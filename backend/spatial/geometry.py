import logging
from collections import defaultdict, deque
from typing import Dict, List, Tuple

from config import settings
from contracts import AudioAlert, Detection

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("safestep.spatial")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "[SPATIAL %(levelname)s] %(asctime)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_h)
logger.setLevel(logging.DEBUG)

# How many frames of bounding-box area history to keep per object
_APPROACH_WINDOW = 5
# Fractional area growth per frame considered "approaching fast"
_APPROACH_THRESHOLD = 0.08


class SpatialAnalyzer:
    """
    Converts pixel bounding boxes into clock positions, distance estimates,
    approach velocity, and priority-ranked AudioAlerts.
    """

    # Object class groups — mirrors inference.py groupings for consistency
    CRITICAL_HAZARDS  = {"car", "motorcycle", "bus", "truck"}
    DYNAMIC_OBSTACLES = {"person", "bicycle", "dog"}
    NAV_MARKERS       = {"traffic light", "stop sign", "fire hydrant"}
    ENVIRONMENT       = {"bench", "cat", "backpack", "umbrella",
                         "handbag", "suitcase", "chair", "couch"}

    # 7-zone clock map: (label, left_boundary_fraction)
    # Zones are left-to-right across the frame
    _CLOCK_ZONES = [
        ("9 o'clock",  0.00),
        ("10 o'clock", 1/7),
        ("11 o'clock", 2/7),
        ("12 o'clock", 3/7),
        ("1 o'clock",  4/7),
        ("2 o'clock",  5/7),
        ("3 o'clock",  6/7),
    ]

    # Zones considered "directly ahead" — highest danger if blocked
    _CENTER_ZONE = {"11 o'clock", "12 o'clock", "1 o'clock"}

    # Human-readable direction labels for message generation
    _DIRECTION_LABELS = {
        "9 o'clock":  "on your far left",
        "10 o'clock": "on your left",
        "11 o'clock": "slightly left",
        "12 o'clock": "directly ahead",
        "1 o'clock":  "slightly right",
        "2 o'clock":  "on your right",
        "3 o'clock":  "on your far right",
    }

    def __init__(self, frame_width: int, frame_height: int):
        self.width  = frame_width
        self.height = frame_height
        self._frame_area = frame_width * frame_height

        # Per object_id history of bounding box areas for approach detection
        # { object_id: deque([area1, area2, ...]) }
        self._area_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=_APPROACH_WINDOW)
        )

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------

    def process_detections(self, detections: List[Detection]) -> List[AudioAlert]:
        """
        Converts a list of Detections into a deduplicated, priority-sorted
        list of AudioAlerts.
        """
        # { object_id: AudioAlert } — keeps only the highest-priority alert per id
        best_alerts: Dict[str, AudioAlert] = {}

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            center_x = (x1 + x2) // 2
            bottom_y = y2
            box_area = (x2 - x1) * (y2 - y1)

            clock_pos   = self._get_clock_position(center_x)
            dist_str, is_close = self._get_distance_estimate(bottom_y)
            is_approaching  = self._check_approach(det.label, clock_pos, box_area)
            priority    = self._determine_priority(
                det.label, clock_pos, is_close, is_approaching
            )
            message     = self._build_message(
                det.label, clock_pos, dist_str, priority, is_approaching
            )
            object_id   = f"{det.label}_{clock_pos}_{dist_str}"

            alert = AudioAlert(
                priority=priority,
                message=message,
                object_id=object_id,
            )

            # Deduplication: keep only the highest-priority alert per object_id
            if object_id not in best_alerts or alert.priority < best_alerts[object_id].priority:
                best_alerts[object_id] = alert
                logger.debug(
                    "Alert: [P%d] %s | %s", priority, object_id, message
                )

        # Return sorted by priority (P1 first) so the audio engine gets them
        # in urgency order even before its own queue sorts them
        result = sorted(best_alerts.values(), key=lambda a: a.priority)
        logger.debug("Scene processed: %d alert(s) generated.", len(result))
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_clock_position(self, center_x: int) -> str:
        """Maps the object's X center to a 7-zone clock position."""
        zone_width = self.width / 7
        zone_index = min(int(center_x / zone_width), 6)
        return self._CLOCK_ZONES[zone_index][0]

    def _get_distance_estimate(self, bottom_y: int) -> Tuple[str, bool]:
        """
        Estimates distance from how low the bounding box sits in the frame.
        Lower = closer (for a chest-mounted, forward-facing camera).

        Returns: (distance_string, is_dangerously_close)
        """
        ratio = bottom_y / self.height
        if ratio > settings.CLOSE_DISTANCE_RATIO:
            return "1 step away", True
        elif ratio > settings.MID_DISTANCE_RATIO:
            return "3 steps away", False
        else:
            return "ahead", False

    def _check_approach(self, label: str, clock_pos: str, box_area: int) -> bool:
        """
        Returns True if the object's bounding box has been consistently
        growing over the last few frames — indicating it is approaching.
        """
        obj_key = f"{label}_{clock_pos}"
        history = self._area_history[obj_key]
        history.append(box_area)

        if len(history) < 3:
            return False  # not enough history yet

        # Check if area is on an upward trend
        areas = list(history)
        growth_rates = [
            (areas[i] - areas[i - 1]) / max(areas[i - 1], 1)
            for i in range(1, len(areas))
        ]
        avg_growth = sum(growth_rates) / len(growth_rates)
        return avg_growth > _APPROACH_THRESHOLD

    def _determine_priority(
        self,
        label: str,
        clock_pos: str,
        is_close: bool,
        is_approaching: bool,
    ) -> int:
        """
        Priority 1 — Immediate danger: stop now
        Priority 2 — Caution: hazard present or approaching
        Priority 3 — Awareness: environment context
        """
        in_center = clock_pos in self._CENTER_ZONE

        # P1: close critical hazard or dynamic obstacle dead ahead
        if is_close and in_center:
            if label in self.CRITICAL_HAZARDS or label in self.DYNAMIC_OBSTACLES:
                return 1

        # P1 escalation: approaching fast even if not yet close
        if is_approaching and in_center and label in self.CRITICAL_HAZARDS:
            return 1

        # P2: any vehicle, navigation marker, or approaching obstacle
        if label in self.CRITICAL_HAZARDS or label in self.NAV_MARKERS:
            return 2
        if is_approaching and label in self.DYNAMIC_OBSTACLES:
            return 2

        # P3: everything else
        return 3

    def _build_message(
        self,
        label: str,
        clock_pos: str,
        dist_str: str,
        priority: int,
        is_approaching: bool,
    ) -> str:
        """
        Builds a natural-language spoken message.
        Aims for the concise, clear phrasing of real assistive audio systems.
        """
        direction = self._DIRECTION_LABELS.get(clock_pos, clock_pos)
        name = label.capitalize()

        if priority == 1:
            return f"Stop! {name} {direction}, {dist_str}!"

        if is_approaching:
            return f"Heads up — {label} approaching {direction}."

        if dist_str == "1 step away":
            return f"{name} very close, {direction}."
        elif dist_str == "3 steps away":
            return f"{name} nearby, {direction}."
        else:
            return f"{name} {direction}."


# ---------------------------------------------------------------------------
# Module-level singleton analyzer + public contract function
# ---------------------------------------------------------------------------

_analyzer_instance = None


def analyze_scene(
    detections: List[Detection],
    frame_width: int,
    frame_height: int,
) -> List[AudioAlert]:
    
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SpatialAnalyzer(frame_width, frame_height)
    return _analyzer_instance.process_detections(detections)
