"""
spatial/obstacle_heuristic.py

SafeStep Web — Unclassified Obstacle Heuristic.
New module — does not exist in SafeStep Final.

WHY THIS MODULE EXISTS:

YOLOv8 (and every other COCO-trained detector) only recognizes the 80
object categories COCO was labeled with — person, car, chair, dog, and so
on. There is no "wall" or "door" category in COCO. This is not a confidence-
threshold or class-list tuning problem; it is a structural limitation of
object detection as an approach. A user can walk straight into a wall or a
closed door and YOLO will correctly report "no objects detected," because
from its point of view, that's true — a wall isn't one of the things it was
trained to name.

This was confirmed as a real gap during testing: a person using SafeStep
Web bumped into walls and doors with zero alert. The fix here is NOT to try
to make YOLO detect walls (it structurally can't), but to add a cheap,
parallel heuristic that asks a more basic question: "is something large,
flat, and featureless rapidly filling my forward view?" That pattern fits
a close wall/door/blank surface even though we can never know its label.

DESIGN CONSTRAINTS (deliberately chosen over a depth-estimation model):
  - Must add near-zero latency/CPU on top of the existing YOLO pipeline,
    since the team is already CPU-constrained on free-tier hosting with
    multiple simultaneous users (see server.py's SessionState).
  - Must run as a fallback ONLY when YOLO found nothing in the danger
    zone for that frame — this heuristic never overrides or competes with
    a real labeled detection, it only fills the silence YOLO leaves behind.
  - Deliberately approximate: it cannot say "wall" vs "door" vs "unlabeled
    parked truck" — it only says "something solid is close ahead, stop or
    slow down." That is the actual safety-critical signal that was
    previously missing entirely.

ALGORITHM:
  1. Look only at the center third of the frame (the "danger zone" a
     person is walking toward) — matches geometry.py's _CENTER_ZONE concept.
  2. Compute edge density in that region (Laplacian variance). A close,
     flat, featureless surface (wall, door, blank surface) has LOW edge
     density compared to a normal scene with texture/depth variation.
  3. Track edge density across the last few frames per session. A
     SUSTAINED low value (not just one blurry frame) is required before
     firing, to avoid false positives from motion blur, a dark room, or a
     single bad frame.
  4. If sustained low edge density persists for several consecutive
     frames, fire a single generic priority-1 AudioAlert, cooldown-gated
     exactly like every other alert in the system.

This module is intentionally separate from spatial/geometry.py (Lexmi's
module stays byte-for-byte unchanged) and is merged into the alert list in
server.py's WebSocket loop, alongside the analyzer's own output.

Public interface:
    from spatial.obstacle_heuristic import ObstacleHeuristic
    heuristic = ObstacleHeuristic()                 # one per session, like SpatialAnalyzer
    alert: Optional[AudioAlert] = heuristic.check(frame, detections)
"""

import logging
from collections import deque
from typing import List, Optional

import cv2
import numpy as np

from contracts import AudioAlert, Detection

logger = logging.getLogger("safestep.obstacle_heuristic")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "[OBSTACLE %(levelname)s] %(asctime)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_h)
logger.setLevel(logging.DEBUG)


# How many consecutive frames of low edge density are required before
# firing. Higher = fewer false positives, but slightly slower to react.
_SUSTAINED_FRAMES_REQUIRED = 4

# Laplacian variance below this is considered "flat / featureless".
# Tuned conservatively high-confidence-flat; lower this if real obstacles
# are being missed, raise it if normal scenes (e.g. a plain hallway) are
# triggering false alerts.
_FLATNESS_THRESHOLD = 35.0

# Only look at the center fraction of the frame width — matches the
# "directly ahead" danger zone a person is walking toward, not peripheral
# objects that aren't in their path.
_CENTER_REGION_FRACTION = 1 / 3

# Cooldown for this alert, in seconds. Uses the same value as
# settings.CRITICAL_COOLDOWN conceptually (kept as a local constant rather
# than importing settings, since this check happens before geometry.py's
# cooldown-aware queueing — server.py's audio.js cooldown logic handles the
# actual suppression using this alert's object_id).
OBSTACLE_ALERT_PRIORITY = 1


class ObstacleHeuristic:
    """
    Stateful, per-session check for a large, close, unclassified obstacle
    filling the center of the frame. Create one instance per WebSocket
    session (same lifecycle as InferenceEngine / SpatialAnalyzer in
    server.py's SessionState) so frame history isn't shared between users.
    """

    def __init__(self):
        # Rolling history of "is this frame flat?" booleans.
        self._flatness_history: deque = deque(maxlen=_SUSTAINED_FRAMES_REQUIRED)

    def check(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> Optional[AudioAlert]:
        """
        Returns a generic "obstacle ahead" AudioAlert if a sustained,
        large, featureless region is filling the center of the frame AND
        YOLO has not already labeled something there. Returns None
        otherwise — this is a fallback, not a replacement for real
        detections.
        """
        if self._yolo_already_covers_center(frame, detections):
            # A real labeled object is already being reported for this
            # area — don't pile a redundant generic alert on top of it.
            self._flatness_history.clear()
            return None

        is_flat_now = self._is_center_flat(frame)
        self._flatness_history.append(is_flat_now)

        if len(self._flatness_history) < _SUSTAINED_FRAMES_REQUIRED:
            return None  # not enough history yet to be confident

        if not all(self._flatness_history):
            return None  # not sustained across every recent frame

        logger.debug(
            "Sustained flat/close obstacle detected in center zone "
            "(last %d frames all below flatness threshold %.1f).",
            _SUSTAINED_FRAMES_REQUIRED, _FLATNESS_THRESHOLD,
        )

        return AudioAlert(
            priority=OBSTACLE_ALERT_PRIORITY,
            message="Stop! Obstacle directly ahead.",
            object_id="obstacle_ahead_unclassified",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _yolo_already_covers_center(
        self,
        frame: np.ndarray,
        detections: List[Detection],
    ) -> bool:
        """
        True if any YOLO detection's box already overlaps the center danger
        zone — in that case, geometry.py's normal alert is the right one to
        speak, and this heuristic should stay silent rather than duplicate
        or contradict it.
        """
        frame_width = frame.shape[1]
        center_left = frame_width * (0.5 - _CENTER_REGION_FRACTION / 2)
        center_right = frame_width * (0.5 + _CENTER_REGION_FRACTION / 2)

        for det in detections:
            x1, _, x2, _ = det.bbox
            box_center_x = (x1 + x2) / 2
            if center_left <= box_center_x <= center_right:
                return True
        return False

    def _is_center_flat(self, frame: np.ndarray) -> bool:
        """
        Computes Laplacian variance (a standard, cheap focus/edge-density
        measure) over the center region of the frame. Low variance means
        few edges/texture — consistent with a close, flat surface filling
        the view. This is a single OpenCV call on a cropped region, so the
        added cost per frame is small relative to YOLO inference.
        """
        height, width = frame.shape[0], frame.shape[1]
        center_left = int(width * (0.5 - _CENTER_REGION_FRACTION / 2))
        center_right = int(width * (0.5 + _CENTER_REGION_FRACTION / 2))

        center_region = frame[:, center_left:center_right]
        gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        return laplacian_var < _FLATNESS_THRESHOLD
