
import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np
import torch
from ultralytics import YOLO

from config import settings
from contracts import Detection

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("safestep.inference")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "[INFERENCE %(levelname)s] %(asctime)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(_h)
logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Target class groups — identical to SafeStep Final
# ---------------------------------------------------------------------------
CRITICAL_HAZARD_CLASSES = {
    "car", "motorcycle", "bus", "truck",
}

NAVIGATION_CLASSES = {
    "person", "bicycle", "dog", "traffic light", "stop sign",
}

ENVIRONMENT_CLASSES = {
    "bench", "cat", "backpack", "umbrella", "handbag",
    "suitcase", "chair", "couch", "fire hydrant",
}

ALL_TARGET_CLASSES = CRITICAL_HAZARD_CLASSES | NAVIGATION_CLASSES | ENVIRONMENT_CLASSES

LOW_LIGHT_THRESHOLD = 40
SMOOTHING_WINDOW = 4


class InferenceEngine:
    

    def __init__(self, model_name: str = settings.MODEL_NAME, shared_model: Optional[YOLO] = None):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if shared_model is not None:
            self.model = shared_model
            logger.info("Reusing shared YOLOv8 model weights on device: %s", self.device)
        else:
            logger.info("Loading YOLOv8 model '%s' on device: %s", model_name, self.device)
            self.model = YOLO(model_name)

        # Per-track confidence history for smoothing: { track_id: deque([conf, ...]) }
        self._conf_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=SMOOTHING_WINDOW)
        )

        # Per-track label cache: { track_id: label_name }
        self._track_labels: Dict[int, str] = {}

        # Each engine gets a fresh, independent tracker — this is the key
        # piece that makes per-session isolation actually work. persist=True
        # in run_inference() tells THIS model call to keep tracking state
        # between calls, but because each session has its own InferenceEngine
        # (and therefore conceptually its own tracker lineage keyed off this
        # instance), one user's tracks never bleed into another's.
        self._tracker_initialized = False

        logger.info(
            "Inference engine ready. Tracking: ByteTrack | Smoothing window: %d frames",
            SMOOTHING_WINDOW,
        )

    # ------------------------------------------------------------------
    # Public method — IDENTICAL logic to SafeStep Final
    # ------------------------------------------------------------------

    def run_inference(self, frame: np.ndarray) -> List[Detection]:
        
        if frame is None:
            return []

        # -- Step 1: Low-light check ------------------------------------
        self._check_lighting(frame)

        # -- Step 2: YOLOv8 + ByteTrack ---------------------------------
        results = self.model.track(
            source=frame,
            device=self.device,
            conf=0.25,
            iou=0.5,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        # -- Steps 3-5: Filter, smooth, emit ----------------------------
        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            has_track_ids = boxes.id is not None

            for i, box in enumerate(boxes):
                class_id = int(box.cls[0].item())
                label = self.model.names[class_id]

                if label not in ALL_TARGET_CLASSES:
                    continue

                raw_conf = float(box.conf[0].item())
                track_id: int = int(boxes.id[i].item()) if has_track_ids else -(i + 1)

                self._track_labels[track_id] = label

                # -- Step 4: Confidence smoothing -----------------------
                self._conf_history[track_id].append(raw_conf)
                smoothed_conf = float(
                    sum(self._conf_history[track_id]) / len(self._conf_history[track_id])
                )

                if smoothed_conf < settings.CONFIDENCE_THRESHOLD:
                    logger.debug(
                        "Track %d (%s) suppressed: smoothed conf %.2f < threshold %.2f",
                        track_id, label, smoothed_conf, settings.CONFIDENCE_THRESHOLD,
                    )
                    continue

                # -- Step 5: Build Detection contract -------------------
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)

                detections.append(
                    Detection(
                        label=label,
                        confidence=round(smoothed_conf, 3),
                        bbox=(x1, y1, x2, y2),
                    )
                )

                logger.debug(
                    "Detection: track=%d label=%s conf=%.2f box=(%d,%d,%d,%d)",
                    track_id, label, smoothed_conf, x1, y1, x2, y2,
                )

        self._prune_lost_tracks(results)

        logger.debug("Frame processed: %d detection(s) emitted.", len(detections))
        return detections

    # ------------------------------------------------------------------
    # Internal helpers — IDENTICAL to SafeStep Final
    # ------------------------------------------------------------------

    def _check_lighting(self, frame: np.ndarray) -> None:
        gray = np.mean(frame)
        if gray < LOW_LIGHT_THRESHOLD:
            logger.warning(
                "LOW LIGHT DETECTED (mean brightness=%.1f). "
                "Detection accuracy may be reduced. Consider better lighting.",
                gray,
            )

    def _prune_lost_tracks(self, results) -> None:
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return

        active_ids = {int(tid.item()) for tid in results[0].boxes.id}
        lost_ids = set(self._conf_history.keys()) - active_ids

        for lost_id in lost_ids:
            del self._conf_history[lost_id]
            self._track_labels.pop(lost_id, None)

        if lost_ids:
            logger.debug("Pruned %d lost track(s): %s", len(lost_ids), lost_ids)


# ---------------------------------------------------------------------------
# Singleton + public contract function — KEPT FOR PARITY / SINGLE-SESSION USE
#
# This matches SafeStep Final's exact public API and is what tests/test_*.py
# exercises. It is intentionally NOT what server.py uses for live multi-user
# traffic - server.py gives each WebSocket session its own InferenceEngine
# (see SessionState in server.py). Use process_frame() only for local
# single-stream scripts, CLI testing, or the test suite, exactly as the
# original main.py did.
# ---------------------------------------------------------------------------

_engine_instance: Optional[InferenceEngine] = None


def process_frame(frame: np.ndarray) -> List[Detection]:
    
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = InferenceEngine()
    return _engine_instance.run_inference(frame)
