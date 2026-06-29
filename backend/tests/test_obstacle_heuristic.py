"""
tests/test_obstacle_heuristic.py

SafeStep Web — Tests for spatial/obstacle_heuristic.py (new module).

Background: testing revealed that walking into walls/doors triggered NO
alert at all, because YOLO has no "wall" or "door" class in COCO — it's a
structural gap in pure object detection, not a tuning issue. This module
adds a cheap, parallel check for "something large, flat, and close filling
the forward view" that fires even when YOLO finds nothing to label. These
tests protect that behavior and its safeguards against regressions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from spatial.obstacle_heuristic import ObstacleHeuristic, _SUSTAINED_FRAMES_REQUIRED
from contracts import Detection


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def test_sustained_flat_frame_triggers_alert():
    section("TEST 1: sustained flat/featureless frames trigger an obstacle alert")

    heuristic = ObstacleHeuristic()
    flat_frame = np.full((480, 640, 3), 130, dtype=np.uint8)

    alert = None
    for _ in range(_SUSTAINED_FRAMES_REQUIRED + 1):
        alert = heuristic.check(flat_frame, detections=[])

    assert alert is not None, "Expected an alert after sustained flat frames"
    assert alert.priority == 1
    assert alert.object_id == "obstacle_ahead_unclassified"
    print("✓ TEST 1 PASSED")


def test_single_flat_frame_does_not_trigger():
    section("TEST 2: a single flat frame alone should NOT trigger (avoid false positives)")

    heuristic = ObstacleHeuristic()
    flat_frame = np.full((480, 640, 3), 130, dtype=np.uint8)

    alert = heuristic.check(flat_frame, detections=[])
    assert alert is None, "A single flat frame should not be enough to fire an alert"
    print("✓ TEST 2 PASSED")


def test_textured_scene_never_triggers():
    section("TEST 3: a normal/textured scene should never trigger a false positive")

    heuristic = ObstacleHeuristic()
    np.random.seed(7)
    textured_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    alert = None
    for _ in range(_SUSTAINED_FRAMES_REQUIRED + 2):
        alert = heuristic.check(textured_frame, detections=[])

    assert alert is None, "Textured scenes must not trigger the obstacle heuristic"
    print("✓ TEST 3 PASSED")


def test_defers_to_existing_center_detection():
    section("TEST 4: heuristic stays silent when YOLO already labeled the center object")

    heuristic = ObstacleHeuristic()
    flat_frame = np.full((480, 640, 3), 130, dtype=np.uint8)
    center_detection = Detection(label="car", confidence=0.9, bbox=(250, 200, 400, 450))

    alert = None
    for _ in range(_SUSTAINED_FRAMES_REQUIRED + 1):
        alert = heuristic.check(flat_frame, detections=[center_detection])

    assert alert is None, "Should defer to YOLO's own alert when center is already labeled"
    print("✓ TEST 4 PASSED")


def test_off_center_detection_does_not_suppress():
    section("TEST 5: an off-center detection must NOT suppress a real center obstacle")

    heuristic = ObstacleHeuristic()
    flat_frame = np.full((480, 640, 3), 130, dtype=np.uint8)
    side_detection = Detection(label="chair", confidence=0.8, bbox=(0, 200, 100, 450))

    alert = None
    for _ in range(_SUSTAINED_FRAMES_REQUIRED + 1):
        alert = heuristic.check(flat_frame, detections=[side_detection])

    assert alert is not None, "Off-center detections must not suppress the obstacle alert"
    print("✓ TEST 5 PASSED")


def test_history_resets_after_real_detection_appears():
    section("TEST 6: flatness history resets once YOLO starts covering the center "
            "(prevents an alert firing right after YOLO's own detection ends)")

    heuristic = ObstacleHeuristic()
    flat_frame = np.full((480, 640, 3), 130, dtype=np.uint8)
    center_detection = Detection(label="car", confidence=0.9, bbox=(250, 200, 400, 450))

    # Build up partial flat history...
    for _ in range(_SUSTAINED_FRAMES_REQUIRED - 1):
        heuristic.check(flat_frame, detections=[])
    # ...then YOLO starts covering the center (should clear history)...
    heuristic.check(flat_frame, detections=[center_detection])
    # ...then YOLO stops covering it again. History should have been reset,
    # so this single frame alone should NOT be enough to fire immediately.
    alert = heuristic.check(flat_frame, detections=[])

    assert alert is None, (
        "History should have reset when a real detection appeared, "
        "so one frame afterward should not be enough to fire"
    )
    print("✓ TEST 6 PASSED")


if __name__ == "__main__":
    print("\nSafeStep Web — Obstacle Heuristic Test Suite (new module)")

    test_sustained_flat_frame_triggers_alert()
    test_single_flat_frame_does_not_trigger()
    test_textured_scene_never_triggers()
    test_defers_to_existing_center_detection()
    test_off_center_detection_does_not_suppress()
    test_history_resets_after_real_detection_appears()

    print("\n" + "="*60)
    print("  ALL TESTS COMPLETE")
    print("="*60)
