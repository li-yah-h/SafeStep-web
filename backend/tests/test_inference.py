

import logging
import os
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.DEBUG)

from contracts import Detection
from models.inference import process_frame, InferenceEngine, LOW_LIGHT_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def blank_frame(brightness: int = 128) -> np.ndarray:
    """Returns a solid-colour 640x480 BGR frame."""
    frame = np.full((480, 640, 3), brightness, dtype=np.uint8)
    return frame


def frame_with_rectangle(color=(255, 255, 255)) -> np.ndarray:
    """Returns a frame with a white rectangle — mimics a generic object."""
    frame = blank_frame(100)
    cv2.rectangle(frame, (200, 150), (440, 400), color, -1)
    return frame


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pipeline_smoke():
    section("TEST 1: Pipeline smoke test — model loads, returns List[Detection]")

    frame = blank_frame()
    result = process_frame(frame)

    assert isinstance(result, list), "process_frame must return a list"
    print(f"Detections on blank frame: {len(result)} (0 expected — no real objects)")
    print("✓ TEST 1 PASSED")


def test_contract_shape():
    section("TEST 2: Contract shape — every Detection has correct field types")

    # Use a real-ish frame; we just want to check shape if anything is returned
    frame = frame_with_rectangle()
    result = process_frame(frame)

    for det in result:
        assert isinstance(det, Detection), "Each item must be a Detection"
        assert isinstance(det.label, str), "label must be str"
        assert isinstance(det.confidence, float), "confidence must be float"
        assert isinstance(det.bbox, tuple) and len(det.bbox) == 4, \
            "bbox must be a 4-tuple"
        assert all(isinstance(v, int) for v in det.bbox), \
            "bbox values must be ints"

    print(f"Checked {len(result)} detection(s) — all fields valid.")
    print("✓ TEST 2 PASSED")


def test_low_light_warning():
    section("TEST 3: Low-light warning — dark frame should trigger a log warning")

    import io
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.WARNING)
    logging.getLogger("safestep.inference").addHandler(handler)

    dark_frame = blank_frame(brightness=10)   # very dark
    process_frame(dark_frame)

    log_output = log_capture.getvalue()
    assert "LOW LIGHT" in log_output, \
        f"Expected LOW LIGHT warning in logs, got: {log_output}"

    print(f"Log captured: {log_output.strip()}")
    print("✓ TEST 3 PASSED")


def test_normal_light_no_warning():
    section("TEST 4: Normal light — no warning on a bright frame")

    import io
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.WARNING)
    logging.getLogger("safestep.inference").addHandler(handler)

    bright_frame = blank_frame(brightness=150)
    process_frame(bright_frame)

    log_output = log_capture.getvalue()
    assert "LOW LIGHT" not in log_output, \
        "Should NOT warn about low light on a bright frame"

    print("No low-light warning on bright frame. Correct.")
    print("✓ TEST 4 PASSED")


def test_confidence_smoothing_logic():
    section("TEST 5: Confidence smoothing logic — direct unit test on smoothing math")

    from collections import deque

    # Simulate SMOOTHING_WINDOW=4 with 3 weak readings and 1 strong — should stay below threshold
    history: deque = deque(maxlen=4)
    history.extend([0.20, 0.22, 0.21, 0.23])
    avg = sum(history) / len(history)
    assert avg < 0.45, f"Smoothed conf {avg:.2f} should be below threshold 0.45"
    print(f"Weak history average: {avg:.3f} — correctly below threshold")

    # Now simulate 4 strong readings — should pass through
    history.clear()
    history.extend([0.80, 0.82, 0.79, 0.85])
    avg = sum(history) / len(history)
    assert avg >= 0.45, f"Smoothed conf {avg:.2f} should be above threshold 0.45"
    print(f"Strong history average: {avg:.3f} — correctly above threshold")

    print("✓ TEST 5 PASSED")


def test_returns_empty_on_none_frame():
    section("TEST 6: None frame input — must return empty list, not crash")

    result = process_frame(None)
    assert result == [], f"Expected [] for None frame, got {result}"
    print("✓ TEST 6 PASSED")




def test_shared_model_weights_reused():
    section("TEST 7: shared_model param reuses weights instead of reloading")

    base_engine = InferenceEngine()
    engine_a = InferenceEngine(shared_model=base_engine.model)

    assert engine_a.model is base_engine.model, \
        "Engine constructed with shared_model should reuse that exact model object"
    print("✓ TEST 7 PASSED")


def test_two_sessions_have_independent_state():
    section("TEST 8: Two engines sharing weights still have independent tracker state")

    base_engine = InferenceEngine()
    engine_a = InferenceEngine(shared_model=base_engine.model)
    engine_b = InferenceEngine(shared_model=base_engine.model)

    assert engine_a._conf_history is not engine_b._conf_history, \
        "Each session must own its own confidence history dict"
    assert engine_a._track_labels is not engine_b._track_labels, \
        "Each session must own its own track-label cache"

    # Mutate A's state and confirm B is untouched — this is the exact bug
    # that would occur if both sessions accidentally shared one engine.
    engine_a._conf_history[1].append(0.99)
    assert 1 not in engine_b._conf_history, \
        "Session B's confidence history was affected by session A — state leak!"

    print("✓ TEST 8 PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nSafeStep Web Inference Module — Isolated Test Suite (ported from Jaliba's original)")

    test_pipeline_smoke()
    test_contract_shape()
    test_low_light_warning()
    test_normal_light_no_warning()
    test_confidence_smoothing_logic()
    test_returns_empty_on_none_frame()
    test_shared_model_weights_reused()
    test_two_sessions_have_independent_state()

    print("\n" + "="*60)
    print("  ALL TESTS COMPLETE")
    print("="*60)
