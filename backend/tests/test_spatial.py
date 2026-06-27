

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts import Detection
from spatial.geometry import SpatialAnalyzer, analyze_scene


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


W, H = 640, 480


def det(label, x1, y1, x2, y2, conf=0.90):
    return Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2))


class TestSpatialAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = SpatialAnalyzer(W, H)

    def test_clock_position_center(self):
        section("TEST 1: Object at frame center → 12 o'clock")
        pos = self.analyzer._get_clock_position(W // 2)
        self.assertEqual(pos, "12 o'clock")
        print(f"  center_x={W//2} → {pos}")
        print("✓ TEST 1 PASSED")

    def test_clock_position_far_left(self):
        section("TEST 2: Object at far left → 9 o'clock")
        pos = self.analyzer._get_clock_position(10)
        self.assertEqual(pos, "9 o'clock")
        print(f"  center_x=10 → {pos}")
        print("✓ TEST 2 PASSED")

    def test_clock_position_far_right(self):
        section("TEST 3: Object at far right → 3 o'clock")
        pos = self.analyzer._get_clock_position(W - 10)
        self.assertEqual(pos, "3 o'clock")
        print(f"  center_x={W-10} → {pos}")
        print("✓ TEST 3 PASSED")

    def test_close_distance(self):
        section("TEST 4: Bottom of frame → 1 step away, is_close=True")
        dist, is_close = self.analyzer._get_distance_estimate(int(H * 0.9))
        self.assertEqual(dist, "1 step away")
        self.assertTrue(is_close)
        print(f"  bottom_y={int(H*0.9)} → '{dist}', is_close={is_close}")
        print("✓ TEST 4 PASSED")

    def test_far_distance(self):
        section("TEST 5: Top of frame → ahead, is_close=False")
        dist, is_close = self.analyzer._get_distance_estimate(int(H * 0.3))
        self.assertEqual(dist, "ahead")
        self.assertFalse(is_close)
        print(f"  bottom_y={int(H*0.3)} → '{dist}', is_close={is_close}")
        print("✓ TEST 5 PASSED")

    def test_priority_1_close_car_center(self):
        section("TEST 6: Close car dead center → Priority 1")
        alerts = analyze_scene(
            [det("car", 200, 100, 440, 450)], W, H
        )
        self.assertTrue(any(a.priority == 1 for a in alerts))
        print(f"  Alert: {alerts[0].message}")
        print("✓ TEST 6 PASSED")

    def test_priority_3_distant_bench(self):
        section("TEST 7: Distant bench → Priority 3")
        alerts = analyze_scene(
            [det("bench", 10, 10, 80, 100)], W, H
        )
        self.assertTrue(any(a.priority == 3 for a in alerts))
        print(f"  Alert: {alerts[0].message}")
        print("✓ TEST 7 PASSED")

    def test_deduplication(self):
        section("TEST 8: Two overlapping detections → only one alert per object_id")
        # Two nearly identical car boxes — same zone, same distance
        detections = [
            det("car", 200, 100, 440, 450),
            det("car", 205, 105, 445, 452),
        ]
        alerts = analyze_scene(detections, W, H)
        object_ids = [a.object_id for a in alerts]
        self.assertEqual(len(object_ids), len(set(object_ids)), "Duplicate object_ids found")
        print(f"  {len(alerts)} unique alert(s) from 2 overlapping detections")
        print("✓ TEST 8 PASSED")

    def test_message_is_natural_language(self):
        section("TEST 9: Messages use natural language direction words")
        alerts = analyze_scene([det("person", 50, 200, 150, 400)], W, H)
        self.assertTrue(len(alerts) > 0)
        msg = alerts[0].message
        natural_words = ["left", "right", "ahead", "directly", "slightly", "far"]
        self.assertTrue(
            any(w in msg for w in natural_words),
            f"Message '{msg}' contains no natural direction word"
        )
        print(f"  Message: '{msg}'")
        print("✓ TEST 9 PASSED")

    def test_priority_sorted(self):
        section("TEST 10: Output is sorted P1 first")
        detections = [
            det("bench", 300, 10, 380, 80),      # P3 far bench
            det("car",   200, 100, 440, 450),     # P1 close car
        ]
        alerts = analyze_scene(detections, W, H)
        self.assertEqual(alerts[0].priority, 1, "First alert should be P1")
        print(f"  First alert: [P{alerts[0].priority}] {alerts[0].message}")
        print("✓ TEST 10 PASSED")


if __name__ == "__main__":
    print("\nSafeStep Spatial Module — Isolated Test Suite (Lexmi)")
    unittest.main(verbosity=0, exit=False)
    print("\n" + "="*60 + "\n  ALL TESTS COMPLETE\n" + "="*60)
