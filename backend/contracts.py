

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Detection:
    
    label: str                         # COCO class name, e.g. "person", "car"
    confidence: float                  # Smoothed confidence score, 0.0–1.0
    bbox: Tuple[int, int, int, int]    # (x1, y1, x2, y2) pixel coords

    def __post_init__(self):
        assert 0.0 <= self.confidence <= 1.0, "confidence must be in [0, 1]"
        assert len(self.bbox) == 4,           "bbox must have exactly 4 values"


@dataclass
class AudioAlert:
    
    priority:  int    # 1 = Immediate danger | 2 = Caution | 3 = Environment
    message:   str    # Exact sentence to speak, e.g. "Stop! Car directly ahead!"
    object_id: str    # Stable ID for cooldown tracking, e.g. "car_12 o'clock_1 step away"

    def __post_init__(self):
        assert self.priority in (1, 2, 3), "priority must be 1, 2, or 3"
        assert self.message.strip(),        "message must not be empty"
