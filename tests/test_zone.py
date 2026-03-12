"""Tests for polygon queue zone filtering."""

import numpy as np
import supervision as sv

from queue_monitor.config import ZoneConfig
from queue_monitor.detection.zone import QueueZone


def _detections_at(centers: list[tuple[int, int]], size: int = 50) -> sv.Detections:
    """Create detections centered at given points."""
    bboxes = np.array(
        [[cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2] for cx, cy in centers],
        dtype=np.float32,
    )
    return sv.Detections(
        xyxy=bboxes,
        confidence=np.full(len(centers), 0.9, dtype=np.float32),
        class_id=np.zeros(len(centers), dtype=int),
    )


def test_zone_filters_inside():
    config = ZoneConfig(name="test", polygon=[[200, 200], [400, 200], [400, 400], [200, 400]])
    zone = QueueZone(config, (640, 480))

    # One inside (300, 300), one outside (50, 50)
    dets = _detections_at([(300, 300), (50, 50)])
    filtered = zone.filter(dets)

    assert len(filtered) == 1


def test_zone_empty_polygon_uses_full_frame():
    config = ZoneConfig(name="test", polygon=[])
    zone = QueueZone(config, (640, 480))

    # Detection in the middle of the frame should be inside
    dets = _detections_at([(320, 240)])
    filtered = zone.filter(dets)

    assert len(filtered) == 1


def test_zone_properties():
    config = ZoneConfig(
        name="main",
        polygon=[[0, 0], [100, 0], [100, 100]],
        num_servers=3,
        service_time=20.0,
    )
    zone = QueueZone(config, (640, 480))

    assert zone.name == "main"
    assert zone.num_servers == 3
    assert zone.service_time == 20.0
