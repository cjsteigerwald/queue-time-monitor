"""Tests for the processing pipeline (integration-style with mocks)."""

from unittest.mock import MagicMock, patch

import numpy as np
import supervision as sv

from queue_monitor.config import AppConfig, ZoneConfig
from queue_monitor.pipeline import Pipeline


def _make_person_detections(n: int) -> sv.Detections:
    """Create n person detections spread across the frame."""
    bboxes = np.array(
        [[200 + i * 100, 200, 250 + i * 100, 400] for i in range(n)],
        dtype=np.float32,
    )
    return sv.Detections(
        xyxy=bboxes,
        confidence=np.full(n, 0.9, dtype=np.float32),
        class_id=np.zeros(n, dtype=int),
    )


@patch("queue_monitor.pipeline.PersonDetector")
def test_process_frame_returns_metrics(mock_detector_cls, tmp_path):
    config = AppConfig(
        zones=[ZoneConfig(name="test", polygon=[[0, 0], [1280, 0], [1280, 720], [0, 720]])],
    )
    config.storage.database = str(tmp_path / "test.db")

    mock_detector = MagicMock()
    mock_detector.detect.return_value = _make_person_detections(3)
    mock_detector_cls.return_value = mock_detector

    pipeline = Pipeline(config)
    pipeline._db.open()
    pipeline._init_zones((1280, 720))

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = pipeline.process_frame(frame)

    assert result.annotated_frame is not None
    assert len(result.metrics) == 1
    assert result.metrics[0].zone_name == "test"
    assert result.metrics[0].raw_count >= 0

    pipeline._db.close()


@patch("queue_monitor.pipeline.PersonDetector")
def test_process_frame_empty(mock_detector_cls, tmp_path):
    config = AppConfig()
    config.storage.database = str(tmp_path / "test.db")

    mock_detector = MagicMock()
    mock_detector.detect.return_value = sv.Detections.empty()
    mock_detector_cls.return_value = mock_detector

    pipeline = Pipeline(config)
    pipeline._db.open()
    pipeline._init_zones((1280, 720))

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = pipeline.process_frame(frame)

    assert result.metrics[0].raw_count == 0
    assert result.metrics[0].wait_time == 0.0

    pipeline._db.close()
