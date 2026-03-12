"""Tests for the processing pipeline (integration-style with mocks)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
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


@patch("queue_monitor.pipeline.PersonDetector")
def test_pause_and_resume(mock_detector_cls, tmp_path):
    config = AppConfig()
    config.storage.database = str(tmp_path / "test.db")
    mock_detector_cls.return_value = MagicMock()

    pipeline = Pipeline(config)

    assert not pipeline.is_paused
    pipeline.pause()
    assert pipeline.is_paused
    pipeline.resume()
    assert not pipeline.is_paused


@patch("queue_monitor.pipeline.PersonDetector")
def test_toggle_pause(mock_detector_cls, tmp_path):
    config = AppConfig()
    config.storage.database = str(tmp_path / "test.db")
    mock_detector_cls.return_value = MagicMock()

    pipeline = Pipeline(config)

    assert not pipeline.is_paused
    result = pipeline.toggle_pause()
    assert result is True
    assert pipeline.is_paused
    result = pipeline.toggle_pause()
    assert result is False
    assert not pipeline.is_paused


@patch("queue_monitor.pipeline.PersonDetector")
def test_error_initially_none(mock_detector_cls, tmp_path):
    config = AppConfig()
    config.storage.database = str(tmp_path / "test.db")
    mock_detector_cls.return_value = MagicMock()

    pipeline = Pipeline(config)
    assert pipeline.error is None


@patch("queue_monitor.pipeline.PersonDetector")
def test_set_error_stores_info(mock_detector_cls, tmp_path):
    config = AppConfig()
    config.storage.database = str(tmp_path / "test.db")
    mock_detector_cls.return_value = MagicMock()

    pipeline = Pipeline(config)
    try:
        raise ValueError("boom")
    except ValueError as exc:
        pipeline._set_error(exc)

    err = pipeline.error
    assert err is not None
    assert err.message == "boom"
    assert "ValueError" in err.traceback
    assert "boom" in err.traceback
    assert err.timestamp  # ISO format string


@patch("queue_monitor.pipeline.PersonDetector")
def test_run_stores_error_on_crash(mock_detector_cls, tmp_path):
    config = AppConfig()
    config.storage.database = str(tmp_path / "test.db")
    mock_detector_cls.return_value = MagicMock()

    pipeline = Pipeline(config)

    # Mock source to raise after open
    mock_source = MagicMock()
    mock_source.open.return_value = None
    mock_source.frame_size = (1280, 720)
    mock_source.fps = 30
    mock_source.__iter__ = MagicMock(side_effect=RuntimeError("source failed"))
    pipeline._source = mock_source

    with pytest.raises(RuntimeError, match="source failed"):
        pipeline.run()

    assert pipeline.error is not None
    assert pipeline.error.message == "source failed"
    assert not pipeline.is_running
