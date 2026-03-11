"""Tests for person detector (mocked YOLO to avoid model download in CI)."""

from unittest.mock import MagicMock, patch

import numpy as np
import supervision as sv

from queue_monitor.config import DetectionConfig


def _make_detections(class_ids, bboxes=None, confidences=None):
    """Helper to create sv.Detections."""
    n = len(class_ids)
    if bboxes is None:
        bboxes = np.array(
            [[i * 100, i * 100, i * 100 + 50, i * 100 + 50] for i in range(n)],
            dtype=np.float32,
        )
    if confidences is None:
        confidences = np.full(n, 0.9, dtype=np.float32)
    return sv.Detections(
        xyxy=bboxes,
        confidence=confidences,
        class_id=np.array(class_ids, dtype=int),
    )


@patch("queue_monitor.detection.detector.YOLO")
def test_detector_filters_to_persons(mock_yolo_cls):
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    # Simulate YOLO returning persons (0) and cars (2)
    mock_result = MagicMock()
    mock_result.boxes = MagicMock()

    # We'll patch from_ultralytics to return mixed detections
    mixed = _make_detections([0, 2, 0, 1])

    patch_target = "queue_monitor.detection.detector.sv.Detections.from_ultralytics"
    with patch(patch_target, return_value=mixed):
        from queue_monitor.detection.detector import PersonDetector

        detector = PersonDetector(DetectionConfig())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_model.return_value = [mock_result]
        result = detector.detect(frame)

        assert len(result) == 2
        assert all(cid == 0 for cid in result.class_id)


@patch("queue_monitor.detection.detector.YOLO")
def test_detector_empty_frame(mock_yolo_cls):
    mock_model = MagicMock()
    mock_yolo_cls.return_value = mock_model

    empty = sv.Detections.empty()

    patch_target = "queue_monitor.detection.detector.sv.Detections.from_ultralytics"
    with patch(patch_target, return_value=empty):
        from queue_monitor.detection.detector import PersonDetector

        detector = PersonDetector(DetectionConfig())
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        mock_model.return_value = [MagicMock()]
        result = detector.detect(frame)

        assert len(result) == 0
