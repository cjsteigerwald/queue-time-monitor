"""YOLOv8 person detector."""

from __future__ import annotations

import numpy as np
import supervision as sv
from ultralytics import YOLO

from queue_monitor.config import DetectionConfig

PERSON_CLASS_ID = 0


class PersonDetector:
    """Detect people in video frames using YOLOv8."""

    def __init__(self, config: DetectionConfig):
        self._config = config
        device = config.device or None
        self._model = YOLO(config.model)
        if device:
            self._model.to(device)

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run detection and return only person detections."""
        results = self._model(frame, conf=self._config.confidence, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        # Filter to person class only
        person_mask = detections.class_id == PERSON_CLASS_ID
        return detections[person_mask]
