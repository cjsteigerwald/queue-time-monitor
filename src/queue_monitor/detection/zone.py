"""Polygon-based queue zone filtering."""

from __future__ import annotations

import numpy as np
import supervision as sv

from queue_monitor.config import ZoneConfig


class QueueZone:
    """Filter detections to those within a polygon zone."""

    def __init__(self, config: ZoneConfig, frame_resolution: tuple[int, int]):
        self._config = config
        self._name = config.name
        polygon = np.array(config.polygon, dtype=np.int32)
        if len(polygon) < 3:
            # Default to full frame if no polygon defined
            w, h = frame_resolution
            polygon = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
        self._polygon = polygon
        self._zone = sv.PolygonZone(polygon=self._polygon)

    @property
    def name(self) -> str:
        return self._name

    @property
    def polygon(self) -> np.ndarray:
        return self._polygon

    @property
    def num_servers(self) -> int:
        return self._config.num_servers

    @property
    def service_time(self) -> float:
        return self._config.service_time

    def filter(self, detections: sv.Detections) -> sv.Detections:
        """Return only detections whose anchors fall within the zone polygon."""
        mask = self._zone.trigger(detections=detections)
        return detections[mask]
