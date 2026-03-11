"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from queue_monitor.config import (
    AppConfig,
    DetectionConfig,
    EstimationConfig,
    StorageConfig,
    VideoConfig,
    ZoneConfig,
)


@pytest.fixture
def sample_frame() -> np.ndarray:
    """A blank 720p frame for testing."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_config(tmp_path: Path) -> AppConfig:
    """A test configuration with a temporary database."""
    return AppConfig(
        video=VideoConfig(source="0", width=1280, height=720),
        detection=DetectionConfig(model="yolov8n.pt", confidence=0.35),
        zones=[
            ZoneConfig(
                name="test_queue",
                polygon=[[100, 100], [600, 100], [600, 500], [100, 500]],
                num_servers=2,
                service_time=25.0,
            )
        ],
        estimation=EstimationConfig(mode="adaptive", ema_alpha=0.3, departure_window=20),
        storage=StorageConfig(database=str(tmp_path / "test.db"), snapshot_interval=10),
    )


@pytest.fixture
def zone_config() -> ZoneConfig:
    return ZoneConfig(
        name="test_queue",
        polygon=[[100, 100], [600, 100], [600, 500], [100, 500]],
        num_servers=2,
        service_time=25.0,
    )
