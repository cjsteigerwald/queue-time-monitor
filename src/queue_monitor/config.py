"""Pydantic configuration models with YAML loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


class VideoConfig(BaseModel):
    source: str = "0"
    width: int = 1280
    height: int = 720
    fps: int = 30


class DetectionConfig(BaseModel):
    model: str = "yolov8n.pt"
    confidence: float = 0.35
    device: str = ""


class ZoneConfig(BaseModel):
    name: str = "main_queue"
    polygon: list[list[int]] = Field(default_factory=list)
    num_servers: int = 1
    service_time: float = 30.0


class EstimationConfig(BaseModel):
    mode: Literal["static", "adaptive"] = "adaptive"
    ema_alpha: float = 0.3
    departure_window: int = 20


class StorageConfig(BaseModel):
    database: str = "queue_monitor.db"
    snapshot_interval: int = 10


class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class AppConfig(BaseModel):
    video: VideoConfig = Field(default_factory=VideoConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    zones: list[ZoneConfig] = Field(default_factory=lambda: [ZoneConfig()])
    estimation: EstimationConfig = Field(default_factory=EstimationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    web: WebConfig = Field(default_factory=WebConfig)


def load_config(path: Path | None = None) -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults."""
    config_path = path or DEFAULT_CONFIG
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        return AppConfig.model_validate(data)
    return AppConfig()


def save_config(config: AppConfig, path: Path) -> None:
    """Save configuration to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
