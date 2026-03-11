"""Tests for configuration loading and saving."""

from pathlib import Path

import yaml

from queue_monitor.config import AppConfig, ZoneConfig, load_config, save_config


def test_default_config():
    config = AppConfig()
    assert config.video.width == 1280
    assert config.detection.confidence == 0.35
    assert len(config.zones) == 1
    assert config.estimation.ema_alpha == 0.3


def test_load_config_from_file(tmp_path: Path):
    config_data = {
        "video": {"source": "test.mp4", "width": 640, "height": 480},
        "detection": {"confidence": 0.5},
        "zones": [
            {"name": "my_queue", "polygon": [[0, 0], [100, 0], [100, 100]], "num_servers": 3},
        ],
    }
    config_file = tmp_path / "test.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    config = load_config(config_file)
    assert config.video.source == "test.mp4"
    assert config.video.width == 640
    assert config.detection.confidence == 0.5
    assert config.zones[0].name == "my_queue"
    assert config.zones[0].num_servers == 3


def test_load_missing_file():
    config = load_config(Path("/nonexistent/config.yaml"))
    assert config == AppConfig()


def test_save_and_reload(tmp_path: Path):
    config = AppConfig(
        zones=[ZoneConfig(name="saved_zone", polygon=[[10, 20], [30, 40], [50, 60]])]
    )
    config_file = tmp_path / "saved.yaml"
    save_config(config, config_file)

    reloaded = load_config(config_file)
    assert reloaded.zones[0].name == "saved_zone"
    assert reloaded.zones[0].polygon == [[10, 20], [30, 40], [50, 60]]
