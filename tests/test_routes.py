"""Tests for web API routes."""

import json
import threading
from unittest.mock import MagicMock, PropertyMock

import numpy as np
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from queue_monitor.web.routes import create_router

TEMPLATES_DIR = "src/queue_monitor/web/templates"


def _make_app(pipeline=None, config=None):
    app = FastAPI()
    templates = Jinja2Templates(directory=TEMPLATES_DIR)
    router = create_router(pipeline, config, templates)
    app.include_router(router)
    return app


def test_toggle_endpoint():
    pipeline = MagicMock()
    pipeline.toggle_pause.return_value = True
    app = _make_app(pipeline=pipeline)
    client = TestClient(app)

    resp = client.post("/api/pipeline/toggle")
    assert resp.status_code == 200
    data = resp.json()
    assert data["paused"] is True
    pipeline.toggle_pause.assert_called_once()


def test_toggle_endpoint_resume():
    pipeline = MagicMock()
    pipeline.toggle_pause.return_value = False
    app = _make_app(pipeline=pipeline)
    client = TestClient(app)

    resp = client.post("/api/pipeline/toggle")
    assert resp.status_code == 200
    assert resp.json()["paused"] is False


def test_status_endpoint():
    pipeline = MagicMock()
    type(pipeline).is_paused = PropertyMock(return_value=False)
    type(pipeline).is_running = PropertyMock(return_value=True)
    app = _make_app(pipeline=pipeline)
    client = TestClient(app)

    resp = client.get("/api/pipeline/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["paused"] is False
    assert data["running"] is True


def test_toggle_without_pipeline():
    app = _make_app(pipeline=None)
    client = TestClient(app)

    resp = client.post("/api/pipeline/toggle")
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data


def test_status_without_pipeline():
    app = _make_app(pipeline=None)
    client = TestClient(app)

    resp = client.get("/api/pipeline/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["paused"] is False
    assert data["running"] is False


def _make_metric(
    zone_name="zone1",
    raw_count=5,
    smoothed_count=4.5,
    wait_time=30.0,
    estimation_mode="bayesian",
    service_time=60.0,
):
    m = MagicMock()
    m.zone_name = zone_name
    m.raw_count = raw_count
    m.smoothed_count = smoothed_count
    m.wait_time = wait_time
    m.estimation_mode = estimation_mode
    m.service_time = service_time
    return m


def test_concurrent_frame_writes_and_reads():
    """Writer thread rapidly updates frames while reader reads via WebSocket."""
    pipeline = MagicMock()
    type(pipeline).is_paused = PropertyMock(return_value=False)
    frame_callback = None

    def capture_on_frame(cb):
        nonlocal frame_callback
        frame_callback = cb

    pipeline.on_frame.side_effect = capture_on_frame
    pipeline.on_metrics.side_effect = lambda cb: None

    app = _make_app(pipeline=pipeline)
    assert frame_callback is not None

    errors = []
    stop = threading.Event()

    def writer():
        """Simulate pipeline thread writing frames."""
        try:
            for i in range(100):
                if stop.is_set():
                    break
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                frame[:] = i % 256
                frame_callback(frame)
        except Exception as e:
            errors.append(e)

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()

    client = TestClient(app)
    with client.websocket_connect("/ws/video") as ws:
        # Read a few frames while writer is running
        for _ in range(5):
            data = ws.receive_bytes()
            assert isinstance(data, bytes)
            assert len(data) > 0

    stop.set()
    writer_thread.join(timeout=5)
    assert not errors, f"Writer thread raised: {errors}"


def test_concurrent_metrics_writes_and_reads():
    """Writer thread rapidly updates metrics while reader reads via WebSocket."""
    pipeline = MagicMock()
    type(pipeline).is_paused = PropertyMock(return_value=False)
    metrics_callback = None

    def capture_on_metrics(cb):
        nonlocal metrics_callback
        metrics_callback = cb

    pipeline.on_frame.side_effect = lambda cb: None
    pipeline.on_metrics.side_effect = capture_on_metrics

    app = _make_app(pipeline=pipeline)
    assert metrics_callback is not None

    errors = []
    stop = threading.Event()

    def writer():
        """Simulate pipeline thread writing metrics."""
        try:
            for i in range(100):
                if stop.is_set():
                    break
                metrics_callback([_make_metric(raw_count=i)])
        except Exception as e:
            errors.append(e)

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()

    client = TestClient(app)
    with client.websocket_connect("/ws/metrics") as ws:
        data = ws.receive_text()
        parsed = json.loads(data)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert "zone_name" in parsed[0]

    stop.set()
    writer_thread.join(timeout=5)
    assert not errors, f"Writer thread raised: {errors}"


def test_lock_prevents_partial_metrics():
    """Each read returns either None or a complete valid metrics list."""
    pipeline = MagicMock()
    type(pipeline).is_paused = PropertyMock(return_value=False)
    metrics_callback = None

    def capture_on_metrics(cb):
        nonlocal metrics_callback
        metrics_callback = cb

    pipeline.on_frame.side_effect = lambda cb: None
    pipeline.on_metrics.side_effect = capture_on_metrics

    app = _make_app(pipeline=pipeline)
    assert metrics_callback is not None

    errors = []
    stop = threading.Event()
    expected_fields = {
        "zone_name",
        "raw_count",
        "smoothed_count",
        "wait_time",
        "estimation_mode",
        "service_time",
        "timestamp",
    }

    def writer():
        try:
            for i in range(200):
                if stop.is_set():
                    break
                metrics_callback(
                    [_make_metric(zone_name=f"zone_{j}", raw_count=i) for j in range(3)]
                )
        except Exception as e:
            errors.append(e)

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()

    client = TestClient(app)
    with client.websocket_connect("/ws/metrics") as ws:
        for _ in range(3):
            data = ws.receive_text()
            parsed = json.loads(data)
            assert isinstance(parsed, list)
            # Must be a complete list of 3 metrics, never partial
            assert len(parsed) == 3
            for item in parsed:
                assert set(item.keys()) == expected_fields

    stop.set()
    writer_thread.join(timeout=5)
    assert not errors, f"Writer thread raised: {errors}"
