"""Tests for web API routes."""

from unittest.mock import MagicMock, PropertyMock

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
