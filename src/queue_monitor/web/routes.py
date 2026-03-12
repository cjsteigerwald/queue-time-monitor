"""HTTP and WebSocket routes for the dashboard."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import TYPE_CHECKING

import cv2
from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from queue_monitor.storage.database import MetricsDatabase

if TYPE_CHECKING:
    from queue_monitor.config import AppConfig
    from queue_monitor.pipeline import Pipeline


def create_router(
    pipeline: Pipeline | None,
    config: AppConfig | None,
    templates: Jinja2Templates,
) -> APIRouter:
    router = APIRouter()

    # Shared state for WebSocket broadcasting
    _latest_frame: dict = {"data": None}
    _latest_metrics: dict = {"data": None}
    _state_lock = threading.Lock()

    if pipeline is not None:

        def on_frame(frame):
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            data = buf.tobytes()
            with _state_lock:
                _latest_frame["data"] = data

        def on_metrics(metrics_list):
            payload = [
                {
                    "zone_name": m.zone_name,
                    "raw_count": m.raw_count,
                    "smoothed_count": m.smoothed_count,
                    "wait_time": m.wait_time,
                    "estimation_mode": m.estimation_mode,
                    "service_time": m.service_time,
                    "timestamp": time.time(),
                }
                for m in metrics_list
            ]
            with _state_lock:
                _latest_metrics["data"] = payload

        pipeline.on_frame(on_frame)
        pipeline.on_metrics(on_metrics)

    @router.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @router.get("/configure", response_class=HTMLResponse)
    async def configure_page(request: Request):
        return templates.TemplateResponse("configure.html", {"request": request})

    def _is_paused() -> bool:
        return pipeline is not None and pipeline.is_paused

    @router.websocket("/ws/video")
    async def ws_video(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                with _state_lock:
                    frame_data = _latest_frame["data"]
                if not _is_paused() and frame_data is not None:
                    await websocket.send_bytes(frame_data)
                await asyncio.sleep(0.033)  # ~30 FPS
        except WebSocketDisconnect:
            pass

    @router.websocket("/ws/metrics")
    async def ws_metrics(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                with _state_lock:
                    metrics_data = _latest_metrics["data"]
                if not _is_paused() and metrics_data is not None:
                    await websocket.send_text(json.dumps(metrics_data))
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            pass

    @router.post("/api/pipeline/toggle")
    async def toggle_pipeline():
        if pipeline is None:
            return {"error": "No pipeline available", "paused": False}
        paused = pipeline.toggle_pause()
        return {"paused": paused}

    @router.get("/api/pipeline/status")
    async def pipeline_status():
        if pipeline is None:
            return {"paused": False, "running": False, "error": None}
        error = pipeline.error
        return {
            "paused": pipeline.is_paused,
            "running": pipeline.is_running,
            "error": {
                "message": error.message,
                "traceback": error.traceback,
                "timestamp": error.timestamp,
            }
            if error
            else None,
        }

    @router.get("/api/history")
    async def get_history(
        zone: str | None = Query(None),
        minutes: int = Query(60, ge=1, le=1440),
    ):
        if config is None:
            return {"data": []}
        db = MetricsDatabase(config.storage.database)
        with db:
            return {"data": db.get_history(zone_name=zone, minutes=minutes)}

    return router
