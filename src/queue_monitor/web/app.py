"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from queue_monitor.web.routes import create_router

if TYPE_CHECKING:
    from queue_monitor.config import AppConfig
    from queue_monitor.pipeline import Pipeline

WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATE_DIR = WEB_DIR / "templates"


def create_app(pipeline: Pipeline | None = None, config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Queue Time Monitor", version="0.1.0")

    templates = Jinja2Templates(directory=str(TEMPLATE_DIR))
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    router = create_router(pipeline, config, templates)
    app.include_router(router)

    return app
