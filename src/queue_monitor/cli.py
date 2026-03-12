"""Typer CLI for queue-monitor commands."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from queue_monitor.config import DEFAULT_CONFIG, load_config, save_config

app = typer.Typer(
    name="queue-monitor",
    help="Video-based queue monitoring and wait time estimation.",
)


@app.command()
def run(
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="Video source (file/webcam/RTSP)",
    ),
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Path to config YAML"),
    show: bool = typer.Option(False, "--show", help="Show OpenCV window with annotated frames"),
    web: bool = typer.Option(False, "--web", help="Start web dashboard"),
) -> None:
    """Run the queue monitoring pipeline."""
    cfg = load_config(config)
    if source is not None:
        cfg.video.source = source

    if web:
        _run_with_web(cfg, show)
    else:
        from queue_monitor.pipeline import Pipeline

        pipeline = Pipeline(cfg)
        pipeline.run(show_window=show)


def _run_with_web(cfg, show: bool) -> None:
    """Run pipeline alongside the web dashboard."""
    import threading

    import uvicorn

    from queue_monitor.pipeline import Pipeline
    from queue_monitor.web.app import create_app

    pipeline = Pipeline(cfg)
    web_app = create_app(pipeline, cfg)

    # Run pipeline in background thread
    def _run_pipeline():
        try:
            pipeline.run(show_window=show)
        except Exception:
            import traceback
            traceback.print_exc()

    pipeline_thread = threading.Thread(target=_run_pipeline, daemon=True)
    pipeline_thread.start()

    uvicorn.run(web_app, host=cfg.web.host, port=cfg.web.port)


@app.command()
def configure(
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="Video source to configure zones on",
    ),
    config: Path = typer.Option(
        DEFAULT_CONFIG, "--config", "-c", help="Path to config YAML",
    ),
    zone_name: str = typer.Option(
        "main_queue", "--zone", "-z", help="Zone name to configure",
    ),
) -> None:
    """Open an interactive window to define queue zone polygons."""
    import cv2
    import numpy as np

    cfg = load_config(config)
    if source is not None:
        cfg.video.source = source

    from queue_monitor.video.source import VideoSource

    vs = VideoSource(cfg.video)
    vs.open()
    ret, frame = vs.read()
    vs.release()

    if not ret or frame is None:
        typer.echo("Error: Could not read a frame from the video source.")
        raise typer.Exit(1)

    points: list[list[int]] = []
    display = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            display = frame.copy()
            pts = np.array(points, dtype=np.int32)
            if len(pts) > 1:
                cv2.polylines(display, [pts], isClosed=False, color=(0, 255, 0), thickness=2)
            for p in points:
                cv2.circle(display, tuple(p), 5, (0, 0, 255), -1)

    cv2.namedWindow("Configure Zone")
    cv2.setMouseCallback("Configure Zone", mouse_callback)

    typer.echo("Click to add polygon vertices. Press 'Enter' to save, 'r' to reset, 'q' to quit.")

    while True:
        cv2.imshow("Configure Zone", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Enter
            if len(points) >= 3:
                break
            typer.echo("Need at least 3 points for a polygon.")
        elif key == ord("r"):
            points.clear()
            display = frame.copy()
        elif key == ord("q"):
            cv2.destroyAllWindows()
            raise typer.Exit(0)

    cv2.destroyAllWindows()

    # Save polygon to config
    zone_found = False
    for zone in cfg.zones:
        if zone.name == zone_name:
            zone.polygon = points
            zone_found = True
            break

    if not zone_found:
        from queue_monitor.config import ZoneConfig
        cfg.zones.append(ZoneConfig(name=zone_name, polygon=points))

    save_config(cfg, config)
    typer.echo(f"Zone '{zone_name}' saved with {len(points)} vertices to {config}")


if __name__ == "__main__":
    app()
