"""Main processing pipeline — ties detection, tracking, and estimation together."""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import structlog
import supervision as sv

from queue_monitor.config import AppConfig
from queue_monitor.detection.detector import PersonDetector
from queue_monitor.detection.tracker import QueueTracker
from queue_monitor.detection.zone import QueueZone
from queue_monitor.estimation.counter import QueueCounter
from queue_monitor.estimation.wait_time import WaitTimeEstimator
from queue_monitor.storage.database import MetricsDatabase
from queue_monitor.video.source import VideoSource

logger = structlog.get_logger()


@dataclass
class ZoneState:
    """Per-zone processing state."""

    zone: QueueZone
    tracker: QueueTracker
    counter: QueueCounter
    estimator: WaitTimeEstimator
    last_snapshot: float = 0.0


@dataclass
class FrameMetrics:
    """Metrics for a single processed frame."""

    zone_name: str
    raw_count: int
    smoothed_count: float
    wait_time: float
    estimation_mode: str
    service_time: float


@dataclass
class PipelineResult:
    """Result of processing a single frame."""

    annotated_frame: np.ndarray
    metrics: list[FrameMetrics] = field(default_factory=list)
    fps: float = 0.0


class Pipeline:
    """Main video processing pipeline."""

    def __init__(self, config: AppConfig):
        self._config = config
        self._running = False
        self._paused = threading.Event()  # unset = not paused
        self._pause_lock = threading.Lock()
        self._source = VideoSource(config.video)
        self._detector = PersonDetector(config.detection)
        self._db = MetricsDatabase(config.storage.database)
        self._zones: list[ZoneState] = []

        # Annotators
        self._box_annotator = sv.BoxAnnotator(thickness=2)
        self._label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

        # Callbacks for live streaming
        self._frame_callbacks: list = []
        self._metrics_callbacks: list = []

    def _init_zones(self, frame_size: tuple[int, int]) -> None:
        """Initialize zone states after video source is opened."""
        self._zones = []
        for zone_cfg in self._config.zones:
            zone = QueueZone(zone_cfg, frame_size)
            tracker = QueueTracker()
            counter = QueueCounter(alpha=self._config.estimation.ema_alpha)
            estimator = WaitTimeEstimator(
                mode=self._config.estimation.mode,
                service_time=zone_cfg.service_time,
                num_servers=zone_cfg.num_servers,
                departure_window=self._config.estimation.departure_window,
            )
            self._zones.append(ZoneState(
                zone=zone, tracker=tracker, counter=counter, estimator=estimator,
            ))

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def toggle_pause(self) -> bool:
        """Toggle paused state. Returns the new paused state."""
        with self._pause_lock:
            if self._paused.is_set():
                self._paused.clear()
            else:
                self._paused.set()
            return self._paused.is_set()

    def on_frame(self, callback) -> None:
        self._frame_callbacks.append(callback)

    def on_metrics(self, callback) -> None:
        self._metrics_callbacks.append(callback)

    def process_frame(self, frame: np.ndarray) -> PipelineResult:
        """Process a single frame through the pipeline."""
        detections = self._detector.detect(frame)
        annotated = frame.copy()
        all_metrics = []
        now = time.monotonic()

        for zs in self._zones:
            # Filter to zone
            zone_detections = zs.zone.filter(detections)

            # Track
            tracked = zs.tracker.update(zone_detections)

            # Count
            raw_count = zs.tracker.active_count
            smoothed = zs.counter.update(raw_count)

            # Departures → wait time
            departures = zs.tracker.pop_departures()
            if departures:
                zs.estimator.record_departures(departures)
            wait_time = zs.estimator.estimate(smoothed)

            # Build metrics
            metrics = FrameMetrics(
                zone_name=zs.zone.name,
                raw_count=raw_count,
                smoothed_count=round(smoothed, 1),
                wait_time=round(wait_time, 1),
                estimation_mode=zs.estimator.mode_active,
                service_time=round(zs.estimator.service_time, 1),
            )
            all_metrics.append(metrics)

            # Annotate frame
            tracker_ids = tracked.tracker_id if tracked.tracker_id is not None else []
            labels = [f"person #{tid}" for tid in tracker_ids]
            annotated = self._box_annotator.annotate(annotated, tracked)
            annotated = self._label_annotator.annotate(annotated, tracked, labels=labels)

            # Draw zone polygon
            pts = zs.zone.polygon.reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw metrics overlay
            text = f"{zs.zone.name}: {zs.counter.count_rounded} people | ~{wait_time:.0f}s wait"
            cv2.putText(annotated, text, (10, 30 + 40 * self._zones.index(zs)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Snapshot to DB
            if now - zs.last_snapshot >= self._config.storage.snapshot_interval:
                try:
                    self._db.record(
                        zone_name=metrics.zone_name,
                        queue_count=metrics.raw_count,
                        smoothed_count=metrics.smoothed_count,
                        wait_time_seconds=metrics.wait_time,
                        estimation_mode=metrics.estimation_mode,
                        service_time=metrics.service_time,
                    )
                except Exception:
                    logger.exception("db_snapshot_failed")
                zs.last_snapshot = now

        return PipelineResult(annotated_frame=annotated, metrics=all_metrics)

    def run(self, show_window: bool = False) -> None:
        """Run the main processing loop."""
        self._running = True
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except ValueError:
            pass  # Not in main thread (e.g., --web mode)

        self._source.open()
        self._db.open()
        self._init_zones(self._source.frame_size)

        source_fps = self._source.fps
        frame_interval = 1.0 / source_fps if source_fps > 0 else 0.033
        frame_count = 0
        fps_start = time.monotonic()

        logger.info("pipeline_started", source=self._config.video.source, fps=source_fps)

        try:
            for frame in self._source:
                if not self._running:
                    break

                if self._paused.is_set():
                    time.sleep(0.1)
                    continue

                t0 = time.monotonic()
                result = self.process_frame(frame)

                # FPS calculation
                frame_count += 1
                elapsed = time.monotonic() - fps_start
                if elapsed >= 1.0:
                    result.fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.monotonic()

                # Callbacks
                for cb in self._frame_callbacks:
                    cb(result.annotated_frame)
                for cb in self._metrics_callbacks:
                    cb(result.metrics)

                # Print metrics
                for m in result.metrics:
                    logger.info(
                        "queue_metrics",
                        zone=m.zone_name,
                        count=m.raw_count,
                        smoothed=m.smoothed_count,
                        wait_time=m.wait_time,
                        mode=m.estimation_mode,
                    )

                # Display window
                if show_window:
                    cv2.imshow("Queue Monitor", result.annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Frame timing
                proc_time = time.monotonic() - t0
                sleep_time = frame_interval - proc_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            self._cleanup()

    def _handle_signal(self, signum, frame):
        logger.info("shutdown_signal", signal=signum)
        self._running = False

    def _cleanup(self) -> None:
        self._source.release()
        self._db.close()
        cv2.destroyAllWindows()
        logger.info("pipeline_stopped")
