"""Video source abstraction for files, webcams, and RTSP streams."""

from __future__ import annotations

import time

import cv2
import numpy as np
import structlog

from queue_monitor.config import VideoConfig

logger = structlog.get_logger()


class VideoSource:
    """Unified video capture for files, webcams, and RTSP streams."""

    def __init__(self, config: VideoConfig, max_retries: int = 5):
        self._config = config
        self._max_retries = max_retries
        self._cap: cv2.VideoCapture | None = None
        self._source = self._parse_source(config.source)

    @staticmethod
    def _parse_source(source: str) -> int | str:
        """Convert source string to int (webcam) or leave as path/URL."""
        try:
            return int(source)
        except ValueError:
            return source

    def open(self) -> None:
        """Open the video source."""
        self.release()
        self._cap = cv2.VideoCapture(self._source)
        if isinstance(self._source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self._source}")
        logger.info("video_source_opened", source=self._source)

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read a frame, reconnecting on failure for RTSP streams."""
        if self._cap is None:
            self.open()
        ret, frame = self._cap.read()
        if not ret and isinstance(self._source, str) and self._source.startswith("rtsp"):
            frame = self._reconnect()
            ret = frame is not None
        return ret, frame

    def _reconnect(self) -> np.ndarray | None:
        """Reconnect to RTSP stream with exponential backoff."""
        for attempt in range(self._max_retries):
            delay = min(2**attempt, 30)
            logger.warning("rtsp_reconnecting", attempt=attempt + 1, delay=delay)
            time.sleep(delay)
            self.release()
            try:
                self.open()
                ret, frame = self._cap.read()
                if ret:
                    logger.info("rtsp_reconnected", attempt=attempt + 1)
                    return frame
            except RuntimeError:
                continue
        logger.error("rtsp_reconnect_failed", max_retries=self._max_retries)
        return None

    @property
    def fps(self) -> float:
        if self._cap is None:
            return self._config.fps
        return self._cap.get(cv2.CAP_PROP_FPS) or self._config.fps

    @property
    def frame_size(self) -> tuple[int, int]:
        if self._cap is None:
            return self._config.width, self._config.height
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        ret, frame = self.read()
        if not ret or frame is None:
            # Loop file-based sources by releasing and reopening
            is_file = isinstance(self._source, str) and not self._source.startswith("rtsp")
            if self._cap is not None and is_file:
                self._cap.release()
                self._cap = cv2.VideoCapture(self._source)
                ret, frame = self._cap.read()
                if ret and frame is not None:
                    return frame
            raise StopIteration
        return frame
