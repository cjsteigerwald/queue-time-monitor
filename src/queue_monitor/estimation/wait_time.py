"""Wait time estimation with static and adaptive modes."""

from __future__ import annotations

from collections import deque
from typing import Literal


class WaitTimeEstimator:
    """Estimate queue wait time using static or adaptive methods."""

    def __init__(
        self,
        mode: Literal["static", "adaptive"] = "adaptive",
        service_time: float = 30.0,
        num_servers: int = 1,
        departure_window: int = 20,
    ):
        self._mode = mode
        self._static_service_time = service_time
        self._num_servers = max(1, num_servers)
        self._departure_times: deque[float] = deque(maxlen=departure_window)
        self._adaptive_service_time: float | None = None

    def record_departures(self, dwell_times: list[float]) -> None:
        """Record observed dwell times from departed tracks."""
        self._departure_times.extend(dwell_times)
        if len(self._departure_times) >= 2:
            self._adaptive_service_time = sum(self._departure_times) / len(self._departure_times)

    def estimate(self, queue_length: int | float) -> float:
        """Estimate wait time in seconds for someone joining the queue now."""
        if queue_length <= 0:
            return 0.0

        service_time = self._get_service_time()
        return (queue_length * service_time) / self._num_servers

    def _get_service_time(self) -> float:
        """Get per-person service time based on mode."""
        if self._mode == "adaptive" and self._adaptive_service_time is not None:
            return self._adaptive_service_time
        return self._static_service_time

    @property
    def service_time(self) -> float:
        return self._get_service_time()

    @property
    def mode_active(self) -> str:
        """Return which mode is actually in use (adaptive falls back to static)."""
        if self._mode == "adaptive" and self._adaptive_service_time is not None:
            return "adaptive"
        return "static"

    def reset(self) -> None:
        self._departure_times.clear()
        self._adaptive_service_time = None
