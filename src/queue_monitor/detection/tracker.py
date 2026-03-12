"""ByteTrack wrapper with dwell time tracking."""

from __future__ import annotations

import time

import supervision as sv


class QueueTracker:
    """Track people across frames and record dwell times."""

    def __init__(self, max_departures: int = 1000):
        self._tracker = sv.ByteTrack()
        self._entry_times: dict[int, float] = {}
        self._active_ids: set[int] = set()
        self._departed: list[float] = []  # dwell times of departed tracks
        self._max_departures = max_departures

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracker with new detections, return tracked detections."""
        tracked = self._tracker.update_with_detections(detections)
        now = time.monotonic()

        current_ids = set()
        if tracked.tracker_id is not None:
            current_ids = set(tracked.tracker_id.tolist())

        # Record entry times for new tracks
        for tid in current_ids - self._active_ids:
            self._entry_times[tid] = now

        # Record departures
        for tid in self._active_ids - current_ids:
            if tid in self._entry_times:
                dwell = now - self._entry_times.pop(tid)
                self._departed.append(dwell)

        # Cap departed list to prevent unbounded memory growth
        if len(self._departed) > self._max_departures:
            self._departed = self._departed[-self._max_departures :]

        self._active_ids = current_ids
        return tracked

    @property
    def active_count(self) -> int:
        return len(self._active_ids)

    @property
    def dwell_times(self) -> dict[int, float]:
        """Current dwell times for active tracks."""
        now = time.monotonic()
        return {
            tid: now - self._entry_times[tid]
            for tid in self._active_ids
            if tid in self._entry_times
        }

    def pop_departures(self) -> list[float]:
        """Return and clear the list of departed dwell times since last call."""
        departed = self._departed.copy()
        self._departed.clear()
        return departed

    def reset(self) -> None:
        self._entry_times.clear()
        self._active_ids.clear()
        self._departed.clear()
