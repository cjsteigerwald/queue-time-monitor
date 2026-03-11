"""EMA-smoothed queue count."""

from __future__ import annotations


class QueueCounter:
    """Exponential moving average smoothed queue counter."""

    def __init__(self, alpha: float = 0.3):
        self._alpha = alpha
        self._smoothed: float | None = None

    def update(self, raw_count: int) -> float:
        """Update with a new raw count and return the smoothed value."""
        if self._smoothed is None:
            self._smoothed = float(raw_count)
        else:
            self._smoothed = self._alpha * raw_count + (1 - self._alpha) * self._smoothed
        return self._smoothed

    @property
    def count(self) -> float:
        """Current smoothed count."""
        return self._smoothed if self._smoothed is not None else 0.0

    @property
    def count_rounded(self) -> int:
        """Current smoothed count, rounded to nearest integer."""
        return round(self.count)

    def reset(self) -> None:
        self._smoothed = None
