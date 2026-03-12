"""SQLite storage for historical queue metrics."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class MetricsDatabase:
    """SQLite database for storing queue metrics history."""

    def __init__(self, db_path: str = "queue_monitor.db"):
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                zone_name TEXT NOT NULL,
                queue_count INTEGER NOT NULL,
                smoothed_count REAL NOT NULL,
                wait_time_seconds REAL NOT NULL,
                estimation_mode TEXT NOT NULL,
                service_time REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
            ON metrics (timestamp)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_metrics_zone
            ON metrics (zone_name, timestamp)
        """)
        self._conn.commit()

    def record(
        self,
        zone_name: str,
        queue_count: int,
        smoothed_count: float,
        wait_time_seconds: float,
        estimation_mode: str,
        service_time: float,
    ) -> None:
        """Record a metrics snapshot."""
        self._conn.execute(
            """
            INSERT INTO metrics
                (timestamp, zone_name, queue_count, smoothed_count,
                 wait_time_seconds, estimation_mode, service_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(UTC).isoformat(),
                zone_name,
                queue_count,
                smoothed_count,
                wait_time_seconds,
                estimation_mode,
                service_time,
            ),
        )
        self._conn.commit()

    def get_history(
        self,
        zone_name: str | None = None,
        minutes: int = 60,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Get recent metrics history."""
        minutes = int(minutes)
        query = "SELECT * FROM metrics WHERE 1=1"
        params: list[Any] = []

        if zone_name:
            query += " AND zone_name = ?"
            params.append(zone_name)

        query += " AND timestamp >= datetime('now', '-' || ? || ' minutes')"
        params.append(minutes)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in reversed(rows)]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def session(self):
        self.open()
        try:
            yield self
        finally:
            self.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()
