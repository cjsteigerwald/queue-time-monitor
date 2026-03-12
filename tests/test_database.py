"""Tests for SQLite metrics database."""

from queue_monitor.storage.database import MetricsDatabase


def _record_sample(db: MetricsDatabase, zone: str = "test_queue", count: int = 5):
    db.record(
        zone_name=zone,
        queue_count=count,
        smoothed_count=4.8,
        wait_time_seconds=30.0,
        estimation_mode="static",
        service_time=25.0,
    )


def test_open_creates_tables(tmp_path):
    db = MetricsDatabase(str(tmp_path / "test.db"))
    db.open()
    # Verify table exists by querying it
    rows = db._conn.execute("SELECT count(*) FROM metrics").fetchone()
    assert rows[0] == 0
    db.close()


def test_record_inserts_row(tmp_path):
    db = MetricsDatabase(str(tmp_path / "test.db"))
    db.open()
    _record_sample(db)
    rows = db._conn.execute("SELECT count(*) FROM metrics").fetchone()
    assert rows[0] == 1
    db.close()


def test_record_stores_correct_values(tmp_path):
    db = MetricsDatabase(str(tmp_path / "test.db"))
    db.open()
    db.record(
        zone_name="lobby",
        queue_count=3,
        smoothed_count=2.7,
        wait_time_seconds=45.5,
        estimation_mode="adaptive",
        service_time=15.0,
    )
    row = db._conn.execute("SELECT * FROM metrics WHERE id=1").fetchone()
    assert row["zone_name"] == "lobby"
    assert row["queue_count"] == 3
    assert row["smoothed_count"] == 2.7
    assert row["wait_time_seconds"] == 45.5
    assert row["estimation_mode"] == "adaptive"
    assert row["service_time"] == 15.0
    assert row["timestamp"] is not None
    db.close()


def test_get_history_returns_all(tmp_path):
    db = MetricsDatabase(str(tmp_path / "test.db"))
    db.open()
    for i in range(5):
        _record_sample(db, count=i)
    history = db.get_history()
    assert len(history) == 5
    db.close()


def test_get_history_filters_by_zone(tmp_path):
    db = MetricsDatabase(str(tmp_path / "test.db"))
    db.open()
    _record_sample(db, zone="zone_a")
    _record_sample(db, zone="zone_a")
    _record_sample(db, zone="zone_b")
    history = db.get_history(zone_name="zone_a")
    assert len(history) == 2
    assert all(r["zone_name"] == "zone_a" for r in history)
    db.close()


def test_get_history_respects_limit(tmp_path):
    db = MetricsDatabase(str(tmp_path / "test.db"))
    db.open()
    for _ in range(10):
        _record_sample(db)
    history = db.get_history(limit=3)
    assert len(history) == 3
    db.close()


def test_get_history_returns_dicts(tmp_path):
    db = MetricsDatabase(str(tmp_path / "test.db"))
    db.open()
    _record_sample(db)
    history = db.get_history()
    assert isinstance(history[0], dict)
    assert "zone_name" in history[0]
    assert "timestamp" in history[0]
    db.close()


def test_get_history_chronological_order(tmp_path):
    db = MetricsDatabase(str(tmp_path / "test.db"))
    db.open()
    _record_sample(db, count=1)
    _record_sample(db, count=2)
    _record_sample(db, count=3)
    history = db.get_history()
    counts = [r["queue_count"] for r in history]
    assert counts == [1, 2, 3]
    db.close()


def test_get_history_empty(tmp_path):
    db = MetricsDatabase(str(tmp_path / "test.db"))
    db.open()
    history = db.get_history()
    assert history == []
    db.close()


def test_close_sets_conn_none(tmp_path):
    db = MetricsDatabase(str(tmp_path / "test.db"))
    db.open()
    db.close()
    assert db._conn is None


def test_close_idempotent(tmp_path):
    db = MetricsDatabase(str(tmp_path / "test.db"))
    db.open()
    db.close()
    db.close()  # Should not raise


def test_context_manager(tmp_path):
    db_path = str(tmp_path / "test.db")
    with MetricsDatabase(db_path) as db:
        _record_sample(db)
        history = db.get_history()
        assert len(history) == 1
    assert db._conn is None


def test_session_context_manager(tmp_path):
    db = MetricsDatabase(str(tmp_path / "test.db"))
    with db.session() as s:
        _record_sample(s)
        assert len(s.get_history()) == 1
    assert db._conn is None


def test_multiple_records(tmp_path):
    db = MetricsDatabase(str(tmp_path / "test.db"))
    db.open()
    for i in range(100):
        _record_sample(db, count=i)
    history = db.get_history(limit=1000)
    assert len(history) == 100
    db.close()
