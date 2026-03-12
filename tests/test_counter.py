"""Tests for EMA-smoothed queue counter."""

from queue_monitor.estimation.counter import QueueCounter


def test_first_update_initializes_to_raw():
    counter = QueueCounter(alpha=0.3)
    result = counter.update(5)
    assert result == 5.0


def test_ema_smoothing():
    counter = QueueCounter(alpha=0.5)
    counter.update(10)
    result = counter.update(20)
    # EMA: 0.5 * 20 + 0.5 * 10 = 15
    assert result == 15.0


def test_ema_with_low_alpha_favors_history():
    counter = QueueCounter(alpha=0.1)
    counter.update(100)
    result = counter.update(0)
    # 0.1 * 0 + 0.9 * 100 = 90
    assert result == 90.0


def test_ema_with_alpha_one_tracks_raw():
    counter = QueueCounter(alpha=1.0)
    counter.update(10)
    result = counter.update(50)
    assert result == 50.0


def test_ema_with_alpha_zero_stays_at_initial():
    counter = QueueCounter(alpha=0.0)
    counter.update(10)
    result = counter.update(50)
    assert result == 10.0


def test_count_before_any_update():
    counter = QueueCounter()
    assert counter.count == 0.0


def test_count_after_update():
    counter = QueueCounter()
    counter.update(7)
    assert counter.count == 7.0


def test_count_rounded():
    counter = QueueCounter(alpha=0.5)
    counter.update(3)
    counter.update(4)
    # smoothed = 0.5 * 4 + 0.5 * 3 = 3.5, rounds to 4
    assert counter.count_rounded == 4


def test_count_rounded_down():
    counter = QueueCounter(alpha=0.3)
    counter.update(10)
    counter.update(11)
    # smoothed = 0.3 * 11 + 0.7 * 10 = 10.3, rounds to 10
    assert counter.count_rounded == 10


def test_reset_clears_state():
    counter = QueueCounter()
    counter.update(10)
    counter.reset()
    assert counter.count == 0.0
    assert counter.count_rounded == 0


def test_update_after_reset_reinitializes():
    counter = QueueCounter(alpha=0.5)
    counter.update(100)
    counter.reset()
    result = counter.update(5)
    assert result == 5.0


def test_multiple_updates_converge():
    counter = QueueCounter(alpha=0.3)
    for _ in range(50):
        counter.update(10)
    assert abs(counter.count - 10.0) < 0.01


def test_zero_count():
    counter = QueueCounter()
    result = counter.update(0)
    assert result == 0.0
    assert counter.count_rounded == 0
