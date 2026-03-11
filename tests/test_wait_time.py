"""Tests for wait time estimation."""

from queue_monitor.estimation.wait_time import WaitTimeEstimator


def test_static_mode():
    est = WaitTimeEstimator(mode="static", service_time=30.0, num_servers=1)
    assert est.estimate(0) == 0.0
    assert est.estimate(3) == 90.0
    assert est.mode_active == "static"


def test_static_mode_multi_server():
    est = WaitTimeEstimator(mode="static", service_time=30.0, num_servers=3)
    assert est.estimate(6) == 60.0


def test_adaptive_mode_fallback():
    est = WaitTimeEstimator(mode="adaptive", service_time=30.0)
    # No departures recorded yet — should fall back to static
    assert est.estimate(2) == 60.0
    assert est.mode_active == "static"


def test_adaptive_mode_with_data():
    est = WaitTimeEstimator(mode="adaptive", service_time=30.0, num_servers=1, departure_window=10)
    # Record departures with ~20s dwell times
    est.record_departures([18.0, 22.0, 20.0])
    assert est.mode_active == "adaptive"
    # avg service time = 20, queue of 3 => 60s
    result = est.estimate(3)
    assert abs(result - 60.0) < 1.0


def test_adaptive_multi_server():
    est = WaitTimeEstimator(mode="adaptive", service_time=30.0, num_servers=2, departure_window=10)
    est.record_departures([20.0, 20.0, 20.0])
    # queue 4, service 20s, 2 servers => 40s
    assert est.estimate(4) == 40.0


def test_empty_queue():
    est = WaitTimeEstimator(mode="adaptive", service_time=30.0)
    est.record_departures([10.0, 15.0])
    assert est.estimate(0) == 0.0


def test_reset():
    est = WaitTimeEstimator(mode="adaptive", service_time=30.0)
    est.record_departures([10.0, 15.0])
    est.reset()
    assert est.mode_active == "static"
    assert est.service_time == 30.0
