"""Tests for ByteTrack queue tracker."""

import time
from unittest.mock import patch

import numpy as np
import supervision as sv

from queue_monitor.detection.tracker import QueueTracker


def _make_tracked_detections(tracker_ids: list[int], bboxes=None) -> sv.Detections:
    n = len(tracker_ids)
    if bboxes is None:
        bboxes = np.array(
            [[i * 100, i * 100, i * 100 + 50, i * 100 + 50] for i in range(n)],
            dtype=np.float32,
        )
    dets = sv.Detections(
        xyxy=bboxes,
        confidence=np.full(n, 0.9, dtype=np.float32),
        class_id=np.zeros(n, dtype=int),
    )
    dets.tracker_id = np.array(tracker_ids, dtype=int)
    return dets


def test_tracker_records_entries():
    tracker = QueueTracker()
    dets = sv.Detections(
        xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=int),
    )

    with patch.object(tracker, "_tracker") as mock_bt:
        result = _make_tracked_detections([1])
        mock_bt.update_with_detections.return_value = result
        tracker.update(dets)

    assert tracker.active_count == 1


def test_tracker_records_departures():
    tracker = QueueTracker()
    dets = sv.Detections(
        xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=int),
    )

    with patch.object(tracker, "_tracker") as mock_bt:
        # Frame 1: person enters
        mock_bt.update_with_detections.return_value = _make_tracked_detections([1])
        tracker.update(dets)

        # Frame 2: person leaves
        empty = sv.Detections.empty()
        empty.tracker_id = np.array([], dtype=int)
        mock_bt.update_with_detections.return_value = empty
        tracker.update(sv.Detections.empty())

    assert tracker.active_count == 0
    departures = tracker.pop_departures()
    assert len(departures) == 1
    assert departures[0] >= 0


def test_tracker_dwell_times():
    tracker = QueueTracker()
    dets = sv.Detections(
        xyxy=np.array([[100, 100, 200, 200]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=int),
    )

    with patch.object(tracker, "_tracker") as mock_bt:
        mock_bt.update_with_detections.return_value = _make_tracked_detections([1])
        tracker.update(dets)
        time.sleep(0.05)
        dwells = tracker.dwell_times
        assert 1 in dwells
        assert dwells[1] >= 0.04


def test_tracker_reset():
    tracker = QueueTracker()
    tracker._active_ids = {1, 2}
    tracker._entry_times = {1: 0.0, 2: 1.0}
    tracker.reset()
    assert tracker.active_count == 0
    assert tracker.pop_departures() == []


def test_departed_list_bounded():
    """Departed list never exceeds max_departures."""
    tracker = QueueTracker(max_departures=5)

    with patch.object(tracker, "_tracker") as mock_bt:
        for i in range(10):
            # Person enters
            mock_bt.update_with_detections.return_value = _make_tracked_detections([i])
            tracker.update(
                sv.Detections(
                    xyxy=np.array([[0, 0, 50, 50]], dtype=np.float32),
                    confidence=np.array([0.9], dtype=np.float32),
                    class_id=np.array([0], dtype=int),
                )
            )
            # Person leaves
            empty = sv.Detections.empty()
            empty.tracker_id = np.array([], dtype=int)
            mock_bt.update_with_detections.return_value = empty
            tracker.update(sv.Detections.empty())

    assert len(tracker._departed) <= 5


def test_departed_list_keeps_newest():
    """When capped, the most recent dwell times are kept."""
    tracker = QueueTracker(max_departures=3)

    # Directly populate to test trimming behavior
    tracker._departed = [1.0, 2.0, 3.0, 4.0, 5.0]
    # Simulate an update that triggers trimming
    with patch.object(tracker, "_tracker") as mock_bt:
        # Person enters then leaves to trigger the trim path
        mock_bt.update_with_detections.return_value = _make_tracked_detections([99])
        tracker.update(
            sv.Detections(
                xyxy=np.array([[0, 0, 50, 50]], dtype=np.float32),
                confidence=np.array([0.9], dtype=np.float32),
                class_id=np.array([0], dtype=int),
            )
        )
        empty = sv.Detections.empty()
        empty.tracker_id = np.array([], dtype=int)
        mock_bt.update_with_detections.return_value = empty
        tracker.update(sv.Detections.empty())

    departed = tracker._departed
    assert len(departed) == 3
    # Should keep the 3 newest: 5.0, and the new dwell time for track 99
    assert 1.0 not in departed
    assert 2.0 not in departed
    assert 3.0 not in departed
