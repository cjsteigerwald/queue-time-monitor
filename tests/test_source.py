"""Tests for video source abstraction (mocked cv2 to avoid needing a camera)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from queue_monitor.config import VideoConfig


@patch("queue_monitor.video.source.cv2")
def test_parse_source_webcam(mock_cv2):
    from queue_monitor.video.source import VideoSource

    vs = VideoSource(VideoConfig(source="0"))
    assert vs._source == 0


@patch("queue_monitor.video.source.cv2")
def test_parse_source_file(mock_cv2):
    from queue_monitor.video.source import VideoSource

    vs = VideoSource(VideoConfig(source="video.mp4"))
    assert vs._source == "video.mp4"


@patch("queue_monitor.video.source.cv2")
def test_parse_source_rtsp(mock_cv2):
    from queue_monitor.video.source import VideoSource

    vs = VideoSource(VideoConfig(source="rtsp://192.168.1.1/stream"))
    assert vs._source == "rtsp://192.168.1.1/stream"


@patch("queue_monitor.video.source.cv2")
def test_open_webcam_sets_properties(mock_cv2):
    from queue_monitor.video.source import VideoSource

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cv2.VideoCapture.return_value = mock_cap

    vs = VideoSource(VideoConfig(source="0", width=640, height=480, fps=15))
    vs.open()

    mock_cap.set.assert_any_call(mock_cv2.CAP_PROP_FRAME_WIDTH, 640)
    mock_cap.set.assert_any_call(mock_cv2.CAP_PROP_FRAME_HEIGHT, 480)
    mock_cap.set.assert_any_call(mock_cv2.CAP_PROP_FPS, 15)


@patch("queue_monitor.video.source.cv2")
def test_open_file_does_not_set_properties(mock_cv2):
    from queue_monitor.video.source import VideoSource

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cv2.VideoCapture.return_value = mock_cap

    vs = VideoSource(VideoConfig(source="video.mp4"))
    vs.open()

    mock_cap.set.assert_not_called()


@patch("queue_monitor.video.source.cv2")
def test_open_raises_on_failure(mock_cv2):
    from queue_monitor.video.source import VideoSource

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_cv2.VideoCapture.return_value = mock_cap

    vs = VideoSource(VideoConfig(source="bad_source"))
    with pytest.raises(RuntimeError, match="Cannot open video source"):
        vs.open()


@patch("queue_monitor.video.source.cv2")
def test_read_returns_frame(mock_cv2):
    from queue_monitor.video.source import VideoSource

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, frame)
    mock_cv2.VideoCapture.return_value = mock_cap

    vs = VideoSource(VideoConfig())
    vs.open()
    ret, result = vs.read()

    assert ret is True
    assert result is not None


@patch("queue_monitor.video.source.cv2")
def test_read_auto_opens(mock_cv2):
    from queue_monitor.video.source import VideoSource

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, frame)
    mock_cv2.VideoCapture.return_value = mock_cap

    vs = VideoSource(VideoConfig())
    ret, result = vs.read()

    assert ret is True
    mock_cv2.VideoCapture.assert_called_once()


@patch("queue_monitor.video.source.cv2")
def test_release_clears_capture(mock_cv2):
    from queue_monitor.video.source import VideoSource

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cv2.VideoCapture.return_value = mock_cap

    vs = VideoSource(VideoConfig())
    vs.open()
    vs.release()

    mock_cap.release.assert_called_once()
    assert vs._cap is None


@patch("queue_monitor.video.source.cv2")
def test_release_noop_when_not_opened(mock_cv2):
    from queue_monitor.video.source import VideoSource

    vs = VideoSource(VideoConfig())
    vs.release()  # Should not raise


@patch("queue_monitor.video.source.cv2")
def test_fps_before_open(mock_cv2):
    from queue_monitor.video.source import VideoSource

    vs = VideoSource(VideoConfig(fps=25))
    assert vs.fps == 25


@patch("queue_monitor.video.source.cv2")
def test_fps_from_capture(mock_cv2):
    from queue_monitor.video.source import VideoSource

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0
    mock_cv2.VideoCapture.return_value = mock_cap

    vs = VideoSource(VideoConfig())
    vs.open()
    assert vs.fps == 30.0


@patch("queue_monitor.video.source.cv2")
def test_frame_size_before_open(mock_cv2):
    from queue_monitor.video.source import VideoSource

    vs = VideoSource(VideoConfig(width=800, height=600))
    assert vs.frame_size == (800, 600)


@patch("queue_monitor.video.source.cv2")
def test_frame_size_from_capture(mock_cv2):
    from queue_monitor.video.source import VideoSource

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: {
        mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
        mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
    }.get(prop, 0)
    mock_cv2.VideoCapture.return_value = mock_cap

    vs = VideoSource(VideoConfig())
    vs.open()
    assert vs.frame_size == (1920, 1080)


@patch("queue_monitor.video.source.cv2")
def test_context_manager(mock_cv2):
    from queue_monitor.video.source import VideoSource

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cv2.VideoCapture.return_value = mock_cap

    with VideoSource(VideoConfig()) as vs:
        assert vs._cap is not None
    mock_cap.release.assert_called_once()


@patch("queue_monitor.video.source.cv2")
def test_iterator_yields_frames(mock_cv2):
    from queue_monitor.video.source import VideoSource

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Return 3 frames then stop
    mock_cap.read.side_effect = [
        (True, frame),
        (True, frame),
        (True, frame),
        (False, None),
    ]
    mock_cv2.VideoCapture.return_value = mock_cap

    vs = VideoSource(VideoConfig())
    vs.open()
    frames = list(vs)
    assert len(frames) == 3


@patch("queue_monitor.video.source.cv2")
def test_iterator_stops_on_failure(mock_cv2):
    from queue_monitor.video.source import VideoSource

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)
    mock_cv2.VideoCapture.return_value = mock_cap

    vs = VideoSource(VideoConfig())
    vs.open()
    frames = list(vs)
    assert frames == []


@patch("queue_monitor.video.source.cv2")
def test_open_releases_previous_capture(mock_cv2):
    from queue_monitor.video.source import VideoSource

    first_cap = MagicMock()
    first_cap.isOpened.return_value = True
    second_cap = MagicMock()
    second_cap.isOpened.return_value = True
    mock_cv2.VideoCapture.side_effect = [first_cap, second_cap]

    vs = VideoSource(VideoConfig(source="video.mp4"))
    vs.open()
    assert vs._cap is first_cap

    vs.open()
    first_cap.release.assert_called_once()
    assert vs._cap is second_cap
