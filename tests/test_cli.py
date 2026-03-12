"""Tests for CLI commands (mocked pipeline and video to avoid hardware deps)."""

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from queue_monitor.cli import app

runner = CliRunner()


@patch("queue_monitor.pipeline.Pipeline")
@patch("queue_monitor.cli.load_config")
def test_run_default(mock_load, mock_pipeline_cls):
    mock_load.return_value = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline

    result = runner.invoke(app, ["run"])

    assert result.exit_code == 0
    mock_pipeline.run.assert_called_once_with(show_window=False)


@patch("queue_monitor.pipeline.Pipeline")
@patch("queue_monitor.cli.load_config")
def test_run_with_source(mock_load, mock_pipeline_cls):
    cfg = MagicMock()
    mock_load.return_value = cfg
    mock_pipeline_cls.return_value = MagicMock()

    result = runner.invoke(app, ["run", "--source", "test.mp4"])

    assert result.exit_code == 0
    assert cfg.video.source == "test.mp4"


@patch("queue_monitor.cli._run_with_web")
@patch("queue_monitor.cli.load_config")
def test_run_with_web_flag(mock_load, mock_run_web):
    mock_load.return_value = MagicMock()

    result = runner.invoke(app, ["run", "--web"])

    assert result.exit_code == 0
    mock_run_web.assert_called_once()


@patch("queue_monitor.pipeline.Pipeline")
@patch("queue_monitor.cli.load_config")
def test_run_with_show(mock_load, mock_pipeline_cls):
    mock_load.return_value = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline

    result = runner.invoke(app, ["run", "--show"])

    assert result.exit_code == 0
    mock_pipeline.run.assert_called_once_with(show_window=True)


@patch("queue_monitor.cli.load_config")
def test_run_web_logs_pipeline_crash(mock_load):
    """When the pipeline thread crashes, structlog.exception should be called."""
    cfg = MagicMock()
    mock_load.return_value = cfg

    mock_pipeline = MagicMock()
    mock_pipeline.run.side_effect = RuntimeError("boom")

    def fake_uvicorn_run(app, host, port):
        import time

        time.sleep(0.1)

    with (
        patch("queue_monitor.pipeline.Pipeline", return_value=mock_pipeline),
        patch("queue_monitor.web.app.create_app", return_value=MagicMock()),
        patch("uvicorn.run", side_effect=fake_uvicorn_run),
        patch("queue_monitor.cli.logger") as mock_logger,
    ):
        runner.invoke(app, ["run", "--web"])
        import time

        time.sleep(0.2)
        mock_logger.exception.assert_called_with("pipeline_crashed")
