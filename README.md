# Queue Time Monitor

A Python application that processes video streams to detect people standing in queues, count queue size over time, and predict approximate wait times using computer vision and real-time tracking.

## Features

- **Person Detection** — YOLOv8 nano model for real-time person detection (40+ FPS on CPU)
- **Polygon Zone Filtering** — Define custom queue regions to monitor specific areas
- **Multi-Object Tracking** — ByteTrack maintains stable IDs through occlusions
- **Wait Time Estimation** — Static (configured service time) or adaptive (learned from observed departures)
- **EMA-Smoothed Counts** — Dampens per-frame noise for stable queue length readings
- **Web Dashboard** — Live video feed and Chart.js time-series via WebSocket
- **SQLite History** — Persistent metrics for historical analysis
- **Multi-Queue Support** — Monitor multiple named zones with independent estimators
- **RTSP Reconnection** — Exponential backoff for stream recovery

## Architecture

```
Video Source → YOLO Detect → Zone Filter → ByteTrack → Metrics
                (persons)     (polygon)    (stable IDs)    │
                                                           ├── Annotated Frame (WebSocket → Dashboard)
                                                           ├── Queue Count + Wait Time
                                                           └── SQLite Snapshots
```

## Installation

```bash
# Clone and install
git clone https://github.com/cjsteigerwald/queue-time-monitor.git
cd queue-time-monitor
pip install -e ".[dev]"
```

## Usage

### Run with a video file

```bash
python -m queue_monitor run --source video.mp4
```

### Run with webcam

```bash
python -m queue_monitor run --source 0
```

### Run with RTSP stream

```bash
python -m queue_monitor run --source rtsp://camera-ip:554/stream
```

### Show annotated OpenCV window

```bash
python -m queue_monitor run --source video.mp4 --show
```

### Start with web dashboard (localhost:8000)

```bash
python -m queue_monitor run --source video.mp4 --web
```

### Configure queue zone polygon interactively

```bash
python -m queue_monitor configure --source video.mp4 --zone main_queue
```

Click to place polygon vertices over the video frame. Press Enter to save, R to reset, Q to quit.

## Configuration

Settings are stored in `config/default.yaml`:

```yaml
video:
  source: "0"
  width: 1280
  height: 720

detection:
  model: "yolov8n.pt"
  confidence: 0.35

zones:
  - name: "main_queue"
    polygon: []          # set via 'configure' command
    num_servers: 1
    service_time: 30.0   # seconds per person (static mode fallback)

estimation:
  mode: "adaptive"       # "static" or "adaptive"
  ema_alpha: 0.3
  departure_window: 20

storage:
  database: "queue_monitor.db"
  snapshot_interval: 10

web:
  host: "0.0.0.0"
  port: 8000
```

## Wait Time Estimation

**Static mode**: `wait = queue_length × service_time / num_servers`

**Adaptive mode**: Tracks when people leave the queue zone. Computes a rolling average dwell time from the last N departures and uses that as the learned service interval. Falls back to static mode until enough departures are observed.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection | YOLOv8 (ultralytics) |
| Video I/O | OpenCV (headless) |
| Tracking | supervision + ByteTrack |
| Web | FastAPI + Jinja2 + Chart.js |
| Storage | SQLite |
| Config | Pydantic v2 + PyYAML |
| CLI | Typer |

## Testing

```bash
# Run tests
pytest

# Lint
ruff check src/ tests/
```

## License

See [LICENSE](LICENSE) for details.
