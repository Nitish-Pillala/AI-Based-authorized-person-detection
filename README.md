# AI-Based Authorized Person Detection

Real-time CCTV surveillance system that detects, recognizes, and tracks people across multiple IP cameras. Known persons (registered employees, family, staff) are matched against a face database; unknown persons are captured once per appearance with deduplication, so you get one clean record per unique stranger instead of thousands of repeated frames.

## Features

- **Multi-camera streaming** — connect 1–50 RTSP / IP cameras simultaneously
- **Face detection** — YOLOv8 (with OpenCV Haar Cascade fallback)
- **Face recognition** — InsightFace 512-dim embeddings, cosine similarity matching
- **Person tracking** — DeepSORT (Kalman filter + Hungarian assignment) for stable IDs across frames
- **Three-layer deduplication** — in-memory set + DB lookup + UNIQUE constraint = no duplicate unknown-person rows
- **REST API** — 20+ FastAPI endpoints with JWT auth, rate limiting, auto-generated Swagger docs
- **WebSocket streaming** — live frames, metrics, and alerts pushed to the dashboard
- **Streamlit dashboard** — 6 pages: overview, unknown gallery, live cameras, analytics, configuration, logs
- **MySQL persistence** — 7 related tables with connection pooling

## Tech Stack

| Layer | Tech |
|---|---|
| Detection | YOLOv8 (Ultralytics) |
| Recognition | InsightFace |
| Tracking | DeepSORT (FilterPy + SciPy) |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Database | MySQL 8 |
| Video I/O | OpenCV |
| Auth | JWT + bcrypt |
| Logging | Loguru |

## Prerequisites

- Python 3.10 or 3.11
- MySQL 8.0 (local install, or use the included `docker-compose.yml`)
- ~2 GB free disk for model weights and face data
- (Optional) NVIDIA GPU + CUDA for faster inference

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Nitish-Pillala/AI-Based-authorized-person-detection.git
cd AI-Based-authorized-person-detection
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv myenv
# Windows
myenv\Scripts\activate
# macOS / Linux
source myenv/bin/activate

pip install -r requirements.txt
```

### 3. Configure environment

```bash
# Windows
copy .env.template .env
# macOS / Linux
cp .env.template .env
```

Open `.env` and set at minimum:
- `DB_PASSWORD` — your MySQL root password
- `JWT_SECRET_KEY` — any long random string

### 4. Initialize the database

Make sure MySQL is running, then:

```bash
python setup_db.py
```

This creates the `cctv_surveillance` database and all 7 tables.

### 5. Start the backend (Terminal 1)

```bash
python start_backend.py
```

Backend runs on `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

### 6. Start the dashboard (Terminal 2)

```bash
python start_frontend.py
```

Dashboard opens at `http://localhost:8501`.

## Adding Known Persons

Two ways to register people the system should recognize:

**Option A — via the dashboard:** Open `Configuration` page → upload photo + enter name.

**Option B — bulk import via script:** Drop photos into a folder named after the person and run:

```bash
python add_known_faces.py
```

## Adding Cameras

From the `Configuration` page, add a camera with:
- **Camera ID** — short unique name (e.g. `entrance_01`)
- **RTSP URL** — `rtsp://username:password@<ip>:554/<stream>`
- **Location** — descriptive label
- **Processing FPS** — 5–15 is a good range; higher needs a GPU

For testing without real CCTV, you can use a webcam (`rtsp://` URL not needed — pass `0` as the source) or a public RTSP test stream.

## Docker (alternative)

If you'd rather not install MySQL locally:

```bash
docker compose up -d
```

This brings up MySQL, the backend, and the frontend. Note: Dockerfiles for backend/frontend are referenced in `docker-compose.yml` but not included — you'll need to add them, or just use the MySQL service and run the Python apps natively.

## Project Structure

```
.
├── backend/                  # AI + API
│   ├── api/fastapi_server.py # REST + WebSocket endpoints
│   ├── cctv_stream.py        # Camera connection / frame reader
│   ├── face_detection.py     # YOLOv8 wrapper
│   ├── face_recognition.py   # InsightFace embeddings + matching
│   ├── tracking.py           # DeepSORT
│   ├── unknown_face_handler.py
│   ├── database.py           # MySQL DAOs
│   ├── config.py             # Pydantic settings from .env
│   └── main.py               # Pipeline orchestrator
├── frontend/                 # Streamlit dashboard
│   ├── app.py
│   ├── pages/                # 6 dashboard pages
│   ├── utils/                # API client, cache, formatters
│   └── styles/custom.css
├── sql/init_db.sql           # Schema for the 7 tables
├── data/
│   ├── known_faces/          # Registered person photos (seed data)
│   ├── unknown_faces/        # Captured strangers (gitignored)
│   └── logs/                 # Daily log files (gitignored)
├── requirements.txt
├── .env.template
├── setup_db.py
├── start_backend.py
├── start_frontend.py
├── add_known_faces.py
├── docker-compose.yml
└── yolov8m-face.pt           # YOLOv8 face detection weights
```

## Configuration Reference

Key values in `.env`:

| Variable | Purpose | Default |
|---|---|---|
| `FACE_MATCH_THRESHOLD` | Cosine similarity above which a face is "known" | `0.75` |
| `YOLO_CONFIDENCE` | YOLO detection minimum confidence | `0.5` |
| `DEFAULT_PROCESSING_FPS` | Per-camera processing rate | `15` |
| `MAX_CAMERAS` | Concurrent camera limit | `50` |
| `DEEPSORT_MAX_AGE` | Frames to keep a track alive after last sighting | `30` |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | Login token lifetime | `60` |

See `.env.template` for the full list.

## API

Once the backend is running, browse `http://localhost:8000/docs` for the interactive Swagger UI. Common endpoints:

- `GET  /api/cameras` — list cameras
- `POST /api/cameras` — add a camera
- `GET  /api/unknown-persons` — list captured unknowns
- `POST /api/known-persons` — register a known person
- `GET  /api/system/health` — liveness check
- `WS   /ws/live` — live frame stream
- `WS   /ws/alerts` — real-time alert pushes

## How It Works

```
IP Camera (RTSP)
    │
    ▼
OpenCV ──► YOLOv8 (detect faces) ──► DeepSORT (assign track IDs)
                                            │
                                            ▼
                                  InsightFace (embed + match)
                                            │
                            ┌───────────────┴──────────────┐
                            ▼                              ▼
                      Known person                  Unknown person
                      → update count                → save once per track
                            │                              │
                            └──────────────┬───────────────┘
                                           ▼
                                       MySQL
                                           │
                                           ▼
                                  FastAPI + WebSocket
                                           │
                                           ▼
                                  Streamlit Dashboard
```

For a deep dive into each component, see [PROJECT_EXPLANATION.txt](PROJECT_EXPLANATION.txt) and [ENHANCED_SPECIFICATION.md](ENHANCED_SPECIFICATION.md).

## Troubleshooting

- **`Can't connect to MySQL`** — make sure MySQL is running and the password in `.env` matches.
- **Backend starts but no detections** — check that `yolov8m-face.pt` is in the project root and that your camera RTSP URL is reachable (`ffplay <rtsp_url>` to verify).
- **Slow on CPU** — drop `DEFAULT_PROCESSING_FPS` to 5, or downsample frames more aggressively in `backend/cctv_stream.py`.
- **`ImportError: insightface`** — InsightFace needs Visual C++ build tools on Windows; install "Desktop development with C++" via the Visual Studio Installer.

## License

Not yet licensed. Add a `LICENSE` file (MIT or Apache-2.0 are common choices) if you want others to be able to reuse this code.
