# 🎥 ENHANCED PROJECT SPECIFICATION
## Real-Time CCTV Unknown Person Identification System
### Industry-Level Implementation with Advanced Features

---

## 📋 EXECUTIVE SUMMARY

Build a **production-grade, real-time CCTV surveillance system** that:
- ✅ Processes **multiple RTSP camera streams simultaneously**
- ✅ Detects faces using **YOLOv8 + InsightFace**
- ✅ Tracks persons across frames using **DeepSORT** (prevents duplicate logs)
- ✅ Recognizes known persons from **MySQL database**
- ✅ Logs unknown persons intelligently (avoiding duplicates)
- ✅ Provides **real-time WebSocket dashboard** via Streamlit
- ✅ Exposes **industry-level REST API** via FastAPI
- ✅ Handles **system failures gracefully**
- ✅ Scales to **10+ cameras** without degradation

---

## 🎯 CORE OBJECTIVES

### Primary Goals
1. **Real-Time Multi-Camera Streaming**
   - Support unlimited RTSP/IP cameras
   - Process multiple streams in parallel
   - Handle camera disconnections gracefully

2. **Intelligent Face Recognition**
   - Detect faces with YOLOv8 (fast, accurate)
   - Extract embeddings with InsightFace
   - Match against known faces database

3. **Smart Person Tracking (New Requirement)**
   - Use DeepSORT for frame-to-frame tracking
   - Maintain consistent person ID across frames
   - **Prevent duplicate logging** of same person
   - Track person movement between cameras

4. **Unknown Person Management**
   - Save only truly unknown faces (no duplicates)
   - Store in MySQL + local filesystem
   - Generate unique IDs per unique person
   - Track appearance frequency and timeline

5. **Real-Time Monitoring Dashboard**
   - Live WebSocket updates (not polling)
   - Multi-camera view
   - Real-time person detection metrics
   - Unknown person gallery with filters
   - Camera status monitoring

6. **Industry-Level REST API**
   - FastAPI with auto-documentation
   - Comprehensive endpoints for all operations
   - JWT authentication ready
   - Rate limiting and throttling
   - Request/response validation

---

## 🏗️ ENHANCED ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                     STREAMLIT DASHBOARD                     │
│  ✓ Real-time WebSocket updates                             │
│  ✓ Multi-camera grid view                                  │
│  ✓ Live metrics & alerts                                   │
│  ✓ Unknown person gallery                                  │
│  ✓ System health monitoring                                │
└────────────────┬────────────────────────────────────────────┘
                 │ WebSocket + REST
                 │
┌────────────────▼────────────────────────────────────────────┐
│                  FASTAPI BACKEND (REST API)                 │
│  ✓ 20+ endpoints                                            │
│  ✓ WebSocket support for real-time updates                 │
│  ✓ JWT authentication                                       │
│  ✓ Request validation with Pydantic                         │
│  ✓ API documentation (/docs)                                │
│  ✓ Rate limiting & throttling                               │
│  ✓ Error handling & logging                                 │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│           FACE RECOGNITION PROCESSING ENGINE                │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ MULTI-CAMERA STREAM MANAGER (Threading)              │  │
│  │ ✓ Parallel processing for N cameras                  │  │
│  │ ✓ Connection health monitoring                       │  │
│  │ ✓ Automatic reconnection on failure                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ FACE DETECTION PIPELINE                              │  │
│  │ ├─ YOLOv8-Face (ultra-fast, accurate)               │  │
│  │ ├─ Real-time bounding box generation                │  │
│  │ ├─ Adaptive resolution handling                      │  │
│  │ └─ Fallback to InsightFace detection if needed       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ DEEPSORT TRACKING SYSTEM (NEW)                       │  │
│  │ ├─ Kalman filter for motion prediction              │  │
│  │ ├─ Feature matching for appearance consistency       │  │
│  │ ├─ Track ID assignment (consistent across frames)   │  │
│  │ ├─ Cross-camera tracking support                     │  │
│  │ └─ Configurable tracking parameters                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ FACE RECOGNITION ENGINE                              │  │
│  │ ├─ InsightFace (512-dim embeddings)                  │  │
│  │ ├─ Cosine similarity matching                        │  │
│  │ ├─ Configurable confidence thresholds                │  │
│  │ └─ Known face cache (in-memory)                      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ DUPLICATE PREVENTION SYSTEM (NEW)                    │  │
│  │ ├─ Track-based deduplication                         │  │
│  │ ├─ Temporal clustering (time-based grouping)         │  │
│  │ ├─ Spatial similarity check                          │  │
│  │ └─ Confidence-based merging                          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ LOGGING & PERSISTENCE LAYER                          │  │
│  │ ├─ Unknown face disk storage                         │  │
│  │ ├─ MySQL database operations                         │  │
│  │ ├─ Transaction management                            │  │
│  │ └─ Audit trail logging                               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└────────────────┬────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────┐
│                    MYSQL DATABASE                           │
│  ✓ known_faces (registered database)                       │
│  ✓ unknown_faces (detected strangers)                      │
│  ✓ track_logs (person tracking history)                    │
│  ✓ detections (detection events)                           │
│  ✓ cameras (camera metadata)                               │
│  ✓ alerts (important events)                               │
│  ✓ system_logs (audit trail)                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 📡 ENHANCED CORE REQUIREMENTS

### 1. RTSP Multi-Camera Integration

#### Requirements
```
✅ Support 5-50+ simultaneous RTSP streams
✅ Handle variable network quality
✅ Auto-reconnect on disconnection
✅ Frame dropping/skipping on lag
✅ Per-camera FPS control
✅ Per-camera resolution management
✅ Per-camera detection confidence threshold
```

#### Implementation Details
```python
# Pseudo-code structure
class MultiCameraStreamManager:
    def __init__(self, max_cameras=50):
        self.cameras = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=max_cameras)
        self.stream_buffer = Queue(maxsize=100)
    
    def add_camera(self, camera_id, rtsp_url, config):
        # Start dedicated thread for each camera
        # Monitor connection health
        # Implement backoff reconnection strategy
    
    def process_frame(self, camera_id, frame):
        # Put frame in shared buffer
        # Non-blocking operation
        # Drop frames if buffer full (indicate lag)
```

---

### 2. YOLOv8 Face Detection (Enhanced)

#### Why YOLOv8?
- **Ultra-fast**: 30-50ms per frame (GPU)
- **Accurate**: 96%+ detection rate
- **Small**: Minimal memory footprint
- **Easy to train**: Custom models possible

#### Requirements
```
✅ Real-time detection (30+ FPS on GPU)
✅ Multi-scale face detection (small to large)
✅ Low-resolution CCTV support (360p, 480p, 720p)
✅ Poor lighting handling
✅ Partial face detection (profile view)
✅ Face confidence filtering
✅ NMS (Non-Maximum Suppression) for overlaps
```

#### Model Selection
```python
# Available YOLO models for faces
- yolov8n-face.pt    # Nano (fastest, smallest)
- yolov8s-face.pt    # Small (balanced)
- yolov8m-face.pt    # Medium (recommended)
- yolov8l-face.pt    # Large (most accurate)
- yolov8x-face.pt    # Extra Large (highest accuracy)

# Recommended: yolov8m-face for production
# Balance between speed and accuracy
```

---

### 3. DeepSORT Multi-Person Tracking (NEW FEATURE)

#### Purpose
**Prevent duplicate logging of same person** across frames

#### How It Works
```
Frame 1: Detect Face A at position (100, 50)
         ↓ DeepSORT assigns Track ID = 1
Frame 2: Detect Face A at position (105, 55)
         ↓ DeepSORT recognizes same person
         ↓ Updates Track ID = 1 (not new)
Frame 3: Detect Face B at position (300, 100)
         ↓ DeepSORT assigns Track ID = 2 (new person)

Result: Only 2 unique persons logged, not 3+ duplicates
```

#### Implementation Requirements
```
✅ Kalman Filter for motion prediction
✅ Feature matching (appearance consistency)
✅ Track ID persistence across frames
✅ Cross-camera tracking (optional)
✅ Track lifecycle management (creation → death)
✅ Configurable track parameters
   - max_age: frames to keep inactive track
   - min_hits: detections before creating track
   - max_iou_distance: overlap threshold
   - max_cosine_distance: appearance threshold
```

#### Key Benefits
```
❌ Without DeepSORT:
   - Same person detected 30 times per second
   - 1800 database entries per minute (one person!)
   - Duplicate logs everywhere
   - Impossible to analyze patterns

✅ With DeepSORT:
   - One track ID per unique person
   - One database entry (with updated timestamp)
   - Clean, deduplicated logs
   - Accurate person analytics
```

---

### 4. InsightFace Face Recognition (Enhanced)

#### Face Embedding Workflow
```
Input: Cropped face image from YOLO
  ↓
InsightFace Extraction: 512-dimensional embedding
  ↓
Cosine Similarity Comparison with Known Faces
  ↓
Threshold-based Decision:
  - Similarity >= 0.65 → Known person ✓
  - Similarity < 0.65  → Unknown person ✗
```

#### Requirements
```
✅ Fast embedding generation (<10ms per face)
✅ 99%+ accuracy on LFW benchmark
✅ Handle extreme poses (profile, tilted)
✅ Handle occlusions (masks, glasses, hats)
✅ Low-resolution face support (64x64 minimum)
✅ Caching of known embeddings (in-memory)
✅ Batch embedding generation (5+ faces)
```

#### Confidence Threshold Strategy
```
Threshold Tuning:
- 0.85+: Highest accuracy, few false positives (secure)
- 0.75-0.85: Balanced (recommended for most use cases)
- 0.65-0.75: Higher recall, some false positives (survey)
- <0.65: Recognize barely similar faces (testing only)

Recommended: 0.75 with manual review option
```

---

### 5. Unknown Person Handling (Enhanced with Deduplication)

#### Original Workflow (Problems)
```
Frame 1-30: Same person appears
  ↓ Save to unknown_faces 30 times
  ↓ Insert to MySQL 30 times
  ↓ Duplicate entries in database
```

#### Enhanced Workflow (With DeepSORT)
```
Frame 1: Detect unknown person, Track ID = 5
  ↓ Check: Is track ID 5 already in DB?
  ↓ NO → Save face + Insert to MySQL
  ↓ Mark Track ID 5 as "logged"

Frame 2-30: Same unknown person appears, Track ID = 5
  ↓ Check: Is track ID 5 already in DB?
  ↓ YES → Skip saving (avoid duplicate)
  ↓ Update last_seen timestamp only

Result: One database entry per unique person
```

#### Database Storage Strategy
```python
# Table: unknown_faces (Enhanced)
columns:
  - id: INT (auto-increment, primary key)
  - track_id: VARCHAR (DeepSORT Track ID)  ← NEW
  - image_path: VARCHAR (best quality image)
  - first_seen: TIMESTAMP
  - last_seen: TIMESTAMP  ← Updated each encounter
  - appearance_count: INT  ← Increment on each encounter
  - camera_id: VARCHAR (which camera)
  - confidence: FLOAT (highest confidence)
  - face_embedding: BLOB (512-dim vector)  ← NEW
  - marked_as_known: BOOLEAN (manually identified)
  - notes: TEXT
  - created_at: TIMESTAMP
  - updated_at: TIMESTAMP

# Unique constraint: (track_id, camera_id)
# Prevents duplicate entry per camera per track
```

---

### 6. Real-Time Dashboard (WebSocket-based)

#### Why WebSocket? (Not Polling)
```
❌ HTTP Polling:
   - Dashboard requests data every 1 second
   - High latency (1-5 second delay)
   - Wasteful bandwidth
   - Looks laggy

✅ WebSocket:
   - Server pushes updates instantly
   - <100ms latency
   - Efficient bandwidth usage
   - Smooth real-time feel
```

#### Dashboard Features

**Main Page: Live Multi-Camera View**
```
┌─────────────────────────────────────────┐
│  🎥 Camera 1       │  🎥 Camera 2       │
│  Known: 5 / Unknown: 2                  │
│  FPS: 28 | Persons: 7                   │
│  [LIVE VIDEO FEED] │ [LIVE VIDEO FEED]  │
│  ✓ Active         │ ✓ Active            │
├─────────────────────────────────────────┤
│  🎥 Camera 3       │  🎥 Camera 4       │
│  Known: 0 / Unknown: 1                  │
│  FPS: 25 | Persons: 1                   │
│  [LIVE VIDEO FEED] │ [LIVE VIDEO FEED]  │
│  ✓ Active         │ ⚠ Buffering         │
└─────────────────────────────────────────┘
```

**Dashboard Tabs**
```
1. 📊 OVERVIEW
   ├─ Real-time metrics (people count, FPS)
   ├─ Camera status grid
   ├─ Alerts and notifications
   ├─ System health stats
   └─ Timeline of events

2. 👥 UNKNOWN PERSONS GALLERY
   ├─ Grid of unknown faces
   ├─ Sort by: first_seen, last_seen, appearance_count
   ├─ Filter by: camera_id, time range, confidence
   ├─ Bulk actions: mark as known, delete
   ├─ Person details:
   │  - Image gallery (all appearances)
   │  - Timeline of sightings
   │  - Cameras visited
   │  - Frequency statistics
   └─ Manual identification interface

3. 📹 LIVE CAMERAS
   ├─ Multi-camera live view
   ├─ Per-camera controls:
   │  - Pause / Resume
   │  - Adjust detection confidence
   │  - Change tracking parameters
   │  - View camera health
   └─ Frame-by-frame controls

4. 📊 ANALYTICS & REPORTING
   ├─ Known person detection trends
   ├─ Unknown person frequency heatmap
   ├─ Camera usage patterns
   ├─ Peak times analysis
   ├─ Custom report generation
   └─ Export data (CSV, JSON)

5. ⚙️ CONFIGURATION
   ├─ Camera management (add/remove/edit)
   ├─ Detection settings
   │  - YOLO confidence threshold
   │  - Face size minimum/maximum
   │  - Processing FPS limit
   ├─ Recognition settings
   │  - InsightFace model selection
   │  - Matching threshold
   │  - Known face refresh interval
   ├─ DeepSORT tracking parameters
   │  - max_age, min_hits, max_iou_distance
   │  - max_cosine_distance
   └─ System settings
      - Database connection
      - Storage location
      - Logging level
      - Alert thresholds

6. 📜 LOGS & MONITORING
   ├─ System logs (real-time)
   ├─ Detection events log
   ├─ Person recognition events
   ├─ Error logs
   ├─ Performance metrics
   └─ Download logs
```

**Real-Time Updates (WebSocket)**
```python
# WebSocket events pushed to dashboard every 100ms

{
    "type": "frame_update",
    "camera_id": "camera_1",
    "frame_number": 1543,
    "detections": [
        {
            "track_id": 5,
            "person_name": "John Doe",
            "confidence": 0.92,
            "bbox": [100, 50, 150, 180],
            "is_new": False
        },
        {
            "track_id": 12,
            "person_name": "Unknown",
            "confidence": None,
            "bbox": [300, 100, 350, 230],
            "is_new": True
        }
    ],
    "metrics": {
        "fps": 28,
        "active_persons": 2,
        "known_persons": 1,
        "unknown_persons": 1
    },
    "timestamp": "2024-05-05T10:30:45.123Z"
}
```

---

### 7. Industry-Level FastAPI Backend

#### REST API Endpoints (20+)

**Camera Management**
```
POST   /api/cameras                    - Add new camera
GET    /api/cameras                    - List all cameras
GET    /api/cameras/{camera_id}        - Get camera details
PUT    /api/cameras/{camera_id}        - Update camera
DELETE /api/cameras/{camera_id}        - Remove camera
POST   /api/cameras/{camera_id}/test   - Test RTSP connection
GET    /api/cameras/{camera_id}/stats  - Get camera statistics
```

**Known Persons Management**
```
POST   /api/known-persons              - Add new known person
GET    /api/known-persons              - List all known persons
GET    /api/known-persons/{person_id}  - Get person details
PUT    /api/known-persons/{person_id}  - Update person
DELETE /api/known-persons/{person_id}  - Remove person
POST   /api/known-persons/{person_id}/image - Add face image
GET    /api/known-persons/{person_id}/images - Get all images
```

**Unknown Persons Management**
```
GET    /api/unknown-persons            - List unknowns with filters
GET    /api/unknown-persons/{unknown_id} - Get details
DELETE /api/unknown-persons/{unknown_id} - Delete entry
POST   /api/unknown-persons/{unknown_id}/mark-known - Identify person
GET    /api/unknown-persons/stats      - Unknown persons stats
GET    /api/unknown-persons/trending   - Most frequent unknowns
```

**Detection & Recognition**
```
POST   /api/detections/search          - Search detections
GET    /api/detections/timeline        - Timeline view
GET    /api/detections/person/{person_id} - Person's timeline
POST   /api/recognition/test           - Test face recognition
```

**System Management**
```
GET    /api/system/health              - System health check
GET    /api/system/stats               - Overall statistics
GET    /api/system/logs                - System logs
POST   /api/system/restart             - Restart system
GET    /api/system/config              - Get configuration
PUT    /api/system/config              - Update configuration
```

**WebSocket Endpoint**
```
WS     /ws/live                        - Live frame stream
WS     /ws/metrics                     - Real-time metrics
WS     /ws/alerts                      - Alert notifications
WS     /ws/camera/{camera_id}          - Single camera stream
```

#### API Features
```
✅ Request/Response validation (Pydantic)
✅ JWT authentication & authorization
✅ Rate limiting (1000 req/min per IP)
✅ Request logging & audit trail
✅ CORS headers configured
✅ Compression (gzip)
✅ Caching headers (ETags)
✅ Pagination support
✅ Search & filtering
✅ Sorting by multiple fields
✅ Full-text search on logs
✅ Bulk operations
✅ Idempotency keys
✅ Error handling with proper HTTP codes
✅ Swagger/OpenAPI documentation
✅ Request tracing (correlation IDs)
```

#### Example API Usage
```python
# Add a camera
POST /api/cameras
{
    "camera_id": "entrance_01",
    "name": "Main Entrance",
    "rtsp_url": "rtsp://user:pass@192.168.1.100:554/stream",
    "location": "Building A - Ground Floor",
    "enabled": true,
    "processing_fps": 15,
    "confidence_threshold": 0.75,
    "track_config": {
        "max_age": 30,
        "min_hits": 3,
        "max_iou_distance": 0.7,
        "max_cosine_distance": 0.4
    }
}

# Response
{
    "status": "success",
    "camera_id": "entrance_01",
    "message": "Camera added successfully",
    "connection_status": "connected"
}

# Add known person
POST /api/known-persons
{
    "name": "John Doe",
    "employee_id": "EMP001",
    "department": "Engineering",
    "face_image": <base64_encoded_image>
}

# Response
{
    "status": "success",
    "person_id": 1,
    "message": "Person registered successfully",
    "embedding_generated": true
}

# Get unknown persons with filters
GET /api/unknown-persons?camera_id=entrance_01&min_appearances=2&sort_by=appearance_count&order=desc

# Response
{
    "status": "success",
    "count": 5,
    "unknowns": [
        {
            "id": 101,
            "image_path": "/data/unknown_faces/unknown_001.jpg",
            "first_seen": "2024-05-05T10:30:45Z",
            "last_seen": "2024-05-05T14:55:32Z",
            "appearance_count": 7,
            "camera_id": "entrance_01",
            "confidence": 0.92
        },
        ...
    ]
}
```

---

### 8. Enhanced Database Schema

#### Tables Overview

**1. cameras** - RTSP camera metadata
```sql
CREATE TABLE cameras (
    id INT PRIMARY KEY AUTO_INCREMENT,
    camera_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100),
    location VARCHAR(200),
    rtsp_url VARCHAR(500),
    enabled BOOLEAN DEFAULT TRUE,
    processing_fps INT DEFAULT 15,
    confidence_threshold FLOAT DEFAULT 0.75,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_camera_id (camera_id),
    INDEX idx_enabled (enabled)
);
```

**2. known_faces** - Registered persons
```sql
CREATE TABLE known_faces (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    employee_id VARCHAR(50),
    department VARCHAR(100),
    description TEXT,
    face_embedding BLOB,  -- 512-dim InsightFace vector
    total_detections INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_name (name),
    INDEX idx_employee_id (employee_id),
    FULLTEXT idx_search (name, department)
);
```

**3. known_face_images** - Multiple images per person
```sql
CREATE TABLE known_face_images (
    id INT PRIMARY KEY AUTO_INCREMENT,
    person_id INT NOT NULL,
    image_path VARCHAR(500),
    face_embedding BLOB,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES known_faces(id) ON DELETE CASCADE,
    INDEX idx_person_id (person_id)
);
```

**4. unknown_faces** - Detected strangers (Enhanced with DeepSORT)
```sql
CREATE TABLE unknown_faces (
    id INT PRIMARY KEY AUTO_INCREMENT,
    track_id VARCHAR(50),  -- DeepSORT Track ID
    best_image_path VARCHAR(500),
    face_embedding BLOB,  -- 512-dim vector
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    appearance_count INT DEFAULT 1,
    camera_id VARCHAR(50),
    highest_confidence FLOAT,
    marked_as_known BOOLEAN DEFAULT FALSE,
    marked_as_person_id INT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY unique_track_camera (track_id, camera_id),
    INDEX idx_first_seen (first_seen),
    INDEX idx_last_seen (last_seen),
    INDEX idx_appearance_count (appearance_count),
    INDEX idx_camera_id (camera_id),
    INDEX idx_marked_as_known (marked_as_known),
    FOREIGN KEY (marked_as_person_id) REFERENCES known_faces(id) ON DELETE SET NULL
);
```

**5. unknown_face_images** - All images of unknown person
```sql
CREATE TABLE unknown_face_images (
    id INT PRIMARY KEY AUTO_INCREMENT,
    unknown_id INT NOT NULL,
    image_path VARCHAR(500),
    camera_id VARCHAR(50),
    detected_at TIMESTAMP,
    confidence FLOAT,
    FOREIGN KEY (unknown_id) REFERENCES unknown_faces(id) ON DELETE CASCADE,
    INDEX idx_unknown_id (unknown_id),
    INDEX idx_detected_at (detected_at)
);
```

**6. detections** - All detection events
```sql
CREATE TABLE detections (
    id INT PRIMARY KEY AUTO_INCREMENT,
    camera_id VARCHAR(50),
    track_id VARCHAR(50),  -- DeepSORT Track ID
    person_id INT,  -- For known persons
    unknown_id INT,  -- For unknown persons
    person_name VARCHAR(100),
    confidence FLOAT,
    bbox_x1 INT, bbox_y1 INT, bbox_x2 INT, bbox_y2 INT,
    detected_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES known_faces(id),
    FOREIGN KEY (unknown_id) REFERENCES unknown_faces(id),
    INDEX idx_camera_id (camera_id),
    INDEX idx_track_id (track_id),
    INDEX idx_detected_at (detected_at),
    INDEX idx_person_id (person_id),
    INDEX idx_unknown_id (unknown_id)
);
```

**7. system_logs** - Audit trail
```sql
CREATE TABLE system_logs (
    id INT PRIMARY KEY AUTO_INCREMENT,
    event_type VARCHAR(50),  -- detection, person_added, etc.
    camera_id VARCHAR(50),
    description TEXT,
    details JSON,
    severity VARCHAR(20),  -- info, warning, error
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_event_type (event_type),
    INDEX idx_severity (severity),
    INDEX idx_created_at (created_at)
);
```

---

## 🚀 ENHANCED IMPLEMENTATION REQUIREMENTS

### Backend Structure
```
backend/
├── cctv_stream.py
│   ├── MultiCameraStreamManager          ← Multiple RTSP support
│   ├── RTSPStreamHandler                 ← Per-camera stream
│   ├── FrameBuffer                       ← Thread-safe queue
│   └── ConnectionHealthMonitor           ← Auto-reconnect
│
├── face_detection.py
│   ├── YOLOv8FaceDetector                ← Primary detector
│   ├── InsightFaceDetector               ← Fallback detector
│   ├── DetectionProcessor                ← Batch processing
│   └── ConfidenceFilter                  ← Threshold-based filtering
│
├── face_recognition.py
│   ├── InsightFaceEncoder                ← Embedding generation
│   ├── FaceEmbeddingCache                ← In-memory cache
│   ├── SimilarityMatcher                 ← Cosine similarity
│   └── RecognitionResult                 ← Result structure
│
├── tracking.py (NEW)
│   ├── DeepSortTracker                   ← Multi-object tracking
│   ├── TrackManager                      ← Track lifecycle
│   ├── DuplicateDetector                 ← Deduplication logic
│   ├── KalmanFilter                      ← Motion prediction
│   └── FeatureExtractor                  ← Appearance features
│
├── database.py
│   ├── DatabaseManager                   ← Connection pooling
│   ├── KnownPersonsDAO                   ← Known persons CRUD
│   ├── UnknownPersonsDAO                 ← Unknown persons CRUD
│   ├── DetectionDAO                      ← Detection logging
│   ├── CameraDAO                         ← Camera management
│   └── SystemLogDAO                      ← Audit logging
│
├── unknown_face_handler.py
│   ├── UnknownFaceSaver                  ← File + DB saving
│   ├── DuplicateChecker                  ← Face similarity check
│   ├── ImageOptimizer                    ← Compression
│   └── StorageManager                    ← File organization
│
├── api/fastapi_server.py
│   ├── CameraEndpoints                   ← /api/cameras/*
│   ├── KnownPersonsEndpoints             ← /api/known-persons/*
│   ├── UnknownPersonsEndpoints           ← /api/unknown-persons/*
│   ├── DetectionEndpoints                ← /api/detections/*
│   ├── SystemEndpoints                   ← /api/system/*
│   ├── WebSocketManager                  ← /ws/* handlers
│   ├── AuthMiddleware                    ← JWT authentication
│   └── RateLimiter                       ← Rate limiting
│
├── config.py
│   ├── Config class
│   ├── Camera defaults
│   ├── Detection settings
│   ├── Recognition settings
│   ├── DeepSORT parameters
│   ├── Database configuration
│   └── API settings
│
├── utils.py
│   ├── Logger setup
│   ├── Image utilities
│   ├── Video utilities
│   ├── Performance monitoring
│   ├── Exception handlers
│   └── Helper functions
│
└── main.py
    ├── Application setup
    ├── Thread initialization
    ├── Signal handlers
    └── Shutdown cleanup
```

### Frontend Structure
```
frontend/
├── app.py                    ← Main Streamlit app
├── pages/
│   ├── 01_Overview.py        ← Dashboard with real-time metrics
│   ├── 02_UnknownGallery.py  ← Unknown persons gallery
│   ├── 03_LiveCameras.py     ← Multi-camera live view
│   ├── 04_Analytics.py       ← Reporting & analytics
│   ├── 05_Configuration.py   ← Settings management
│   └── 06_Logs.py            ← System logs viewer
│
├── components/
│   ├── camera_grid.py        ← Multi-camera display
│   ├── metrics_widget.py     ← Real-time metrics
│   ├── unknown_gallery.py    ← Face gallery
│   ├── timeline.py           ← Event timeline
│   └── charts.py             ← Analytics charts
│
├── utils/
│   ├── api_client.py         ← FastAPI communication
│   ├── websocket_handler.py  ← WebSocket management
│   ├── cache.py              ← Caching layer
│   └── formatters.py         ← Data formatting
│
└── styles/
    └── custom.css            ← Custom styling
```

---

## ⚡ PERFORMANCE OPTIMIZATION STRATEGIES

### 1. Multi-Camera Parallel Processing
```python
# ThreadPoolExecutor for parallel camera processing
executor = ThreadPoolExecutor(max_workers=10)

for camera_id, rtsp_url in cameras.items():
    executor.submit(process_camera_stream, camera_id, rtsp_url)
```

### 2. Frame Skipping (Configurable)
```python
# Process every Nth frame based on FPS target
frame_skip = int(camera_fps / target_fps)  # e.g., every 2nd frame

if frame_count % frame_skip == 0:
    # Process this frame
    detections = detect_faces(frame)
```

### 3. Resolution Downsampling
```python
# Downsize frame for faster processing
processing_height = 480
aspect_ratio = frame.shape[1] / frame.shape[0]
processing_width = int(processing_height * aspect_ratio)
resized_frame = cv2.resize(frame, (processing_width, processing_height))
```

### 4. Batch Processing
```python
# Process multiple faces in batch
faces = get_detected_faces(frame)
if len(faces) > 1:
    embeddings = face_encoder.encode_batch(faces)  # Faster than loop
else:
    embeddings = [face_encoder.encode(faces[0])]
```

### 5. Database Connection Pooling
```python
# Use connection pool for concurrent queries
db_pool = mysql.connector.pooling.MySQLConnectionPool(
    pool_name="cctv_pool",
    pool_size=10,
    pool_reset_session=True,
    host="localhost",
    user="root",
    password="password",
    database="cctv_system"
)
```

### 6. In-Memory Caching
```python
# Cache known face embeddings in memory
class EmbeddingCache:
    def __init__(self):
        self.cache = {}
        self.last_refresh = None
    
    def refresh_if_needed(self, timeout_minutes=60):
        # Periodically reload from database
        if time.time() - self.last_refresh > timeout_minutes * 60:
            self.cache = db.get_all_known_embeddings()
```

### 7. GPU Acceleration
```python
# Use GPU for heavy operations
- YOLO detection: GPU (30-50ms)
- InsightFace embedding: GPU (10-15ms)
- DeepSORT: GPU/CPU hybrid (5-10ms)
```

### 8. Smart Frame Buffering
```python
# Drop frames if processing lags
if buffer.full():
    # System is lagging, drop oldest frame
    buffer.get()
    logger.warning(f"Dropped frame from {camera_id}")
```

---

## 🔄 ENHANCED WORKFLOW DIAGRAM

```
FRAME CAPTURE
    ↓
MULTI-CAMERA STREAM MANAGER
    ├─ Camera 1: RTSP stream
    ├─ Camera 2: RTSP stream
    ├─ Camera N: RTSP stream
    ↓
FRAME PREPROCESSING
    ├─ Resize (480p)
    ├─ Normalize
    └─ Frame skip (if needed)
    ↓
YOLOV8 FACE DETECTION
    ├─ Detect faces
    ├─ Get bounding boxes
    └─ Filter by confidence threshold
    ↓
DEEPSORT TRACKING (NEW)
    ├─ Assign Track IDs
    ├─ Predict motion
    ├─ Match with previous frames
    └─ Maintain consistent person IDs
    ↓
FACE CROPPING & ALIGNMENT
    ├─ Extract face ROI
    ├─ Align face orientation
    └─ Resize to standard size
    ↓
INSIGHTFACE EMBEDDING EXTRACTION
    ├─ Generate 512-dim embedding
    └─ Normalize vector
    ↓
KNOWN FACE MATCHING
    ├─ Load cached known embeddings
    ├─ Compute cosine similarity
    ├─ Compare with threshold
    ↓
    ├─ MATCH FOUND (Known person)
    │   ├─ Get person name
    │   ├─ Update detection count
    │   └─ Log to MySQL (detections table)
    │
    └─ NO MATCH (Unknown person)
        ├─ Check if already in unknown_faces (via track_id)
        ├─ If NEW unknown:
        │   ├─ Save image to disk
        │   ├─ Save embedding to MySQL
        │   ├─ Create unknown_faces entry
        │   └─ Send alert to dashboard
        └─ If EXISTING unknown:
            ├─ Update last_seen timestamp
            ├─ Increment appearance_count
            └─ Save additional image (if better quality)
    ↓
ANNOTATION & VISUALIZATION
    ├─ Draw bounding box
    ├─ Write name/label
    ├─ Draw track ID
    └─ Display confidence
    ↓
REAL-TIME DASHBOARD UPDATE (WebSocket)
    ├─ Push frame to all connected clients
    ├─ Send detection events
    └─ Send metrics update
    ↓
METRICS CALCULATION & LOGGING
    ├─ Calculate FPS
    ├─ Count active persons
    ├─ Track detection events
    └─ Log to system_logs table
```

---

## 🎯 KEY ENHANCEMENTS SUMMARY

| Feature | Original | Enhanced |
|---------|----------|----------|
| **Cameras** | 1 RTSP | 5-50+ parallel RTSP |
| **Detection** | OpenCV Haar | YOLOv8 (faster, more accurate) |
| **Recognition** | InsightFace only | InsightFace with caching |
| **Tracking** | None | DeepSORT (prevents duplicates) |
| **Duplicates** | 1000s per hour | Single entry per unique person |
| **Frontend** | Basic Streamlit | Real-time WebSocket dashboard |
| **Backend** | Simple scripts | Industry-level FastAPI |
| **Database** | Basic MySQL | Full-featured schema (7 tables) |
| **API** | None | 20+ REST endpoints + WebSocket |
| **Performance** | Basic | Multi-threading, GPU, caching |
| **Monitoring** | No | Real-time metrics + logging |
| **Authentication** | No | JWT-ready |
| **Scalability** | Single system | Production-grade |

---

## 📦 DELIVERABLES

### Code
- ✅ Complete backend system (10+ modules)
- ✅ Industry-level FastAPI server
- ✅ Real-time Streamlit dashboard
- ✅ MySQL database with full schema
- ✅ RTSP multi-camera handler
- ✅ DeepSORT tracking integration
- ✅ YOLOv8 detection module
- ✅ InsightFace recognition module

### Documentation
- ✅ Complete API documentation
- ✅ Database schema documentation
- ✅ Deployment guide
- ✅ Configuration guide
- ✅ Troubleshooting guide
- ✅ Performance tuning guide

### Configuration
- ✅ Docker compose for deployment
- ✅ Requirements.txt for dependencies
- ✅ Config file for customization
- ✅ Environment variables template

---

## ⚙️ TECHNICAL SPECIFICATIONS

### Hardware Requirements
```
Minimum:
  - CPU: 4 cores
  - RAM: 8GB
  - GPU: Optional (RTX 1060+)
  - Storage: 500GB

Recommended (for 10+ cameras):
  - CPU: 8+ cores
  - RAM: 16GB+
  - GPU: RTX 3060+ (or A100 for enterprise)
  - Storage: 2TB+ SSD

Enterprise (50+ cameras):
  - CPU: 16+ cores
  - RAM: 32GB+
  - GPU: Multiple RTX 3090 or A100
  - Storage: 10TB+ distributed
```

### Software Stack
```
OS: Ubuntu 20.04 LTS or later
Python: 3.9+
CUDA: 11.8+
cuDNN: 8.6+

Core Libraries:
  - OpenCV 4.8+
  - YOLOv8 (ultralytics)
  - InsightFace 0.7.3+
  - DeepSORT
  - FastAPI 0.104+
  - Streamlit 1.28+
  - MySQL Connector 8.0+
  - NumPy 1.24+
  - scikit-learn (for metrics)
```

### Network Requirements
```
- Minimum bandwidth: 5 Mbps per camera (RTSP)
- Recommended: 10+ Mbps per camera
- API port: 8000
- Streamlit port: 8501
- MySQL port: 3306
- Low-latency network (< 50ms RTT)
```

---

## 🎓 IMPLEMENTATION TIMELINE

```
Week 1: Core Backend
  - RTSP multi-camera handler
  - YOLOv8 face detection
  - InsightFace recognition
  - MySQL database setup

Week 2: Advanced Features
  - DeepSORT tracking integration
  - Duplicate detection logic
  - Unknown face handling
  - Performance optimization

Week 3: API & Integration
  - FastAPI REST endpoints
  - WebSocket implementation
  - API authentication
  - Error handling

Week 4: Frontend & Deployment
  - Streamlit dashboard
  - Real-time WebSocket updates
  - Docker containerization
  - Production deployment
```

---

## ✅ QUALITY ASSURANCE CHECKLIST

- [ ] All RTSP cameras connect reliably
- [ ] Face detection works in low light
- [ ] Face recognition accuracy > 95%
- [ ] DeepSORT prevents duplicate logging
- [ ] Unknown faces saved correctly
- [ ] Dashboard updates in real-time
- [ ] API returns correct responses
- [ ] Database queries are optimized
- [ ] No memory leaks in threads
- [ ] Graceful handling of camera disconnections
- [ ] System recovers from errors
- [ ] Logs are comprehensive
- [ ] Performance under load is acceptable
- [ ] Security headers are set correctly
- [ ] Documentation is complete

---

This enhanced specification provides a **production-grade, industry-level CCTV surveillance system** with all requested features!
