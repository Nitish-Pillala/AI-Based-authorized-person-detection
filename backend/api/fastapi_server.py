from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, status, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import json
import asyncio
import base64
import cv2
import numpy as np
from pathlib import Path
from jose import JWTError, jwt
from passlib.context import CryptContext
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.config import settings
from backend.database import CameraDAO, KnownPersonsDAO, UnknownPersonsDAO, DetectionDAO, SystemLogDAO, DatabaseManager
from backend.cctv_stream import CameraConfig
from backend.utils import app_logger as logger, get_system_stats, save_face_image, embedding_to_bytes


# ── Auth ──────────────────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.jwt_access_token_expire_minutes))
    to_encode["exp"] = expire
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])


# ── Pydantic models ───────────────────────────────────────────────────────────
class TrackConfig(BaseModel):
    max_age: int = 30
    min_hits: int = 3
    max_iou_distance: float = 0.7
    max_cosine_distance: float = 0.4

class CameraCreate(BaseModel):
    camera_id: str
    name: Optional[str] = None
    rtsp_url: str
    location: Optional[str] = None
    enabled: bool = True
    processing_fps: int = 15
    confidence_threshold: float = 0.75
    track_config: Optional[TrackConfig] = None

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = None
    location: Optional[str] = None
    enabled: Optional[bool] = None
    processing_fps: Optional[int] = None
    confidence_threshold: Optional[float] = None

class KnownPersonCreate(BaseModel):
    name: str
    employee_id: Optional[str] = None
    department: Optional[str] = None
    description: Optional[str] = None
    face_image: Optional[str] = None  # base64

class KnownPersonUpdate(BaseModel):
    name: Optional[str] = None
    employee_id: Optional[str] = None
    department: Optional[str] = None
    description: Optional[str] = None

class MarkAsKnownRequest(BaseModel):
    person_id: int

class ConfigUpdate(BaseModel):
    face_match_threshold: Optional[float] = None
    default_processing_fps: Optional[int] = None
    log_level: Optional[str] = None

class TokenRequest(BaseModel):
    username: str
    password: str


# ── App setup ─────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="CCTV Surveillance System API",
    description="Production-grade CCTV face recognition and tracking API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"], allow_credentials=True)

# Serve static unknown faces
data_dir = Path(settings.data_dir)
if data_dir.exists():
    app.mount("/static", StaticFiles(directory=str(data_dir)), name="static")

# WebSocket manager
class WSManager:
    def __init__(self):
        self._connections: set = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self._connections.discard(ws)

    async def broadcast(self, data: dict):
        if not self._connections:
            return
        msg = json.dumps(data, default=str)
        dead = set()
        async with self._lock:
            conns = set(self._connections)
        for ws in conns:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._lock:
                self._connections -= dead

ws_manager = WSManager()
camera_ws_managers: Dict[str, WSManager] = {}


# ── Startup / Shutdown ────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    from backend.main import cctv_system
    cctv_system.initialize()
    asyncio.create_task(_broadcast_loop())
    logger.info("FastAPI started")

@app.on_event("shutdown")
async def shutdown():
    from backend.main import cctv_system
    cctv_system.shutdown()


async def _broadcast_loop():
    from backend.main import cctv_system
    while True:
        await asyncio.sleep(0.1)
        results = cctv_system.get_latest_frame()
        if results:
            await ws_manager.broadcast({"type": "metrics_update", "data": results})


# ── Auth endpoints ─────────────────────────────────────────────────────────────
@app.post("/api/auth/token")
async def get_token(req: TokenRequest):
    # Simple demo auth - extend with real user DB in production
    if req.username == "admin" and req.password == "admin":
        token = create_access_token({"sub": req.username})
        return {"access_token": token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")


# ── Camera endpoints ──────────────────────────────────────────────────────────
@app.post("/api/cameras", status_code=201)
async def add_camera(cam: CameraCreate):
    try:
        existing = CameraDAO.get_by_camera_id(cam.camera_id)
        if existing:
            raise HTTPException(status_code=409, detail="Camera ID already exists")

        data = cam.dict()
        track_cfg = data.pop("track_config") or {}
        if track_cfg:
            data["track_max_age"] = track_cfg.get("max_age", 30)
            data["track_min_hits"] = track_cfg.get("min_hits", 3)
            data["track_max_iou_distance"] = track_cfg.get("max_iou_distance", 0.7)
            data["track_max_cosine_distance"] = track_cfg.get("max_cosine_distance", 0.4)

        CameraDAO.create(data)

        from backend.main import cctv_system
        config = CameraConfig(
            camera_id=cam.camera_id, rtsp_url=cam.rtsp_url, name=cam.name or "",
            location=cam.location or "", processing_fps=cam.processing_fps,
            confidence_threshold=cam.confidence_threshold, enabled=cam.enabled)
        cctv_system.stream_manager.add_camera(config)

        return {"status": "success", "camera_id": cam.camera_id, "message": "Camera added"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add camera error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cameras")
async def list_cameras():
    cameras = CameraDAO.get_all()
    from backend.main import cctv_system
    statuses = cctv_system.stream_manager.get_status()
    for cam in cameras:
        cam_status = statuses.get(cam["camera_id"], {})
        cam["runtime_status"] = cam_status
    return {"status": "success", "cameras": cameras}


@app.get("/api/cameras/{camera_id}")
async def get_camera(camera_id: str):
    cam = CameraDAO.get_by_camera_id(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"status": "success", "camera": cam}


@app.put("/api/cameras/{camera_id}")
async def update_camera(camera_id: str, data: CameraUpdate):
    cam = CameraDAO.get_by_camera_id(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    update_data = {k: v for k, v in data.dict().items() if v is not None}
    CameraDAO.update(camera_id, update_data)
    return {"status": "success", "message": "Camera updated"}


@app.delete("/api/cameras/{camera_id}")
async def delete_camera(camera_id: str):
    cam = CameraDAO.get_by_camera_id(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    from backend.main import cctv_system
    cctv_system.stream_manager.remove_camera(camera_id)
    CameraDAO.delete(camera_id)
    return {"status": "success", "message": "Camera removed"}


@app.post("/api/cameras/{camera_id}/test")
async def test_camera(camera_id: str):
    cam = CameraDAO.get_by_camera_id(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    from backend.main import cctv_system
    result = cctv_system.stream_manager.test_connection(cam["rtsp_url"])
    return {"status": "success", **result}


@app.get("/api/cameras/{camera_id}/stats")
async def camera_stats(camera_id: str):
    from backend.main import cctv_system
    status = cctv_system.stream_manager.get_status(camera_id)
    return {"status": "success", "camera_id": camera_id, "stats": status}


@app.get("/api/cameras/{camera_id}/frame")
async def camera_frame(camera_id: str):
    """Return the latest annotated frame for a camera as base64 JPEG."""
    from backend.main import cctv_system
    frame_data = cctv_system.get_latest_frame(camera_id)
    if not frame_data:
        return {"status": "no_frame", "frame_b64": None, "detections": []}
    return {
        "status": "success",
        "frame_b64": frame_data.get("frame_b64"),
        "detections": frame_data.get("detections", []),
        "metrics": frame_data.get("metrics", {}),
        "timestamp": frame_data.get("timestamp"),
    }


# ── Known Persons endpoints ───────────────────────────────────────────────────
@app.post("/api/known-persons", status_code=201)
async def add_known_person(person: KnownPersonCreate):
    try:
        person_id = KnownPersonsDAO.create(
            name=person.name, employee_id=person.employee_id,
            department=person.department, description=person.description)

        embedding_generated = False
        if person.face_image:
            from backend.main import cctv_system
            from backend.utils import decode_image_base64
            img = decode_image_base64(person.face_image)
            img_path = save_face_image(img, settings.known_faces_dir, f"person_{person_id}")
            embedding = cctv_system.recognition_engine.encoder.encode(img)
            emb_bytes = embedding_to_bytes(embedding) if embedding is not None else None
            KnownPersonsDAO.add_image(person_id, img_path, emb_bytes)
            if embedding is not None:
                cctv_system.recognition_engine.cache.add(person_id, person.name, embedding)
            embedding_generated = embedding is not None

        return {"status": "success", "person_id": person_id,
                "message": "Person registered", "embedding_generated": embedding_generated}
    except Exception as e:
        logger.error(f"Add known person error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/known-persons")
async def list_known_persons():
    persons = KnownPersonsDAO.get_all()
    return {"status": "success", "persons": persons, "count": len(persons)}


@app.get("/api/known-persons/{person_id}")
async def get_known_person(person_id: int):
    person = KnownPersonsDAO.get_by_id(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    person.pop("face_embedding", None)
    return {"status": "success", "person": person}


@app.put("/api/known-persons/{person_id}")
async def update_known_person(person_id: int, data: KnownPersonUpdate):
    person = KnownPersonsDAO.get_by_id(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    update_data = {k: v for k, v in data.dict().items() if v is not None}
    KnownPersonsDAO.update(person_id, update_data)
    return {"status": "success", "message": "Person updated"}


@app.delete("/api/known-persons/{person_id}")
async def delete_known_person(person_id: int):
    person = KnownPersonsDAO.get_by_id(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    KnownPersonsDAO.delete(person_id)
    return {"status": "success", "message": "Person deleted"}


@app.post("/api/known-persons/{person_id}/image")
async def add_person_image(person_id: int, file: UploadFile = File(...)):
    person = KnownPersonsDAO.get_by_id(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Person not found")
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    img_path = save_face_image(img, settings.known_faces_dir, f"person_{person_id}")
    from backend.main import cctv_system
    embedding = cctv_system.recognition_engine.encoder.encode(img)
    emb_bytes = embedding_to_bytes(embedding) if embedding is not None else None
    KnownPersonsDAO.add_image(person_id, img_path, emb_bytes)
    if embedding is not None:
        cctv_system.recognition_engine.cache.add(person_id, person["name"], embedding)
    return {"status": "success", "image_path": img_path, "embedding_generated": embedding is not None}


@app.get("/api/known-persons/{person_id}/images")
async def get_person_images(person_id: int):
    images = KnownPersonsDAO.get_images(person_id)
    for img in images:
        img.pop("face_embedding", None)
    return {"status": "success", "images": images}


# ── Unknown Persons endpoints ─────────────────────────────────────────────────
@app.get("/api/unknown-persons")
async def list_unknown_persons(
    camera_id: Optional[str] = None,
    min_appearances: Optional[int] = None,
    marked_as_known: Optional[bool] = None,
    sort_by: str = "last_seen",
    order: str = "desc",
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    filters = {"camera_id": camera_id, "min_appearances": min_appearances,
               "marked_as_known": marked_as_known, "sort_by": sort_by,
               "order": order, "limit": limit, "offset": offset}
    unknowns = UnknownPersonsDAO.get_all(filters)
    for u in unknowns:
        u.pop("face_embedding", None)
    return {"status": "success", "unknowns": unknowns, "count": len(unknowns)}


@app.get("/api/unknown-persons/stats")
async def unknown_persons_stats():
    stats = UnknownPersonsDAO.get_stats()
    for u in stats.get("trending", []):
        u.pop("face_embedding", None)
    return {"status": "success", "stats": stats}


@app.get("/api/unknown-persons/trending")
async def trending_unknowns():
    unknowns = UnknownPersonsDAO.get_all({"sort_by": "appearance_count", "order": "desc", "limit": 20})
    for u in unknowns:
        u.pop("face_embedding", None)
    return {"status": "success", "trending": unknowns}


@app.get("/api/unknown-persons/{unknown_id}")
async def get_unknown_person(unknown_id: int):
    person = UnknownPersonsDAO.get_by_id(unknown_id)
    if not person:
        raise HTTPException(status_code=404, detail="Unknown person not found")
    person.pop("face_embedding", None)
    return {"status": "success", "person": person}


@app.delete("/api/unknown-persons/{unknown_id}")
async def delete_unknown_person(unknown_id: int):
    person = UnknownPersonsDAO.get_by_id(unknown_id)
    if not person:
        raise HTTPException(status_code=404, detail="Unknown person not found")
    UnknownPersonsDAO.delete(unknown_id)
    return {"status": "success", "message": "Unknown person deleted"}


@app.post("/api/unknown-persons/{unknown_id}/mark-known")
async def mark_unknown_as_known(unknown_id: int, req: MarkAsKnownRequest):
    person = UnknownPersonsDAO.get_by_id(unknown_id)
    if not person:
        raise HTTPException(status_code=404, detail="Unknown person not found")
    known = KnownPersonsDAO.get_by_id(req.person_id)
    if not known:
        raise HTTPException(status_code=404, detail="Known person not found")
    UnknownPersonsDAO.mark_as_known(unknown_id, req.person_id)
    return {"status": "success", "message": "Marked as known"}


# ── Detection endpoints ────────────────────────────────────────────────────────
@app.get("/api/detections/timeline")
async def detection_timeline(
    camera_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    events = DetectionDAO.get_timeline(camera_id, limit, offset)
    return {"status": "success", "events": events, "count": len(events)}


@app.get("/api/detections/person/{person_id}")
async def person_detection_timeline(person_id: int):
    events = DetectionDAO.get_person_timeline(person_id)
    return {"status": "success", "events": events, "count": len(events)}


@app.post("/api/recognition/test")
async def test_recognition(file: UploadFile = File(...)):
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    from backend.main import cctv_system
    result = cctv_system.recognition_engine.recognize(img)
    return {
        "status": "success",
        "person_name": result.person_name,
        "is_known": result.is_known,
        "confidence": result.confidence,
        "person_id": result.person_id,
    }


# ── System endpoints ───────────────────────────────────────────────────────────
@app.get("/api/system/health")
async def health_check():
    stats = get_system_stats()
    try:
        DatabaseManager.execute_one("SELECT 1")
        db_healthy = True
    except Exception:
        db_healthy = False
    from backend.main import cctv_system
    active_cameras = len(cctv_system.stream_manager.get_all_camera_ids()) if cctv_system.stream_manager else 0
    return {
        "status": "healthy" if db_healthy else "degraded",
        "database": "connected" if db_healthy else "error",
        "active_cameras": active_cameras,
        "system": stats,
    }


@app.get("/api/system/stats")
async def system_stats():
    stats = get_system_stats()
    total_known = len(KnownPersonsDAO.get_all())
    unknown_stats = UnknownPersonsDAO.get_stats()
    return {
        "status": "success",
        "system": stats,
        "known_persons": total_known,
        "unknown_persons_total": unknown_stats["total"],
        "unknown_persons_today": unknown_stats["today"],
    }


@app.get("/api/system/logs")
async def get_system_logs(
    severity: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    logs = SystemLogDAO.get_logs(severity, limit, offset)
    return {"status": "success", "logs": logs, "count": len(logs)}


@app.post("/api/system/restart")
async def restart_system():
    from backend.main import cctv_system
    cctv_system.shutdown()
    cctv_system.initialize()
    return {"status": "success", "message": "System restarted"}


@app.get("/api/system/config")
async def get_config():
    return {
        "status": "success",
        "config": {
            "face_match_threshold": settings.face_match_threshold,
            "yolo_confidence": settings.yolo_confidence,
            "default_processing_fps": settings.default_processing_fps,
            "deepsort_max_age": settings.deepsort_max_age,
            "deepsort_min_hits": settings.deepsort_min_hits,
            "log_level": settings.log_level,
        }
    }


# ── WebSocket endpoints ────────────────────────────────────────────────────────
@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)


@app.websocket("/ws/camera/{camera_id}")
async def ws_camera(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    try:
        from backend.main import cctv_system
        while True:
            await asyncio.sleep(0.1)
            frame_data = cctv_system.get_latest_frame(camera_id)
            if frame_data:
                await websocket.send_text(json.dumps(frame_data, default=str))
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/metrics")
async def ws_metrics(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await asyncio.sleep(1.0)
            stats = get_system_stats()
            from backend.main import cctv_system
            cam_statuses = cctv_system.stream_manager.get_status() if cctv_system.stream_manager else {}
            await websocket.send_text(json.dumps({
                "type": "metrics",
                "system": stats,
                "cameras": cam_statuses,
                "timestamp": datetime.utcnow().isoformat(),
            }, default=str))
    except WebSocketDisconnect:
        pass
