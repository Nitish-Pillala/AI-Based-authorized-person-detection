import cv2
import threading
import time
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from backend.utils import app_logger as logger, FPSCounter


@dataclass
class CameraConfig:
    camera_id: str
    rtsp_url: str
    name: str = ""
    location: str = ""
    processing_fps: int = 15
    confidence_threshold: float = 0.75
    enabled: bool = True
    track_max_age: int = 30
    track_min_hits: int = 3
    track_max_iou_distance: float = 0.7
    track_max_cosine_distance: float = 0.4


@dataclass
class CameraStatus:
    camera_id: str
    connected: bool = False
    fps: float = 0.0
    frames_processed: int = 0
    frames_dropped: int = 0
    last_frame_at: Optional[datetime] = None
    error: Optional[str] = None
    reconnect_count: int = 0


class RTSPStreamHandler:
    MAX_RECONNECT_WAIT = 30

    def __init__(self, config: CameraConfig, frame_callback: Callable):
        self.config = config
        self.frame_callback = frame_callback
        self.status = CameraStatus(camera_id=config.camera_id)
        # If URL is a number like "0", "1", "2" → use as webcam index
        url = int(self.config.rtsp_url) if self.config.rtsp_url.isdigit() else self.config.rtsp_url
        self._cap = cv2.VideoCapture(url)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._fps_counter = FPSCounter()
        self._frame_interval = 1.0 / max(1, config.processing_fps)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True,
                                        name=f"cam-{self.config.camera_id}")
        self._thread.start()
        logger.info(f"Started stream handler for camera {self.config.camera_id}")

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"Stopped stream handler for camera {self.config.camera_id}")

    def _connect(self) -> bool:
        if self._cap:
            self._cap.release()
        url = int(self.config.rtsp_url) if self.config.rtsp_url.isdigit() else self.config.rtsp_url
        self._cap = cv2.VideoCapture(url)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        connected = self._cap.isOpened()
        self.status.connected = connected
        if connected:
            logger.info(f"Camera {self.config.camera_id} connected")
        else:
            logger.warning(f"Camera {self.config.camera_id} connection failed")
        return connected

    def _stream_loop(self):
        reconnect_wait = 1
        while self._running:
            if not self._connect():
                self.status.error = "Connection failed"
                time.sleep(min(reconnect_wait, self.MAX_RECONNECT_WAIT))
                reconnect_wait = min(reconnect_wait * 2, self.MAX_RECONNECT_WAIT)
                self.status.reconnect_count += 1
                continue

            reconnect_wait = 1
            last_process_time = 0

            while self._running and self._cap.isOpened():
                ret, frame = self._cap.read()
                if not ret:
                    self.status.connected = False
                    self.status.error = "Stream read failed"
                    break

                now = time.time()
                if now - last_process_time < self._frame_interval:
                    continue
                last_process_time = now

                self._fps_counter.tick()
                self.status.fps = self._fps_counter.fps
                self.status.frames_processed += 1
                self.status.last_frame_at = datetime.utcnow()
                self.status.error = None

                try:
                    self.frame_callback(self.config.camera_id, frame, self.config)
                except Exception as e:
                    logger.error(f"Frame callback error [{self.config.camera_id}]: {e}")

        self.status.connected = False

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()


class MultiCameraStreamManager:
    def __init__(self, frame_callback: Callable, max_cameras: int = 50):
        self._cameras: Dict[str, RTSPStreamHandler] = {}
        self._configs: Dict[str, CameraConfig] = {}
        self._frame_callback = frame_callback
        self._max_cameras = max_cameras
        self._lock = threading.Lock()

    def add_camera(self, config: CameraConfig) -> bool:
        with self._lock:
            if config.camera_id in self._cameras:
                logger.warning(f"Camera {config.camera_id} already exists")
                return False
            if len(self._cameras) >= self._max_cameras:
                logger.error("Max cameras reached")
                return False
            handler = RTSPStreamHandler(config, self._frame_callback)
            self._cameras[config.camera_id] = handler
            self._configs[config.camera_id] = config
            if config.enabled:
                handler.start()
            return True

    def remove_camera(self, camera_id: str) -> bool:
        with self._lock:
            handler = self._cameras.pop(camera_id, None)
            self._configs.pop(camera_id, None)
            if handler:
                handler.stop()
                return True
            return False

    def update_camera(self, camera_id: str, config: CameraConfig) -> bool:
        self.remove_camera(camera_id)
        return self.add_camera(config)

    def get_status(self, camera_id: str = None) -> Dict:
        with self._lock:
            if camera_id:
                handler = self._cameras.get(camera_id)
                return vars(handler.status) if handler else {}
            return {cid: vars(h.status) for cid, h in self._cameras.items()}

    def get_all_camera_ids(self) -> list:
        return list(self._cameras.keys())

    def test_connection(self, rtsp_url: str) -> dict:
        url = int(rtsp_url) if rtsp_url.isdigit() else rtsp_url
        cap = cv2.VideoCapture(url)
        connected = cap.isOpened()
        if connected:
            ret, _ = cap.read()
            connected = ret
        cap.release()
        return {"connected": connected, "url": rtsp_url}

    def stop_all(self):
        with self._lock:
            for handler in self._cameras.values():
                handler.stop()
            self._cameras.clear()
            self._configs.clear()
        logger.info("All camera streams stopped")
