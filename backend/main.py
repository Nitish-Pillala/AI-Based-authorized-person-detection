import signal
import sys
import threading
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any
from backend.config import settings
from backend.utils import app_logger as logger
from backend.cctv_stream import MultiCameraStreamManager, CameraConfig
from backend.face_detection import DetectionProcessor
from backend.face_recognition import FaceRecognitionEngine
from backend.tracking import TrackManager
from backend.unknown_face_handler import UnknownFaceHandler, KnownFaceHandler
from backend.database import DatabaseManager, CameraDAO, SystemLogDAO, DetectionDAO


class CCTVSystem:
    def __init__(self):
        self.detection_processor = DetectionProcessor()
        self.recognition_engine = FaceRecognitionEngine()
        self.track_manager = TrackManager()
        self.unknown_handler = UnknownFaceHandler()
        self.known_handler = KnownFaceHandler()
        self.stream_manager: MultiCameraStreamManager = None
        self._running = False
        self._frame_results: Dict[str, Any] = {}
        self._results_lock = threading.Lock()
        self._ws_clients = set()
        self._ws_lock = threading.Lock()

    def initialize(self):
        logger.info("Initializing CCTV System...")
        self.recognition_engine.refresh_cache(force=True)
        self.stream_manager = MultiCameraStreamManager(
            frame_callback=self._on_frame,
            max_cameras=settings.max_cameras,
        )
        cameras = CameraDAO.get_enabled()
        for cam in cameras:
            config = CameraConfig(
                camera_id=cam["camera_id"],
                rtsp_url=cam["rtsp_url"],
                name=cam.get("name", ""),
                location=cam.get("location", ""),
                processing_fps=cam.get("processing_fps", 15),
                confidence_threshold=cam.get("confidence_threshold", 0.75),
                enabled=cam.get("enabled", True),
                track_max_age=cam.get("track_max_age", 30),
                track_min_hits=cam.get("track_min_hits", 3),
                track_max_iou_distance=cam.get("track_max_iou_distance", 0.7),
                track_max_cosine_distance=cam.get("track_max_cosine_distance", 0.4),
            )
            self.stream_manager.add_camera(config)

        self._running = True
        SystemLogDAO.log("system_start", "CCTV System started", severity="info")
        logger.info(f"CCTV System initialized with {len(cameras)} cameras")

    def _on_frame(self, camera_id: str, frame, camera_config: CameraConfig):
        try:
            detections, _ = self.detection_processor.process_frame(
                frame, camera_config.confidence_threshold)

            if not detections:
                self._update_results(camera_id, frame, [], {
                    "fps": 0, "active_persons": 0, "known_persons": 0, "unknown_persons": 0})
                return

            tracker = self.track_manager.get_tracker(camera_id, {
                "max_age": camera_config.track_max_age,
                "min_hits": camera_config.track_min_hits,
                "max_iou_distance": camera_config.track_max_iou_distance,
                "max_cosine_distance": camera_config.track_max_cosine_distance,
            })

            face_images = [d.face_image for d in detections]
            rec_results = self.recognition_engine.recognize_batch(face_images)

            track_inputs = []
            for det, rec in zip(detections, rec_results):
                track_inputs.append({
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                    "feature": rec.embedding,
                })

            tracked = tracker.update(track_inputs)

            frame_dets = []
            known_count = 0
            unknown_count = 0

            for i, track_info in enumerate(tracked):
                if i >= len(detections):
                    break
                det = detections[i]
                rec = rec_results[i]
                track_id = track_info["track_id"]
                is_new = track_info.get("is_new", False)

                if rec.is_known:
                    known_count += 1
                    if is_new or not self.track_manager.is_track_logged(camera_id, track_id):
                        self.known_handler.handle(
                            camera_id, track_id, rec.person_id, rec.person_name,
                            rec.confidence, det.bbox)
                        self.track_manager.mark_track_logged(camera_id, track_id)
                else:
                    unknown_count += 1
                    if is_new or not self.track_manager.is_track_logged(camera_id, track_id):
                        self.unknown_handler.handle(
                            camera_id, str(track_id), det.face_image,
                            det.confidence, rec.embedding)
                        self.track_manager.mark_track_logged(camera_id, track_id)

                frame_dets.append({
                    "track_id": int(track_id),
                    "person_name": rec.person_name if rec.is_known else "Unknown",
                    "confidence": float(rec.confidence) if rec.confidence is not None else None,
                    "bbox": [int(v) for v in det.bbox],
                    "is_known": bool(rec.is_known),
                    "is_new": bool(is_new),
                })

            metrics = {
                "active_persons": len(tracked),
                "known_persons": known_count,
                "unknown_persons": unknown_count,
            }

            self._update_results(camera_id, frame, frame_dets, metrics)

        except Exception as e:
            logger.error(f"Frame processing error [{camera_id}]: {e}")

    def _update_results(self, camera_id: str, frame, detections: list, metrics: dict):
        import cv2
        import base64
        from backend.utils import draw_detection

        for det in detections:
            draw_detection(frame, det["bbox"], det["person_name"],
                           det.get("confidence"), det.get("track_id"), det.get("is_known", False))

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        # Ensure all values are native Python types (no numpy scalars)
        safe_metrics = {k: int(v) if hasattr(v, 'item') else v
                        for k, v in metrics.items()}

        result = {
            "type": "frame_update",
            "camera_id": str(camera_id),
            "frame_b64": frame_b64,
            "detections": detections,   # already sanitized in _on_frame
            "metrics": safe_metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }

        with self._results_lock:
            self._frame_results[camera_id] = result

    def get_latest_frame(self, camera_id: str = None) -> Dict:
        with self._results_lock:
            if camera_id:
                return self._frame_results.get(camera_id, {})
            return dict(self._frame_results)

    def add_ws_client(self, client):
        with self._ws_lock:
            self._ws_clients.add(client)

    def remove_ws_client(self, client):
        with self._ws_lock:
            self._ws_clients.discard(client)

    def shutdown(self):
        self._running = False
        if self.stream_manager:
            self.stream_manager.stop_all()
        SystemLogDAO.log("system_stop", "CCTV System stopped", severity="info")
        logger.info("CCTV System shutdown complete")


# Global instance
cctv_system = CCTVSystem()


def main():
    def handle_signal(sig, frame):
        logger.info("Shutdown signal received")
        cctv_system.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    cctv_system.initialize()

    import uvicorn
    from backend.api.fastapi_server import app
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, reload=settings.api_reload)


if __name__ == "__main__":
    main()
