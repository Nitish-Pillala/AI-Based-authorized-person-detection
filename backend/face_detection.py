import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from backend.config import settings
from backend.utils import app_logger as logger


@dataclass
class DetectionResult:
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    face_image: Optional[np.ndarray] = None


class YOLOv8FaceDetector:
    def __init__(self, model_path: str = None, confidence: float = None):
        self.model_path = model_path or settings.yolo_model
        self.confidence = confidence or settings.yolo_confidence
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            logger.info(f"YOLOv8 model loaded: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}. Will use fallback detector.")
            self._model = None

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        if self._model is None:
            return self._fallback_detect(frame)
        try:
            results = self._model(frame, conf=self.confidence, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < self.confidence:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    if (x2 - x1) < settings.min_face_size or (y2 - y1) < settings.min_face_size:
                        continue
                    face_img = frame[y1:y2, x1:x2]
                    detections.append(DetectionResult(
                        bbox=[x1, y1, x2, y2], confidence=conf, face_image=face_img))
            return detections
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return self._fallback_detect(frame)

    def _fallback_detect(self, frame: np.ndarray) -> List[DetectionResult]:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(settings.min_face_size, settings.min_face_size))
            detections = []
            for (x, y, w, h) in faces:
                x2, y2 = x + w, y + h
                face_img = frame[y:y2, x:x2]
                detections.append(DetectionResult(bbox=[x, y, x2, y2], confidence=0.8, face_image=face_img))
            return detections
        except Exception as e:
            logger.error(f"Fallback detection error: {e}")
            return []

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[DetectionResult]]:
        return [self.detect(f) for f in frames]


class DetectionProcessor:
    def __init__(self, detector: YOLOv8FaceDetector = None, processing_height: int = 480):
        self.detector = detector or YOLOv8FaceDetector()
        self.processing_height = processing_height

    def process_frame(self, frame: np.ndarray, confidence_threshold: float = None) -> Tuple[List[DetectionResult], np.ndarray]:
        orig_h, orig_w = frame.shape[:2]
        if orig_h > self.processing_height:
            scale = self.processing_height / orig_h
            proc_w = int(orig_w * scale)
            proc_frame = cv2.resize(frame, (proc_w, self.processing_height))
        else:
            proc_frame = frame
            scale = 1.0

        if confidence_threshold is not None:
            orig_conf = self.detector.confidence
            self.detector.confidence = confidence_threshold

        detections = self.detector.detect(proc_frame)

        if confidence_threshold is not None:
            self.detector.confidence = orig_conf

        if scale != 1.0:
            for d in detections:
                d.bbox = [int(v / scale) for v in d.bbox]
                x1, y1, x2, y2 = d.bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                d.bbox = [x1, y1, x2, y2]
                d.face_image = frame[y1:y2, x1:x2]

        return detections, frame
