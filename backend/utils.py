import cv2
import numpy as np
import base64
import uuid
import time
import psutil
from datetime import datetime
from loguru import logger
from pathlib import Path
import sys
from backend.config import settings


def setup_logger():
    logger.remove()
    logger.add(sys.stdout, level=settings.log_level, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")
    log_path = Path(settings.logs_dir) / "cctv_{time:YYYY-MM-DD}.log"
    logger.add(str(log_path), level="DEBUG", rotation="1 day", retention="30 days",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    return logger


def encode_image_base64(image: np.ndarray, quality: int = 85) -> str:
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode("utf-8")


def decode_image_base64(b64_string: str) -> np.ndarray:
    img_bytes = base64.b64decode(b64_string)
    arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    return embedding.astype(np.float32).tobytes()


def bytes_to_embedding(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a_norm, b_norm))


def generate_unique_id() -> str:
    return str(uuid.uuid4())[:8].upper()


def save_face_image(face_img: np.ndarray, directory: str, prefix: str = "face") -> str:
    Path(directory).mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{generate_unique_id()}.jpg"
    filepath = str(Path(directory) / filename)
    cv2.imwrite(filepath, face_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return filepath


def calculate_face_quality(face_img: np.ndarray) -> float:
    if face_img is None or face_img.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    h, w = gray.shape[:2]
    size_score = min(1.0, (h * w) / (128 * 128))
    sharpness_score = min(1.0, laplacian_var / 500.0)
    return 0.4 * size_score + 0.6 * sharpness_score


def draw_detection(frame: np.ndarray, bbox: list, label: str, confidence: float,
                   track_id: int = None, is_known: bool = True) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = (0, 255, 0) if is_known else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    display = f"{label}"
    if confidence:
        display += f" ({confidence:.0%})"
    if track_id is not None:
        display += f" #{track_id}"

    text_size = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_size[0] + 4, y1), color, -1)
    cv2.putText(frame, display, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def get_system_stats() -> dict:
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_gb": round(psutil.virtual_memory().used / (1024 ** 3), 2),
        "disk_percent": psutil.disk_usage("/").percent,
        "timestamp": datetime.utcnow().isoformat(),
    }


class FPSCounter:
    def __init__(self, window: int = 30):
        self.timestamps = []
        self.window = window

    def tick(self):
        now = time.time()
        self.timestamps.append(now)
        cutoff = now - 1.0
        self.timestamps = [t for t in self.timestamps if t > cutoff]

    @property
    def fps(self) -> float:
        return float(len(self.timestamps))


app_logger = setup_logger()
