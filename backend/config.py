from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Database
    db_host: str = "localhost"
    db_port: int = 3306
    db_user: str = "root"
    db_password: str = "123456789"
    db_name: str = "cctv_surveillance"
    db_pool_size: int = 10

    # JWT
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False

    # Storage paths
    data_dir: str = "./data"
    unknown_faces_dir: str = "./data/unknown_faces"
    known_faces_dir: str = "./data/known_faces"
    logs_dir: str = "./data/logs"

    # YOLO
    yolo_model: str = "yolov8m-face.pt"
    yolo_confidence: float = 0.5

    # Face recognition
    face_match_threshold: float = 0.45
    min_face_size: int = 64
    embedding_cache_refresh_minutes: int = 60

    # Processing
    default_processing_fps: int = 15
    max_cameras: int = 50
    frame_buffer_size: int = 100

    # DeepSORT defaults
    deepsort_max_age: int = 30
    deepsort_min_hits: int = 3
    deepsort_max_iou_distance: float = 0.7
    deepsort_max_cosine_distance: float = 0.4

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def ensure_dirs(self):
        for d in [self.data_dir, self.unknown_faces_dir, self.known_faces_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
