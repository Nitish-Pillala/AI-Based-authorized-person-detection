import mysql.connector
from mysql.connector import pooling
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
from backend.config import settings
from backend.utils import app_logger as logger


class DatabaseManager:
    _pool: Optional[pooling.MySQLConnectionPool] = None

    @classmethod
    def get_pool(cls) -> pooling.MySQLConnectionPool:
        if cls._pool is None:
            cls._pool = pooling.MySQLConnectionPool(
                pool_name="cctv_pool",
                pool_size=settings.db_pool_size,
                pool_reset_session=True,
                host=settings.db_host,
                port=settings.db_port,
                user=settings.db_user,
                password=settings.db_password,
                database=settings.db_name,
                charset="utf8mb4",
                autocommit=False,
            )
        return cls._pool

    @classmethod
    def get_connection(cls):
        return cls.get_pool().get_connection()

    @classmethod
    def execute(cls, query: str, params: tuple = (), fetch: bool = False) -> Any:
        conn = cls.get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params)
            if fetch:
                result = cursor.fetchall()
                return result
            conn.commit()
            return cursor.lastrowid
        except Exception as e:
            conn.rollback()
            logger.error(f"DB error: {e} | query: {query[:100]}")
            raise
        finally:
            cursor.close()
            conn.close()

    @classmethod
    def execute_one(cls, query: str, params: tuple = ()) -> Optional[Dict]:
        conn = cls.get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params)
            return cursor.fetchone()
        finally:
            cursor.close()
            conn.close()


class CameraDAO:
    @staticmethod
    def create(camera_data: dict) -> int:
        q = """INSERT INTO cameras (camera_id, name, location, rtsp_url, enabled,
               processing_fps, confidence_threshold, track_max_age, track_min_hits,
               track_max_iou_distance, track_max_cosine_distance)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        p = (camera_data["camera_id"], camera_data.get("name"), camera_data.get("location"),
             camera_data.get("rtsp_url"), camera_data.get("enabled", True),
             camera_data.get("processing_fps", 15), camera_data.get("confidence_threshold", 0.75),
             camera_data.get("track_max_age", 30), camera_data.get("track_min_hits", 3),
             camera_data.get("track_max_iou_distance", 0.7), camera_data.get("track_max_cosine_distance", 0.4))
        return DatabaseManager.execute(q, p)

    @staticmethod
    def get_all() -> List[Dict]:
        return DatabaseManager.execute("SELECT * FROM cameras ORDER BY id", fetch=True)

    @staticmethod
    def get_by_camera_id(camera_id: str) -> Optional[Dict]:
        return DatabaseManager.execute_one("SELECT * FROM cameras WHERE camera_id=%s", (camera_id,))

    @staticmethod
    def update(camera_id: str, data: dict) -> None:
        sets = ", ".join(f"{k}=%s" for k in data)
        values = list(data.values()) + [camera_id]
        DatabaseManager.execute(f"UPDATE cameras SET {sets} WHERE camera_id=%s", tuple(values))

    @staticmethod
    def delete(camera_id: str) -> None:
        DatabaseManager.execute("DELETE FROM cameras WHERE camera_id=%s", (camera_id,))

    @staticmethod
    def get_enabled() -> List[Dict]:
        return DatabaseManager.execute("SELECT * FROM cameras WHERE enabled=TRUE", fetch=True)


class KnownPersonsDAO:
    @staticmethod
    def create(name: str, employee_id: str = None, department: str = None,
               description: str = None, embedding: bytes = None) -> int:
        q = "INSERT INTO known_faces (name, employee_id, department, description, face_embedding) VALUES (%s,%s,%s,%s,%s)"
        return DatabaseManager.execute(q, (name, employee_id, department, description, embedding))

    @staticmethod
    def get_all() -> List[Dict]:
        return DatabaseManager.execute(
            "SELECT id, name, employee_id, department, description, total_detections, created_at FROM known_faces ORDER BY name",
            fetch=True)

    @staticmethod
    def get_by_id(person_id: int) -> Optional[Dict]:
        return DatabaseManager.execute_one("SELECT * FROM known_faces WHERE id=%s", (person_id,))

    @staticmethod
    def get_all_embeddings() -> List[Dict]:
        return DatabaseManager.execute(
            "SELECT kf.id, kf.name, kfi.face_embedding FROM known_faces kf "
            "JOIN known_face_images kfi ON kf.id = kfi.person_id WHERE kfi.face_embedding IS NOT NULL",
            fetch=True)

    @staticmethod
    def update(person_id: int, data: dict) -> None:
        sets = ", ".join(f"{k}=%s" for k in data)
        values = list(data.values()) + [person_id]
        DatabaseManager.execute(f"UPDATE known_faces SET {sets} WHERE id=%s", tuple(values))

    @staticmethod
    def delete(person_id: int) -> None:
        DatabaseManager.execute("DELETE FROM known_faces WHERE id=%s", (person_id,))

    @staticmethod
    def increment_detection(person_id: int) -> None:
        DatabaseManager.execute(
            "UPDATE known_faces SET total_detections=total_detections+1 WHERE id=%s", (person_id,))

    @staticmethod
    def add_image(person_id: int, image_path: str, embedding: bytes = None) -> int:
        return DatabaseManager.execute(
            "INSERT INTO known_face_images (person_id, image_path, face_embedding) VALUES (%s,%s,%s)",
            (person_id, image_path, embedding))

    @staticmethod
    def get_images(person_id: int) -> List[Dict]:
        return DatabaseManager.execute(
            "SELECT * FROM known_face_images WHERE person_id=%s ORDER BY uploaded_at DESC",
            (person_id,), fetch=True)


class UnknownPersonsDAO:
    @staticmethod
    def create(track_id: str, image_path: str, camera_id: str,
               confidence: float, embedding: bytes = None, first_seen: datetime = None) -> int:
        ts = first_seen or datetime.utcnow()
        q = """INSERT INTO unknown_faces (track_id, best_image_path, face_embedding,
               first_seen, last_seen, camera_id, highest_confidence)
               VALUES (%s,%s,%s,%s,%s,%s,%s)"""
        return DatabaseManager.execute(q, (track_id, image_path, embedding, ts, ts, camera_id, confidence))

    @staticmethod
    def get_by_track_camera(track_id: str, camera_id: str) -> Optional[Dict]:
        return DatabaseManager.execute_one(
            "SELECT * FROM unknown_faces WHERE track_id=%s AND camera_id=%s", (track_id, camera_id))

    @staticmethod
    def get_all_embeddings() -> List[Dict]:
        """Return id + embedding for all unknown faces that have an embedding stored."""
        return DatabaseManager.execute(
            "SELECT id, track_id, camera_id, face_embedding FROM unknown_faces "
            "WHERE face_embedding IS NOT NULL AND marked_as_known=FALSE",
            fetch=True)

    @staticmethod
    def update_track_id(unknown_id: int, new_track_id: str, camera_id: str) -> None:
        """Update track_id so current-session dedup (Layer 1/2) can find it faster."""
        try:
            DatabaseManager.execute(
                "UPDATE unknown_faces SET track_id=%s, camera_id=%s WHERE id=%s",
                (new_track_id, camera_id, unknown_id))
        except Exception:
            pass  # Ignore if UNIQUE constraint blocks it (another record already has this track)

    @staticmethod
    def update_encounter(unknown_id: int, last_seen: datetime, confidence: float,
                         best_image_path: str = None) -> None:
        if best_image_path:
            DatabaseManager.execute(
                "UPDATE unknown_faces SET last_seen=%s, appearance_count=appearance_count+1, "
                "highest_confidence=GREATEST(highest_confidence,%s), best_image_path=%s WHERE id=%s",
                (last_seen, confidence, best_image_path, unknown_id))
        else:
            DatabaseManager.execute(
                "UPDATE unknown_faces SET last_seen=%s, appearance_count=appearance_count+1, "
                "highest_confidence=GREATEST(highest_confidence,%s) WHERE id=%s",
                (last_seen, confidence, unknown_id))

    @staticmethod
    def get_all(filters: dict = None) -> List[Dict]:
        q = "SELECT * FROM unknown_faces WHERE 1=1"
        params = []
        if filters:
            if filters.get("camera_id"):
                q += " AND camera_id=%s"
                params.append(filters["camera_id"])
            if filters.get("min_appearances"):
                q += " AND appearance_count>=%s"
                params.append(filters["min_appearances"])
            if filters.get("marked_as_known") is not None:
                q += " AND marked_as_known=%s"
                params.append(filters["marked_as_known"])
            sort = filters.get("sort_by", "last_seen")
            order = filters.get("order", "desc").upper()
            if sort in ("last_seen", "first_seen", "appearance_count", "highest_confidence"):
                q += f" ORDER BY {sort} {order}"
            limit = filters.get("limit", 100)
            offset = filters.get("offset", 0)
            q += f" LIMIT %s OFFSET %s"
            params.extend([limit, offset])
        return DatabaseManager.execute(q, tuple(params), fetch=True)

    @staticmethod
    def get_by_id(unknown_id: int) -> Optional[Dict]:
        return DatabaseManager.execute_one("SELECT * FROM unknown_faces WHERE id=%s", (unknown_id,))

    @staticmethod
    def mark_as_known(unknown_id: int, person_id: int) -> None:
        DatabaseManager.execute(
            "UPDATE unknown_faces SET marked_as_known=TRUE, marked_as_person_id=%s WHERE id=%s",
            (person_id, unknown_id))

    @staticmethod
    def delete(unknown_id: int) -> None:
        DatabaseManager.execute("DELETE FROM unknown_faces WHERE id=%s", (unknown_id,))

    @staticmethod
    def add_image(unknown_id: int, image_path: str, camera_id: str,
                  confidence: float, detected_at: datetime = None) -> int:
        ts = detected_at or datetime.utcnow()
        return DatabaseManager.execute(
            "INSERT INTO unknown_face_images (unknown_id, image_path, camera_id, detected_at, confidence) VALUES (%s,%s,%s,%s,%s)",
            (unknown_id, image_path, camera_id, ts, confidence))

    @staticmethod
    def get_stats() -> Dict:
        total = DatabaseManager.execute_one("SELECT COUNT(*) as cnt FROM unknown_faces")
        today = DatabaseManager.execute_one(
            "SELECT COUNT(*) as cnt FROM unknown_faces WHERE DATE(first_seen)=CURDATE()")
        trending = DatabaseManager.execute(
            "SELECT * FROM unknown_faces ORDER BY appearance_count DESC LIMIT 10", fetch=True)
        return {"total": total["cnt"], "today": today["cnt"], "trending": trending}


class DetectionDAO:
    @staticmethod
    def log(camera_id: str, track_id: str, person_name: str, confidence: float,
            bbox: list, person_id: int = None, unknown_id: int = None,
            detected_at: datetime = None) -> int:
        ts = detected_at or datetime.utcnow()
        return DatabaseManager.execute(
            "INSERT INTO detections (camera_id, track_id, person_id, unknown_id, person_name, "
            "confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, detected_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (camera_id, track_id, person_id, unknown_id, person_name, confidence,
             *bbox[:4], ts))

    @staticmethod
    def get_timeline(camera_id: str = None, limit: int = 100, offset: int = 0) -> List[Dict]:
        q = "SELECT * FROM detections WHERE 1=1"
        params = []
        if camera_id:
            q += " AND camera_id=%s"
            params.append(camera_id)
        q += " ORDER BY detected_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        return DatabaseManager.execute(q, tuple(params), fetch=True)

    @staticmethod
    def get_person_timeline(person_id: int) -> List[Dict]:
        return DatabaseManager.execute(
            "SELECT * FROM detections WHERE person_id=%s ORDER BY detected_at DESC LIMIT 200",
            (person_id,), fetch=True)


class SystemLogDAO:
    @staticmethod
    def log(event_type: str, description: str, severity: str = "info",
            camera_id: str = None, details: dict = None) -> None:
        DatabaseManager.execute(
            "INSERT INTO system_logs (event_type, camera_id, description, details, severity) VALUES (%s,%s,%s,%s,%s)",
            (event_type, camera_id, description, json.dumps(details) if details else None, severity))

    @staticmethod
    def get_logs(severity: str = None, limit: int = 100, offset: int = 0) -> List[Dict]:
        q = "SELECT * FROM system_logs WHERE 1=1"
        params = []
        if severity:
            q += " AND severity=%s"
            params.append(severity)
        q += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        return DatabaseManager.execute(q, tuple(params), fetch=True)
