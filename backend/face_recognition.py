import numpy as np
import time
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from backend.config import settings
from backend.utils import app_logger as logger, cosine_similarity, embedding_to_bytes, bytes_to_embedding


@dataclass
class RecognitionResult:
    person_id: Optional[int]
    person_name: str
    confidence: float
    is_known: bool
    embedding: Optional[np.ndarray] = None


class InsightFaceEncoder:
    def __init__(self):
        self._app = None
        self._load_model()

    def _load_model(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace loaded successfully")
        except Exception as e:
            logger.error(f"InsightFace load error: {e}")
            self._app = None

    def encode(self, face_image) -> Optional[np.ndarray]:
        if self._app is None:
            return self._fallback_encode(face_image)
        try:
            import cv2
            if face_image is None or face_image.size == 0:
                return None
            if face_image.shape[0] < 32 or face_image.shape[1] < 32:
                face_image = cv2.resize(face_image, (112, 112))

            # InsightFace's app.get() runs its own internal face detector which
            # needs surrounding context to locate a face.  A tightly-cropped
            # YOLOv8 output often has zero context, causing the detector to
            # return an empty list.  Adding ~30% padding on each side gives the
            # internal detector enough border to succeed.
            h, w = face_image.shape[:2]
            pad_h = int(h * 0.3)
            pad_w = int(w * 0.3)
            padded = cv2.copyMakeBorder(
                face_image, pad_h, pad_h, pad_w, pad_w,
                cv2.BORDER_REPLICATE
            )

            faces = self._app.get(padded)
            if not faces:
                # Fallback: try on the original crop in case padding confused
                # the detector (e.g. very low-res crop)
                faces = self._app.get(face_image)

            if faces:
                emb = faces[0].embedding
                norm = np.linalg.norm(emb)
                return emb / (norm + 1e-10)
            return None
        except Exception as e:
            logger.error(f"InsightFace encode error: {e}")
            return None

    def encode_batch(self, face_images: List) -> List[Optional[np.ndarray]]:
        return [self.encode(img) for img in face_images]

    def _fallback_encode(self, face_image) -> Optional[np.ndarray]:
        try:
            import cv2
            if face_image is None or face_image.size == 0:
                return None
            resized = cv2.resize(face_image, (64, 64))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            flat = gray.flatten().astype(np.float32)
            norm = np.linalg.norm(flat)
            return flat / (norm + 1e-10)
        except Exception as e:
            logger.error(f"Fallback encode error: {e}")
            return None


class FaceEmbeddingCache:
    def __init__(self, refresh_minutes: int = None):
        self._cache: Dict[int, List[Tuple[str, np.ndarray]]] = {}
        self._last_refresh: Optional[float] = None
        self._refresh_minutes = refresh_minutes or settings.embedding_cache_refresh_minutes
        self._lock = __import__("threading").Lock()

    def load(self, embeddings_data: List[Dict]) -> None:
        with self._lock:
            self._cache.clear()
            for row in embeddings_data:
                pid = row["id"]
                name = row["name"]
                emb_bytes = row.get("face_embedding")
                if emb_bytes:
                    emb = bytes_to_embedding(emb_bytes)
                    if pid not in self._cache:
                        self._cache[pid] = []
                    self._cache[pid].append((name, emb))
            self._last_refresh = time.time()
            total = sum(len(v) for v in self._cache.values())
            logger.info(f"Embedding cache loaded: {len(self._cache)} persons, {total} embeddings")

    def needs_refresh(self) -> bool:
        if self._last_refresh is None:
            return True
        return time.time() - self._last_refresh > self._refresh_minutes * 60

    def get_all(self) -> Dict[int, List[Tuple[str, np.ndarray]]]:
        with self._lock:
            return dict(self._cache)

    def add(self, person_id: int, name: str, embedding: np.ndarray) -> None:
        with self._lock:
            if person_id not in self._cache:
                self._cache[person_id] = []
            self._cache[person_id].append((name, embedding))


class SimilarityMatcher:
    def __init__(self, threshold: float = None):
        self.threshold = threshold or settings.face_match_threshold

    def match(self, query_embedding: np.ndarray,
              cache: FaceEmbeddingCache) -> RecognitionResult:
        if query_embedding is None:
            return RecognitionResult(None, "Unknown", 0.0, False, None)

        best_score = 0.0
        best_pid = None
        best_name = "Unknown"

        for pid, face_list in cache.get_all().items():
            for name, emb in face_list:
                score = cosine_similarity(query_embedding, emb)
                if score > best_score:
                    best_score = score
                    best_pid = pid
                    best_name = name

        is_known = best_score >= self.threshold
        return RecognitionResult(
            person_id=best_pid if is_known else None,
            person_name=best_name if is_known else "Unknown",
            confidence=best_score,
            is_known=is_known,
            embedding=query_embedding,
        )


class FaceRecognitionEngine:
    def __init__(self):
        self.encoder = InsightFaceEncoder()
        self.cache = FaceEmbeddingCache()
        self.matcher = SimilarityMatcher()
        self._db_loaded = False

    def refresh_cache(self, force: bool = False) -> None:
        if force or self.cache.needs_refresh():
            from backend.database import KnownPersonsDAO
            embeddings = KnownPersonsDAO.get_all_embeddings()
            self.cache.load(embeddings)
            self._db_loaded = True

    def recognize(self, face_image) -> RecognitionResult:
        if not self._db_loaded or self.cache.needs_refresh():
            self.refresh_cache()
        embedding = self.encoder.encode(face_image)
        if embedding is None:
            return RecognitionResult(None, "Unknown", 0.0, False, None)
        result = self.matcher.match(embedding, self.cache)
        result.embedding = embedding
        return result

    def recognize_batch(self, face_images: List) -> List[RecognitionResult]:
        if not self._db_loaded or self.cache.needs_refresh():
            self.refresh_cache()
        embeddings = self.encoder.encode_batch(face_images)
        return [self.matcher.match(e, self.cache) if e is not None
                else RecognitionResult(None, "Unknown", 0.0, False, None)
                for e in embeddings]
