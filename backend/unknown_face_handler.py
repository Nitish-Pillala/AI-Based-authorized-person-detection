import numpy as np
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from backend.config import settings
from backend.database import UnknownPersonsDAO, DetectionDAO, SystemLogDAO
from backend.utils import (app_logger as logger, save_face_image,
                            calculate_face_quality, embedding_to_bytes,
                            bytes_to_embedding, cosine_similarity)

# Similarity threshold for matching an unknown face against existing unknowns
# (lower than known-person threshold — we just need to avoid duplicates)
UNKNOWN_DEDUP_THRESHOLD = 0.40


class UnknownFaceHandler:
    def __init__(self):
        self._storage_dir = settings.unknown_faces_dir
        Path(self._storage_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core handler
    # ------------------------------------------------------------------
    def handle(self, camera_id: str, track_id: str, face_image: np.ndarray,
               confidence: float, embedding: Optional[np.ndarray],
               track_manager=None) -> Optional[Dict]:
        """
        Process an unknown face detection.
        Deduplication strategy (3 layers):
          1. In-memory track set (same session, same track_id)
          2. DB lookup by (track_id, camera_id) — same session, same camera
          3. Embedding similarity against ALL stored unknowns — cross-session
        Returns dict with action taken and unknown_id.
        """
        str_track_id = str(track_id)

        # ── Layer 1 & 2: track_id + camera_id DB lookup ────────────────
        existing = UnknownPersonsDAO.get_by_track_camera(str_track_id, camera_id)
        if existing:
            return self._update_existing(existing, face_image, camera_id, confidence)

        # ── Layer 3: embedding similarity across ALL unknowns ──────────
        if embedding is not None:
            matched = self._find_by_embedding(embedding)
            if matched:
                logger.info(
                    f"Duplicate unknown detected via embedding match "
                    f"(new track={str_track_id} → existing id={matched['id']})"
                )
                # Update the existing record with the new track_id so future
                # frames in this session hit Layer 1/2 directly (faster)
                UnknownPersonsDAO.update_track_id(matched["id"], str_track_id, camera_id)
                return self._update_existing(matched, face_image, camera_id, confidence)

        # ── No match anywhere — genuinely new unknown person ───────────
        if face_image is None or face_image.size == 0:
            return None

        img_path = save_face_image(face_image, self._storage_dir, f"unknown_{camera_id}")
        emb_bytes = embedding_to_bytes(embedding) if embedding is not None else None

        unknown_id = UnknownPersonsDAO.create(
            track_id=str_track_id,
            image_path=img_path,
            camera_id=camera_id,
            confidence=confidence,
            embedding=emb_bytes,
            first_seen=datetime.utcnow(),
        )
        UnknownPersonsDAO.add_image(unknown_id, img_path, camera_id,
                                    confidence, datetime.utcnow())

        SystemLogDAO.log(
            event_type="unknown_person_detected",
            camera_id=camera_id,
            description=f"New unknown person detected (track_id={str_track_id})",
            severity="warning",
            details={"track_id": str_track_id, "confidence": confidence},
        )

        logger.info(
            f"New unknown person saved: id={unknown_id} "
            f"camera={camera_id} track={str_track_id}"
        )
        return {"action": "created", "unknown_id": unknown_id, "image_path": img_path}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_existing(self, existing: Dict, face_image, camera_id: str,
                          confidence: float) -> Dict:
        unknown_id = existing["id"]
        quality = calculate_face_quality(face_image) if face_image is not None else 0
        best_image_path = None

        if quality > 0.5 and face_image is not None:
            img_path = save_face_image(
                face_image, self._storage_dir, f"unknown_{camera_id}")
            UnknownPersonsDAO.add_image(
                unknown_id, img_path, camera_id, confidence, datetime.utcnow())
            if quality > float(existing.get("highest_confidence") or 0):
                best_image_path = img_path

        UnknownPersonsDAO.update_encounter(
            unknown_id, datetime.utcnow(), confidence, best_image_path)
        return {"action": "updated", "unknown_id": unknown_id}

    def _find_by_embedding(self, query_emb: np.ndarray) -> Optional[Dict]:
        """Compare query embedding against all stored unknown embeddings."""
        try:
            rows = UnknownPersonsDAO.get_all_embeddings()
            best_score = 0.0
            best_row = None
            for row in rows:
                raw = row.get("face_embedding")
                if not raw:
                    continue
                stored_emb = bytes_to_embedding(raw)
                if stored_emb.shape != query_emb.shape:
                    continue
                score = cosine_similarity(query_emb, stored_emb)
                if score > best_score:
                    best_score = score
                    best_row = row
            if best_score >= UNKNOWN_DEDUP_THRESHOLD:
                logger.debug(f"Embedding match found: similarity={best_score:.3f}")
                return best_row
        except Exception as e:
            logger.error(f"Embedding dedup check error: {e}")
        return None


class KnownFaceHandler:
    """Handles known person detection events."""

    def handle(self, camera_id: str, track_id: str, person_id: int,
               person_name: str, confidence: float, bbox: list) -> None:
        try:
            from backend.database import KnownPersonsDAO
            KnownPersonsDAO.increment_detection(person_id)
            DetectionDAO.log(
                camera_id=camera_id,
                track_id=str(track_id),
                person_name=person_name,
                confidence=confidence,
                bbox=bbox,
                person_id=person_id,
                detected_at=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"Known face handler error: {e}")
