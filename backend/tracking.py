import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from backend.utils import app_logger as logger


@dataclass
class Track:
    track_id: int
    bbox: List[float]
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    features: List[np.ndarray] = field(default_factory=list)
    is_confirmed: bool = False

    def __post_init__(self):
        self._kf = _init_kalman(self.bbox)

    def predict(self) -> List[float]:
        self._kf.predict()
        self.age += 1
        self.time_since_update += 1
        state = self._kf.x.flatten()
        cx, cy, s, r = state[0], state[1], state[2], state[3]
        w = np.sqrt(max(s * r, 1))
        h = max(s / w, 1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        self.bbox = [x1, y1, x1 + w, y1 + h]
        return self.bbox

    def update(self, bbox: List[float], feature: Optional[np.ndarray] = None) -> None:
        self._kf.update(_bbox_to_z(bbox))
        self.bbox = bbox
        self.hits += 1
        self.time_since_update = 0
        if feature is not None:
            self.features.append(feature)
            if len(self.features) > 10:
                self.features.pop(0)

    def get_mean_feature(self) -> Optional[np.ndarray]:
        if not self.features:
            return None
        stacked = np.stack(self.features)
        mean = stacked.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / (norm + 1e-10)


def _init_kalman(bbox: List[float]) -> KalmanFilter:
    kf = KalmanFilter(dim_x=7, dim_z=4)
    kf.F = np.array([
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]])
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0]])
    kf.R[2:, 2:] *= 10.0
    kf.P[4:, 4:] *= 1000.0
    kf.P *= 10.0
    kf.Q[-1, -1] *= 0.01
    kf.Q[4:, 4:] *= 0.01
    kf.x[:4] = _bbox_to_z(bbox)
    return kf


def _bbox_to_z(bbox: List[float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    s = (x2 - x1) * (y2 - y1)
    r = (x2 - x1) / max((y2 - y1), 1)
    return np.array([[cx], [cy], [s], [r]])


def _iou(bbox1: List[float], bbox2: List[float]) -> float:
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-10)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return 1.0 - float(np.dot(a_norm, b_norm))


class DeepSortTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 max_iou_distance: float = 0.7, max_cosine_distance: float = 0.4):
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        detections: list of {"bbox": [x1,y1,x2,y2], "confidence": float, "feature": ndarray|None}
        returns: list of {"track_id": int, "bbox": [...], "confidence": float, "is_new": bool}
        """
        # Predict all tracks
        for track in self._tracks.values():
            track.predict()

        if not detections:
            self._prune_tracks()
            return []

        det_bboxes = [d["bbox"] for d in detections]
        det_features = [d.get("feature") for d in detections]
        track_ids = list(self._tracks.keys())
        track_bboxes = [self._tracks[tid].bbox for tid in track_ids]

        matched, unmatched_dets, unmatched_tracks = self._associate(
            det_bboxes, track_bboxes, det_features, track_ids)

        # Update matched
        for det_idx, track_id in matched:
            self._tracks[track_id].update(det_bboxes[det_idx], det_features[det_idx])

        # Create new tracks for unmatched detections
        new_track_ids = []
        for det_idx in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            t = Track(track_id=tid, bbox=det_bboxes[det_idx])
            if det_features[det_idx] is not None:
                t.features.append(det_features[det_idx])
            self._tracks[tid] = t
            new_track_ids.append((det_idx, tid))

        self._prune_tracks()

        results = []
        # Matched
        for det_idx, track_id in matched:
            track = self._tracks.get(track_id)
            if track and track.hits >= self.min_hits:
                if not track.is_confirmed:
                    track.is_confirmed = True
                results.append({
                    "track_id": track_id,
                    "bbox": detections[det_idx]["bbox"],
                    "confidence": detections[det_idx]["confidence"],
                    "is_new": False,
                })

        # New tracks (only confirm after min_hits)
        for det_idx, track_id in new_track_ids:
            track = self._tracks.get(track_id)
            if track and track.hits >= self.min_hits:
                track.is_confirmed = True
                results.append({
                    "track_id": track_id,
                    "bbox": detections[det_idx]["bbox"],
                    "confidence": detections[det_idx]["confidence"],
                    "is_new": True,
                })

        return results

    def _associate(self, det_bboxes, track_bboxes, det_features, track_ids):
        if not track_bboxes or not det_bboxes:
            return [], list(range(len(det_bboxes))), list(range(len(track_bboxes)))

        n_det = len(det_bboxes)
        n_trk = len(track_bboxes)
        cost = np.zeros((n_det, n_trk))

        for i, db in enumerate(det_bboxes):
            for j, tb in enumerate(track_bboxes):
                iou_dist = 1.0 - _iou(db, tb)
                cost_val = iou_dist

                # Use appearance feature if available
                df = det_features[i]
                tid = track_ids[j]
                tf = self._tracks[tid].get_mean_feature()
                if df is not None and tf is not None:
                    cos_dist = _cosine_distance(df, tf)
                    cost_val = 0.4 * iou_dist + 0.6 * cos_dist

                cost[i, j] = cost_val

        row_ind, col_ind = linear_sum_assignment(cost)
        matched, unmatched_dets, unmatched_trks = [], [], []

        matched_det_set = set()
        matched_trk_set = set()

        for r, c in zip(row_ind, col_ind):
            iou_val = 1.0 - (1.0 - _iou(det_bboxes[r], track_bboxes[c]))
            if cost[r, c] > self.max_iou_distance:
                continue
            matched.append((r, track_ids[c]))
            matched_det_set.add(r)
            matched_trk_set.add(c)

        for i in range(n_det):
            if i not in matched_det_set:
                unmatched_dets.append(i)

        for j in range(n_trk):
            if j not in matched_trk_set:
                unmatched_trks.append(j)

        return matched, unmatched_dets, unmatched_trks

    def _prune_tracks(self):
        to_delete = [tid for tid, t in self._tracks.items()
                     if t.time_since_update > self.max_age]
        for tid in to_delete:
            del self._tracks[tid]

    def get_track_count(self) -> int:
        return len(self._tracks)


class TrackManager:
    """Manages per-camera DeepSORT trackers and deduplication state."""

    def __init__(self):
        self._trackers: Dict[str, DeepSortTracker] = {}
        self._logged_tracks: Dict[str, set] = {}  # camera_id -> set of track_ids logged

    def get_tracker(self, camera_id: str, config: Dict = None) -> DeepSortTracker:
        if camera_id not in self._trackers:
            cfg = config or {}
            self._trackers[camera_id] = DeepSortTracker(
                max_age=cfg.get("max_age", 30),
                min_hits=cfg.get("min_hits", 3),
                max_iou_distance=cfg.get("max_iou_distance", 0.7),
                max_cosine_distance=cfg.get("max_cosine_distance", 0.4),
            )
            self._logged_tracks[camera_id] = set()
        return self._trackers[camera_id]

    def is_track_logged(self, camera_id: str, track_id: int) -> bool:
        return track_id in self._logged_tracks.get(camera_id, set())

    def mark_track_logged(self, camera_id: str, track_id: int) -> None:
        if camera_id not in self._logged_tracks:
            self._logged_tracks[camera_id] = set()
        self._logged_tracks[camera_id].add(track_id)

    def remove_camera(self, camera_id: str) -> None:
        self._trackers.pop(camera_id, None)
        self._logged_tracks.pop(camera_id, None)
