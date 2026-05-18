import httpx
import os
from typing import Optional, Dict, Any, List

API_BASE = os.getenv("API_URL", "http://localhost:8000")


class APIClient:
    def __init__(self, base_url: str = API_BASE, token: str = None):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}

    def _get(self, path: str, params: dict = None) -> Optional[Dict]:
        try:
            resp = httpx.get(f"{self.base_url}{path}", params=params,
                             headers=self._headers, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def _post(self, path: str, data: dict = None, files=None) -> Optional[Dict]:
        try:
            if files:
                resp = httpx.post(f"{self.base_url}{path}", files=files,
                                  headers=self._headers, timeout=30)
            else:
                resp = httpx.post(f"{self.base_url}{path}", json=data,
                                  headers=self._headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def _put(self, path: str, data: dict) -> Optional[Dict]:
        try:
            resp = httpx.put(f"{self.base_url}{path}", json=data,
                             headers=self._headers, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def _delete(self, path: str) -> Optional[Dict]:
        try:
            resp = httpx.delete(f"{self.base_url}{path}",
                                headers=self._headers, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    # ── Cameras ──────────────────────────────────────────────────────────────
    def get_cameras(self) -> List[Dict]:
        r = self._get("/api/cameras")
        return r.get("cameras", []) if r else []

    def add_camera(self, data: dict) -> Dict:
        return self._post("/api/cameras", data) or {}

    def update_camera(self, camera_id: str, data: dict) -> Dict:
        return self._put(f"/api/cameras/{camera_id}", data) or {}

    def delete_camera(self, camera_id: str) -> Dict:
        return self._delete(f"/api/cameras/{camera_id}") or {}

    def test_camera(self, camera_id: str) -> Dict:
        return self._post(f"/api/cameras/{camera_id}/test") or {}

    def get_camera_stats(self, camera_id: str) -> Dict:
        return self._get(f"/api/cameras/{camera_id}/stats") or {}

    # ── Known Persons ─────────────────────────────────────────────────────────
    def get_known_persons(self) -> List[Dict]:
        r = self._get("/api/known-persons")
        return r.get("persons", []) if r else []

    def add_known_person(self, data: dict) -> Dict:
        return self._post("/api/known-persons", data) or {}

    def update_known_person(self, person_id: int, data: dict) -> Dict:
        return self._put(f"/api/known-persons/{person_id}", data) or {}

    def delete_known_person(self, person_id: int) -> Dict:
        return self._delete(f"/api/known-persons/{person_id}") or {}

    def add_person_image(self, person_id: int, file_bytes: bytes, filename: str) -> Dict:
        files = {"file": (filename, file_bytes, "image/jpeg")}
        return self._post(f"/api/known-persons/{person_id}/image", files=files) or {}

    def get_person_images(self, person_id: int) -> List[Dict]:
        r = self._get(f"/api/known-persons/{person_id}/images")
        return r.get("images", []) if r else []

    # ── Unknown Persons ───────────────────────────────────────────────────────
    def get_unknown_persons(self, **filters) -> List[Dict]:
        r = self._get("/api/unknown-persons", params={k: v for k, v in filters.items() if v is not None})
        return r.get("unknowns", []) if r else []

    def get_unknown_stats(self) -> Dict:
        r = self._get("/api/unknown-persons/stats")
        return r.get("stats", {}) if r else {}

    def delete_unknown(self, unknown_id: int) -> Dict:
        return self._delete(f"/api/unknown-persons/{unknown_id}") or {}

    def mark_unknown_as_known(self, unknown_id: int, person_id: int) -> Dict:
        return self._post(f"/api/unknown-persons/{unknown_id}/mark-known", {"person_id": person_id}) or {}

    # ── Detections ────────────────────────────────────────────────────────────
    def get_timeline(self, camera_id: str = None, limit: int = 100) -> List[Dict]:
        r = self._get("/api/detections/timeline",
                      params={"camera_id": camera_id, "limit": limit})
        return r.get("events", []) if r else []

    def test_recognition(self, file_bytes: bytes, filename: str) -> Dict:
        files = {"file": (filename, file_bytes, "image/jpeg")}
        return self._post("/api/recognition/test", files=files) or {}

    # ── System ────────────────────────────────────────────────────────────────
    def get_health(self) -> Dict:
        return self._get("/api/system/health") or {}

    def get_system_stats(self) -> Dict:
        return self._get("/api/system/stats") or {}

    def get_logs(self, severity: str = None, limit: int = 100) -> List[Dict]:
        r = self._get("/api/system/logs", params={"severity": severity, "limit": limit})
        return r.get("logs", []) if r else []

    def restart_system(self) -> Dict:
        return self._post("/api/system/restart") or {}

    def get_config(self) -> Dict:
        r = self._get("/api/system/config")
        return r.get("config", {}) if r else {}


# Default client instance
client = APIClient()
