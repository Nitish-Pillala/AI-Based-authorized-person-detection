import time
from typing import Any, Optional, Callable


class TTLCache:
    def __init__(self, ttl_seconds: int = 30):
        self._store = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry and time.time() - entry["ts"] < self._ttl:
            return entry["val"]
        return None

    def set(self, key: str, value: Any) -> None:
        self._store[key] = {"val": value, "ts": time.time()}

    def get_or_fetch(self, key: str, fetch_fn: Callable) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = fetch_fn()
        self.set(key, value)
        return value

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()


dashboard_cache = TTLCache(ttl_seconds=5)
