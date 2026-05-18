from datetime import datetime
from typing import Optional


def format_datetime(dt) -> str:
    if dt is None:
        return "—"
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except Exception:
            return dt
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_confidence(conf: Optional[float]) -> str:
    if conf is None:
        return "—"
    return f"{conf:.1%}"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds/60)}m {int(seconds%60)}s"
    return f"{int(seconds/3600)}h {int((seconds%3600)/60)}m"


def truncate(text: str, max_len: int = 50) -> str:
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def severity_icon(severity: str) -> str:
    return {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}.get(severity, "•")
