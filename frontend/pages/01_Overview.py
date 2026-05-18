import streamlit as st
import time
from frontend.utils.api_client import client
from frontend.utils.formatters import format_datetime, format_confidence, severity_icon

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")
st.title("📊 System Overview")

auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
if auto_refresh:
    st.sidebar.info("Dashboard refreshes every 5 seconds")

# ── Metrics row ───────────────────────────────────────────────────────────────
health = client.get_health()
stats = client.get_system_stats()
unknown_stats = client.get_unknown_stats()

col1, col2, col3, col4, col5 = st.columns(5)
sys = stats.get("system", {})
col1.metric("CPU", f"{sys.get('cpu_percent', 0):.0f}%")
col2.metric("Memory", f"{sys.get('memory_percent', 0):.0f}%")
col3.metric("Known Persons", stats.get("known_persons", 0))
col4.metric("Unknown (Total)", unknown_stats.get("total", 0))
col5.metric("Unknown (Today)", unknown_stats.get("today", 0))

st.divider()

# ── Camera status ─────────────────────────────────────────────────────────────
st.subheader("Camera Status")
cameras = client.get_cameras()

if cameras:
    cols = st.columns(min(len(cameras), 4))
    for idx, cam in enumerate(cameras):
        col = cols[idx % len(cols)]
        rt = cam.get("runtime_status", {})
        connected = rt.get("connected", False)
        fps = rt.get("fps", 0)
        with col:
            status_icon = "🟢" if connected else "🔴"
            st.markdown(f"""
            <div class='camera-card {"camera-online" if connected else "camera-offline"}'>
                <b>{status_icon} {cam.get("name") or cam["camera_id"]}</b><br>
                <small>{cam.get("location", "—")}</small><br>
                FPS: {fps:.1f} | Reconnects: {rt.get("reconnect_count", 0)}
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("No cameras configured. Add cameras in Configuration.")

st.divider()

# ── Recent alerts ─────────────────────────────────────────────────────────────
st.subheader("Recent Events")
logs = client.get_logs(limit=20)
if logs:
    for log in logs[:10]:
        icon = severity_icon(log.get("severity", "info"))
        ts = format_datetime(log.get("created_at"))
        cam = f" [{log['camera_id']}]" if log.get("camera_id") else ""
        st.write(f"{icon} `{ts}`{cam} — {log.get('description', '')}")
else:
    st.info("No recent events")

st.divider()

# ── Trending unknowns ─────────────────────────────────────────────────────────
st.subheader("Trending Unknown Persons")
trending = unknown_stats.get("trending", [])
if trending:
    for u in trending[:5]:
        ts = format_datetime(u.get("last_seen"))
        st.write(f"• Track `{u.get('track_id')}` | Camera: `{u.get('camera_id')}` | "
                 f"Appearances: **{u.get('appearance_count', 1)}** | Last seen: {ts}")
else:
    st.info("No trending unknowns")

if auto_refresh:
    time.sleep(5)
    st.rerun()
