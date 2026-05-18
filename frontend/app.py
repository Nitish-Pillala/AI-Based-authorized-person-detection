import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="CCTV Surveillance System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = Path(__file__).parent / "styles" / "custom.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Sidebar branding
st.sidebar.markdown("""
<div style='text-align:center; padding: 20px 0;'>
    <h2 style='color:#00d4ff; margin:0;'>🎥 CCTV System</h2>
    <p style='color:#a0b4c8; font-size:0.8rem; margin:4px 0;'>Surveillance Dashboard</p>
</div>
""", unsafe_allow_html=True)
st.sidebar.divider()

# Connection status check
from frontend.utils.api_client import client
health = client.get_health()
is_healthy = "error" not in str(health.get("status", "")) and health.get("status") in ("healthy", "degraded")

if is_healthy:
    st.sidebar.success(f"API Connected — {health.get('active_cameras', 0)} cameras active")
else:
    st.sidebar.error("API Disconnected — Start backend first")

# Main landing page
st.title("🎥 Real-Time CCTV Surveillance System")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

stats = client.get_system_stats()
unknown_stats = client.get_unknown_stats()

with col1:
    st.metric("Known Persons", stats.get("known_persons", 0), help="Registered persons in database")
with col2:
    st.metric("Unknown (Total)", unknown_stats.get("total", 0), help="All detected unknown persons")
with col3:
    st.metric("Unknown (Today)", unknown_stats.get("today", 0), help="New unknowns detected today")
with col4:
    cameras = client.get_cameras()
    active = sum(1 for c in cameras if c.get("runtime_status", {}).get("connected"))
    st.metric("Active Cameras", f"{active}/{len(cameras)}", help="Connected cameras")

st.markdown("---")
st.markdown("""
### Navigate Using Sidebar Pages

| Page | Description |
|------|-------------|
| 📊 **Overview** | Real-time metrics & system health |
| 👥 **Unknown Gallery** | Browse & manage unknown persons |
| 📹 **Live Cameras** | Live RTSP camera feeds |
| 📈 **Analytics** | Detection trends & reports |
| ⚙️ **Configuration** | System settings & camera management |
| 📜 **Logs** | System & detection event logs |
""")

sys_stats = stats.get("system", {})
if sys_stats:
    st.markdown("---")
    st.subheader("System Health")
    c1, c2, c3 = st.columns(3)
    c1.metric("CPU", f"{sys_stats.get('cpu_percent', 0):.0f}%")
    c2.metric("Memory", f"{sys_stats.get('memory_percent', 0):.0f}%")
    c3.metric("Disk", f"{sys_stats.get('disk_percent', 0):.0f}%")
