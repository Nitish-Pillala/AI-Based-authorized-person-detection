import streamlit as st
import time
import base64
import numpy as np
import cv2
from frontend.utils.api_client import client

st.set_page_config(page_title="Live Cameras", page_icon="📹", layout="wide")
st.title("📹 Live Camera Feeds")

cameras = client.get_cameras()
if not cameras:
    st.warning("No cameras configured. Add cameras in ⚙️ Configuration.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
camera_ids = [c["camera_id"] for c in cameras]
selected   = st.sidebar.multiselect("Cameras to display", camera_ids,
                                     default=camera_ids[:4])
fps_target  = st.sidebar.slider("Refresh rate (FPS)", 1, 15, 5)
show_stats  = st.sidebar.checkbox("Show stats below feed", value=True)

st.sidebar.divider()
st.sidebar.subheader("Adjust Camera")
ctrl_id  = st.sidebar.selectbox("Camera", camera_ids)
ctrl_cam = next((c for c in cameras if c["camera_id"] == ctrl_id), {})
new_fps  = st.sidebar.slider("Processing FPS", 1, 30,
                               int(ctrl_cam.get("processing_fps", 15)))
new_conf = st.sidebar.slider("Confidence", 0.1, 1.0,
                               float(ctrl_cam.get("confidence_threshold", 0.75)),
                               step=0.05)
if st.sidebar.button("Apply", use_container_width=True):
    r = client.update_camera(ctrl_id,
                             {"processing_fps": new_fps,
                              "confidence_threshold": new_conf})
    st.sidebar.success("Applied!" if "error" not in r else r.get("error"))

if not selected:
    st.info("Select at least one camera from the sidebar.")
    st.stop()

# ── Build grid of image + stat placeholders ───────────────────────────────────
COLS = min(len(selected), 2)
img_placeholders  = {}
stat_placeholders = {}

rows = [selected[i:i+COLS] for i in range(0, len(selected), COLS)]
for row_cams in rows:
    cols = st.columns(COLS)
    for col, cam_id in zip(cols, row_cams):
        cam_info = next((c for c in cameras if c["camera_id"] == cam_id), {})
        with col:
            st.markdown(f"**📷 {cam_info.get('name') or cam_id}** "
                        f"— `{cam_info.get('location','')}`")
            img_placeholders[cam_id]  = st.empty()   # ← live image here
            stat_placeholders[cam_id] = st.empty()   # ← stats below

# ── Helper: decode base64 → numpy → display ───────────────────────────────────
def show_frame(img_ph, stat_ph, cam_id):
    resp = client._get(f"/api/cameras/{cam_id}/frame")

    # ── IMAGE ─────────────────────────────────────────────────────────────────
    if resp and resp.get("frame_b64"):
        try:
            raw   = base64.b64decode(resp["frame_b64"])
            arr   = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                # Convert BGR → RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_ph.image(frame_rgb, use_column_width=True)
            else:
                img_ph.warning("⚠ Could not decode frame")
        except Exception as e:
            img_ph.error(f"Frame error: {e}")
    else:
        # Show camera placeholder while waiting for first frame
        img_ph.markdown(
            "<div style='background:#0d1117;height:240px;border:1px dashed #333;"
            "border-radius:8px;display:flex;align-items:center;"
            "justify-content:center;font-size:3rem;'>📷</div>",
            unsafe_allow_html=True,
        )

    # ── STATS ─────────────────────────────────────────────────────────────────
    if show_stats:
        rt = client.get_camera_stats(cam_id).get("stats", {})
        connected = rt.get("connected", False)
        dets = resp.get("detections", []) if resp else []
        known   = sum(1 for d in dets if d.get("is_known"))
        unknown = sum(1 for d in dets if not d.get("is_known"))

        status_dot = "🟢" if connected else "🔴"
        stat_ph.markdown(
            f"{status_dot} **FPS:** {rt.get('fps', 0):.1f} &nbsp;|&nbsp; "
            f"**Frames:** {rt.get('frames_processed', 0):,} &nbsp;|&nbsp; "
            f"**Known:** {known} &nbsp;|&nbsp; **Unknown:** {unknown}"
        )

# ── Live loop ─────────────────────────────────────────────────────────────────
interval = 1.0 / fps_target

for _ in range(300):          # ~300 ticks then st.rerun() clears memory
    for cam_id in selected:
        show_frame(
            img_placeholders[cam_id],
            stat_placeholders[cam_id],
            cam_id,
        )
    time.sleep(interval)

st.rerun()                     # restart loop to prevent memory accumulation
