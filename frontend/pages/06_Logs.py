import streamlit as st
import pandas as pd
from frontend.utils.api_client import client
from frontend.utils.formatters import format_datetime, severity_icon

st.set_page_config(page_title="Logs", page_icon="📜", layout="wide")
st.title("📜 System Logs & Events")

tab_sys, tab_det = st.tabs(["📋 System Logs", "🔍 Detection Events"])

# ── SYSTEM LOGS ───────────────────────────────────────────────────────────────
with tab_sys:
    col1, col2 = st.columns(2)
    severity_filter = col1.selectbox("Severity", ["All", "info", "warning", "error"])
    log_limit = col2.slider("Limit", 20, 500, 100)

    logs = client.get_logs(
        severity=severity_filter if severity_filter != "All" else None,
        limit=log_limit
    )

    if logs:
        df = pd.DataFrame(logs)
        df["created_at"] = pd.to_datetime(df["created_at"]).astype(str)
        df = df[["created_at", "severity", "event_type", "camera_id", "description"]]
        df.columns = ["Timestamp", "Severity", "Event Type", "Camera", "Description"]

        def color_severity(row):
            colors = {"error": "background-color: #450a0a", "warning": "background-color: #451a03"}
            return [colors.get(row["Severity"], "") for _ in row]

        st.dataframe(df.style.apply(color_severity, axis=1), use_container_width=True, height=500)

        csv = df.to_csv(index=False)
        st.download_button("Download Logs (CSV)", csv, "system_logs.csv", "text/csv")
    else:
        st.info("No logs found.")

# ── DETECTION EVENTS ──────────────────────────────────────────────────────────
with tab_det:
    cameras = client.get_cameras()
    camera_opts = ["All"] + [c["camera_id"] for c in cameras]
    sel_cam = st.selectbox("Filter by camera", camera_opts)
    det_limit = st.slider("Limit", 20, 1000, 200, key="det_limit")

    events = client.get_timeline(
        camera_id=sel_cam if sel_cam != "All" else None,
        limit=det_limit
    )

    if events:
        df = pd.DataFrame(events)
        keep_cols = [c for c in ["detected_at", "camera_id", "track_id",
                                  "person_name", "confidence"] if c in df.columns]
        df = df[keep_cols]
        df.columns = [c.replace("_", " ").title() for c in keep_cols]

        st.dataframe(df, use_container_width=True, height=500)

        csv = df.to_csv(index=False)
        st.download_button("Download Events (CSV)", csv, "detection_events.csv", "text/csv")
    else:
        st.info("No detection events found.")

    st.divider()
    st.subheader("Person Tracking History")
    persons = client.get_known_persons()
    if persons:
        sel_person = st.selectbox("Select person", [p["name"] for p in persons])
        sel_pid = next(p["id"] for p in persons if p["name"] == sel_person)
        p_events = client._get(f"/api/detections/person/{sel_pid}")
        if p_events and p_events.get("events"):
            df_p = pd.DataFrame(p_events["events"])
            keep = [c for c in ["detected_at", "camera_id", "confidence"] if c in df_p.columns]
            st.dataframe(df_p[keep], use_container_width=True)
