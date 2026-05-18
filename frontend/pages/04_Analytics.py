import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from frontend.utils.api_client import client

st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")
st.title("📈 Analytics & Reporting")

# Date range filter
col1, col2 = st.columns(2)
with col1:
    date_from = st.date_input("From", value=datetime.now().date() - timedelta(days=7))
with col2:
    date_to = st.date_input("To", value=datetime.now().date())

cameras = client.get_cameras()
camera_options = ["All"] + [c["camera_id"] for c in cameras]
sel_camera = st.selectbox("Filter by camera", camera_options)
filter_camera = sel_camera if sel_camera != "All" else None

st.divider()

# ── Detection timeline chart ───────────────────────────────────────────────────
st.subheader("Detection Timeline")
events = client.get_timeline(camera_id=filter_camera, limit=500)

if events:
    df = pd.DataFrame(events)
    if "detected_at" in df.columns:
        df["detected_at"] = pd.to_datetime(df["detected_at"])
        df["hour"] = df["detected_at"].dt.floor("H")
        hourly = df.groupby(["hour"]).size().reset_index(name="count")
        fig = px.bar(hourly, x="hour", y="count", title="Detections per Hour",
                     color_discrete_sequence=["#00d4ff"])
        fig.update_layout(template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827")
        st.plotly_chart(fig, use_container_width=True)

        # Known vs Unknown
        if "person_name" in df.columns:
            df["type"] = df["person_name"].apply(lambda x: "Known" if x and x != "Unknown" else "Unknown")
            type_counts = df["type"].value_counts()
            fig2 = px.pie(values=type_counts.values, names=type_counts.index,
                          title="Known vs Unknown Detections",
                          color_discrete_sequence=["#10b981", "#ef4444"])
            fig2.update_layout(template="plotly_dark", paper_bgcolor="#111827")
            st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No detection events found for selected period.")

st.divider()

# ── Unknown persons trend ─────────────────────────────────────────────────────
st.subheader("Unknown Persons — Most Frequent")
unknowns = client.get_unknown_persons(sort_by="appearance_count", order="desc", limit=20)

if unknowns:
    df_u = pd.DataFrame(unknowns)
    if "appearance_count" in df_u.columns:
        df_u["label"] = df_u["track_id"].astype(str) + " (" + df_u["camera_id"].astype(str) + ")"
        fig3 = px.bar(df_u.head(15), x="label", y="appearance_count",
                      title="Top Unknown Persons by Appearance Count",
                      color_discrete_sequence=["#f59e0b"])
        fig3.update_layout(template="plotly_dark", plot_bgcolor="#111827", paper_bgcolor="#111827",
                           xaxis_tickangle=-45)
        st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── System stats ──────────────────────────────────────────────────────────────
st.subheader("System Statistics")
stats = client.get_system_stats()
sys_info = stats.get("system", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("CPU Usage", f"{sys_info.get('cpu_percent', 0):.0f}%")
c2.metric("Memory Used", f"{sys_info.get('memory_used_gb', 0):.1f} GB")
c3.metric("Memory %", f"{sys_info.get('memory_percent', 0):.0f}%")
c4.metric("Disk %", f"{sys_info.get('disk_percent', 0):.0f}%")

st.divider()

# ── Export ────────────────────────────────────────────────────────────────────
st.subheader("Export Data")
col_e1, col_e2 = st.columns(2)

with col_e1:
    if st.button("Export Unknown Persons (CSV)"):
        data = client.get_unknown_persons(limit=500)
        if data:
            df_exp = pd.DataFrame(data).drop(columns=["face_embedding"], errors="ignore")
            csv = df_exp.to_csv(index=False)
            st.download_button("Download CSV", csv, "unknown_persons.csv", "text/csv")

with col_e2:
    if st.button("Export Detection Events (CSV)"):
        data = client.get_timeline(limit=1000)
        if data:
            df_exp = pd.DataFrame(data)
            csv = df_exp.to_csv(index=False)
            st.download_button("Download CSV", csv, "detections.csv", "text/csv")
