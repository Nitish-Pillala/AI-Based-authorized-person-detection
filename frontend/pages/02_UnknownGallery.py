import streamlit as st
import os
from frontend.utils.api_client import client
from frontend.utils.formatters import format_datetime, format_confidence

st.set_page_config(page_title="Unknown Gallery", page_icon="👥", layout="wide")
st.title("👥 Unknown Persons Gallery")

# ── Filters sidebar ───────────────────────────────────────────────────────────
st.sidebar.subheader("Filters")
cameras_list = client.get_cameras()
camera_options = ["All"] + [c["camera_id"] for c in cameras_list]
sel_camera = st.sidebar.selectbox("Camera", camera_options)
min_appearances = st.sidebar.number_input("Min Appearances", min_value=1, value=1)
show_marked = st.sidebar.selectbox("Status", ["All", "Unknown", "Identified"])
sort_by = st.sidebar.selectbox("Sort By", ["last_seen", "first_seen", "appearance_count", "highest_confidence"])
order = st.sidebar.selectbox("Order", ["desc", "asc"])
limit = st.sidebar.slider("Results", 10, 200, 50)

# Build filters
filters = {
    "sort_by": sort_by, "order": order, "limit": limit,
    "min_appearances": min_appearances if min_appearances > 1 else None,
    "camera_id": sel_camera if sel_camera != "All" else None,
}
if show_marked == "Unknown":
    filters["marked_as_known"] = False
elif show_marked == "Identified":
    filters["marked_as_known"] = True

unknowns = client.get_unknown_persons(**filters)

# ── Stats banner ──────────────────────────────────────────────────────────────
stats = client.get_unknown_stats()
c1, c2, c3 = st.columns(3)
c1.metric("Total Unknown", stats.get("total", 0))
c2.metric("Today", stats.get("today", 0))
c3.metric("Showing", len(unknowns))
st.divider()

# ── Gallery grid ──────────────────────────────────────────────────────────────
if not unknowns:
    st.info("No unknown persons found with current filters.")
    st.stop()

API_BASE = os.getenv("API_URL", "http://localhost:8000")
COLS = 4

known_persons = client.get_known_persons()
person_options = {p["name"]: p["id"] for p in known_persons}

rows = [unknowns[i:i+COLS] for i in range(0, len(unknowns), COLS)]

for row in rows:
    cols = st.columns(COLS)
    for col, person in zip(cols, row):
        with col:
            img_path = person.get("best_image_path", "")
            if img_path and os.path.exists(img_path):
                st.image(img_path, use_column_width=True)
            else:
                st.markdown("🚫 No image")

            status = "✅ Identified" if person.get("marked_as_known") else "❓ Unknown"
            st.markdown(f"**{status}**")
            st.caption(f"Track: `{person.get('track_id')}`")
            st.caption(f"Camera: `{person.get('camera_id')}`")
            st.caption(f"Appearances: **{person.get('appearance_count', 1)}**")
            st.caption(f"First: {format_datetime(person.get('first_seen'))}")
            st.caption(f"Last: {format_datetime(person.get('last_seen'))}")
            st.caption(f"Confidence: {format_confidence(person.get('highest_confidence'))}")

            uid = person["id"]

            with st.expander("Actions"):
                if not person.get("marked_as_known") and known_persons:
                    sel_name = st.selectbox("Identify as", list(person_options.keys()),
                                            key=f"sel_{uid}")
                    if st.button("Mark as Known", key=f"mk_{uid}"):
                        result = client.mark_unknown_as_known(uid, person_options[sel_name])
                        if "error" not in result:
                            st.success("Marked as known!")
                            st.rerun()
                        else:
                            st.error(result["error"])

                if st.button("Delete", key=f"del_{uid}", type="secondary"):
                    result = client.delete_unknown(uid)
                    if "error" not in result:
                        st.success("Deleted")
                        st.rerun()
