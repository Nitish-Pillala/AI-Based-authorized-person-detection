import streamlit as st
import base64
from frontend.utils.api_client import client

st.set_page_config(page_title="Configuration", page_icon="⚙️", layout="wide")
st.title("⚙️ Configuration")

tab_cameras, tab_persons, tab_system = st.tabs(["📹 Cameras", "👤 Known Persons", "🔧 System"])

# ── CAMERAS TAB ───────────────────────────────────────────────────────────────
with tab_cameras:
    st.subheader("Camera Management")

    cameras = client.get_cameras()

    if cameras:
        st.write(f"**{len(cameras)} camera(s) configured:**")
        for cam in cameras:
            rt = cam.get("runtime_status", {})
            connected = rt.get("connected", False)
            with st.expander(f"{'🟢' if connected else '🔴'} {cam.get('name') or cam['camera_id']} ({cam['camera_id']})"):
                c1, c2 = st.columns(2)
                c1.text_input("Name", value=cam.get("name", ""), key=f"name_{cam['camera_id']}", disabled=True)
                c1.text_input("Location", value=cam.get("location", ""), key=f"loc_{cam['camera_id']}", disabled=True)
                c2.text_input("RTSP URL", value=cam.get("rtsp_url", ""), key=f"url_{cam['camera_id']}", disabled=True)
                c2.write(f"FPS: {cam.get('processing_fps')} | Confidence: {cam.get('confidence_threshold')}")

                btn1, btn2, btn3 = st.columns(3)
                with btn1:
                    if st.button("Test Connection", key=f"test_{cam['camera_id']}"):
                        result = client.test_camera(cam['camera_id'])
                        if result.get("connected"):
                            st.success("Connection OK!")
                        else:
                            st.error("Connection failed")
                with btn3:
                    if st.button("Remove Camera", key=f"del_{cam['camera_id']}", type="secondary"):
                        client.delete_camera(cam['camera_id'])
                        st.success("Camera removed")
                        st.rerun()
    else:
        st.info("No cameras configured yet.")

    st.divider()
    st.subheader("Add New Camera")

    with st.form("add_camera_form"):
        c1, c2 = st.columns(2)
        cam_id = c1.text_input("Camera ID *", placeholder="entrance_01")
        cam_name = c1.text_input("Display Name", placeholder="Main Entrance")
        rtsp_url = c2.text_input("RTSP URL *", placeholder="rtsp://user:pass@192.168.1.100:554/stream")
        location = c2.text_input("Location", placeholder="Building A - Ground Floor")

        c3, c4 = st.columns(2)
        proc_fps = c3.slider("Processing FPS", 1, 30, 15)
        confidence = c4.slider("Confidence Threshold", 0.1, 1.0, 0.75)
        enabled = st.checkbox("Enable immediately", value=True)

        st.markdown("**DeepSORT Tracking Parameters**")
        t1, t2, t3, t4 = st.columns(4)
        max_age = t1.number_input("Max Age", 1, 100, 30)
        min_hits = t2.number_input("Min Hits", 1, 20, 3)
        max_iou = t3.number_input("Max IoU Dist", 0.0, 1.0, 0.7, step=0.05)
        max_cos = t4.number_input("Max Cosine Dist", 0.0, 1.0, 0.4, step=0.05)

        submitted = st.form_submit_button("Add Camera", use_container_width=True)
        if submitted:
            if not cam_id or not rtsp_url:
                st.error("Camera ID and RTSP URL are required")
            else:
                result = client.add_camera({
                    "camera_id": cam_id, "name": cam_name, "rtsp_url": rtsp_url,
                    "location": location, "enabled": enabled,
                    "processing_fps": proc_fps, "confidence_threshold": confidence,
                    "track_config": {
                        "max_age": max_age, "min_hits": min_hits,
                        "max_iou_distance": max_iou, "max_cosine_distance": max_cos,
                    }
                })
                if "error" not in result:
                    st.success(f"Camera '{cam_id}' added successfully!")
                    st.rerun()
                else:
                    st.error(result.get("error"))


# ── KNOWN PERSONS TAB ─────────────────────────────────────────────────────────
with tab_persons:
    st.subheader("Known Persons Database")

    persons = client.get_known_persons()

    if persons:
        for p in persons:
            with st.expander(f"👤 {p['name']} — {p.get('employee_id', 'No ID')} ({p.get('department', '')})"):
                c1, c2 = st.columns(2)
                c1.write(f"**Detections:** {p.get('total_detections', 0)}")
                c2.write(f"**Added:** {p.get('created_at', '—')}")
                if p.get("description"):
                    st.write(f"**Note:** {p['description']}")

                images = client.get_person_images(p["id"])
                if images:
                    img_cols = st.columns(min(len(images), 5))
                    for ic, img in zip(img_cols, images[:5]):
                        import os
                        if os.path.exists(img.get("image_path", "")):
                            ic.image(img["image_path"], width=80)

                col_a, col_b = st.columns(2)
                with col_a:
                    up_img = st.file_uploader("Add face image", type=["jpg", "jpeg", "png"],
                                              key=f"img_{p['id']}")
                    if up_img:
                        result = client.add_person_image(p["id"], up_img.read(), up_img.name)
                        if "error" not in result:
                            st.success("Image added!")
                with col_b:
                    if st.button("Delete Person", key=f"delp_{p['id']}", type="secondary"):
                        client.delete_known_person(p["id"])
                        st.success("Person deleted")
                        st.rerun()
    else:
        st.info("No known persons registered.")

    st.divider()
    st.subheader("Register New Person")

    with st.form("add_person_form"):
        c1, c2 = st.columns(2)
        p_name = c1.text_input("Full Name *")
        p_emp_id = c1.text_input("Employee ID")
        p_dept = c2.text_input("Department")
        p_desc = c2.text_input("Notes")
        face_file = st.file_uploader("Face Image *", type=["jpg", "jpeg", "png"])

        if st.form_submit_button("Register Person", use_container_width=True):
            if not p_name:
                st.error("Name is required")
            elif not face_file:
                st.error("Face image is required")
            else:
                import base64
                img_b64 = base64.b64encode(face_file.read()).decode()
                result = client.add_known_person({
                    "name": p_name, "employee_id": p_emp_id or None,
                    "department": p_dept or None, "description": p_desc or None,
                    "face_image": img_b64,
                })
                if "error" not in result:
                    st.success(f"Person '{p_name}' registered! Embedding: {result.get('embedding_generated')}")
                    st.rerun()
                else:
                    st.error(result.get("error"))


# ── SYSTEM TAB ────────────────────────────────────────────────────────────────
with tab_system:
    st.subheader("System Configuration")

    config = client.get_config()
    c1, c2 = st.columns(2)
    c1.metric("Face Match Threshold", config.get("face_match_threshold", 0.75))
    c1.metric("YOLO Confidence", config.get("yolo_confidence", 0.5))
    c2.metric("Processing FPS", config.get("default_processing_fps", 15))
    c2.metric("DeepSORT Max Age", config.get("deepsort_max_age", 30))

    st.divider()
    st.subheader("Test Face Recognition")

    test_file = st.file_uploader("Upload a face image to test", type=["jpg", "jpeg", "png"])
    if test_file and st.button("Test Recognition"):
        result = client.test_recognition(test_file.read(), test_file.name)
        if "error" not in result:
            col1, col2 = st.columns(2)
            col1.metric("Person", result.get("person_name", "Unknown"))
            col2.metric("Confidence", f"{result.get('confidence', 0):.1%}")
            if result.get("is_known"):
                st.success("Known person detected!")
            else:
                st.warning("Unknown person")
        else:
            st.error(result.get("error"))

    st.divider()
    if st.button("Restart System", type="secondary"):
        with st.spinner("Restarting..."):
            result = client.restart_system()
            if "error" not in result:
                st.success("System restarted!")
            else:
                st.error(result.get("error"))
