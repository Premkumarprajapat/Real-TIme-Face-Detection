"""
Streamlit UI — Advanced Face Recognition System.
Two modes: User (recognition) and Admin (management).
"""

import streamlit as st
import cv2
import numpy as np
import face_recognition as fr
import time
import os
import glob
import re
from datetime import datetime

import config
from face_encoder import encode_known_faces, load_encodings
from liveness import LivenessDetector
from depth_motion_check import DepthMotionChecker
from database import AttendanceDB
from unknown_handler import UnknownFaceHandler
from security import SecurityManager
from tamper_proof_log import TamperProofLog
from adaptive_learning import AdaptiveLearner
from preprocessing import enhance_frame, get_brightness_status

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
        text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: #fff; font-size: 2rem; font-weight: 700; margin: 0; }
    .main-header p { color: #a5b4fc; font-size: 0.95rem; margin: 0.3rem 0 0 0; }

    .status-card {
        background: linear-gradient(135deg, #1e1e2e, #2d2d44);
        border: 1px solid #3d3d5c; border-radius: 12px;
        padding: 1.2rem; margin: 0.5rem 0; color: #e2e8f0;
    }
    .status-card h3 {
        color: #a5b4fc; margin: 0 0 0.5rem 0; font-size: 0.85rem;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .status-card .value { font-size: 1.5rem; font-weight: 700; color: #fff; }

    .access-granted {
        background: linear-gradient(135deg, #065f46, #047857);
        border: 2px solid #10b981; border-radius: 16px; padding: 2rem;
        text-align: center; color: white;
        animation: pulse 1.5s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 20px rgba(16,185,129,0.3); }
        50% { box-shadow: 0 0 40px rgba(16,185,129,0.6); }
    }

    .access-denied {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 2px solid #ef4444; border-radius: 16px; padding: 2rem;
        text-align: center; color: white;
    }

    .info-badge {
        display: inline-block; background: #3b82f6; color: white;
        padding: 0.2rem 0.6rem; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600;
    }

    .attendance-table { width: 100%; border-collapse: collapse; }
    .attendance-table th {
        background: #302b63; color: #fff; padding: 10px 15px;
        text-align: left; font-size: 0.85rem;
    }
    .attendance-table td {
        padding: 8px 15px; border-bottom: 1px solid #2d2d44;
        color: #e2e8f0; font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔐 Advanced Face Recognition System</h1>
    <p>Offline · Secure · Adaptive · Tamper-Proof</p>
</div>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "admin_face_feedback" not in st.session_state:
    st.session_state.admin_face_feedback = None

# ─── Cached Resources ───────────────────────────────────────────────────────
@st.cache_resource
def get_data():
    return load_encodings()

@st.cache_resource
def get_db():
    return AttendanceDB()

@st.cache_resource
def get_security():
    return SecurityManager()

@st.cache_resource
def get_log():
    return TamperProofLog()

@st.cache_resource
def get_adaptive():
    return AdaptiveLearner()


def sanitize_person_name(name: str) -> str:
    """Normalize a person name so it can be used safely as a folder name."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name or "")
    return " ".join(cleaned.strip().split())


def get_pending_face_batches() -> list:
    """Return people waiting in known_faces with their pending image counts."""
    pending = []
    if not os.path.exists(config.KNOWN_FACES_DIR):
        return pending

    for person_name in sorted(os.listdir(config.KNOWN_FACES_DIR)):
        person_dir = os.path.join(config.KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        image_count = len(
            glob.glob(os.path.join(person_dir, "*.jpg")) +
            glob.glob(os.path.join(person_dir, "*.jpeg")) +
            glob.glob(os.path.join(person_dir, "*.png")) +
            glob.glob(os.path.join(person_dir, "*.bmp")) +
            glob.glob(os.path.join(person_dir, "*.webp"))
        )

        if image_count:
            pending.append({
                "name": person_name,
                "count": image_count,
                "path": person_dir,
            })

    return pending


def save_uploaded_faces(person_name: str, uploaded_files) -> tuple:
    """Save uploaded face images to known_faces/<PersonName>/."""
    safe_name = sanitize_person_name(person_name)
    if not safe_name:
        raise ValueError("Enter a valid person name.")

    if not uploaded_files:
        raise ValueError("Upload at least one image.")

    person_dir = os.path.join(config.KNOWN_FACES_DIR, safe_name)
    os.makedirs(person_dir, exist_ok=True)

    saved_count = 0
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        extension = os.path.splitext(uploaded_file.name)[1].lower() or ".jpg"
        if extension not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            extension = ".jpg"

        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{index}{extension}"
        file_path = os.path.join(person_dir, filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_count += 1

    return safe_name, saved_count


# ═══════════════════════════════════════════════════════════════════════════════
tab_user, tab_admin = st.tabs(["👤 Recognition", "🛡️ Admin Panel"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1: USER — Recognition
# ═══════════════════════════════════════════════════════════════════════════════
with tab_user:
    st.markdown("### 🎥 Face Recognition & Attendance")
    st.markdown("Click **Start Camera** to begin a recognition session with anti-spoofing verification.")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        start_btn = st.button("▶️ Start Camera", type="primary")
    with col2:
        stop_btn = st.button("⏹️ Stop Camera")
    with col3:
        threshold = st.slider("Threshold", 0.35, 0.60, 0.50, 0.01)
    with col4:
        depth_enabled = st.checkbox("Depth Check", value=config.DEPTH_CHECK_ENABLED)

    # Camera display in centered smaller column
    _, cam_col, _ = st.columns([1, 2, 1])
    with cam_col:
        status_area = st.empty()
        depth_status_area = st.empty()
        frame_display = st.empty()
    result_area = st.empty()

    if stop_btn:
        st.session_state.camera_running = False

    if start_btn:
        st.session_state.camera_running = True

        data = get_data()
        security = get_security()
        log = get_log()
        db = get_db()

        if security.is_locked():
            result_area.markdown("""
            <div class="access-denied">
                <h2>🔒 SYSTEM LOCKED</h2>
                <p>Too many failed attempts. Go to Admin Panel to unlock.</p>
            </div>
            """, unsafe_allow_html=True)
            st.session_state.camera_running = False
        elif not data["encodings"]:
            st.warning("⚠️ No face encodings found. Run `python admin.py --encode` first.")
            st.session_state.camera_running = False
        else:
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

            if not cap.isOpened():
                st.error("❌ Cannot open camera!")
                st.session_state.camera_running = False
            else:
                liveness = LivenessDetector()
                depth_checker = DepthMotionChecker() if depth_enabled else None
                unknown_handler = UnknownFaceHandler()

                attempt_start = time.time()
                blink_count = 0
                name_counter = {}
                last_name = "UNKNOWN"
                last_confidence = 0.0
                last_unknown_capture = time.time()
                attempt_recorded = False
                frame_count = 0
                final_result = None

                # Phases: "recognition", "depth", "done"
                phase = "recognition"
                confirmed_name = None
                confirmed_conf = 0.0

                # Persistent face box state (drawn every frame)
                draw_box = None
                draw_name = "UNKNOWN"
                draw_conf = 0.0

                while st.session_state.camera_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Camera error!")
                        break

                    frame_count += 1
                    now = time.time()
                    elapsed = now - attempt_start
                    remaining = max(0, int(config.UNKNOWN_TIMEOUT - elapsed))

                    # ── 30s TIMEOUT (only in recognition phase) ──
                    if elapsed > config.UNKNOWN_TIMEOUT and phase == "recognition":
                        if last_name == "UNKNOWN":
                            unknown_handler._capture_image(frame, None)
                            log.append_log("TIMEOUT", "Unknown", 0, "30s expired")
                            final_result = ("denied", "UNKNOWN", 0)
                        else:
                            log.append_log("TIMEOUT", last_name, last_confidence / 100.0,
                                           "Blink not completed")
                            final_result = ("timeout", last_name, last_confidence)
                        break

                    # ── PREPROCESSING ──
                    enhanced = enhance_frame(frame.copy())
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # ═══════════════════════════════════════════════
                    #  PHASE: DEPTH / MOTION CHECK
                    # ═══════════════════════════════════════════════
                    if phase == "depth" and depth_checker:
                        depth_checker.process_frame(frame)
                        task_info = depth_checker.get_current_task_info()

                        # Draw depth UI
                        display_frame = enhanced.copy()
                        state = task_info.get("state", "")
                        instruction = task_info.get("instruction", "")
                        icon = task_info.get("icon", "")
                        task_idx = task_info.get("task_index", 0)
                        total = task_info.get("total_tasks", 0)
                        time_rem = task_info.get("time_remaining", 0)
                        progress = task_info.get("progress", [])

                        # Draw task instruction on frame
                        h_f, w_f = display_frame.shape[:2]
                        overlay = display_frame.copy()
                        cv2.rectangle(overlay, (0, 0), (w_f, 60), (80, 40, 0), -1)
                        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                        cv2.putText(display_frame,
                                    f"DEPTH CHECK: {confirmed_name} ({confirmed_conf:.1f}%)",
                                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                        task_text = f"Task {task_idx + 1}/{total}"
                        cv2.putText(display_frame, task_text,
                                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                        if state == "BASELINE":
                            text = "Hold STILL — capturing baseline..."
                            tsz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                            tx = (w_f - tsz[0]) // 2
                            ty = h_f // 2
                            cv2.rectangle(display_frame, (tx - 10, ty - 30),
                                          (tx + tsz[0] + 10, ty + 10), (0, 0, 0), -1)
                            cv2.putText(display_frame, text, (tx, ty),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                        elif state == "TASK_ACTIVE":
                            text = f"{icon}  {instruction}"
                            tsz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            tx = (w_f - tsz[0]) // 2
                            ty = h_f // 2
                            cv2.rectangle(display_frame, (tx - 15, ty - 35),
                                          (tx + tsz[0] + 15, ty + 15), (0, 80, 0), -1)
                            cv2.putText(display_frame, text, (tx, ty),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                        # Status bars
                        depth_info = f"🔍 Depth Check: Task {task_idx + 1}/{total}"
                        if state == "BASELINE":
                            depth_info += " | 📷 Hold still..."
                        elif state == "TASK_ACTIVE":
                            depth_info += f" | {icon} {instruction} ({time_rem:.1f}s)"
                        elif state == "ALL_DONE":
                            depth_info += " | ✅ Complete"

                        # Show task results so far
                        for i, res in enumerate(progress):
                            r_icon = "✅" if res["passed"] else "❌"
                            depth_info += f" | T{i+1}:{r_icon}"

                        depth_status_area.warning(depth_info)

                        frame_display.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                                            channels="RGB")

                        # Check if depth check is complete
                        if depth_checker.is_complete():
                            if depth_checker.is_passed():
                                db.log_attendance(confirmed_name, confirmed_conf / 100.0)
                                log.append_log("ACCESS_GRANTED", confirmed_name,
                                               confirmed_conf / 100.0,
                                               f"Blinks: {blink_count} | Depth: PASS")
                                security.record_attempt(True)
                                final_result = ("granted", confirmed_name, confirmed_conf)
                            else:
                                log.append_log("DEPTH_FAIL", confirmed_name,
                                               confirmed_conf / 100.0,
                                               "Depth/motion check failed")
                                security.record_attempt(False)
                                final_result = ("spoofing", confirmed_name, confirmed_conf)

                                # Save spoofing image to digital_theft folder
                                try:
                                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    theft_path = os.path.join(
                                        config.DIGITAL_THEFT_DIR,
                                        f"spoof_{confirmed_name}_{ts}.jpg"
                                    )
                                    cv2.imwrite(theft_path, frame)
                                except Exception:
                                    pass
                            break

                        continue

                    # ═══════════════════════════════════════════════
                    #  PHASE: RECOGNITION + BLINK
                    # ═══════════════════════════════════════════════

                    # ── BLINK DETECTION (every frame) ──
                    try:
                        blink_result = liveness.detect_blink(frame)
                        blink_count = blink_result["blink_count"]
                    except Exception:
                        pass

                    # ── FACE RECOGNITION (every 3rd frame) ──
                    if frame_count % 3 == 0:
                        try:
                            boxes = fr.face_locations(rgb, model=config.FACE_RECOGNITION_MODEL)
                            encodings = fr.face_encodings(rgb, boxes)
                        except Exception:
                            boxes, encodings = [], []

                        if not boxes:
                            draw_box = None

                        for (top, right, bottom, left), encoding in zip(boxes, encodings):
                            name = "UNKNOWN"
                            confidence = 0.0

                            if len(data["encodings"]) > 0:
                                distances = fr.face_distance(data["encodings"], encoding)
                                min_dist = float(np.min(distances))
                                index = int(np.argmin(distances))
                                confidence = round((1 - min_dist) * 100, 2)

                                if min_dist < threshold and confidence > 50:
                                    name = data["names"][index]

                            if confidence < 45:
                                name = "UNKNOWN"

                            # Multi-frame confirmation
                            if name != "UNKNOWN":
                                name_counter[name] = name_counter.get(name, 0) + 1
                            else:
                                name_counter.clear()

                            last_name = name
                            last_confidence = confidence

                            draw_box = (top, right, bottom, left)
                            draw_name = name
                            draw_conf = confidence

                            # FACE CONFIRMED + BLINKS DONE → start depth check or grant
                            if (name != "UNKNOWN"
                                    and name_counter.get(name, 0) >= 5
                                    and blink_count >= config.REQUIRED_BLINKS):

                                confirmed_name = name
                                confirmed_conf = confidence

                                if depth_enabled and depth_checker:
                                    phase = "depth"
                                    depth_checker.start_session()
                                    liveness.reset()
                                    status_area.info(
                                        f"🟢 {name} confirmed! Starting depth/motion check...")
                                else:
                                    db.log_attendance(name, confidence / 100.0)
                                    log.append_log("ACCESS_GRANTED", name, confidence / 100.0,
                                                   f"Blinks: {blink_count}")
                                    security.record_attempt(True)
                                    final_result = ("granted", name, confidence)
                                break

                            # Unknown tracking
                            if name == "UNKNOWN" and confidence < 45:
                                if now - last_unknown_capture >= config.UNKNOWN_CAPTURE_INTERVAL:
                                    unknown_handler.process_unknown(
                                        frame, (top, right, bottom, left))
                                    last_unknown_capture = now

                                if not attempt_recorded:
                                    attempt_recorded = True
                                    security.record_attempt(False)
                                    log.append_log("ACCESS_DENIED", "Unknown",
                                                   confidence / 100.0, "Unknown face")

                        if final_result:
                            break

                    # ── DRAW FACE BOX ON EVERY FRAME ──
                    display_frame = enhanced.copy()
                    if draw_box is not None:
                        top, right, bottom, left = draw_box
                        color = (0, 255, 0) if draw_name != "UNKNOWN" else (0, 0, 255)
                        cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                        label = f"{draw_name} ({draw_conf:.1f}%)"
                        lsz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(display_frame, (left, top - lsz[1] - 10),
                                      (left + lsz[0] + 6, top), color, -1)
                        cv2.putText(display_frame, label, (left + 3, top - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # HUD
                    cv2.putText(display_frame, f"Blinks: {blink_count}/{config.REQUIRED_BLINKS}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(display_frame, f"Time: {remaining}s",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    if depth_enabled:
                        cv2.putText(display_frame, "Depth Check: ON",
                                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

                    # ── STATUS BAR ──
                    blink_text = f"👁️ Blinks: {blink_count}/{config.REQUIRED_BLINKS}"
                    confirmed = max(name_counter.values()) if name_counter else 0
                    confirm_text = f"✅ Confirmed: {min(confirmed, 5)}/5"
                    depth_text = " | 🔍 Depth: ON" if depth_enabled else ""
                    status_text = f"⏱️ {remaining}s | {blink_text} | {confirm_text}{depth_text}"
                    if last_name != "UNKNOWN":
                        status_text += f" | 🟢 {last_name} ({last_confidence:.1f}%)"
                    status_area.info(status_text)

                    # ── DISPLAY FRAME ──
                    frame_display.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                                        channels="RGB")

                # ── CLEANUP ──
                cap.release()
                liveness.release()
                if depth_checker:
                    depth_checker.release()
                st.session_state.camera_running = False

                # ── SHOW RESULT ──
                if final_result:
                    status, name, conf = final_result
                    if status == "granted":
                        result_area.markdown(f"""
                        <div class="access-granted">
                            <h2>✅ ACCESS GRANTED</h2>
                            <p style="font-size:1.3rem; margin:0.5rem 0;">Welcome, <strong>{name}</strong></p>
                            <p>Confidence: {conf:.1f}% | Attendance logged</p>
                            <p style="font-size:0.85rem; color: #a7f3d0;">{'🔍 Depth/motion verified' if depth_enabled else '👁️ Blink verified'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        status_area.success(f"✅ Access granted to {name}! Attendance logged.")
                        depth_status_area.empty()
                    elif status == "spoofing":
                        result_area.markdown(f"""
                        <div class="access-denied">
                            <h2>🚨 SPOOFING DETECTED</h2>
                            <p>Face matched <strong>{name}</strong> ({conf:.1f}%), but <strong>depth/motion check failed</strong>.</p>
                            <p>Screen replay or photo detected. Access denied.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        status_area.error("🚨 Spoofing detected — depth/motion check failed!")
                        depth_status_area.empty()
                    elif status == "timeout":
                        result_area.markdown(f"""
                        <div class="access-denied">
                            <h2>⏱️ TIME OUT</h2>
                            <p>Detected <strong>{name}</strong> ({conf:.1f}%) but verification was not completed.</p>
                            <p>Please try again.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif status == "denied":
                        result_area.markdown(f"""
                        <div class="access-denied">
                            <h2>❌ ACCESS DENIED</h2>
                            <p>Unknown face detected. Image saved for review.</p>
                        </div>
                        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2: ADMIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_admin:
    st.markdown("### 🛡️ Admin Panel")

    if not st.session_state.admin_authenticated:
        st.markdown("Enter admin password to access the control panel.")

        with st.form("admin_login"):
            password = st.text_input("Admin Password", type="password")
            login_btn = st.form_submit_button("🔓 Login", type="primary")

        if login_btn:
            security = get_security()
            if security.verify_admin_password(password):
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("❌ Invalid password!")
    else:
        col_top1, col_top2 = st.columns([4, 1])
        with col_top2:
            if st.button("🚪 Logout"):
                st.session_state.admin_authenticated = False
                st.rerun()

        security = get_security()
        db = get_db()
        log = get_log()
        adaptive = get_adaptive()

        admin_tab1, admin_tab2, admin_tab3, admin_tab4, admin_tab5, admin_tab6 = st.tabs(["📊 Status", "📋 Attendance", "🔑 Password", "👤 Unknown Faces", "🚨 Digital Theft", "🔓 Unlock"])

        # ─── 1. STATUS ────────────────────────────────────────────────
        with admin_tab1:
            st.markdown("#### System Status Overview")

            if st.button("🔄 Refresh Status", key="refresh_status"):
                st.cache_resource.clear()

            sec_status = security.get_status()
            data = get_data()
            integrity = log.verify_integrity()
            adaptive_stats = adaptive.get_stats()
            today_count = db.get_count()
            registered = list(set(data["names"])) if data["names"] else []

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                lock_icon = "🔒 LOCKED" if sec_status["locked"] else "🔓 Unlocked"
                lock_color = "🔴" if sec_status["locked"] else "🟢"
                st.markdown(f"""
                <div class="status-card">
                    <h3>Security</h3>
                    <div class="value">{lock_color} {lock_icon}</div>
                    <p>Failed: {sec_status['failed_attempts']}/{sec_status['max_attempts']}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="status-card">
                    <h3>Registered People</h3>
                    <div class="value">{len(registered)}</div>
                    <p>{len(data['encodings'])} embeddings</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="status-card">
                    <h3>Today's Attendance</h3>
                    <div class="value">{today_count}</div>
                    <p>{datetime.now().strftime('%Y-%m-%d')}</p>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                log_icon = "✅" if integrity["valid"] else "❌ TAMPERED"
                st.markdown(f"""
                <div class="status-card">
                    <h3>Log Integrity</h3>
                    <div class="value">{log_icon}</div>
                    <p>{integrity['total_entries']} entries</p>
                </div>
                """, unsafe_allow_html=True)

            if registered:
                st.markdown("**Registered Names:**")
                name_tags = " ".join([f'<span class="info-badge">{n}</span>' for n in sorted(registered)])
                st.markdown(name_tags, unsafe_allow_html=True)

            st.markdown(f"**Adaptive Learning:** {adaptive_stats['pending']} pending, "
                        f"{adaptive_stats['reviewed']} reviewed")

            st.markdown("---")
            st.markdown("#### Add New Faces And Train Model")
            st.caption(
                "Upload new user images into `known_faces`, then train only those pending folders. "
                "Existing names and embeddings stay untouched, and only the processed source images "
                "for newly added users are deleted after embedding."
            )

            feedback = st.session_state.pop("admin_face_feedback", None)
            if feedback:
                if feedback["level"] == "success":
                    st.success(feedback["message"])
                elif feedback["level"] == "warning":
                    st.warning(feedback["message"])
                else:
                    st.error(feedback["message"])

            pending_batches = get_pending_face_batches()

            add_col, queue_col = st.columns([3, 2])
            with add_col:
                with st.form("add_new_faces_form", clear_on_submit=True):
                    person_name = st.text_input("Person Name")
                    uploaded_files = st.file_uploader(
                        "Upload face images",
                        type=["jpg", "jpeg", "png", "bmp", "webp"],
                        accept_multiple_files=True
                    )
                    save_faces_btn = st.form_submit_button("Add To Known Faces", type="primary")

                if save_faces_btn:
                    try:
                        saved_name, saved_count = save_uploaded_faces(person_name, uploaded_files)
                        st.session_state.admin_face_feedback = {
                            "level": "success",
                            "message": (
                                f"Saved {saved_count} image(s) for {saved_name} in known_faces. "
                                "Click Train Model to create embeddings."
                            ),
                        }
                    except ValueError as exc:
                        st.session_state.admin_face_feedback = {
                            "level": "error",
                            "message": str(exc),
                        }
                    st.rerun()

            with queue_col:
                st.markdown("**Pending Training Queue**")
                if pending_batches:
                    for batch in pending_batches:
                        st.markdown(
                            f"- **{batch['name']}**: {batch['count']} image(s) in `{batch['path']}`"
                        )
                else:
                    st.info("No pending face images in known_faces.")

            if st.button("Train Model", type="primary", key="train_model_btn"):
                if not pending_batches:
                    st.session_state.admin_face_feedback = {
                        "level": "warning",
                        "message": "There are no pending images to train.",
                    }
                    st.rerun()

                before_embeddings = len(data["encodings"])
                updated_data = encode_known_faces(delete_originals=True, preserve_existing=True)
                added_embeddings = max(0, len(updated_data["encodings"]) - before_embeddings)

                st.cache_resource.clear()

                if added_embeddings:
                    st.session_state.admin_face_feedback = {
                        "level": "success",
                        "message": (
                            f"Training complete. Added {added_embeddings} new embedding(s). "
                            "Existing users were preserved, and processed source images were "
                            "deleted from known_faces."
                        ),
                    }
                else:
                    st.session_state.admin_face_feedback = {
                        "level": "warning",
                        "message": (
                            "Training finished, but no new embeddings were created. "
                            "The pending images were kept so you can review them."
                        ),
                    }

                st.rerun()

        # ─── 2. ATTENDANCE ────────────────────────────────────────────
        with admin_tab2:
            st.markdown("#### 📋 Attendance Records")

            col_d1, col_d2 = st.columns([2, 1])
            with col_d1:
                selected_date = st.date_input("Select Date", datetime.now())
            with col_d2:
                show_all = st.checkbox("Show all dates")

            if show_all:
                records = db.get_all_attendance()
                st.markdown(f"**All Records** ({len(records)} total)")
            else:
                date_str = selected_date.strftime("%Y-%m-%d")
                records = db.get_attendance(date_str)
                st.markdown(f"**{date_str}** — {len(records)} entries")

            if records:
                table_html = '<table class="attendance-table"><tr>'
                table_html += '<th>#</th><th>Name</th><th>Date</th><th>Time</th><th>Confidence</th></tr>'
                for i, r in enumerate(records, 1):
                    conf = f"{r['confidence'] * 100:.1f}%" if r['confidence'] else "N/A"
                    table_html += f"<tr><td>{i}</td><td>{r['name']}</td><td>{r['date']}</td>"
                    table_html += f"<td>{r['time']}</td><td>{conf}</td></tr>"
                table_html += '</table>'
                st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.info("No attendance records found for this date.")

        # ─── 3. PASSWORD ─────────────────────────────────────────────
        with admin_tab3:
            st.markdown("#### 🔑 Change Admin Password")

            with st.form("change_password"):
                new_pass = st.text_input("New Password", type="password")
                confirm_pass = st.text_input("Confirm Password", type="password")
                change_btn = st.form_submit_button("🔄 Change Password", type="primary")

            if change_btn:
                if not new_pass or len(new_pass) < 4:
                    st.error("Password must be at least 4 characters.")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match!")
                else:
                    security.set_admin_password(new_pass)
                    st.success("✅ Admin password changed successfully!")

        # ─── 4. UNKNOWN FACES ────────────────────────────────────────
        with admin_tab4:
            st.markdown("#### 👤 Captured Unknown Faces")

            unknown_dir = config.UNKNOWN_FACES_DIR
            if os.path.exists(unknown_dir):
                images = sorted(glob.glob(os.path.join(unknown_dir, "*.jpg")), reverse=True)

                if images:
                    st.markdown(f"**{len(images)} unknown face captures**")

                    cols_per_row = 4
                    for i in range(0, len(images), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            idx = i + j
                            if idx < len(images):
                                img_path = images[idx]
                                filename = os.path.basename(img_path)
                                with col:
                                    img = cv2.imread(img_path)
                                    if img is not None:
                                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        st.image(img_rgb, caption=filename)

                    st.markdown("---")
                    if st.button("🗑️ Delete All Unknown Faces", type="secondary"):
                        for img in images:
                            try:
                                os.remove(img)
                            except Exception:
                                pass
                        st.success(f"Deleted {len(images)} images.")
                        st.rerun()
                else:
                    st.info("No unknown face captures found.")
            else:
                st.info("Unknown faces directory does not exist yet.")

        # ─── 5. DIGITAL THEFT ─────────────────────────────────────────
        with admin_tab5:
            st.markdown("#### 🚨 Digital Theft — Spoofing Attempts")
            st.markdown("Images captured when someone **failed the depth/motion check** "
                        "(screen replay, photo, or video spoofing detected).")

            theft_dir = config.DIGITAL_THEFT_DIR
            if os.path.exists(theft_dir):
                theft_images = sorted(
                    glob.glob(os.path.join(theft_dir, "*.jpg")), reverse=True
                )

                if theft_images:
                    st.error(f"⚠️ **{len(theft_images)} spoofing attempt(s) detected!**")

                    cols_per_row = 3
                    for i in range(0, len(theft_images), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            idx = i + j
                            if idx < len(theft_images):
                                img_path = theft_images[idx]
                                filename = os.path.basename(img_path)
                                # Parse info from filename: spoof_Name_20260407_120000.jpg
                                parts = filename.replace(".jpg", "").split("_")
                                who = parts[1] if len(parts) > 1 else "Unknown"
                                when = ""
                                if len(parts) >= 4:
                                    when = f"{parts[2][:4]}-{parts[2][4:6]}-{parts[2][6:]} {parts[3][:2]}:{parts[3][2:4]}:{parts[3][4:]}"

                                with col:
                                    img = cv2.imread(img_path)
                                    if img is not None:
                                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        st.image(img_rgb, caption=f"🚨 {who}")
                                        if when:
                                            st.caption(f"📅 {when}")

                    st.markdown("---")
                    col_del1, col_del2 = st.columns([1, 3])
                    with col_del1:
                        if st.button("🗑️ Delete All", type="secondary", key="del_theft"):
                            for img in theft_images:
                                try:
                                    os.remove(img)
                                except Exception:
                                    pass
                            st.success(f"Deleted {len(theft_images)} spoofing images.")
                            st.rerun()
                else:
                    st.success("✅ No spoofing attempts detected. All clear!")
            else:
                st.success("✅ No spoofing attempts detected. All clear!")

        # ─── 6. UNLOCK ───────────────────────────────────────────────
        with admin_tab6:
            st.markdown("#### 🔓 System Lock Control")

            sec_status = security.get_status()

            if sec_status["locked"]:
                st.error(f"🔒 System is **LOCKED** — {sec_status['failed_attempts']} failed attempts")
                if st.button("🔓 Unlock System", type="primary"):
                    security.locked = False
                    security.failed_attempts = 0
                    security.lock_time = None
                    security._save_lock_state()
                    st.success("✅ System unlocked!")
                    st.rerun()
            else:
                st.success(f"🔓 System is unlocked. Failed attempts: "
                           f"{sec_status['failed_attempts']}/{sec_status['max_attempts']}")

            if sec_status["failed_attempts"] > 0 and not sec_status["locked"]:
                if st.button("Reset Failed Attempts Counter"):
                    security.failed_attempts = 0
                    security._save_lock_state()
                    st.success("Counter reset to 0.")
                    st.rerun()

            st.markdown("---")
            st.markdown("#### 📝 Log Integrity Check")
            if st.button("🔍 Verify Log Integrity"):
                result = log.verify_integrity()
                if result["valid"]:
                    st.success(f"✅ {result['details']}")
                else:
                    st.error(f"❌ INTEGRITY VIOLATION: {result['details']}")
