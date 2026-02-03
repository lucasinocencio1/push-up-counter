import time
import threading
from collections import deque
from dataclasses import asdict

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

import mediapipe as mp

# reuse your modules (when app lives in src/)
from settings import (
    ANGLE_UP_THRESHOLD, ANGLE_DOWN_THRESHOLD,
    MIN_FRAMES_IN_STATE, SMOOTHING_WINDOW,
    USE_HIP_CHECK, HIP_MIN_DELTA,
    DEFAULT_TARGET_REPS, INACTIVITY_SECONDS,
)
from session_logger import SessionSummary, save_session

# ------------- Utils -------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def calc_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def to_px(pt, w, h):
    return int(pt.x * w), int(pt.y * h)

def fmt_time(ts):
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

# ------------- Thread-safe shared state (video callback runs in another thread) -------------
_sync_lock = threading.Lock()
_shared = {
    "config": {},
    "reps_total": 0,
    "sets_total": 0,
    "reps_current_set": 0,
    "reps_per_set": [],
    "state": "TOP",
    "frames_in_state": 0,
    "angle_hist": deque(maxlen=30),
    "hip_y_hist": deque(maxlen=30),
    "last_movement_time": 0.0,
    "last_rep_time": None,
    "rep_times": [],
    "angle_display": None,
    "hip_move_ok": True,
    "_pose": None,
}

def _make_processor():
    """Lazy-init MediaPipe Pose (used inside callback)."""
    with _sync_lock:
        if _shared["_pose"] is None:
            _shared["_pose"] = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
                smooth_landmarks=True,
            )
        return _shared["_pose"]

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Process each video frame; must not use st.session_state (runs in worker thread)."""
    pose = _make_processor()
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    h, w = img.shape[:2]

    with _sync_lock:
        cfg = _shared["config"]
        angle_up = cfg.get("angle_up", ANGLE_UP_THRESHOLD)
        angle_down = cfg.get("angle_down", ANGLE_DOWN_THRESHOLD)
        min_frames = cfg.get("min_frames", MIN_FRAMES_IN_STATE)
        smooth_win = cfg.get("smooth_win", SMOOTHING_WINDOW)
        use_hip = cfg.get("use_hip", USE_HIP_CHECK)
        hip_min_delta = cfg.get("hip_min_delta", HIP_MIN_DELTA)
        target_reps = cfg.get("target_reps", DEFAULT_TARGET_REPS)
        idle_secs = cfg.get("idle_secs", INACTIVITY_SECONDS)
        angle_hist = _shared["angle_hist"]
        hip_y_hist = _shared["hip_y_hist"]
        if _shared["last_movement_time"] == 0.0:
            _shared["last_movement_time"] = time.time()
        # ensure deque maxlen matches smooth_win
        if angle_hist.maxlen != smooth_win:
            _shared["angle_hist"] = deque(angle_hist, maxlen=smooth_win)
            angle_hist = _shared["angle_hist"]

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    angle = None
    hip_ok = True

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark

        mp_drawing.draw_landmarks(
            img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )

        r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        r_el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
        r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
        sx, sy = to_px(r_sh, w, h); ex, ey = to_px(r_el, w, h); wx, wy = to_px(r_wr, w, h)

        angle = calc_angle((sx, sy), (ex, ey), (wx, wy))
        with _sync_lock:
            angle_hist.append(angle)
            angle_s = float(np.mean(angle_hist)) if angle_hist else angle
            _shared["angle_display"] = angle_s

            if use_hip:
                r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
                _, hip_y = to_px(r_hip, w, h)
                hip_y_norm = hip_y / float(h)
                hip_y_hist.append(hip_y_norm)
                if len(hip_y_hist) > 5:
                    recent = list(hip_y_hist)[-5:]
                    amp = max(recent) - min(recent)
                    hip_ok = amp >= hip_min_delta
                _shared["hip_move_ok"] = hip_ok

            # state machine
            _shared["frames_in_state"] += 1
            frames_in_state = _shared["frames_in_state"]
            state = _shared["state"]
            moved = False

            if state == "TOP":
                if angle_s <= angle_down and frames_in_state >= min_frames:
                    _shared["state"] = "BOTTOM"
                    _shared["frames_in_state"] = 0
                    moved = True
            elif state == "BOTTOM":
                if angle_s >= angle_up and frames_in_state >= min_frames:
                    if (not use_hip) or hip_ok:
                        _shared["reps_total"] += 1
                        _shared["reps_current_set"] += 1
                        now = time.time()
                        if _shared["last_rep_time"] is not None:
                            _shared["rep_times"].append(now - _shared["last_rep_time"])
                        _shared["last_rep_time"] = now
                        if _shared["reps_current_set"] >= target_reps:
                            _shared["sets_total"] += 1
                            _shared["reps_per_set"].append(_shared["reps_current_set"])
                            _shared["reps_current_set"] = 0
                    _shared["state"] = "TOP"
                    _shared["frames_in_state"] = 0
                    moved = True

            if moved:
                _shared["last_movement_time"] = time.time()

        # overlay
        cv2.rectangle(img, (0, 0), (w, 40), (0, 0, 0), -1)
        with _sync_lock:
            rt, st_num, rcs = _shared["reps_total"], _shared["sets_total"], _shared["reps_current_set"]
        cv2.putText(img, f"Reps {rt}  |  Set {st_num+1} ({rcs}/{target_reps})",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if angle is not None:
            cv2.putText(img, f"Ang: {int(angle)}", (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if use_hip:
            cv2.putText(img, f"Hip: {'OK' if hip_ok else 'NO'}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255) if hip_ok else (0, 0, 255), 2)

    # idle → close current set
    with _sync_lock:
        last_mov = _shared["last_movement_time"]
        rcs = _shared["reps_current_set"]
    idle = time.time() - last_mov
    if idle >= idle_secs and rcs > 0:
        with _sync_lock:
            _shared["sets_total"] += 1
            _shared["reps_per_set"].append(_shared["reps_current_set"])
            _shared["reps_current_set"] = 0
            _shared["last_movement_time"] = time.time()

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ------------- Session state init -------------
if "session_start" not in st.session_state:
    st.session_state.session_start = time.time()
    st.session_state.reps_total = 0
    st.session_state.sets_total = 0
    st.session_state.reps_current_set = 0
    st.session_state.reps_per_set = []
    st.session_state.last_movement_time = time.time()
    st.session_state.last_rep_time = None
    st.session_state.rep_times = []
    st.session_state.state = "TOP"
    st.session_state.frames_in_state = 0
    st.session_state.angle_hist = deque(maxlen=SMOOTHING_WINDOW)
    st.session_state.hip_y_hist = deque(maxlen=30)
    st.session_state.hip_move_ok = True
    st.session_state.angle_display = None

# ------------- Sidebar (UI) -------------
st.sidebar.title("⚙️ Settings")

angle_up = st.sidebar.slider("Top angle (°)", 140, 180, ANGLE_UP_THRESHOLD, 1)
angle_down = st.sidebar.slider("Bottom angle (°)", 30, 120, ANGLE_DOWN_THRESHOLD, 1)
min_frames = st.sidebar.slider("Min frames per state", 1, 10, MIN_FRAMES_IN_STATE, 1)
smooth_win = st.sidebar.slider("Smoothing (moving avg window)", 1, 15, SMOOTHING_WINDOW, 1)
use_hip = st.sidebar.checkbox("Require hip movement", value=USE_HIP_CHECK)
hip_min_delta = st.sidebar.slider("Min hip amplitude", 0.0, 0.1, HIP_MIN_DELTA, 0.005)
target_reps = st.sidebar.number_input("Target reps per set", 1, 200, DEFAULT_TARGET_REPS, 1)
idle_secs = st.sidebar.slider("Idle time to close set (s)", 2, 15, int(INACTIVITY_SECONDS), 1)

reset_btn = st.sidebar.button("🔁 Reset session")

if reset_btn:
    t0 = time.time()
    st.session_state.session_start = t0
    st.session_state.reps_total = 0
    st.session_state.sets_total = 0
    st.session_state.reps_current_set = 0
    st.session_state.reps_per_set = []
    st.session_state.last_movement_time = t0
    st.session_state.last_rep_time = None
    st.session_state.rep_times = []
    st.session_state.state = "TOP"
    st.session_state.frames_in_state = 0
    st.session_state.angle_hist = deque(maxlen=smooth_win)
    st.session_state.hip_y_hist = deque(maxlen=30)
    st.session_state.hip_move_ok = True
    st.session_state.angle_display = None
    with _sync_lock:
        _shared["reps_total"] = 0
        _shared["sets_total"] = 0
        _shared["reps_current_set"] = 0
        _shared["reps_per_set"] = []
        _shared["state"] = "TOP"
        _shared["frames_in_state"] = 0
        _shared["last_movement_time"] = t0
        _shared["last_rep_time"] = None
        _shared["rep_times"] = []
        _shared["angle_display"] = None
        _shared["hip_move_ok"] = True
        _shared["angle_hist"] = deque(maxlen=smooth_win)
        _shared["hip_y_hist"] = deque(maxlen=30)

# Push config to shared state so the video callback can read it
with _sync_lock:
    _shared["config"] = {
        "angle_up": angle_up,
        "angle_down": angle_down,
        "min_frames": min_frames,
        "smooth_win": smooth_win,
        "use_hip": use_hip,
        "hip_min_delta": hip_min_delta,
        "target_reps": target_reps,
        "idle_secs": float(idle_secs),
    }

# Sync from shared state to session state for display (callback runs in another thread)
with _sync_lock:
    st.session_state.reps_total = _shared["reps_total"]
    st.session_state.sets_total = _shared["sets_total"]
    st.session_state.reps_current_set = _shared["reps_current_set"]
    st.session_state.reps_per_set = list(_shared["reps_per_set"])
    st.session_state.state = _shared["state"]
    st.session_state.angle_display = _shared["angle_display"]
    st.session_state.hip_move_ok = _shared["hip_move_ok"]
    st.session_state.last_movement_time = _shared["last_movement_time"]
    st.session_state.last_rep_time = _shared["last_rep_time"]
    st.session_state.rep_times = list(_shared["rep_times"])

# ------------- Header -------------
st.title("Push-up Counter — Streamlit v1")

# Top KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Reps", st.session_state.reps_total)
col2.metric("Current set", st.session_state.sets_total + 1)
col3.metric("In set", f"{st.session_state.reps_current_set}/{target_reps}")
col4.metric("State", st.session_state.state)

# Rep progress bar
prog = min(1.0, max(0.0, (st.session_state.angle_display or 0.0 - angle_down) / (angle_up - angle_down) if angle_up != angle_down else 0.0))
st.progress(prog)

# ------------- WebRTC Video -------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

try:
    webrtc_streamer(
        key="pushup",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        video_frame_callback=video_frame_callback,
    )
except Exception as e:
    if "NoSessionError" in type(e).__name__ or "thread context" in str(e).lower():
        st.error(
            "WebRTC needs a proper Streamlit session. **Run the app from a terminal with:**\n\n"
            "`streamlit run src/streamlit_app.py`\n\n"
            "Then open http://localhost:8501 in your browser. Do not run the script with `python` or from an IDE run button."
        )
    else:
        raise

# ------------- Status panel -------------
st.subheader("Status")
col_a, col_b = st.columns(2)
col_a.write(f"Angle (smoothed): **{int(st.session_state.angle_display or 0)}°**")
if use_hip:
    col_b.write(f"Hip: **{'OK' if st.session_state.hip_move_ok else 'NO'}**")

idle_now = int(time.time() - st.session_state.last_movement_time)
st.caption(f"Idle: {idle_now}s")

# ------------- Save session -------------
def finalize_and_save():
    session_end = time.time()
    with _sync_lock:
        rcs = _shared["reps_current_set"]
        if rcs > 0:
            _shared["sets_total"] += 1
            _shared["reps_per_set"].append(rcs)
            _shared["reps_current_set"] = 0
        st.session_state.reps_total = _shared["reps_total"]
        st.session_state.sets_total = _shared["sets_total"]
        st.session_state.reps_current_set = _shared["reps_current_set"]
        st.session_state.reps_per_set = list(_shared["reps_per_set"])
        st.session_state.rep_times = list(_shared["rep_times"])
    avg_rep = (sum(st.session_state.rep_times) / len(st.session_state.rep_times)) if st.session_state.rep_times else None
    summary = SessionSummary(
        start_time=fmt_time(st.session_state.session_start),
        end_time=fmt_time(session_end),
        duration_sec=session_end - st.session_state.session_start,
        total_reps=st.session_state.reps_total,
        total_sets=st.session_state.sets_total,
        reps_per_set=st.session_state.reps_per_set,
        avg_sec_per_rep=avg_rep,
        params={
            "ANGLE_UP_THRESHOLD": angle_up,
            "ANGLE_DOWN_THRESHOLD": angle_down,
            "MIN_FRAMES_IN_STATE": min_frames,
            "SMOOTHING_WINDOW": smooth_win,
            "USE_HIP_CHECK": use_hip,
            "HIP_MIN_DELTA": hip_min_delta,
            "DEFAULT_TARGET_REPS": target_reps,
            "INACTIVITY_SECONDS": idle_secs,
        }
    )
    save_session(summary)
    st.success("Session summary saved to `data/sessions/` ✅")
    st.json(asdict(summary))

st.button("💾 Save session now", on_click=finalize_and_save)
