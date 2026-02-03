# Push-up Counter v2
# - Count by elbow angle (MediaPipe Pose)
# - Hip-check (anti-cheat)
# - Sets with target reps and auto-pause on inactivity
# - Beep on rep and when finishing a set
# - Saves session summary (JSON + CSV)

import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp

from settings import (
    ANGLE_UP_THRESHOLD, ANGLE_DOWN_THRESHOLD,
    MIN_FRAMES_IN_STATE, SMOOTHING_WINDOW,
    USE_HIP_CHECK, HIP_MIN_DELTA,
    DEFAULT_TARGET_REPS, INACTIVITY_SECONDS,
    CAM_INDEX, DRAW_SKELETON,
    BEEP_ON_REP, BEEP_ON_SET,
)
from beeper import beep
from session_logger import SessionSummary, save_session

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def calc_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def to_px(pt, w, h):
    return int(pt.x * w), int(pt.y * h)

def fmt_time(ts): return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit(f"Could not access webcam (index {CAM_INDEX}).")

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        smooth_landmarks=True,
    )

    # Session
    session_start = time.time()
    reps_total = 0
    sets_total = 0
    reps_current_set = 0
    reps_per_set = []
    target_reps = DEFAULT_TARGET_REPS

    # State
    state = "TOP"
    frames_in_state = 0

    # Smoothing
    angle_hist = deque(maxlen=SMOOTHING_WINDOW)

    # Hip check
    hip_y_hist = deque(maxlen=30)

    # Time metrics
    prev_time = time.time()
    last_movement_time = time.time()
    last_rep_time = None
    rep_times = []  # time between reps

    show_debug = True

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        angle = None
        hip_move_ok = True

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark

            if DRAW_SKELETON:
                mp_drawing.draw_landmarks(
                    frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
                )

            # right side
            r_sh = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            r_el = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
            r_wr = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            sx, sy = to_px(r_sh, w, h); ex, ey = to_px(r_el, w, h); wx, wy = to_px(r_wr, w, h)
            angle = calc_angle((sx, sy), (ex, ey), (wx, wy))
            angle_hist.append(angle)
            angle_s = np.mean(angle_hist) if angle_hist else angle

            # hip check
            if USE_HIP_CHECK:
                r_hip = lm[mp_pose.PoseLandmark.RIGHT_HIP]
                _, hip_y = to_px(r_hip, w, h)
                hip_y_norm = hip_y / float(h)
                hip_y_hist.append(hip_y_norm)
                if len(hip_y_hist) > 5:
                    recent = list(hip_y_hist)[-5:]
                    amp = max(recent) - min(recent)
                    hip_move_ok = amp >= HIP_MIN_DELTA

            # State machine
            frames_in_state += 1
            moved = False
            if angle_s is not None:
                if state == "TOP":
                    if angle_s <= ANGLE_DOWN_THRESHOLD and frames_in_state >= MIN_FRAMES_IN_STATE:
                        state = "BOTTOM"; frames_in_state = 0
                        moved = True
                elif state == "BOTTOM":
                    if angle_s >= ANGLE_UP_THRESHOLD and frames_in_state >= MIN_FRAMES_IN_STATE:
                        if (not USE_HIP_CHECK) or hip_move_ok:
                            # completed rep
                            reps_total += 1
                            reps_current_set += 1
                            if BEEP_ON_REP: beep(990, 100)

                            # time per rep
                            now = time.time()
                            if last_rep_time is not None:
                                rep_times.append(now - last_rep_time)
                            last_rep_time = now

                            # set completed?
                            if reps_current_set >= target_reps:
                                sets_total += 1
                                if BEEP_ON_SET: beep(660, 180)
                                reps_per_set.append(reps_current_set)
                                reps_current_set = 0

                        state = "TOP"; frames_in_state = 0
                        moved = True

            if moved:
                last_movement_time = time.time()

            # Arm drawing
            cv2.circle(frame, (sx, sy), 6, (0, 255, 0), -1)
            cv2.circle(frame, (ex, ey), 6, (0, 255, 0), -1)
            cv2.circle(frame, (wx, wy), 6, (0, 255, 0), -1)
            cv2.line(frame, (sx, sy), (ex, ey), (255, 255, 255), 2)
            cv2.line(frame, (ex, ey), (wx, wy), (255, 255, 255), 2)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-8)
        prev_time = now

        # Inactivity → close current set (if any reps)
        idle = (time.time() - last_movement_time)
        if idle >= INACTIVITY_SECONDS and reps_current_set > 0:
            sets_total += 1
            if BEEP_ON_SET: beep(660, 180)
            reps_per_set.append(reps_current_set)
            reps_current_set = 0
            last_movement_time = time.time()

        # UI header
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(frame, f"Reps: {reps_total}  |  Set: {sets_total+1}  ({reps_current_set}/{target_reps})",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

        cv2.putText(frame, f"State: {state}   FPS: {int(fps)}   Idle: {int(idle)}s",
                    (10, 70+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        if angle is not None and show_debug:
            cv2.putText(frame, f"Elbow angle: {int(angle)} deg", (10, 70+45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if USE_HIP_CHECK and show_debug:
            cv2.putText(frame, f"Hip ok: {'YES' if hip_move_ok else 'NO'}",
                        (10, 70+70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255) if hip_move_ok else (0, 0, 255), 2)

        # Help
        if show_debug:
            cv2.putText(frame, "Controls: [q] quit  [r] reset set  [s] debug on/off  [n/p] target reps -/+",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Push-up Counter v2", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            reps_current_set = 0
            angle_hist.clear(); hip_y_hist.clear()
            state = "TOP"; frames_in_state = 0
            last_movement_time = time.time()
        elif key == ord('s'):
            show_debug = not show_debug
        elif key == ord('n'):  # decrease target
            target_reps = max(1, target_reps - 1)
        elif key == ord('p'):  # increase target
            target_reps = min(200, target_reps + 1)

    # Cleanup and save session
    cap.release(); cv2.destroyAllWindows()
    session_end = time.time()

    # finish pending set
    if reps_current_set > 0:
        sets_total += 1
        reps_per_set.append(reps_current_set)

    avg_rep = (sum(rep_times)/len(rep_times)) if rep_times else None

    summary = SessionSummary(
        start_time=fmt_time(session_start),
        end_time=fmt_time(session_end),
        duration_sec=session_end - session_start,
        total_reps=reps_total,
        total_sets=sets_total,
        reps_per_set=reps_per_set,
        avg_sec_per_rep=avg_rep,
        params={
            "ANGLE_UP_THRESHOLD": ANGLE_UP_THRESHOLD,
            "ANGLE_DOWN_THRESHOLD": ANGLE_DOWN_THRESHOLD,
            "MIN_FRAMES_IN_STATE": MIN_FRAMES_IN_STATE,
            "SMOOTHING_WINDOW": SMOOTHING_WINDOW,
            "USE_HIP_CHECK": USE_HIP_CHECK,
            "HIP_MIN_DELTA": HIP_MIN_DELTA,
            "DEFAULT_TARGET_REPS": DEFAULT_TARGET_REPS,
            "INACTIVITY_SECONDS": INACTIVITY_SECONDS,
        }
    )
    save_session(summary)
    print("\nSession summary saved to data/sessions/")
    print(summary)

if __name__ == "__main__":
    main()
