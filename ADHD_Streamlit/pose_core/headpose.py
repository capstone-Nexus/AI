#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import savgol_filter
from collections import deque
import time, argparse, os
from datetime import datetime

# ===============================
# 파라미터
# ===============================
THETA_YAW_DEG   = 18.0   # 15 -> 18  (좌우 더 많이 돌려야 턴)
THETA_PITCH_DEG = 10.0   #  8 -> 10  (고개 끄덕임 좀 더 커야 인식)
THETA_ROLL_DEG  = 15.0   # 12 -> 15  (기울임도 더 크게)
MIN_TURN_DURATION = 0.25 # 0.19 -> 0.25  (0.25초 이상 유지될 때만 턴으로 인정)

T_LEAVE = 1.5            # 그대로

FACE_CONF_THR = 0.65     # 0.6 -> 0.65 (얼굴 인식 더 확실할 때만 사용)

SMOOTHING_WINDOW = 7     # 6 -> 7  (조금 더 부드럽게)
SMOOTHING_POLYORDER = 2  # 그대로

CALIBRATION_DURATION = 2.5  # 2.3 -> 2.5 (기본 자세를 조금 더 안정적으로 잡기)

BASELINE_SIGMA_MULT_YAW   = 2.3  # 2.0 -> 2.3
BASELINE_SIGMA_MULT_PITCH = 1.8  # 1.5 -> 1.8
BASELINE_SIGMA_MULT_ROLL  = 2.1  # 1.8 -> 2.1

FPS_THRESHOLD = 20        # 그대로
TIME_ADJUSTMENT = 0.1     # 그대로


MAX_CAMERA_INDEX = 6   # 스캔 범위 확대

# ===============================
# MediaPipe 초기화 & 그리기 유틸
# ===============================
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

LANDMARK_IDX = [1,152,33,263,57,287]
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),         # nose
    (0.0, -330.0, -65.0),    # chin
    (-225.0, 170.0, -135.0), # right eye
    (225.0, 170.0, -135.0),  # left eye
    (-150.0, -150.0, -125.0),# right mouth
    (150.0, -150.0, -125.0)  # left mouth
], dtype=np.float64)

def init_mediapipe():
    return mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def estimate_headpose(landmarks, image_shape):
    h, w = image_shape[:2]
    image_points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in LANDMARK_IDX], dtype=np.float64)
    focal_length = w
    center = (w/2, h/2)
    K = np.array([[focal_length,0,center[0]],[0,focal_length,center[1]],[0,0,1]], dtype=np.float64)
    dist = np.zeros((4,1))
    success, rvec, tvec = cv2.solvePnP(MODEL_POINTS_3D, image_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return 0.0, 0.0, 0.0, None, None
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    yaw  = np.degrees(np.arctan2(R[1,0], R[0,0])) if sy >= 1e-6 else 0.0
    pitch= np.degrees(np.arctan2(-R[2,0], sy))
    # 신뢰도(눈 사이 거리 기반)
    eye_dist = np.linalg.norm(image_points[2] - image_points[3])
    conf = np.clip(eye_dist / (w * 0.15), 0, 1)
    return yaw, pitch, conf, rvec, tvec

# 3D 축을 2D로 투영해 그리기 (시각적 피드백)
def draw_pose_axes(frame, rvec, tvec, K):
    if rvec is None or tvec is None: return
    axis_len = 80
    axes_3d = np.float32([[axis_len,0,0],[0,axis_len,0],[0,0,axis_len],[0,0,0]])  # X,Y,Z,origin
    dist = np.zeros((4,1))
    pts, _ = cv2.projectPoints(axes_3d, rvec, tvec, K, dist)
    o = tuple(pts[3].ravel().astype(int))
    x = tuple(pts[0].ravel().astype(int))
    y = tuple(pts[1].ravel().astype(int))
    z = tuple(pts[2].ravel().astype(int))
    cv2.line(frame, o, x, (0,0,255), 2)   # X:red
    cv2.line(frame, o, y, (0,255,0), 2)   # Y:green
    cv2.line(frame, o, z, (255,0,0), 2)   # Z:blue

# FaceMesh 시각화 (임계 초과 시 빨강)
def draw_landmarks_with_alert(frame, face_landmarks, turning_now):
    if face_landmarks is None: return
    color_norm = (255,255,255)
    color_alert= (0,0,255)
    c = color_alert if turning_now else color_norm

    # 컨투어 + 티셀레이션(간결 버전)
    mp_draw.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_draw.DrawingSpec(color=c, thickness=1, circle_radius=0)
    )
    mp_draw.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=mp_draw.DrawingSpec(color=c, thickness=1, circle_radius=1),
        connection_drawing_spec=mp_draw.DrawingSpec(color=c, thickness=1, circle_radius=1)
    )

# ===============================
# 카메라 선택(아이폰 후면 회피 휴리스틱)
# ===============================
def try_open(index, backend=None):
    try:
        cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return None
        # 테스트 프레임 2~3회 읽어 안정화
        for _ in range(3):
            ret, frm = cap.read()
            if not ret: 
                time.sleep(0.05)
        ret, frm = cap.read()
        if not ret or frm is None:
            cap.release()
            return None
        h, w = frm.shape[:2]
        aspect = w / max(1,h)
        # 아이폰/세로/초고해상도 휴리스틱 제외
        if h > w*1.05:  # 세로 프레임(아이폰에서 흔함)
            return None
        if max(w,h) >= 2500: # 4K급 초고해상도 → 외부 카메라일 확률 ↑
            return None
        return cap
    except Exception:
        return None

def open_camera(mode:str, preferred_idx:int|None):
    # 우선순위 세트
    av_backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, 'CAP_AVFOUNDATION') else None
    backends = [av_backend, cv2.CAP_ANY] if av_backend else [cv2.CAP_ANY]

    if mode == "index" and preferred_idx is not None:
        for be in backends:
            cap = try_open(preferred_idx, be)
            if cap: return cap
        return None

    # internal 선호: 작은 해상도/가로형 우선
    index_order = list(range(0, MAX_CAMERA_INDEX+1))
    if mode == "iphone":
        # 아이폰 선호: 세로/고해상도도 허용(휴리스틱 완화)
        for idx in index_order:
            for be in backends:
                cap = cv2.VideoCapture(idx, be)
                if cap.isOpened(): return cap
        return None

    # auto/internal: 휴리스틱 필터 적용
    for idx in index_order:
        for be in backends:
            cap = try_open(idx, be)
            if cap: return cap
    return None

# ===============================
# 이벤트 감지기
# ===============================
class EventDetector:
    def __init__(self):
        self.yaw_buf, self.pitch_buf = deque(maxlen=SMOOTHING_WINDOW), deque(maxlen=SMOOTHING_WINDOW)
        self.cal_t0, self.cal_yaw, self.cal_pitch = None, [], []
        self.is_calibrated = False
        self.theta_yaw, self.theta_pitch = THETA_YAW_DEG, THETA_PITCH_DEG

        self.is_turning = False
        self.t_turn_start = None
        self.turn_events, self.total_turn = 0, 0.0

        self.is_missing = False
        self.t_missing_start = None
        self.leave_events, self.longest_leave = 0, 0.0

        self.prev_t = None
        self.fps_buf = deque(maxlen=30)
        self.fps = 30.0

        self.min_turn = MIN_TURN_DURATION
        self.t_leave = T_LEAVE

        self.max_abs_yaw, self.max_abs_pitch = 0.0, 0.0

    def update_fps(self):
        now = time.time()
        if self.prev_t is not None:
            dt = now - self.prev_t
            if dt > 0:
                self.fps_buf.append(1.0/dt)
                self.fps = np.mean(self.fps_buf)
                if self.fps < FPS_THRESHOLD:
                    self.min_turn = MIN_TURN_DURATION + TIME_ADJUSTMENT
                    self.t_leave  = T_LEAVE + TIME_ADJUSTMENT
        self.prev_t = now

    def calibrate(self, yaw, pitch):
        if self.cal_t0 is None: self.cal_t0 = time.time()
        self.cal_yaw.append(yaw); self.cal_pitch.append(pitch)
        if (time.time() - self.cal_t0) >= CALIBRATION_DURATION and not self.is_calibrated:
            ym, ys = np.mean(self.cal_yaw), np.std(self.cal_yaw)
            pm, ps = np.mean(self.cal_pitch), np.std(self.cal_pitch)
            self.theta_yaw   = max(THETA_YAW_DEG,   abs(ym) + BASELINE_SIGMA_MULT_YAW   * ys)
            self.theta_pitch = max(THETA_PITCH_DEG, abs(pm) + BASELINE_SIGMA_MULT_PITCH * ps)
            self.is_calibrated = True
            print(f"[INFO] Calibration: yaw={self.theta_yaw:.1f}°, pitch={self.theta_pitch:.1f}°")

    def smooth(self, yaw, pitch):
        self.yaw_buf.append(yaw); self.pitch_buf.append(pitch)
        if len(self.yaw_buf) < SMOOTHING_WINDOW: return yaw, pitch
        return (
            savgol_filter(self.yaw_buf,   SMOOTHING_WINDOW, SMOOTHING_POLYORDER)[-1],
            savgol_filter(self.pitch_buf, SMOOTHING_WINDOW, SMOOTHING_POLYORDER)[-1]
        )

    def detect_turn(self, yaw, pitch, now):
        self.max_abs_yaw = max(self.max_abs_yaw, abs(yaw))
        self.max_abs_pitch = max(self.max_abs_pitch, abs(pitch))
        turning = (abs(yaw) > self.theta_yaw) or (abs(pitch) > self.theta_pitch)
        if turning and not self.is_turning:
            self.is_turning = True; self.t_turn_start = now
        elif not turning and self.is_turning:
            dur = now - self.t_turn_start
            if dur >= self.min_turn:
                self.turn_events += 1
                self.total_turn += dur
            self.is_turning = False; self.t_turn_start = None
        return turning

    def detect_leave(self, present, now):
        if not present:
            if not self.is_missing:
                self.is_missing = True; self.t_missing_start = now
        else:
            if self.is_missing:
                dur = now - self.t_missing_start
                if dur >= self.t_leave:
                    self.leave_events += 1
                    self.longest_leave = max(self.longest_leave, dur)
                self.is_missing = False; self.t_missing_start = None

# ===============================
# 오버레이
# ===============================
def draw_overlay(frame, D, present, conf, yaw, pitch, valid_ratio):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10,10), (450,300), (0,0,0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    y, lh = 35, 30
    def put(t,c=(255,255,255)):
        nonlocal y; cv2.putText(frame, t, (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2); y += lh
    put("=== Face Tracking ===")
    put(f"FPS: {D.fps:.1f}", (0,255,0))
    put(f"Face: {'DETECTED' if present else 'MISSING'} (conf:{conf:.2f})", (0,255,0) if present else (0,0,255))
    put(f"Yaw: {yaw:+.1f}°  (thr {D.theta_yaw:.1f})", (255,200,0))
    put(f"Pitch: {pitch:+.1f}°(thr {D.theta_pitch:.1f})", (255,200,0))
    put(f"Turn: {'TURNING' if D.is_turning else 'Normal'} (count:{D.turn_events})",
        (0,165,255) if D.is_turning else (255,255,255))
    put(f"Seat: {'AWAY' if D.is_missing else 'Present'} (count:{D.leave_events})",
        (0,0,255) if D.is_missing else (255,255,255))
    put(f"Valid Frames: {valid_ratio:.1f}%", (200,200,200))
    if not D.is_calibrated:
        rem = max(0.0, CALIBRATION_DURATION - (time.time() - (D.cal_t0 or time.time())))
        put(f"Calibrating... {rem:.1f}s", (0,255,255))
    return frame

# ===============================
# 메인
# ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="auto", choices=["auto","internal","iphone","index"], help="camera pick strategy")
    ap.add_argument("--cam", type=int, default=None, help="when --mode index, use this number")
    args, _ = ap.parse_known_args()

    cap = open_camera(args.mode, args.cam)
    if cap is None:
        print("[ERROR] 카메라 열기 실패. --mode index --cam 0 등으로 직접 지정해보세요.")
        return

    # 기본 해상도 세팅(지정 실패해도 무방)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    face_mesh = init_mediapipe()
    D = EventDetector()
    D.cal_t0 = time.time()

    win = "Head Turn & Seat Leave Detection"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    start = time.time()
    log, total, valid = [], 0, 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("[WARN] 프레임 실패"); break

            total += 1
            now = time.time()
            D.update_fps()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            present, yaw, pitch, conf = False, 0.0, 0.0, 0.0
            rvec=tvec=None

            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                yaw, pitch, conf, rvec, tvec = estimate_headpose(lms, frame.shape)
                if conf >= FACE_CONF_THR:
                    present = True
                    valid += 1
                    yaw, pitch = D.smooth(yaw, pitch)
                    turning_now = False
                    if not D.is_calibrated:
                        D.calibrate(yaw, pitch)
                    else:
                        turning_now = D.detect_turn(yaw, pitch, now)

                    # 1) 랜드마크 시각화(임계 초과 시 빨강)
                    draw_landmarks_with_alert(frame, res.multi_face_landmarks[0], turning_now)

                    # 2) pose 축 시각화
                    h, w = frame.shape[:2]
                    K = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
                    draw_pose_axes(frame, rvec, tvec, K)

            D.detect_leave(present, now)
            valid_ratio = (valid / max(1,total)) * 100.0
            frame = draw_overlay(frame, D, present, conf, yaw, pitch, valid_ratio)
            cv2.imshow(win, frame)

            log.append([f"{now-start:.3f}", int(present), f"{conf:.3f}", f"{yaw:.2f}", f"{pitch:.2f}",
                        int(D.is_turning), int(D.is_missing), D.turn_events, D.leave_events])

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'),27): break
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1: break

    finally:
        dur = time.time()-start
        print("="*60)
        print(f"총 실행 시간: {dur:.1f}s, 총 프레임: {total}, 유효: {valid} ({valid/max(1,total)*100:.1f}%)")
        print(f"평균 FPS: {D.fps:.1f}")
        print(f"고개 돌림: {D.turn_events}, 자리 이탈: {D.leave_events}")
        print(f"최대 |Yaw|={D.max_abs_yaw:.1f}°, |Pitch|={D.max_abs_pitch:.1f}°")
        print("="*60)

        os.makedirs("logs", exist_ok=True)
        fn = f"logs/headpose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(fn,"w",encoding="utf-8") as f:
            f.write("time_s,face_present,face_conf,yaw_deg,pitch_deg,turning,seat_missing,turn_events,seat_leave_events\n")
            for r in log: f.write(",".join(map(str,r))+"\n")
        print(f"[INFO] 로그 저장: {fn}")

        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
