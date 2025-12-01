
"""
ADHD Headpose Streamlit App (2025)
---------------------------
- 실시간 얼굴/머리포즈 분석 (OpenCV + Mediapipe + streamlit-webrtc)
- preview/test 모드 지원 (쿼리파라미터로 제어)
- 분석 엔진: pose_core/headpose.py (수정 금지, 그대로 import)
- CSV 저장, 세션 상태 관리, WebRTC 영상 처리, UI 안내 등 완비
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import numpy as np
import pandas as pd
import cv2
import time
import os
from pose_core import headpose

# ===============================
# 1. 쿼리 파라미터 파싱 및 세션 상태 반영
# ===============================
def get_query_params():
    params = st.query_params
    def get(key, default=None, typ=str):
        v = params.get(key, default)
        if v is None:
            return default
        try:
            return typ(v)
        except Exception:
            return default
    return {
        "session_id": get("session_id", "demo", str),
        "mode": get("mode", "preview", str),
        "duration": get("duration", 30, int),
        "yaw_thr": get("yaw_thr", 15, float),
        "pitch_thr": get("pitch_thr", 8, float),
        "reset_cal": get("reset_cal", "0", str),
        "upload_url": get("upload_url", None, str)
    }

params = get_query_params()
for k, v in params.items():
    st.session_state[k] = v


# ===============================
# 3. 세션 상태 안전 초기화 함수
# ===============================
def ensure_session_state():
    if "detector" not in st.session_state:
        st.session_state["detector"] = headpose.EventDetector()
        st.session_state["detector"].theta_yaw = st.session_state.get("yaw_thr", 15)
        st.session_state["detector"].theta_pitch = st.session_state.get("pitch_thr", 8)
    if "face_mesh" not in st.session_state:
        st.session_state["face_mesh"] = headpose.init_mediapipe()
    if "log" not in st.session_state:
        st.session_state["log"] = []
    if "start_time" not in st.session_state:
        st.session_state["start_time"] = None
    if "done" not in st.session_state:
        st.session_state["done"] = False
    if "csv_path" not in st.session_state:
        st.session_state["csv_path"] = None

# ===============================
# 4. VideoProcessor (WebRTC 영상 처리)
# ===============================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        ensure_session_state()
        self.detector = st.session_state["detector"]
        self.face_mesh = st.session_state["face_mesh"]
        self.log = st.session_state["log"]
        self.start_time = st.session_state["start_time"]
        self.mode = st.session_state.get("mode", "preview")
        self.duration = st.session_state.get("duration", 30)
        self.done = st.session_state["done"]

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            now = time.time()
            print(f"[DEBUG] frame shape: {img.shape}")
            if st.session_state.get("done", False):
                if self.mode == "test":
                    return av.VideoFrame.from_ndarray(np.zeros_like(img), format="bgr24")
                else:
                    return av.VideoFrame.from_ndarray(img, format="bgr24")
            if self.start_time is None:
                self.start_time = now
                st.session_state["start_time"] = now
            elapsed = now - self.start_time
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)
            present, yaw, pitch, conf = False, 0.0, 0.0, 0.0
            rvec = tvec = None
            turning_now = False
            if res and getattr(res, "multi_face_landmarks", None):
                lms = res.multi_face_landmarks[0].landmark
                yaw, pitch, conf, rvec, tvec = headpose.estimate_headpose(lms, img.shape)
                if conf >= headpose.FACE_CONF_THR:
                    present = True
                    yaw, pitch = self.detector.smooth(yaw, pitch)
                    if not self.detector.is_calibrated:
                        self.detector.calibrate(yaw, pitch)
                    else:
                        turning_now = self.detector.detect_turn(yaw, pitch, now)
                    if self.mode == "preview":
                        headpose.draw_landmarks_with_alert(img, res.multi_face_landmarks[0], self.detector.is_turning)
                        h, w = img.shape[:2]
                        K = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
                        headpose.draw_pose_axes(img, rvec, tvec, K)
            self.detector.detect_leave(present, now)
            self.detector.update_fps()
            self.log.append([
                f"{elapsed:.3f}", int(present), f"{conf:.3f}", f"{yaw:.2f}", f"{pitch:.2f}",
                int(self.detector.is_turning), int(self.detector.is_missing), self.detector.turn_events, self.detector.leave_events
            ])
            # duration 제한 제거: 항상 분석 계속
            if self.mode == "test":
                return av.VideoFrame.from_ndarray(np.zeros_like(img), format="bgr24")
            else:
                return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[ERROR][recv] {tb}")
            st.warning(f"VideoProcessor.recv() 예외 발생: {e}")
            # 안전하게 검정 프레임 반환
            return av.VideoFrame.from_ndarray(np.zeros((480, 640, 3), dtype=np.uint8), format="bgr24")

        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        # 테스트 종료 후에는 프레임 처리 최소화
        if st.session_state.get("done", False):
            if self.mode == "test":
                return av.VideoFrame.from_ndarray(np.zeros_like(img), format="bgr24")
            else:
                return av.VideoFrame.from_ndarray(img, format="bgr24")

        # 시작 시간 기록
        if self.start_time is None:
            self.start_time = now
            st.session_state["start_time"] = now

        elapsed = now - self.start_time

        # 얼굴 분석 및 로그 기록
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        present, yaw, pitch, conf = False, 0.0, 0.0, 0.0
        rvec = tvec = None
        turning_now = False

        if res and getattr(res, "multi_face_landmarks", None):
            lms = res.multi_face_landmarks[0].landmark
            yaw, pitch, conf, rvec, tvec = headpose.estimate_headpose(lms, img.shape)
            if conf >= headpose.FACE_CONF_THR:
                present = True
                yaw, pitch = self.detector.smooth(yaw, pitch)
                if not self.detector.is_calibrated:
                    self.detector.calibrate(yaw, pitch)
                else:
                    turning_now = self.detector.detect_turn(yaw, pitch, now)
                # preview 모드에서만 오버레이
                if self.mode == "preview":
                    headpose.draw_landmarks_with_alert(img, res.multi_face_landmarks[0], self.detector.is_turning)
                    h, w = img.shape[:2]
                    K = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
                    headpose.draw_pose_axes(img, rvec, tvec, K)
        self.detector.detect_leave(present, now)
        self.detector.update_fps(now)

        # 로그 저장 (CSV 포맷)
        self.log.append([
            f"{elapsed:.3f}", int(present), f"{conf:.3f}", f"{yaw:.2f}", f"{pitch:.2f}",
            int(self.detector.is_turning), int(self.detector.is_missing), self.detector.turn_events, self.detector.leave_events
        ])

        # duration 경과 시 종료
        if elapsed >= self.duration:
            st.session_state["done"] = True

        # 모드별 반환
        if self.mode == "test":
            return av.VideoFrame.from_ndarray(np.zeros_like(img), format="bgr24")
        else:
            return av.VideoFrame.from_ndarray(img, format="bgr24")





st.set_page_config(page_title="ADHD Headpose Analyzer", layout="centered", initial_sidebar_state="collapsed")
st.title("ADHD Headpose Analyzer")
if params["mode"] == "test":
    st.info("테스트 모드: 영상 없이 분석만 진행됩니다.")
else:
    st.info("프리뷰 모드: 얼굴/포즈 오버레이가 표시됩니다.")
st.write(f"**Session ID:** {params['session_id']}  |  **Mode:** {params['mode']}  |  **Duration:** {params['duration']}s")
webrtc_streamer(
    key="headpose-analyzer",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)

# ===============================
# 6. 테스트 종료 후 CSV 저장 및 다운로드
# ===============================
def save_csv_if_needed():
    """테스트 종료 시 logs 폴더에 CSV 저장 및 다운로드 버튼 표시"""
    if st.session_state.get("done", False) and st.session_state.get("csv_path") is None:
        # logs 폴더 생성
        os.makedirs("logs", exist_ok=True)
        # 파일명 생성
        session_id = st.session_state.get("session_id", "demo")
        timestamp = int(time.time())
        csv_path = f"logs/session_{session_id}_{timestamp}.csv"
        # CSV 저장
        df = pd.DataFrame(st.session_state["log"], columns=[
            "time_s","face_present","face_conf","yaw_deg","pitch_deg",
            "turning","seat_missing","turn_events","seat_leave_events"
        ])
        df.to_csv(csv_path, index=False)
        st.session_state["csv_path"] = csv_path

    # 다운로드 버튼
    if st.session_state.get("csv_path"):
        with open(st.session_state["csv_path"], "rb") as f:
            st.download_button("CSV 다운로드", f, file_name=st.session_state["csv_path"], mime="text/csv")

        # (선택) 부모 iframe에 postMessage
        st.markdown(f"""
        <script>
        window.parent.postMessage({{
            type: "done",
            session_id: "{st.session_state.get('session_id', 'demo')}",
            csv_path: "{st.session_state['csv_path']}"
        }}, "*");
        </script>
        """, unsafe_allow_html=True)

# ===============================
# 6. 테스트 종료 후 CSV 저장 및 다운로드
# ===============================
def save_csv_if_needed():
    if st.session_state.get("done", False) and st.session_state.get("csv_path") is None:
        os.makedirs("logs", exist_ok=True)
        session_id = st.session_state.get("session_id", "demo")
        timestamp = int(time.time())
        csv_path = f"logs/session_{session_id}_{timestamp}.csv"
        df = pd.DataFrame(st.session_state["log"], columns=[
            "time_s","face_present","face_conf","yaw_deg","pitch_deg",
            "turning","seat_missing","turn_events","seat_leave_events"
        ])
        df.to_csv(csv_path, index=False)
        st.session_state["csv_path"] = csv_path
    if st.session_state.get("csv_path"):
        with open(st.session_state["csv_path"], "rb") as f:
            st.download_button("CSV 다운로드", f, file_name=st.session_state["csv_path"], mime="text/csv")
        st.markdown(f"""
        <script>
        window.parent.postMessage({{
            type: "done",
            session_id: "{st.session_state.get('session_id', 'demo')}",
            csv_path: "{st.session_state['csv_path']}"
        }}, "*");
        </script>
        """, unsafe_allow_html=True)

save_csv_if_needed()

# ===============================
# 7. 사용 가이드 (아래 출력)
# ===============================
if st.sidebar.button("사용 가이드 보기", key="guide_btn"):
        st.sidebar.markdown("""
### ADHD Headpose Streamlit 앱 사용법

**실행**
- 터미널에서:  
    `streamlit run app.py`

**모드별 접속 예시**
- 프리뷰(얼굴 오버레이):  
    [http://localhost:8501/?mode=preview&session_id=demo&yaw_thr=15&pitch_thr=8](http://localhost:8501/?mode=preview&session_id=demo&yaw_thr=15&pitch_thr=8)
- 테스트(분석만, 검정화면):  
    [http://localhost:8501/?mode=test&session_id=demo&duration=30&yaw_thr=18&pitch_thr=9](http://localhost:8501/?mode=test&session_id=demo&duration=30&yaw_thr=18&pitch_thr=9)

**동작 설명**
- preview: 웹캠 영상 + 얼굴/포즈 오버레이 실시간 표시
- test: 검정 화면만 보이고, 내부적으로 분석/로깅/CSV 저장
- 테스트 종료 후 logs/ 폴더에 CSV 자동 저장, 다운로드 버튼 제공
- iframe 연동 시 postMessage로 완료 알림 가능

**참고**
- 분석 엔진은 pose_core/headpose.py를 그대로 import하여 사용합니다.
- app.py만 수정/확장하면 됩니다.
""")

# ===============================
# 7. 사용 가이드 (아래 출력)
# ===============================
if st.sidebar.button("사용 가이드 보기"):
    st.sidebar.markdown("""
### ADHD Headpose Streamlit 앱 사용법

**실행**
- 터미널에서:  
  `streamlit run app.py`

**모드별 접속 예시**
- 프리뷰(얼굴 오버레이):  
  [http://localhost:8501/?mode=preview&session_id=demo&yaw_thr=15&pitch_thr=8](http://localhost:8501/?mode=preview&session_id=demo&yaw_thr=15&pitch_thr=8)
- 테스트(분석만, 검정화면):  
  [http://localhost:8501/?mode=test&session_id=demo&duration=30&yaw_thr=18&pitch_thr=9](http://localhost:8501/?mode=test&session_id=demo&duration=30&yaw_thr=18&pitch_thr=9)

**동작 설명**
- preview: 웹캠 영상 + 얼굴/포즈 오버레이 실시간 표시
- test: 검정 화면만 보이고, 내부적으로 분석/로깅/CSV 저장
- 테스트 종료 후 logs/ 폴더에 CSV 자동 저장, 다운로드 버튼 제공
- iframe 연동 시 postMessage로 완료 알림 가능

**참고**
- 분석 엔진은 pose_core/headpose.py를 그대로 import하여 사용합니다.
- app.py만 수정/확장하면 됩니다.
""")