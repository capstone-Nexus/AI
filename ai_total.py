from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import os


# =========================================================
# 0) 모델 경로 (환경변수로 덮어쓰기 가능)
# =========================================================


BASE_DIR = Path(__file__).resolve().parent

SURVEY_MODEL_PATH = os.getenv("SURVEY_MODEL_PATH", str(BASE_DIR / "models" / "adhd_pipe_smote.pkl"))
CAT_MODEL_PATH    = os.getenv("CAT_MODEL_PATH",    str(BASE_DIR / "models" / "cat_late_fusion_model.pkl"))
FUSION_MODEL_PATH = os.getenv("FUSION_MODEL_PATH", str(BASE_DIR / "models" / "meta_fusion_model.pkl"))


def safe_load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"모델 파일이 존재하지 않음: {path}")
    return joblib.load(path)


# =========================================================
# 1) FastAPI 앱 + 모델(Startup에서 로드)
#    - import 단계에서 터지지 않게 startup에서 로드하는 형태 권장
# =========================================================
app = FastAPI(title="ADHD Prediction API")

MODELS = {}


@app.on_event("startup")
def load_models():
    try:
        MODELS["survey"] = safe_load_model(SURVEY_MODEL_PATH)
        MODELS["cat"] = safe_load_model(CAT_MODEL_PATH)
        MODELS["fusion"] = safe_load_model(FUSION_MODEL_PATH)
    except Exception as e:
        # 서버 부팅 로그에서 원인 확인 가능
        raise RuntimeError(f"모델 로드 실패: {type(e).__name__}: {e}")


# =========================================================
# 2) 입력 스키마 (프론트 Raw JSON)
# =========================================================
class Trial(BaseModel):
    trial_index: int
    is_target: bool
    clicked: bool
    reaction_time_ms: Optional[float] = None


class WMTrial(BaseModel):
    trial_index: int
    type: str  # "forward" / "backward"
    presented: List[int]
    user_answer: List[int]
    correct: bool


class Survey(BaseModel):
    answers: Dict[str, int] = Field(..., description="q1~q20 각 1~6 점수")
    full4_iq: float


class UserInfo(BaseModel):
    user_id: Optional[str] = None
    age: int
    gender: int  # 0=여, 1=남
    test_id: Optional[str] = None


class CatRaw(BaseModel):
    simple_trials: List[Trial]
    sustained_trials: List[Trial]
    interference_trials: List[Trial]
    divided_trials: List[Trial]
    wm_trials: List[WMTrial]


class RequestPayload(BaseModel):
    user_info: UserInfo
    survey: Survey
    cat_raw: CatRaw


# =========================================================
# 3) 컬럼 순서(학습 순서와 동일해야 함)
#    - Survey model feature 순서(사용자 제공)
# =========================================================
SURVEY_COLS = ["Age", "Gender", "Full4 IQ", "Inattentive", "Hyper/Impulsive"]

# CAT 모델 입력(사용자 제공: 27개, 마지막이 p_survey)
CAT_COLS = [
    "simple_sel_omission", "simple_sel_commission", "simple_sel_rt_mean", "simple_sel_rt_sd", "simple_sel_correct_rate",
    "sustained_omission", "sustained_commission", "sustained_rt_mean", "sustained_rt_sd", "sustained_correct_rate",
    "interference_omission", "interference_commission", "interference_rt_mean", "interference_rt_sd", "interference_correct_rate",
    "divided_omission", "divided_commission", "divided_rt_mean", "divided_rt_sd", "divided_correct_rate",
    "wm_forward_span", "wm_backward_span",
    "aq_simple_sel", "aq_sustained", "aq_interference", "aq_divided",
    "p_survey"
]

# ✅ fusion_model 입력(28개)
# - 에러 메시지로 "28개 기대"가 확정이라서, 28개를 여기서 만들어 넣어야 함
FUSION_COLS = [
    "p_survey",
    "p_cat",
    "simple_sel_omission", "simple_sel_commission", "simple_sel_rt_mean", "simple_sel_rt_sd", "simple_sel_correct_rate",
    "sustained_omission", "sustained_commission", "sustained_rt_mean", "sustained_rt_sd", "sustained_correct_rate",
    "interference_omission", "interference_commission", "interference_rt_mean", "interference_rt_sd", "interference_correct_rate",
    "divided_omission", "divided_commission", "divided_rt_mean", "divided_rt_sd", "divided_correct_rate",
    "wm_forward_span", "wm_backward_span",
    "aq_simple_sel", "aq_sustained", "aq_interference", "aq_divided"
]


# =========================================================
# 4) 입력 검증(설문 누락 방지)
# =========================================================
def validate_survey_answers(answers: Dict[str, int]) -> None:
    missing = [f"q{i}" for i in range(1, 21) if f"q{i}" not in answers]
    if missing:
        raise HTTPException(status_code=400, detail=f"설문 문항 누락: {missing}")

    bad = {k: v for k, v in answers.items() if not (1 <= int(v) <= 6)}
    if bad:
        raise HTTPException(status_code=400, detail=f"설문 점수 범위 오류(1~6): {bad}")


# =========================================================
# 5) CAT 연산
#   - omission: 타겟인데 클릭X
#   - commission: 노타겟인데 클릭O  (노타겟 클릭은 오답 처리)
#   - correct_rate: (타겟+클릭) 또는 (노타겟+미클릭)만 정답
#   - RT: 타겟+클릭인 trial의 reaction_time만 사용, ms -> sec
# =========================================================
def compute_block_features(trials: List[Trial]):
    total = len(trials)
    if total == 0:
        # 데이터가 비정상인 경우, 400으로 막고 싶으면 여기서 HTTPException으로 바꾸면 됨
        return 0.0, 0.0, 0.0, 0.0, 0.0

    target_trials = [t for t in trials if t.is_target]
    nontarget_trials = [t for t in trials if not t.is_target]

    omission = sum(1 for t in target_trials if not t.clicked) / max(len(target_trials), 1)
    commission = sum(1 for t in nontarget_trials if t.clicked) / max(len(nontarget_trials), 1)

    correct = 0
    rts = []

    for t in trials:
        # 정답: 타겟 + 클릭
        if t.is_target and t.clicked:
            correct += 1
            if t.reaction_time_ms is not None:
                rts.append(t.reaction_time_ms / 1000.0)  # sec
        # 정답: 노타겟 + 미클릭
        elif (not t.is_target) and (not t.clicked):
            correct += 1
        # 노타겟 클릭은 오답(정답 처리 안 함)

    correct_rate = correct / total
    rt_mean = float(np.mean(rts)) if rts else 0.0
    rt_sd = float(np.std(rts)) if rts else 0.0

    # clip/정리
    omission = float(np.clip(omission, 0, 1))
    commission = float(np.clip(commission, 0, 1))
    correct_rate = float(np.clip(correct_rate, 0, 1))
    rt_mean = max(0.0, rt_mean)
    rt_sd = max(0.0, rt_sd)

    return omission, commission, rt_mean, rt_sd, correct_rate


# =========================================================
# 6) WM span (0~1 정답률)
# =========================================================
def compute_wm_spans(wm_trials: List[WMTrial]):
    forward = [t for t in wm_trials if t.type == "forward"]
    backward = [t for t in wm_trials if t.type == "backward"]

    def acc(trials):
        if not trials:
            return 0.0
        return sum(1 for t in trials if t.correct) / len(trials)

    wm_forward = float(np.clip(acc(forward), 0, 1))
    wm_backward = float(np.clip(acc(backward), 0, 1)) if backward else wm_forward
    return wm_forward, wm_backward


# =========================================================
# 7) AQ 공식 (사용자 제공 그대로)
# =========================================================
def compute_aq(om: float, com: float, w1: float, w2: float) -> float:
    v = 1 - w1 * om - w2 * com
    return float(np.clip(v, 0, 1))


# =========================================================
# 8) 상태 확인
# =========================================================
@app.get("/health")
def health():
    survey = MODELS.get("survey")
    cat = MODELS.get("cat")
    fusion = MODELS.get("fusion")

    return {
        "status": "ok",
        "survey_model_loaded": survey is not None,
        "cat_model_loaded": cat is not None,
        "fusion_model_loaded": fusion is not None,
        "survey_n_features": getattr(survey, "n_features_in_", None),
        "cat_n_features": getattr(cat, "n_features_in_", None),
        "fusion_n_features": getattr(fusion, "n_features_in_", None),
    }


# =========================================================
# 9) 예측 엔드포인트
# =========================================================
@app.post("/predict")
def predict(payload: RequestPayload):
    try:
        survey_model = MODELS["survey"]
        cat_model = MODELS["cat"]
        fusion_model = MODELS["fusion"]

        # -----------------------------
        # A) Survey -> p_survey
        # -----------------------------
        answers = payload.survey.answers
        validate_survey_answers(answers)

        # 현재 합의: q1~q10 부주의, q11~q20 충동/억제
        inatt = float(np.mean([answers[f"q{i}"] for i in range(1, 11)]))
        hyper = float(np.mean([answers[f"q{i}"] for i in range(11, 21)]))

        survey_row = {
            "Age": payload.user_info.age,
            "Gender": payload.user_info.gender,
            "Full4 IQ": payload.survey.full4_iq,
            "Inattentive": inatt,
            "Hyper/Impulsive": hyper
        }
        survey_X = pd.DataFrame([survey_row], columns=SURVEY_COLS)
        p_survey = float(survey_model.predict_proba(survey_X)[0][1])

        # -----------------------------
        # B) CAT trial logs -> 27개 요약 + p_cat
        # -----------------------------
        s_om, s_com, s_rt_m, s_rt_s, s_cr = compute_block_features(payload.cat_raw.simple_trials)
        su_om, su_com, su_rt_m, su_rt_s, su_cr = compute_block_features(payload.cat_raw.sustained_trials)
        i_om, i_com, i_rt_m, i_rt_s, i_cr = compute_block_features(payload.cat_raw.interference_trials)
        d_om, d_com, d_rt_m, d_rt_s, d_cr = compute_block_features(payload.cat_raw.divided_trials)

        wm_f, wm_b = compute_wm_spans(payload.cat_raw.wm_trials)

        aq_simple = compute_aq(s_om,  s_com,  0.8686213105, 0.8257700590)
        aq_sust   = compute_aq(su_om, su_com, 0.8588605831, 0.8421784891)
        aq_inter  = compute_aq(i_om,  i_com,  0.8519097689, 0.8812066974)
        aq_div    = compute_aq(d_om,  d_com,  0.8752559235, 0.8557618929)

        cat_row = {
            "simple_sel_omission": s_om,
            "simple_sel_commission": s_com,
            "simple_sel_rt_mean": s_rt_m,
            "simple_sel_rt_sd": s_rt_s,
            "simple_sel_correct_rate": s_cr,

            "sustained_omission": su_om,
            "sustained_commission": su_com,
            "sustained_rt_mean": su_rt_m,
            "sustained_rt_sd": su_rt_s,
            "sustained_correct_rate": su_cr,

            "interference_omission": i_om,
            "interference_commission": i_com,
            "interference_rt_mean": i_rt_m,
            "interference_rt_sd": i_rt_s,
            "interference_correct_rate": i_cr,

            "divided_omission": d_om,
            "divided_commission": d_com,
            "divided_rt_mean": d_rt_m,
            "divided_rt_sd": d_rt_s,
            "divided_correct_rate": d_cr,

            "wm_forward_span": wm_f,
            "wm_backward_span": wm_b,

            "aq_simple_sel": aq_simple,
            "aq_sustained": aq_sust,
            "aq_interference": aq_inter,
            "aq_divided": aq_div,

            "p_survey": p_survey
        }

        cat_X = pd.DataFrame([cat_row], columns=CAT_COLS)
        p_cat = float(cat_model.predict_proba(cat_X)[0][1])

        # -----------------------------
        # C) Fusion model -> p_final (✅ 28개 입력)
        # -----------------------------
        fusion_row = {
            "p_survey": p_survey,
            "p_cat": p_cat,

            "simple_sel_omission": s_om,
            "simple_sel_commission": s_com,
            "simple_sel_rt_mean": s_rt_m,
            "simple_sel_rt_sd": s_rt_s,
            "simple_sel_correct_rate": s_cr,

            "sustained_omission": su_om,
            "sustained_commission": su_com,
            "sustained_rt_mean": su_rt_m,
            "sustained_rt_sd": su_rt_s,
            "sustained_correct_rate": su_cr,

            "interference_omission": i_om,
            "interference_commission": i_com,
            "interference_rt_mean": i_rt_m,
            "interference_rt_sd": i_rt_s,
            "interference_correct_rate": i_cr,

            "divided_omission": d_om,
            "divided_commission": d_com,
            "divided_rt_mean": d_rt_m,
            "divided_rt_sd": d_rt_s,
            "divided_correct_rate": d_cr,

            "wm_forward_span": wm_f,
            "wm_backward_span": wm_b,

            "aq_simple_sel": aq_simple,
            "aq_sustained": aq_sust,
            "aq_interference": aq_inter,
            "aq_divided": aq_div
        }

        fusion_X = pd.DataFrame([fusion_row], columns=FUSION_COLS)

        # 안전 체크: (1, 28) 보장
        if fusion_X.shape[1] != 28:
            raise HTTPException(status_code=500, detail=f"fusion_X 컬럼 수가 28이 아님: {fusion_X.shape}")

        p_final = float(fusion_model.predict_proba(fusion_X)[0][1])
        label_final = int(p_final >= 0.5)

        return {
            "p_survey": round(p_survey, 4),
            "p_cat": round(p_cat, 4),
            "p_final": round(p_final, 4),
            "label_final": label_final
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {type(e).__name__}: {e}")
