from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import os

# =========================================================
# 0) 모델 경로
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
# 1) 설정(여기 두 개가 제일 중요)
# =========================================================
# 설문 모델이 학습한 입력 스케일과 동일하게 맞춰야 함
# - "sum": (각 요인 합계) => 보통 10문항이면 10~60, 20문항이면 10~60/또는 10~60 각각
# - "0to100": 0~100 정규화 점수
SURVEY_FEATURE_MODE = os.getenv("SURVEY_FEATURE_MODE", "sum")  # "sum" or "0to100"

# CAT 점수 산출 방식 (프론트 표시용)
CAT_SCORE_MODE = os.getenv("CAT_SCORE_MODE", "mix")  # "aq" / "correct_rate" / "mix"

# =========================================================
# 2) FastAPI
# =========================================================
app = FastAPI(title="ADHD Prediction API")

# 고정된 CAT 점수 반환용 GET 엔드포인트
@app.get("/cat_scores_100")
def get_cat_scores_100():
    return {
        "cat_scores_100": {
            "simple": 26,
            "sustained": 40,
            "interference": 24,
            "divided": 25,
            "working_memory": 67
        }
    }

# CORS 설정
origins = [
    "http://localhost:3000/",
    "https://acts-front.vercel.app/"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODELS: Dict[str, Any] = {}

@app.on_event("startup")
def load_models():
    try:
        MODELS["survey"] = safe_load_model(SURVEY_MODEL_PATH)
        MODELS["cat"] = safe_load_model(CAT_MODEL_PATH)
        MODELS["fusion"] = safe_load_model(FUSION_MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"모델 로드 실패: {type(e).__name__}: {e}")

# =========================================================
# 3) 입력 스키마
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
# 4) 모델 입력 컬럼 (학습 순서대로)
# =========================================================
SURVEY_COLS = ["Age", "Gender", "Full4 IQ", "Inattentive", "Hyper/Impulsive"]

CAT_COLS = [
    "simple_sel_omission", "simple_sel_commission", "simple_sel_rt_mean", "simple_sel_rt_sd", "simple_sel_correct_rate",
    "sustained_omission", "sustained_commission", "sustained_rt_mean", "sustained_rt_sd", "sustained_correct_rate",
    "interference_omission", "interference_commission", "interference_rt_mean", "interference_rt_sd", "interference_correct_rate",
    "divided_omission", "divided_commission", "divided_rt_mean", "divided_rt_sd", "divided_correct_rate",
    "wm_forward_span", "wm_backward_span",
    "aq_simple_sel", "aq_sustained", "aq_interference", "aq_divided",
    "p_survey"
]

FUSION_COLS = [
    "p_survey", "p_cat",
    "simple_sel_omission", "simple_sel_commission", "simple_sel_rt_mean", "simple_sel_rt_sd", "simple_sel_correct_rate",
    "sustained_omission", "sustained_commission", "sustained_rt_mean", "sustained_rt_sd", "sustained_correct_rate",
    "interference_omission", "interference_commission", "interference_rt_mean", "interference_rt_sd", "interference_correct_rate",
    "divided_omission", "divided_commission", "divided_rt_mean", "divided_rt_sd", "divided_correct_rate",
    "wm_forward_span", "wm_backward_span",
    "aq_simple_sel", "aq_sustained", "aq_interference", "aq_divided"
]

# =========================================================
# 5) 유틸: 설문 검증 + 점수화(20문항 -> 요약)
# =========================================================
def validate_survey_answers(answers: Dict[str, int]) -> None:
    missing = [f"q{i}" for i in range(1, 21) if f"q{i}" not in answers]
    if missing:
        raise HTTPException(status_code=400, detail=f"설문 문항 누락: {missing}")
    bad = {k: v for k, v in answers.items() if not (1 <= int(v) <= 6)}
    if bad:
        raise HTTPException(status_code=400, detail=f"설문 점수 범위 오류(1~6): {bad}")

def to_0_100_from_sum(raw_sum: float, n_items: int, min_scale=1, max_scale=6) -> float:
    min_sum = n_items * min_scale
    max_sum = n_items * max_scale
    return float(np.clip((raw_sum - min_sum) / (max_sum - min_sum) * 100.0, 0.0, 100.0))

def compute_survey_features(answers: Dict[str, int]) -> Dict[str, float]:
    """
    프론트 q1~q20 -> (inatt, hyper)을 만들어 SURVEY_FEATURE_MODE에 맞춰 반환
    """
    # 그룹핑: (현재 FastAPI 로직 기준) q1~q10 부주의, q11~q20 과잉/충동
    inatt_vals = [float(answers[f"q{i}"]) for i in range(1, 11)]
    hyper_vals = [float(answers[f"q{i}"]) for i in range(11, 21)]

    inatt_sum = float(np.sum(inatt_vals))   # 10~60
    hyper_sum = float(np.sum(hyper_vals))   # 10~60

    inatt_100 = to_0_100_from_sum(inatt_sum, n_items=10)
    hyper_100 = to_0_100_from_sum(hyper_sum, n_items=10)

    if SURVEY_FEATURE_MODE == "0to100":
        inatt_feat, hyper_feat = inatt_100, hyper_100
    else:
        inatt_feat, hyper_feat = inatt_sum, hyper_sum

    return {
        "inatt_sum": inatt_sum,
        "hyper_sum": hyper_sum,
        "inatt_0to100": inatt_100,
        "hyper_0to100": hyper_100,
        "inatt_feat": inatt_feat,
        "hyper_feat": hyper_feat,
    }

# =========================================================
# 6) CAT 연산
# =========================================================
def compute_block_features(trials: List[Trial]):
    total = len(trials)
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    target_trials = [t for t in trials if t.is_target]
    nontarget_trials = [t for t in trials if not t.is_target]

    omission = sum(1 for t in target_trials if not t.clicked) / max(len(target_trials), 1)
    commission = sum(1 for t in nontarget_trials if t.clicked) / max(len(nontarget_trials), 1)

    correct = 0
    rts = []
    for t in trials:
        if t.is_target and t.clicked:
            correct += 1
            if t.reaction_time_ms is not None:
                rts.append(t.reaction_time_ms / 1000.0)
        elif (not t.is_target) and (not t.clicked):
            correct += 1

    correct_rate = correct / total
    rt_mean = float(np.mean(rts)) if rts else 0.0
    rt_sd = float(np.std(rts)) if rts else 0.0

    omission = float(np.clip(omission, 0, 1))
    commission = float(np.clip(commission, 0, 1))
    correct_rate = float(np.clip(correct_rate, 0, 1))
    rt_mean = max(0.0, rt_mean)
    rt_sd = max(0.0, rt_sd)
    return omission, commission, rt_mean, rt_sd, correct_rate

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

def compute_aq(om: float, com: float, w1: float, w2: float) -> float:
    v = 1 - w1 * om - w2 * com
    return float(np.clip(v, 0, 1))

# CAT 점수(표시용) 0~100
def score_100(aq: float, correct_rate: float, mode: str) -> int:
    if mode == "aq":
        return int(round(aq * 100))
    if mode == "correct_rate":
        return int(round(correct_rate * 100))
    # mix: AQ 70% + 정답률 30% (표시용 임의 가중)
    v = 0.7 * aq + 0.3 * correct_rate
    return int(round(float(np.clip(v, 0, 1)) * 100))

def wm_score_100(wm_f: float, wm_b: float) -> int:
    v = 0.5 * wm_f + 0.5 * wm_b
    return int(round(float(np.clip(v, 0, 1)) * 100))

# =========================================================
# 7) 엔드포인트
# =========================================================
@app.post("/predict")
def predict(payload: RequestPayload):
    try:
        survey_model = MODELS["survey"]
        cat_model = MODELS["cat"]
        fusion_model = MODELS["fusion"]

        # A) Survey -> p_survey
        answers = payload.survey.answers
        validate_survey_answers(answers)

        sf = compute_survey_features(answers)

        survey_row = {
            "Age": payload.user_info.age,
            "Gender": payload.user_info.gender,
            "Full4 IQ": payload.survey.full4_iq,
            "Inattentive": sf["inatt_feat"],
            "Hyper/Impulsive": sf["hyper_feat"],
        }
        survey_X = pd.DataFrame([survey_row], columns=SURVEY_COLS)
        p_survey = float(survey_model.predict_proba(survey_X)[0][1])

        # B) CAT -> features + p_cat
        s_om, s_com, s_rt_m, s_rt_s, s_cr = compute_block_features(payload.cat_raw.simple_trials)
        su_om, su_com, su_rt_m, su_rt_s, su_cr = compute_block_features(payload.cat_raw.sustained_trials)
        i_om, i_com, i_rt_m, i_rt_s, i_cr = compute_block_features(payload.cat_raw.interference_trials)
        d_om, d_com, d_rt_m, d_rt_s, d_cr = compute_block_features(payload.cat_raw.divided_trials)
        wm_f, wm_b = compute_wm_spans(payload.cat_raw.wm_trials)

        aq_simple = compute_aq(s_om,  s_com,  0.8686213105, 0.8257700590)
        aq_sust   = compute_aq(su_om, su_com, 0.8588605831, 0.8421784891)
        aq_inter  = compute_aq(i_om,  i_com,  0.8519097689, 0.8812066974)
        aq_div    = compute_aq(d_om,  d_com,  0.8752559235, 0.8557618929)

        # 표시용 점수
        cat_scores = {
            "simple": score_100(aq_simple, s_cr, CAT_SCORE_MODE),
            "sustained": score_100(aq_sust, su_cr, CAT_SCORE_MODE),
            "interference": score_100(aq_inter, i_cr, CAT_SCORE_MODE),
            "divided": score_100(aq_div, d_cr, CAT_SCORE_MODE),
            "working_memory": wm_score_100(wm_f, wm_b),
            "mode": CAT_SCORE_MODE,
        }

        cat_row = {
            "simple_sel_omission": s_om, "simple_sel_commission": s_com, "simple_sel_rt_mean": s_rt_m, "simple_sel_rt_sd": s_rt_s, "simple_sel_correct_rate": s_cr,
            "sustained_omission": su_om, "sustained_commission": su_com, "sustained_rt_mean": su_rt_m, "sustained_rt_sd": su_rt_s, "sustained_correct_rate": su_cr,
            "interference_omission": i_om, "interference_commission": i_com, "interference_rt_mean": i_rt_m, "interference_rt_sd": i_rt_s, "interference_correct_rate": i_cr,
            "divided_omission": d_om, "divided_commission": d_com, "divided_rt_mean": d_rt_m, "divided_rt_sd": d_rt_s, "divided_correct_rate": d_cr,
            "wm_forward_span": wm_f, "wm_backward_span": wm_b,
            "aq_simple_sel": aq_simple, "aq_sustained": aq_sust, "aq_interference": aq_inter, "aq_divided": aq_div,
            "p_survey": p_survey
        }
        cat_X = pd.DataFrame([cat_row], columns=CAT_COLS)
        p_cat = float(cat_model.predict_proba(cat_X)[0][1])

        # C) Fusion -> p_final
        fusion_row = {
            "p_survey": p_survey, "p_cat": p_cat,
            "simple_sel_omission": s_om, "simple_sel_commission": s_com, "simple_sel_rt_mean": s_rt_m, "simple_sel_rt_sd": s_rt_s, "simple_sel_correct_rate": s_cr,
            "sustained_omission": su_om, "sustained_commission": su_com, "sustained_rt_mean": su_rt_m, "sustained_rt_sd": su_rt_s, "sustained_correct_rate": su_cr,
            "interference_omission": i_om, "interference_commission": i_com, "interference_rt_mean": i_rt_m, "interference_rt_sd": i_rt_s, "interference_correct_rate": i_cr,
            "divided_omission": d_om, "divided_commission": d_com, "divided_rt_mean": d_rt_m, "divided_rt_sd": d_rt_s, "divided_correct_rate": d_cr,
            "wm_forward_span": wm_f, "wm_backward_span": wm_b,
            "aq_simple_sel": aq_simple, "aq_sustained": aq_sust, "aq_interference": aq_inter, "aq_divided": aq_div
        }
        fusion_X = pd.DataFrame([fusion_row], columns=FUSION_COLS)
        if fusion_X.shape[1] != 28:
            raise HTTPException(status_code=500, detail=f"fusion_X 컬럼 수가 28이 아님: {fusion_X.shape}")

        p_final = float(fusion_model.predict_proba(fusion_X)[0][1])
        label_final = int(p_final >= 0.5)

        return {
            "p_survey": round(p_survey, 4),
            "p_cat": round(p_cat, 4),
            "p_final": round(p_final, 4),
            "label_final": label_final,

            # 설문 요약(프론트가 표시 가능)
            "survey_summary": {
                "mode_used_for_model": SURVEY_FEATURE_MODE,
                "inatt_sum": round(sf["inatt_sum"], 3),
                "hyper_sum": round(sf["hyper_sum"], 3),
                "inatt_0to100": round(sf["inatt_0to100"], 3),
                "hyper_0to100": round(sf["hyper_0to100"], 3),
            },

            # CAT 점수(0~100)
            "cat_scores_100": cat_scores,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {type(e).__name__}: {e}")
