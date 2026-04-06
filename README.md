# ADHD Prediction API

설문(Survey)과 인지 주의력 테스트(CAT) 데이터를 **Late Fusion** 방식으로 결합하여 ADHD 가능성을 예측하는 FastAPI 기반 **설명 가능한 AI(XAI)** 진단 솔루션입니다.

---

## 📹 동기

최근 ADHD에 대한 사회적 관심이 높아짐에 따라 다양한 진단 서비스가 등장하고 있지만, 기존 프로젝트들은 대부분 주관적인 '설문'에만 의존하는 수준에 머물러 있었습니다.

의료 및 심리 도메인에서 AI가 실질적인 가치를 가지려면 무엇보다 **'예측의 신뢰도'** 와 **'판단 근거'** 가 완벽해야 한다고 생각했습니다. 이에 단일 데이터에 의존할 때 발생하는 AI의 과신(Overconfidence) 오류를 구조적으로 제어하고, 주관적 데이터와 객관적 행동 데이터를 확률 수준에서 결합(Late Fusion)하여 **누구나 결과를 납득할 수 있는 '설명 가능한 AI(XAI)' 진단 솔루션** 을 직접 설계해 보고자 기획을 시작했습니다.

---

## 🧠 ML Model & Architecture

- 설문(Survey)과 행동(CAT) 데이터의 이질적인 신뢰도를 반영하기 위해, 두 지표를 독립적으로 예측한 후 결합하는 **'3단 분리형 Late Fusion' 융합 아키텍처 직접 설계 및 파이프라인 구축**
- 실시간 데이터의 이상치로 인한 확률 폭주(과신)를 막는 **'안정화 레이어'(`adjust_p_cat`)** 도입 및 하드코딩 인덱스 오류를 방지하는 **'동적 확률 추출(`model.classes_`)'** 로직을 구현하여 모델의 구조적 안정성 극대화
- 사용 스택: `scikit-learn`, `Logistic Regression`, `CalibratedClassifierCV`, `LightGBM`

### 3단 분리형 Late Fusion 구조

```
 ┌──────────────┐      ┌────────────────────┐
 │  Survey 20Q  │─────▶│  Survey Model      │──▶ p_survey
 │  (q1~q20)    │      │  (adhd_pipe_smote) │
 └──────────────┘      └────────────────────┘
                                │
                                ▼
 ┌──────────────┐      ┌────────────────────┐
 │  CAT raw     │─────▶│  CAT Model         │──▶ p_cat_raw ──▶ [Stabilizer] ──▶ p_cat_used
 │  (trials)    │      │  (late_fusion)     │      (cap / soft / none)
 └──────────────┘      └────────────────────┘
                                │
                                ▼
                       ┌────────────────────┐
                       │  Meta Fusion Model │──▶ p_final, label_final
                       │  (28 features)     │
                       └────────────────────┘
```

1. **Survey 모델** (`adhd_pipe_smote.pkl`) — 설문 기반 확률 `p_survey`
2. **CAT 모델** (`cat_late_fusion_model.pkl`) — CAT 성능 지표 27개 + `p_survey` 기반 확률 `p_cat`
3. **Fusion 메타 모델** (`meta_fusion_model.pkl`) — `p_survey`, 보정 `p_cat`, CAT 피처 26개를 결합한 최종 확률 `p_final`

---

## 📊 Data Processing & Feature Engineering

- CAT 행동 로그에서 단순 점수를 넘어 **타겟 누락 비율(omission), 허위 반응 비율(commission), 반응 속도 변동성(rt_sd)** 등 사용자의 '행동 패턴' 자체를 정량화하는 **고도화된 피처 엔지니어링(Feature Engineering)** 수행
- 학습 데이터의 통계적 분포(`describe`)를 기준으로 실시간 입력 데이터의 이상치(Outlier)를 사전에 검증·필터링하는 **견고한 데이터 전처리 파이프라인 구현**
- 사용 스택: `pandas`, `numpy`, `StandardScaler`

### CAT 블록별 피처 (simple / sustained / interference / divided)

| 피처 | 의미 |
|---|---|
| `omission` | 타겟인데 클릭 안 한 비율 |
| `commission` | 논타겟인데 클릭한 비율 |
| `rt_mean` | 정답 타겟의 반응시간 평균 (초) |
| `rt_sd` | 정답 타겟의 반응시간 표준편차 (행동 변동성) |
| `correct_rate` | 전체 정답률 |
| `aq_*` | `1 - w1·omission - w2·commission` 블록별 정확도 지수 |

Working Memory 블록은 `forward` / `backward` 정답률로 span을 산출합니다.

---

## ⚙️ Backend Framework & XAI Integration

- 복잡한 3단 융합 모델의 추론 과정을 프론트엔드와 실시간으로 연동하는 **FastAPI 기반의 빠르고 안정적인 API 서버 개발**
- 의료/심리 도메인의 필수 요소인 신뢰도 확보를 위해, 단계별 판단 근거(**설문 확률 → 원본 CAT 확률 → 보정 확률 → 최종 확률**)를 모두 투명하게 반환하는 **설명 가능한 AI(XAI) JSON 응답 구조 직접 기획 및 적용**
- 사용 스택: `FastAPI`, `Pydantic`, `Uvicorn`

### 안정화 레이어 (`adjust_p_cat`)

실시간 이상치로 인한 CAT 확률 폭주를 구조적으로 차단합니다.

| 모드 | 동작 |
|---|---|
| `none` | 보정 없이 그대로 사용 |
| `cap` *(기본)* | `P_CAT_CAP` 상한값으로 클리핑 (기본 0.85) |
| `soft` | 0.5 중심으로 `P_CAT_SOFT_SCALE` 스케일링하여 과신 완화 |

---

## 프로젝트 구조

```
AI/
├── ai_total.py           # FastAPI 엔트리포인트 (전체 파이프라인)
├── requirements.txt
└── models/
    ├── adhd_pipe_smote.pkl         # Survey 모델
    ├── cat_late_fusion_model.pkl   # CAT 모델
    └── meta_fusion_model.pkl       # Fusion 메타 모델
```

## 설치 및 실행

```bash
pip install -r requirements.txt
uvicorn ai_total:app --host 0.0.0.0 --port 8000
```

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `SURVEY_MODEL_PATH` | `models/adhd_pipe_smote.pkl` | Survey 모델 경로 |
| `CAT_MODEL_PATH` | `models/cat_late_fusion_model.pkl` | CAT 모델 경로 |
| `FUSION_MODEL_PATH` | `models/meta_fusion_model.pkl` | Fusion 모델 경로 |
| `SURVEY_FEATURE_MODE` | `sum` | 설문 피처 스케일 (`sum` / `0to100`) |
| `CAT_SCORE_MODE` | `mix` | CAT 점수화 방식 (`aq` / `correct_rate` / `mix`) |
| `P_CAT_ADJUST_MODE` | `cap` | 안정화 레이어 (`none` / `cap` / `soft`) |
| `P_CAT_CAP` | `0.85` | `cap` 모드 상한값 |
| `P_CAT_SOFT_SCALE` | `0.6` | `soft` 모드 스케일 |

---

## API

### `POST /predict`

**Request Body**
```json
{
  "user_info": { "user_id": "u-123", "age": 20, "gender": 1, "test_id": "t-1" },
  "survey": {
    "answers": { "q1": 3, "q2": 4, "q3": 2, "...": 0, "q20": 5 },
    "full4_iq": 105.0
  },
  "cat_raw": {
    "simple_trials":       [{ "trial_index": 0, "is_target": true, "clicked": true, "reaction_time_ms": 430 }],
    "sustained_trials":    [],
    "interference_trials": [],
    "divided_trials":      [],
    "wm_trials": [
      { "trial_index": 0, "type": "forward",  "presented": [1,2,3], "user_answer": [1,2,3], "correct": true },
      { "trial_index": 1, "type": "backward", "presented": [4,5,6], "user_answer": [6,5,4], "correct": true }
    ]
  }
}
```

- 설문 `answers`는 `q1`~`q20` 모두 필요하며 각 값은 **1~6** 범위
- `q1~q10` → **Inattentive**, `q11~q20` → **Hyper/Impulsive** 합산
- `gender`: 0=여, 1=남

**Response — XAI 투명 응답 구조**

```json
{
  "p_survey":    0.1234,
  "p_cat_raw":   0.9123,
  "p_cat_used":  0.85,
  "p_final":     0.6421,
  "label_final": 1,

  "survey_summary": {
    "mode_used_for_model": "sum",
    "inatt_sum": 35.0, "hyper_sum": 28.0,
    "inatt_0to100": 50.0, "hyper_0to100": 36.0
  },
  "cat_scores_100": {
    "simple": 82, "sustained": 74, "interference": 69,
    "divided": 71, "working_memory": 80, "mode": "mix"
  },
  "p_cat_adjust": { "mode": "cap", "cap": 0.85, "soft_scale": 0.6 }
}
```

단계별 판단 근거를 모두 노출하여 프론트에서 **"어떤 신호가 최종 라벨을 이끌었는가"** 를 사용자에게 설명할 수 있습니다.

### `GET /cat_scores_100`
프론트 프로토타이핑용 예시 CAT 점수 응답.

---

## CORS

기본 허용 오리진:
- `http://localhost:3000`
- `https://acts-front.vercel.app`

## 의존성

`fastapi`, `uvicorn`, `pydantic`, `numpy`, `pandas`, `joblib`, `scikit-learn`, `lightgbm` — 상세 버전은 `requirements.txt` 참조.
