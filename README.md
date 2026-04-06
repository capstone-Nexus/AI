# ADHD Prediction API

FastAPI 기반 ADHD 예측 서버. 설문(ASRS 계열 20문항)과 CAT(인지 주의력 테스트) 원시 데이터를 받아 **Late Fusion** 방식으로 최종 ADHD 확률을 산출합니다.

## 개요

3개의 사전학습 모델을 조합하여 예측을 수행합니다.

1. **Survey 모델** (`adhd_pipe_smote.pkl`) — 설문 기반 확률 `p_survey`
2. **CAT 모델** (`cat_late_fusion_model.pkl`) — CAT 성능 지표 + `p_survey` 기반 확률 `p_cat`
3. **Fusion 메타 모델** (`meta_fusion_model.pkl`) — `p_survey`, `p_cat`, CAT 피처 28개를 결합한 최종 확률 `p_final`

## 프로젝트 구조

```
AI/
├── ai_total.py           # FastAPI 엔트리포인트
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
| `P_CAT_ADJUST_MODE` | `cap` | CAT 과신 완화 (`none` / `cap` / `soft`) |
| `P_CAT_CAP` | `0.85` | `cap` 모드 상한값 |
| `P_CAT_SOFT_SCALE` | `0.6` | `soft` 모드 스케일 (0.5 중심 수렴) |

## API

### `GET /cat_scores_100`
예시 CAT 점수 응답(더미).

### `POST /predict`
ADHD 예측 수행.

**Request Body**
```json
{
  "user_info": { "user_id": "string", "age": 20, "gender": 1, "test_id": "string" },
  "survey": {
    "answers": { "q1": 3, "q2": 4, "...": 0, "q20": 2 },
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

- 설문 `answers`는 `q1`~`q20` 모두 필요하며 각 값은 1~6 범위.
- `q1~q10` → Inattentive, `q11~q20` → Hyper/Impulsive 합산.
- `gender`: 0=여, 1=남.

**Response**
```json
{
  "p_survey": 0.1234,
  "p_cat_raw": 0.9123,
  "p_cat_used": 0.85,
  "p_final": 0.6421,
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

## 파이프라인 상세

### 1) Survey → `p_survey`
입력 피처: `Age`, `Gender`, `Full4 IQ`, `Inattentive`, `Hyper/Impulsive`.

### 2) CAT 블록 피처 계산
각 블록(`simple` / `sustained` / `interference` / `divided`)마다:
- **omission**: 타겟인데 클릭 안 한 비율
- **commission**: 논타겟인데 클릭한 비율
- **rt_mean / rt_sd**: 정답 타겟 반응시간 평균 / 표준편차(초)
- **correct_rate**: 전체 정답률
- **AQ**: `1 - w1*omission - w2*commission` (블록별 고정 가중치)

Working Memory는 `forward` / `backward` 정답률로 span 산출.

### 3) CAT → `p_cat`
CAT 피처 27개 + `p_survey`를 모델에 투입하여 `p_cat_raw` 산출 후 `P_CAT_ADJUST_MODE`에 따라 보정:
- `none`: 그대로 사용
- `cap`: `P_CAT_CAP` 상한 적용 (기본)
- `soft`: 0.5 중심으로 `P_CAT_SOFT_SCALE` 스케일링

### 4) Fusion → `p_final`
`p_survey`, 보정된 `p_cat`, CAT 피처 26개(총 28 컬럼)를 메타 모델에 투입하여 최종 확률과 라벨(`>= 0.5`)을 반환합니다.

## CORS

기본 허용 오리진:
- `http://localhost:3000`
- `https://acts-front.vercel.app`

## 의존성

`fastapi`, `uvicorn`, `pydantic`, `numpy`, `pandas`, `joblib`, `scikit-learn`, `lightgbm` — 상세 버전은 `requirements.txt` 참조.
