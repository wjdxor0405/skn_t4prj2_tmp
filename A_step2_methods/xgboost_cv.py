"""
XGBoost 기반 K별 교차검증 성능 비교.
기획구현.md 3번 "2단계 -- 머신러닝 기반 표본충분성 검증 + 사후 통합":
  "K별로 XGBoost(또는 동일 계열 분류기)를 학습시켜 교차검증 성능 비교"

⚠️ 파일명 메모: 사용자가 "xgboost.py 같은 파일"이라 요청했으나, 패키지명
'xgboost'와 동일한 파일명을 쓰면 이 파일 내부의 `import xgboost`가 자기
자신(이 파일)을 다시 import하는 순환 참조를 일으켜 ImportError가 난다
(실제로 재현 후 확인됨). 따라서 역할을 그대로 살려 xgboost_cv.py로 명명한다.

이 모듈은 "분석 A 1단계가 제안한 변화점 후보가, 실제로 분류 성능에
기여하는가"를 교차검증으로 검증하는 역할만 한다. segment_merge.py의
사후 통합과는 별도 절차이며, run_step2.py(또는 main.py)에서 두 결과를
나란히 제시해 사람이 최종 K를 판단하게 한다 (기획구현.md 4번-②:
"임계값은 자동 탐지하지 않음 ... 사람이 종합 판단").

입력 피처 범위 (중요, 누수 방지)
--------------------------------
여기서 학습하는 XGBoost는 "세그먼트 경계 후보 자체가 쓸모 있는지"를
보는 것이 목적이지, 기획구현.md 5번의 최종 예측모델(1·2·3단계)이 아니다.
따라서 피처는 의도적으로 단순하게 "tenure 원본 + 세그먼트 라벨"만
사용한다. 다른 전체 속성(Contract, InternetService 등)을 섞으면
"세그먼트 라벨의 순수한 기여도"가 다른 변수 효과와 뒤섞여 해석이
어려워지기 때문이다 (5번 표의 1·2·3단계 비교 구조와는 다른, 더 좁은
범위의 검증임에 유의).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from segment_merge import boundaries_to_labels  # noqa: E402

_SCORING = ["roc_auc", "f1", "recall", "precision"]


@dataclass
class CVResult:
    """K(세그먼트 라벨 포함 여부에 따른) 1개 구조의 교차검증 결과."""

    n_segments: int
    boundaries_tenure: list
    mean_scores: dict  # {"roc_auc": 0.83, "f1": ..., ...}
    std_scores: dict
    n_folds: int


def _build_features(
    df_train: pd.DataFrame,
    boundaries_tenure: list[int],
    tenure_col: str = "tenure",
    use_segment_label: bool = True,
) -> pd.DataFrame:
    """
    tenure 원본 + (옵션) 세그먼트 라벨(원-핫)로 X를 만든다.
    boundaries_tenure가 빈 리스트면 세그먼트가 1개뿐이라는 뜻이므로
    use_segment_label을 True로 둬도 라벨 컬럼이 상수가 되어 자동으로
    무의미해진다(트리 분류기가 알아서 무시).
    """
    X = pd.DataFrame({"tenure": df_train[tenure_col].astype(float).values})
    if use_segment_label and len(boundaries_tenure) > 0:
        labels = boundaries_to_labels(df_train[tenure_col], boundaries_tenure)
        dummies = pd.get_dummies(labels, prefix="segment")
        X = pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    return X


def cross_validate_segment_structure(
    df_train: pd.DataFrame,
    boundaries_tenure: list[int],
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
    n_splits: int = 5,
    random_state: int = 42,
) -> CVResult:
    """
    주어진 경계(boundaries_tenure)로 만든 세그먼트 라벨을 피처에 추가했을 때
    XGBoost 분류 성능을 계층화 K-fold 교차검증으로 측정한다.

    boundaries_tenure=[] 이면 "세그먼트 라벨 없음"(tenure 원본만 사용)
    구조와 동일해지므로, K=1(분할 없음) 베이스라인도 같은 함수로 잴 수 있다.
    """
    y = df_train[churn_col].map({"Yes": 1, "No": 0})
    if y.isna().any():
        y = pd.to_numeric(df_train[churn_col], errors="coerce")
    X = _build_features(df_train, boundaries_tenure, tenure_col, use_segment_label=True)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores = cross_validate(
        model, X, y, cv=cv, scoring=_SCORING, n_jobs=-1, error_score="raise"
    )

    mean_scores = {m: float(np.mean(scores[f"test_{m}"])) for m in _SCORING}
    std_scores = {m: float(np.std(scores[f"test_{m}"])) for m in _SCORING}

    n_segments = len(boundaries_tenure) + 1
    return CVResult(
        n_segments=n_segments,
        boundaries_tenure=sorted(boundaries_tenure),
        mean_scores=mean_scores,
        std_scores=std_scores,
        n_folds=n_splits,
    )


def compare_candidates(
    df_train: pd.DataFrame,
    boundary_candidates: list[list[int]],
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
    n_splits: int = 5,
    random_state: int = 42,
) -> list[CVResult]:
    """
    여러 경계 후보(분석 A 1단계의 K=2~8 후보들)에 대해 일괄적으로
    교차검증을 수행한다. K=1(세그먼트 없음) 베이스라인도 자동으로
    맨 앞에 포함시켜, "세그먼트 라벨이 없을 때 대비 성능이 실제로
    오르는가"를 바로 비교할 수 있게 한다.
    """
    results = []

    # K=1 베이스라인 (세그먼트 라벨 없음)
    baseline = cross_validate_segment_structure(
        df_train, [], tenure_col, churn_col, n_splits, random_state
    )
    results.append(baseline)

    for boundaries in boundary_candidates:
        r = cross_validate_segment_structure(
            df_train, boundaries, tenure_col, churn_col, n_splits, random_state
        )
        results.append(r)

    return results


def results_to_table(results: list[CVResult]) -> pd.DataFrame:
    """CVResult 목록을 사람이 읽기 좋은 표(DataFrame)로 변환."""
    rows = []
    for r in results:
        row = {"n_segments": r.n_segments, "boundaries_tenure": r.boundaries_tenure}
        for metric in _SCORING:
            row[f"{metric}_mean"] = round(r.mean_scores[metric], 4)
            row[f"{metric}_std"] = round(r.std_scores[metric], 4)
        rows.append(row)
    return pd.DataFrame(rows)
