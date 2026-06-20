"""
전원 머신러닝(All-ML) 경로 -- 3단계: 정확도 검증 (StratifiedKFold CV).

2단계(step2_tree.py)가 찾은 경계를 세그먼트 라벨로 변환해 입력 피처에
추가하고, 분류기를 StratifiedKFold 교차검증으로 학습/평가한다. 통계
공식(p-value, 검정력 분석 등)을 전혀 쓰지 않고, "다른 데이터 묶음에서도
일관되게 잘 맞추는가"만을 ROC-AUC / F1-score로 직접 측정한다.

기존 경로(A_step2_methods/xgboost_cv.py)와 평가 방식(StratifiedKFold +
ROC-AUC/F1)은 같지만, 이 모듈은 "전원 머신러닝 경로" 전용으로 별도
유지한다 -- 입력 경계의 출처(통계 PELT vs 머신러닝 트리)가 다르므로
결과를 섞지 않고 독립적으로 비교 가능하게 두는 것이 기획구현.md 3번의
"방법론적 위치를 구분해서 서술" 원칙과 합치한다.

⚠️ 해석 메모 (Precision/Recall=0이 나올 수 있는 이유):
  K=1(베이스라인, 경계 없음)이나 세그먼트가 적은 K=2~3에서는 모든
  표본에 같은 라벨(또는 정보량이 적은 라벨)이 부여되므로, 기본 분류
  임계값(0.5)에서 모델이 "이탈"으로 단 한 번도 예측하지 않을 수 있다
  (이탈률 42%인 구간도 다수는 여전히 '비이탈'이므로, 0.5 임계값
  기준으로는 비이탈로 분류됨). 이 경우 Precision/Recall이 0으로
  나오는 것은 버그가 아니라 "0.5 임계값에서 이 구조로는 이탈을 한
  건도 잡아내지 못한다"는 사실을 정확히 보여주는 것이다. 반면
  ROC-AUC는 확률 기반 지표라 이런 경우에도 구조의 변별력 차이를
  반영한다 (예: K=1은 정확히 0.5, K=2부터는 0.5보다 유의미하게 높음).
  사람이 K를 판단할 때는 ROC-AUC를 우선 참고하고, Precision/Recall은
  "실제 운영 임계값에서 쓸만한지"를 보는 보조 지표로 해석하는 것을
  권장한다.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

_SCORING = ["roc_auc", "f1", "recall", "precision"]


@dataclass
class MLPathCVResult:
    """ML경로 K(세그먼트 구조) 1개에 대한 교차검증 결과."""

    n_segments: int
    boundaries_tenure: list
    mean_scores: dict
    std_scores: dict
    n_folds: int


def _boundaries_to_segment_features(
    df_train: pd.DataFrame,
    boundaries_tenure: list[int],
    tenure_col: str = "tenure",
) -> pd.DataFrame:
    """
    세그먼트 라벨(원-핫)만으로 입력 피처를 구성한다. tenure 원본은
    포함하지 않는다.

    ⚠️ 설계 메모 (tenure 원본을 빼는 이유):
      처음에는 A_step2_methods.xgboost_cv와 동일하게 "tenure 원본 +
      세그먼트 더미"를 함께 넣었으나, 단일 의사결정나무
      (DecisionTreeClassifier)로 검증한 결과 트리가 거의 항상 tenure
      원본만으로 분기하고 세그먼트 더미는 무시하는 현상을 확인했다
      (feature_importances_에서 tenure가 0.85 이상, 세그먼트 더미는
      합쳐서 0.15 이하). 세그먼트 라벨이 tenure로부터 결정론적으로
      파생된 중복 정보이기 때문에, 정보량 기준으로 더 세밀한 tenure
      원본 쪽을 트리가 선호하는 것이 당연한 결과다. 그 결과 K(세그먼트
      개수)를 바꿔도 트리의 실제 분기가 전혀 달라지지 않아, 3단계
      교차검증 점수가 모든 K에서 완전히 동일하게 나오는 문제가 있었다
      (직접 재현해서 확인함).

      이 ML 경로의 3단계는 "2단계가 찾은 절단면 구조 자체가 분류에
      얼마나 기여하는지"를 보는 것이 목적이므로, tenure 원본을 빼고
      세그먼트 라벨만 입력해 K에 따라 점수가 실제로 달라지도록 한다
      (A_step2_methods.xgboost_cv는 기존 경로의 설계를 그대로 유지하며
      건드리지 않았다 -- 그쪽은 "세그먼트 라벨이 전체 속성+tenure
      조합에서 추가로 기여하는지"를 보는 다른 목적의 검증이라 동일한
      이슈가 있어도 별개로 둔다).
    """
    if len(boundaries_tenure) == 0:
        # 세그먼트가 1개뿐이면 모든 표본이 같은 카테고리이므로, 상수
        # 피처 1개로 표현한다 (K=1 베이스라인 -- 사실상 무정보 피처).
        return pd.DataFrame({"segment_0": np.ones(len(df_train))})

    edges = [-np.inf] + sorted(boundaries_tenure) + [np.inf]
    labels = pd.cut(
        df_train[tenure_col], bins=edges, labels=list(range(len(edges) - 1)), right=False
    ).astype(int)
    X = pd.get_dummies(labels, prefix="segment").reset_index(drop=True)
    return X


def cross_validate_tree_structure(
    df_train: pd.DataFrame,
    boundaries_tenure: list[int],
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
    n_splits: int = 5,
    random_state: int = 42,
    classifier: str = "tree",
) -> MLPathCVResult:
    """
    주어진 경계로 만든 세그먼트 라벨을 피처에 추가했을 때 분류 성능을
    StratifiedKFold 교차검증으로 측정한다.

    Parameters
    ----------
    n_splits : int
        요청사항 "5~10개로 쪼개어" 그대로 -- 기본값 5, 호출 시 최대 10까지
        자유롭게 조정 가능하다.
    classifier : {"tree", "logreg"}
        평가에 사용할 분류기. 기본은 2단계와 같은 계열인 의사결정나무
        (DecisionTreeClassifier)를 쓴다 -- "이 경계 구조가 다른 분류기로
        재현해도 잘 갈리는가"를 보려면 logreg도 선택 가능하게 했다.
    """
    y = df_train[churn_col].map({"Yes": 1, "No": 0})
    if y.isna().any():
        y = pd.to_numeric(df_train[churn_col], errors="coerce")
    X = _boundaries_to_segment_features(df_train, boundaries_tenure, tenure_col)

    if classifier == "tree":
        model = DecisionTreeClassifier(max_depth=4, random_state=random_state)
    elif classifier == "logreg":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        raise ValueError("classifier는 'tree' 또는 'logreg'여야 합니다.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    with warnings.catch_warnings():
        # Precision/Recall=0 케이스(설명은 모듈 docstring 참고)에서 발생하는
        # UndefinedMetricWarning은 의도된 결과이므로 출력에서만 억제한다.
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        scores = cross_validate(
            model, X, y, cv=cv, scoring=_SCORING, n_jobs=-1, error_score="raise"
        )

    mean_scores = {m: float(np.mean(scores[f"test_{m}"])) for m in _SCORING}
    std_scores = {m: float(np.std(scores[f"test_{m}"])) for m in _SCORING}

    return MLPathCVResult(
        n_segments=len(boundaries_tenure) + 1,
        boundaries_tenure=sorted(boundaries_tenure),
        mean_scores=mean_scores,
        std_scores=std_scores,
        n_folds=n_splits,
    )


def compare_tree_candidates(
    df_train: pd.DataFrame,
    boundary_candidates: list[list[int]],
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
    n_splits: int = 5,
    random_state: int = 42,
    classifier: str = "tree",
) -> list[MLPathCVResult]:
    """
    여러 경계 후보(2단계의 K별 트리 결과들)에 대해 일괄적으로 교차검증을
    수행한다. K=1(세그먼트 없음) 베이스라인도 자동으로 맨 앞에 포함한다.
    """
    results = []
    results.append(
        cross_validate_tree_structure(
            df_train, [], tenure_col, churn_col, n_splits, random_state, classifier
        )
    )
    for boundaries in boundary_candidates:
        results.append(
            cross_validate_tree_structure(
                df_train, boundaries, tenure_col, churn_col, n_splits, random_state, classifier
            )
        )
    return results


def results_to_table(results: list[MLPathCVResult]) -> pd.DataFrame:
    """MLPathCVResult 목록을 사람이 읽기 좋은 표(DataFrame)로 변환."""
    rows = []
    for r in results:
        row = {"n_segments": r.n_segments, "boundaries_tenure": r.boundaries_tenure}
        for metric in _SCORING:
            row[f"{metric}_mean"] = round(r.mean_scores[metric], 4)
            row[f"{metric}_std"] = round(r.std_scores[metric], 4)
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_by_roc_auc(results: list[MLPathCVResult]) -> MLPathCVResult:
    """
    ROC-AUC mean이 가장 높은 결과를 "참고용 추천"으로 반환한다.

    ⚠️ 주의: 이 함수는 기획구현.md 4번-②의 "임계값은 자동 탐지하지
    않음, 사람이 종합 판단" 원칙과 같은 맥락에서, *최종 채택을 자동
    확정하는 함수가 아니라* 사람이 표를 보기 전에 참고할 수 있는
    "추천 1개"를 보여주는 용도로만 쓴다. main.py 등 호출부에서는
    반드시 전체 results_to_table()도 함께 출력해 사람이 직접 비교할
    수 있게 한다.
    """
    return max(results, key=lambda r: r.mean_scores["roc_auc"])
