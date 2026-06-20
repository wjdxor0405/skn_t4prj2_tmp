"""
분석 A ② 단계: 적절성 검증 -- 세그먼트단독 AUC + 순열검정.

기획구현가이드(A,B확정판).md 3번 섹션:
  "①이 찾은 세그먼트 라벨만(다른 속성 없이) 입력으로 분류모델
   (RandomForestClassifier) 학습, 5-fold 교차검증 AUC 측정"
  "이탈여부 라벨을 수백 회(200회 이상) 무작위로 섞어 같은 절차 반복 ->
   실제 AUC가 그 분포보다 압도적으로 높으면 경계가 통계적으로 실재"
  "이론적 분포(카이제곱 등) 전혀 사용 않는 완전 데이터기반 검증"

⚠️ 4번 섹션의 경고: 이 모듈의 부트스트랩/순열검정은 ③단계
   (step3_bootstrap_ci.py)의 부트스트랩과 "같은 재추출 기법이지만 다른
   목적"이다. ②는 "경계~이탈여부 관련성이 우연이 아닌지" 보는 것이고,
   ③은 "그 측정값(AUC) 자체가 표본크기 면에서 안정적인지" 보는 것이다.
   절대 한 함수로 합치지 않는다 (가이드 명시 요구사항).

⚠️ 중요(가이드 80번 줄): 이 검증은 "경계 자체가 실재하는가"를 보는 것이며,
   "예측모델(20개 변수 포함)에 추가했을 때 분류성능이 오르는가"와는 다른
   질문이다. 5번 섹션(예측모델)의 XGBoost 비교와 혼동하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


@dataclass
class PermutationTestResult:
    """② 단계 순열검정 결과."""

    boundaries_tenure: list
    observed_auc: float  # 실제 라벨로 측정한 세그먼트단독 AUC
    null_auc_distribution: np.ndarray  # 라벨을 섞어서 측정한 AUC 분포 (n_permutations개)
    p_value: float  # observed_auc 이상이 null 분포에서 나온 비율
    n_permutations: int


def boundaries_to_segment_labels(
    tenure: pd.Series,
    boundaries_tenure: list[int],
) -> pd.DataFrame:
    """
    tenure 값과 경계 목록으로 세그먼트 라벨(원-핫)을 만든다.
    A_step1_methods/A_step2_methods의 동명 함수와 동일한 정의(경계는
    "구간 시작 tenure")를 따르되, 이 경로 전용으로 독립 보존한다
    (비교군과 결과를 섞지 않기 위함).
    """
    edges = [-np.inf] + sorted(boundaries_tenure) + [np.inf]
    n_segments = len(edges) - 1
    labels = pd.cut(tenure, bins=edges, labels=list(range(n_segments)), right=False).astype(int)
    return pd.get_dummies(labels, prefix="segment")


def segment_only_auc(
    X_segment: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_estimators: int = 100,
    random_state: int = 42,
) -> float:
    """
    세그먼트 라벨만(다른 속성 없이)을 입력으로 RandomForestClassifier를
    학습시켜 5-fold 교차검증 평균 AUC를 측정한다.

    n_estimators 기본값을 100으로 둔 이유: 가이드는 ①의 보조검증 RF만
    "200~300그루"로 명시했고, ②의 RF는 트리 개수를 지정하지 않았다.
    이 함수는 순열검정에서 200회 이상 반복 호출되므로(가이드 권장),
    트리 개수가 속도에 거의 선형으로 영향을 준다(실측: 200그루 1.3초,
    100그루 0.64초 -- 순열 200회 기준 약 4.3분 -> 약 2.1분). observed_auc와
    null 분포 양쪽에 동일한 n_estimators를 일관되게 적용하므로, 통계적
    비교(p-value 산출)의 타당성에는 영향이 없다 -- 더 적은 트리로도
    "관측값이 우연보다 압도적으로 높은가"라는 상대 비교는 유효하다.
    """
    if X_segment.shape[1] <= 1:
        # 세그먼트가 1개뿐(K=1)이면 모든 표본이 같은 카테고리라 AUC를
        # 정의할 수 없는 상수 입력이다. 정보가 전혀 없다는 의미로
        # 0.5(무작위 수준)를 반환한다.
        return 0.5

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=1)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, X_segment, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


def run_permutation_test(
    df: pd.DataFrame,
    boundaries_tenure: list[int],
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
    n_permutations: int = 200,
    n_splits: int = 5,
    n_estimators: int = 100,
    random_state: int = 42,
) -> PermutationTestResult:
    """
    ② 단계 전체 실행: 세그먼트단독 AUC 측정 + 라벨 순열검정.

    Parameters
    ----------
    n_permutations : int
        가이드 권장 "수백 회(200회 이상)". 기본값 200.
    n_estimators : int
        순열검정에 쓸 RandomForestClassifier 트리 개수. 기본 100
        (segment_only_auc의 속도 메모 참고).
    """
    y = df[churn_col].map({"Yes": 1, "No": 0})
    if y.isna().any():
        y = pd.to_numeric(df[churn_col], errors="coerce")
    y = y.reset_index(drop=True)

    X_segment = boundaries_to_segment_labels(df[tenure_col], boundaries_tenure).reset_index(drop=True)

    observed_auc = segment_only_auc(
        X_segment, y, n_splits=n_splits, n_estimators=n_estimators, random_state=random_state
    )

    rng = np.random.default_rng(random_state)
    null_aucs = np.empty(n_permutations)
    for i in range(n_permutations):
        y_shuffled = y.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
        null_aucs[i] = segment_only_auc(
            X_segment, y_shuffled, n_splits=n_splits, n_estimators=n_estimators, random_state=random_state
        )

    # p-value: 순열(우연) 분포에서 관측 AUC 이상이 나온 비율.
    # 이론적 분포를 전혀 가정하지 않는 완전 데이터기반 추정 (가이드 명시).
    p_value = float(np.mean(null_aucs >= observed_auc))

    return PermutationTestResult(
        boundaries_tenure=sorted(boundaries_tenure),
        observed_auc=observed_auc,
        null_auc_distribution=null_aucs,
        p_value=p_value,
        n_permutations=n_permutations,
    )


def permutation_summary(result: PermutationTestResult) -> dict:
    """순열검정 결과를 사람이 읽기 좋은 요약 딕셔너리로 변환."""
    null = result.null_auc_distribution
    return {
        "boundaries_tenure": result.boundaries_tenure,
        "observed_auc": round(result.observed_auc, 4),
        "null_auc_mean": round(float(null.mean()), 4),
        "null_auc_std": round(float(null.std()), 4),
        "null_auc_95th_pct": round(float(np.percentile(null, 95)), 4),
        "p_value": result.p_value,
        "n_permutations": result.n_permutations,
        "is_significant_p<0.05": result.p_value < 0.05,
    }
