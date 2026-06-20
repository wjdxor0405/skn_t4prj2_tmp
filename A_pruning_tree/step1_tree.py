"""
분석 A ① 단계: 경계 탐지 -- 가지치기 회귀나무 (메인) + 랜덤포레스트 투표 (보조).

기획구현가이드(A,B확정판).md 3번 섹션:
  "① 경계 탐지: 가지치기 회귀나무 (메인, 머신러닝) + 랜덤포레스트 투표 (보조 검증)"

⚠️ 이 ①은 main.py/run_cycle.py의 "①②③ 반복 사이클" 중 첫 단계다.
   예측모델의 "1단계"(기획구현가이드 5번)와는 다른 번호 체계이므로 혼동 주의.

입력: tenure 단일변수, 출력: tenure별 평균 이탈률
가중치: 각 tenure의 표본수 sqrt(count) -- 표본이 많은 tenure일수록 평균
        이탈률 추정이 더 안정적이므로, 회귀나무 학습 시 더 신뢰도 있게
        반영되도록 가중치를 준다. count 그대로 쓰지 않고 sqrt를 쓰는
        이유는 표본수 차이가 매우 클 때(예: 10명 vs 500명) 가중치가
        과도하게 쏠리는 것을 완화하기 위함(가이드 명시 권장값).

핵심 구현 규칙(가이드 필수 사항):
  - cost_complexity_pruning_path가 반환하는 alpha 후보를 절대 샘플링하지
    않고 전체를 다 탐색한다 (일부만 쓰면 그리드 민감성 발생, 가이드에서
    실제로 확인됨).
  - 보조 검증(랜덤포레스트 투표)으로 ①을 통째로 대체하지 않는다. 어디까지나
    "CV 성능 상위 alpha 후보의 경계가 투표 상위 결과와 일치하는지" 확인하는
    교차검증 역할만 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor


@dataclass
class AlphaCandidate:
    """ccp_alpha 후보 1개의 평가 결과."""

    ccp_alpha: float
    cv_mse_mean: float  # 5-fold CV 평균 제곱오차 (작을수록 좋음)
    cv_mse_std: float
    n_leaves: int  # 이 alpha로 전체 데이터를 학습했을 때 리프(세그먼트) 개수
    boundaries_tenure: list  # 이 alpha로 전체 데이터를 학습했을 때의 경계


@dataclass
class RandomForestVoteResult:
    """랜덤포레스트 분기점 투표 결과."""

    n_estimators: int
    threshold_votes: pd.Series  # index=tenure(반올림), value=득표수(여러 트리에서 등장한 횟수). 내림차순 정렬.


@dataclass
class TreeStep1Result:
    """① 단계 전체 결과."""

    signal: pd.DataFrame  # tenure별 (평균 이탈률, 표본수, 가중치)
    alpha_candidates: list  # list[AlphaCandidate], ccp_alpha 전수탐색 결과
    best_alpha: AlphaCandidate  # CV MSE가 가장 낮은 후보
    rf_vote: RandomForestVoteResult
    boundaries_tenure: list  # 최종 채택 경계 (best_alpha 기준)


def build_tenure_signal(
    df: pd.DataFrame,
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
) -> pd.DataFrame:
    """
    tenure별 평균 이탈률 신호를 만든다. 이탈 여부와 무관하게 시간 분포만
    보는 기존 경로(A_step1_methods)의 누수 방지 원칙을 그대로 따른다 --
    여기서도 입력은 tenure->이탈률 집계뿐, 개별 행 단위 다른 속성은
    전혀 사용하지 않는다.
    """
    churn_binary = df[churn_col].map({"Yes": 1, "No": 0})
    if churn_binary.isna().any():
        churn_binary = pd.to_numeric(df[churn_col], errors="coerce")

    tmp = pd.DataFrame({"tenure": df[tenure_col].values, "churn": churn_binary.values})
    grouped = tmp.groupby("tenure")["churn"].agg(["count", "mean"])
    grouped = grouped.rename(columns={"count": "n_customers", "mean": "churn_rate"})

    full_index = pd.RangeIndex(
        start=int(grouped.index.min()), stop=int(grouped.index.max()) + 1
    )
    grouped = grouped.reindex(full_index)
    # 표본이 아예 없는 tenure(결측 구간)는 이탈률 0, 표본수 0으로 채운다.
    grouped["n_customers"] = grouped["n_customers"].fillna(0).astype(int)
    grouped["churn_rate"] = grouped["churn_rate"].fillna(0.0)
    grouped["weight"] = np.sqrt(grouped["n_customers"])
    grouped.index.name = "tenure"
    return grouped


def _extract_thresholds_from_tree(tree: DecisionTreeRegressor) -> list[float]:
    """학습된 단일 트리에서 분기 threshold들을 추출한다 (리프 제외)."""
    t = tree.tree_
    thresholds = []
    for node_id in range(t.node_count):
        if t.children_left[node_id] != t.children_right[node_id]:
            # 리프가 아닌 노드(분기 노드)만 threshold가 의미를 가진다.
            thresholds.append(float(t.threshold[node_id]))
    return thresholds


def _thresholds_to_boundaries(thresholds: list[float]) -> list[int]:
    """실수 threshold들을 정수 tenure 경계로 변환한다 (올림, 중복 제거, 정렬)."""
    return sorted(set(int(np.ceil(th)) for th in thresholds))


def search_alpha_candidates(
    signal: pd.DataFrame,
    n_splits: int = 5,
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> list[AlphaCandidate]:
    """
    cost_complexity_pruning_path가 반환하는 ccp_alpha 후보를 전수
    탐색하며, 각 alpha에 대해 5-fold 교차검증 MSE를 측정한다.

    ⚠️ 필수 구현 규칙(가이드 명시): alpha 후보를 샘플링하지 않고
    전체를 다 쓴다.

    Parameters
    ----------
    min_samples_leaf : int
        리프 1개가 가져야 할 최소 "개월 수"(tenure 신호 단위 표본 개수,
        고객 머릿수가 아님에 유의). 기본 1(제약 없음). run_cycle.py가
        ②③ 검증에 실패했을 때, 이 값을 점진적으로 늘려 ①을 재실행하는
        용도로 쓴다 -- 데이터를 변형하지 않고도 회귀나무가 같은 좁은
        구간을 다시 제안하지 못하게 막는 가드레일 역할.
    """
    X = signal.index.to_numpy().astype(float).reshape(-1, 1)
    y = signal["churn_rate"].to_numpy()
    sample_weight = signal["weight"].to_numpy()

    base_tree = DecisionTreeRegressor(
        min_samples_leaf=min_samples_leaf, random_state=random_state
    )
    path = base_tree.cost_complexity_pruning_path(X, y, sample_weight=sample_weight)
    # ccp_alphas는 오름차순으로 정렬되어 반환된다. 마지막 값은 트리 전체를
    # 가지치기해서 루트 노드 1개(=세그먼트 1개)로 만드는 alpha이므로
    # 포함해도 무방하다(K=1 베이스라인 역할).
    ccp_alphas = path.ccp_alphas

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    candidates = []
    for alpha in ccp_alphas:  # 전수 탐색 (샘플링 금지)
        alpha = float(alpha)
        tree = DecisionTreeRegressor(
            ccp_alpha=alpha, min_samples_leaf=min_samples_leaf, random_state=random_state
        )

        # cross_val_score는 sample_weight를 params로 받을 수 있다
        # (sklearn 1.8 기준; 구버전은 fit_params를 쓰지만 여기서는 설치된
        # 버전에 맞춰 params를 사용한다).
        scores = cross_val_score(
            tree, X, y, cv=cv, scoring="neg_mean_squared_error",
            params={"sample_weight": sample_weight},
        )
        mse_scores = -scores

        # 전체 데이터로 다시 학습해 실제 경계(리프 개수)를 확인한다.
        full_tree = DecisionTreeRegressor(
            ccp_alpha=alpha, min_samples_leaf=min_samples_leaf, random_state=random_state
        )
        full_tree.fit(X, y, sample_weight=sample_weight)
        thresholds = _extract_thresholds_from_tree(full_tree)
        boundaries = _thresholds_to_boundaries(thresholds)

        candidates.append(
            AlphaCandidate(
                ccp_alpha=alpha,
                cv_mse_mean=float(mse_scores.mean()),
                cv_mse_std=float(mse_scores.std()),
                n_leaves=int(full_tree.get_n_leaves()),
                boundaries_tenure=boundaries,
            )
        )
    return candidates


def select_best_alpha(candidates: list[AlphaCandidate]) -> AlphaCandidate:
    """CV MSE가 가장 낮은(예측오차가 가장 작은) alpha를 선택한다."""
    return min(candidates, key=lambda c: c.cv_mse_mean)


def random_forest_vote(
    signal: pd.DataFrame,
    n_estimators: int = 250,
    random_state: int = 42,
) -> RandomForestVoteResult:
    """
    RandomForestRegressor를 학습시켜, 모든 개별 트리의 분기점을 전부
    수집하고 빈도를 집계한다 (보조 검증 -- ①을 대체하지 않음).
    """
    X = signal.index.to_numpy().astype(float).reshape(-1, 1)
    y = signal["churn_rate"].to_numpy()
    sample_weight = signal["weight"].to_numpy()

    rf = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    rf.fit(X, y, sample_weight=sample_weight)

    all_thresholds = []
    for estimator in rf.estimators_:
        all_thresholds.extend(_extract_thresholds_from_tree(estimator))

    # 실수 threshold를 정수 tenure로 반올림해 득표를 집계한다.
    rounded = [int(np.round(th)) for th in all_thresholds]
    votes = pd.Series(rounded).value_counts().sort_values(ascending=False)
    votes.index.name = "tenure"

    return RandomForestVoteResult(n_estimators=n_estimators, threshold_votes=votes)


def run_step1(
    df: pd.DataFrame,
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
    n_splits: int = 5,
    min_samples_leaf: int = 1,
    n_estimators: int = 250,
    random_state: int = 42,
) -> TreeStep1Result:
    """① 단계 진입점: 신호 생성 -> alpha 전수탐색 -> 최적 alpha 선택 -> RF 투표."""
    signal = build_tenure_signal(df, tenure_col=tenure_col, churn_col=churn_col)
    alpha_candidates = search_alpha_candidates(
        signal, n_splits=n_splits, min_samples_leaf=min_samples_leaf, random_state=random_state
    )
    best = select_best_alpha(alpha_candidates)
    rf_vote = random_forest_vote(
        signal, n_estimators=n_estimators, random_state=random_state
    )

    return TreeStep1Result(
        signal=signal,
        alpha_candidates=alpha_candidates,
        best_alpha=best,
        rf_vote=rf_vote,
        boundaries_tenure=best.boundaries_tenure,
    )


def alpha_candidates_table(candidates: list[AlphaCandidate]) -> pd.DataFrame:
    """alpha 후보 전체를 사람이 보기 좋은 표로 변환 (CV MSE 오름차순)."""
    rows = [
        {
            "ccp_alpha": round(c.ccp_alpha, 6),
            "cv_mse_mean": round(c.cv_mse_mean, 6),
            "cv_mse_std": round(c.cv_mse_std, 6),
            "n_leaves": c.n_leaves,
            "boundaries_tenure": c.boundaries_tenure,
        }
        for c in candidates
    ]
    df = pd.DataFrame(rows)
    return df.sort_values("cv_mse_mean").reset_index(drop=True)


def check_rf_agreement(
    best_alpha: AlphaCandidate,
    rf_vote: RandomForestVoteResult,
    top_n: int = 10,
    tolerance: int = 2,
) -> pd.DataFrame:
    """
    best_alpha의 경계들이 RF 투표 상위 top_n 안에 (오차범위 tolerance
    이내로) 들어있는지 확인하는 표를 만든다. 가이드 권장 사용법:
    "CV 성능 상위 alpha 후보의 경계가 투표 결과의 상위 후보와 일치하는지
    확인 -- 일치하면 신뢰, 아니면 더 보수적인 후보로 한 단계 내려갈 것"
    """
    top_voted = rf_vote.threshold_votes.head(top_n)
    rows = []
    for b in best_alpha.boundaries_tenure:
        # tolerance 범위 내 가장 가까운 투표 tenure를 찾는다.
        diffs = (top_voted.index - b).map(abs)
        if len(diffs) > 0 and diffs.min() <= tolerance:
            matched_tenure = top_voted.index[diffs.argmin()]
            votes = int(top_voted.loc[matched_tenure])
            agrees = True
        else:
            matched_tenure = None
            votes = 0
            agrees = False
        rows.append(
            {
                "boundary": b,
                "matched_rf_tenure": matched_tenure,
                "rf_votes": votes,
                "agrees_with_top_n": agrees,
            }
        )
    return pd.DataFrame(rows)
