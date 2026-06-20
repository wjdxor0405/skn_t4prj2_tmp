"""
전원 머신러닝(All-ML) 경로 전체 오케스트레이션.

기존 경로(A_step1_methods: PELT 통계, A_step2_methods: XGBoost 검증)와는
독립적인 "대안 경로"다. main.py에서 선택적으로 호출한다(기존 경로를
대체하지 않음 -- "기존 코드도 실행될 수 있어야 함" 요구사항).

절차 (사람이 합의한 요청 그대로):
  1단계 (쏠림 해결): 비지도학습(K-means/분위 클러스터링)으로 tenure
                      밀도를 자동으로 정돈
  2단계 (경계선 추출): 지도학습(의사결정나무)으로 이탈률을 가장 잘
                        가르는 절단면을 K=2~8 범위에서 탐색
  3단계 (정확도 검증): StratifiedKFold 교차검증(5~10폴드)으로 ROC-AUC/
                        F1-score를 측정, 통계 공식(p-value 등) 미사용

1단계 결과(클러스터 라벨)는 2단계 트리 학습의 "입력 정돈" 역할만 하고,
실제 경계는 2단계 트리가 스스로 찾는다. 1단계의 클러스터 개수
(density_n_clusters)는 2단계가 탐색할 K 범위의 "출발 해상도"로 쓰인다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

import step1_density  # noqa: E402
import step2_tree  # noqa: E402
import step3_cv_validation as step3  # noqa: E402


@dataclass
class MLPathResult:
    """전원 머신러닝 경로 전체 실행 결과."""

    density_results: list  # list[step1_density.DensitySplitResult], 클러스터 수별
    tree_results: list  # list[step2_tree.TreeBoundaryResult], K별
    cv_results: list  # list[step3.MLPathCVResult]
    best_by_roc_auc: "step3.MLPathCVResult"

    def cv_table(self) -> pd.DataFrame:
        return step3.results_to_table(self.cv_results)

    def density_summary_table(self) -> pd.DataFrame:
        rows = []
        for d in self.density_results:
            rows.append(
                {
                    "method": d.method,
                    "n_clusters_requested": d.n_clusters,
                    "boundaries_tenure": d.boundaries_tenure,
                    "cluster_sizes": d.cluster_sizes.to_dict(),
                }
            )
        return pd.DataFrame(rows)

    def tree_summary_table(self) -> pd.DataFrame:
        rows = []
        for t in self.tree_results:
            rows.append(
                {
                    "n_leaves": t.n_leaves,
                    "boundaries_tenure": t.boundaries_tenure,
                    "tenure_feature_importance": round(t.feature_importance, 4),
                }
            )
        return pd.DataFrame(rows)


def run_ml_path(
    df_train: pd.DataFrame,
    k_search_range: range | None = None,
    density_method: str = "kmeans",
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
    n_cv_splits: int = 5,
    min_samples_leaf_ratio: float = 0.02,
    random_state: int = 42,
) -> MLPathResult:
    """
    전원 머신러닝 경로 전체 실행 (1->2->3단계).

    Parameters
    ----------
    df_train : pd.DataFrame
        Train 데이터 (기존 경로와 동일하게 Test 누수 방지를 위해 Train만 사용).
    k_search_range : range, optional
        탐색할 세그먼트 개수(K) 범위. 기본값은 기존 경로와 동일하게
        K=2~8 (range(2, 9)) -- 두 경로를 같은 조건에서 비교할 수 있도록.
    density_method : {"kmeans", "quantile"}
        1단계 밀도 분할 방법.
    n_cv_splits : int
        3단계 StratifiedKFold 폴드 수 (요청사항: 5~10 범위 권장).
    """
    if k_search_range is None:
        k_search_range = range(2, 9)

    # 1단계: 밀도(쏠림) 해결. k_search_range의 각 K에 대해 1단계도 함께
    # 실행해, "이 정도 쏠림 정돈에서 표본이 어떻게 분포하는지" 사람이
    # 참고할 수 있게 모두 보존한다 (2단계 트리 입력 자체는 tenure 원본을
    # 그대로 쓰므로, 1단계 결과가 2단계를 강제하지는 않는다 -- 아래 설명 참고).
    density_results = [
        step1_density.run_density_split(
            df_train, n_clusters=k, method=density_method,
            tenure_col=tenure_col, random_state=random_state,
        )
        for k in k_search_range
    ]

    # 2단계: 의사결정나무로 경계 추출. min_samples_leaf_ratio가 "쏠림이
    # 심한 구간에서 과도하게 좁은 리프가 나오지 않도록" 막아주므로,
    # 1단계가 명시적으로 정돈한 데이터를 직접 분할해 넘기지 않아도
    # "쏠림 해결"의 효과가 트리 학습 자체에 반영된다. (1단계 결과는
    # density_results로 별도 제공해 사람이 "쏠림이 실제로 어떻게
    # 생겼는지"를 K-means/분위수 관점에서 비교할 수 있게 한다.)
    tree_results = step2_tree.run_tree_boundary_search(
        df_train,
        k_search_range=k_search_range,
        tenure_col=tenure_col,
        churn_col=churn_col,
        min_samples_leaf_ratio=min_samples_leaf_ratio,
        random_state=random_state,
    )

    # 3단계: StratifiedKFold 교차검증으로 ROC-AUC/F1 측정
    boundary_candidates = [t.boundaries_tenure for t in tree_results]
    cv_results = step3.compare_tree_candidates(
        df_train,
        boundary_candidates,
        tenure_col=tenure_col,
        churn_col=churn_col,
        n_splits=n_cv_splits,
        random_state=random_state,
        classifier="tree",
    )

    best = step3.select_best_by_roc_auc(cv_results)

    return MLPathResult(
        density_results=density_results,
        tree_results=tree_results,
        cv_results=cv_results,
        best_by_roc_auc=best,
    )
