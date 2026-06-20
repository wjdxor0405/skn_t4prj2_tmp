"""
전원 머신러닝(All-ML) 경로 -- 1단계: 밀도(쏠림) 해결.

기존 A_step1_methods(PELT, 순수 통계)와는 별도의 "대안 경로"다.
기획구현.md 3번이 "1단계는 통계, 2단계는 머신러닝"이라고 역할을 분리해둔
기존 경로와 달리, 이 경로는 경계 탐지 자체를 처음부터 끝까지 머신러닝으로
수행한다 (사람이 합의한 요구사항: "전원 머신러닝 처리 전략").

목적: tenure 분포에 사람이 개입하지 않고, 데이터의 밀도(쏠림)를 비지도
학습으로 파악해 자동으로 구간을 1차로 정돈한다. 이 1단계 결과는 "최종
경계"가 아니라, 2단계(의사결정나무)가 더 정교한 절단면을 찾을 수 있도록
입력 신호를 정돈하는 전처리 단계다.

제공하는 두 가지 방법 (요청 그대로):
  - K-means: tenure 값(1차원)을 K개 클러스터로 묶는다. 클러스터 중심이
    가까운 tenure끼리 묶이므로, 데이터가 특정 tenure 구간에 쏠려 있으면
    그 쏠린 구간 안에서 더 세밀하게, 희박한 구간은 넓게 묶이는 효과가
    자연스럽게 나온다 (밀도 적응적).
  - 정량적 분위(quantile) 클러스터링: 표본 수가 클러스터마다 똑같아지도록
    분위수로 자른다. "쏠림 해결"을 가장 직접적으로 보장하는 방법
    (각 구간의 표본 수가 사람이 보기에도 균등해짐).

두 방법 모두 "구간 경계 후보"를 만들어내는 것이 산출물이며, 최종 채택이
아니다 -- 다음 단계(step2_tree.py)가 이 정돈된 입력으로 실제 분류
경계를 학습한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


@dataclass
class DensitySplitResult:
    """1단계 밀도기반 분할 결과."""

    method: str  # "kmeans" 또는 "quantile"
    n_clusters: int
    boundaries_tenure: list  # 정렬된 tenure 경계값 (구간 시작점들, 0번째 제외)
    cluster_label_per_tenure: pd.Series  # index=tenure, value=cluster id (참고/시각화용)
    cluster_sizes: pd.Series  # 클러스터별 표본 수 (쏠림이 실제로 해소됐는지 확인용)


def _labels_to_boundaries(tenure_values: np.ndarray, labels: np.ndarray) -> list[int]:
    """
    정렬된 (tenure, cluster_label) 쌍에서, 클러스터가 바뀌는 지점을
    "구간 경계(boundaries_tenure)"로 변환한다.

    주의: K-means는 라벨 순서가 tenure 순서와 자동으로 일치하지 않을 수
    있으므로(클러스터 ID 0이 반드시 가장 작은 tenure를 의미하지 않음),
    클러스터 중심값 기준으로 라벨을 재정렬한 뒤 경계를 추출한다.
    """
    order = np.argsort(tenure_values)
    sorted_tenure = tenure_values[order]
    sorted_labels = labels[order]

    boundaries = []
    for i in range(1, len(sorted_tenure)):
        if sorted_labels[i] != sorted_labels[i - 1]:
            boundaries.append(int(sorted_tenure[i]))
    return boundaries


def split_by_kmeans(
    df_train: pd.DataFrame,
    n_clusters: int,
    tenure_col: str = "tenure",
    random_state: int = 42,
) -> DensitySplitResult:
    """
    tenure 값(1차원)을 K-means로 n_clusters개 그룹으로 묶는다.

    비지도 학습이므로 Churn 레이블은 전혀 사용하지 않는다 (기획구현.md
    3번 "입력: tenure별 이탈률 분포만 ... 이탈 여부와 무관하게 시간
    분포만 봄 -> 누수 방지" 원칙을 그대로 따른다 -- 여기서는 이탈률
    분포조차 보지 않고 tenure 값 자체의 밀도만 본다).
    """
    tenure_values = df_train[tenure_col].to_numpy().astype(float)
    X = tenure_values.reshape(-1, 1)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    # 클러스터 중심값 기준으로 라벨을 정렬 (0번이 가장 작은 tenure를 갖도록)
    center_order = np.argsort(km.cluster_centers_.ravel())
    relabel_map = {old: new for new, old in enumerate(center_order)}
    relabeled = np.array([relabel_map[l] for l in labels])

    boundaries = _labels_to_boundaries(tenure_values, relabeled)

    cluster_label_per_tenure = pd.Series(
        relabeled, index=pd.Index(tenure_values.astype(int), name="tenure")
    ).groupby(level=0).first().sort_index()

    cluster_sizes = pd.Series(relabeled).value_counts().sort_index()
    cluster_sizes.index.name = "cluster"

    return DensitySplitResult(
        method="kmeans",
        n_clusters=n_clusters,
        boundaries_tenure=sorted(set(boundaries)),
        cluster_label_per_tenure=cluster_label_per_tenure,
        cluster_sizes=cluster_sizes,
    )


def split_by_quantile(
    df_train: pd.DataFrame,
    n_clusters: int,
    tenure_col: str = "tenure",
) -> DensitySplitResult:
    """
    정량적 분위(quantile) 클러스터링: 각 구간의 "표본 수"가 똑같아지도록
    tenure를 n_clusters개 분위로 자른다. pd.qcut을 사용한다.

    K-means보다 "쏠림 해소"를 더 직접적으로 보장하는 방법이다 -- 클러스터
    개수만 정하면 표본 수가 자동으로 균등해진다 (밀집 구간은 좁게,
    희박 구간은 넓게 잘림).
    """
    tenure_values = df_train[tenure_col].to_numpy().astype(float)

    # qcut은 동일한 값이 많으면(중복 경계) 구간 수가 줄어들 수 있어
    # duplicates="drop"으로 안전하게 처리한다.
    labels, bin_edges = pd.qcut(
        tenure_values, q=n_clusters, retbins=True, labels=False, duplicates="drop"
    )
    actual_n_clusters = len(bin_edges) - 1

    boundaries = _labels_to_boundaries(tenure_values, labels)

    cluster_label_per_tenure = pd.Series(
        labels, index=pd.Index(tenure_values.astype(int), name="tenure")
    ).groupby(level=0).first().sort_index()

    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    cluster_sizes.index.name = "cluster"

    return DensitySplitResult(
        method="quantile",
        n_clusters=actual_n_clusters,
        boundaries_tenure=sorted(set(boundaries)),
        cluster_label_per_tenure=cluster_label_per_tenure,
        cluster_sizes=cluster_sizes,
    )


def run_density_split(
    df_train: pd.DataFrame,
    n_clusters: int,
    method: str = "kmeans",
    tenure_col: str = "tenure",
    random_state: int = 42,
) -> DensitySplitResult:
    """
    1단계 진입점. method에 따라 K-means 또는 분위 클러스터링을 실행한다.
    """
    if method == "kmeans":
        return split_by_kmeans(df_train, n_clusters, tenure_col, random_state)
    elif method == "quantile":
        return split_by_quantile(df_train, n_clusters, tenure_col)
    else:
        raise ValueError("method는 'kmeans' 또는 'quantile'이어야 합니다.")
