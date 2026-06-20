"""
전원 머신러닝(All-ML) 경로 -- 2단계: 경계선 추출 (의사결정나무).

1단계(step1_density.py)가 밀도 기준으로 1차 정돈한 데이터를 입력받아,
지도학습(의사결정나무)으로 "이탈률을 가장 잘 가르는 절단면(threshold)"을
스스로 찾는다. 여기서부터는 Churn 레이블을 사용한다 (지도학습이므로
당연히 필요 -- 기획구현.md 3번의 "이탈 여부와 무관하게 시간 분포만 봄"
원칙은 *비지도* 1단계에만 해당되고, 지도학습인 2단계는 애초에 다른
성격의 단계다. 이는 기존 경로의 1단계(PELT, 통계)와 2단계(XGBoost,
머신러닝)가 입력을 다르게 가져가는 것과 같은 논리다).

핵심 아이디어:
  - sklearn.tree.DecisionTreeClassifier를 tenure(1개 피처) 단독으로
    학습시키면, 트리가 분기할 때 선택하는 threshold 값들이 곧
    "이탈률이 가장 크게 갈리는 tenure 절단점"이 된다.
  - max_leaf_nodes(또는 max_depth)로 분기 개수를 제어해 K(세그먼트 수)를
    조절한다. max_leaf_nodes=K로 두면 정확히 K개의 리프(=K-1개의 분기
    경계)가 나온다.
  - 1단계의 클러스터 라벨(cluster_label_per_tenure)은 "밀도가 균등하게
    조정된 입력 구조"를 의사결정나무에 넘겨주는 용도로, 트리 학습 시
    min_samples_leaf의 기준이 되는 "표본 밀도가 보정된 tenure 좌표"로
    활용한다 (자세한 내용은 build_features 참고).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


@dataclass
class TreeBoundaryResult:
    """2단계 의사결정나무 경계 추출 결과."""

    n_leaves: int  # 실제로 만들어진 리프(=세그먼트) 개수
    boundaries_tenure: list  # 정렬된 tenure 경계값 (분기 threshold들)
    tree: DecisionTreeClassifier  # 학습된 트리 객체 (해석/시각화용으로 보존)
    feature_importance: float  # tenure 피처의 중요도 (단일 피처라 항상 1.0에 가까움, 참고용)


def _extract_thresholds(tree: DecisionTreeClassifier, feature_index: int = 0) -> list[float]:
    """
    학습된 sklearn 트리 내부 구조(_tree.Tree)를 순회하며, 지정한 피처
    인덱스로 분기하는 모든 노드의 threshold 값을 추출한다.

    sklearn은 트리를 배열 기반(children_left, children_right, feature,
    threshold)으로 저장한다. 리프 노드는 feature == -2(TREE_UNDEFINED)로
    표시되므로, 그 외 노드만 모아 분기 기준값을 뽑는다.
    """
    t = tree.tree_
    thresholds = []
    for node_id in range(t.node_count):
        if t.feature[node_id] == feature_index:
            thresholds.append(float(t.threshold[node_id]))
    return thresholds


def extract_boundaries_via_tree(
    df_train: pd.DataFrame,
    max_leaf_nodes: int,
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
    min_samples_leaf_ratio: float = 0.02,
    random_state: int = 42,
) -> TreeBoundaryResult:
    """
    tenure 단일 피처로 의사결정나무를 학습시켜, 이탈률을 가장 잘 가르는
    절단면(threshold)들을 경계로 추출한다.

    Parameters
    ----------
    max_leaf_nodes : int
        만들고자 하는 세그먼트 수(K)에 해당. sklearn의 max_leaf_nodes를
        그대로 K로 사용하면, 트리는 정확히 K개의 리프(=K-1개의 분기)를
        만들도록 최선의 절단면을 탐욕적으로(greedy, best-first) 선택한다.
    min_samples_leaf_ratio : float
        리프 1개가 가져야 할 최소 표본 비율 (전체 표본 대비). 너무 작은
        리프(=너무 좁은 tenure 구간)가 나오지 않도록 막는 가드레일이다.
        기본 2% -- 1단계의 "쏠림 해결"과 별개로, 트리 자체가 극단적으로
        좁은 구간을 만들지 않도록 하는 안전장치 (A_step1/2 경로의
        min_size, 392/393명 기준과 같은 역할을 의사결정나무 버전으로
        둔 것).

    Returns
    -------
    TreeBoundaryResult
    """
    y = df_train[churn_col].map({"Yes": 1, "No": 0})
    if y.isna().any():
        y = pd.to_numeric(df_train[churn_col], errors="coerce")
    X = df_train[[tenure_col]].astype(float)

    min_samples_leaf = max(1, int(len(df_train) * min_samples_leaf_ratio))

    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree.fit(X, y)

    thresholds = _extract_thresholds(tree, feature_index=0)
    # threshold는 실수(예: 23.5)로 나오므로, tenure가 정수 개월이라는
    # 점을 이용해 "경계 이상부터 다음 구간"으로 해석되게 올림 처리한다.
    boundaries = sorted(set(int(np.ceil(th)) for th in thresholds))

    n_leaves = int(tree.get_n_leaves())
    feature_importance = float(tree.feature_importances_[0]) if len(tree.feature_importances_) else 0.0

    return TreeBoundaryResult(
        n_leaves=n_leaves,
        boundaries_tenure=boundaries,
        tree=tree,
        feature_importance=feature_importance,
    )


def run_tree_boundary_search(
    df_train: pd.DataFrame,
    k_search_range: range,
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
    min_samples_leaf_ratio: float = 0.02,
    random_state: int = 42,
) -> list[TreeBoundaryResult]:
    """
    k_search_range의 각 K(세그먼트 수)에 대해 의사결정나무로 경계를
    추출한다. 기존 경로의 1단계가 K=2~8 범위를 후보로 내는 것과
    동일한 인터페이스를 맞추기 위함 (3단계 교차검증에서 K별로 비교).
    """
    results = []
    seen_boundaries = set()
    for k in k_search_range:
        result = extract_boundaries_via_tree(
            df_train,
            max_leaf_nodes=k,
            tenure_col=tenure_col,
            churn_col=churn_col,
            min_samples_leaf_ratio=min_samples_leaf_ratio,
            random_state=random_state,
        )
        key = tuple(result.boundaries_tenure)
        if key in seen_boundaries:
            # min_samples_leaf 제약 때문에 더 큰 K를 요청해도 실제로는
            # 같은 경계로 수렴할 수 있다 (중복 결과는 생략).
            continue
        seen_boundaries.add(key)
        results.append(result)
    return results
