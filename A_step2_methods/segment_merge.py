"""
세그먼트 라벨링 + 사후 표본 점검(392건 미달 구간 통합).

기획구현.md 3번 "2단계 -- 머신러닝 기반 표본충분성 검증 + 사후 통합":
  "[필수, 빠뜨리지 말 것] 사후 표본 점검: 1단계가 제안한 각 K 구조에서
   구간별 실제 표본수·이탈건수를 직접 집계 -> 이탈 392건 미달 구간은
   인접 구간과 통합 후 재검증 (4번에서 확정한 사전 범용 기준과 동일하게
   적용 -- 500건이 아닌 392건이 코드 전체에서 일관되게 쓰이는 임계값)"

이 모듈은 "라벨링"과 "통합" 두 가지를 책임진다. 통합 여부 판단에 쓰는
임계값(392건)은 A_step1_methods.pelt.compute_min_group_size()를 그대로
재사용한다 (하드코딩 금지 원칙 일관 적용, 기획구현.md 4번-①).

xgboost_cv.py, bootstrap_cv.py 등 다른 2단계 모듈에서 공통으로 쓰는
경계 -> 라벨 변환 로직이라 별도 파일로 분리했다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def boundaries_to_labels(
    tenure: pd.Series,
    boundaries_tenure: list[int],
) -> pd.Series:
    """
    tenure 값과 경계 목록(예: [24, 48])을 받아 세그먼트 라벨(0, 1, 2, ...)을 만든다.

    경계는 "구간 시작 tenure"를 의미한다 (pelt.py의 boundaries_tenure 정의와 동일).
    예: boundaries_tenure=[24, 48] -> 구간0: tenure<24, 구간1: 24<=tenure<48,
        구간2: tenure>=48

    세그먼트 이름은 여기서 부여하지 않는다 (기획구현.md 3번:
    "사전에 '신규/중간/장기'로 고정하지 않음. ... 사후에 이름 부여").
    임시 라벨은 정수 인덱스(0, 1, 2, ...)만 쓴다.
    """
    edges = [-np.inf] + sorted(boundaries_tenure) + [np.inf]
    labels = pd.cut(
        tenure, bins=edges, labels=list(range(len(edges) - 1)), right=False
    )
    return labels.astype(int)


@dataclass
class SegmentStats:
    """세그먼트 1개의 표본 현황."""

    segment: int
    tenure_range: tuple  # (시작 tenure, 끝 tenure) - 사람이 읽기 위한 참고용
    n_customers: int
    n_churn: int
    churn_rate: float
    meets_threshold: bool  # n_churn >= min_group_churn_count 여부


@dataclass
class PostMergeResult:
    """사후 표본 점검 + 통합 결과."""

    original_boundaries: list
    final_boundaries: list  # 통합 후 최종 경계 (빈 리스트면 단일 세그먼트)
    min_group_churn_count: int  # 판정에 쓰인 임계값 (392건, compute_min_group_size 출력)
    merge_log: list  # 통합 과정 로그 (사람이 따라갈 수 있도록 단계별 기록)
    before_stats: list  # list[SegmentStats], 통합 전
    after_stats: list  # list[SegmentStats], 통합 후

    def before_stats_table(self) -> pd.DataFrame:
        return _stats_list_to_df(self.before_stats)

    def after_stats_table(self) -> pd.DataFrame:
        return _stats_list_to_df(self.after_stats)


def _stats_list_to_df(stats_list: list[SegmentStats]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "segment": s.segment,
                "tenure_range": s.tenure_range,
                "n_customers": s.n_customers,
                "n_churn": s.n_churn,
                "churn_rate": s.churn_rate,
                "meets_threshold(392)": s.meets_threshold,
            }
            for s in stats_list
        ]
    )


def compute_segment_stats(
    df_train: pd.DataFrame,
    boundaries_tenure: list[int],
    min_group_churn_count: int,
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
) -> list[SegmentStats]:
    """
    주어진 경계로 세그먼트를 나눴을 때, 구간별 표본수·이탈건수를 집계한다.
    """
    churn_binary = df_train[churn_col].map({"Yes": 1, "No": 0})
    if churn_binary.isna().any():
        churn_binary = pd.to_numeric(df_train[churn_col], errors="coerce")

    labels = boundaries_to_labels(df_train[tenure_col], boundaries_tenure)

    edges = [-np.inf] + sorted(boundaries_tenure) + [np.inf]
    tenure_min, tenure_max = df_train[tenure_col].min(), df_train[tenure_col].max()
    range_edges = [tenure_min] + sorted(boundaries_tenure) + [tenure_max + 1]

    stats = []
    for seg in sorted(labels.unique()):
        mask = labels == seg
        n_customers = int(mask.sum())
        n_churn = int(churn_binary[mask].sum())
        churn_rate = n_churn / n_customers if n_customers > 0 else 0.0
        lo = int(range_edges[seg])
        hi = int(range_edges[seg + 1]) - 1 if seg + 1 < len(range_edges) else int(tenure_max)
        stats.append(
            SegmentStats(
                segment=int(seg),
                tenure_range=(lo, hi),
                n_customers=n_customers,
                n_churn=n_churn,
                churn_rate=churn_rate,
                meets_threshold=(n_churn >= min_group_churn_count),
            )
        )
    return stats


def merge_undersized_segments(
    df_train: pd.DataFrame,
    boundaries_tenure: list[int],
    min_group_churn_count: int,
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
) -> PostMergeResult:
    """
    이탈 건수가 min_group_churn_count(392건) 미달인 구간을 인접 구간과
    통합한다. "인접 구간 중 표본이 더 작은 쪽"이 아니라, tenure 축에서
    물리적으로 이웃한 구간과 합치는 것이 자연스러우므로, 항상 "오른쪽
    이웃이 있으면 오른쪽과, 없으면(맨 끝 구간이면) 왼쪽과" 합치는 규칙을
    쓴다. 한 번 합친 뒤 표본을 재집계하고, 모든 구간이 기준을 충족할
    때까지(또는 단일 구간이 될 때까지) 반복한다.

    기획구현.md 3번: "이탈 392건 미달 구간은 인접 구간과 통합 후 재검증"
    """
    before_stats = compute_segment_stats(
        df_train, boundaries_tenure, min_group_churn_count, tenure_col, churn_col
    )

    current_boundaries = sorted(boundaries_tenure)
    merge_log = []

    while True:
        stats = compute_segment_stats(
            df_train, current_boundaries, min_group_churn_count, tenure_col, churn_col
        )

        undersized = [s for s in stats if not s.meets_threshold]
        if not undersized or len(current_boundaries) == 0:
            break

        # 가장 표본이 적은(가장 시급한) 미달 구간부터 처리
        target = min(undersized, key=lambda s: s.n_churn)
        seg_idx = target.segment
        n_segments = len(stats)

        if seg_idx < n_segments - 1:
            # 오른쪽 이웃과 합침 -> 오른쪽 경계(현재 구간과 다음 구간 사이)를 제거
            boundary_to_remove = current_boundaries[seg_idx]
            merge_direction = "오른쪽"
        else:
            # 맨 끝 구간이면 왼쪽과 합침 -> 왼쪽 경계를 제거
            boundary_to_remove = current_boundaries[seg_idx - 1]
            merge_direction = "왼쪽"

        merge_log.append(
            f"구간{seg_idx}(이탈 {target.n_churn}건, tenure {target.tenure_range})이 "
            f"{min_group_churn_count}건 미달 -> {merge_direction} 구간과 통합 "
            f"(경계 {boundary_to_remove} 제거)"
        )
        current_boundaries = [b for b in current_boundaries if b != boundary_to_remove]

    after_stats = compute_segment_stats(
        df_train, current_boundaries, min_group_churn_count, tenure_col, churn_col
    )

    return PostMergeResult(
        original_boundaries=sorted(boundaries_tenure),
        final_boundaries=current_boundaries,
        min_group_churn_count=min_group_churn_count,
        merge_log=merge_log,
        before_stats=before_stats,
        after_stats=after_stats,
    )
