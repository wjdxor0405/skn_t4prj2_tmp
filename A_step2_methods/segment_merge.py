"""
세그먼트 라벨링 + 사후 표본 점검(전체 표본 수 393명 미달 구간 통합).

기획구현.md 3번 "2단계 -- 머신러닝 기반 표본충분성 검증 + 사후 통합":
  "[필수, 빠뜨리지 말 것] 사후 표본 점검: 1단계가 제안한 각 K 구조에서
   구간별 실제 표본수·이탈건수를 직접 집계 -> 이탈 392건 미달 구간은
   인접 구간과 통합 후 재검증 (4번에서 확정한 사전 범용 기준과 동일하게
   적용 -- 500건이 아닌 392건이 코드 전체에서 일관되게 쓰이는 임계값)"

이 모듈은 "라벨링"과 "통합" 두 가지를 책임진다. 통합 여부 판단에 쓰는
임계값(393)은 A_step1_methods.pelt.compute_min_group_size()를 그대로
재사용한다 (하드코딩 금지 원칙 일관 적용, 기획구현.md 4번-①).

⚠️ 정정 이력 (표본충분성 논리 재검증, 두 번째 발견 -- 분모 정의 오류)
--------------------------------------------------------------------
이전 버전은 compute_min_group_size()의 출력값(393)을 "그룹당 필요한
최소 *이탈 건수*"로 해석해 `n_churn >= 393`으로 판정했다. 이는 통계적
정의와 어긋난다.

statsmodels.stats.proportion.power_proportions_2indep의 nobs1
파라미터는 공식 문서에 "number of observations in sample 1"로
명시되어 있다 -- "표본의 전체 관측치 수"이지 "표본 안에서 어떤 사건이
일어난 건수"가 아니다. 실제로 nobs1=393(전체 표본)으로 교차검증하면
정확히 검정력 80%가 재현되므로, 393은 처음부터 "그룹당 전체 표본 수
(고객 머릿수)" 기준이었다.

"이탈 건수" 기준으로 잘못 적용했을 때의 문제점 (실데이터 기준):
  - 이탈률 26.5% 구간에서 같은 검정력을 얻으려면 실제로는
    393 ÷ 0.265 ≈ 1,483건의 전체 표본이 필요해져, 검정력 분석이
    원래 요구한 기준보다 약 3.8배 더 엄격해진다.
  - 이탈률이 낮은 구간(예: 7~8%, 장기 고객)에서는 393건의 "이탈"을
    채우려면 전체 표본 5,000명 이상이 한 구간에 몰려야 하므로 Train
    전체(약 4,930건)보다 큰 표본을 요구하는 셈이 되어, 저이탈률
    구간이 구조적으로 항상 통합되어 사라진다 (장기 고객의 변곡점을
    모델이 아예 학습할 기회조차 얻지 못함).
  - Train 전체 이탈 건수(~1,308건) 기준으로는 K=4 이상이 거의 항상
    미달 처리되어, 3번 섹션이 명시한 "K=2~8 넓게 탐색"이 코딩 전부터
    수학적으로 무의미해진다 (1단계 min_size를 완화해도 2단계의 이
    오류 때문에 같은 자리에서 다시 막힘).

따라서 이번 수정에서는 판정 기준을 `n_customers >= min_group_sample_size`
(구간의 "전체 표본 수", 이탈 여부와 무관한 고객 머릿수)로 바꾼다. 이렇게
하면:
  - Train 전체(약 4,930건) 기준으로 최대 4,930 ÷ 393 ≈ 12.5개 구간까지
    구조적으로 수용 가능해져 K=2~8 탐색이 실제로 의미를 가진다.
  - 저이탈률 구간도 "사람 수"만 충분하면 생존할 수 있어, 이탈률이
    극단적으로 불균등한(54% vs 7~8%) 이 데이터의 특성에 더 적합하다.
  - "이탈 여부와 무관하게 시간 분포만 봄"(기획구현.md 3번, 1단계
    입력 정의)이라는 원칙과도 더 잘 맞는다 -- 표본충분성도 같은
    원칙으로 "이탈 여부와 무관하게 고객 수"만 본다.

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
    n_customers: int  # 구간의 전체 표본 수 (표본충분성 판정 기준 -- 정정 이력 참고)
    n_churn: int  # 구간의 이탈 건수 (참고/해석용, 판정 기준 아님)
    churn_rate: float
    meets_threshold: bool  # n_customers >= min_group_sample_size 여부


@dataclass
class PostMergeResult:
    """사후 표본 점검 + 통합 결과."""

    original_boundaries: list
    final_boundaries: list  # 통합 후 최종 경계 (빈 리스트면 단일 세그먼트)
    min_group_sample_size: int  # 판정에 쓰인 임계값 (393명, "전체 표본 수" 기준)
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
                "meets_threshold(전체표본기준)": s.meets_threshold,
            }
            for s in stats_list
        ]
    )


def compute_segment_stats(
    df_train: pd.DataFrame,
    boundaries_tenure: list[int],
    min_group_sample_size: int,
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
) -> list[SegmentStats]:
    """
    주어진 경계로 세그먼트를 나눴을 때, 구간별 전체 표본수·이탈건수를 집계한다.
    표본충분성 판정(meets_threshold)은 전체 표본 수(n_customers) 기준이다
    (정정 이력 참고 -- 이탈 건수 기준 아님).
    """
    churn_binary = df_train[churn_col].map({"Yes": 1, "No": 0})
    if churn_binary.isna().any():
        churn_binary = pd.to_numeric(df_train[churn_col], errors="coerce")

    labels = boundaries_to_labels(df_train[tenure_col], boundaries_tenure)

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
                meets_threshold=(n_customers >= min_group_sample_size),
            )
        )
    return stats


def merge_undersized_segments(
    df_train: pd.DataFrame,
    boundaries_tenure: list[int],
    min_group_sample_size: int,
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
) -> PostMergeResult:
    """
    전체 표본 수(n_customers)가 min_group_sample_size(393명) 미달인 구간을
    인접 구간과 통합한다 (정정 이력 참고 -- 이탈 건수 기준이 아니라 전체
    표본 수 기준). "인접 구간 중 표본이 더 작은 쪽"이 아니라, tenure
    축에서 물리적으로 이웃한 구간과 합치는 것이 자연스러우므로, 항상
    "오른쪽 이웃이 있으면 오른쪽과, 없으면(맨 끝 구간이면) 왼쪽과" 합치는
    규칙을 쓴다. 한 번 합친 뒤 표본을 재집계하고, 모든 구간이 기준을
    충족할 때까지(또는 단일 구간이 될 때까지) 반복한다.

    기획구현.md 3번: "이탈 392건 미달 구간은 인접 구간과 통합 후 재검증"
    -- 단, 이 393이라는 숫자 자체는 "전체 표본 수" 기준임 (정정 이력 참고).
    """
    before_stats = compute_segment_stats(
        df_train, boundaries_tenure, min_group_sample_size, tenure_col, churn_col
    )

    current_boundaries = sorted(boundaries_tenure)
    merge_log = []

    while True:
        stats = compute_segment_stats(
            df_train, current_boundaries, min_group_sample_size, tenure_col, churn_col
        )

        undersized = [s for s in stats if not s.meets_threshold]
        if not undersized or len(current_boundaries) == 0:
            break

        # 가장 표본이 적은(가장 시급한) 미달 구간부터 처리
        target = min(undersized, key=lambda s: s.n_customers)
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
            f"구간{seg_idx}(전체표본 {target.n_customers}명, 이탈 {target.n_churn}건, "
            f"tenure {target.tenure_range})이 {min_group_sample_size}명 미달 -> "
            f"{merge_direction} 구간과 통합 (경계 {boundary_to_remove} 제거)"
        )
        current_boundaries = [b for b in current_boundaries if b != boundary_to_remove]

    after_stats = compute_segment_stats(
        df_train, current_boundaries, min_group_sample_size, tenure_col, churn_col
    )

    return PostMergeResult(
        original_boundaries=sorted(boundaries_tenure),
        final_boundaries=current_boundaries,
        min_group_sample_size=min_group_sample_size,
        merge_log=merge_log,
        before_stats=before_stats,
        after_stats=after_stats,
    )
