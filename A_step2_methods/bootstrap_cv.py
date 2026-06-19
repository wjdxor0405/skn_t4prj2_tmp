"""
부트스트랩 변동계수(CV = 표준편차/평균) 기반 표본충분성 사후검증.
기획구현.md 4번-② "사후 데이터 특화 검증 (현재 데이터에 맞춰 직접 확인)":
  "이탈 '건수'의 부트스트랩 변동계수(CV = 표준편차/평균)를 K별로 계산해
   표로 제시"
  "평균 이탈률의 신뢰구간 폭은 쓰지 않음 (K=1이 항상 유리해지는 왜곡
   확인됨, 폐기됨)"
  "임계값은 자동 탐지하지 않음 (2차 미분 기반 꺾임 탐지는 부트스트랩
   시드 바꾸면 결론이 바뀌는 재현성 문제로 폐기 확인됨) -> CV·성능 표를
   투명하게 제시하고 사람이 종합 판단"

⚠️ 기획구현.md 4번 "7번과의 관계" 섹션에 따라, 이 모듈은 7번(경계
재현성 검증, "PELT가 찾은 변화점의 위치가 재추출해도 같은 곳에서
나오는가")과는 측정 대상이 다른 별도 함수다.
  - 이 모듈: "이미 정해진 경계로 나눈 구간의 이탈 '건수'가 재추출해도
    안정적인가" (boundaries_tenure를 고정하고 부트스트랩)
  - 7번(추후 별도 구현 예정): "경계의 '위치' 자체가 재추출해도 같은
    곳에서 나오는가" (매 부트스트랩마다 PELT를 다시 돌림)
같은 재추출(resample) 기법을 쓰지만 자동으로 결과를 공유하지 않으며,
이 파일은 7번 구현이 추가되더라도 그대로 둔다(목적이 다르므로 통합 금지).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from segment_merge import boundaries_to_labels


@dataclass
class SegmentBootstrapCV:
    """세그먼트 1개의 이탈 건수 부트스트랩 변동계수."""

    segment: int
    observed_n_churn: int
    bootstrap_mean: float
    bootstrap_std: float
    cv: float  # 변동계수 = std/mean. 작을수록 안정적(표본이 충분).


@dataclass
class BootstrapCVResult:
    """K(경계 구조) 1개에 대한 전체 부트스트랩 CV 결과."""

    n_segments: int
    boundaries_tenure: list
    n_bootstrap: int
    per_segment: list  # list[SegmentBootstrapCV]
    overall_cv: float  # 세그먼트별 CV의 평균 (K 구조 전체를 한 줄로 비교하기 위한 요약치)

    def to_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "segment": s.segment,
                    "observed_n_churn": s.observed_n_churn,
                    "bootstrap_mean": round(s.bootstrap_mean, 2),
                    "bootstrap_std": round(s.bootstrap_std, 2),
                    "cv": round(s.cv, 4),
                }
                for s in self.per_segment
            ]
        )


def bootstrap_churn_count_cv(
    df_train: pd.DataFrame,
    boundaries_tenure: list[int],
    n_bootstrap: int = 1000,
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
    random_state: int = 42,
) -> BootstrapCVResult:
    """
    고정된 경계(boundaries_tenure)로 세그먼트를 나눈 뒤, 행 단위
    복원추출(resample with replacement)을 n_bootstrap회 반복하면서
    구간별 "이탈 건수"의 평균/표준편차/변동계수(CV)를 계산한다.

    평균 이탈률(비율)의 신뢰구간이 아니라 "건수"의 CV를 쓰는 이유는
    기획구현.md 4번-②에 명시된 대로, 비율 기반 지표는 K=1(구간을
    안 나누는 경우)이 표본이 가장 많아 항상 유리하게 왜곡되는 문제가
    확인되어 폐기되었기 때문이다.
    """
    rng = np.random.default_rng(random_state)

    churn_binary = df_train[churn_col].map({"Yes": 1, "No": 0})
    if churn_binary.isna().any():
        churn_binary = pd.to_numeric(df_train[churn_col], errors="coerce")

    if len(boundaries_tenure) == 0:
        labels = pd.Series(0, index=df_train.index)
        n_segments = 1
    else:
        labels = boundaries_to_labels(df_train[tenure_col], boundaries_tenure)
        n_segments = len(boundaries_tenure) + 1

    n = len(df_train)
    labels_arr = labels.to_numpy()
    churn_arr = churn_binary.to_numpy()

    # 세그먼트별 이탈 건수를 담을 (n_bootstrap, n_segments) 배열
    boot_counts = np.zeros((n_bootstrap, n_segments), dtype=int)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)  # 복원추출
        seg_b = labels_arr[idx]
        churn_b = churn_arr[idx]
        for seg in range(n_segments):
            boot_counts[b, seg] = int(churn_b[seg_b == seg].sum())

    observed_counts = [
        int(churn_arr[labels_arr == seg].sum()) for seg in range(n_segments)
    ]

    per_segment = []
    for seg in range(n_segments):
        col = boot_counts[:, seg]
        mean_ = float(col.mean())
        std_ = float(col.std(ddof=1))
        cv_ = std_ / mean_ if mean_ > 0 else np.inf
        per_segment.append(
            SegmentBootstrapCV(
                segment=seg,
                observed_n_churn=observed_counts[seg],
                bootstrap_mean=mean_,
                bootstrap_std=std_,
                cv=cv_,
            )
        )

    overall_cv = float(np.mean([s.cv for s in per_segment]))

    return BootstrapCVResult(
        n_segments=n_segments,
        boundaries_tenure=sorted(boundaries_tenure),
        n_bootstrap=n_bootstrap,
        per_segment=per_segment,
        overall_cv=overall_cv,
    )


def compare_candidates(
    df_train: pd.DataFrame,
    boundary_candidates: list[list[int]],
    n_bootstrap: int = 1000,
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
    random_state: int = 42,
) -> list[BootstrapCVResult]:
    """
    여러 경계 후보에 대해 일괄적으로 부트스트랩 CV를 계산한다.
    K=1(세그먼트 없음) 베이스라인도 자동으로 맨 앞에 포함한다.
    """
    results = []
    results.append(
        bootstrap_churn_count_cv(
            df_train, [], n_bootstrap, tenure_col, churn_col, random_state
        )
    )
    for boundaries in boundary_candidates:
        results.append(
            bootstrap_churn_count_cv(
                df_train, boundaries, n_bootstrap, tenure_col, churn_col, random_state
            )
        )
    return results


def overall_summary_table(results: list[BootstrapCVResult]) -> pd.DataFrame:
    """K별 overall_cv를 한 줄씩 비교할 수 있는 요약 표."""
    return pd.DataFrame(
        [
            {
                "n_segments": r.n_segments,
                "boundaries_tenure": r.boundaries_tenure,
                "overall_cv": round(r.overall_cv, 4),
                "n_bootstrap": r.n_bootstrap,
            }
            for r in results
        ]
    )
