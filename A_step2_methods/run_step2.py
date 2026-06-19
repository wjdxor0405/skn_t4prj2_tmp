"""
분석 A 2단계 전체 오케스트레이션.
기획구현.md 3번 "2단계 -- 머신러닝 기반 표본충분성 검증 + 사후 통합" 전체를
하나의 함수로 묶어서 실행한다. main.py가 직접 이 모듈만 호출하면 되도록
인터페이스를 단순화하는 역할.

절차 (기획구현.md 3, 4번 그대로):
  1. (4번-①) 검정력 분석으로 임계값(392건) 계산 -- A_step1_methods.pelt 재사용
  2. (3번) 1단계가 제안한 K=2~8 후보별로:
       a. XGBoost 교차검증 성능 측정 (xgboost_cv.py)
       b. 사후 표본 점검 + 392건 미달 구간 통합 (segment_merge.py)
       c. 부트스트랩 변동계수(CV) 계산 (bootstrap_cv.py, 통합 "전" 원본 경계 기준)
  3. 위 표들을 모두 사람이 볼 수 있게 반환 -- 최종 K 채택은 자동화하지
     않고 사람이 종합 판단 (기획구현.md 4번-②)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "A_step1_methods"))

import bootstrap_cv  # noqa: E402
import segment_merge  # noqa: E402
import xgboost_cv  # noqa: E402
import pelt  # noqa: E402


@dataclass
class Step2Result:
    """분석 A 2단계 실행 결과 전체. 사람이 검토할 모든 표를 담는다."""

    min_group_churn_count: int
    cv_results: list  # list[xgboost_cv.CVResult]
    bootstrap_results: list  # list[bootstrap_cv.BootstrapCVResult]
    merge_results: list  # list[segment_merge.PostMergeResult], 1단계 후보별

    def cv_table(self) -> pd.DataFrame:
        return xgboost_cv.results_to_table(self.cv_results)

    def bootstrap_table(self) -> pd.DataFrame:
        return bootstrap_cv.overall_summary_table(self.bootstrap_results)

    def merge_summary_table(self) -> pd.DataFrame:
        """후보별 원래 경계 -> 통합 후 경계를 한눈에 보는 표."""
        columns = [
            "original_n_segments",
            "original_boundaries",
            "final_n_segments",
            "final_boundaries",
            "n_merges",
        ]
        rows = []
        for m in self.merge_results:
            rows.append(
                {
                    "original_n_segments": len(m.original_boundaries) + 1,
                    "original_boundaries": m.original_boundaries,
                    "final_n_segments": len(m.final_boundaries) + 1,
                    "final_boundaries": m.final_boundaries,
                    "n_merges": len(m.merge_log),
                }
            )
        return pd.DataFrame(rows, columns=columns)


def run_step2(
    df_train: pd.DataFrame,
    pelt_result: "pelt.PeltResult",
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
    n_cv_splits: int = 5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> Step2Result:
    """
    분석 A 2단계 전체 실행.

    Parameters
    ----------
    df_train : pd.DataFrame
        1단계와 동일한 Train 데이터 (누수 방지를 위해 반드시 Train만 사용).
    pelt_result : pelt.PeltResult
        1단계(A_step1_methods.pelt.run_pelt) 실행 결과. K=2~8 후보들과
        검정력 분석 기반 min_group_churn_count를 이미 담고 있다.
    """
    boundary_candidates = [c.boundaries_tenure for c in pelt_result.candidates]
    min_group_churn_count = pelt_result.min_group_churn_count

    # a. XGBoost 교차검증 (K=1 베이스라인 포함, compare_candidates 내부 처리)
    cv_results = xgboost_cv.compare_candidates(
        df_train,
        boundary_candidates,
        tenure_col=tenure_col,
        churn_col=churn_col,
        n_splits=n_cv_splits,
        random_state=random_state,
    )

    # b. 사후 표본 점검 + 392건 미달 구간 통합 (후보별로 개별 수행)
    merge_results = [
        segment_merge.merge_undersized_segments(
            df_train,
            boundaries,
            min_group_churn_count,
            tenure_col=tenure_col,
            churn_col=churn_col,
        )
        for boundaries in boundary_candidates
    ]

    # c. 부트스트랩 변동계수 (통합 전 원본 경계 기준 -- "이 구조가 애초에
    #    얼마나 불안정한지"를 보여주는 게 목적이므로 원본을 그대로 사용)
    bootstrap_results = bootstrap_cv.compare_candidates(
        df_train,
        boundary_candidates,
        n_bootstrap=n_bootstrap,
        tenure_col=tenure_col,
        churn_col=churn_col,
        random_state=random_state,
    )

    return Step2Result(
        min_group_churn_count=min_group_churn_count,
        cv_results=cv_results,
        bootstrap_results=bootstrap_results,
        merge_results=merge_results,
    )
