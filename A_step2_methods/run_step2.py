"""
분석 A 2단계 전체 오케스트레이션.
기획구현.md 3번 "2단계 -- 머신러닝 기반 표본충분성 검증 + 사후 통합" 전체를
하나의 함수로 묶어서 실행한다. main.py가 직접 이 모듈만 호출하면 되도록
인터페이스를 단순화하는 역할.

⚠️ 정정 이력 (실행 순서 오류 -- 세 번째 발견)
--------------------------------------------------------------------
이전 버전은 a(XGBoost 교차검증), b(사후 표본 점검+통합), c(부트스트랩 CV)
세 단계가 전부 1단계 원본 경계(boundary_candidates)를 독립적으로 입력받아
"병렬로" 실행됐다. 즉 b가 만든 "통합 후 경계"(final_boundaries)가 a의
입력으로 전혀 쓰이지 않았다.

기획구현.md 3번 원문은 이를 명시적으로 금지한다:
  "이탈 392건 미달 구간은 인접 구간과 통합 후 재검증"
  "최종 K와 경계는 위 사후 점검을 통과한 구조로 확정"
"통합 후 재검증"이라는 표현과 "사후 점검을 통과한 구조"로 최종 K를
확정한다는 표현 모두, XGBoost 성능 평가가 통합 *이후* 구조를 대상으로
이뤄져야 한다는 뜻이다. 실제로 같은 K=8 후보를 통합 전/후로 각각 학습
시켜보면 성능 수치가 달라지는 것을 확인했다(통합 전: 표본부족 구간이
섞인 채로 측정된, 신뢰할 수 없는 성능).

따라서 이번 수정에서는 실행 순서를 다음과 같이 바꾼다:
  1. (4번-①) 검정력 분석으로 임계값(393명, "전체 표본 수" 기준) 계산
  2. (3번) 1단계가 제안한 K=2~8 후보별로:
       a. 먼저 사후 표본 점검 + 393명 미달 구간 통합을 수행해
          "최종 경계"(final_boundaries)를 확정한다 (segment_merge.py)
       b. 그 최종 경계로 XGBoost 교차검증을 재실행해 성능을 측정한다
          (xgboost_cv.py) -- "통합 후 재검증"을 글자 그대로 구현
       c. 부트스트랩 변동계수(CV)는 "이 1단계 제안 구조가 애초에 얼마나
          불안정했는지"를 보여주는 게 목적이므로, 여전히 통합 *전* 원본
          경계 기준으로 계산한다 (이 부분은 의도적으로 원본 유지)
  3. 위 표들을 모두 사람이 볼 수 있게 반환 -- 최종 K 채택은 자동화하지
     않고 사람이 종합 판단 (기획구현.md 4번-②)

cv_results의 boundaries_tenure는 이제 "통합 후 최종 경계"를 담는다.
원본(1단계 제안) 경계와 비교하고 싶다면 merge_results의
original_boundaries를 함께 참고하면 된다. 통합으로 여러 원본 후보가
같은 최종 구조로 수렴하는 경우, XGBoost CV는 같은 구조를 중복 계산하지
않도록 최종 경계 기준으로 중복 제거한다.
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

    min_group_sample_size: int
    cv_results: list  # list[xgboost_cv.CVResult], "통합 후 최종 경계" 기준
    bootstrap_results: list  # list[bootstrap_cv.BootstrapCVResult], 통합 전 원본 경계 기준
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
        검정력 분석 기반 min_group_sample_size를 이미 담고 있다.
    """
    boundary_candidates = [c.boundaries_tenure for c in pelt_result.candidates]
    min_group_sample_size = pelt_result.min_group_sample_size

    # a. 사후 표본 점검 + 393명("전체 표본 수" 기준) 미달 구간 통합을
    #    먼저 수행해 "최종 경계"를 확정한다 (정정 이력 참고 -- XGBoost
    #    CV보다 먼저 실행되어야 "통합 후 재검증"이 성립한다)
    merge_results = [
        segment_merge.merge_undersized_segments(
            df_train,
            boundaries,
            min_group_sample_size,
            tenure_col=tenure_col,
            churn_col=churn_col,
        )
        for boundaries in boundary_candidates
    ]

    # b. 위에서 확정한 "최종 경계"로 XGBoost 교차검증을 재실행한다.
    #    여러 원본 후보가 통합 후 같은 구조로 수렴할 수 있으므로
    #    (예: K=5와 K=6이 둘 다 K=4로 통합되는 경우), 최종 경계 기준으로
    #    중복을 제거해 같은 구조를 두 번 학습하지 않는다.
    seen_final = set()
    unique_final_boundaries = []
    for m in merge_results:
        key = tuple(m.final_boundaries)
        if key not in seen_final:
            seen_final.add(key)
            unique_final_boundaries.append(m.final_boundaries)

    cv_results = xgboost_cv.compare_candidates(
        df_train,
        unique_final_boundaries,
        tenure_col=tenure_col,
        churn_col=churn_col,
        n_splits=n_cv_splits,
        random_state=random_state,
    )

    # c. 부트스트랩 변동계수는 "1단계가 제안한 원본 구조가 애초에 얼마나
    #    불안정했는지"를 보여주는 게 목적이므로, 의도적으로 통합 *전*
    #    원본 경계 기준 그대로 둔다.
    bootstrap_results = bootstrap_cv.compare_candidates(
        df_train,
        boundary_candidates,
        n_bootstrap=n_bootstrap,
        tenure_col=tenure_col,
        churn_col=churn_col,
        random_state=random_state,
    )

    return Step2Result(
        min_group_sample_size=min_group_sample_size,
        cv_results=cv_results,
        bootstrap_results=bootstrap_results,
        merge_results=merge_results,
    )
