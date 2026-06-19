"""
PELT(Pruned Exact Linear Time) 기반 변화점 탐지.
기획구현.md 3번 섹션 "분석 A -- 세그먼트 경계 탐지(시간축)" 의 1단계.

⚠️ 방법론적 위치 (기획구현.md 3번):
  이 모듈은 "순수 통계학"이다. 훈련 데이터로 일반화를 학습하는 과정이 없고,
  주어진 분포(tenure별 이탈률)에 대한 1회의 수학적 최적화일 뿐이다.
  "AI/머신러닝"이라고 불러서는 안 된다.
  머신러닝(교차검증 기반 표본충분성 검증)은 A_step2_methods의 책임이며
  이 모듈은 그 후보를 "제안"하는 역할만 한다.

입력: df_train의 tenure별 이탈률 분포만 사용한다 (이탈 여부 자체가 아니라
"tenure 값에 따른 이탈률"이라는 시간축 분포만 봄으로써 Test 누수를 방지).

핵심 절차
---------
1. (기획구현.md 4-① 사전 범용 기준) statsmodels의 검정력 분석으로 그룹당
   최소 표본수(392건)를 계산한다. 이는 데이터셋과 무관한 재사용 가능 값이며,
   하드코딩하지 않고 함수 출력값을 그대로 쓴다.
2. 392건을 "이탈 건수" 기준으로 잡고, 전체 이탈률로 역산해 tenure 개월
   수 기준 "느슨한 시작값"(약 15.3개월)을 구한다. 이 값은 PELT의
   min_size 파라미터로 쓰이는 시작점일 뿐, 구간별 실제 안정성을 보장하지
   않는다(사후 검증은 2단계의 몫).
3. 위 최소 길이로 "전체기간 ÷ 최소길이"를 역산한 기술적 상한(K≈4~5)을
   참고치로만 두고, 실제 탐색은 K=2~8로 넓게 잡는다. 사전 상한으로
   강제하지 않는다.
4. ruptures.Pelt로 BIC(또는 AIC) 페널티를 적용해 변화점을 탐지하고,
   탐색한 K 범위 전체에 대해 결과를 같이 반환한다 (최종 채택은
   2단계의 사후 표본 점검이 결정하므로 여기서는 후보만 만든다).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt
from statsmodels.stats.power import NormalIndPower

# bic.py / aic.py를 같은 폴더(A_step1_methods)에서 import.
sys.path.append(str(Path(__file__).resolve().parent))
import aic as aic_module  # noqa: E402
import bic as bic_module  # noqa: E402

_CRITERIA = {
    "bic": bic_module,
    "aic": aic_module,
}


# ---------------------------------------------------------------------------
# 4-① 사전 범용 기준: 검정력 분석 (데이터셋 무관, 재사용 가능)
# ---------------------------------------------------------------------------
def compute_min_group_size(
    effect_size: float = 0.2,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """
    Cohen(1988) 표준 효과크기 기준 그룹당 최소 표본수 계산.

    기획구현.md 4번:
      "Cohen(1988) 표준 효과크기(작은 효과크기 h=0.2) + 유의수준 5%
       + 검정력 80%로 검정력 분석. 계산 결과: 그룹당 약 392건"
      "392라는 숫자를 하드코딩하지 말고, NormalIndPower().solve_power(...)
       함수를 코드에 남겨 그 출력값을 임계값으로 사용 (가정값이 바뀌면
       임계값도 자동으로 같이 바뀌도록)"

    Returns
    -------
    int
        그룹(세그먼트)당 필요한 최소 표본(이탈 건수) 수. 올림 처리.
    """
    analysis = NormalIndPower()
    n = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )
    return int(np.ceil(n))


# ---------------------------------------------------------------------------
# tenure별 이탈률 분포 (PELT 입력 신호) 만들기
# ---------------------------------------------------------------------------
def build_tenure_churn_rate_signal(
    df_train: pd.DataFrame,
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
) -> pd.DataFrame:
    """
    tenure(개월) 값별 이탈률을 집계해 PELT 입력 신호를 만든다.

    기획구현.md 3번: "입력: df_train의 tenure별 이탈률 분포만
    (이탈 여부와 무관하게 시간 분포만 봄 -> 누수 방지)"
    즉 개별 행 단위 이탈 여부가 아니라, tenure 값으로 group-by 한
    "이탈률"이라는 1차원 시계열만 PELT에 넣는다.

    Returns
    -------
    pd.DataFrame
        index: tenure (0, 1, 2, ... 정렬됨, 빈 개월은 0건/이탈률 0으로 채움)
        columns: ['n_customers', 'n_churn', 'churn_rate']
    """
    churn_binary = df_train[churn_col].map({"Yes": 1, "No": 0})
    if churn_binary.isna().any():
        # 이미 0/1로 인코딩되어 들어온 경우도 허용
        churn_binary = pd.to_numeric(df_train[churn_col], errors="coerce")

    tmp = pd.DataFrame(
        {"tenure": df_train[tenure_col].values, "churn": churn_binary.values}
    )
    grouped = tmp.groupby("tenure")["churn"].agg(["count", "sum"])
    grouped = grouped.rename(columns={"count": "n_customers", "sum": "n_churn"})

    full_index = pd.RangeIndex(
        start=int(grouped.index.min()), stop=int(grouped.index.max()) + 1
    )
    grouped = grouped.reindex(full_index, fill_value=0)
    grouped["n_churn"] = grouped["n_churn"].astype(int)
    grouped["n_customers"] = grouped["n_customers"].astype(int)
    grouped["churn_rate"] = np.where(
        grouped["n_customers"] > 0,
        grouped["n_churn"] / grouped["n_customers"],
        0.0,
    )
    grouped.index.name = "tenure"
    return grouped


# ---------------------------------------------------------------------------
# 392건(이탈 건수) -> tenure 개월 수로 환산한 "느슨한 시작값"
# ---------------------------------------------------------------------------
def estimate_min_segment_months(
    df_train: pd.DataFrame,
    min_group_churn_count: int,
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
) -> float:
    """
    이탈 건수 기준 최소 표본(예: 392건)을, 전체 평균 이탈률로 역산해
    "tenure 개월 수" 단위의 느슨한 시작값으로 변환한다.

    기획구현.md 3번:
      "최소 세그먼트 길이: 이탈 392건(4번의 사전 범용 기준) 역산값을
       느슨한 시작값으로 사용 -> 약 15.3개월(전체 73개월의 21%).
       단, 이 값은 전체 평균 기반 근사치이며 구간별 실제 안정성을
       보장하지 않는다 (0-17개월/44-72개월의 이탈률이 54%/8%로 크게
       달라 평균 가정이 깨짐). 정밀 검증은 2단계가 담당."

    Returns
    -------
    float
        tenure 개월 수 기준 최소 세그먼트 길이(느슨한 시작값).
    """
    churn_binary = df_train[churn_col].map({"Yes": 1, "No": 0})
    if churn_binary.isna().any():
        churn_binary = pd.to_numeric(df_train[churn_col], errors="coerce")

    overall_churn_rate = churn_binary.mean()
    n_per_month_avg = len(df_train) / (
        df_train[tenure_col].max() - df_train[tenure_col].min() + 1
    )
    # 평균적으로 한 달에 발생하는 이탈 건수
    churn_per_month_avg = n_per_month_avg * overall_churn_rate
    if churn_per_month_avg <= 0:
        raise ValueError("평균 이탈률이 0이라 개월 수 환산이 불가능합니다.")
    return float(min_group_churn_count / churn_per_month_avg)


# ---------------------------------------------------------------------------
# PELT 실행 결과 컨테이너
# ---------------------------------------------------------------------------
@dataclass
class PeltCandidate:
    """K(변화점 수+1 = 세그먼트 수)별 PELT 탐지 결과 1건."""

    n_breakpoints: int  # 변화점 개수 (세그먼트 수 = n_breakpoints + 1)
    breakpoints: list  # tenure 값 기준 경계 (마지막 원소는 신호 끝 인덱스 포함, ruptures 관례)
    boundaries_tenure: list  # 신호 끝 인덱스 제외, 실제 "구간 시작 tenure" 경계만 정리
    penalty_used: float


@dataclass
class PeltResult:
    """PELT 탐지 전체 결과. criterion, 탐색 범위, 신호, 후보 목록을 모두 보존."""

    criterion: str
    signal: pd.DataFrame  # build_tenure_churn_rate_signal 결과
    min_group_churn_count: int  # 검정력 분석 결과 (예: 392)
    min_segment_months: float  # 느슨한 시작값 (예: 약 15.3개월)
    k_search_range: list  # 실제 탐색한 K 범위 (변화점 개수 기준)
    candidates: list = field(default_factory=list)  # list[PeltCandidate]

    def get_candidate(self, n_breakpoints: int) -> PeltCandidate:
        for c in self.candidates:
            if c.n_breakpoints == n_breakpoints:
                return c
        raise KeyError(f"n_breakpoints={n_breakpoints} 후보가 없습니다.")

    def summary_table(self) -> pd.DataFrame:
        """K(세그먼트 수)별 경계 목록을 표로 정리 (사람이 검토하기 쉽게)."""
        columns = ["n_segments", "n_breakpoints", "boundaries_tenure", "penalty"]
        rows = []
        for c in self.candidates:
            rows.append(
                {
                    "n_segments": c.n_breakpoints + 1,
                    "n_breakpoints": c.n_breakpoints,
                    "boundaries_tenure": c.boundaries_tenure,
                    "penalty": c.penalty_used,
                }
            )
        return pd.DataFrame(rows, columns=columns)


def run_pelt(
    df_train: pd.DataFrame,
    criterion: str = "bic",
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
    k_search_range: range | None = None,
    effect_size: float = 0.2,
    alpha: float = 0.05,
    power: float = 0.8,
) -> PeltResult:
    """
    PELT 변화점 탐지 1단계 전체 실행.

    기획구현.md 3번 1단계 절차를 그대로 따른다:
      1) tenure별 이탈률 신호 생성
      2) 검정력 분석으로 최소 표본(이탈 건수) 계산 -> 개월 수로 환산
         (PELT의 min_size로 사용할 "느슨한 시작값")
      3) K=2~8 범위를 폭넓게 탐색 (사전 상한으로 강제하지 않음)
      4) 각 K에 대해 ruptures.Pelt(model="l2")로 변화점 후보를 만든다

    Parameters
    ----------
    df_train : pd.DataFrame
        전처리가 끝난 학습 데이터 (Test 누수 방지를 위해 반드시 Train만).
    criterion : {"bic", "aic"}
        모델 선택 기준. 기본 BIC (기획구현.md: "표본 적은 데이터에 더 안전").
    k_search_range : range, optional
        탐색할 "세그먼트 개수"(K) 범위. 기본값은 기획구현.md 권장대로
        K=2~8 (range(2, 9)).

    Returns
    -------
    PeltResult
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion은 {list(_CRITERIA)} 중 하나여야 합니다.")
    if k_search_range is None:
        k_search_range = range(2, 9)  # K=2~8

    # 1) tenure별 이탈률 신호
    signal_df = build_tenure_churn_rate_signal(
        df_train, tenure_col=tenure_col, churn_col=churn_col
    )
    signal = signal_df["churn_rate"].to_numpy().reshape(-1, 1)
    n_samples = len(signal)

    # 2) 검정력 분석 기반 최소 표본 -> 개월 수 환산 (느슨한 시작값)
    min_group_churn_count = compute_min_group_size(
        effect_size=effect_size, alpha=alpha, power=power
    )
    min_segment_months = estimate_min_segment_months(
        df_train,
        min_group_churn_count=min_group_churn_count,
        tenure_col=tenure_col,
        churn_col=churn_col,
    )
    # PELT min_size는 정수 인덱스 단위(개월 수)라야 하므로 내림 처리하되 최소 1.
    min_size = max(1, int(np.floor(min_segment_months)))

    criterion_module = _CRITERIA[criterion]

    # 3) & 4) K=2~8 폭넓게 탐색. ruptures.Pelt는 pen(페널티)으로 변화점
    #    개수가 결정되는 구조라, K를 직접 지정할 수 없다. 따라서
    #    페널티를 점차 낮춰가며 원하는 K 범위에 도달하는 변화점 집합들을
    #    수집한다 (model="l2": 분산 기반 비용함수).
    algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(signal)

    base_penalty = criterion_module.penalty_value(n_samples=n_samples, n_dims=1)

    candidates: list[PeltCandidate] = []
    seen_breakpoint_counts = set()
    # 페널티 배율을 넓게 스캔하면서 K=2~8에 해당하는 결과들을 모은다.
    for scale in np.linspace(4.0, 0.05, 80):
        pen = base_penalty * scale
        try:
            bkps = algo.predict(pen=pen)
        except Exception:
            continue
        n_breakpoints = len(bkps) - 1  # 마지막 원소는 신호 길이(끝) 포함 -> 실제 경계 수는 -1
        n_segments = n_breakpoints + 1
        if n_segments not in k_search_range:
            continue
        if n_breakpoints in seen_breakpoint_counts:
            continue
        seen_breakpoint_counts.add(n_breakpoints)

        boundary_tenures = [
            int(signal_df.index[idx]) for idx in bkps[:-1]
        ]  # 끝 인덱스 제외, 실제 구간 시작점만

        candidates.append(
            PeltCandidate(
                n_breakpoints=n_breakpoints,
                breakpoints=list(bkps),
                boundaries_tenure=boundary_tenures,
                penalty_used=float(pen),
            )
        )

    candidates.sort(key=lambda c: c.n_breakpoints)

    return PeltResult(
        criterion=criterion_module.name(),
        signal=signal_df,
        min_group_churn_count=min_group_churn_count,
        min_segment_months=min_segment_months,
        k_search_range=list(k_search_range),
        candidates=candidates,
    )
