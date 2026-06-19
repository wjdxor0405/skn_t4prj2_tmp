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

⚠️ 수정 이력 ② (분모 정의 오류 수정, 표본충분성 논리 재검증 -- 두 번째 발견)
--------------------------------------------------------------------
①의 수정만으로는 부족했다. compute_min_group_size()가 계산하는
392.44(올림 393)를 "그룹당 필요한 최소 *이탈 건수*"로 잘못 해석해
2단계(segment_merge.py)에서 `n_churn >= 393`으로 판정하고 있었다.

statsmodels.stats.proportion.power_proportions_2indep의 nobs1
파라미터는 공식 문서에 "number of observations in sample 1"로
명시되어 있다 -- "표본의 전체 관측치 수"이지 "표본 안에서 어떤 사건이
일어난 건수"가 아니다. 실제로 nobs1=393(전체 표본)으로 교차검증하면
정확히 검정력 80%가 재현된다. 즉 393은 처음부터 "그룹당 전체 표본
수(고객 머릿수)" 기준이었다. 이를 "이탈 건수" 기준으로 잘못 쓰면:
  - 이탈률 26.5% 구간에서 같은 검정력을 얻으려면 실제로는
    393 ÷ 0.265 ≈ 1,483건의 전체 표본이 필요해져, 검정력 분석이
    원래 요구한 기준보다 약 3.8배 더 엄격해진다.
  - 이탈률이 낮은 구간(예: 7~8%, 장기 고객)에서는 393건의 "이탈"을
    채우려면 전체 표본 5,000명 이상이 한 구간에 몰려야 하므로 Train
    전체보다 큰 표본을 요구하는 셈이 되어, 저이탈률 구간이 구조적으로
    항상 통합되어 사라진다 (장기 고객의 변곡점을 모델이 아예 학습할
    기회조차 얻지 못함).
  - Train 전체 이탈 건수(~1,308건) 기준으로는 ①의 수정(min_size 완화)
    으로 1단계가 K=8까지 후보를 내더라도, 2단계가 같은 자리(K≈3.3)에서
    다시 막아버려 ①의 수정 효과가 무력화된다.

따라서 이번 수정에서는:
  - compute_min_group_size()의 반환값(393)과 의미는 그대로 두되,
    변수명을 min_group_churn_count -> min_group_sample_size로 바꿔
    "전체 표본 수" 기준임을 명확히 한다.
  - 2단계(segment_merge.py)의 판정 기준을 `n_churn >= 임계값`에서
    `n_customers >= 임계값`으로 바꾼다 (이탈 여부와 무관하게 구간의
    고객 머릿수만 본다 -- 오히려 3번 섹션의 "이탈 여부와 무관하게
    시간 분포만 봄" 원칙과 더 합치한다).
  - estimate_min_segment_months()도 더 이상 이탈률로 나누지 않고
    "월별 평균 고객 수"로 바로 나누도록 단순화한다.

⚠️ 수정 이력 ① (min_size 산정 방식 변경)
--------------------------------------------------------------------
이전 버전은 392건(검정력 분석 출력값) 역산 개월수(약 15.3개월)를 PELT의
min_size로 그대로 사용했다. 그런데 실데이터로 검증하면:
  - Train 이탈 건수 ≈ 1,308건 (전체 1,869건 × 0.7)
  - 1,308건 ÷ 392건 ≈ 3.3  ->  2단계(사후 통합) 기준으로도 K=4 이상은
    이론상 거의 항상 미달 통합된다.
  - 게다가 15.3개월을 1단계 min_size로 "하드 제약"까지 걸면
    73개월 ÷ 15.3개월 ≈ 4.8  ->  1단계 PELT 자체가 K=4~5조차 후보로
    내지 못하는 경우가 생긴다.
기획구현.md 3번은 이를 "기술적 상한 K=4~5는 참고치일 뿐, 사전 상한으로
강제하지 않고 K=2~8로 넓게 탐색"하라고 명시했고, "이 15.3개월은 못 믿는
근사치이니 느슨한 시작값으로만 쓰고 정밀 검증은 2단계가 담당"한다고도
명시했다. 즉 392건 기준으로 1단계 min_size를 하드 제약하는 것은
문서가 명시적으로 금지한 행동이었다.

따라서 이 수정에서는:
  - 392건(검정력 분석) 기준은 여전히 compute_min_group_size()로 계산하되,
    더 이상 PELT의 min_size로 쓰지 않는다. 표본충분성 "판정"은 전부
    2단계(A_step2_methods.segment_merge)에서만 일어난다.
  - 1단계 min_size는 k_search_range의 최대 K(기본 8)가 물리적으로
    후보로 나올 수 있도록 보장하는, 훨씬 더 작은 "진짜 느슨한" 가드레일
    (estimate_loose_min_size)로 바꾼다. 이 값은 "통계적으로 그럴듯한
    최소 길이"가 아니라 "이보다 작으면 ruptures가 분산을 계산할 수조차
    없다"는 최소한의 하한일 뿐이다.

핵심 절차 (수정 후)
-------------------
1. (기획구현.md 4-① 사전 범용 기준) statsmodels 검정력 분석으로
   "전체 표본 수" 기준 393명을 계산한다 (참고치 + 2단계 전달용. 1단계
   후보 생성에는 더 이상 하드 제약으로 쓰지 않는다).
2. 393명의 개월 환산값도 함께 계산해 "기술적 상한"을 사람이 확인할
   수 있게 남긴다 (참고용, PELT 실행에는 영향 없음).
3. k_search_range의 최댓값(기본 K=8)이 후보로 나올 수 있는 만큼만 작은
   min_size를 따로 계산해 실제 PELT 실행에 사용한다.
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
    Cohen(1988) 표준 효과크기 기준, 두 그룹의 비율(이탈률) 차이를
    통계적으로 유의하게 검출하기 위해 그룹당 필요한 "전체 표본 수"
    (분모, 고객 머릿수)를 계산한다.

    기획구현.md 4번:
      "Cohen(1988) 표준 효과크기(작은 효과크기 h=0.2) + 유의수준 5%
       + 검정력 80%로 검정력 분석. 계산 결과: 그룹당 약 392건"
      "392라는 숫자를 하드코딩하지 말고, NormalIndPower().solve_power(...)
       함수를 코드에 남겨 그 출력값을 임계값으로 사용 (가정값이 바뀌면
       임계값도 자동으로 같이 바뀌도록)"

    ⚠️ 정정 이력 (표본충분성 논리 재검증, 두 번째 발견 -- 분모 정의 오류):
      이전 버전은 이 함수의 출력값을 "그룹당 필요한 최소 *이탈 건수*"로
      해석해 변수명도 min_group_churn_count로 두고, 2단계
      (segment_merge.py)에서 `n_churn >= 임계값`으로 판정했다. 이는
      통계적으로 틀렸다.

      NormalIndPower().solve_power()와 같은 계열인
      statsmodels.stats.proportion.power_proportions_2indep의 nobs1
      파라미터는 공식 문서에 "number of observations in sample 1"로
      명시되어 있다 -- 즉 "표본 1의 *전체 관측치 수*"이지 "표본 1
      안에서 어떤 사건이 일어난 건수"가 아니다. 실제로
      power_proportions_2indep(diff=..., prop2=..., nobs1=393, ...)으로
      교차검증하면 nobs1=393(전체 표본)일 때 정확히 검정력 80%가
      재현된다 -- 즉 이 함수가 계산하는 392.44(올림 393)는 처음부터
      "그룹당 전체 표본 수"였다. "이탈 건수"로 쓰면:
        - 이탈률이 26.5%인 경우, 같은 검정력을 얻으려면 실제로는
          392 ÷ 0.265 ≈ 1,479건의 전체 표본이 필요해져, 검정력
          분석이 원래 요구한 기준보다 약 3.8배 더 엄격해진다.
        - 이탈률이 낮은 구간(예: 7~8%)에서는 392건의 이탈을 채우려면
          전체 표본 5,000~5,600명이 한 구간에 몰려야 하므로, Train
          전체보다 큰 표본을 요구하는 셈이 되어 저이탈률(장기 고객)
          구간이 구조적으로 항상 통합되어 사라진다.
        - Train 전체 이탈 건수(~1,308건) 기준으로는 K=4 이상이 거의
          항상 미달 처리되어, 3번 섹션이 명시한 "K=2~8 넓게 탐색"이
          무의미해진다.
      따라서 이 함수의 반환값은 변경 없이 그대로 두되(392.44를 올림해
      393 반환, 여전히 statsmodels 출력값을 하드코딩 없이 그대로 씀),
      호출하는 쪽(segment_merge.py)의 판정 기준을 "구간의 전체 표본 수
      (n_customers)"로 바꾼다. 이름도 compute_min_group_size로 그대로
      두되, 사용처의 변수명을 min_group_churn_count ->
      min_group_sample_size로 바꿔 의미를 명확히 한다.

    Returns
    -------
    int
        그룹(세그먼트)당 필요한 최소 "전체 표본 수"(고객 머릿수). 올림 처리.
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
# 393명(전체 표본) -> tenure 개월 수로 환산한 값 (참고용, 기술적 상한 설명용)
# ---------------------------------------------------------------------------
def estimate_min_segment_months(
    df_train: pd.DataFrame,
    min_group_sample_size: int,
    tenure_col: str = "tenure",
) -> float:
    """
    검정력 분석 기준 최소 "전체 표본 수"(예: 393명)를, 개월당 평균
    고객 수로 역산해 "tenure 개월 수" 단위로 환산한다.

    ⚠️ 정정 이력 (분모 정의 오류 수정 -- 표본충분성 논리 재검증, 두 번째 발견):
      이전 버전은 393을 "이탈 건수" 기준으로 보고 전체 이탈률(약 26.5%)로
      나눠서 개월수를 역산했다 (393 ÷ 평균이탈률 ÷ 월별고객수). 그런데
      compute_min_group_size()의 393은 애초에 "전체 표본 수"였으므로
      (해당 함수 docstring의 정정 이력 참고), 이탈률로 나누는 절차 자체가
      불필요하고 틀렸다. 전체 표본 수 기준이면 그냥 "월별 평균 고객 수"로
      바로 나누면 된다 (이탈 여부와 무관하게 고객 머릿수만 보면 됨 --
      오히려 원래 기획구현.md 3번이 명시한 "이탈 여부와 무관하게 시간
      분포만 봄" 원칙과 더 잘 맞는다).

    ⚠️ 이 값은 더 이상 PELT의 min_size로 쓰이지 않는다 (이전 수정 이력
    참고). "전체기간 ÷ 이 값"이라는 기술적 상한을 사람이 보고서에서
    확인할 수 있도록 참고치로만 PeltResult에 남겨둔다. 393명 기준의
    실제 판정(통과/미달, 구간 통합)은 전부 2단계
    (A_step2_methods.segment_merge)에서 "전체 표본 수" 기준으로 수행한다.

    Returns
    -------
    float
        tenure 개월 수 기준 환산값 (참고용).
    """
    n_per_month_avg = len(df_train) / (
        df_train[tenure_col].max() - df_train[tenure_col].min() + 1
    )
    if n_per_month_avg <= 0:
        raise ValueError("월별 평균 고객 수가 0이라 개월 수 환산이 불가능합니다.")
    return float(min_group_sample_size / n_per_month_avg)


# ---------------------------------------------------------------------------
# 1단계 PELT가 실제로 사용할 "진짜 느슨한" min_size
# (k_search_range의 최대 K가 물리적으로 후보로 나올 수 있도록 보장)
# ---------------------------------------------------------------------------
def estimate_loose_min_size(
    df_train: pd.DataFrame,
    k_search_range: range,
    tenure_col: str = "tenure",
    floor_months: int = 2,
) -> int:
    """
    PELT의 min_size로 실제 사용할 값을 계산한다.

    설계 원칙: 이 값은 "표본충분성을 판정"하는 기준이 아니라, ruptures가
    구간의 분산을 계산할 수 있는 최소한의 길이를 보장하기 위한
    가드레일일 뿐이다. 392/393건 기준의 표본충분성 판정은 전부 2단계
    (segment_merge.py)가 담당하므로(기획구현.md 3번: "정밀 검증은
    2단계가 담당"), 여기서는 k_search_range가 요구하는 최대 K
    (기본 8)가 최소한 PELT 후보로는 나올 수 있을 만큼 작게 잡는다.

    계산: 전체 tenure 기간을 k_search_range의 최댓값으로 균등 분할했을
    때의 구간 길이보다 살짝 작게 잡는다(균등분할보다 작아야 비균등
    경계도 K=max까지 표현 가능). 다만 floor_months(기본 2개월) 미만으로는
    내려가지 않는다 -- 1개월 단위로는 분산 추정 자체가 불안정해지므로
    "느슨함"에도 최소한의 하한은 둔다.

    Parameters
    ----------
    k_search_range : range
        1단계가 탐색할 K(세그먼트 수) 범위. 이 범위의 최댓값이 기준이 된다.

    Returns
    -------
    int
        PELT min_size로 쓸 tenure 개월 수.
    """
    tenure_span = (
        df_train[tenure_col].max() - df_train[tenure_col].min() + 1
    )
    k_max = max(k_search_range)
    # 균등분할 길이보다 확실히 작게(여유 0.7배) 잡아서, 비균등 경계도
    # k_max까지 표현 가능하게 한다.
    candidate = int(np.floor((tenure_span / k_max) * 0.7))
    return max(floor_months, candidate)


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
    min_group_sample_size: int  # 검정력 분석 결과 (392/393명, "전체 표본 수" 기준. 2단계 판정에 사용)
    min_segment_months_reference: float  # 393명 역산 개월수 (참고용, 더 이상 min_size 아님)
    technical_k_upper_bound: float  # 위 참고치로 역산한 "기술적 상한" K (참고용, 강제 안 함)
    min_size_used: int  # 실제 PELT에 사용된 min_size (estimate_loose_min_size 출력)
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

    기획구현.md 3번 1단계 절차 + 수정 이력 ①②반영:
      1) tenure별 이탈률 신호 생성
      2) 검정력 분석으로 "전체 표본 수" 기준 393명을 계산한다 (이탈
         건수가 아님 -- 수정 이력 ② 참고). 이 값과 그 개월 환산치
         (min_segment_months_reference)는 "기술적 상한"을 설명하는
         참고치로만 PeltResult에 남기고, 더 이상 PELT의 min_size로
         쓰지 않는다 (수정 이력 ① 참고).
      3) k_search_range의 최대 K(기본 8)가 실제로 후보로 나올 수 있도록,
         훨씬 더 작은 min_size(estimate_loose_min_size)를 따로 계산해
         PELT 실행에 사용한다. -> "사전 상한으로 강제하지 않고 K=2~8
         넓게 탐색"이라는 기획 의도를 실제로 충족.
      4) 각 K에 대해 ruptures.Pelt(model="l2")로 변화점 후보를 만든다.
         393명("전체 표본 수") 기준의 표본충분성 "판정"(통과/미달,
         구간 통합)은 여기서 하지 않고 전부 2단계(segment_merge.py)로
         넘긴다.

    Parameters
    ----------
    df_train : pd.DataFrame
        전처리가 끝난 학습 데이터 (Test 누수 방지를 위해 반드시 Train만).
    criterion : {"bic", "aic"}
        모델 선택 기준. 기본 BIC (기획구현.md: "표본 적은 데이터에 더 안전").
    k_search_range : range, optional
        탐색할 "세그먼트 개수"(K) 범위. 기본값은 기획구현.md 권장대로
        K=2~8 (range(2, 9)). 이 범위의 최댓값이 min_size 산정 기준이 된다.

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

    # 2) 검정력 분석 기반 392/393명 기준("전체 표본 수") -- 참고치로만 계산
    #    (min_size로 쓰지 않음; 정정 이력: 이탈 건수가 아니라 전체 표본 수 기준)
    min_group_sample_size = compute_min_group_size(
        effect_size=effect_size, alpha=alpha, power=power
    )
    min_segment_months_reference = estimate_min_segment_months(
        df_train,
        min_group_sample_size=min_group_sample_size,
        tenure_col=tenure_col,
    )
    tenure_span = df_train[tenure_col].max() - df_train[tenure_col].min() + 1
    technical_k_upper_bound = float(tenure_span / min_segment_months_reference)

    # 3) 실제 PELT에 사용할 "진짜 느슨한" min_size
    #    (k_search_range 최대 K가 물리적으로 후보로 나올 수 있도록 보장)
    min_size = estimate_loose_min_size(
        df_train, k_search_range=k_search_range, tenure_col=tenure_col
    )

    criterion_module = _CRITERIA[criterion]

    # 3) & 4) K=2~8 폭넓게 탐색. ruptures.Pelt는 pen(페널티)으로 변화점
    #    개수가 결정되는 구조라, K를 직접 지정할 수 없다. 따라서
    #    페널티를 점차 낮춰가며 원하는 K 범위에 도달하는 변화점 집합들을
    #    수집한다 (model="l2": 분산 기반 비용함수).
    #    min_size를 줄인 만큼(수정 이력 참고) 더 잘게 쪼개려면 더 작은
    #    페널티가 필요할 수 있어, 스캔 하한을 0.05 -> 0.001로 낮춰
    #    k_search_range 최댓값까지 도달할 여지를 넉넉히 둔다.
    algo = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(signal)

    base_penalty = criterion_module.penalty_value(n_samples=n_samples, n_dims=1)

    candidates: list[PeltCandidate] = []
    seen_breakpoint_counts = set()
    # 페널티 배율을 넓게(로그 스케일로) 스캔하면서 k_search_range에
    # 해당하는 결과들을 모은다.
    scales = np.concatenate(
        [
            np.linspace(4.0, 0.05, 60),
            np.logspace(np.log10(0.05), np.log10(0.0005), 60),
        ]
    )
    for scale in scales:
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
        min_group_sample_size=min_group_sample_size,
        min_segment_months_reference=min_segment_months_reference,
        technical_k_upper_bound=technical_k_upper_bound,
        min_size_used=min_size,
        k_search_range=list(k_search_range),
        candidates=candidates,
    )
