"""
BIC(Bayesian Information Criterion) 기반 페널티 계산.

기획구현.md 3번 섹션:
  "1단계(PELT+BIC/AIC)는 순수 통계학 -- 훈련 데이터로 일반화를 학습하는
   과정이 없고, 주어진 분포에 대한 1회의 수학적 최적화. 'AI/머신러닝'이라고
   부르면 안 됨"
  "모델선택 기준: BIC 기본 사용 (AIC보다 페널티가 강해 표본 적은 우리
   데이터에 더 안전)"

ruptures의 Pelt 알고리즘은 다음 형태의 비용함수를 최소화한다.

    sum(구간별 비용) + penalty * (변화점 개수)

여기서 penalty 값을 어떻게 주느냐가 곧 "정보기준"의 선택이다.
BIC 형태의 페널티: penalty = log(n) * sigma^2 * dim
(n: 표본 수, dim: 신호 차원. 우리는 tenure별 이탈률이라는 1차원 신호를 쓰므로 dim=1)

aic.py와 동일한 인터페이스(penalty_value(n, ...) -> float)를 제공해
pelt.py에서 BIC/AIC를 손쉽게 교체할 수 있도록 한다.
"""

from __future__ import annotations

import numpy as np


def penalty_value(n_samples: int, n_dims: int = 1, sigma2: float = 1.0) -> float:
    """
    BIC 페널티 값 계산.

    Parameters
    ----------
    n_samples : int
        신호의 길이(관측치 개수). 우리 맥락에서는 tenure 구간 수(0~72개월 -> 73).
    n_dims : int
        신호의 차원 수. 기본값 1 (tenure별 이탈률이라는 단일 시계열).
    sigma2 : float
        신호의 분산 추정치. 기본값 1.0이면 흔히 쓰이는
        penalty = log(n) * n_dims 형태가 된다.
        ruptures 예제 관례를 따라 신호 분산을 곱해 스케일을 맞출 수도 있다.

    Returns
    -------
    float
        ruptures.Pelt(...).predict(pen=penalty_value(...)) 에 넘길 페널티 값.
    """
    if n_samples <= 1:
        raise ValueError("n_samples는 2 이상이어야 BIC 페널티를 계산할 수 있습니다.")
    return float(np.log(n_samples) * n_dims * sigma2)


def name() -> str:
    """pelt.py / main.py 등에서 어떤 기준이 쓰였는지 로깅할 때 사용."""
    return "BIC"
