"""
AIC(Akaike Information Criterion) 기반 페널티 계산.

기획구현.md 3번 섹션: 기본은 BIC를 사용하지만(표본이 적은 우리 데이터에
페널티가 더 강한 BIC가 더 안전), 확장성/비교 목적으로 AIC도 같은
인터페이스(penalty_value)로 제공해 pelt.py에서 criterion 파라미터로
손쉽게 교체할 수 있게 한다.

AIC 페널티는 표본 수에 무관하게 차원당 상수 페널티를 준다.
    penalty = 2 * n_dims * sigma2
(BIC는 log(n)이 곱해져 표본이 많을수록 페널티가 커지지만, AIC는 고정.
 -> 표본이 적을 때 AIC가 BIC보다 더 많은 변화점을 허용하는 경향이 있어
    기획구현.md가 "표본 적은 우리 데이터에는 BIC가 더 안전"이라고 명시한 이유)
"""

from __future__ import annotations


def penalty_value(n_samples: int, n_dims: int = 1, sigma2: float = 1.0) -> float:
    """
    AIC 페널티 값 계산.

    Parameters
    ----------
    n_samples : int
        신호의 길이(관측치 개수). bic.penalty_value와 동일한 시그니처를
        맞추기 위해 받지만 AIC 계산 자체에는 사용하지 않는다(상수 페널티).
    n_dims : int
        신호의 차원 수. 기본값 1.
    sigma2 : float
        신호의 분산 추정치. 기본값 1.0.

    Returns
    -------
    float
        ruptures.Pelt(...).predict(pen=penalty_value(...)) 에 넘길 페널티 값.
    """
    if n_samples <= 1:
        raise ValueError("n_samples는 2 이상이어야 합니다.")
    return float(2 * n_dims * sigma2)


def name() -> str:
    """pelt.py / main.py 등에서 어떤 기준이 쓰였는지 로깅할 때 사용."""
    return "AIC"
