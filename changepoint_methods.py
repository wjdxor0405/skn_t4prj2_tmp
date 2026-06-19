"""
changepoint_methods.py
======================
PELT / BIC / AIC 기반 변화점 탐지 메서드 모듈
기존 tenure_binning.py 의 BaseBinningMethod 를 상속하여 구현.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import ruptures as rpt

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# tenure → churn-rate 시그널 변환 유틸
# ─────────────────────────────────────────────────────────────

def build_churn_signal(
    X: np.ndarray,
    y: np.ndarray,
    window: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    tenure 값별 churn rate 를 구해 1-D 시그널로 반환.

    Parameters
    ----------
    X       : tenure 원본 배열
    y       : churn 레이블 (0/1)
    window  : 이동평균 스무딩 윈도우 크기

    Returns
    -------
    unique_tenures : 정렬된 고유 tenure 값
    signal_1d     : 각 tenure 에서의 (스무딩된) churn rate
    """

    # 호출 전에 NaN·이상값이 이미 제거된 X를 받는 것이 원칙이지만,
    # 방어적으로 한 줄 추가
    tenures = np.sort(np.unique(X[~np.isnan(X)])).astype(int)
    
    tenures = np.sort(np.unique(X)).astype(int)
    rates = np.array([
        y[X == t].mean() if (X == t).sum() > 0 else 0.0
        for t in tenures
    ])

    # 이동평균 스무딩 (엣지 처리: same padding)
    if window > 1:
        kernel = np.ones(window) / window
        rates = np.convolve(rates, kernel, mode="same")

    return tenures, rates


def breakpoints_to_thresholds(
    breakpoints: list[int],
    tenures: np.ndarray,
) -> list[float]:
    """
    ruptures 의 분기점(인덱스) → 실제 tenure 임계값 변환.
    ruptures 는 세그먼트 끝 인덱스(exclusive)를 반환하므로
    인덱스-1 위치의 tenure 를 임계값으로 사용.
    """
    thresholds = []
    for bp in breakpoints:
        if bp < len(tenures):
            thresholds.append(float(tenures[bp - 1]))
    return sorted(set(thresholds))


# ─────────────────────────────────────────────────────────────
# BIC / AIC 계산
# ─────────────────────────────────────────────────────────────

def _segment_log_likelihood(signal: np.ndarray) -> float:
    """정규분포 가정 하의 세그먼트 log-likelihood."""
    n = len(signal)
    if n < 2:
        return 0.0
    mu  = signal.mean()
    var = signal.var()
    if var <= 0:
        var = 1e-10
    return -0.5 * n * (np.log(2 * np.pi * var) + 1)


def compute_information_criterion(
    signal: np.ndarray,
    breakpoints: list[int],
    criterion: str = "bic",
) -> float:
    """
    BIC / AIC 계산.

    Parameters
    ----------
    signal      : 1-D churn-rate 시그널
    breakpoints : ruptures 출력 (세그먼트 끝 인덱스, 마지막은 len(signal))
    criterion   : 'bic' | 'aic'

    Returns
    -------
    정보기준 값 (낮을수록 좋음)
    """
    n = len(signal)
    segments = list(zip([0] + list(breakpoints[:-1]), breakpoints))

    total_ll = sum(_segment_log_likelihood(signal[s:e]) for s, e in segments)

    # 파라미터 수: 각 세그먼트의 평균·분산 2개 + 분기점 수
    k = 2 * len(segments) + (len(breakpoints) - 1)

    if criterion == "bic":
        return -2 * total_ll + k * np.log(n)
    elif criterion == "aic":
        return -2 * total_ll + 2 * k
    else:
        raise ValueError(f"criterion must be 'bic' or 'aic', got '{criterion}'")


# ─────────────────────────────────────────────────────────────
# PELT 기반 분할기
# ─────────────────────────────────────────────────────────────

class PELTBinner:
    """
    PELT(Pruned Exact Linear Time) 알고리즘으로 최적 변화점 탐지.
    penalty 파라미터를 그리드 탐색하여 최적 분기점 집합 반환.
    """
    PARAM_GRID: list[dict[str, Any]] = [
        {"model": m, "penalty": p, "min_size": ms}
        for m  in ["rbf", "l2", "normal"]
        for p  in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0]
        for ms in [3, 5, 7]
    ]

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> tuple[list[float], list[int]]:
        """
        Returns
        -------
        thresholds : 실제 tenure 임계값 리스트
        raw_bkps   : ruptures 인덱스 기반 breakpoints
        """
        tenures, signal = build_churn_signal(X, y)
        signal_2d = signal.reshape(-1, 1)

        algo = rpt.Pelt(
            model    = params["model"],
            min_size = params["min_size"],
            jump     = 1,
        ).fit(signal_2d)

        raw_bkps = algo.predict(pen=params["penalty"])
        thresholds = breakpoints_to_thresholds(raw_bkps[:-1], tenures)  # 마지막 제외
        return thresholds, raw_bkps

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        evaluator_fn,                    # (thresholds, X, y) → score
        higher_is_better: bool = True,
    ) -> tuple[dict[str, Any], list[float], float]:
        """전체 파라미터 그리드 탐색 → 최적 파라미터·임계값·점수 반환."""
        best_params     = None
        best_thresholds: list[float] = []
        best_score      = -np.inf if higher_is_better else np.inf
        all_records     = []

        for params in self.PARAM_GRID:
            try:
                thresholds, _ = self.run(X, y, params)
                score = evaluator_fn(thresholds, X, y)
            except Exception:
                continue

            all_records.append((params, thresholds, score))

            improved = (
                score > best_score if higher_is_better else score < best_score
            )
            if improved:
                best_score      = score
                best_params     = params
                best_thresholds = thresholds

        return best_params, best_thresholds, best_score, all_records


# ─────────────────────────────────────────────────────────────
# BIC / AIC 기반 최적 분기점 수 선택 (Binseg)
# ─────────────────────────────────────────────────────────────

class InfoCriterionBinner:
    """
    Binary Segmentation 으로 후보 분기점을 생성하고
    BIC / AIC 로 최적 개수를 선택.
    """

    def __init__(
        self,
        criterion: str = "bic",
        model: str = "normal",
        max_bkps: int = 30,#10,
        min_size: int = 3,
    ):
        self.criterion = criterion
        self.model     = model
        self.max_bkps  = max_bkps
        self.min_size  = min_size

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[list[float], int, list[float]]:
        """
        Returns
        -------
        best_thresholds : 최적 tenure 임계값 리스트
        best_n_bkps     : 선택된 분기점 수
        ic_scores       : 각 n_bkps 별 정보기준 값 리스트
        """
        tenures, signal = build_churn_signal(X, y)

        algo = rpt.Binseg(
            model    = self.model,
            min_size = self.min_size,
            jump     = 1,
        ).fit(signal.reshape(-1, 1))

        ic_scores: list[float] = []
        all_bkps:  list[list[int]] = []

        for n in range(0, self.max_bkps + 1):
            if n == 0:
                bkps = [len(signal)]
            else:
                try:
                    bkps = algo.predict(n_bkps=n)
                except Exception:
                    break

            ic = compute_information_criterion(signal, bkps, self.criterion)
            ic_scores.append(ic)
            all_bkps.append(bkps)

        best_n = int(np.argmin(ic_scores))
        best_bkps = all_bkps[best_n]
        best_thresholds = breakpoints_to_thresholds(best_bkps[:-1], tenures)

        return best_thresholds, best_n, ic_scores

    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: list[dict[str, Any]],
        evaluator_fn,
        higher_is_better: bool = True,
    ) -> tuple[dict[str, Any], list[float], float]:
        """model / max_bkps 파라미터 그리드 탐색."""
        best_params     = None
        best_thresholds: list[float] = []
        best_score      = -np.inf if higher_is_better else np.inf
        all_records     = []

        for params in param_grid:
            binner = InfoCriterionBinner(
                criterion = self.criterion,
                model     = params.get("model", self.model),
                max_bkps  = params.get("max_bkps", self.max_bkps),
                min_size  = params.get("min_size", self.min_size),
            )
            try:
                thresholds, n_bkps, ic_scores = binner.run(X, y)
                score = evaluator_fn(thresholds, X, y)
            except Exception:
                continue

            all_records.append((params, thresholds, score, n_bkps, min(ic_scores)))

            improved = (
                score > best_score if higher_is_better else score < best_score
            )
            if improved:
                best_score      = score
                best_params     = {**params, "selected_n_bkps": n_bkps}
                best_thresholds = thresholds

        return best_params, best_thresholds, best_score, all_records


IC_PARAM_GRID: list[dict[str, Any]] = [
    {"model": m, "max_bkps": mb, "min_size": ms}
    for m  in ["normal", "l2", "rbf"]
    for mb in [5, 8, 12]
    for ms in [3, 5, 7]
]
