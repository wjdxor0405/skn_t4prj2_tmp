"""
compare_methods.py
==================
결정트리(DT) vs PELT vs BIC/AIC 기반 변화점 탐지 방법 비교.

실행:
    python compare_methods.py

출력:
    - 콘솔: 메서드별 최적 결과 요약 + 비교표
    - CSV : comparison_report.csv
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

# 기존 모듈
from tenure_binning import (
    BinningResult,
    EvaluatorRegistry,
    TenureBinner,
    PARAM_GRID as DT_PARAM_GRID,
)

# 변화점 탐지 모듈
from changepoint_methods import (
    PELTBinner,
    InfoCriterionBinner,
    IC_PARAM_GRID,
    build_churn_signal,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 공통 평가 유틸 (메서드 독립적)
# ─────────────────────────────────────────────────────────────

def thresholds_to_bin_index(
    X: np.ndarray,
    thresholds: list[float],
) -> np.ndarray:
    """임계값 리스트 → 각 샘플의 구간 인덱스."""
    edges = [-np.inf] + sorted(thresholds) + [np.inf]
    idx = np.zeros(len(X), dtype=int)
    for i in range(len(edges) - 1):
        mask = (X > edges[i]) & (X <= edges[i + 1])
        idx[mask] = i
    return idx


def find_optimal_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """
    precision-recall curve 상의 모든 후보 threshold를 훑어
    F1을 최대화하는 지점을 찾는다.

    Returns
    -------
    best_threshold : F1을 최대화하는 확률 임계값 (0~1)
    best_f1        : 그 지점에서의 F1 score
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # precision_recall_curve는 thresholds가 precision/recall보다 1개 적음
    precision = precision[:-1]
    recall    = recall[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

    if len(f1_scores) == 0:
        return 0.5, 0.0

    best_idx = int(np.argmax(f1_scores))
    return round(float(thresholds[best_idx]), 4), round(float(f1_scores[best_idx]), 4)


def eval_predictive(
    thresholds: list[float],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
) -> dict[str, float]:
    """
    F1 / AUC-ROC 평가.

    F1은 두 가지를 함께 반환:
    - f1            : 기본 임계값 0.5에서의 F1 (기존 출력과 동일)
    - f1_optimal    : precision-recall curve에서 F1을 최대화하는 임계값에서의 F1
    - f1_opt_thresh : 그 최적 임계값 (확률 기준, 0~1)
    """
    if not thresholds:
        return {
            "f1": 0.0, "auc": 0.5, "predictive_score": 0.25,
            "f1_optimal": 0.0, "f1_opt_thresh": 0.5,
        }

    X_bin = thresholds_to_bin_index(X, thresholds).reshape(-1, 1)
    n_bins = len(np.unique(X_bin))

    if n_bins < 2:
        return {
            "f1": 0.0, "auc": 0.5, "predictive_score": 0.25,
            "f1_optimal": 0.0, "f1_opt_thresh": 0.5,
        }

    clf = DecisionTreeClassifier(max_depth=n_bins, random_state=42)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    y_pred = cross_val_predict(clf, X_bin, y, cv=skf)
    y_prob = cross_val_predict(clf, X_bin, y, cv=skf, method="predict_proba")[:, 1]

    f1  = round(float(f1_score(y, y_pred, zero_division=0)), 4)
    auc = round(float(roc_auc_score(y, y_prob)), 4)

    f1_opt_thresh, f1_optimal = find_optimal_f1_threshold(y, y_prob)

    return {
        "f1":               f1,             # threshold=0.5 기준 (기존 출력)
        "f1_optimal":       f1_optimal,      # 최적 threshold 기준 F1
        "f1_opt_thresh":    f1_opt_thresh,   # 최적 threshold 값
        "auc":              auc,
        "predictive_score": round((f1 + auc) / 2, 4),
    }


def eval_significance(
    thresholds: list[float],
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Chi-square p-value 평가."""
    if not thresholds:
        return {"mean_pval": 1.0, "min_pval": 1.0, "n_splits": 0}

    X_bin = thresholds_to_bin_index(X, thresholds)
    n_bins = len(np.unique(X_bin))
    pvals: list[float] = []

    for i in range(n_bins - 1):
        left  = X_bin <= i
        right = X_bin  > i
        if left.sum() < 5 or right.sum() < 5:
            continue
        ct = np.array([
            [(y[left] == 0).sum(),  (y[left] == 1).sum()],
            [(y[right] == 0).sum(), (y[right] == 1).sum()],
        ])
        _, p, _, _ = stats.chi2_contingency(ct)
        pvals.append(p)

    if not pvals:
        return {"mean_pval": 1.0, "min_pval": 1.0, "n_splits": 0}

    return {
        "mean_pval": round(float(np.mean(pvals)), 8),
        "min_pval":  round(float(min(pvals)), 8),
        "n_splits":  len(pvals),
    }


# ─────────────────────────────────────────────────────────────
# 비교 결과 데이터 클래스
# ─────────────────────────────────────────────────────────────

@dataclass
class MethodResult:
    method_name:   str
    variant:       str                        # e.g. "predictive", "significance", "bic", "aic"
    thresholds:    list[float]
    best_params:   dict[str, Any]
    predictive:    dict[str, float] = field(default_factory=dict)
    significance:  dict[str, float] = field(default_factory=dict)
    extra:         dict[str, Any]   = field(default_factory=dict)

    @property
    def n_bins(self) -> int:
        return len(self.thresholds) + 1


# ─────────────────────────────────────────────────────────────
# 비교 실행기
# ─────────────────────────────────────────────────────────────

class MethodComparator:
    """
    등록된 메서드를 실행하고 결과를 수집·비교합니다.

    새 메서드 추가:
        comparator.register_method("my_method", run_fn, tune_kwargs)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y
        self._results: list[MethodResult] = []

    # ── 결정트리 기반 방법 ────────────────────────────────────
    def run_decision_tree(self) -> None:
        binner = TenureBinner()

        for ev_name in ["predictive", "significance"]:
            print(f"  [DT/{ev_name}] 튜닝 중...")
            tuning = binner.tune(self.X, self.y, DT_PARAM_GRID, ev_name)
            best   = tuning.best_result

            pred_scores = eval_predictive(best.thresholds, self.X, self.y)
            sig_scores  = eval_significance(best.thresholds, self.X, self.y)

            self._results.append(MethodResult(
                method_name  = "DecisionTree",
                variant      = ev_name,
                thresholds   = best.thresholds,
                best_params  = best.params,
                predictive   = pred_scores,
                significance = sig_scores,
                extra        = {"dt_score": tuning.best_score},
            ))

    # ── PELT 기반 방법 ────────────────────────────────────────
    def run_pelt(self) -> None:
        pelt = PELTBinner()

        for ev_name, higher in [("predictive", True), ("significance", False)]:
            print(f"  [PELT/{ev_name}] 튜닝 중...")

            if ev_name == "predictive":
                ev_fn: Callable = lambda t, X, y: eval_predictive(t, X, y)["predictive_score"]
            else:
                # significance: mean_pval 최소화
                ev_fn = lambda t, X, y: eval_significance(t, X, y)["mean_pval"]

            best_params, best_thr, best_score, records = pelt.tune(
                self.X, self.y, ev_fn, higher_is_better=higher
            )

            pred_scores = eval_predictive(best_thr, self.X, self.y)
            sig_scores  = eval_significance(best_thr, self.X, self.y)

            self._results.append(MethodResult(
                method_name  = "PELT",
                variant      = ev_name,
                thresholds   = best_thr,
                best_params  = best_params or {},
                predictive   = pred_scores,
                significance = sig_scores,
                extra        = {"pelt_score": best_score, "n_candidates": len(records)},
            ))

    # ── BIC / AIC 기반 방법 ───────────────────────────────────
    def run_info_criterion(self) -> None:
        for criterion in ["bic", "aic"]:
            for ev_name, higher in [("predictive", True), ("significance", False)]:
                print(f"  [{criterion.upper()}/{ev_name}] 튜닝 중...")

                ic_binner = InfoCriterionBinner(criterion=criterion)

                if ev_name == "predictive":
                    ev_fn: Callable = lambda t, X, y: eval_predictive(t, X, y)["predictive_score"]
                else:
                    ev_fn = lambda t, X, y: eval_significance(t, X, y)["mean_pval"]

                best_params, best_thr, best_score, records = ic_binner.tune(
                    self.X, self.y, IC_PARAM_GRID, ev_fn, higher_is_better=higher
                )

                pred_scores = eval_predictive(best_thr, self.X, self.y)
                sig_scores  = eval_significance(best_thr, self.X, self.y)

                self._results.append(MethodResult(
                    method_name  = criterion.upper(),
                    variant      = ev_name,
                    thresholds   = best_thr,
                    best_params  = best_params or {},
                    predictive   = pred_scores,
                    significance = sig_scores,
                    extra        = {"ic_score": best_score, "criterion": criterion},
                ))

    # ── 전체 실행 ─────────────────────────────────────────────
    def run_all(self) -> list[MethodResult]:
        print("\n[1/4] 결정트리(DecisionTree) 실행...")
        self.run_decision_tree()

        print("\n[2/4] PELT 변화점 탐지 실행...")
        self.run_pelt()

        print("\n[3/4] BIC 기반 분기점 선택 실행...")
        self.run_info_criterion()   # BIC + AIC 둘 다 포함

        return self._results

    # ── 결과 반환 ─────────────────────────────────────────────
    def get_results(self) -> list[MethodResult]:
        return self._results


# ─────────────────────────────────────────────────────────────
# 비교 리포터
# ─────────────────────────────────────────────────────────────

class ComparisonReporter:

    SEP = "─" * 80

    @staticmethod
    def _fmt_thr(thresholds: list[float]) -> str:
        if not thresholds:
            return "(없음)"
        return ", ".join(f"{t:.1f}" for t in sorted(thresholds))

    @staticmethod
    def _fmt_bins(thresholds: list[float]) -> str:
        edges = [-np.inf] + sorted(thresholds) + [np.inf]
        parts = []
        for lo, hi in zip(edges, edges[1:]):
            lo_s = "-∞"   if lo == -np.inf else f"{lo:.1f}"
            hi_s = "+∞"   if hi ==  np.inf else f"{hi:.1f}"
            parts.append(f"({lo_s}, {hi_s}]")
        return "  |  ".join(parts)

    def print_detail(self, result: MethodResult) -> None:
        print(f"\n{'='*80}")
        print(f"  방법: {result.method_name}  /  최적화 관점: {result.variant}")
        print(f"{'='*80}")
        print(f"  최적 파라미터  : {result.best_params}")
        print(f"  분기 임계값    : {self._fmt_thr(result.thresholds)}")
        print(f"  구간 수        : {result.n_bins}")
        print(f"\n  ── 구간 목록 ──")
        for i, seg in enumerate(self._fmt_bins(result.thresholds).split("  |  ")):
            print(f"    구간 {i}: {seg}")
        print(f"\n  ── 예측 성능 ──")
        print(f"    F1-Score (threshold=0.5)   : {result.predictive.get('f1', '-')}")
        print(f"    F1-Score (최적 threshold)  : {result.predictive.get('f1_optimal', '-')}"
              f"   (threshold={result.predictive.get('f1_opt_thresh', '-')})")
        print(f"    AUC-ROC                    : {result.predictive.get('auc', '-')}")
        print(f"    종합                       : {result.predictive.get('predictive_score', '-')}")
        print(f"\n  ── 분기점 유의성 ──")
        print(f"    Mean p-value : {result.significance.get('mean_pval', '-')}")
        print(f"    Min  p-value : {result.significance.get('min_pval', '-')}")
        print(f"    검정 횟수    : {result.significance.get('n_splits', '-')}")
        if result.extra:
            print(f"\n  ── 기타 ──")
            for k, v in result.extra.items():
                print(f"    {k}: {v}")

    def print_comparison_table(self, results: list[MethodResult]) -> None:
        """메서드 × 관점 비교표를 콘솔에 출력."""
        print(f"\n\n{'#'*80}")
        print("##  전체 비교 요약표")
        print(f"{'#'*80}")

        header = (
            f"{'방법':<15} {'관점':<14} {'구간수':>5} "
            f"{'F1(0.5)':>8} {'F1(최적)':>9} {'최적Th':>7} "
            f"{'AUC':>7} {'종합점수':>9} "
            f"{'Mean p-val':>12} {'임계값'}"
        )
        print(f"\n{header}")
        print(self.SEP)

        for r in results:
            row = (
                f"{r.method_name:<15} {r.variant:<14} {r.n_bins:>5} "
                f"{r.predictive.get('f1', 0):>8.4f} "
                f"{r.predictive.get('f1_optimal', 0):>9.4f} "
                f"{r.predictive.get('f1_opt_thresh', 0):>7.3f} "
                f"{r.predictive.get('auc', 0):>7.4f} "
                f"{r.predictive.get('predictive_score', 0):>9.4f} "
                f"{r.significance.get('mean_pval', 1):>12.2e} "
                f"{self._fmt_thr(r.thresholds)}"
            )
            print(row)

        print(self.SEP)

        # 관점별 최고 방법 강조
        pred_best  = max(results, key=lambda r: r.predictive.get("predictive_score", 0))
        sig_best   = min(results, key=lambda r: r.significance.get("mean_pval", 1.0))

        print(f"\n  🏆 예측 성능 최고  : {pred_best.method_name} / {pred_best.variant}"
              f"  (종합={pred_best.predictive.get('predictive_score',0):.4f})")
        print(f"  🏆 유의성 최고     : {sig_best.method_name} / {sig_best.variant}"
              f"  (Mean p={sig_best.significance.get('mean_pval',1):.2e})")

    def to_dataframe(self, results: list[MethodResult]) -> pd.DataFrame:
        rows = []
        for r in results:
            rows.append({
                "method":            r.method_name,
                "variant":           r.variant,
                "n_bins":            r.n_bins,
                "thresholds":        self._fmt_thr(r.thresholds),
                "f1":                r.predictive.get("f1"),
                "f1_optimal":        r.predictive.get("f1_optimal"),
                "f1_opt_thresh":     r.predictive.get("f1_opt_thresh"),
                "auc":               r.predictive.get("auc"),
                "predictive_score":  r.predictive.get("predictive_score"),
                "mean_pval":         r.significance.get("mean_pval"),
                "min_pval":          r.significance.get("min_pval"),
                "n_splits":          r.significance.get("n_splits"),
                "best_params":       str(r.best_params),
                **{f"extra_{k}": v for k, v in r.extra.items()},
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────

def main(csv_path: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> None:
    from tenure_binning import TenureBinner

    print("=" * 80)
    print("  Tenure 구간 분할 방법 비교")
    print("  DecisionTree  vs  PELT  vs  BIC  vs  AIC")
    print("=" * 80)

    binner = TenureBinner()
    X, y = binner.load_and_prepare(csv_path)
    print(f"\n  데이터: {csv_path}")
    print(f"  샘플 수={len(X)}, Churn 비율={y.mean():.3f}, "
          f"tenure 범위={int(X.min())}~{int(X.max())}")

    # ── 비교 실행 ─────────────────────────────
    comparator = MethodComparator(X, y)
    results = comparator.run_all()

    # ── 상세 출력 ──────────────────────────────
    reporter = ComparisonReporter()
    for r in results:
        reporter.print_detail(r)

    # ── 비교 요약표 ───────────────────────────
    reporter.print_comparison_table(results)

    # ── CSV 저장 ──────────────────────────────
    df = reporter.to_dataframe(results)
    out_path = "comparison_report.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  결과 저장 → {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
