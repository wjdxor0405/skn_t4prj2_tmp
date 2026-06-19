"""
tenure_binning.py
=================
결정트리를 이용한 tenure 구간 분할 + 확장 가능한 평가 프레임워크

평가 관점
---------
1. PredictiveEvaluator  : F1-Score / AUC-ROC (예측 성능)
2. SignificanceEvaluator: p-value (분기점 유의성)
(추가 평가 시 BaseEvaluator를 상속하여 register_evaluator()로 등록)
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class BinningResult:
    """결정트리 하나의 실행 결과"""
    thresholds: list[float]          # 분기 임계값 (오름차순)
    bins: list[tuple[float, float]]  # (lower, upper) 구간 목록
    params: dict[str, Any]           # 사용된 DecisionTree 파라미터
    scores: dict[str, float] = field(default_factory=dict)  # 평가 점수


@dataclass
class TuningResult:
    """파라미터 튜닝 전체 결과"""
    evaluator_name: str
    best_result: BinningResult
    best_score: float
    all_results: list[BinningResult]
    higher_is_better: bool           # True면 점수 높을수록 좋음


# ──────────────────────────────────────────────
# 평가자(Evaluator) 추상 기반 클래스
# ──────────────────────────────────────────────

class BaseEvaluator(ABC):
    """
    새 평가 관점 추가 시 이 클래스를 상속하여
    EvaluatorRegistry.register()로 등록하면 됩니다.
    """

    name: str                  # 서브클래스에서 정의
    higher_is_better: bool     # True = 높을수록 좋음

    @abstractmethod
    def evaluate(
        self,
        X_binned: np.ndarray,   # tenure를 bin 인덱스로 변환한 1-D 배열
        y: np.ndarray,          # 타깃 (0/1)
        result: BinningResult,  # 메타 정보
    ) -> float:
        """점수(스칼라) 반환"""


# ──────────────────────────────────────────────
# 평가자 구현
# ──────────────────────────────────────────────

class PredictiveEvaluator(BaseEvaluator):
    """F1-Score + AUC-ROC 기반 예측 성능 평가 (두 점수의 평균)"""

    name = "predictive"
    higher_is_better = True

    def __init__(self, cv: int = 5):
        self.cv = cv

    def evaluate(self, X_binned: np.ndarray, y: np.ndarray,
                 result: BinningResult) -> float:
        if len(np.unique(X_binned)) < 2:
            return 0.0

        clf = DecisionTreeClassifier(max_depth=len(result.bins), random_state=42)
        X2d = X_binned.reshape(-1, 1)
        cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

        y_pred = cross_val_predict(clf, X2d, y, cv=cv)
        y_prob = cross_val_predict(clf, X2d, y, cv=cv, method="predict_proba")[:, 1]

        f1  = f1_score(y, y_pred, zero_division=0)
        auc = roc_auc_score(y, y_prob)

        result.scores["f1"]  = round(f1,  4)
        result.scores["auc"] = round(auc, 4)
        return round((f1 + auc) / 2, 4)


class SignificanceEvaluator(BaseEvaluator):
    """
    각 분기점에서 Chi-square 검정 → p-value 집계.
    "유의성"은 낮은 p-value가 좋으므로 higher_is_better=False.
    여러 분기점이 있을 때 p-value의 평균을 사용.
    """

    name = "significance"
    higher_is_better = False

    def evaluate(self, X_binned: np.ndarray, y: np.ndarray,
                 result: BinningResult) -> float:
        pvals: list[float] = []

        for i in range(len(result.bins) - 1):
            mask_left  = X_binned <= i
            mask_right = X_binned  > i
            if mask_left.sum() < 5 or mask_right.sum() < 5:
                continue

            contingency = np.array([
                [(y[mask_left] == 0).sum(), (y[mask_left] == 1).sum()],
                [(y[mask_right] == 0).sum(), (y[mask_right] == 1).sum()],
            ])
            _, p, _, _ = stats.chi2_contingency(contingency)
            pvals.append(p)

        if not pvals:
            return 1.0

        mean_pval = float(np.mean(pvals))
        result.scores["mean_pval"]  = round(mean_pval, 6)
        result.scores["min_pval"]   = round(min(pvals), 6)
        result.scores["n_splits"]   = len(pvals)
        return round(mean_pval, 6)


# ──────────────────────────────────────────────
# 평가자 레지스트리
# ──────────────────────────────────────────────

class EvaluatorRegistry:
    """평가자를 등록·조회하는 싱글턴 레지스트리"""

    _registry: dict[str, BaseEvaluator] = {}

    @classmethod
    def register(cls, evaluator: BaseEvaluator) -> None:
        cls._registry[evaluator.name] = evaluator

    @classmethod
    def get(cls, name: str) -> BaseEvaluator:
        if name not in cls._registry:
            raise KeyError(f"'{name}' 평가자가 등록되지 않았습니다. "
                           f"등록된 평가자: {list(cls._registry)}")
        return cls._registry[name]

    @classmethod
    def list_names(cls) -> list[str]:
        return list(cls._registry)


# 기본 평가자 등록
EvaluatorRegistry.register(PredictiveEvaluator(cv=5))
EvaluatorRegistry.register(SignificanceEvaluator())


# ──────────────────────────────────────────────
# 결정트리 기반 구간 분할기
# ──────────────────────────────────────────────

class TenureBinner:
    """
    결정트리를 이용해 tenure 컬럼의 분기점을 찾고
    등록된 평가자로 품질을 측정합니다.
    """

    def __init__(self, feature_col: str = "tenure", target_col: str = "Churn"):
        self.feature_col = feature_col
        self.target_col  = target_col
        self._le = LabelEncoder()

    # ── 데이터 전처리 ──────────────────────────
    def load_and_prepare(
            self,
            path: str,
            valid_range: tuple[float, float] = (0.0, 72.0),  # 데이터셋 명세 기준
            na_strategy: str = "drop",  # "drop" | "median" | "mean"
    ) -> tuple[np.ndarray, np.ndarray]:

        df = pd.read_csv(path)

        # 1. 필수 컬럼 존재 확인
        for col in [self.feature_col, self.target_col]:
            if col not in df.columns:
                raise KeyError(f"필수 컬럼 '{col}'이 데이터에 없습니다.")

        # 2. tenure 타입 강제 변환 (문자열 "  " 등 포함 대비)
        df[self.feature_col] = pd.to_numeric(df[self.feature_col], errors="coerce")

        # 3. 범위 이탈값 → NaN으로 마킹
        lo, hi = valid_range
        out_of_range = ~df[self.feature_col].between(lo, hi, inclusive="both")
        if out_of_range.any():
            print(f"  [경고] 범위 이탈 {out_of_range.sum()}건 → NaN 처리")
            df.loc[out_of_range, self.feature_col] = np.nan

        # 4. NaN 처리
        na_mask = df[self.feature_col].isna() | df[self.target_col].isna()
        if na_mask.any():
            print(f"  [경고] NaN {na_mask.sum()}건 감지")
            if na_strategy == "drop":
                df = df[~na_mask].reset_index(drop=True)
                print(f"  → {na_mask.sum()}행 제거, 잔여 {len(df)}행")
            elif na_strategy in ("median", "mean"):
                fill_val = getattr(df[self.feature_col], na_strategy)()
                df[self.feature_col] = df[self.feature_col].fillna(fill_val)
                print(f"  → tenure NaN을 {na_strategy}({fill_val:.1f})로 대체")
            else:
                raise ValueError(f"na_strategy='{na_strategy}'는 지원하지 않습니다.")

        # 5. Churn 인코딩
        df[self.target_col] = self._le.fit_transform(df[self.target_col])

        X = df[self.feature_col].values.astype(float)
        y = df[self.target_col].values
        return X, y

    # ── 결정트리로 임계값 추출 ─────────────────
    def _fit_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: dict[str, Any],
    ) -> BinningResult:
        clf = DecisionTreeClassifier(**params, random_state=42)
        clf.fit(X.reshape(-1, 1), y)

        tree = clf.tree_
        thresholds = sorted(set(
            round(float(t), 2)
            for t in tree.threshold
            if t != -2.0           # -2.0 은 리프 노드 표시
        ))

        edges = [-np.inf] + thresholds + [np.inf]
        bins  = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

        return BinningResult(thresholds=thresholds, bins=bins, params=params)

    # ── 구간 인덱스 변환 ──────────────────────
    @staticmethod
    def _to_bin_index(X: np.ndarray, result: BinningResult) -> np.ndarray:
        idx = np.zeros(len(X), dtype=int)
        for i, (lo, hi) in enumerate(result.bins):
            mask = (X > lo) & (X <= hi)
            idx[mask] = i
        return idx

    # ── 단일 파라미터 세트 평가 ───────────────
    def evaluate_single(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: dict[str, Any],
        evaluator_name: str,
    ) -> BinningResult:
        result   = self._fit_tree(X, y, params)
        X_binned = self._to_bin_index(X, result)
        evaluator = EvaluatorRegistry.get(evaluator_name)
        score = evaluator.evaluate(X_binned, y, result)
        result.scores["__main__"] = score
        return result

    # ── 파라미터 그리드 튜닝 ──────────────────
    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: list[dict[str, Any]],
        evaluator_name: str,
    ) -> TuningResult:
        evaluator = EvaluatorRegistry.get(evaluator_name)
        all_results: list[BinningResult] = []

        for params in param_grid:
            r = self.evaluate_single(X, y, params, evaluator_name)
            all_results.append(r)

        scores = [r.scores["__main__"] for r in all_results]

        if evaluator.higher_is_better:
            best_idx = int(np.argmax(scores))
        else:
            best_idx = int(np.argmin(scores))

        return TuningResult(
            evaluator_name  = evaluator_name,
            best_result     = all_results[best_idx],
            best_score      = scores[best_idx],
            all_results     = all_results,
            higher_is_better= evaluator.higher_is_better,
        )


# ──────────────────────────────────────────────
# 결과 리포터
# ──────────────────────────────────────────────

class ResultReporter:
    """TuningResult 를 콘솔에 출력"""

    @staticmethod
    def print_tuning_summary(tuning: TuningResult) -> None:
        best = tuning.best_result
        print(f"\n{'='*60}")
        print(f"  평가자  : {tuning.evaluator_name.upper()}")
        print(f"  최적 점수: {tuning.best_score}  "
              f"({'높을수록 좋음' if tuning.higher_is_better else '낮을수록 좋음'})")
        print(f"  최적 파라미터: {best.params}")
        print(f"  분기 임계값  : {best.thresholds}")
        print(f"  구간 수      : {len(best.bins)}")
        print(f"  상세 점수    : {best.scores}")
        print(f"\n  ── 구간 목록 ──")
        for i, (lo, hi) in enumerate(best.bins):
            lo_str = f"{lo:.1f}" if lo != -np.inf else "-∞"
            hi_str = f"{hi:.1f}" if hi !=  np.inf else "+∞"
            print(f"    구간 {i}: ({lo_str}, {hi_str}]")
        print(f"{'='*60}")

    @staticmethod
    def print_all_scores(tuning: TuningResult) -> None:
        print(f"\n  ── 전체 파라미터 결과 ({tuning.evaluator_name}) ──")
        for r in tuning.all_results:
            score = r.scores.get("__main__", "-")
            print(f"    params={r.params}  score={score}  "
                  f"thresholds={r.thresholds}")

    @staticmethod
    def to_dataframe(tuning: TuningResult) -> pd.DataFrame:
        rows = []
        for r in tuning.all_results:
            row = {**r.params, **r.scores, "thresholds": str(r.thresholds)}
            rows.append(row)
        return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# 파라미터 그리드 정의
# ──────────────────────────────────────────────

PARAM_GRID: list[dict[str, Any]] = [
    {"max_depth": d, "min_samples_leaf": m, "min_impurity_decrease": imp}
    for d   in [2, 3, 4, 5]
    for m   in [50, 100, 200]
    for imp in [0.0, 0.001, 0.005]
]


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────

def main(csv_path: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> None:
    binner   = TenureBinner()
    reporter = ResultReporter()

    print(f"데이터 로드: {csv_path}")
    X, y = binner.load_and_prepare(csv_path)
    print(f"  샘플 수={len(X)}, Churn 비율={y.mean():.3f}")
    print(f"  tenure 범위: {X.min():.0f} ~ {X.max():.0f}")
    print(f"  파라미터 조합 수: {len(PARAM_GRID)}")
    print(f"  등록된 평가자: {EvaluatorRegistry.list_names()}")

    all_tuning: dict[str, TuningResult] = {}

    # ── 등록된 모든 평가자로 튜닝 ─────────────
    for ev_name in EvaluatorRegistry.list_names():
        print(f"\n[{ev_name}] 튜닝 시작...")
        tuning = binner.tune(X, y, PARAM_GRID, ev_name)
        all_tuning[ev_name] = tuning
        reporter.print_tuning_summary(tuning)

    # ── 상세 결과표 저장 ──────────────────────
    for ev_name, tuning in all_tuning.items():
        df_out = reporter.to_dataframe(tuning)
        out_path = f"tuning_results_{ev_name}.csv"
        df_out.to_csv(out_path, index=False)
        print(f"\n[{ev_name}] 전체 결과 저장 → {out_path}")
        reporter.print_all_scores(tuning)

    return all_tuning


if __name__ == "__main__":
    main("WA_Fn-UseC_-Telco-Customer-Churn.csv")
