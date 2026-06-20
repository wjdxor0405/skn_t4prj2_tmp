"""
분석 A ③ 단계: 표본충분성 확인 -- AUC 측정값의 부트스트랩 신뢰구간.

기획구현가이드(A,B확정판).md 3번 섹션:
  "②에서 측정한 세그먼트단독 AUC를 부트스트랩(수백 회)으로 재추정해
   신뢰구간 계산"
  "신뢰구간이 좁으면(예: ±0.02) 그 표본 크기에서 측정이 안정적.
   넓으면(예: ±0.15 이상) 표본부족 신호 -- 인접 구간과 통합 후 ②부터
   재검증"
  "사전 외부 공식(검정력분석 등) 없이, 측정 자체의 안정성을 사후에
   직접 확인"

4번 섹션 [정정]: Cohen 효과크기 기반 검정력 분석(NormalIndPower 등)은
완전히 폐기됐다. 이 모듈은 그런 사전 공식을 전혀 쓰지 않고, "AUC
측정 자체"를 부트스트랩으로 재추정해서 그 변동폭만으로 표본충분성을
판단한다.

⚠️ ②(step2_permutation.py)의 순열검정과는 완전히 다른 목적의 별도
   함수다 (가이드 114~118번 줄 명시):
   - ②: 라벨을 "섞어서" -> 경계가 이탈여부와 실재 관련 있는지 검증
   - ③(여기): 데이터를 "복원추출"해서 -> AUC 측정 자체가 표본크기
     면에서 안정적인지 확인 (라벨을 섞지 않음, 원래 라벨 그대로 재추출)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from step2_permutation import boundaries_to_segment_labels  # noqa: E402


@dataclass
class BootstrapCIResult:
    """③ 단계 부트스트랩 신뢰구간 결과."""

    boundaries_tenure: list
    observed_auc: float
    bootstrap_auc_distribution: np.ndarray  # 복원추출 재측정 AUC 분포
    ci_lower: float  # 95% 신뢰구간 하한
    ci_upper: float  # 95% 신뢰구간 상한
    ci_width: float  # ci_upper - ci_lower (좁을수록 안정적)
    n_bootstrap: int
    is_stable: bool  # ci_width가 narrow_threshold 이하인지


def bootstrap_auc_ci(
    df: pd.DataFrame,
    boundaries_tenure: list[int],
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
    n_bootstrap: int = 200,
    n_splits: int = 5,
    n_estimators: int = 100,
    narrow_threshold: float = 0.04,  # ±0.02 -> 폭 0.04를 "안정적" 기준으로 사용
    random_state: int = 42,
) -> BootstrapCIResult:
    """
    주어진 경계의 세그먼트단독 AUC를, 행 단위 복원추출(resample with
    replacement)로 n_bootstrap회 재측정해 95% 신뢰구간을 계산한다.

    가이드: "신뢰구간이 좁으면(예: ±0.02) ... 넓으면(예: ±0.15 이상)
    표본부족 신호". ±0.02는 신뢰구간 폭으로 환산하면 0.04(상하한 합)에
    해당하므로, narrow_threshold 기본값을 0.04로 둔다. 이 임계값은
    참고용 기본값일 뿐 자동 의사결정에 강제로 쓰지 않는다 (가이드 4번:
    "절대임계값은 자동 탐지하지 않음 -- K별 신뢰구간 표를 투명하게
    제시하고 사람이 종합 판단"). is_stable 필드는 그 참고 판단을 보여주는
    용도로만 제공한다.
    """
    y_full = df[churn_col].map({"Yes": 1, "No": 0})
    if y_full.isna().any():
        y_full = pd.to_numeric(df[churn_col], errors="coerce")
    y_full = y_full.reset_index(drop=True)

    X_full = boundaries_to_segment_labels(df[tenure_col], boundaries_tenure).reset_index(drop=True)

    def measure_auc(X: pd.DataFrame, y: pd.Series) -> float:
        if X.shape[1] <= 1 or y.nunique() < 2:
            return 0.5
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=1)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        try:
            scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        except ValueError:
            # 복원추출 후 특정 fold에 한 클래스만 남는 등 극단적 표본부족 케이스
            return 0.5
        return float(scores.mean())

    observed_auc = measure_auc(X_full, y_full)

    n = len(df)
    rng = np.random.default_rng(random_state)
    boot_aucs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)  # 복원추출 (라벨은 섞지 않음 -- ②와의 차이)
        X_boot = X_full.iloc[idx].reset_index(drop=True)
        y_boot = y_full.iloc[idx].reset_index(drop=True)
        boot_aucs[i] = measure_auc(X_boot, y_boot)

    ci_lower = float(np.percentile(boot_aucs, 2.5))
    ci_upper = float(np.percentile(boot_aucs, 97.5))
    ci_width = ci_upper - ci_lower

    return BootstrapCIResult(
        boundaries_tenure=sorted(boundaries_tenure),
        observed_auc=observed_auc,
        bootstrap_auc_distribution=boot_aucs,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_width=ci_width,
        n_bootstrap=n_bootstrap,
        is_stable=(ci_width <= narrow_threshold),
    )


def bootstrap_ci_summary(result: BootstrapCIResult) -> dict:
    """부트스트랩 신뢰구간 결과를 사람이 읽기 좋은 요약 딕셔너리로 변환."""
    return {
        "boundaries_tenure": result.boundaries_tenure,
        "observed_auc": round(result.observed_auc, 4),
        "ci_95": (round(result.ci_lower, 4), round(result.ci_upper, 4)),
        "ci_width": round(result.ci_width, 4),
        "n_bootstrap": result.n_bootstrap,
        "is_stable_reference(width<=0.04)": result.is_stable,
    }
