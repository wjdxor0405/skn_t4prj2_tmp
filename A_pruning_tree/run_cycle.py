"""
분석 A ①②③ 반복 사이클.

기획구현가이드(A,B확정판).md 1번 섹션:
  "[분석A 반복 사이클, Train만]
    ① 가지치기 회귀나무로 세그먼트 경계 탐지 (tenure 단일변수)
    ② 세그먼트단독 AUC + 순열검정으로 적절성 검증
    ③ AUC 부트스트랩 신뢰구간으로 표본충분성 점검
    └─ ②에서 우연(p값 높음) 또는 ③에서 신뢰구간 넓음(표본부족)
       -> 문제 구간을 인접 구간과 통합 후 ①부터 재실행
    └─ ②③ 모두 통과 -> 경계 확정, 다음 단계로"

이 모듈은 위 제어 흐름 전체를 구현한다.

⚠️ 설계 메모 ("인접 구간과 통합 후 ①부터 재실행"을 구현한 방식):
  처음에는 실패한 두 구간의 tenure 값을 중앙값으로 강제 재매핑해서
  "데이터 자체를 합치는" 방식을 시도했으나, 직접 검증한 결과 부작용이
  있었다 -- 한 구간(예: 0~25개월, 1754명)이 단일 tenure 값으로 뭉개지면서
  그 지점에 거대한 스파이크가 생기고, 다음 회차 회귀나무가 오히려 그
  주변에서 더 잘게 쪼개려는 역효과가 발생했다(실측: 경계 11 -> 64개로
  급증). 데이터를 왜곡하는 방식은 안전하지 않다고 판단해 폐기했다.

  대신 sklearn DecisionTreeRegressor의 min_samples_leaf를 점진적으로
  늘려가며 ①을 재실행하는 방식을 쓴다. min_samples_leaf는 "리프 1개가
  최소 몇 개월을 포함해야 하는가"를 직접 제어하므로(실측: sample_weight와
  무관하게 원시 표본개수, 즉 개월수 기준으로 작동함을 확인), 데이터를
  전혀 바꾸지 않고도 회귀나무가 같은 좁은 구간을 다시 제안하지 못하게
  막을 수 있다. 첫 회차는 min_samples_leaf=1(제약 없음)로 시작하고,
  실패할 때마다 "직전에 실패를 유발한 가장 좁은 세그먼트의 개월 수 + 1"
  만큼으로 끌어올린다 -- 가이드의 "문제 구간을 인접 구간과 통합"이라는
  취지를(좁은 구간이 다시 나타나지 못하게 막는다는 의미로) 데이터 변형
  없이 구현한 것이다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import step1_tree  # noqa: E402
import step2_permutation  # noqa: E402
import step3_bootstrap_ci  # noqa: E402


@dataclass
class CycleIteration:
    """반복 사이클 1회(①②③ 한 바퀴) 결과."""

    iteration: int
    min_samples_leaf_used: int
    boundaries_tried: list  # ①이 이번 회차에 제안한 경계
    step1_result: "step1_tree.TreeStep1Result"
    step2_result: "step2_permutation.PermutationTestResult | None"
    step3_result: "step3_bootstrap_ci.BootstrapCIResult | None"
    passed: bool  # ②③ 모두 통과했는지
    fail_reason: str | None  # 실패 시 사유 ("permutation" / "bootstrap_ci" / "permutation+bootstrap_ci" / None)
    next_min_samples_leaf: int | None  # 실패 시 다음 회차에 적용할 min_samples_leaf


@dataclass
class CycleResult:
    """①②③ 반복 사이클 전체 결과."""

    iterations: list  # list[CycleIteration]
    final_boundaries: list  # 최종 확정 경계
    converged: bool  # ②③를 모두 통과해서 정상 종료했는지
    stop_reason: str  # "passed" / "single_segment" / "max_iterations"


def _smallest_segment_size(
    df: pd.DataFrame,
    boundaries_tenure: list[int],
    tenure_col: str,
) -> int:
    """현재 경계 구조에서 가장 작은 세그먼트의 "개월 수"(tenure 신호
    단위 표본개수)를 구한다 -- min_samples_leaf를 얼마나 올려야 이
    구간이 다시 나타나지 못하는지 판단하는 기준."""
    edges = [-float("inf")] + sorted(boundaries_tenure) + [float("inf")]
    labels = pd.cut(
        df[tenure_col], bins=edges, labels=list(range(len(edges) - 1)), right=False
    ).astype(int)
    # "개월 수" 기준이므로 고유 tenure 값 개수를 센다(고객 수가 아님 --
    # step1_tree의 입력 신호 자체가 tenure별 1행 집계이기 때문).
    sizes = df.loc[labels.index, tenure_col].groupby(labels).nunique()
    return int(sizes.min())


def run_cycle(
    df_train: pd.DataFrame,
    tenure_col: str = "tenure",
    churn_col: str = "Churn",
    n_permutations: int = 200,
    n_bootstrap: int = 200,
    p_value_threshold: float = 0.05,
    ci_width_threshold: float = 0.04,
    max_iterations: int = 10,
    step1_n_splits: int = 5,
    step1_rf_n_estimators: int = 250,
    step23_n_estimators: int = 100,
    random_state: int = 42,
) -> CycleResult:
    """
    ①②③ 반복 사이클 전체 실행.

    실패 판정 기준 (가이드 그대로):
      - ②: p_value >= p_value_threshold(기본 0.05) -> "우연일 가능성을
        배제 못함" -> 실패
      - ③: ci_width > ci_width_threshold(기본 0.04, ±0.02에 해당) ->
        "표본부족" -> 실패
      둘 중 하나라도 실패하면 min_samples_leaf를 끌어올려 ①부터 재실행한다
      (설계 메모 참고 -- 데이터를 합치는 대신 가드레일을 강화하는 방식).

    Parameters
    ----------
    n_permutations, n_bootstrap : int
        가이드 권장 "수백 회(200회 이상)". 기본 200이지만, 반복
        사이클 전체가 여러 번 도는 점을 감안해 빠른 탐색 시에는
        호출 측에서 50~100 정도로 낮춰 쓰는 것을 권장한다(결과의
        통계적 해석은 동일, 속도만 다름).
    max_iterations : int
        무한루프 방지 가드레일. 가이드에 명시된 값은 아니며, 안전장치다.
    """
    iterations: list[CycleIteration] = []
    stop_reason = "max_iterations"
    final_boundaries: list[int] = []
    min_samples_leaf = 1

    for it in range(1, max_iterations + 1):
        step1_result = step1_tree.run_step1(
            df_train,
            tenure_col=tenure_col,
            churn_col=churn_col,
            n_splits=step1_n_splits,
            min_samples_leaf=min_samples_leaf,
            n_estimators=step1_rf_n_estimators,
            random_state=random_state,
        )
        boundaries = step1_result.boundaries_tenure

        if len(boundaries) == 0:
            # ①이 더 이상 쪼갤 게 없다고 판단(K=1) -> 사이클 종료.
            iterations.append(
                CycleIteration(
                    iteration=it,
                    min_samples_leaf_used=min_samples_leaf,
                    boundaries_tried=boundaries,
                    step1_result=step1_result,
                    step2_result=None,
                    step3_result=None,
                    passed=True,
                    fail_reason=None,
                    next_min_samples_leaf=None,
                )
            )
            final_boundaries = []
            stop_reason = "single_segment"
            break

        step2_result = step2_permutation.run_permutation_test(
            df_train,
            boundaries,
            tenure_col=tenure_col,
            churn_col=churn_col,
            n_permutations=n_permutations,
            n_estimators=step23_n_estimators,
            random_state=random_state,
        )
        step3_result = step3_bootstrap_ci.bootstrap_auc_ci(
            df_train,
            boundaries,
            tenure_col=tenure_col,
            churn_col=churn_col,
            n_bootstrap=n_bootstrap,
            n_estimators=step23_n_estimators,
            narrow_threshold=ci_width_threshold,
            random_state=random_state,
        )

        passes_permutation = step2_result.p_value < p_value_threshold
        passes_bootstrap = step3_result.ci_width <= ci_width_threshold
        passed = passes_permutation and passes_bootstrap

        fail_reason = None
        if not passed:
            if not passes_permutation and not passes_bootstrap:
                fail_reason = "permutation+bootstrap_ci"
            elif not passes_permutation:
                fail_reason = "permutation"
            else:
                fail_reason = "bootstrap_ci"

        next_min_samples_leaf = None
        if not passed:
            smallest = _smallest_segment_size(df_train, boundaries, tenure_col)
            next_min_samples_leaf = smallest + 1

        iterations.append(
            CycleIteration(
                iteration=it,
                min_samples_leaf_used=min_samples_leaf,
                boundaries_tried=boundaries,
                step1_result=step1_result,
                step2_result=step2_result,
                step3_result=step3_result,
                passed=passed,
                fail_reason=fail_reason,
                next_min_samples_leaf=next_min_samples_leaf,
            )
        )

        if passed:
            final_boundaries = boundaries
            stop_reason = "passed"
            break

        if next_min_samples_leaf is None or next_min_samples_leaf <= min_samples_leaf:
            # 더 끌어올릴 수 없거나(이미 충분히 큼) 진전이 없으면 종료.
            final_boundaries = []
            stop_reason = "single_segment"
            break

        min_samples_leaf = next_min_samples_leaf

    return CycleResult(
        iterations=iterations,
        final_boundaries=final_boundaries,
        converged=(stop_reason == "passed"),
        stop_reason=stop_reason,
    )


def cycle_summary_table(result: CycleResult) -> pd.DataFrame:
    """반복 사이클 전체를 한눈에 보는 표 (회차별 경계/통과여부/사유)."""
    rows = []
    for it in result.iterations:
        rows.append(
            {
                "iteration": it.iteration,
                "min_samples_leaf": it.min_samples_leaf_used,
                "boundaries_tried": it.boundaries_tried,
                "passed": it.passed,
                "fail_reason": it.fail_reason,
                "p_value": round(it.step2_result.p_value, 4) if it.step2_result else None,
                "observed_auc": round(it.step2_result.observed_auc, 4) if it.step2_result else None,
                "ci_width": round(it.step3_result.ci_width, 4) if it.step3_result else None,
                "next_min_samples_leaf": it.next_min_samples_leaf,
            }
        )
    return pd.DataFrame(rows)


def cycle_result_to_dict(result: CycleResult) -> dict:
    """
    CycleResult를 results_io.save_result()로 그대로 저장 가능한
    JSON 친화적 딕셔너리로 변환한다. 시각화 스크립트가 다시 읽어
    그래프를 그릴 때 필요한 정보를 전부 포함한다(회차별 alpha 후보
    표, RF 투표, 순열검정 null 분포, 부트스트랩 분포 등).
    """
    iterations_payload = []
    for it in result.iterations:
        entry = {
            "iteration": it.iteration,
            "min_samples_leaf_used": it.min_samples_leaf_used,
            "boundaries_tried": it.boundaries_tried,
            "passed": it.passed,
            "fail_reason": it.fail_reason,
            "next_min_samples_leaf": it.next_min_samples_leaf,
            "step1": {
                "best_alpha": it.step1_result.best_alpha.ccp_alpha,
                "best_alpha_n_leaves": it.step1_result.best_alpha.n_leaves,
                "alpha_candidates": [
                    {
                        "ccp_alpha": c.ccp_alpha,
                        "cv_mse_mean": c.cv_mse_mean,
                        "cv_mse_std": c.cv_mse_std,
                        "n_leaves": c.n_leaves,
                        "boundaries_tenure": c.boundaries_tenure,
                    }
                    for c in it.step1_result.alpha_candidates
                ],
                "rf_vote_top20": it.step1_result.rf_vote.threshold_votes.head(20).to_dict(),
                "signal_tenure": it.step1_result.signal.index.tolist(),
                "signal_churn_rate": it.step1_result.signal["churn_rate"].tolist(),
                "signal_n_customers": it.step1_result.signal["n_customers"].tolist(),
            },
        }
        if it.step2_result is not None:
            entry["step2"] = {
                "observed_auc": it.step2_result.observed_auc,
                "p_value": it.step2_result.p_value,
                "null_auc_distribution": it.step2_result.null_auc_distribution.tolist(),
                "n_permutations": it.step2_result.n_permutations,
            }
        if it.step3_result is not None:
            entry["step3"] = {
                "observed_auc": it.step3_result.observed_auc,
                "ci_lower": it.step3_result.ci_lower,
                "ci_upper": it.step3_result.ci_upper,
                "ci_width": it.step3_result.ci_width,
                "bootstrap_auc_distribution": it.step3_result.bootstrap_auc_distribution.tolist(),
                "n_bootstrap": it.step3_result.n_bootstrap,
            }
        iterations_payload.append(entry)

    return {
        "final_boundaries": result.final_boundaries,
        "converged": result.converged,
        "stop_reason": result.stop_reason,
        "iterations": iterations_payload,
    }
