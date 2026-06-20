"""
실행 진입점 (CLI, argparse 기반).

구현 범위:
  - 전처리 (기획구현가이드 1~2번)
  - 분석 A -- 두 가지 경로를 --method로 선택 실행
      "pelt"         : 기존 PELT+BIC 통계 경로 (A_step1_methods/A_step2_methods)
                       -- 비교군으로 보존, 코드 변경 없음
      "pruning_tree" : 기획구현가이드(A,B확정판).md 확정 방법론
                       (A_pruning_tree, ①②③ 반복 사이클)
      "both"         : 두 경로 모두 실행 (기본값)
  - 분석 A 대안 경로(A_ml_path, K-means/의사결정나무) -- 선택 실행(--ml-path)
  - 모든 실행 결과는 results/ 폴더에 JSON으로 자동 저장된다.
    -- visualize_results.py가 이 JSON만 읽어 그래프를 그리므로, 무거운
       재계산 없이 나중에 언제든 결과를 다시 비교/시각화할 수 있다.

사용 예:
  python main.py                                  # 기본: pelt + pruning_tree 둘 다, 결과 저장
  python main.py --method=pelt                    # PELT만
  python main.py --method=pruning_tree             # 가지치기 회귀나무만
  python main.py --method=pruning_tree --fast       # 빠른 설정(순열/부트스트랩 반복 축소, 개발용)
  python main.py --ml-path                          # + 전원 머신러닝 대안 경로까지
  python main.py --no-save                          # 결과 파일 저장 안 함
  python main.py data/다른파일.csv                  # 다른 CSV 사용

분석 B, 보조 분석 Q, 예측모델 1·2·3단계는 아직 작성하지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "A_step1_methods"))
sys.path.append(str(PROJECT_ROOT / "A_step2_methods"))
sys.path.append(str(PROJECT_ROOT / "A_ml_path"))
sys.path.append(str(PROJECT_ROOT / "A_pruning_tree"))

from src.preprocessing import run_preprocessing, split_train_test  # noqa: E402
from A_step1_methods import pelt  # noqa: E402
from A_step2_methods import run_step2 as step2_module  # noqa: E402
from A_ml_path import run_ml_path as ml_path_module  # noqa: E402
import run_cycle as pruning_cycle_module  # noqa: E402
import step1_tree as pruning_step1_module  # noqa: E402
import results_io  # noqa: E402

DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "WA_FnUseC_TelcoCustomerChurn.csv"


# ---------------------------------------------------------------------------
# 전처리 (공통)
# ---------------------------------------------------------------------------
def run_preprocessing_step(csv_path: str | Path = DEFAULT_CSV_PATH):
    """전처리 -> Train/Test 계층화 분할. 모든 경로가 공유하는 첫 단계."""
    print("=" * 70)
    print("[전처리] 로드 -> 자료형 정리 -> 결측치 처리 -> 중복정보 통합")
    print("=" * 70)
    df = run_preprocessing(str(csv_path))
    print(f"  - 전체 데이터: {df.shape[0]}건, {df.shape[1]}개 컬럼")
    print(f"  - TotalCharges 결측 처리 후 NaN 개수: {df['TotalCharges'].isna().sum()}")
    print(f"  - 이탈 건수: {(df['Churn'] == 'Yes').sum()}건 "
          f"({(df['Churn'] == 'Yes').mean():.1%})")

    print()
    print("[Train/Test 계층화 분할] test_size=0.3")
    df_train, df_test = split_train_test(df, test_size=0.3, random_state=42)
    print(f"  - Train: {df_train.shape[0]}건 / Test: {df_test.shape[0]}건")
    print(f"  - Train 이탈률: {(df_train['Churn'] == 'Yes').mean():.1%} / "
          f"Test 이탈률: {(df_test['Churn'] == 'Yes').mean():.1%}")

    return df, df_train, df_test


# ---------------------------------------------------------------------------
# 경로 A: 기존 PELT 통계 경로 (비교군, 코드 변경 없음)
# ---------------------------------------------------------------------------
def run_pelt_path(df_train, criterion: str = "bic", n_cv_splits: int = 5, n_bootstrap: int = 1000):
    """
    기존 PELT+BIC 경로 (1단계 통계적 변화점 탐지 -> 2단계 머신러닝 검증).
    A_step1_methods/A_step2_methods를 그대로 호출한다 (비교군, 변경 없음).
    """
    print()
    print("=" * 70)
    print(f"[비교군] 분석 A - PELT 경로 (criterion={criterion.upper()})")
    print("=" * 70)
    pelt_result = pelt.run_pelt(df_train, criterion=criterion)
    print(f"  - 탐지된 K 후보: {len(pelt_result.candidates)}개")
    print(pelt_result.summary_table().to_string(index=False))

    print()
    print("  [2단계: 사후 표본 점검 + XGBoost 재검증]")
    step2_result = step2_module.run_step2(
        df_train, pelt_result, n_cv_splits=n_cv_splits, n_bootstrap=n_bootstrap
    )
    print(step2_result.merge_summary_table().to_string(index=False))
    print()
    print(step2_result.cv_table().to_string(index=False))

    payload = {
        "pelt_step1": {
            "criterion": pelt_result.criterion,
            "min_group_sample_size": pelt_result.min_group_sample_size,
            "min_size_used": pelt_result.min_size_used,
            "k_search_range": list(pelt_result.k_search_range),
            "candidates": [
                {
                    "n_breakpoints": c.n_breakpoints,
                    "boundaries_tenure": c.boundaries_tenure,
                    "penalty_used": c.penalty_used,
                }
                for c in pelt_result.candidates
            ],
        },
        "step2_merge": step2_result.merge_summary_table().to_dict(orient="records"),
        "step2_cv": step2_result.cv_table().to_dict(orient="records"),
        "step2_bootstrap": step2_result.bootstrap_table().to_dict(orient="records"),
    }
    return pelt_result, step2_result, payload


# ---------------------------------------------------------------------------
# 경로 B: 가지치기 회귀나무 ①②③ 반복 사이클 (확정판 메인 방법론)
# ---------------------------------------------------------------------------
def run_pruning_tree_path(
    df_train,
    n_permutations: int = 200,
    n_bootstrap: int = 200,
    max_iterations: int = 10,
):
    """
    기획구현가이드(A,B확정판).md 확정 방법론.
    ① 가지치기 회귀나무(+RF 투표 보조) -> ② 세그먼트단독 AUC+순열검정
    -> ③ AUC 부트스트랩 신뢰구간, 실패 시 ①부터 재실행하는 반복 사이클.
    """
    print()
    print("=" * 70)
    print("[메인] 분석 A - 가지치기 회귀나무 ①②③ 반복 사이클")
    print("=" * 70)
    print(f"  (순열검정 {n_permutations}회, 부트스트랩 {n_bootstrap}회, "
          f"최대 반복 {max_iterations}회)")
    print("  실행 중... (순열검정/부트스트랩은 다소 시간이 걸릴 수 있습니다)")

    result = pruning_cycle_module.run_cycle(
        df_train,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        max_iterations=max_iterations,
    )

    print()
    print(f"  - 수렴 여부: {result.converged} (종료 사유: {result.stop_reason})")
    print(f"  - 최종 경계: {result.final_boundaries}")
    print()
    print("  [회차별 진행 기록]")
    print(pruning_cycle_module.cycle_summary_table(result).to_string(index=False))

    if result.converged and result.iterations:
        last = result.iterations[-1]
        print()
        print("  [최종 회차: ① RF 투표와의 일치성 점검]")
        agreement = pruning_step1_module.check_rf_agreement(
            last.step1_result.best_alpha, last.step1_result.rf_vote
        )
        print(agreement.to_string(index=False))

    payload = pruning_cycle_module.cycle_result_to_dict(result)
    return result, payload


# ---------------------------------------------------------------------------
# 경로 C: 전원 머신러닝 대안 경로 (선택 실행, 기존 그대로)
# ---------------------------------------------------------------------------
def run_ml_path_analysis(df_train, k_search_range=None, density_method: str = "kmeans", n_cv_splits: int = 5):
    """분석 A 대안 경로 - 전원 머신러닝(All-ML) 경계 탐지 실행 (선택 사항)."""
    print()
    print("=" * 70)
    print("[대안 경로] 분석 A - 전원 머신러닝(All-ML) 경계 탐지")
    print("=" * 70)

    result = ml_path_module.run_ml_path(
        df_train, k_search_range=k_search_range, density_method=density_method, n_cv_splits=n_cv_splits
    )
    print(result.cv_table().to_string(index=False))

    payload = {
        "density": result.density_summary_table().to_dict(orient="records"),
        "tree": result.tree_summary_table().to_dict(orient="records"),
        "cv": result.cv_table().to_dict(orient="records"),
    }
    return result, payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="가입 고객 이탈 예측 - 분석 A 실행 (PELT 비교군 vs 가지치기 회귀나무 확정 방법론)"
    )
    parser.add_argument(
        "csv_path", nargs="?", default=str(DEFAULT_CSV_PATH),
        help="입력 CSV 경로 (기본: data/WA_FnUseC_TelcoCustomerChurn.csv)",
    )
    parser.add_argument(
        "--method", choices=["pelt", "pruning_tree", "both"], default="both",
        help="실행할 분석 A 방법론 (기본: both)",
    )
    parser.add_argument(
        "--ml-path", action="store_true",
        help="전원 머신러닝 대안 경로(K-means->의사결정나무)도 함께 실행",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="개발/디버깅용 빠른 설정 (순열검정/부트스트랩 반복 횟수를 대폭 축소)",
    )
    parser.add_argument(
        "--n-permutations", type=int, default=None,
        help="② 순열검정 반복 횟수 (기본 200, --fast 시 30)",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=None,
        help="③ 부트스트랩 반복 횟수 (기본 200, --fast 시 30)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="결과를 results/ 폴더에 JSON으로 저장하지 않음 (기본은 저장함)",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    n_permutations = args.n_permutations if args.n_permutations is not None else (30 if args.fast else 200)
    n_bootstrap = args.n_bootstrap if args.n_bootstrap is not None else (30 if args.fast else 200)
    save_results = not args.no_save

    df, df_train, df_test = run_preprocessing_step(args.csv_path)

    if args.method in ("pelt", "both"):
        pelt_result, step2_result, pelt_payload = run_pelt_path(df_train)
        if save_results:
            path = results_io.save_result("pelt", pelt_payload)
            print(f"\n  [저장됨] {path}")

    if args.method in ("pruning_tree", "both"):
        cycle_result, tree_payload = run_pruning_tree_path(
            df_train, n_permutations=n_permutations, n_bootstrap=n_bootstrap
        )
        if save_results:
            path = results_io.save_result("pruning_tree", tree_payload)
            print(f"\n  [저장됨] {path}")

    if args.ml_path:
        ml_result, ml_payload = run_ml_path_analysis(df_train)
        if save_results:
            path = results_io.save_result("ml_path", ml_payload)
            print(f"\n  [저장됨] {path}")

    if save_results:
        print()
        print("=" * 70)
        print("결과가 results/ 폴더에 저장됐습니다.")
        print("나중에 시각화하려면: python visualize_results.py")
        print("=" * 70)


if __name__ == "__main__":
    main()
