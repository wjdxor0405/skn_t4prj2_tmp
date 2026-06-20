"""
저장된 분석 A 결과(results/*.json)를 읽어 비교 시각화 PNG를 생성한다.

main.py와 완전히 분리된 스크립트다 -- 무거운 학습/검증(PELT, XGBoost,
가지치기 회귀나무, 순열검정, 부트스트랩)을 다시 돌리지 않고, 이미 저장된
JSON 결과만으로 그래프를 그린다. 그래서 결과 비교가 빠르고(수초 내),
같은 실행 결과를 여러 번 다른 방식으로 그려볼 수 있다.

사용법:
  python visualize_results.py                  # results/ 폴더의 가장 최근
                                                 # pelt / pruning_tree 결과를 자동으로 찾아 비교
  python visualize_results.py --pelt-file results/pelt_xxx.json \
                               --tree-file results/pruning_tree_xxx.json
  python visualize_results.py --out-dir figures/

생성되는 그림 (output 디렉토리에 PNG로 저장):
  1. boundaries_comparison.png  : PELT vs 가지치기 회귀나무 K별 경계 비교
  2. tree_alpha_curve.png        : ① ccp_alpha vs 교차검증 MSE 곡선
  3. permutation_test.png        : ② 순열검정 null 분포 vs 관측 AUC
  4. bootstrap_ci.png             : ③ AUC 부트스트랩 분포 + 신뢰구간
  5. cycle_progress.png           : ①②③ 반복 사이클 진행 (회차별 통과/실패)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 화면 없는 환경에서도 PNG 저장 가능하도록
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

import results_io

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = PROJECT_ROOT / "figures"

# 한글 폰트 설정 (이 환경에 설치된 Noto Sans CJK 사용, 없으면 기본 폰트로
# 폴백 -- 한글이 깨지더라도 스크립트 자체는 동작하게 한다).
_NOTO_CJK_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_NOTO_CJK_PATH).exists():
    fm.fontManager.addfont(_NOTO_CJK_PATH)
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 1. K별 경계 비교 (PELT vs 가지치기 회귀나무)
# ---------------------------------------------------------------------------
def plot_boundaries_comparison(pelt_data: dict | None, tree_data: dict | None, out_path: Path):
    """
    PELT가 K별로 제안한 경계들과, 가지치기 회귀나무가 최종 확정한 경계를
    tenure 축 위에 나란히 그려서 비교한다.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    y = 0
    yticks, yticklabels = [], []

    if pelt_data is not None:
        candidates = pelt_data["pelt_step1"]["candidates"]
        for c in sorted(candidates, key=lambda c: c["n_breakpoints"]):
            boundaries = c["boundaries_tenure"]
            k = c["n_breakpoints"] + 1
            ax.scatter(boundaries, [y] * len(boundaries), marker="|", s=300, color="tab:blue")
            ax.plot([0, 73], [y, y], color="tab:blue", alpha=0.15, linewidth=8)
            yticks.append(y)
            yticklabels.append(f"PELT K={k}")
            y += 1
        y += 0.5

    if tree_data is not None:
        final_boundaries = tree_data["final_boundaries"]
        if final_boundaries:
            ax.scatter(final_boundaries, [y] * len(final_boundaries), marker="|", s=300, color="tab:red")
        ax.plot([0, 73], [y, y], color="tab:red", alpha=0.15, linewidth=8)
        k = len(final_boundaries) + 1
        yticks.append(y)
        yticklabels.append(f"가지치기회귀나무 K={k} (최종)")

        # 시도했지만 통과하지 못한 회차들도 흐리게 함께 표시
        for it in tree_data["iterations"][:-1]:
            y += 1
            b = it["boundaries_tried"]
            ax.scatter(b, [y] * len(b), marker="|", s=200, color="tab:orange", alpha=0.5)
            ax.plot([0, 73], [y, y], color="tab:orange", alpha=0.08, linewidth=8)
            yticks.append(y)
            yticklabels.append(f"  (시도 {it['iteration']}, 실패: {it['fail_reason']})")

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("tenure (개월)")
    ax.set_xlim(0, 73)
    ax.set_title("분석 A: PELT(비교군) vs 가지치기 회귀나무(메인) 경계 비교")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. ① ccp_alpha vs CV MSE 곡선
# ---------------------------------------------------------------------------
def plot_tree_alpha_curve(tree_data: dict, out_path: Path):
    """최종(또는 최초) 회차의 alpha 후보 전수탐색 결과를 곡선으로 그린다."""
    iteration = tree_data["iterations"][-1] if tree_data["converged"] else tree_data["iterations"][0]
    candidates = iteration["step1"]["alpha_candidates"]
    candidates = sorted(candidates, key=lambda c: c["ccp_alpha"])

    alphas = [c["ccp_alpha"] for c in candidates]
    mse_means = [c["cv_mse_mean"] for c in candidates]
    mse_stds = [c["cv_mse_std"] for c in candidates]
    n_leaves = [c["n_leaves"] for c in candidates]

    best_alpha = iteration["step1"]["best_alpha"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.errorbar(alphas, mse_means, yerr=mse_stds, fmt="o-", markersize=3, color="tab:green", alpha=0.7)
    ax1.axvline(best_alpha, color="red", linestyle="--", label=f"선택된 alpha={best_alpha:.6f}")
    ax1.set_ylabel("5-fold CV MSE (낮을수록 좋음)")
    ax1.set_title("① 가지치기 회귀나무: ccp_alpha 전수탐색")
    ax1.legend()

    ax2.plot(alphas, n_leaves, "o-", markersize=3, color="tab:purple", alpha=0.7)
    ax2.axvline(best_alpha, color="red", linestyle="--")
    ax2.set_xlabel("ccp_alpha")
    ax2.set_ylabel("리프 개수 (= 세그먼트 수 K)")
    ax2.set_xscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. ② 순열검정 분포
# ---------------------------------------------------------------------------
def plot_permutation_test(tree_data: dict, out_path: Path):
    """마지막(또는 첫) 회차의 순열검정 null 분포와 관측 AUC를 히스토그램으로 비교."""
    iteration = tree_data["iterations"][-1] if tree_data["converged"] else tree_data["iterations"][0]
    step2 = iteration.get("step2")
    if step2 is None:
        return

    null_dist = np.array(step2["null_auc_distribution"])
    observed = step2["observed_auc"]
    p_value = step2["p_value"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null_dist, bins=20, color="lightgray", edgecolor="gray", label=f"순열(우연) 분포 (n={step2['n_permutations']})")
    ax.axvline(observed, color="red", linewidth=2, label=f"관측 AUC={observed:.4f}")
    ax.axvline(null_dist.mean(), color="gray", linestyle="--", label=f"순열 평균={null_dist.mean():.4f}")
    ax.set_xlabel("세그먼트단독 AUC")
    ax.set_ylabel("빈도")
    ax.set_title(f"② 순열검정: 경계 {iteration['boundaries_tried']} (p-value={p_value:.4f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. ③ 부트스트랩 신뢰구간
# ---------------------------------------------------------------------------
def plot_bootstrap_ci(tree_data: dict, out_path: Path):
    """마지막(또는 첫) 회차의 AUC 부트스트랩 분포와 95% 신뢰구간을 그린다."""
    iteration = tree_data["iterations"][-1] if tree_data["converged"] else tree_data["iterations"][0]
    step3 = iteration.get("step3")
    if step3 is None:
        return

    boot_dist = np.array(step3["bootstrap_auc_distribution"])
    observed = step3["observed_auc"]
    ci_lower, ci_upper = step3["ci_lower"], step3["ci_upper"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(boot_dist, bins=20, color="lightblue", edgecolor="steelblue", label=f"부트스트랩 분포 (n={step3['n_bootstrap']})")
    ax.axvline(observed, color="red", linewidth=2, label=f"관측 AUC={observed:.4f}")
    ax.axvspan(ci_lower, ci_upper, color="orange", alpha=0.2, label=f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}] (폭={step3['ci_width']:.4f})")
    ax.set_xlabel("세그먼트단독 AUC")
    ax.set_ylabel("빈도")
    ax.set_title(f"③ 부트스트랩 신뢰구간: 경계 {iteration['boundaries_tried']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. 반복 사이클 진행 요약
# ---------------------------------------------------------------------------
def plot_cycle_progress(tree_data: dict, out_path: Path):
    """회차별 min_samples_leaf, p-value, ci_width, 통과여부를 한 화면에 정리."""
    iterations = tree_data["iterations"]
    its = [it["iteration"] for it in iterations]
    msl = [it["min_samples_leaf_used"] for it in iterations]
    passed = [it["passed"] for it in iterations]
    p_values = [it["step2"]["p_value"] if it.get("step2") else None for it in iterations]
    ci_widths = [it["step3"]["ci_width"] if it.get("step3") else None for it in iterations]

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    colors = ["tab:green" if p else "tab:red" for p in passed]
    axes[0].bar(its, msl, color=colors)
    axes[0].set_ylabel("min_samples_leaf")
    axes[0].set_title("①②③ 반복 사이클 진행 (초록=통과, 빨강=실패->재실행)")

    valid_p = [(i, p) for i, p in zip(its, p_values) if p is not None]
    if valid_p:
        xs, ys = zip(*valid_p)
        axes[1].plot(xs, ys, "o-", color="tab:purple")
        axes[1].axhline(0.05, color="gray", linestyle="--", label="p=0.05 기준")
        axes[1].set_ylabel("② p-value")
        axes[1].legend()

    valid_ci = [(i, c) for i, c in zip(its, ci_widths) if c is not None]
    if valid_ci:
        xs, ys = zip(*valid_ci)
        axes[2].plot(xs, ys, "o-", color="tab:blue")
        axes[2].axhline(0.04, color="gray", linestyle="--", label="폭 0.04(±0.02) 기준")
        axes[2].set_ylabel("③ CI 폭")
        axes[2].set_xlabel("반복 회차")
        axes[2].legend()

    axes[0].set_xticks(its)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def run_visualization(pelt_file: str | None, tree_file: str | None, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    pelt_data = results_io.load_result(pelt_file) if pelt_file else results_io.latest_result("pelt")
    tree_data = results_io.load_result(tree_file) if tree_file else results_io.latest_result("pruning_tree")

    if pelt_data is None and tree_data is None:
        print("results/ 폴더에 저장된 결과가 없습니다. 먼저 main.py를 실행해 결과를 저장하세요.")
        return

    generated = []

    if pelt_data is not None or tree_data is not None:
        path = out_dir / "boundaries_comparison.png"
        plot_boundaries_comparison(pelt_data, tree_data, path)
        generated.append(path)

    if tree_data is not None:
        path = out_dir / "tree_alpha_curve.png"
        plot_tree_alpha_curve(tree_data, path)
        generated.append(path)

        path = out_dir / "permutation_test.png"
        plot_permutation_test(tree_data, path)
        generated.append(path)

        path = out_dir / "bootstrap_ci.png"
        plot_bootstrap_ci(tree_data, path)
        generated.append(path)

        path = out_dir / "cycle_progress.png"
        plot_cycle_progress(tree_data, path)
        generated.append(path)

    print(f"총 {len(generated)}개 그림을 {out_dir}/ 에 저장했습니다:")
    for p in generated:
        print(f"  - {p.name}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="저장된 분석 A 결과를 읽어 비교 시각화 PNG 생성")
    parser.add_argument("--pelt-file", default=None, help="PELT 결과 JSON 경로 (기본: results/의 가장 최근 pelt_*.json)")
    parser.add_argument("--tree-file", default=None, help="가지치기 회귀나무 결과 JSON 경로 (기본: 가장 최근 pruning_tree_*.json)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="그림 저장 폴더 (기본: figures/)")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_visualization(args.pelt_file, args.tree_file, Path(args.out_dir))
