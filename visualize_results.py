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

한글 폰트 관련 안내:
  이 스크립트는 시스템에 설치된 한글 폰트(맑은 고딕/Apple SD Gothic Neo/
  Noto Sans CJK/나눔고딕 등)를 자동으로 찾아 사용한다. 한글 폰트를 찾지
  못하면 한글이 깨지는 대신 모든 라벨을 자동으로 영문으로 표시한다
  (실행 시 콘솔에 어느 쪽인지 안내 메시지가 뜬다). 한글로 보고 싶은데
  영문으로 나온다면, OS에 한글 폰트를 설치한 뒤(Windows/Mac은 보통
  기본 포함, Linux는 `sudo apt install fonts-nanum` 등) 다시 실행하면 된다.
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


# ---------------------------------------------------------------------------
# 한글 폰트 자동 탐색
# ---------------------------------------------------------------------------
# ⚠️ 이전 버전은 "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"라는
# 절대경로를 하드코딩했다. 이는 코드를 작성한 컨테이너 환경에만 존재하는
# 경로라, 다른 PC(특히 Windows/Mac, 또는 이 폰트가 없는 Linux)에서 실행하면
# 이 경로가 존재하지 않아 폰트 설정이 조용히 건너뛰어지고, matplotlib
# 기본 폰트(한글 글리프가 없는 DejaVu Sans)로 폴백되어 모든 한글이 빈
# 사각형(□, tofu)으로 깨지는 문제가 실제로 재현됐다.
#
# 이번 버전은 절대경로를 직접 지정하지 않고, matplotlib이 인식하는 시스템
# 폰트 전체를 스캔해서 이름에 한글 지원 폰트로 흔히 쓰이는 키워드가
# 포함된 폰트를 자동으로 찾는다. OS별로 기본 탑재되는 한글 폰트가 다르므로
# (Windows: 맑은 고딕, Mac: Apple SD Gothic Neo, Linux: Noto Sans CJK/나눔고딕
# 등) 여러 후보 키워드를 순서대로 시도한다. 그래도 못 찾으면(한글 폰트가
# 전혀 설치되지 않은 환경) 한글 텍스트를 깨진 채로 그리는 대신, 그래프의
# 모든 한글 라벨을 영문으로 자동 치환해서 최소한 읽을 수 있는 결과물을
# 만든다 (_LABELS 딕셔너리, _t() 함수 참고).
_KOREAN_FONT_KEYWORDS = [
    "malgun gothic",       # Windows 기본 한글 폰트
    "apple sd gothic neo", # macOS 기본 한글 폰트
    "applegothic",
    "noto sans cjk kr",
    "noto sans kr",
    "noto sans cjk jp",    # 이 환경(컨테이너)에 설치된 폰트
    "noto sans cjk",
    "nanum gothic",        # Linux에 흔히 깔리는 한글 폰트
    "nanumgothic",
    "gulim", "batang", "dotum",  # 구형 Windows 한글 폰트
]


def _find_korean_font() -> str | None:
    """
    시스템에 설치된 폰트 중 한글 지원 가능성이 높은 폰트의 family 이름을
    찾는다. 찾으면 그 이름을, 못 찾으면 None을 반환한다.
    """
    try:
        font_paths = fm.findSystemFonts()
    except Exception:
        return None

    name_to_path: dict[str, str] = {}
    for fpath in font_paths:
        try:
            name = fm.FontProperties(fname=fpath).get_name()
        except Exception:
            continue
        name_to_path.setdefault(name.lower(), fpath)

    for keyword in _KOREAN_FONT_KEYWORDS:
        for lower_name, fpath in name_to_path.items():
            if keyword in lower_name:
                fm.fontManager.addfont(fpath)
                return fm.FontProperties(fname=fpath).get_name()
    return None


_KOREAN_FONT_FOUND = _find_korean_font()
if _KOREAN_FONT_FOUND:
    plt.rcParams["font.family"] = _KOREAN_FONT_FOUND
    print(f"[visualize_results] 한글 폰트 사용: {_KOREAN_FONT_FOUND}")
else:
    print(
        "[visualize_results] 경고: 시스템에서 한글 폰트를 찾지 못했습니다. "
        "그래프 라벨을 영문으로 표시합니다. 한글로 보려면 나눔고딕 등 한글 "
        "폰트를 설치한 뒤 다시 실행하세요."
    )
plt.rcParams["axes.unicode_minus"] = False


# 한글 폰트가 없을 때 쓸 영문 대체 라벨. 한글 폰트를 찾았으면 키(한글)를
# 그대로 쓰고, 못 찾았으면 값(영문)으로 자동 치환한다 (_t 함수).
_LABELS = {
    "분석 A: PELT(비교군) vs 가지치기 회귀나무(메인) 경계 비교":
        "Analysis A: PELT (baseline) vs Pruning Tree (main) - boundary comparison",
    "tenure (개월)": "tenure (months)",
    "PELT K=": "PELT K=",
    "가지치기회귀나무 K=": "PruningTree K=",
    " (최종)": " (final)",
    "  (시도 ": "  (try ",
    ", 실패: ": ", failed: ",
    "① 가지치기 회귀나무: ccp_alpha 전수탐색": "Step1: Pruning tree - ccp_alpha full search",
    "선택된 alpha=": "selected alpha=",
    "5-fold CV MSE\n(낮을수록 좋음)": "5-fold CV MSE\n(lower is better)",
    "리프 개수\n(=세그먼트 수 K)": "Number of leaves\n(=segments K)",
    "ccp_alpha": "ccp_alpha",
    "② 순열검정: 경계 ": "Step2: Permutation test - boundary ",
    " (p-value=": " (p-value=",
    "세그먼트단독 AUC": "Segment-only AUC",
    "빈도": "Frequency",
    "순열(우연) 분포 (n=": "Permuted (null) dist. (n=",
    "관측 AUC=": "Observed AUC=",
    "순열 평균=": "Null mean=",
    "③ 부트스트랩 신뢰구간: 경계 ": "Step3: Bootstrap CI - boundary ",
    "부트스트랩 분포 (n=": "Bootstrap dist. (n=",
    "95% CI [": "95% CI [",
    "] (폭=": "] (width=",
    "①②③ 반복 사이클 진행 (초록=통과, 빨강=실패->재실행)":
        "Cycle 1-2-3 progress (green=passed, red=failed->retry)",
    "min_samples_leaf": "min_samples_leaf",
    "② p-value": "Step2 p-value",
    "p=0.05 기준": "p=0.05 threshold",
    "③ CI 폭": "Step3 CI width",
    "폭 0.04(±0.02) 기준": "width=0.04 (±0.02) threshold",
    "반복 회차": "iteration",
}


def _t(text: str) -> str:
    """
    한글 폰트를 찾았으면 입력 텍스트를 그대로 반환하고, 못 찾았으면
    _LABELS에 등록된 한글 조각들을 영문으로 치환해서 반환한다.
    완전 일치가 아니라 부분 치환이므로, f-string으로 동적 생성된
    텍스트(예: "PELT K=4")에도 적용 가능하다.
    """
    if _KOREAN_FONT_FOUND:
        return text
    result = text
    for kr, en in _LABELS.items():
        result = result.replace(kr, en)
    return result


def _set_ylabel_kr(ax, text: str, labelpad: int = 28):
    """
    한글이 포함된 ylabel을 깨지지 않게 설정한다.

    matplotlib 기본 ylabel은 90도 회전(세로쓰기)되는데, 한글 폰트를
    회전된 상태로 렌더링하면 글리프가 뭉개져 보이는 경우가 있다
    (이 환경에서 Noto Sans CJK로 실제 확인됨). rotation=0으로
    가로쓰기 그대로 두고 축 왼쪽에 배치하는 방식으로 우회한다.
    """
    ax.set_ylabel(_t(text), rotation=0, labelpad=labelpad, ha="right", va="center")


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
        tree_label = "PruningTree" if not _KOREAN_FONT_FOUND else "가지치기회귀나무"
        final_label = "(final)" if not _KOREAN_FONT_FOUND else "(최종)"
        yticklabels.append(f"{tree_label} K={k} {final_label}")

        # 시도했지만 통과하지 못한 회차들도 흐리게 함께 표시
        for it in tree_data["iterations"][:-1]:
            y += 1
            b = it["boundaries_tried"]
            ax.scatter(b, [y] * len(b), marker="|", s=200, color="tab:orange", alpha=0.5)
            ax.plot([0, 73], [y, y], color="tab:orange", alpha=0.08, linewidth=8)
            yticks.append(y)
            if _KOREAN_FONT_FOUND:
                yticklabels.append(f"  (시도 {it['iteration']}, 실패: {it['fail_reason']})")
            else:
                yticklabels.append(f"  (try {it['iteration']}, failed: {it['fail_reason']})")

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel(_t("tenure (개월)"))
    ax.set_xlim(0, 73)
    ax.set_title(_t("분석 A: PELT(비교군) vs 가지치기 회귀나무(메인) 경계 비교"))
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
    ax1.axvline(best_alpha, color="red", linestyle="--", label=_t(f"선택된 alpha={best_alpha:.6f}"))
    _set_ylabel_kr(ax1, "5-fold CV MSE\n(낮을수록 좋음)")
    ax1.set_title(_t("① 가지치기 회귀나무: ccp_alpha 전수탐색"))
    ax1.legend()

    ax2.plot(alphas, n_leaves, "o-", markersize=3, color="tab:purple", alpha=0.7)
    ax2.axvline(best_alpha, color="red", linestyle="--")
    ax2.set_xlabel("ccp_alpha")
    _set_ylabel_kr(ax2, "리프 개수\n(=세그먼트 수 K)")
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
    ax.hist(null_dist, bins=20, color="lightgray", edgecolor="gray", label=_t(f"순열(우연) 분포 (n={step2['n_permutations']})"))
    ax.axvline(observed, color="red", linewidth=2, label=_t(f"관측 AUC={observed:.4f}"))
    ax.axvline(null_dist.mean(), color="gray", linestyle="--", label=_t(f"순열 평균={null_dist.mean():.4f}"))
    ax.set_xlabel(_t("세그먼트단독 AUC"))
    _set_ylabel_kr(ax, "빈도")
    ax.set_title(_t(f"② 순열검정: 경계 {iteration['boundaries_tried']} (p-value={p_value:.4f})"))
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
    ax.hist(boot_dist, bins=20, color="lightblue", edgecolor="steelblue", label=_t(f"부트스트랩 분포 (n={step3['n_bootstrap']})"))
    ax.axvline(observed, color="red", linewidth=2, label=_t(f"관측 AUC={observed:.4f}"))
    ax.axvspan(ci_lower, ci_upper, color="orange", alpha=0.2, label=_t(f"95% CI [{ci_lower:.4f}, {ci_upper:.4f}] (폭={step3['ci_width']:.4f})"))
    ax.set_xlabel(_t("세그먼트단독 AUC"))
    _set_ylabel_kr(ax, "빈도")
    ax.set_title(_t(f"③ 부트스트랩 신뢰구간: 경계 {iteration['boundaries_tried']}"))
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
    _set_ylabel_kr(axes[0], "min_samples_leaf", labelpad=10)
    axes[0].set_title(_t("①②③ 반복 사이클 진행 (초록=통과, 빨강=실패->재실행)"))

    valid_p = [(i, p) for i, p in zip(its, p_values) if p is not None]
    if valid_p:
        xs, ys = zip(*valid_p)
        axes[1].plot(xs, ys, "o-", color="tab:purple")
        axes[1].axhline(0.05, color="gray", linestyle="--", label=_t("p=0.05 기준"))
        _set_ylabel_kr(axes[1], "② p-value", labelpad=10)
        axes[1].legend()

    valid_ci = [(i, c) for i, c in zip(its, ci_widths) if c is not None]
    if valid_ci:
        xs, ys = zip(*valid_ci)
        axes[2].plot(xs, ys, "o-", color="tab:blue")
        axes[2].axhline(0.04, color="gray", linestyle="--", label=_t("폭 0.04(±0.02) 기준"))
        _set_ylabel_kr(axes[2], "③ CI 폭", labelpad=10)
        axes[2].set_xlabel(_t("반복 회차"))
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
