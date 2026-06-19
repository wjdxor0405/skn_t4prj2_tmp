"""
실행 진입점.
현재 구현 범위: 전처리(기획구현.md 1~2번) + 분석 A 1단계(기획구현.md 3번 1단계).

분석 A 2단계(머신러닝 기반 표본충분성 검증, A_step2_methods/xgboost.py 등)는
아직 작성하지 않으므로 여기서 실행하지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "A_step1_methods"))

from src.preprocessing import run_preprocessing, split_train_test  # noqa: E402
from A_step1_methods import pelt  # noqa: E402

DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def run_step1(csv_path: str | Path = DEFAULT_CSV_PATH, criterion: str = "bic"):
    """
    전처리 -> Train/Test 분할 -> 분석 A 1단계(PELT) 실행.

    Returns
    -------
    dict
        df_train, df_test, pelt_result 를 담은 딕셔너리.
        (이후 분석 A 2단계, 분석 B 등에서 그대로 이어받아 쓸 수 있도록)
    """
    print("=" * 70)
    print("[1/3] 전처리 시작")
    print("=" * 70)
    df = run_preprocessing(str(csv_path))
    print(f"  - 전체 데이터: {df.shape[0]}건, {df.shape[1]}개 컬럼")
    print(f"  - TotalCharges 결측 처리 후 NaN 개수: {df['TotalCharges'].isna().sum()}")
    print(f"  - 이탈 건수: {(df['Churn'] == 'Yes').sum()}건 "
          f"({(df['Churn'] == 'Yes').mean():.1%})")

    print()
    print("=" * 70)
    print("[2/3] Train/Test 계층화 분할 (test_size=0.3)")
    print("=" * 70)
    df_train, df_test = split_train_test(df, test_size=0.3, random_state=42)
    print(f"  - Train: {df_train.shape[0]}건 / Test: {df_test.shape[0]}건")
    print(f"  - Train 이탈률: {(df_train['Churn'] == 'Yes').mean():.1%} / "
          f"Test 이탈률: {(df_test['Churn'] == 'Yes').mean():.1%}")

    print()
    print("=" * 70)
    print(f"[3/3] 분석 A 1단계 - PELT 변화점 탐지 (criterion={criterion.upper()})")
    print("=" * 70)
    pelt_result = pelt.run_pelt(df_train, criterion=criterion)

    print(f"  - 검정력 분석 기반 최소 표본(그룹당 이탈 건수): "
          f"{pelt_result.min_group_churn_count}건")
    print(f"  - 위 값을 개월 수로 환산한 '느슨한 시작값': "
          f"약 {pelt_result.min_segment_months:.1f}개월")
    print(f"  - 탐색한 K(세그먼트 수) 범위: {min(pelt_result.k_search_range)}"
          f"~{max(pelt_result.k_search_range)}")
    print(f"  - 실제 탐지된 후보 개수: {len(pelt_result.candidates)}개")
    print()
    print("  [K별 경계 후보 (tenure 개월 기준)]")
    summary = pelt_result.summary_table()
    if summary.empty:
        print("  (탐색 범위 내에서 후보를 찾지 못했습니다. "
              "k_search_range 또는 penalty scale 범위를 조정하세요.)")
    else:
        for _, row in summary.iterrows():
            print(f"    K={row['n_segments']}: 경계 tenure = {row['boundaries_tenure']}")

    print()
    print("  ※ 위 후보들은 '통계적 제안'일 뿐 최종 채택이 아닙니다.")
    print("    사후 표본 점검(이탈 392건 미달 구간 통합 여부)과 머신러닝")
    print("    기반 교차검증 성능 비교는 분석 A 2단계(A_step2_methods, 추후 구현)")
    print("    에서 수행됩니다. (기획구현.md 3번)")

    return {
        "df": df,
        "df_train": df_train,
        "df_test": df_test,
        "pelt_result": pelt_result,
    }


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_PATH
    run_step1(csv_path)
