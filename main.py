"""
실행 진입점.
현재 구현 범위:
  - 전처리 (기획구현.md 1~2번)
  - 분석 A 1단계 - PELT 통계적 변화점 탐지 (기획구현.md 3번 1단계)
  - 분석 A 2단계 - 머신러닝 기반 표본충분성 검증 + 사후 통합 (기획구현.md 3번 2단계)

분석 B, 보조 분석 Q, 예측모델 1·2·3단계(기획구현.md 5번)는 아직 작성하지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "A_step1_methods"))
sys.path.append(str(PROJECT_ROOT / "A_step2_methods"))

from src.preprocessing import run_preprocessing, split_train_test  # noqa: E402
from A_step1_methods import pelt  # noqa: E402
from A_step2_methods import run_step2 as step2_module  # noqa: E402

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

    print(f"  - 검정력 분석 기반 표본충분성 기준(그룹당 '전체 표본 수', 참고치): "
          f"{pelt_result.min_group_sample_size}명")
    print(f"    -> 개월 환산 약 {pelt_result.min_segment_months_reference:.1f}개월, "
          f"기술적 상한 K≈{pelt_result.technical_k_upper_bound:.1f}")
    print(f"    (이 값은 1단계 PELT의 제약으로 쓰지 않습니다. 표본충분성")
    print(f"     실제 판정은 2단계에서 '전체 표본 수' 기준으로 수행됩니다")
    print(f"     -- 이탈 건수 기준이 아님에 유의. 표본충분성 논리 재검증 결과 반영)")
    print(f"  - 1단계 PELT에 실제 사용된 min_size: {pelt_result.min_size_used}개월 "
          f"(k_search_range 최대 K까지 후보가 나올 수 있도록 보장하는 느슨한 가드레일)")
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
    print("    사후 표본 점검('전체 표본 수' 393명 미달 구간 통합 여부)과")
    print("    머신러닝 기반 교차검증 성능 비교는 분석 A 2단계에서 수행됩니다.")
    print("    (기획구현.md 3번)")

    return {
        "df": df,
        "df_train": df_train,
        "df_test": df_test,
        "pelt_result": pelt_result,
    }


def run_step2(
    df_train,
    pelt_result: "pelt.PeltResult",
    n_cv_splits: int = 5,
    n_bootstrap: int = 1000,
):
    """
    분석 A 2단계 - 머신러닝 기반 표본충분성 검증 + 사후 통합 실행.
    1단계(run_step1) 결과를 그대로 입력받는다.

    Returns
    -------
    A_step2_methods.run_step2.Step2Result
    """
    print()
    print("=" * 70)
    print("분석 A 2단계 - 머신러닝 기반 표본충분성 검증 + 사후 통합")
    print("=" * 70)
    print(f"  (대상 후보 {len(pelt_result.candidates)}개, "
          f"교차검증 {n_cv_splits}-fold, 부트스트랩 {n_bootstrap}회)")
    print("  실행 중... (XGBoost 교차검증 + 부트스트랩은 다소 시간이 걸릴 수 있습니다)")

    result = step2_module.run_step2(
        df_train,
        pelt_result,
        n_cv_splits=n_cv_splits,
        n_bootstrap=n_bootstrap,
    )

    print()
    print(f"  - 표본충분성 임계값('전체 표본 수' 기준, 1단계와 동일 출처): "
          f"{result.min_group_sample_size}명")

    print()
    print("  [a. XGBoost 교차검증 성능 (K=1 베이스라인 포함)]")
    cv_table = result.cv_table()
    print(cv_table.to_string(index=False))

    print()
    print("  [b. 사후 표본 점검: '전체 표본 수' 393명 미달 구간 통합 전/후 비교]")
    merge_table = result.merge_summary_table()
    print(merge_table.to_string(index=False))
    for m in result.merge_results:
        if m.merge_log:
            print(f"    - 원본 경계 {m.original_boundaries}:")
            for line in m.merge_log:
                print(f"        · {line}")

    print()
    print("  [c. 부트스트랩 변동계수(CV) 요약 - K별 안정성]")
    boot_table = result.bootstrap_table()
    print(boot_table.to_string(index=False))

    print()
    print("  ※ 최종 K와 경계는 자동으로 채택하지 않습니다. 위 세 표(성능 /")
    print("    통합결과 / CV안정성)를 사람이 종합 판단해 확정합니다.")
    print("    (기획구현.md 4번-②: '임계값은 자동 탐지하지 않음')")

    return result


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV_PATH
    step1_out = run_step1(csv_path)
    step2_out = run_step2(step1_out["df_train"], step1_out["pelt_result"])
