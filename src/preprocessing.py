"""
전처리 모듈
기획구현.md 2번 섹션 "전처리 세부 사항" 구현

핵심 원칙(기획구현.md 1번):
- pandas DataFrame 하나(df)를 끝까지 유지. customerID 등 식별 컬럼을 위한
  별도 테이블을 만들지 않는다. 모델 입력 X는 필요한 시점에 df에서 컬럼만 선택.
- 누수 방지: 더미 컬럼 목록, 스케일러의 fit은 전부 Train으로만 수행하고
  Test에는 적용만 한다.

이 모듈은 아래까지만 책임진다 (기획구현.md 1번 파이프라인 순서 기준):
  원본 로드 → 자료형 정리 → 결측치 처리 → 중복정보 통합(No internet/phone service)
  → Train/Test 계층화 분할

세그먼트 컬럼 추가, 인코딩, 스케일링은 이후 단계(분석 A 확정 이후)의 책임이므로
여기서는 다루지 않는다.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

# "No internet service" / "No phone service" 가 등장하는 컬럼들.
# 기획구현.md 2번 표: InternetService_No / PhoneService_No 와 100% 중복되므로
# 원-핫 인코딩 시 다중공선성을 유발한다 -> "No"로 통합해서 제거.
_NO_INTERNET_SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
_NO_PHONE_SERVICE_COLS = ["MultipleLines"]


def load_raw(csv_path: str) -> pd.DataFrame:
    """원본 CSV 로드. 변형 없이 그대로 읽는다."""
    df = pd.read_csv(csv_path)
    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    자료형 정리.
    기획구현.md: TotalCharges는 문자열로 들어오므로 pd.to_numeric(errors='coerce')
    로 숫자 변환 (공백 문자열 등은 NaN이 된다).
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    결측치 처리.
    기획구현.md: TotalCharges 결측(11건)은 0으로 대체.
    실데이터 확인: 전부 tenure=0인 신규고객(아직 청구 전), 전부 비이탈.
    """
    df = df.copy()
    n_missing = df["TotalCharges"].isna().sum()
    if n_missing > 0:
        df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def consolidate_no_service_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    "No internet service" / "No phone service" 값을 "No"로 통합.

    기획구현.md 2번: 정보손실 방지 목적이 아니라, InternetService_No /
    PhoneService_No 컬럼과 원-핫 인코딩 시 100% 중복되어 다중공선성을
    유발하기 때문에 통합한다.
    """
    df = df.copy()
    for col in _NO_INTERNET_SERVICE_COLS:
        if col in df.columns:
            df[col] = df[col].replace("No internet service", "No")
    for col in _NO_PHONE_SERVICE_COLS:
        if col in df.columns:
            df[col] = df[col].replace("No phone service", "No")
    return df


def run_preprocessing(csv_path: str) -> pd.DataFrame:
    """
    전처리 파이프라인 일괄 실행.
    원본 로드 -> 자료형 정리 -> 결측치 처리 -> 중복정보 통합.
    customerID 등 모든 컬럼을 df에 그대로 보존한다(식별/매칭/손해추정용).
    """
    df = load_raw(csv_path)
    df = fix_dtypes(df)
    df = handle_missing(df)
    df = consolidate_no_service_categories(df)
    return df


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
    target_col: str = "Churn",
):
    """
    Train/Test 계층화 분할.
    기획구현.md: train_test_split(stratify=Churn, test_size=0.3).

    df 자체를 분할해 반환한다 (X/y로 미리 쪼개지 않음 -> "DataFrame 하나를
    끝까지 유지" 원칙 준수, 모델 입력은 이후 단계에서 필요한 컬럼만 선택).
    """
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    return df_train, df_test
