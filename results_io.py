"""
분석 A 실행 결과 저장/로드 유틸.

main.py가 --method 옵션으로 PELT 경로 또는 가지치기 회귀나무 경로(또는
둘 다)를 실행한 뒤, 그 결과를 results/ 폴더에 JSON 파일로 저장한다.
visualize_results.py는 이 JSON만 읽어서 시각화하므로, 실행(main.py)과
시각화(visualize_results.py)가 완전히 분리된다 -- 무거운 학습/검증을
다시 돌리지 않고도 나중에 언제든 결과를 다시 그려볼 수 있다.

저장 형식: 1회 실행 = 파일 1개. 파일명에 method와 timestamp를 넣어
여러 번 실행한 결과가 서로 덮어쓰지 않게 한다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"


class _NumpyJSONEncoder(json.JSONEncoder):
    """numpy 스칼라/배열, pandas Series/Index를 JSON 직렬화 가능한
    파이썬 기본 타입으로 변환하는 인코더."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Series, pd.Index)):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        return super().default(obj)


def save_result(method: str, payload: dict, results_dir: Path | None = None) -> Path:
    """
    결과 딕셔너리를 JSON 파일로 저장한다.

    Parameters
    ----------
    method : str
        "pelt" 또는 "pruning_tree" 등 실행 방법 식별자. 파일명에 들어간다.
    payload : dict
        저장할 결과. 중첩된 dict/list/DataFrame/numpy 타입을 그대로 넣어도
        _NumpyJSONEncoder가 처리한다.

    Returns
    -------
    Path
        저장된 파일 경로.
    """
    results_dir = results_dir or RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = results_dir / f"{method}_{timestamp}.json"

    payload_with_meta = {
        "method": method,
        "saved_at": timestamp,
        **payload,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload_with_meta, f, ensure_ascii=False, indent=2, cls=_NumpyJSONEncoder)

    return filepath


def load_result(filepath: str | Path) -> dict:
    """저장된 JSON 결과 파일을 읽어 딕셔너리로 반환한다."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def list_results(method: str | None = None, results_dir: Path | None = None) -> list[Path]:
    """
    results/ 폴더에 저장된 결과 파일 목록을 최신순으로 반환한다.

    Parameters
    ----------
    method : str, optional
        지정하면 해당 method로 시작하는 파일만 반환 (예: "pelt", "pruning_tree").
    """
    results_dir = results_dir or RESULTS_DIR
    if not results_dir.exists():
        return []
    pattern = f"{method}_*.json" if method else "*.json"
    files = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def latest_result(method: str, results_dir: Path | None = None) -> dict | None:
    """method에 해당하는 가장 최근 결과를 로드한다. 없으면 None."""
    files = list_results(method=method, results_dir=results_dir)
    if not files:
        return None
    return load_result(files[0])
