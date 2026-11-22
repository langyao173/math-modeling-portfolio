"""Minimal submission schema and atomic output."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .input_schema import ID_COLUMNS, first_existing


PROBABILITY_COLUMNS = ["主队胜概率", "平局概率", "客队胜概率"]


def build_minimal_submission(query: pd.DataFrame, rich: pd.DataFrame) -> pd.DataFrame:
    if len(query) != len(rich):
        raise ValueError("预测输出行数与查询输入不一致。")
    id_col = first_existing(query.columns, ID_COLUMNS)
    if id_col and not query[id_col].notna().any():
        id_col = None
    elif id_col and query[id_col].isna().any():
        raise ValueError(f"编号列“{id_col}”部分为空；请全部填写或全部留空。")

    result = pd.DataFrame(index=np.arange(len(query)))
    if id_col:
        result["id"] = query[id_col].to_numpy()
    result["prediction"] = rich["prediction"].to_numpy()
    if result["prediction"].isna().any():
        raise ValueError("预测结果存在空值。")
    numeric = pd.to_numeric(result["prediction"], errors="coerce")
    if numeric.isna().any() or not set(numeric.astype(int)).issubset({0, 1, 2}):
        raise ValueError("prediction 必须为 0、1、2 之一。")
    result["prediction"] = numeric.astype(int)

    expected = ["id", "prediction"] if id_col else ["prediction"]
    if result.columns.tolist() != expected:
        raise ValueError("cla_pre.csv 列协议不符合要求。")
    if all(name in rich for name in PROBABILITY_COLUMNS):
        probabilities = rich[PROBABILITY_COLUMNS].to_numpy(float)
        if not np.isfinite(probabilities).all() or not np.allclose(
            probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-10
        ):
            raise ValueError("分类概率含非有限值，或每行概率之和不为 1。")
        if not np.array_equal(probabilities.argmax(axis=1), result["prediction"].to_numpy()):
            raise ValueError("prediction 必须由同一行三分类概率的 argmax 生成。")
    return result


def write_atomic_submission(frame: pd.DataFrame, destination: Path) -> Path:
    destination = Path(destination)
    temporary = destination.with_name("cla_pre.tmp.csv")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        check = pd.read_csv(temporary)
        if check.columns.tolist() != frame.columns.tolist() or len(check) != len(frame):
            raise OSError("临时输出文件回读校验失败。")
        os.replace(temporary, destination)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise OSError(
            f"无法写入输出文件“{destination}”；请检查目录权限。原始错误：{exc}"
        ) from exc
    return destination
