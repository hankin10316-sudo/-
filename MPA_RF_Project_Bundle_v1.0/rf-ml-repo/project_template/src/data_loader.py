"""data_loader.py — 读取数据并按 (target, id_col) 推断特征矩阵 X 与目标 y。

通用化设计：不预设任何列名，特征 = 除 target 与 id_col 外的所有列。
"""

from __future__ import annotations

import pandas as pd

from . import config


def load_clean(path=None) -> pd.DataFrame:
    """读取清洗后的 CSV。"""
    path = path or config.DATA_PROCESSED / "dataset.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到数据文件: {path}\n请将 CSV 放入 data/processed/ 并指定 --data 路径。"
        )
    return pd.read_csv(path, encoding="utf-8-sig")


def resolve_features(df: pd.DataFrame, target: str, id_col: str | None = None) -> list[str]:
    """返回用于建模的特征列：全部列去掉 target 与 id_col。"""
    drop = {target}
    if id_col:
        drop.add(id_col)
    missing = [c for c in drop if c not in df.columns]
    if missing:
        raise ValueError(f"数据中缺少指定列: {missing}；可用列: {list(df.columns)}")
    return [c for c in df.columns if c not in drop]


def get_X_y(df: pd.DataFrame, target: str, id_col: str | None = None):
    """返回 (X, y)。"""
    feats = resolve_features(df, target, id_col)
    X = df[feats].copy()
    y = df[target].copy()
    return X, y


def get_named_features(df: pd.DataFrame, target: str, id_col: str | None = None):
    """返回带标识列名称的 (names, X, y)，便于结果回溯。"""
    names = df[id_col].copy() if id_col and id_col in df.columns else pd.Series([""] * len(df))
    X, y = get_X_y(df, target, id_col)
    return names, X, y
