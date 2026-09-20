"""data_loader.py — 读取清洗后的数据，提供训练/推理所需的数据接口。

对外暴露：
- load_clean()        : 读取 data/processed/mpa_linkers_clean.csv 为 DataFrame
- get_X_y()           : 返回特征矩阵 X 与目标向量 y（自动剔除标识列）
- get_named_features(): 返回带化合物名称的特征/目标，便于结果回溯
"""

from __future__ import annotations

import pandas as pd

from . import config


def load_clean(path=None) -> pd.DataFrame:
    """读取清洗后的 CSV。若文件不存在给出友好提示。"""
    path = path or config.CLEAN_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"未找到清洗数据: {path}\n"
            f"请先运行 `python src/parse_raw.py` 生成清洗数据，"
            f"或确认 data/processed/mpa_linkers_clean.csv 存在。"
        )
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def get_X_y(df: pd.DataFrame):
    """返回 (X, y)。X 为 FEATURES 指定的特征，y 为目标列。"""
    X = df[config.FEATURES].copy()
    y = df[config.TARGET].copy()
    return X, y


def get_named_features(df: pd.DataFrame):
    """返回带化合物名称的 (names, X, y)，便于将预测结果映射回具体化合物。"""
    names = df[config.IDENTIFIER].copy()
    X, y = get_X_y(df)
    return names, X, y
