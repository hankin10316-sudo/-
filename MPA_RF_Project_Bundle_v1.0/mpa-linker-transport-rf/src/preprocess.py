"""preprocess.py — 数据预处理与训练/测试切分。

随机森林对特征的量纲不敏感，本数据集无需标准化/归一化。
此处提供：
- build_pipeline() : 返回一个 sklearn Pipeline（当前为 RandomForestRegressor，
                     便于后续插入缺失值填充、编码等步骤而无需改动训练脚本）。
- split_data()     : 可复现的训练/测试切分。
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config


def build_pipeline() -> Pipeline:
    """构造模型流水线。当前仅含随机森林回归器。"""
    model = RandomForestRegressor(**config.RF_PARAMS)
    return Pipeline([("regressor", model)])


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """按固定随机种子切分训练集/测试集。"""
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
