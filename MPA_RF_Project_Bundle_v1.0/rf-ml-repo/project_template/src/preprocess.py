"""preprocess.py — 数据切分与 sklearn Pipeline 构造。

随机森林对量纲不敏感，无需标准化。Pipeline 便于后续插入缺失值填充/编码步骤。
支持回归（RandomForestRegressor）与分类（RandomForestClassifier）两种任务。
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from . import config


def detect_task(y: pd.Series) -> str:
    """根据目标列类型推断任务：类别/字符串 -> classification；数值 -> regression。"""
    if y.dtype == object or str(y.dtype).startswith("category"):
        return "classification"
    uniq = y.nunique(dropna=True)
    # 数值但取值很少且为整数 -> 视为分类
    if uniq <= 20 and (y.dropna() % 1 == 0).all():
        return "classification"
    return "regression"


def build_pipeline(task: str = "regression") -> Pipeline:
    """构造模型流水线。"""
    if task == "classification":
        est = RandomForestClassifier(**config.RF_PARAMS)
    else:
        est = RandomForestRegressor(**config.RF_PARAMS)
    return Pipeline([("model", est)])


def split_data(X: pd.DataFrame, y: pd.Series, task: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """可复现训练/测试切分；分类任务按类别分层抽样。"""
    stratify = y if task == "classification" else None
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=stratify
    )
