"""config.py — 项目统一配置（路径、列定义、模型超参）。

集中管理所有可调参数，便于复现与二次开发。
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 目录
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

CLEAN_CSV = DATA_PROCESSED / "mpa_linkers_clean.csv"
MODEL_PATH = MODELS_DIR / "random_forest_transport.pkl"
METRICS_PATH = RESULTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_CSV = RESULTS_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_PNG = RESULTS_DIR / "feature_importance.png"
PREDICTION_SCATTER_PNG = RESULTS_DIR / "predicted_vs_actual.png"

# 数据字典（英文列名 -> 中文含义 / 说明）
COLUMN_DOC = {
    "compound_name": "化合物名称（仅作标识，不作为模型特征）",
    "logp_2mg": "2-MG LogP，亲脂性描述符（数值）",
    "asi": "是否含 ASI 自毁/连接基团（0/1 指示）",
    "masi": "是否含 MASI 基团（0/1 指示）",
    "phb": "是否含 PHB 基团（0/1 指示）",
    "dmphb": "是否含 DMPHB 基团（0/1 指示）",
    "cphb": "是否含 CPHB 基团（0/1 指示）",
    "cdmphb": "是否含 CDMPHB 基团（0/1 指示）",
    "fsi": "是否含 FSI 基团（0/1 指示）",
    "tml": "是否含 TML 基团（0/1 指示）",
    "ce4": "是否含 CE4 基团（0/1 指示）",
    "casi": "是否含 CASI 基团（0/1 指示）",
    "chain_length": "连接子链长（碳原子数，整数）",
    "self_immolative": "是否含自毁基团（0/1 指示）",
    "tg_linkage": "与 TG 的连接键类型（0/1 指示）",
    "branch_modifications": "支链修饰个数（整数）",
    "mw": "分子量 MW（数值）",
    "transport_rate": "转运率（回归目标，数值）",
}

# 模型使用的特征（排除标识列与目标列）
TARGET = "transport_rate"
IDENTIFIER = "compound_name"
FEATURES = [
    "logp_2mg",
    "asi", "masi", "phb", "dmphb", "cphb", "cdmphb",
    "fsi", "tml", "ce4", "casi",
    "chain_length",
    "self_immolative",
    "tg_linkage",
    "branch_modifications",
    "mw",
]

# 随机森林超参数（已做轻量调优，固定随机种子以保证可复现）
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
