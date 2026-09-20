"""config.py — 随机森林项目统一配置（路径与超参数）。

特征列和目标列不在此硬编码，而是在运行时由数据自动推断
（见 src/data_loader.resolve_features），因此本模板可直接套用任意 CSV。
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_PATH = MODELS_DIR / "random_forest.pkl"
METRICS_PATH = RESULTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_CSV = RESULTS_DIR / "feature_importance.csv"
FEATURE_IMPORTANCE_PNG = RESULTS_DIR / "feature_importance.png"
PREDICTED_VS_ACTUAL_PNG = RESULTS_DIR / "predicted_vs_actual.png"
CONFUSION_PNG = RESULTS_DIR / "confusion_matrix.png"

# 默认超参数（固定随机种子以保证可复现）
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
