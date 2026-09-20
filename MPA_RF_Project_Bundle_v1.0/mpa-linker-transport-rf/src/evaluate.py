"""evaluate.py — 加载已保存的模型并在清洗数据上重新评估。

可用于：训练后复核、模型对比、CI 中验证模型仍可用。

运行：
    python src/evaluate.py [--model models/random_forest_transport.pkl]
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config
from .data_loader import get_named_features, load_clean


def main():
    parser = argparse.ArgumentParser(description="评估已保存的随机森林模型。")
    parser.add_argument("--model", type=str, default=str(config.MODEL_PATH))
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    df = load_clean()
    names, X, y = get_named_features(df)

    y_pred = pipeline.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred) ** 0.5

    print(f"[INFO] 在全部 {len(y)} 个样本上的表现:")
    print(f"       R²   = {r2:.4f}")
    print(f"       MAE  = {mae:.4f}")
    print(f"       RMSE = {rmse:.4f}")

    # 输出每个化合物的预测误差，便于定位难样本
    result = pd.DataFrame({
        "compound_name": names.values,
        "true": y.values,
        "predicted": y_pred,
        "abs_error": (y.values - y_pred),
    })
    out = config.RESULTS_DIR / "per_compound_predictions.csv"
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[OK] 逐化合物预测结果已保存: {out}")


if __name__ == "__main__":
    main()
