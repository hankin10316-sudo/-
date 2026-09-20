"""evaluate.py — 加载已保存模型并在全部数据上复核（可选按 train/test 仅看测试集）。

运行：
    python -m src.evaluate --target <目标列> [--id-col <标识列>] [--data data/processed/dataset.csv]
"""

from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from . import config
from .data_loader import get_named_features, load_clean
from .preprocess import detect_task


def main():
    parser = argparse.ArgumentParser(description="评估已保存的随机森林模型。")
    parser.add_argument("--model", type=str, default=str(config.MODEL_PATH))
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--id-col", type=str, default=None)
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    df = load_clean(args.data)
    names, X, y = get_named_features(df, args.target, args.id_col)
    y_pred = pipeline.predict(X)

    task = detect_task(df[args.target])
    if task == "classification":
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")
        print(f"[INFO] full-data: ACC={acc:.4f} F1_macro={f1:.4f}")
    else:
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = mean_squared_error(y, y_pred) ** 0.5
        print(f"[INFO] full-data: R2={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f}")

    out = df.copy()
    if args.id_col and args.id_col in out.columns:
        out = out[[args.id_col]].copy()
    out["true"] = y.values
    out["predicted"] = y_pred
    out["abs_error"] = y.values - y_pred
    p = config.RESULTS_DIR / "per_sample_predictions.csv"
    out.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"[OK] per-sample predictions -> {p}")


if __name__ == "__main__":
    main()
