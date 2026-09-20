"""predict.py — 使用已训练模型对新样本推理。

输入 CSV 只需包含与训练一致的特征列（目标列无需提供）。
运行：
    python -m src.predict --input new_samples.csv --output results/predictions.csv [--model models/random_forest.pkl]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from . import config


def main():
    parser = argparse.ArgumentParser(description="对新样本预测目标值。")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=config.RESULTS_DIR / "predictions.csv")
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--target", type=str, default=None,
                        help="目标列名（仅用于输出列命名，可选）")
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    model = pipeline.named_steps["model"]
    expected = list(getattr(model, "feature_names_in_", []))
    if not expected:
        raise RuntimeError("模型缺少 feature_names_in_，无法校验输入特征。")

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"输入缺少训练时的特征列: {missing}")
    extra = [c for c in df.columns if c not in expected]
    X = df[expected]
    preds = pipeline.predict(X)

    out = df.drop(columns=extra).copy() if extra else df.copy()
    out_col = "predicted_" + args.target if args.target else "prediction"
    out[out_col] = preds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[OK] predicted {len(out)} samples -> {args.output}")


if __name__ == "__main__":
    main()
