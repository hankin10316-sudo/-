"""predict.py — 使用已训练模型对新样本推理转运率。

输入 CSV 需包含与训练一致的特征列（见 config.FEATURES）。
可包含 compound_name 列（仅作输出标识，不参与预测）。

运行：
    python src/predict.py --input new_samples.csv --output predictions.csv
    python src/predict.py --input new_samples.csv   # 默认输出到 results/predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from . import config


def main():
    parser = argparse.ArgumentParser(description="对新样本预测转运率。")
    parser.add_argument(
        "--input", type=Path, required=True,
        help="含特征列的 CSV 路径（需包含 config.FEATURES 中的列）。",
    )
    parser.add_argument(
        "--output", type=Path, default=config.RESULTS_DIR / "predictions.csv",
        help="预测结果输出路径。",
    )
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    df = pd.read_csv(args.input, encoding="utf-8-sig")

    missing = [c for c in config.FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"输入缺少必需特征列: {missing}")

    X = df[config.FEATURES]
    preds = pipeline.predict(X)

    out = df.copy()
    out["predicted_transport_rate"] = preds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[OK] 已对 {len(out)} 个样本预测，结果保存至: {args.output}")


if __name__ == "__main__":
    main()
