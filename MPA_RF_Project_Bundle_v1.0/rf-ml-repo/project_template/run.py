"""run.py — 一键运行：清洗原始数据（若存在）-> 训练随机森林 -> 产出评估与图表。

用法：
    python run.py --target <目标列> [--id-col <标识列>] [--task regression|classification]

- 若 data/raw/ 下存在 CSV，会先用 src/clean_csv 清洗为 data/processed/dataset.csv；
- 否则直接使用 data/processed/dataset.csv（请先放入你自己的数据并命名，或用 --data 指定）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import clean_csv, train  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="运行随机森林流水线。")
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--id-col", type=str, default=None)
    parser.add_argument("--task", type=str, default="auto",
                        choices=["auto", "regression", "classification"])
    parser.add_argument("--data", type=Path, default=None, help="清洗后的数据路径（缺省时自动推断）")
    args = parser.parse_args()

    # 1) 推断数据路径：优先用显式 --data；否则尝试清洗 raw，或直接使用 processed/dataset.csv
    data_path = args.data
    if data_path is None:
        raw_csvs = sorted((ROOT / "data" / "raw").glob("*.csv"))
        default_clean = ROOT / "data" / "processed" / "dataset.csv"
        if raw_csvs:
            print(f"[INFO] 发现原始文件 {raw_csvs[0].name}，先清洗...")
            clean_csv.clean(raw_csvs[0], default_clean)
            data_path = default_clean
        elif default_clean.exists():
            data_path = default_clean
        else:
            print("[ERROR] 未找到数据。请将 CSV 放入 data/processed/ 或 data/raw/，或用 --data 指定。", file=sys.stderr)
            sys.exit(1)

    # 2) 训练
    sys.argv = ["train", "--data", str(data_path), "--target", args.target]
    if args.id_col:
        sys.argv += ["--id-col", args.id_col]
    if args.task != "auto":
        sys.argv += ["--task", args.task]
    train.main()

    print("\n[DONE] 流水线完成。查看 results/ 与 models/。")


if __name__ == "__main__":
    main()
