"""run.py — 一键运行完整流水线（解析原始数据 -> 训练 -> 评估 -> 产出结果）。

用法：
    python run.py              # 使用默认路径
    python run.py --input "原始CSV路径"   # 指定原始数据

说明：
- 原始数据（非标准分隔 CSV）由 src/parse_raw.py 清洗为 data/processed/mpa_linkers_clean.csv；
- 模型与评估指标写入 models/ 与 results/。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，使 `python run.py` 与 `python -m src.xxx` 均可运行
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import parse_raw, train  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="运行 MPA linker 随机森林流水线。")
    parser.add_argument(
        "--input", type=Path, default=None,
        help="原始 CSV 路径；缺省时使用 data/raw/MPA(1).csv。",
    )
    args = parser.parse_args()

    print("=== 步骤 1/2: 解析并清洗原始数据 ===")
    parse_raw.run(args.input)

    print("\n=== 步骤 2/2: 训练随机森林并评估 ===")
    train.main()

    print("\n[DONE] 流水线执行完毕。查看 results/ 与 models/ 目录。")


if __name__ == "__main__":
    main()
