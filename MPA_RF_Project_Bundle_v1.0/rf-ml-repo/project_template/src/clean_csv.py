"""clean_csv.py — 通用清洗：处理「Excel 导出产生多余空列」的非标准分隔 CSV。

适用场景：某些导出文件每行字段数是有效列数的约 2 倍，有效列之间夹着空字段
（如 MPA linker 原始数据：每行 36 字段中仅 18 个有效）。本脚本通过表头中
「非空单元格的位置」定位有效列，并据此抽取所有数据行，输出规范 CSV。

对标准 CSV 同样安全（有效列即全部列）。

运行：
    python -m src.clean_csv --input raw.csv --output data/processed/dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from . import config


def clean(input_path: Path, output_path: Path) -> int:
    with input_path.open(encoding="utf-8-sig", newline="") as f:
        raw = [ln for ln in f.read().splitlines() if ln.strip() != ""]
    if not raw:
        raise ValueError("输入文件为空。")

    header = raw[0].split(",")
    valid_idx = [i for i, v in enumerate(header) if v.strip() != ""]
    cols = [header[i].strip() for i in valid_idx]
    if not cols:
        raise ValueError("表头未找到任何有效列。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for line in raw[1:]:
            fields = line.split(",")
            if max(valid_idx) >= len(fields):
                print(f"[WARN] 跳过字段不足的行", file=sys.stderr)
                continue
            w.writerow([fields[i].strip() for i in valid_idx])
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="清洗非标准分隔 CSV。")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=config.DATA_PROCESSED / "dataset.csv")
    args = parser.parse_args()
    n = clean(args.input, args.output)
    print(f"[OK] 清洗完成：{n} 行 -> {args.output}")


if __name__ == "__main__":
    main()
