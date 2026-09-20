"""parse_raw.py — 将原始 MPA linker 数据（非标准分隔的 CSV）清洗为规范化数据表。

原始文件特点
------------
源文件 ``MPA(1).csv`` 由 Excel 导出，存在大量多余的空列：每一行被逗号切分后有 36 个字段，
但其中只有 18 个是有效数据（名称 + 17 个特征/目标列），其余为成对出现的空字段。
有效字段位于固定位置：索引 0, 3, 5, 7, ..., 35。

本脚本将：
1. 按固定位置抽取有效字段；
2. 将中文表头映射为规范的英文列名（便于代码与跨语言协作）；
3. 统一数值类型；
4. 输出清洗后的 CSV 到 ``data/processed/mpa_linkers_clean.csv``。

用法
----
    python src/parse_raw.py --input "C:/path/to/MPA(1).csv" \
        --output data/processed/mpa_linkers_clean.csv
若省略参数，则使用脚本内置的默认路径。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# 项目根目录（本文件位于 <root>/src/parse_raw.py）
ROOT = Path(__file__).resolve().parent.parent

# 原始 CSV 中有效字段所在的位置（0-based）。
# 名称在 idx 0；之后每两个字段（值 + 空）出现一个有效值，即 3,5,7,...,35。
VALID_INDICES = [0] + list(range(3, 36, 2))
assert len(VALID_INDICES) == 18, f"期望 18 个有效列，实际 {len(VALID_INDICES)}"

# 中文表头 -> 英文列名 的映射（顺序与 VALID_INDICES 一一对应）
COLUMN_MAP = {
    "名称": "compound_name",
    "2-MG LogP": "logp_2mg",
    "ASI": "asi",
    "MASI": "masi",
    "PHB": "phb",
    "DMPHB": "dmphb",
    "CPHB": "cphb",
    "CDMPHB": "cdmphb",
    "FSI": "fsi",
    "TML": "tml",
    "CE4": "ce4",
    "CASI": "casi",
    "链长": "chain_length",
    "自毁基团": "self_immolative",
    "连接键（与TG）": "tg_linkage",
    "支链修饰个数": "branch_modifications",
    "MW": "mw",
    "转运率": "transport_rate",
}

# 数值型列（用于类型转换与校验）
NUMERIC_COLUMNS = {
    "logp_2mg", "asi", "masi", "phb", "dmphb", "cphb", "cdmphb",
    "fsi", "tml", "ce4", "casi", "chain_length", "self_immolative",
    "tg_linkage", "branch_modifications", "mw", "transport_rate",
}

HEADER_ZH = list(COLUMN_MAP.keys())


def parse_file(input_path: Path) -> tuple[list[str], list[dict]]:
    """读取原始 CSV，返回 (英文列名列表, 数据行列表)。"""
    with input_path.open(encoding="utf-8-sig", newline="") as f:
        raw_lines = [ln for ln in f.read().splitlines() if ln.strip() != ""]

    if not raw_lines:
        raise ValueError("输入文件为空。")

    # 解析表头，建立「原始中文表头位置 -> 英文列名」的对照
    header_fields = raw_lines[0].split(",")
    header_present = [header_fields[i].strip() for i in VALID_INDICES]
    if header_present != HEADER_ZH:
        # 允许顺序差异：通过中文名定位每一列
        pos_by_zh = {header_fields[i].strip(): i for i in VALID_INDICES}
        ordered_indices = [pos_by_zh[zh] for zh in HEADER_ZH]
    else:
        ordered_indices = VALID_INDICES

    english_cols = [COLUMN_MAP[zh] for zh in HEADER_ZH]

    rows: list[dict] = []
    parse_errors: list[str] = []
    for lineno, line in enumerate(raw_lines[1:], start=2):
        fields = line.split(",")
        if len(fields) < max(ordered_indices) + 1:
            parse_errors.append(f"第 {lineno} 行字段数不足，已跳过。")
            continue
        record = {}
        for zh, idx in zip(HEADER_ZH, ordered_indices):
            eng = COLUMN_MAP[zh]
            raw_val = fields[idx].strip()
            if eng == "compound_name":
                record[eng] = raw_val
            else:
                try:
                    record[eng] = float(raw_val) if "." in raw_val else int(float(raw_val))
                except ValueError:
                    parse_errors.append(f"第 {lineno} 行字段 '{zh}' 非数值: {raw_val!r}")
                    record[eng] = float("nan")
        rows.append(record)

    for err in parse_errors:
        print(f"[WARN] {err}", file=sys.stderr)
    return english_cols, rows


def write_clean(cols: list[str], rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def run(input_path: Path | None = None, output_path: Path | None = None) -> None:
    """程序化入口：解析并写出清洗数据。input_path / output_path 为空时使用默认值。"""
    input_path = input_path or (ROOT / "data" / "raw" / "MPA(1).csv")
    output_path = output_path or (ROOT / "data" / "processed" / "mpa_linkers_clean.csv")

    if not input_path.exists():
        print(f"[SKIP] 未找到原始文件: {input_path}（将直接使用已清洗数据）。")
        return

    cols, rows = parse_file(input_path)
    write_clean(cols, rows, output_path)
    print(f"[OK] 已解析 {len(rows)} 行 -> {output_path}")
    print(f"     列: {cols}")


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗 MPA linker 原始 CSV。")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "raw" / "MPA(1).csv",
        help="原始 CSV 路径（默认 data/raw/MPA(1).csv）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "mpa_linkers_clean.csv",
        help="清洗后输出路径。",
    )
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
