# rf-ml-repo 进阶用法

## 1. 参数速查

| 参数 | 脚本 | 说明 |
|------|------|------|
| `--target` | train / evaluate / predict | 目标列名（必填） |
| `--id-col` | train / evaluate | 标识列（如名称），不参与建模 |
| `--task` | train / run | `auto`(默认) / `regression` / `classification` |
| `--data` | train / evaluate | 清洗后的 CSV 路径 |
| `--cv-folds` | train | 交叉验证折数，默认 5 |
| `--input/--output` | predict / clean_csv | 输入/输出文件 |

## 2. 非标准 CSV 清洗原理

某些 Excel 导出文件每行 36 个字段里只有 18 个有效，有效列之间夹着空字段。
`src/clean_csv.py` 用**表头中非空单元格的位置**定位有效列，再据此抽取所有数据行。
对标准 CSV 也安全（有效列即全部列）。

## 3. 任务类型推断规则

- 目标列为字符串/类别 → 分类
- 目标列为数值且唯一值 ≤20 且全为整数 → 分类
- 其余数值 → 回归

可用 `--task` 显式覆盖。

## 4. 自定义超参数

编辑 `src/config.py`：

```python
RF_PARAMS = {"n_estimators": 500, "max_depth": 12, "max_features": "sqrt", ...}
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42
```

## 5. 复现与共享

```bash
# 他人克隆后
pip install -r requirements.txt
python run.py --target <目标列>
```

`models/` 与 `results/` 默认不入库，运行即生成，保证仓库精简且可复现。
