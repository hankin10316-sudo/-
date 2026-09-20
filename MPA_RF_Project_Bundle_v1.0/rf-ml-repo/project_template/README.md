# 随机森林建模项目（通用模板）

> 由 **rf-ml-repo** skill 生成：给定任意表格数据（CSV），用随机森林预测指定目标列，并自动产出规范、可克隆运行的 GitHub 仓库。

## 1. 项目简介

- **任务**：监督学习 · 回归或分类（按目标列类型自动推断，可用 `--task` 覆盖）
- **模型**：`RandomForestRegressor` / `RandomForestClassifier`（sklearn）
- **输入**：一张 CSV，指定**目标列**与（可选）**标识列**
- **输出**：训练好的模型、评估指标（交叉验证 + 测试集）、特征重要性、可视化图表、规范目录

## 2. 目录结构

```
.
├── README.md / LICENSE / .gitignore / requirements.txt / run.py
├── data/
│   ├── raw/              # 原始 CSV（可选，运行时会自动清洗）
│   └── processed/        # 清洗后的规范数据（dataset.csv）
├── src/                  # 流水线源码
│   ├── config.py         # 路径与超参数
│   ├── data_loader.py    # 读取与特征推断
│   ├── preprocess.py     # 切分 + Pipeline
│   ├── clean_csv.py      # 非标准分隔 CSV 清洗
│   ├── train.py          # 训练 + 评估 + 绘图
│   ├── evaluate.py       # 全量复核
│   └── predict.py        # 新样本推理
├── models/               # 训练好的模型（运行时生成）
├── results/              # 指标 / 特征重要性 / 图表（运行时生成）
└── examples/            # 推理示例输入
```

## 3. 数据说明

- 把你的 CSV 放入 `data/processed/`（命名为 `dataset.csv`），或放入 `data/raw/`（运行时会自动清洗多余空列）。
- 确定**目标列**（要预测的列）与可选的**标识列**（如样本名称，仅作标识、不参与建模）。
- 其余所有列自动作为特征。

## 4. 环境配置

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. 运行

```bash
# 一键：清洗(若有原始) + 训练 + 评估
python run.py --target <目标列> [--id-col <标识列>] [--task regression|classification]

# 或分步
python -m src.clean_csv --input data/raw/raw.csv --output data/processed/dataset.csv
python -m src.train --data data/processed/dataset.csv --target <目标列> [--id-col <标识列>]
python -m src.predict --input examples/new_samples.csv --target <目标列> --output results/predictions.csv
```

## 6. 结果解读

- `results/metrics.json`：交叉验证与测试集指标（回归：R²/MAE/RMSE；分类：Accuracy/F1）。
- `results/feature_importance.png` / `.csv`：各特征重要性。
- `results/predicted_vs_actual.png`（回归）或 `confusion_matrix.png`（分类）：模型表现可视化。

> 小样本提示：样本量较小时交叉验证方差较大，测试集指标更能反映真实泛化；建议结合特征重要性与领域知识解读。

## 7. 许可证

MIT License。
