---
name: rf-ml-repo
description: "将任意表格数据（CSV）用随机森林构建回归/分类模型，并自动生成规范、可克隆运行的 GitHub 仓库（data/src/models/results + README/LICENSE/.gitignore/requirements）。当用户想『基于一份数据建一个随机森林模型并整理成项目』、『丢入自己的 CSV 直接出建模结果』、或提及随机森林/RF/表格建模/特征重要性/转运率预测等场景时触发。"
agent_created: true
---

# rf-ml-repo — 随机森林建模 · 一键产出规范仓库

## 能力概述

把用户提供的**任意 CSV 表格数据**转换成一套**可复现、可克隆运行**的机器学习项目：
自动训练随机森林（回归或分类），输出模型、评估指标、特征重要性与可视化图表，
并按 `data / src / models / results` 的规范目录组织，附带 README、LICENSE、.gitignore、requirements.txt。

**能做什么**
- ✅ 读取标准 CSV，或自动清洗「Excel 导出产生多余空列」的非标准分隔 CSV
- ✅ 按目标列类型**自动推断回归/分类**（可用 `--task` 覆盖）
- ✅ 训练随机森林：可复现切分 + 交叉验证 + 测试集评估
- ✅ 产出特征重要性、预测 vs 真实（回归）/混淆矩阵（分类）可视化
- ✅ 生成规范仓库结构，他人 `git clone` 后 `pip install -r requirements.txt && python run.py` 即可复现
- ✅ 支持对新样本用已训练模型推理

**不能做什么**
- ❌ 不支持图像/文本/时序等非表格数据
- ❌ 不保证模型达到特定精度（结果取决于数据质量与样本量）
- ❌ 不做超参数大规模搜索（默认已用较优参数，可在 `src/config.py` 调整）

## 输入 / 输出

| 项目 | 说明 |
|------|------|
| **输入** | 用户 CSV（含特征列 + 一个目标列；可选一个标识列） |
| **关键参数** | 目标列名 `--target`、标识列名 `--id-col`（可选）、任务类型 `--task`（可选） |
| **输出** | 规范仓库目录 + 训练好的模型 + `results/`（metrics.json、feature_importance.*、可视化图） |
| **产物位置** | 由用户指定或默认在当前工作区下新建项目目录 |

## 触发条件

- "用随机森林建个模型"、"基于这份数据做个 RF 预测"、"把 CSV 整理成建模项目"
- "丢入我自己的数据直接出结果"、"做一个表格建模的 GitHub 仓库"
- 提及 随机森林 / RandomForest / 特征重要性 / 转运率预测 / 表格回归 等

**典型输入示例**
- "用这个 CSV 训练随机森林预测转运率，整理成可运行的仓库"
- "基于我的数据做一个随机森林分类项目"
- "把这份表格数据变成规范 ML 项目，别人能直接 clone 跑"

## 执行流程总览

```
用户给 CSV + 目标列
  → 复制 skill 的 project_template 到目标项目目录
  → 放入数据（data/processed 或 data/raw）
  → 创建虚拟环境并安装依赖
  → 训练（src/train.py）：清洗(若需) → 切分 → CV → 拟合 → 评估 → 绘图 → 保存模型
  → 报告指标与特征重要性，并说明如何运行/复现
```

---

## 阶段 0：确认参数

向用户确认（若未明确）：
1. **目标列** `--target`：要预测的那一列（必填）。
2. **标识列** `--id-col`（可选）：样本名称等只作标识、不参与建模的列。
3. **任务类型**：默认按目标列自动推断（数值→回归，类别/少数值→分类）；用户可指定 `--task regression|classification`。

> 若用户未给 CSV 而只描述需求，先请其提供数据文件（路径或上传）。

## 阶段 1：搭建项目骨架

将本 skill 目录下的 `project_template/` **整体复制**到用户期望的项目路径（如 `./rf_project/` 或用户指定目录）：

```bash
# <skill_dir> 为本 skill 所在目录（如 ~/.workbuddy/skills/rf-ml-repo）
cp -r "<skill_dir>/project_template" "<目标项目目录>"
```

随后把用户的 CSV 放入：
- 标准 CSV → `<目标项目目录>/data/processed/dataset.csv`
- 非标准（多余空列）CSV → `<目标项目目录>/data/raw/原始名.csv`（运行时会自动清洗）

## 阶段 2：环境与依赖

```bash
cd <目标项目目录>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> 若环境已存在可用 Python 与依赖，可跳过虚拟环境步骤，直接 `pip install -r requirements.txt`。

## 阶段 3：训练与产出

```bash
# 一键（自动清洗 raw + 训练 + 评估）
python run.py --target <目标列> [--id-col <标识列>] [--task regression|classification]

# 或显式分步
python -m src.clean_csv --input data/raw/raw.csv --output data/processed/dataset.csv   # 仅非标准数据需要
python -m src.train --data data/processed/dataset.csv --target <目标列> [--id-col <标识列>]
```

脚本会输出：
- 交叉验证指标（训练集泛化估计）
- 测试集指标（R²/MAE/RMSE 或 Accuracy/F1）
- `models/random_forest.pkl`
- `results/metrics.json`、`feature_importance.csv/.png`、回归散点图或分类混淆矩阵

## 阶段 4：复核与新样本推理（可选）

```bash
# 全量数据复核
python -m src.evaluate --target <目标列> [--id-col <标识列>]

# 用训练好的模型对新数据推理
python -m src.predict --input examples/new_samples.csv --target <目标列> --output results/predictions.csv
```

## 阶段 5：结果报告

向用户汇报：
- 任务类型（回归/分类）与样本量、特征数
- 关键指标（交叉验证 + 测试集）
- Top 特征重要性（指出驱动预测的主要变量，结合领域知识解读）
- 仓库位置与复现命令（`git clone` 后 `pip install -r requirements.txt && python run.py --target ...`）

---

## 资源文件说明

本 skill 目录结构：

```
rf-ml-repo/
├── SKILL.md                  # 本文件
├── project_template/         # 可直接复制运行的完整 RF 项目模板
│   ├── README.md
│   ├── LICENSE / .gitignore / requirements.txt / run.py
│   ├── data/{raw,processed}/  # 占位 .gitkeep
│   ├── src/                   # config / data_loader / preprocess / clean_csv / train / evaluate / predict
│   ├── models/  results/  examples/   # 占位 .gitkeep
└── references/usage.md       # 进阶用法与参数说明
```

### 关键脚本

| 脚本 | 作用 |
|------|------|
| `src/train.py` | 训练+交叉验证+评估+绘图+保存模型/指标（核心） |
| `src/clean_csv.py` | 清洗「多余空列」的非标准分隔 CSV |
| `src/data_loader.py` | 读取数据、按 target/id_col 自动推断特征 |
| `src/preprocess.py` | 训练/测试切分 + 回归/分类 Pipeline 构造 |
| `src/evaluate.py` | 加载模型在全部数据上复核 |
| `src/predict.py` | 对新样本推理 |

### 调参提示
- 超参数集中在 `src/config.py`（`RF_PARAMS`、`TEST_SIZE`、`CV_FOLDS`、`RANDOM_STATE`）。
- 样本量很小时，交叉验证方差会偏大；此时以测试集指标 + 特征重要性解读为主。

## 注意事项
- 本项目适合**中小规模表格数据**；极大样本或需深度模型时，建议改用梯度提升/XGBoost 或神经网络。
- 仓库默认**不提交** `models/` 与 `results/`（由 `.gitignore` 排除，运行时再生），符合精简仓库惯例。
