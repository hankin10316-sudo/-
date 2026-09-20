# MPA Linker 转运率预测（随机森林）

> 基于 MPA（麦考酚酸）–甘油三酯（TG）偶联前药的 **linker（连接子）信息**，使用**随机森林回归**预测其 **转运率（transport rate）** 的机器学习项目。仓库结构规范、可直接克隆运行。

---

## 1. 项目简介

本项目的目标是：给定一类 MPA–TG 偶联物的分子/连接子描述符（亲脂性、各类自毁/连接基团指示、链长、支链修饰数、分子量等），训练一个随机森林模型，预测该化合物的**转运率**。

- **任务类型**：监督学习 · 回归（连续值预测）
- **模型**：`RandomForestRegressor`（sklearn）
- **样本量**：86 个化合物
- **特征数**：16
- **目标变量**：`transport_rate`（转运率）

数据来源于实验测定的 MPA 前药库。每条记录对应一种 linker 设计（名称 + 结构描述符 + 实测转运率）。

---

## 2. 目录结构

```
mpa-linker-transport-rf/
├── README.md                      # 本说明文档
├── LICENSE                        # MIT 开源许可证
├── .gitignore                     # 忽略缓存、虚拟环境、训练产物等
├── requirements.txt               # Python 依赖（已锁定版本）
├── run.py                         # 一键运行：解析原始数据 → 训练 → 评估
├── data/
│   ├── raw/                       # 原始（未清洗）CSV，默认不入库
│   │   └── .gitkeep
│   └── processed/
│       └── mpa_linkers_clean.csv  # 清洗后的规范化数据（已入库，可直接训练）
├── src/
│   ├── __init__.py
│   ├── config.py                  # 统一路径、列定义、超参数
│   ├── parse_raw.py               # 解析非标准分隔的原始 CSV → 清洗数据
│   ├── data_loader.py             # 读取数据、拆分 X/y
│   ├── preprocess.py              # 训练/测试切分与 sklearn Pipeline
│   ├── train.py                   # 训练、交叉验证、评估、绘图、保存模型
│   ├── evaluate.py                # 加载模型在全部数据上复核
│   └── predict.py                 # 对新样本推理转运率
├── models/
│   └── random_forest_transport.pkl  # 训练好的模型（运行时生成，git 忽略）
├── results/
│   ├── metrics.json               # 评估指标（交叉验证 + 测试集）
│   ├── feature_importance.csv     # 特征重要性
│   ├── feature_importance.png     # 特征重要性柱状图
│   └── predicted_vs_actual.png    # 测试集预测 vs 真实散点图
└── examples/
    └── new_samples.csv            # 推理示例输入
```

---

## 3. 数据说明

### 3.1 数据来源与清洗

原始文件 `MPA(1).csv` 由 Excel 导出，存在**大量成对出现的空列**（每行 36 个字段中只有 18 个有效）。`src/parse_raw.py` 按固定位置抽取有效字段，并将中文表头映射为英文列名，输出规范化数据 `data/processed/mpa_linkers_clean.csv`。

> 原始文件不含任何个人隐私或敏感信息（仅为化合物结构与实验数值），且因其非标准格式不宜直接入库，故 `data/raw/` 仅保留占位文件；规范化后的数据已入库，克隆后可立即训练。

### 3.2 数据字典

| 英文列名 | 中文含义 | 类型 | 说明 |
|---|---|---|---|
| `compound_name` | 化合物名称 | 标识 | 仅作标识，**不作为模型特征** |
| `logp_2mg` | 2-MG LogP | 数值 | 亲脂性描述符 |
| `asi` / `masi` / `phb` / `dmphb` / `cphb` / `cdmphb` / `fsi` / `tml` / `ce4` / `casi` | 各基团指示 | 0/1 | 是否含对应连接/自毁基团 |
| `chain_length` | 链长 | 整数 | 连接子碳原子数 |
| `self_immolative` | 自毁基团 | 0/1 | 是否含自毁基团 |
| `tg_linkage` | 连接键（与 TG） | 0/1 | 与甘油三酯的连接键类型 |
| `branch_modifications` | 支链修饰个数 | 整数 | 支链修饰数量 |
| `mw` | 分子量 | 数值 | MW |
| `transport_rate` | 转运率 | 数值 | **回归目标（y）** |

**模型使用的特征（16 个）**：`logp_2mg, asi, masi, phb, dmphb, cphb, cdmphb, fsi, tml, ce4, casi, chain_length, self_immolative, tg_linkage, branch_modifications, mw`

---

## 4. 环境配置

要求 Python ≥ 3.9（开发环境为 3.13）。

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd mpa-linker-transport-rf

# 2. 创建并激活虚拟环境（推荐）
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

依赖：numpy, pandas, scikit-learn, matplotlib, joblib（版本见 `requirements.txt`）。

---

## 5. 运行步骤

### 5.1 一键运行（解析 + 训练 + 评估）

```bash
python run.py
```

- 若 `data/raw/` 下存在原始 CSV，会先清洗为 `data/processed/mpa_linkers_clean.csv`；
- 否则直接使用已入库的清洗数据；
- 随后训练随机森林，并将模型、指标、图表写入 `models/` 与 `results/`。

### 5.2 分步运行

```bash
# 仅清洗原始数据（可选）
python -m src.parse_raw --input data/raw/MPA(1).csv --output data/processed/mpa_linkers_clean.csv

# 训练 + 交叉验证 + 评估 + 保存模型与图表
python -m src.train

# 在全部样本上复核模型
python -m src.evaluate

# 对新样本推理（输入需含全部 16 个特征列）
python -m src.predict --input examples/new_samples.csv --output results/predictions.csv
```

> 提示：本项目以 **包** 形式组织，`python -m src.<module>` 与 `python run.py` 均可正确解析路径。

---

## 6. 结果解读

### 6.1 评估指标

| 评估口径 | R² | MAE | RMSE |
|---|---|---|---|
| 5 折交叉验证（训练集） | 0.495 ± 0.418 | 6.03 ± 1.50 | 8.23 ± 1.95 |
| 测试集（约 17 个样本） | 0.285 | 9.70 | 15.19 |
| 全量数据（evaluate 复核） | 0.749 | 3.66 | 7.43 |

**如何解读**：

- **交叉验证 R² 仅约 0.50 且方差很大（±0.42）**：说明在小样本（n=86）下模型泛化能力有限、对数据划分较敏感，属于典型的**小数据过拟合风险**信号。
- **测试集 R²≈0.28**：在新化合物上的预测能力偏弱，提示 linker 描述符对转运率仅有中等解释力，转运率还受未纳入的特征（如立体构型、代谢稳定性）影响。
- **全量拟合 R²≈0.75**：模型能捕捉数据中的主要趋势，但高估了真实泛化能力，不应作为对外汇报的“性能”。

### 6.2 特征重要性

| 排名 | 特征 | 重要性 |
|---|---|---|
| 1 | `logp_2mg`（亲脂性） | 0.339 |
| 2 | `mw`（分子量） | 0.252 |
| 3 | `chain_length`（链长） | 0.110 |
| 4 | `masi`（MASI 基团） | 0.106 |
| 5 | `branch_modifications`（支链修饰数） | 0.099 |
| 6 | `self_immolative`（自毁基团） | 0.025 |
| 7 | `tg_linkage`（连接键） | 0.014 |
| … | 其余基团指示 | < 0.012 |

**结论**：转运率主要由**亲脂性（LogP）**与**分子量（MW）**驱动，其次是链长、MASI 基团与支链修饰；多数单一基团指示变量贡献很小。这与“亲脂性/分子尺寸影响膜转运与药代”的化学直觉一致，也提示后续可重点围绕 LogP、MW、链长做更精细的 linker 设计。

### 6.3 可视化

- `results/predicted_vs_actual.png`：测试集预测值 vs 真实值散点（越贴近对角线越好）。
- `results/feature_importance.png`：各特征重要性柱状图。

---

## 7. 局限与改进方向

1. **样本量偏小（86）**：建议扩充化合物库，或采用留一法（LOOCV）/贝叶斯优化做更稳健的验证。
2. **未做特征工程**：可加入 LogP²、LogP×MW 等交互项，或对基团指示做分组编码。
3. **目标含噪声**：转运率实测存在实验误差，可尝试对异常点做稳健回归（如 `RandomForestRegressor` + 离群裁剪）。
4. **可解释性**：可补充 SHAP 值分析，量化单样本级别的贡献。
5. **模型对比**：可对比 Gradient Boosting、SVM、线性回归等基线，确认随机森林是否最优。

---

## 8. 许可证

本项目基于 [MIT License](./LICENSE) 开源。

---

## 9. 版本管理与运行环境（v1.0）

本项目已进行 **Git 版本管理**，当前发布版本为 **v1.0**（git tag `v1.0.0`）。

### 9.1 运行环境

- **Python**：≥ 3.9（开发验证环境 3.13）
- **依赖**：见根目录 `requirements.txt`，**已锁定版本**，与 v1.0 完全匹配
- **复现环境**：
  ```bash
  python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  ```

### 9.2 两种运行方式

**方式 A — Jupyter Notebook / Lab（推荐交互式）**

在**仓库根目录**启动 Jupyter，打开 Notebook：

```bash
jupyter notebook notebooks/MPA_RF_Training.ipynb
# 或
jupyter lab notebooks/MPA_RF_Training.ipynb
```

Notebook 会依次执行：清洗 → 训练 → 评估 → 特征重要性 → 可视化，结果写入 `results/`，与 CLI 方式一致。
（Notebook 复用 `src/` 流水线，无需单独配置。）

**方式 B — 命令行（CLI）**

```bash
python run.py --target transport_rate --id-col compound_name
# 或分步
python -m src.train --data data/processed/mpa_linkers_clean.csv --target transport_rate --id-col compound_name
```

> 说明：本项目**不强制**依赖 Jupyter；Notebook 仅为可选的交互式入口，核心逻辑在 `src/` 中，两种方式的产物完全相同。

### 9.3 切换到 v1.0 版本

```bash
git fetch --tags
git checkout v1.0.0        # 或 git checkout tags/v1.0.0
```

切回最新开发分支：

```bash
git checkout main
```

> 提示：`v1.0.0` 标签对应的文件集合（含 `notebooks/MPA_RF_Training.ipynb`、`requirements.txt` 锁定版本等）即为本仓库 v1.0 的确定快照；`git checkout v1.0.0` 后配合 `pip install -r requirements.txt` 即可完全复现该版本的运行环境。
