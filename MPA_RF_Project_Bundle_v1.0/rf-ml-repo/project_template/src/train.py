"""train.py — 训练随机森林并产出评估结果与可视化。

通用流程（任意 CSV 均可）：
1. 读取数据（--data）；按 --target 推断任务类型与特征；
2. 可复现训练/测试切分；
3. 交叉验证评估泛化能力；
4. 在全量训练集上拟合最终模型；
5. 测试集评估 + 可视化（回归：预测 vs 真实散点；分类：混淆矩阵）；
6. 特征重要性；
7. 保存模型(pkl)、指标(json)、特征重要性(csv/png)。

运行：
    python -m src.train --data data/processed/dataset.csv --target <目标列> [--id-col <标识列>] [--task regression|classification]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from . import config
from .data_loader import get_named_features, get_X_y, load_clean
from .preprocess import build_pipeline, detect_task, split_data


def _ensure_dirs():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _plot_regression(y_test, y_pred, r2):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.8, edgecolors="k", color="#2c7fb8")
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="ideal (y = x)")
    ax.set_xlabel("True value")
    ax.set_ylabel("Predicted value")
    ax.set_title(f"Test set: predicted vs actual (R2={r2:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.PREDICTED_VS_ACTUAL_PNG, dpi=150)
    plt.close(fig)


def _plot_classification(cm, labels):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(config.CONFUSION_PNG, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="训练随机森林（回归/分类通用）。")
    parser.add_argument("--data", type=Path, default=None, help="清洗后的 CSV 路径")
    parser.add_argument("--target", type=str, required=True, help="目标列名（要预测的列）")
    parser.add_argument("--id-col", type=str, default=None, help="标识列名（如化合物名称，不作为特征）")
    parser.add_argument("--task", type=str, default="auto",
                        choices=["auto", "regression", "classification"], help="任务类型（默认自动推断）")
    parser.add_argument("--cv-folds", type=int, default=config.CV_FOLDS)
    args = parser.parse_args()

    _ensure_dirs()
    df = load_clean(args.data)
    print(f"[INFO] loaded: {df.shape[0]} rows x {df.shape[1]} cols")

    task = args.task if args.task != "auto" else detect_task(df[args.target])
    print(f"[INFO] task type: {task}")

    X, y = get_X_y(df, args.target, args.id_col)
    X_train, X_test, y_train, y_test = split_data(X, y, task)

    # 交叉验证
    if task == "classification":
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
        scoring = ("accuracy", "f1_macro")
        cv_acc = cross_val_score(build_pipeline(task), X_train, y_train, cv=cv, scoring="accuracy")
        cv_f1 = cross_val_score(build_pipeline(task), X_train, y_train, cv=cv, scoring="f1_macro")
        print(f"[INFO] {args.cv_folds}-fold CV (train): ACC={cv_acc.mean():.4f}±{cv_acc.std():.4f} "
              f"F1={cv_f1.mean():.4f}±{cv_f1.std():.4f}")
    else:
        cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
        cv_r2 = cross_val_score(build_pipeline(task), X_train, y_train, cv=cv, scoring="r2")
        cv_mae = -cross_val_score(build_pipeline(task), X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
        cv_rmse = -cross_val_score(build_pipeline(task), X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
        print(f"[INFO] {args.cv_folds}-fold CV (train): R2={cv_r2.mean():.4f}±{cv_r2.std():.4f} "
              f"MAE={cv_mae.mean():.4f}±{cv_mae.std():.4f} RMSE={cv_rmse.mean():.4f}±{cv_rmse.std():.4f}")

    # 最终模型
    pipeline = build_pipeline(task)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics: dict = {"dataset": {"n_samples": int(df.shape[0]),
                                 "n_features": int(X.shape[1]),
                                 "target": args.target,
                                 "id_col": args.id_col,
                                 "features": list(X.columns),
                                 "task": task},
                      "model": {"type": "RandomForest", "params": config.RF_PARAMS}}

    if task == "classification":
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"[INFO] test: ACC={acc:.4f} F1_macro={f1:.4f}")
        from sklearn.metrics import confusion_matrix
        labels = sorted(y.unique().tolist())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        _plot_classification(cm, [str(l) for l in labels])
        metrics["cross_validation"] = {"accuracy_mean": float(cv_acc.mean()),
                                       "accuracy_std": float(cv_acc.std()),
                                       "f1_macro_mean": float(cv_f1.mean()),
                                       "f1_macro_std": float(cv_f1.std())}
        metrics["test_set"] = {"accuracy": float(acc), "f1_macro": float(f1)}
    else:
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        print(f"[INFO] test: R2={r2:.4f} MAE={mae:.4f} RMSE={rmse:.4f}")
        _plot_regression(y_test, y_pred, r2)
        metrics["cross_validation"] = {"r2_mean": float(cv_r2.mean()), "r2_std": float(cv_r2.std()),
                                       "mae_mean": float(cv_mae.mean()), "mae_std": float(cv_mae.std()),
                                       "rmse_mean": float(cv_rmse.mean()), "rmse_std": float(cv_rmse.std())}
        metrics["test_set"] = {"r2": float(r2), "mae": float(mae), "rmse": float(rmse)}

    # 特征重要性
    importances = pipeline.named_steps["model"].feature_importances_
    fi = (pd.DataFrame({"feature": X.columns, "importance": importances})
          .sort_values("importance", ascending=False).reset_index(drop=True))
    fi.to_csv(config.FEATURE_IMPORTANCE_CSV, index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(fi))))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(fi)))
    ax.barh(fi["feature"][::-1], fi["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Feature importance")
    ax.set_title("Random Forest Feature Importance")
    fig.tight_layout()
    fig.savefig(config.FEATURE_IMPORTANCE_PNG, dpi=150)
    plt.close(fig)

    joblib.dump(pipeline, config.MODEL_PATH)
    metrics["feature_importance"] = fi.to_dict(orient="records")
    with config.METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] model -> {config.MODEL_PATH}")
    print(f"[OK] metrics -> {config.METRICS_PATH}")
    print(f"[OK] feature importance -> {config.FEATURE_IMPORTANCE_CSV} / {config.FEATURE_IMPORTANCE_PNG}")


if __name__ == "__main__":
    main()
