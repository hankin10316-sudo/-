"""train.py — 训练随机森林模型并产出评估结果与可视化。

流程：
1. 读取清洗数据；
2. 可复现训练/测试切分；
3. 5 折交叉验证评估泛化能力；
4. 在全量训练集上拟合最终模型；
5. 在测试集上评估并绘制「预测值 vs 真实值」散点图；
6. 计算特征重要性并绘图；
7. 保存模型（pkl）、指标（json）、特征重要性（csv + png）。

运行：
    python src/train.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # 无显示环境下保存图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

from . import config
from .data_loader import get_X_y, load_clean
from .preprocess import build_pipeline, split_data


def _safe_makedirs(*paths: Path):
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)


def main():
    _safe_makedirs(config.MODELS_DIR, config.RESULTS_DIR)

    # 1. 数据
    df = load_clean()
    print(f"[INFO] 载入数据: {df.shape[0]} 行, {df.shape[1]} 列")
    X, y = get_X_y(df)

    # 2. 切分
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 3. 交叉验证（在训练集上评估泛化能力）
    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    pipeline_cv = build_pipeline()
    r2_cv = cross_val_score(pipeline_cv, X_train, y_train, cv=cv, scoring="r2")
    mae_cv = -cross_val_score(pipeline_cv, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error")
    rmse_cv = -cross_val_score(pipeline_cv, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")

    print(f"[INFO] {config.CV_FOLDS} 折交叉验证 (训练集):")
    print(f"       R²  = {r2_cv.mean():.4f} ± {r2_cv.std():.4f}")
    print(f"       MAE = {mae_cv.mean():.4f} ± {mae_cv.std():.4f}")
    print(f"       RMSE= {rmse_cv.mean():.4f} ± {rmse_cv.std():.4f}")

    # 4. 最终模型（训练集全量拟合）
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 5. 测试集评估
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    print(f"[INFO] 测试集表现:")
    print(f"       R²   = {r2:.4f}")
    print(f"       MAE  = {mae:.4f}")
    print(f"       RMSE = {rmse:.4f}")

    # 6. 预测 vs 真实 散点图
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.8, edgecolors="k", color="#2c7fb8")
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=1.2, label="理想拟合 (y = x)")
    ax.set_xlabel("True transport rate")
    ax.set_ylabel("Predicted transport rate")
    ax.set_title(f"Test set: predicted vs actual (R²={r2:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.PREDICTION_SCATTER_PNG, dpi=150)
    plt.close(fig)

    # 7. 特征重要性
    importances = pipeline.named_steps["regressor"].feature_importances_
    fi = (
        pd.DataFrame({"feature": config.FEATURES, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    fi.to_csv(config.FEATURE_IMPORTANCE_CSV, index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(fi)))
    ax.barh(fi["feature"][::-1], fi["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Feature importance")
    ax.set_title("Random Forest Feature Importance")
    fig.tight_layout()
    fig.savefig(config.FEATURE_IMPORTANCE_PNG, dpi=150)
    plt.close(fig)

    # 8. 保存模型与指标
    joblib.dump(pipeline, config.MODEL_PATH)

    metrics = {
        "dataset": {
            "n_samples": int(df.shape[0]),
            "n_features": int(len(config.FEATURES)),
            "target": config.TARGET,
            "features": config.FEATURES,
        },
        "cross_validation": {
            "n_folds": config.CV_FOLDS,
            "r2_mean": float(r2_cv.mean()),
            "r2_std": float(r2_cv.std()),
            "mae_mean": float(mae_cv.mean()),
            "mae_std": float(mae_cv.std()),
            "rmse_mean": float(rmse_cv.mean()),
            "rmse_std": float(rmse_cv.std()),
        },
        "test_set": {
            "n_test": int(len(y_test)),
            "r2": float(r2),
            "mae": float(mae),
            "rmse": float(rmse),
        },
        "model": {
            "type": "RandomForestRegressor",
            "params": config.RF_PARAMS,
        },
        "feature_importance": fi.to_dict(orient="records"),
    }
    with config.METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] 模型已保存: {config.MODEL_PATH}")
    print(f"[OK] 指标已保存: {config.METRICS_PATH}")
    print(f"[OK] 特征重要性: {config.FEATURE_IMPORTANCE_CSV} / {config.FEATURE_IMPORTANCE_PNG}")
    print(f"[OK] 预测散点图: {config.PREDICTION_SCATTER_PNG}")


if __name__ == "__main__":
    main()
