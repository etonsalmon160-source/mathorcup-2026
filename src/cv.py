"""交叉验证与指标。

为什么这份文件很重要：本赛**每天只能提交 1 次**，排行榜几乎给不了反馈。
所有决策都必须靠本地 CV 做，所以 CV 方案必须第一天就搭对。见 docs/备战计划.md。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit,
)

from src.config import get_logger

log = get_logger("cv")


def make_folds(
    X: pd.DataFrame, y: pd.Series, cfg: dict
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """按配置切折。返回 (折列表, 行顺序)。

    「行顺序」是给时间序列策略用的：时序数据必须先按时间排好，否则模型会
    "偷看未来"。调用方拿到后要照它重排 X/y：
        X, y = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)
    """
    c = cfg.get("cv", {})
    strategy = c.get("strategy", "stratified")
    n_splits = int(c.get("n_splits", 5))
    seed = int(c.get("seed", cfg.get("seed", 42)))

    order = np.arange(len(y))
    sort_by = c.get("sort_by")
    if sort_by:
        if sort_by not in X.columns:
            raise KeyError(f"cv.sort_by={sort_by!r} 不在特征里")
        order = np.argsort(pd.to_datetime(X[sort_by]).to_numpy(), kind="stable")
        log.info("按 %s 排序（时序策略必须）", sort_by)

    group = None
    if c.get("group_col"):
        gc = c["group_col"]
        if gc not in X.columns:
            raise KeyError(f"cv.group_col={gc!r} 不在特征里")
        group = X[gc].iloc[order].to_numpy()

    y_ord = y.iloc[order].to_numpy()
    n = len(y_ord)

    if strategy == "timeseries":
        splitter = TimeSeriesSplit(n_splits=n_splits)
        folds = list(splitter.split(np.arange(n)))
    elif group is not None:
        splitter = GroupKFold(n_splits=n_splits)
        folds = list(splitter.split(np.arange(n), y_ord, groups=group))
    elif strategy == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(np.arange(n), y_ord))
    elif strategy == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(np.arange(n)))
    else:
        raise ValueError(f"不认识的 cv.strategy：{strategy}")

    log.info("CV 方案：%s，%d 折，每折验证集约 %d 条",
             strategy, len(folds), int(np.mean([len(v) for _, v in folds])))
    return folds, order


def score(y_true, y_pred, task: str, proba: np.ndarray | None = None) -> dict:
    """算一组指标。分类多给几个，因为不同赛题看重的指标不一样，
    而且**类别不平衡时准确率会骗人**。"""
    from sklearn.metrics import (
        accuracy_score, average_precision_score, f1_score, mean_absolute_error,
        precision_score, r2_score, recall_score, roc_auc_score, root_mean_squared_error,
    )

    y_true = np.asarray(y_true)
    if task == "regression":
        return {
            "rmse": float(root_mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    labels = np.unique(y_true)
    average = "binary" if len(labels) == 2 else "macro"
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    if proba is not None:
        try:
            if proba.ndim == 2 and proba.shape[1] == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
                out["pr_auc"] = float(average_precision_score(y_true, proba[:, 1]))
            elif proba.ndim == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr"))
        except ValueError as e:  # 某折里只有一类，算不了 AUC
            log.warning("这一折算不了 AUC：%s", e)
    return out


def primary_metric(task: str, cfg: dict) -> str:
    """选一个"主指标"用于比较不同实验。默认分类看 f1，回归看 rmse。"""
    return cfg.get("cv", {}).get("primary_metric") or ("rmse" if task == "regression" else "f1")


# 越小越好的指标。写论文时"提升 X"必须按方向算，别把 RMSE 降了 2 个点写成 -2。
LOWER_IS_BETTER = {"rmse", "mae", "mse", "mape", "logloss"}


def improvement(metric: str, model_value: float, baseline_value: float) -> float:
    """相对基线的提升，统一成"正数即变好"。"""
    if metric in LOWER_IS_BETTER:
        return baseline_value - model_value
    return model_value - baseline_value


def baseline_score(y_true, task: str) -> dict:
    """最笨的基线（分类猜多数类，回归猜均值）。

    论文摘要里那句"较基线提升 X 个百分点"就靠它，别省这一步。
    """
    y_true = np.asarray(y_true)
    if task == "regression":
        return score(y_true, np.full_like(y_true, y_true.mean(), dtype=float), "regression")
    vals, counts = np.unique(y_true, return_counts=True)
    return score(y_true, np.full_like(y_true, vals[counts.argmax()]), "classification")
