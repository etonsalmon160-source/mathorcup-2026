"""表格深度模型：RealMLP，作为**异构专家**与其他模型同台比较。

    make dl PY=/home/eto/miniconda3/envs/mathorcup-gpu/bin/python
    python -m src.dl --config src/configs/baseline.yaml --device cpu --epochs 8   # 冒烟

## 为什么是 RealMLP

它不是"又一个神经网络"，而是 2024 NeurIPS《Better by Default》里
**在 118 个表格数据集上调出来、90 个不相交数据集上验证过**的默认配置族。
我们给 LightGBM / XGBoost / CatBoost 用的那套 meta-tuned 默认值，
和 RealMLP 出自**同一篇论文的同一套实验**——所以拿它当第四个专家，
比较的是"方法差异"而不是"谁调参调得多"。论文里这句话很好写。

## 它在这个项目里的定位

**不是主力。** 表格数据上梯度提升树仍然更强，这条线只做两件事：

1. 给 G5（融合门）提供一个**归纳偏置完全不同**的专家；
2. 万一赛题是"样本少、特征平滑"的类型（生信里常见），DL 可能反超。

不赢就不进融合——`src/fuse.py` 的规矩不变。

## 和 src/train.py 的关系

复用同一套特征工程、同一套折、同一个 scoring，所以数字**直接可比**。
不重复实现，只新增"怎么训这个模型"这一段。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import (
    PATHS, SEED, config_hash, get_logger, load_config, record_run, run_dir,
    set_global_seed,
)
from src.cv import baseline_score, improvement, make_folds, primary_metric, score
from src.data import load_table
from src.features import prepare_features
from src.models import detect_task
from src.train import aggregate

log = get_logger("dl")

MISSING_CAT = "__缺失__"


def _ready_for_realmlp(
    parts: list[pd.DataFrame],
) -> tuple[list[pd.DataFrame], list[str]]:
    """把特征整理成 RealMLP 能吃的样子。

    RealMLP 会自己做标准化/分箱等预处理，但**不补缺失值**（官方文档明说）。
    所以这里只做两件事：
      1. 类别列 → 字符串，缺失填一个显式类别 `__缺失__`（缺失本身可能就是信号）
      2. 数值列 → 用**训练集**中位数填补（绝不能用测试集中位数，那是泄漏）
    """
    tr = parts[0]
    cat_cols = [c for c in tr.columns
                if str(tr[c].dtype) in ("category", "object", "bool")]
    num_cols = [c for c in tr.columns if c not in cat_cols]

    medians = tr[num_cols].median(numeric_only=True) if num_cols else pd.Series(dtype=float)

    out = []
    for df in parts:
        d = df.copy()
        for c in num_cols:
            if d[c].isna().any():
                d[c] = d[c].fillna(medians.get(c, 0.0))
        for c in cat_cols:
            d[c] = d[c].astype(object).where(d[c].notna(), MISSING_CAT).astype(str)
        out.append(d)
    return out, cat_cols


def _make_model(task: str, device: str, seed: int, epochs: int | None):
    try:
        from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor
    except ImportError as e:                                  # pragma: no cover
        raise SystemExit(
            "没装 pytabkit。先跑：bash scripts/setup_gpu.sh\n"
            "（它会建一个独立的 mathorcup-gpu 环境，不碰现有环境）"
        ) from e

    cls = RealMLP_TD_Classifier if task == "classification" else RealMLP_TD_Regressor
    kw = dict(device=device, random_state=seed, n_threads=4, verbosity=0)
    if epochs:
        kw["n_epochs"] = epochs
    return cls(**kw)


def main() -> None:
    ap = argparse.ArgumentParser(description="RealMLP：表格深度模型")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="cpu | cuda | 不填=有卡就用卡")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖默认轮数（冒烟用小的）")
    args = ap.parse_args()

    try:
        import torch
    except ImportError as e:                                  # pragma: no cover
        raise SystemExit("没装 torch。先跑 bash scripts/setup_gpu.sh") from e

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config)
    # 目录名加后缀，否则会把同名的树模型结果覆盖掉
    cfg["name"] = f"{cfg['name']}_realmlp"
    cfg["model"] = "realmlp"
    seeds = cfg.get("seeds") or [cfg.get("seed", SEED)]
    set_global_seed(seeds[0])
    out_dir = run_dir(cfg)

    log.info("=" * 60)
    log.info("实验 %s | RealMLP | 设备 %s | 种子 %s | 配置指纹 %s",
             cfg["name"], device, seeds, config_hash(cfg))

    train = load_table(cfg["data"]["train"])
    test = load_table(cfg["data"]["test"]) if cfg["data"].get("test") else None
    X, X_test, names = prepare_features(train, test, cfg)
    cfg["_n_features"] = len(names)

    y_raw = train[cfg["data"]["target"]].reset_index(drop=True)
    task = detect_task(y_raw)
    classes = None
    if task == "classification":
        classes = sorted(y_raw.dropna().unique().tolist(), key=str)
        y = y_raw.map({c: i for i, c in enumerate(classes)}).astype(int)
        log.info("类别映射：%s", {c: i for i, c in enumerate(classes)})
    else:
        y = y_raw.astype(float)

    # ---- 逐种子交叉验证（同一套折，数字才可比）----
    oof_all, metric_rows = [], []
    for seed in seeds:
        fold_cfg = {**cfg, "cv": {**cfg.get("cv", {}), "seed": seed}}
        folds, order = make_folds(X, y, fold_cfg)
        Xs, ys = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)

        oof_pred = np.full(len(ys), np.nan)
        oof_proba = None

        for i, (tr, va) in enumerate(folds):
            parts, cat_cols = _ready_for_realmlp(
                [Xs.iloc[tr], Xs.iloc[va]])
            X_tr, X_va = parts
            y_tr, y_va = ys.iloc[tr], ys.iloc[va]

            model = _make_model(task, device, seed, args.epochs)
            model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va,
                      cat_col_names=cat_cols if cat_cols else None)

            if task == "classification":
                proba = model.predict_proba(X_va)
                oof_pred[va] = np.asarray(proba).argmax(axis=1)
                if oof_proba is None:
                    oof_proba = np.full((len(ys), proba.shape[1]), np.nan)
                oof_proba[va] = proba
            else:
                oof_pred[va] = np.asarray(model.predict(X_va)).ravel()
            log.info("  种子 %d 第 %d/%d 折完成", seed, i + 1, len(folds))

        m = score(ys, oof_pred, task, oof_proba)
        log.info("种子 %d 结果：%s", seed, {k: round(v, 5) for k, v in m.items()})
        metric_rows.append(m)
        oof_all.append((order, oof_pred, oof_proba, ys))

    agg = aggregate(metric_rows)
    ref = baseline_score(y, task)
    pm = primary_metric(task, cfg)
    gain = improvement(pm, agg[pm], ref.get(pm, float("nan")))

    log.info("-" * 60)
    log.info("【主指标 %s】CV %.5f ± %.5f   |   基线 %.5f   |   优于基线 %.5f",
             pm, agg[pm], agg.get(f"{pm}_std", 0.0), ref.get(pm, float("nan")), gain)
    log.info("-" * 60)

    # ---- 全量重训 + 出测试集预测 ----
    if X_test is not None:
        parts, cat_cols = _ready_for_realmlp([X, X_test])
        model = _make_model(task, device, seeds[0], args.epochs)
        model.fit(parts[0], y, cat_col_names=cat_cols if cat_cols else None)

        if task == "classification":
            proba = np.asarray(model.predict_proba(parts[1]))
            pred = proba.argmax(axis=1)
            sub_pred = [classes[i] for i in pred]
            sub = pd.DataFrame({cfg.get("submission", {}).get("pred_col", "prediction"): sub_pred})
            for j in range(proba.shape[1]):
                sub[f"proba_{classes[j]}"] = proba[:, j]
        else:
            sub_pred = np.asarray(model.predict(parts[1])).ravel()
            sub = pd.DataFrame({cfg.get("submission", {}).get("pred_col", "prediction"): sub_pred})

        id_col = (cfg.get("submission", {}) or {}).get("id_col") or cfg["data"].get("id_col")
        if id_col and id_col in test.columns:
            sub.insert(0, id_col, test[id_col].to_numpy())
        sub.to_csv(out_dir / "submission.csv", index=False, encoding="utf-8-sig")
        log.info("提交文件 -> %s", out_dir / "submission.csv")

    # ---- 落盘（格式与 src/train.py 一致，方便 fuse 直接读）----
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {"name": cfg["name"], "model": "realmlp", "task": task,
             "primary_metric": pm, "cv": agg, "baseline": ref, "improvement": gain,
             "per_seed": metric_rows, "seeds": seeds, "n_features": len(names),
             "classes": classes, "device": device},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    order0, pred0, proba0, y0 = oof_all[0]
    oof = pd.DataFrame({"y_true": y0.to_numpy(), "oof_pred": pred0})
    if proba0 is not None:
        for s in range(1, len(oof_all)):
            proba0 = proba0 + oof_all[s][2]
        proba0 = proba0 / len(oof_all)
        for j in range(proba0.shape[1]):
            oof[f"proba_{j}"] = proba0[:, j]
    oof.to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    record_run(cfg, task, pm, agg[pm], agg.get(f"{pm}_std", 0.0),
               ref.get(pm, float("nan")), gain)
    print(f"\n完成。RealMLP 主指标 {pm} = {agg[pm]:.5f}"
          f"（基线 {ref.get(pm, float('nan')):.5f}，优于基线 {gain:.5f}）")
    print(f"产物 -> {out_dir}")


if __name__ == "__main__":
    main()
