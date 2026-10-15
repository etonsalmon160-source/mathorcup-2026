"""训练入口。

    python -m src.train --config src/configs/baseline.yaml

一次跑完：交叉验证 → 报"均值 ± 标准差" → 全量重训 → 存模型和产物 → 记台账。
结果都在 results/<配置名>/ 下。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import (
    PATHS, SEED, config_hash, get_logger, load_config, record_run, run_dir,
    set_global_seed,
)
from src.cv import (
    LOWER_IS_BETTER, baseline_score, improvement, make_folds, primary_metric, score,
)
from src.data import load_table
from src.features import prepare_features
from src.models import (
    MODEL_SUFFIX, detect_task, feature_importance, fit_predict, save_model, train_full,
)

log = get_logger("train")


def aggregate(per_seed: list[dict]) -> dict:
    """跨种子汇总：报均值，也报标准差（证明结果稳，论文里比单个数可信）。"""
    keys = per_seed[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seed], dtype=float)
        out[k] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="训练 + 交叉验证")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = cfg.get("seeds") or [cfg.get("seed", SEED)]
    set_global_seed(seeds[0])
    out_dir = run_dir(cfg)

    log.info("=" * 60)
    log.info("实验 %s | 模型 %s | 种子 %s | 配置指纹 %s",
             cfg["name"], cfg.get("model"), seeds, config_hash(cfg))

    # ---- 数据 ----
    train = load_table(cfg["data"]["train"])
    test = load_table(cfg["data"]["test"]) if cfg["data"].get("test") else None
    X, X_test, names = prepare_features(train, test, cfg)
    cfg["_n_features"] = len(names)

    y_raw = train[cfg["data"]["target"]].reset_index(drop=True)
    task = detect_task(y_raw)
    log.info("任务类型：%s（目标列 %s，%d 个唯一值）",
             task, cfg["data"]["target"], y_raw.nunique())

    # 目标列编码成整数，同时记下映射，预测时再翻回去
    classes: list | None = None
    if task == "classification":
        classes = sorted(y_raw.dropna().unique().tolist(), key=str)
        mapping = {c: i for i, c in enumerate(classes)}
        y = y_raw.map(mapping).astype(int)
        log.info("类别映射：%s", mapping)
    else:
        y = y_raw.astype(float)

    params = cfg.get("params", {}) or {}
    model_name = cfg.get("model", "lightgbm")

    # ---- 逐种子交叉验证 ----
    oof_all, metric_rows = [], []
    for seed in seeds:
        fold_cfg = {**cfg, "cv": {**cfg.get("cv", {}), "seed": seed}}
        folds, order = make_folds(X, y, fold_cfg)
        Xs, ys = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)

        oof_pred = np.full(len(ys), np.nan)
        oof_proba = None

        for i, (tr, va) in enumerate(folds):
            pred, proba = fit_predict(
                model_name, params, task, seed,
                Xs.iloc[tr], ys.iloc[tr], Xs.iloc[va],
            )
            oof_pred[va] = pred if task == "regression" else pred
            if proba is not None:
                if oof_proba is None:
                    oof_proba = np.full((len(ys), proba.shape[1]), np.nan)
                oof_proba[va] = proba
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
    log.info("【主指标 %s】CV %.5f ± %.5f   |   基线 %.5f   |   优于基线 %.5f%s",
             pm, agg[pm], agg.get(f"{pm}_std", 0.0), ref.get(pm, float("nan")), gain,
             "（越小越好，差值为基线减模型）" if pm in LOWER_IS_BETTER else "")
    log.info("-" * 60)

    # ---- 全量重训 + 存模型 ----
    model = train_full(model_name, params, task, seeds[0], X, y)
    model_path = PATHS["models"] / f"{cfg['name']}{MODEL_SUFFIX.get(model_name, '.txt')}"
    save_model(model, model_name, model_path)

    imp = feature_importance(model, model_name, names)
    imp.to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    log.info("Top 10 重要特征：\n%s", imp.head(10).to_string(index=False))

    # ---- 落盘 ----
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {"name": cfg["name"], "model": model_name, "task": task,
             "primary_metric": pm, "cv": agg, "baseline": ref, "improvement": gain,
             "per_seed": metric_rows, "seeds": seeds, "n_features": len(names),
             "classes": classes},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")},
                   open(out_dir / "config_used.yaml", "w", encoding="utf-8"),
                   allow_unicode=True, sort_keys=False)

    # OOF 预测：多个种子取平均，画 ROC / 误差图用它
    order0, pred0, proba0, y0 = oof_all[0]
    oof = pd.DataFrame({"y_true": y0.to_numpy(), "oof_pred": pred0})
    if proba0 is not None:
        n_cls = proba0.shape[1]
        for s in range(1, len(oof_all)):
            proba0 = proba0 + oof_all[s][2]
        proba0 = proba0 / len(oof_all)
        for j in range(n_cls):
            oof[f"proba_{j}"] = proba0[:, j]
    oof.to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")

    PATHS["processed"].mkdir(parents=True, exist_ok=True)
    X.to_parquet(PATHS["processed"] / f"{cfg['name']}_train.parquet", index=False)
    if X_test is not None:
        X_test.to_parquet(PATHS["processed"] / f"{cfg['name']}_test.parquet", index=False)

    record_run(cfg, task, pm, agg[pm], agg.get(f"{pm}_std", 0.0),
               ref.get(pm, float("nan")), gain)
    log.info("产物目录 -> %s", out_dir)
    print(f"\n完成。主指标 {pm} = {agg[pm]:.5f}（基线 {ref.get(pm, float('nan')):.5f}，"
          f"优于基线 {gain:.5f}）")


if __name__ == "__main__":
    main()
