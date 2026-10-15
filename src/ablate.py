"""消融实验：证明你加的每一块特征都真的有用。

    python -m src.ablate --config src/configs/baseline.yaml

做法：把特征分成几组，**每次砍掉一组**重跑同一套交叉验证，看主指标掉多少。

- 掉了 → 这组有用，写进论文（这是"论证有说服力"的直接证据）。
- **没掉甚至变好 → 这组是噪音，必须下线**（G4 护栏会拦）。

这一步是评分第 4 轴最实在的加分项：别人说"我们加了 XX 特征效果更好"，
你们说"我们砍掉 XX 特征，主指标从 A 掉到 B，所以它贡献了 A−B"。后者才叫论证。

产出 `results/<名>/ablation.csv`。
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import PATHS, SEED, get_logger, load_config, run_dir, set_global_seed
from src.cv import improvement, make_folds, primary_metric, score
from src.models import detect_task, fit_predict

log = get_logger("ablate")

# 特征分组规则。**按前缀/后缀自动认**，认不出的归到「基础字段」。
# 开赛加了领域特征之后，在这里加一条规则，消融表会自动带上它。
SURNAME_TO_GROUP = {
    "__年": "日期周期", "__月": "日期周期", "__日": "日期周期", "__星期": "日期周期",
    "__季度": "日期周期", "__年内第几天": "日期周期", "__第几周": "日期周期",
    "__是否周末": "日期周期", "__是否月初": "日期周期", "__是否月末": "日期周期",
    "__时间戳": "日期周期",
    "__频次": "频次编码",
}
PREFIX_TO_GROUP = {
    "行统计__": "行内统计",
    "缺失__": "缺失模式",
}


def assign_group(col: str, extra_rules: list[dict] | None = None) -> str:
    for rule in extra_rules or []:
        kw = rule.get("包含")
        if kw and kw in col:
            return rule["组名"]
    for pre, grp in PREFIX_TO_GROUP.items():
        if col.startswith(pre):
            return grp
    for suf, grp in SURNAME_TO_GROUP.items():
        if col.endswith(suf):
            return grp
    return "基础字段"


def run_cv(model_name, params, task, seed, X, y, cv_cfg: dict) -> dict:
    """按训练时**同一套**折跑交叉验证 —— 折不一样就没法跟参照比。"""
    folds, order = make_folds(X, y, {"cv": {**cv_cfg, "seed": seed}})
    Xs, ys = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)
    oof_pred = np.full(len(ys), np.nan)
    oof_proba = None
    for tr, va in folds:
        pred, proba = fit_predict(model_name, params, task, seed, Xs.iloc[tr], ys.iloc[tr], Xs.iloc[va])
        oof_pred[va] = pred
        if proba is not None:
            if oof_proba is None:
                oof_proba = np.full((len(ys), proba.shape[1]), np.nan)
            oof_proba[va] = proba
    return score(ys, oof_pred, task, oof_proba)


def main() -> None:
    ap = argparse.ArgumentParser(description="特征组消融")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(SEED)
    out_dir = run_dir(cfg)
    name = cfg["name"]

    tr_path = PATHS["processed"] / f"{name}_train.parquet"
    if not tr_path.exists():
        raise FileNotFoundError(f"缺 {tr_path}。先跑 make train。")
    X = pd.read_parquet(tr_path)

    from src.data import load_table
    train_raw = load_table(cfg["data"]["train"])
    y_raw = train_raw[cfg["data"]["target"]].reset_index(drop=True)
    task = detect_task(y_raw)
    if task == "classification":
        classes = sorted(y_raw.dropna().unique().tolist(), key=str)
        y = y_raw.map({c: i for i, c in enumerate(classes)}).astype(int)
    else:
        y = y_raw.astype(float)

    model_name = cfg.get("model", "lightgbm")
    params = cfg.get("params") or {}
    seed = (cfg.get("seeds") or [SEED])[0]
    pm = primary_metric(task, cfg)
    cv_cfg = cfg.get("cv") or {}

    # 先跑完整模型作为参照
    log.info("参照：完整模型（%d 个特征）", X.shape[1])
    t0 = time.perf_counter()
    full_metrics = run_cv(model_name, params, task, seed, X, y, cv_cfg)
    full_time = time.perf_counter() - t0
    full_score = full_metrics[pm]
    log.info("完整模型 %s = %.5f（%.1f 秒）", pm, full_score, full_time)

    groups = cfg.get("ablation", {}).get("groups")
    Xc = X.copy()
    col_group = {c: assign_group(c, groups) for c in Xc.columns}
    counts = pd.Series(col_group).value_counts()
    log.info("特征分组：\n%s", counts.to_string())

    rows = [{"特征组": "（完整模型）", "特征数": X.shape[1], pm: round(full_score, 6),
             "与完整模型的差": 0.0, "耗时秒": round(full_time, 1)}]

    for grp in counts.index:
        if grp == "基础字段":
            log.info("跳过「基础字段」——砍掉它就没模型了")
            continue
        drop = [c for c, g in col_group.items() if g == grp]
        if not drop:
            continue
        log.info("砍掉「%s」(%d 列) 重跑……", grp, len(drop))
        t0 = time.perf_counter()
        m = run_cv(model_name, params, task, seed, Xc.drop(columns=drop), y, cv_cfg)
        dt = time.perf_counter() - t0
        s = m[pm]
        # 与完整模型的差：正数 = 砍掉这组之后反而变好 = 这组是噪音
        delta = improvement(pm, s, full_score)
        rows.append({"特征组": grp, "特征数": len(drop), pm: round(s, 6),
                     "与完整模型的差": round(delta, 6), "耗时秒": round(dt, 1)})
        log.info("  砍掉「%s」后 %s = %.5f（差 %+.5f）", grp, pm, s, delta)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation.csv", index=False, encoding="utf-8-sig")

    print(f"\n=== 消融表（主指标 {pm}）===")
    print(df.to_string(index=False))
    losers = df[(df["特征组"] != "（完整模型）") & (df["与完整模型的差"] > 0)]
    if len(losers):
        print(f"\n⚠ 砍掉后反而变好的组（负贡献，G4 会判定不过）：")
        for _, r in losers.iterrows():
            print(f"   {r['特征组']}：砍掉后 {pm} 提升 {r['与完整模型的差']:.5f} ⇒ 应该下线")
    else:
        print("\n✓ 每个特征组都有正贡献 —— 论文里就说：砍掉任一组主指标都会下降。")
    print(f"\n明细 -> {out_dir / 'ablation.csv'}")


if __name__ == "__main__":
    main()
