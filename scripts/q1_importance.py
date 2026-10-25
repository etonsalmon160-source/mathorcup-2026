#!/usr/bin/env python
# 本文件由 AI 辅助编写；重要性方法（置换重要性）与口径由参赛队员按 C3 检查点选定。
"""问题一主模型的置换重要性：化合物级、模块级，以及与 Q2 Lasso 频率的对照。

做什么
------
主模型是 `avg`（ET 50% + 8 维 PCA 感官潜状态 Ridge 50%）。在按风格分组的
5 折外层 CV 内，逐折训练主模型，再逐列打乱测试折的原始输入，记录标准化
MSE 的增量。3 个种子 × 5 折 × 每列 3 次置换，最后取均值与标准差。

为什么用置换重要性
------------------
- 对多输出 (50 维) 直接成立，不依赖树结构；
- 主模型是两路平均，置换能同时反映 ET 与潜状态两路的贡献；
- 与 Q2 的 Lasso 选择频率形成"模型侧 vs 稀疏回归侧"的对照，满足 C4 的
  「与 MDI/SHAP 类重要性对照」要求。

已知限制（写论文时必须说）
--------------------------
- 共线化合物会分摊重要性（例如乙醇与乙酯类），单个数值不能读成因果效应；
- 置换打破列间相关，若模型依赖联合分布，重要性可能被高估；
- 只覆盖检出的输入分布，不对检出限以下的合成谱外推。

产物（只新建，不改主结果）
--------------------------
results/q1/importance.csv             化合物级：ΔMSE(z)、标准差、相对增幅、排名
results/q1/importance_by_target.csv   232 × 50 的 ΔMSE(z) 矩阵
results/q1/importance_modules.csv     模块级汇总（模块来自 Q2 的 Louvain 划分）
results/q1/importance_vs_lasso.csv    逐感官与 Q2 选择频率的 Spearman / Top-10 重合
figures/q1/fig_q1_importance_top.png
figures/q1/fig_q1_module_importance.png
figures/q1/fig_q1_heatmap.png

运行
----
    python scripts/q1_importance.py --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import run_v1 as R  # noqa: E402
from src.config import PATHS  # noqa: E402

Q1 = R.Q1
FIG = PATHS["figures"] / "q1"


def fit_avg(Xtr: np.ndarray, Ytr: np.ndarray, seed: int) -> dict:
    """与主流程同超参：ET + PCA8 RidgeCV，输出两路模型对象。"""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import RidgeCV

    Aet, _ = R._impute(Xtr, Xtr)
    et = ExtraTreesRegressor(n_estimators=300, max_features=0.3, min_samples_leaf=2,
                             n_jobs=R.N_JOBS, random_state=seed).fit(Aet, Ytr)
    Asc, _ = R._impute_scale(Xtr, Xtr)
    k = min(R.SENSORY_LATENT_K, Ytr.shape[1] - 1)
    pca = PCA(n_components=k, random_state=seed).fit(Ytr)
    ridge = RidgeCV(alphas=np.logspace(-1, 3, 9)).fit(Asc, pca.transform(Ytr))
    return {"Xtr": Xtr, "et": et, "pca": pca, "ridge": ridge}


def predict_avg(m: dict, Xte: np.ndarray) -> np.ndarray:
    _, Bet = R._impute(m["Xtr"], Xte)
    _, Bsc = R._impute_scale(m["Xtr"], Xte)
    return 0.5 * m["et"].predict(Bet) + 0.5 * m["pca"].inverse_transform(
        m["ridge"].predict(Bsc))


def permutation_importance(X, Y, styles, seeds, n_rep, log) -> tuple[pd.DataFrame, np.ndarray]:
    from sklearn.model_selection import GroupKFold

    sd = Y.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = Y / sd  # 标准化后逐维 MSE 可比

    n, m = Y.shape
    p = X.shape[1]
    acc = np.zeros((p, m))
    base = np.zeros((p, m))
    n_runs = 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for fold, (tr, te) in enumerate(
                GroupKFold(5).split(X, Y[:, 0], groups=styles), 1):
            t0 = time.time()
            model = fit_avg(X[tr], Y[tr], seed)
            Xte = X[te]
            pred0 = predict_avg(model, Xte)
            mse0 = ((Z[te] - pred0) ** 2).mean(0)
            acc_fold = np.zeros((p, m))
            for j in range(p):
                ds = np.empty((n_rep, m))
                for r in range(n_rep):
                    Xp = Xte.copy()
                    Xp[:, j] = rng.permutation(Xp[:, j])
                    ds[r] = ((Z[te] - predict_avg(model, Xp)) ** 2).mean(0) - mse0
                acc_fold[j] = ds.mean(0)
            acc += acc_fold
            base += mse0
            n_runs += 1
            log(f"重要性 seed={seed} fold={fold} 完成（{time.time()-t0:.0f}s）")
    return pd.DataFrame(acc / n_runs), base / n_runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(R.SEEDS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--from-cache", action="store_true",
                    help="读已算好的重要性矩阵，只重算汇总表与图")
    args = ap.parse_args()

    R.set_global_seed(R.SEED)
    Q1.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    data = R.load_data()
    X = np.column_stack([data.X[data.chem_all].to_numpy(float), data.cdr])
    Y = data.Y[data.targets].to_numpy(float)
    feats = list(data.chem_all) + ["CDR"]
    assert X.shape[1] == len(feats)

    t0 = time.time()
    imp_file = Q1 / "importance_by_target.csv"
    base_file = Q1 / "importance_base_by_target.csv"
    if args.from_cache:
        imp_by_target = pd.read_csv(imp_file, index_col=0, encoding="utf-8-sig")
        imp_by_target = imp_by_target.reindex(index=feats, columns=data.targets)
        prev = pd.read_csv(Q1 / "importance.csv", encoding="utf-8-sig")
        prev.columns = [c.lstrip("﻿") for c in prev.columns]
        rel = prev.set_index("feature").reindex(feats)["rel_increase"].to_numpy()
        log(f"从缓存读取 {imp_file.name}")
    else:
        imp_by_target, base_by_target = permutation_importance(
            X, Y, data.styles, args.seeds, args.reps, log)
        imp_by_target.index = feats
        imp_by_target.columns = data.targets
        imp_by_target.to_csv(imp_file, encoding="utf-8-sig")
        pd.DataFrame(base_by_target, index=feats,
                     columns=data.targets).to_csv(base_file,
                                                  encoding="utf-8-sig")

    delta = imp_by_target.mean(1)
    delta_std = imp_by_target.std(1, ddof=1)
    if not args.from_cache:
        rel = (imp_by_target / np.maximum(base_by_target, 1e-6)).clip(0, 50).mean(1)
    fam = R._family_map(data)
    fam["CDR"] = "技术协变量"

    # 模块映射：Q2 的 Louvain 划分按 2.csv 全名对齐
    comm = pd.read_csv(R.RESULTS / "q2" / "communities.csv",
                       encoding="utf-8-sig")
    comm.columns = [c.lstrip("﻿") for c in comm.columns]
    disp2mod = dict(zip(comm["node"].astype(str), comm["community"].astype(int)))
    modules = []
    for c in feats:
        if c == "CDR":
            modules.append("CDR")
        elif c not in data.chem_nd:
            modules.append("派生汇总列")
        else:
            modules.append(disp2mod.get(str(data.chem_map.get(c, c)), "未入图"))

    out = pd.DataFrame({
        "feature": feats,
        "display": [data.chem_map.get(c, c) if c != "CDR" else "样本检出率" for c in feats],
        "family": [fam.get(c, "") for c in feats],
        "module": modules,
        "delta_mse_z": delta.to_numpy(),
        "delta_mse_z_std": delta_std.to_numpy(),
        "rel_increase": np.asarray(rel),
    }).sort_values("delta_mse_z", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    out.to_csv(Q1 / "importance.csv", index=False, encoding="utf-8-sig")

    # 模块级汇总
    rows = []
    for mod, sub in out.groupby("module", sort=False):
        tops = sub.head(5)
        rows.append({
            "module": mod,
            "n_members": len(sub),
            "delta_mse_z_sum": float(sub.delta_mse_z.sum()),
            "delta_mse_z_mean": float(sub.delta_mse_z.mean()),
            "rel_increase_mean": float(sub.rel_increase.mean()),
            "top_compounds": "、".join(tops.display.tolist()),
            "top_delta": float(tops.delta_mse_z.iloc[0]),
        })
    mods = pd.DataFrame(rows).sort_values("delta_mse_z_sum", ascending=False)
    pos_sum = mods.delta_mse_z_sum.clip(lower=0).sum()
    mods["share"] = mods.delta_mse_z_sum.clip(lower=0) / pos_sum if pos_sum > 0 else np.nan
    mods.to_csv(Q1 / "importance_modules.csv", index=False, encoding="utf-8-sig")

    # 与 Q2 Lasso 选择频率逐维对照
    ev = pd.read_csv(R.RESULTS / "q2" / "evidence_scores.csv",
                     index_col=0, encoding="utf-8-sig")
    ev.columns = [c.lstrip("﻿") for c in ev.columns]
    from scipy.stats import spearmanr

    cross = []
    for t in data.targets:
        if t not in ev.index:
            continue
        imp_vec = imp_by_target.loc[data.chem_nd, t].to_numpy()
        fre_vec = ev.loc[t, data.chem_nd].to_numpy(float)
        rho, pv = spearmanr(imp_vec, fre_vec)
        k = 10
        A = set(np.argsort(-imp_vec)[:k])
        B = set(np.argsort(-fre_vec)[:k])
        cross.append({"target": t, "spearman": float(rho),
                      "p": float(pv), "top10_overlap": len(A & B) / k})
    cross = pd.DataFrame(cross)
    cross.to_csv(Q1 / "importance_vs_lasso.csv", index=False, encoding="utf-8-sig")

    # 图 1：Top-25 化合物
    fig, ax = plt.subplots(figsize=(8, 9))
    top = out.head(25).iloc[::-1]
    ax.barh(np.arange(len(top)), top.delta_mse_z,
            xerr=top.delta_mse_z_std, color="#4c72b0", alpha=0.85,
            error_kw={"lw": 0.8, "capsize": 2})
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top.display, fontsize=9)
    ax.set_xlabel("ΔMSE(z) increase after permutation (mean over 50 targets)")
    ax.set_title("Q1 main model: top-25 compound permutation importance")
    fig.savefig(FIG / "fig_q1_importance_top.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 图 2：模块级
    fig, ax = plt.subplots(figsize=(9, 5.5))
    mm = mods.iloc[::-1]
    mod_en = {"派生汇总列": "derived summary cols", "未入图": "not in graph",
              "CDR": "CDR"}
    labels = [f"{mod_en.get(r.module, f'Module {r.module}')} ({r.n_members})"
              for r in mm.itertuples()]
    ax.barh(np.arange(len(mm)), mm.delta_mse_z_sum, color="#55a868", alpha=0.85)
    ax.set_yticks(np.arange(len(mm)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Σ ΔMSE(z) within module")
    ax.set_title("Q1 main model: co-expression module importance (members)")
    fig.savefig(FIG / "fig_q1_module_importance.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # 图 3：50 维 R² 热图（主模型 / ET / PCA8 / PLS8 对照）
    comp = pd.read_csv(R.RESULTS / "composite" / "per_target.csv")
    q1pt = pd.read_csv(Q1 / "per_target_all.csv")
    q1pt = q1pt[q1pt.model == "latent_sens_pls"]
    names = {"avg": "main avg", "et": "ET only", "latent_sens_ridge": "PCA8 latent",
             "latent_sens_pls": "PLS8 control"}
    sn = R.load_tables().sensory_names.drop_duplicates("R_name").set_index("R_name")
    cats = sn.reindex(data.targets)["Category"].fillna("Others").astype(str)
    order = sorted(range(len(data.targets)),
                   key=lambda i: (cats.iloc[i], data.targets[i]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 12), sharey=True)
    for ax, scheme, title in zip(axes, ("random", "group"),
                                 ("random 5-fold", "style-grouped 5-fold")):
        mat = []
        for key in ("avg", "et", "latent_sens_ridge", "latent_sens_pls"):
            src = comp if key in comp.variant.unique() else q1pt
            col = "variant" if src is comp else "model"
            sub = src[(src[col] == key) & (src.scheme == scheme)]
            mat.append(sub.groupby("target").r2.mean().reindex(data.targets))
        mat = pd.DataFrame(mat, index=list(names.values())).T.iloc[order]
        im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="RdYlGn",
                       vmin=-0.1, vmax=0.7)
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(mat)))
        ax.set_yticklabels([f"{data.targets[i]} ({cats.iloc[i]})" for i in order],
                           fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="R²")
    fig.suptitle("Q1: per-target R² across 50 sensory dimensions", fontsize=14)
    fig.savefig(FIG / "fig_q1_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "seeds": args.seeds, "reps": args.reps, "n_features": len(feats),
        "runtime_s": round(time.time() - t0, 1),
        "top10": out.head(10)[["display", "delta_mse_z"]].to_dict("records"),
        "modules": mods[["module", "n_members", "delta_mse_z_sum", "share"]]
        .to_dict("records"),
        "vs_lasso": {
            "mean_spearman": float(cross.spearman.mean()),
            "median_spearman": float(cross.spearman.median()),
            "mean_top10_overlap": float(cross.top10_overlap.mean()),
            "n_targets": int(len(cross)),
        },
        "note": ("置换在主模型 avg 上做；共线特征会分摊重要性；"
                 "负值表示打乱该列后预测反而略好，保留不截断。"),
    }
    (Q1 / "importance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"完成，用时 {summary['runtime_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
