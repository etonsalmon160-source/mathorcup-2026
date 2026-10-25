"""论文用补充表：方差分解 + 共表达模块特征基因（ME）与感官类别关联。

AI 辅助声明：本脚本由 AI 辅助编写；分解顺序、模块定义与统计口径由参赛队员
决定，AI 只负责实现与跑数。

方差分解（逐感官维度，5 折随机 CV，多种子平均）
------------------------------------------------
按预先固定的顺序做**序贯残差分解**，每一步只用训练折拟合：
  1. 风格先验：训练折内各风格均值；未见风格退回全局均值。
  2. PLS8 潜流形：在去风格均值后的残差上拟合 8 维有监督 PLS。
  3. ET 非线性残差：对第 2 步的样本内残差拟合 Extra Trees。
R² 在拼接后的外层 OOF 上计算；增量份额 = 后一步 R² − 前一步 R²，
保留负值（表示该层在外层没有增量解释力），同时给出截断到 0 的可绘图份额。

ME × 感官类别
-------------
每个 Louvain 模块取第一主成分得分作为模块特征基因（ME），符号按模块平均
表达方向统一。ME 与 50 个感官维度做 Spearman 相关，BH-FDR 校正；再按感官
Category 汇总显著数与平均 |rho|，供热图使用。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v1 as R  # noqa: E402

OUT = R.RESULTS / "supplementary"
SENSORY_LATENT_K = R.SENSORY_LATENT_K


def style_prior(styles_tr, Ytr, styles_te, Yte):
    prior = pd.DataFrame(Ytr).groupby(styles_tr).mean()
    global_mean = Ytr.mean(0)
    pred = np.vstack([
        prior.loc[s].to_numpy(float) if s in prior.index else global_mean
        for s in styles_te])
    return pred


def latent_residual(Xtr, Ytr, Xte, kind="pca"):
    """在风格残差上拟合 8 维潜状态，返回训练/测试的残差预测。

    pca：主流程口径（PCA8 + RidgeCV(logspace(-1,3,9))）；
    pls8：对照口径（有监督 PLS8），用于敏感性表。
    """
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    A, B = R._impute_scale(Xtr, Xte)
    k = min(SENSORY_LATENT_K, A.shape[1], Ytr.shape[1], A.shape[0] - 1)
    if kind == "pca":
        pca = PCA(n_components=k, random_state=R.SEED)
        T = pca.fit_transform(Ytr)
        reg = RidgeCV(alphas=np.logspace(-1, 3, 9)).fit(A, T)
        return pca.inverse_transform(reg.predict(A)), \
            pca.inverse_transform(reg.predict(B))
    m = PLSRegression(n_components=k, scale=False, max_iter=500)
    m.fit(A, Ytr)
    return m.predict(A), m.predict(B)


def run_seed(X, Y, styles, seed, latent="pca"):
    from sklearn.model_selection import KFold
    from sklearn.ensemble import ExtraTreesRegressor
    n, m = Y.shape
    preds = {k: np.full_like(Y, np.nan) for k in ("style", "plus_pls", "full")}
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        p1_tr = style_prior(styles[tr], Y[tr], styles[tr], Y[tr])
        p1_te = style_prior(styles[tr], Y[tr], styles[te], Y[te])
        preds["style"][te] = p1_te

        s2_tr, s2_te = latent_residual(X[tr], Y[tr] - p1_tr, X[te],
                                       kind=latent)
        preds["plus_pls"][te] = p1_te + s2_te

        r3 = Y[tr] - p1_tr - s2_tr
        et = ExtraTreesRegressor(
            n_estimators=300, max_features="sqrt", min_samples_leaf=2,
            random_state=seed, n_jobs=R.N_JOBS)
        et.fit(X[tr], r3)
        preds["full"][te] = preds["plus_pls"][te] + et.predict(X[te])
    return preds


def r2_cols(y, p):
    ss_res = ((y - p) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    ss_tot = np.where(ss_tot > 1e-12, ss_tot, np.nan)
    return 1.0 - ss_res / ss_tot


def variance_tables(data, seeds, latent="pca"):
    X = np.column_stack([data.X[data.chem_all].to_numpy(float), data.cdr])
    Y = data.Y[data.targets].to_numpy(float)
    acc = []
    for seed in seeds:
        p = run_seed(X, Y, data.styles, seed, latent=latent)
        acc.append({k: r2_cols(Y, v) for k, v in p.items()})
    r2 = {k: np.mean([a[k] for a in acc], 0) for k in acc[0]}

    sn = R.load_tables().sensory_names.drop_duplicates("R_name").set_index("R_name")
    cat = sn.reindex(data.targets)["Category"].fillna("Others").to_dict()
    df = pd.DataFrame({
        "sensory": data.targets,
        "category": [cat[t] for t in data.targets],
        "r2_style": r2["style"],
        "r2_plus_pls": r2["plus_pls"],
        "r2_full_et": r2["full"],
    })
    df["delta_pls"] = df["r2_plus_pls"] - df["r2_style"]
    df["delta_et"] = df["r2_full_et"] - df["r2_plus_pls"]
    for c in ("r2_style", "delta_pls", "delta_et"):
        df[f"share_{c.removeprefix('r2_').removeprefix('delta_')}_plot"] = \
            df[c].clip(lower=0)
    denom = df[["share_style_plot", "share_pls_plot", "share_et_plot"]].sum(1)
    for c in ("style", "pls", "et"):
        df[f"share_{c}"] = np.where(denom > 1e-12,
                                    df[f"share_{c}_plot"] / denom, np.nan)
    suf = "" if latent == "pca" else f"_{latent}"
    df.to_csv(OUT / f"variance_decomposition{suf}.csv", index=False,
              encoding="utf-8-sig")

    summ = df.groupby("category").agg(
        n=("sensory", "size"),
        r2_style=("r2_style", "mean"),
        r2_plus_pls=("r2_plus_pls", "mean"),
        r2_full_et=("r2_full_et", "mean"),
        delta_pls=("delta_pls", "mean"),
        delta_et=("delta_et", "mean"),
        share_style=("share_style", "mean"),
        share_pls=("share_pls", "mean"),
        share_et=("share_et", "mean"),
    ).reset_index()
    summ.to_csv(OUT / f"variance_decomposition_by_category{suf}.csv",
                index=False, encoding="utf-8-sig")
    overall = {
        "n_targets": int(len(df)),
        "mean_r2_style": float(df.r2_style.mean()),
        "mean_r2_plus_pls": float(df.r2_plus_pls.mean()),
        "mean_r2_full_et": float(df.r2_full_et.mean()),
        "mean_delta_pls": float(df.delta_pls.mean()),
        "mean_delta_et": float(df.delta_et.mean()),
        "n_delta_pls_negative": int((df.delta_pls < 0).sum()),
        "n_delta_et_negative": int((df.delta_et < 0).sum()),
        "seeds": list(seeds), "latent": latent,
        "note": "序贯残差分解；负增量表示该层在外层 OOF 上没有增益，未做截断。",
    }
    pd.DataFrame([overall]).to_csv(
        OUT / f"variance_decomposition_overall{suf}.csv",
        index=False, encoding="utf-8-sig")
    return df, overall


def me_tables(data):
    from scipy.stats import spearmanr
    from statsmodels.stats.multitest import multipletests

    comm = pd.read_csv(R.RESULTS / "q2" / "communities.csv")
    comm.columns = [c.lstrip("﻿") for c in comm.columns]
    node2mod = dict(zip(comm["node"].astype(str), comm["community"].astype(int)))
    short2disp = {c: data.chem_map[c] for c in data.chem_nd}
    disp2short = {v: k for k, v in short2disp.items()}
    mods = sorted(set(node2mod.values()))

    X = data.X[data.chem_nd].to_numpy(float)
    A, _ = R._impute_scale(X, X)
    col = {c: i for i, c in enumerate(data.chem_nd)}
    me = np.zeros((len(data.styles), len(mods)))
    top = {}
    for k, mod in enumerate(mods):
        members = [disp2short[disp] for disp, mm in node2mod.items()
                   if mm == mod and disp in disp2short]
        cols = [col[c] for c in members]
        Z = A[:, cols]
        Z = (Z - Z.mean(0)) / np.where(Z.std(0) < 1e-8, 1.0, Z.std(0))
        u, s, vt = np.linalg.svd(Z, full_matrices=False)
        pc = u[:, 0] * s[0]
        if np.corrcoef(pc, Z.mean(1))[0, 1] < 0:
            pc = -pc
        me[:, k] = pc
        deg = comm[comm["community"] == mod].sort_values("degree", ascending=False)
        top[mod] = "、".join(deg["node"].astype(str).head(5))

    Y = data.Y[data.targets].to_numpy(float)
    rho = np.zeros((len(mods), len(data.targets)))
    pval = np.zeros_like(rho)
    for i in range(len(mods)):
        for j in range(Y.shape[1]):
            rho[i, j], pval[i, j] = spearmanr(me[:, i], Y[:, j])
    fdr = multipletests(pval.ravel(), method="fdr_bh")[1].reshape(pval.shape)

    sn = R.load_tables().sensory_names.drop_duplicates("R_name").set_index("R_name")
    cat = sn.reindex(data.targets)["Category"].fillna("Others").astype(str).to_dict()
    rho_df = pd.DataFrame(rho, index=[f"ME{m}" for m in mods],
                          columns=data.targets)
    rho_df.insert(0, "module", [f"ME{m}" for m in mods])
    rho_df.insert(1, "top_compounds", [top[m] for m in mods])
    rho_df.to_csv(OUT / "me_sensory_rho.csv", index=False, encoding="utf-8-sig")

    long = []
    for i, mod in enumerate(mods):
        for j, t in enumerate(data.targets):
            long.append({"module": f"ME{mod}", "target": t, "category": cat[t],
                         "rho": rho[i, j], "p": pval[i, j], "fdr": fdr[i, j],
                         "top_compounds": top[mod]})
    long = pd.DataFrame(long)
    long.to_csv(OUT / "me_sensory_long.csv", index=False, encoding="utf-8-sig")

    rows = []
    for (mod, c), sub in long.groupby(["module", "category"], sort=True):
        rows.append({
            "module": mod, "category": c, "n_targets": len(sub),
            "n_fdr_lt_0.05": int((sub.fdr < 0.05).sum()),
            "mean_abs_rho": float(sub.rho.abs().mean()),
            "median_rho": float(sub.rho.median()),
            "top_target": sub.loc[sub.rho.abs().idxmax(), "target"],
            "top_rho": float(sub.loc[sub.rho.abs().idxmax(), "rho"]),
            "top_compounds": sub.top_compounds.iloc[0],
        })
    bycat = pd.DataFrame(rows)
    bycat.to_csv(OUT / "me_sensory_category.csv", index=False,
                 encoding="utf-8-sig")
    mat = bycat.pivot(index="module", columns="category", values="mean_abs_rho")
    mat.to_csv(OUT / "me_sensory_category_matrix.csv", encoding="utf-8-sig")
    return long, bycat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(R.SEEDS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    R.set_global_seed(R.SEED)
    data = R.load_data()

    print("[A] 序贯方差分解（PCA8 主口径 + PLS8 敏感性）")
    for latent in ("pca", "pls8"):
        df, overall = variance_tables(data, args.seeds, latent=latent)
        print(f"-- {latent} --")
        print(df[["sensory", "r2_style", "r2_plus_pls", "r2_full_et",
                  "share_style", "share_pls", "share_et"]].head(5).to_string(index=False))
        print(overall)

    print("[B] ME × 感官类别")
    long, bycat = me_tables(data)
    print(bycat.to_string(index=False))
    print(f"完成：{OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
