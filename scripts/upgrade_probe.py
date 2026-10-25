#!/usr/bin/env python
# 本文件由 AI 辅助编写；三项升级是否采用由参赛队员决定。
"""三项升级的独立对比评测（只写 results/supplementary/upgrade/）。

不改 Q1/Q2/Q3 任何主结果，也不改主流程：

A. Q1 检出限插补：当前「哨兵置 NaN + 中位数」vs 左截断半检出限
   （log10 尺度减 log10(2)）、KNN 原生缺失、检出指示特征。
B. Q1 感官潜状态：PCA(8) vs FactorAnalysis(8) vs PLS / CCA。
C. Q3 迁移：当前逐坐标裁剪 ot_proj / Schur 补 ot_cond vs
   全通路 + 感官读出联合受控凸二次规划（SLSQP）。
D. Q2 图传播：一阶邻居平均 vs 归一化拉普拉斯正则 vs 带重启随机游走。

用法（单任务 ≤8 线程）：
    python scripts/upgrade_probe.py --only q1
    python scripts/upgrade_probe.py --only q3
    python scripts/upgrade_probe.py --only q3 --pairs
    python scripts/upgrade_probe.py --only q2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_v1 as R  # noqa: E402

OUT = R.RESULTS / "supplementary" / "upgrade"
OUT.mkdir(parents=True, exist_ok=True)
LOG10_2 = float(np.log10(2.0))
N_BOOT_DIM = 2000

# 扩展异味通路：题面四项 + 酵母硫代谢 + 金属味候选。
EXT_DEFECTS = {
    "DMS": {"threshold": 19.0, "scale": "log10", "unit": "ug/L"},
    "X4VG": {"threshold": 0.20, "scale": "log10", "unit": "mg/L"},
    "X2.3.butanedione": {"threshold": 0.09, "scale": "raw", "unit": "mg/L"},
    "X.E..2.nonenal": {"threshold": None, "scale": "raw", "unit": "mg/L"},
    "H2S": {"threshold": 3.5, "scale": "log10", "unit": "ug/L"},
    "MeSH": {"threshold": None, "scale": "log10", "unit": "ug/L"},
    "DES": {"threshold": 1.2, "scale": "log10", "unit": "ug/L"},
    "iron": {"threshold": 0.10, "scale": "log10", "unit": "mg/L"},
}
BASE_DEFECTS = ("DMS", "X4VG", "X2.3.butanedione", "X.E..2.nonenal")
SENS_CONSTR = ["stale_hops", "dms", "diacetyl", "barnyard", "metallic"]


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def save_json(name: str, obj: dict) -> None:
    (OUT / name).write_text(
        json.dumps(_jsonable(obj), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


# ============================================================ A/B. Q1

def _prep_xy(data):
    X = np.column_stack([data.X[data.chem_all].to_numpy(float), data.cdr])
    Y = data.Y[data.targets].to_numpy(float)
    return X, Y


_CHEM_LOG_MASK = None


def _chem_log_mask(n=None):
    """231 个化合物理化列中哪些是 log10 口径（其余为原始量纲）。"""
    global _CHEM_LOG_MASK
    if _CHEM_LOG_MASK is None:
        t = pd.read_csv(ROOT / "docs/data_contract"
                        / "transform_classification_v1.csv")
        m = dict(zip(t["column"].astype(str), t["transform"].astype(str)))
        cols = R.load_data().chem_all
        _CHEM_LOG_MASK = np.array([m.get(c, "log10") == "log10"
                                   for c in cols])
    return _CHEM_LOG_MASK if n is None else _CHEM_LOG_MASK[:n]


def _impute_stats(Xtr):
    """与 _impute_scale 相同：返回训练矩阵、列均值、列标准差与插补器。"""
    from sklearn.impute import SimpleImputer
    Xtr = np.asarray(Xtr, float)
    fv = R._floor_values(Xtr)
    Araw = np.where(R._floor_mask(Xtr, fv), np.nan, Xtr)
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    A = imp.fit_transform(Araw)
    mu = A.mean(0)
    sd = A.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return A, mu, sd, imp


def _impute_variant(Xtr, Xte, mode="median", cdr=True, indicators=False,
                    ind_rate=0.10):
    """训练/测试矩阵的插补 + 标准化。

    mode:
      median  当前口径：检出下限 -> NaN -> 训练折中位数；
      halfmin 左截断：log10 列 -> 下限 - log10(2)，原始量纲列 -> 下限/2
              （都是物理浓度减半），原生缺失仍用训练折中位数；
      knn     同 halfmin，原生缺失改用 KNNImputer(k=5)；
      plain   不识别哨兵（审计前口径），中位数插补。
    indicators: 对训练折检出率 >= ind_rate 的列追加二值「已检出」指示列。
    """
    from sklearn.impute import KNNImputer, SimpleImputer

    Xtr = np.asarray(Xtr, float)
    Xte = np.asarray(Xte, float)
    nchem = Xtr.shape[1] - 1
    if not cdr:
        Xtr, Xte = Xtr[:, :nchem], Xte[:, :nchem]
        nchem = Xtr.shape[1]
    A, B = Xtr.copy(), Xte.copy()
    ind_tr = ind_te = None

    if mode == "plain":
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        A = imp.fit_transform(A)
        B = imp.transform(B)
    else:
        fv = R._floor_values(Xtr[:, :nchem])
        fm_tr = R._floor_mask(Xtr[:, :nchem], fv)
        fm_te = R._floor_mask(Xte[:, :nchem], fv)
        if indicators:
            cols = np.flatnonzero(fm_tr.mean(0) >= ind_rate)
            if len(cols):
                ind_tr = (~fm_tr[:, cols]).astype(float)
                ind_te = (~fm_te[:, cols]).astype(float)
                ind_tr[np.isnan(Xtr[:, cols])] = 0.0
                ind_te[np.isnan(Xte[:, cols])] = 0.0

        if mode == "median":
            A[:, :nchem] = np.where(fm_tr, np.nan, A[:, :nchem])
            B[:, :nchem] = np.where(fm_te, np.nan, B[:, :nchem])
            imp = SimpleImputer(strategy="median", keep_empty_features=True)
            A[:, :nchem] = imp.fit_transform(A[:, :nchem])
            B[:, :nchem] = imp.transform(B[:, :nchem])
        elif mode in ("halfmin", "knn"):
            med = np.nanmedian(np.where(fm_tr, np.nan, A[:, :nchem]), axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            lm = _chem_log_mask(nchem)
            half = np.where(lm, fv - LOG10_2, fv / 2.0)
            for M, fm in ((A, fm_tr), (B, fm_te)):
                cell = np.where(fm, half[None, :], M[:, :nchem])
                M[:, :nchem] = np.where(np.isnan(cell), med[None, :], cell)
            if mode == "knn":
                mu = np.nanmean(A[:, :nchem], 0)
                sd = np.nanstd(A[:, :nchem], 0)
                sd = np.where(sd > 0, sd, 1.0)
                knn = KNNImputer(n_neighbors=5, keep_empty_features=True)
                A[:, :nchem] = knn.fit_transform(
                    (A[:, :nchem] - mu) / sd) * sd + mu
                B[:, :nchem] = knn.transform(
                    (B[:, :nchem] - mu) / sd) * sd + mu
        else:
            raise KeyError(mode)

    if ind_tr is not None:
        A = np.column_stack([A, ind_tr])
        B = np.column_stack([B, ind_te])

    mu = A.mean(0)
    sd = A.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (np.clip((A - mu) / sd, -10, 10),
            np.clip((B - mu) / sd, -10, 10))


def _latent_fit(A, Ytr, B, latent):
    from sklearn.cross_decomposition import CCA, PLSRegression
    from sklearn.decomposition import FactorAnalysis, PCA
    from sklearn.linear_model import RidgeCV

    k = R.SENSORY_LATENT_K
    alphas = [0.1, 1, 10, 100, 1000]
    if latent == "pca":
        pca = PCA(n_components=k, random_state=R.SEED)
        T = pca.fit_transform(Ytr)
        return pca.inverse_transform(RidgeCV(alphas=alphas).fit(A, T).predict(B))
    if latent == "fa":
        fa = FactorAnalysis(n_components=k, random_state=R.SEED, max_iter=1000)
        T = fa.fit_transform(Ytr)
        # FA 没有 inverse_transform：均值重建为 T @ 载荷 + 均值
        return RidgeCV(alphas=alphas).fit(A, T).predict(B) @ fa.components_ \
            + fa.mean_
    if latent.startswith("pls"):
        m = PLSRegression(n_components=int(latent[3:]), scale=False,
                          max_iter=500)
        return m.fit(A, Ytr).predict(B)
    if latent.startswith("cca"):
        cca = CCA(n_components=int(latent[3:]), scale=True, max_iter=2000,
                  tol=1e-8)
        xt, yt = cca.fit_transform(A, Ytr)
        reg = RidgeCV(alphas=alphas).fit(xt, yt)
        xt_te = cca.transform(B)
        _, y_rec = cca.inverse_transform(xt_te, reg.predict(xt_te))
        return y_rec
    raise KeyError(latent)


def _et_fit(A, Ytr, B, seed):
    from sklearn.ensemble import ExtraTreesRegressor
    m = ExtraTreesRegressor(n_estimators=300, max_features=0.3,
                            min_samples_leaf=2, n_jobs=R.N_JOBS,
                            random_state=seed)
    return m.fit(A, Ytr).predict(B)


def _splits(data, scheme, seed):
    from sklearn.model_selection import GroupKFold, KFold
    X, _ = _prep_xy(data)
    if scheme == "random":
        return list(KFold(5, shuffle=True, random_state=seed).split(X))
    return list(GroupKFold(5).split(X, groups=data.styles))


def _oof(data, mode, cdr, indicators, scheme, seed, latents):
    """一次遍历得到 ET 与各潜状态模型的 OOF 预测。"""
    X, Y = _prep_xy(data)
    oof = {"et": np.full_like(Y, np.nan)}
    for lat in latents:
        oof[lat] = np.full_like(Y, np.nan)
    for tr, te in _splits(data, scheme, seed):
        A, B = _impute_variant(X[tr], X[te], mode=mode, cdr=cdr,
                               indicators=indicators)
        oof["et"][te] = _et_fit(A, Y[tr], B, seed)
        for lat in latents:
            oof[lat][te] = _latent_fit(A, Y[tr], B, lat)
    return {k: R._target_metrics(Y, v)[0] for k, v in oof.items()}


LOD_VARIANTS = [
    ("median", True, False, "当前：中位数+CDR"),
    ("median", False, False, "中位数，无CDR"),
    ("halfmin", True, False, "半检出限+CDR"),
    ("halfmin", False, False, "半检出限，无CDR"),
    ("knn", True, False, "半检出限+KNN+CDR"),
    ("halfmin", True, True, "半检出限+检出指示+CDR"),
    ("plain", True, False, "哨兵当真值+CDR"),
    ("plain", False, False, "审计前：哨兵当真值，无CDR"),
]


def q1_lod(data) -> dict:
    rows, dims = [], {}
    for mode, cdr, ind, label in LOD_VARIANTS:
        for scheme in ("random", "group"):
            for seed in R.SEEDS:
                r2 = _oof(data, mode, cdr, ind, scheme, seed, ["pca"])
                for model in ("et", "avg"):
                    v = (r2["et"] if model == "et"
                         else 0.5 * (r2["et"] + r2["pca"]))
                    dims[(label, scheme, seed, model)] = v
                    rows.append({"label": label, "scheme": scheme, "seed": seed,
                                 "model": model, "mean_r2": float(np.mean(v))})
        print(f"  A {label} 完成", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "q1_lod_raw.csv", index=False, encoding="utf-8-sig")

    pivot = raw.pivot_table(index="label", columns=["model", "scheme"],
                            values="mean_r2", aggfunc="mean")
    order = [v[3] for v in LOD_VARIANTS]
    pivot = pivot.reindex(order)
    base = pivot.loc["当前：中位数+CDR"]
    delta = pivot.sub(base, axis=1)
    pivot.to_csv(OUT / "q1_lod_pivot.csv", encoding="utf-8-sig")
    delta.to_csv(OUT / "q1_lod_delta.csv", encoding="utf-8-sig")

    rng = np.random.default_rng(R.SEED)
    boot = rng.integers(0, 50, size=(N_BOOT_DIM, 50))
    ci_rows = []
    for label in order:
        if label == "当前：中位数+CDR":
            continue
        for model in ("et", "avg"):
            for scheme in ("random", "group"):
                d = np.mean([dims[(label, scheme, s, model)]
                             - dims[("当前：中位数+CDR", scheme, s, model)]
                             for s in R.SEEDS], axis=0)
                bs = d[boot].mean(1)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                ci_rows.append({"label": label, "model": model,
                                "scheme": scheme, "delta": float(np.mean(d)),
                                "ci_lo": float(lo), "ci_hi": float(hi),
                                "n_dim_pos": int((d > 0).sum())})
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(OUT / "q1_lod_delta_ci.csv", index=False, encoding="utf-8-sig")
    print(pivot.round(4).to_string())
    return {"pivot": pivot.round(5).to_dict("index"), "delta_ci": ci_rows}


def q1_latent(data) -> dict:
    variants = ["pca", "fa", "pls8", "pls12", "cca8"]
    rows, dims = [], {}
    for scheme in ("random", "group"):
        for seed in R.SEEDS:
            r2 = _oof(data, "median", True, False, scheme, seed, variants)
            for v in variants:
                dims[(v, scheme, seed)] = r2[v]
                rows.append({"latent": v, "scheme": scheme, "seed": seed,
                             "mean_r2": float(np.mean(r2[v]))})
            rows.append({"latent": "et", "scheme": scheme, "seed": seed,
                         "mean_r2": float(np.mean(r2["et"]))})
        print(f"  B {scheme} 完成", flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "q1_latent_raw.csv", index=False, encoding="utf-8-sig")
    pivot = raw.pivot_table(index="latent", columns="scheme",
                            values="mean_r2", aggfunc="mean")
    pivot["fusion_random"] = 0.5 * (pivot["random"] + pivot.loc["et", "random"])
    pivot["fusion_group"] = 0.5 * (pivot["group"] + pivot.loc["et", "group"])
    pivot.to_csv(OUT / "q1_latent_pivot.csv", encoding="utf-8-sig")

    rng = np.random.default_rng(R.SEED)
    boot = rng.integers(0, 50, size=(N_BOOT_DIM, 50))
    ci_rows = []
    for v in variants:
        for scheme in ("random", "group"):
            d = np.mean([dims[(v, scheme, s)] - dims[("pca", scheme, s)]
                         for s in R.SEEDS], axis=0)
            bs = d[boot].mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            ci_rows.append({"latent": v, "scheme": scheme,
                            "delta_vs_pca": float(np.mean(d)),
                            "ci_lo": float(lo), "ci_hi": float(hi),
                            "n_dim_pos": int((d > 0).sum())})
    pd.DataFrame(ci_rows).to_csv(OUT / "q1_latent_delta_ci.csv", index=False,
                                 encoding="utf-8-sig")
    print(pivot.round(4).to_string())
    return {"pivot": pivot.round(5).to_dict("index"), "delta_ci": ci_rows}


# ============================================================ C. Q3 QP

def _thr_feature(d):
    spec = EXT_DEFECTS[d]
    if spec["threshold"] is None:
        return None
    return (float(np.log10(spec["threshold"]))
            if spec["scale"] == "log10" else float(spec["threshold"]))


def _bound(d, xs, mode):
    thr = _thr_feature(d)
    if mode == "none":
        return np.inf
    if thr is None:
        return xs
    return max(xs, thr) if mode == "loose" else thr


def _solve_qp(ctx, x_ot, x_src, bounds, sensory_w=None, sensory_base=None,
              sensory_c=None, lam=0.10, margin=5e-3):
    """全空间凸 QP：

        min_x ||x-x_ot||_W^2 + lam||x-x_src||_W^2,  W = Sigma_x^{-1}
        s.t.  化学通路 x_j <= bound_j；
              感官 readout(x) <= readout(x_src) - margin；
              xmin <= x <= xmax（观测范围，硬约束）。

    感官约束的右端必须用**主流程读出的基线**（检出限值先插补成中位数），
    而不是 w^T x_src：Q1 训练时从未见过检出限值，拿原始 x_src 当基线会
    让约束对一个模型不认识的输入分布生效。同时把下界抬到 floor+1e-3，
    避免解恰好落在检出限上再次触发插补跳变（此时线性读出与主流程读出一致）。

    不在降维后的 8 列上做条件均值消元：协方差近奇异时它会把其余 218 列
    推出观测范围，经验分位映射饱和后风格位移失真。226 维全空间直接解，
    SLSQP 单样本约 0.4–0.8 s。
    """
    W = ctx["W_qp"]
    xmin, xmax = ctx["xmin"], ctx["xmax"]
    n = len(x_src)
    base = (x_ot + lam * x_src) / (1.0 + lam)
    f = lambda x: float((x - x_ot) @ W @ (x - x_ot)
                        + lam * (x - x_src) @ W @ (x - x_src))
    g = lambda x: 2.0 * (W @ (x - x_ot) + lam * W @ (x - x_src))

    lo = np.maximum(xmin + 1e-3, ctx["nd_lo"])
    hi = np.minimum(xmax, ctx["nd_hi"])
    lo = np.minimum(lo, hi)
    A_chem, b_chem = [], []
    for j, bnd in bounds.items():
        j = int(j)
        if bnd is not None and np.isfinite(bnd):
            row = np.zeros(n)
            row[j] = 1.0
            A_chem.append(row)
            # 化学上界低于"离开检出限"的下界时，放宽 ≤1e-3，避免解落回
            # 检出限触发中位数插补跳变
            b_chem.append(float(max(bnd, lo[j])))
    A_chem = np.asarray(A_chem).reshape(-1, n)
    b_chem = np.asarray(b_chem, float)
    A_sens, b_sens = [], []
    if sensory_w:
        if sensory_base is None:
            raise ValueError("sensory_w 需要配套的 sensory_base")
        cvec = np.zeros(len(sensory_w)) if sensory_c is None else sensory_c
        for w, s0, c0 in zip(sensory_w, sensory_base, cvec):
            A_sens.append(w)
            b_sens.append(float(s0) - margin - float(c0))
    A_sens = np.asarray(A_sens).reshape(-1, n)
    b_sens = np.asarray(b_sens, float)
    box = list(zip(lo, hi))

    lp = linprog(np.zeros(n), A_ub=np.vstack([A_chem, A_sens]),
                 b_ub=np.r_[b_chem, b_sens], bounds=box, method="highs")
    feasible_sens = bool(lp.success)
    if not lp.success:
        # 感官硬约束与化学上界/观测范围冲突：退回化学约束，如实标记
        lp = linprog(np.zeros(n), A_ub=A_chem, b_ub=b_chem, bounds=box,
                     method="highs")
        if not lp.success:
            return None, False, np.inf
        A_sens, b_sens = np.zeros((0, n)), np.zeros(0)
    A = np.vstack([A_chem, A_sens]) if len(A_sens) else A_chem
    b = np.r_[b_chem, b_sens] if len(b_sens) else b_chem

    x0 = np.clip(base, lo, hi)
    for j, bnd in bounds.items():
        j = int(j)
        if bnd is not None and np.isfinite(bnd):
            x0[j] = min(x0[j], max(float(bnd), lo[j]))
    cons = [{"type": "ineq", "fun": lambda x: b - A @ x,
             "jac": lambda x: -A}] if len(A) else []
    best = None
    starts = [x0] + ([lp.x] if lp.success else [])
    for start in starts:
        r = minimize(f, start, jac=g, method="SLSQP", bounds=box,
                     constraints=cons,
                     options={"ftol": 1e-9, "maxiter": 400})
        ok = bool(np.all(A @ r.x - b <= 1e-5)) if len(A) else True
        if ok and np.all(np.isfinite(r.x)):
            val = f(r.x)
            if best is None or val < best[0]:
                best = (val, r.x)
    if best is None:
        return None, feasible_sens, np.inf
    return best[1], feasible_sens, float(best[0])


def _setup_q3(data):
    X = data.X[data.chem_nd].to_numpy(float)
    Xdf = data.X[data.chem_nd]
    Y = data.Y[data.targets].to_numpy(float)
    styles = data.styles
    k = R.LATENT_K

    Ns = R._normal_scores_matrix(Xdf, data.chem_nd)
    mu_ns, sd_ns = Ns.mean(0), Ns.std(0, ddof=1)
    sd_ns = np.where(sd_ns > 0, sd_ns, 1.0)
    M = (Ns - mu_ns) / sd_ns
    c0 = data.cdr - data.cdr.mean()
    beta = (M.T @ c0) / max(float(c0 @ c0), 1e-12)
    M = M - np.outer(c0, beta)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    Z = U[:, :k] * S[:k]
    Spool = np.cov(Z, rowvar=False)
    mu = Z.mean(0)
    Si = np.linalg.pinv(np.cov(Z, rowvar=False))
    maha = lambda z: float(np.sqrt((z - mu) @ Si @ (z - mu)))

    raw_tables = R._raw_quantile_tables(data.X, data.chem_nd)
    X_clean = np.where(R._floor_mask(X), np.nan, X)
    xmin = np.nanmin(X_clean, axis=0)
    xmax = np.nanmax(X_clean, axis=0)
    xsd = np.nanstd(X_clean, axis=0, ddof=1)
    xsd = np.where(xsd > 0, xsd, 1.0)
    xmed = np.nanmedian(X_clean, axis=0)
    xmed = np.where(np.isfinite(xmed), xmed, 0.0)
    Zs = np.where(np.isfinite(X_clean), (X_clean - xmed) / xsd, 0.0)
    Sigma = np.cov(Zs, rowvar=False) + 1.0 * np.eye(len(data.chem_nd))
    col = {c: j for j, c in enumerate(data.chem_nd)}

    Xall = data.X[data.chem_all].to_numpy(float)
    A_all, mu_a, sd_a, imp_a = _impute_stats(Xall)
    A_clean = np.clip((A_all - mu_a) / sd_a, -10, 10)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    sens_ridge = Ridge(alpha=10.0).fit(A_clean, Y)

    # 主流程口径：Q3 诊断读出在「检出限先置缺失」的标准化矩阵上训练
    fv_leg = R._floor_values(Xall)
    Araw_leg = np.where(R._floor_mask(Xall, fv_leg), np.nan, Xall)
    imp_leg = SimpleImputer(strategy="median", keep_empty_features=True)
    Araw_leg = imp_leg.fit_transform(Araw_leg)
    mu_leg, sd_leg = Araw_leg.mean(0), Araw_leg.std(0)
    sd_leg = np.where(sd_leg < 1e-8, 1.0, sd_leg)
    A_leg = np.clip((Araw_leg - mu_leg) / sd_leg, -10, 10)
    sens_ridge_leg = Ridge(alpha=10.0).fit(A_leg, Y)
    coef_leg = sens_ridge_leg.coef_
    W_sens = []
    for dim in SENS_CONSTR:
        j = data.targets.index(dim)
        w = np.zeros(len(data.chem_nd))
        for jj, c in enumerate(data.chem_all):
            if c in col:
                w[col[c]] = coef_leg[j, jj] / sd_leg[jj]
        W_sens.append(w)

    nd_in_all = np.array([{c: j for j, c in enumerate(data.chem_all)}[c]
                          for c in data.chem_nd])
    nd_lo = np.maximum(xmin, mu_leg[nd_in_all] - 10.0 * sd_leg[nd_in_all])
    nd_hi = np.minimum(xmax, mu_leg[nd_in_all] + 10.0 * sd_leg[nd_in_all])
    ctx = {"data": data, "X": X, "Y": Y, "styles": styles, "Z": Z,
           "Spool": Spool, "mu_ns": mu_ns, "sd_ns": sd_ns, "Vt": Vt,
           "raw_tables": raw_tables, "X_clean": X_clean, "xmin": xmin,
           "xmax": xmax, "xsd": xsd, "xmed": xmed, "Sigma": Sigma,
           "col": col, "Xall": Xall, "sens_ridge": sens_ridge,
           "W_sens": W_sens, "maha": maha,
           "floor_cols": np.any(R._floor_mask(X), axis=0),
           "W_qp": np.linalg.inv(
               (np.diag(xsd) @ Sigma @ np.diag(xsd))
               + 1e-8 * np.eye(len(xsd))),
           "imp_a": imp_a, "mu_a": mu_a, "sd_a": sd_a,
           "fv_leg": fv_leg, "imp_leg": imp_leg, "mu_leg": mu_leg,
           "sd_leg": sd_leg, "sens_ridge_leg": sens_ridge_leg,
           "nd_in_all": nd_in_all, "nd_lo": nd_lo, "nd_hi": nd_hi}
    # 在"离开检出限 + ±10σ 截断域"内取探针点，此时读出对 x 严格线性
    lo_p = np.maximum(xmin + 1e-3, nd_lo)
    hi_p = np.minimum(xmax, nd_hi)
    lo_p = np.minimum(lo_p, hi_p)
    probe = np.clip(xmed, lo_p, hi_p)
    xf = _to_full(ctx, probe)[None, :]
    braw = np.where(R._floor_mask(xf, fv_leg), np.nan, xf)
    a = np.clip((imp_leg.transform(braw) - mu_leg) / sd_leg, -10, 10)
    pred = sens_ridge_leg.predict(a)[0]
    ctx["W_sens_c"] = np.array([
        pred[data.targets.index(dim)] - W_sens[k] @ probe
        for k, dim in enumerate(SENS_CONSTR)])
    return ctx


def _to_full(ctx, x):
    data = ctx["data"]
    x_full = data.X[data.chem_all].mean().to_numpy(float)
    for j, c in enumerate(data.chem_all):
        if c in ctx["col"]:
            x_full[j] = x[ctx["col"][c]]
    return x_full


def _readout(ctx, x):
    """线性读出：固定插补器，不再把合成谱的检出限值二次置 NaN。

    主流程用 `_impute_scale(Xall, x)` 时会把「恰好等于检出下限」的合成值
    当成缺失再填中位数，单样本可差 2 个感官单位以上；对约束求解必须用
    固定插补器，否则约束梯度与实际读出不一致。
    """
    xf = _to_full(ctx, x)[None, :]
    a = np.clip((ctx["imp_a"].transform(xf) - ctx["mu_a"]) / ctx["sd_a"],
                -10, 10)
    return ctx["sens_ridge"].predict(a)[0]


def _legacy_readout(ctx, x):
    """主流程口径：检出限值按训练规则置缺失并插补中位数。"""
    xf = _to_full(ctx, x)[None, :]
    braw = np.where(R._floor_mask(xf, ctx["fv_leg"]), np.nan, xf)
    a = np.clip((ctx["imp_leg"].transform(braw) - ctx["mu_leg"])
                / ctx["sd_leg"], -10, 10)
    return ctx["sens_ridge_leg"].predict(a)[0]


def _sens_affine(ctx, x):
    """在 x 处把主流程读出写成 w·x + c，并返回 ±10 截断掩码。

    被截断的坐标梯度为 0、贡献固定为系数×±10；未截断坐标保留斜率。
    该仿射形式在掩码不变时对任意 y 精确成立。
    """
    data = ctx["data"]
    xf = _to_full(ctx, x)[None, :]
    braw = np.where(R._floor_mask(xf, ctx["fv_leg"]), np.nan, xf)
    a_raw = (ctx["imp_leg"].transform(braw) - ctx["mu_leg"]) / ctx["sd_leg"]
    a = np.clip(a_raw, -10, 10)
    pred = ctx["sens_ridge_leg"].predict(a)[0]
    coef = ctx["sens_ridge_leg"].coef_
    mask = np.abs(a_raw[0]) > 10
    ws, cs = [], []
    for k, dim in enumerate(SENS_CONSTR):
        jt = data.targets.index(dim)
        w = np.zeros(len(x))
        for jj in ctx["nd_in_all"]:
            if mask[jj]:
                continue
            w[ctx["col"][data.chem_all[jj]]] = coef[jt, jj] / ctx["sd_leg"][jj]
        ws.append(w)
        cs.append(float(pred[jt] - w @ x))
    return ws, np.asarray(cs), mask


def _cond_adjust(ctx, x_base, x_des, def_idx):
    xmed, xsd, Sigma = ctx["xmed"], ctx["xsd"], ctx["Sigma"]
    zb = (x_base - xmed) / xsd
    zd = (x_des - xmed) / xsd
    dd = zd[def_idx] - zb[def_idx]
    Sig_DD_inv = np.linalg.inv(Sigma[np.ix_(def_idx, def_idx)])
    delta = Sigma[:, def_idx] @ (Sig_DD_inv @ dd)
    x_new = xmed + xsd * (zb + delta)
    x_new[def_idx] = x_des[def_idx]
    return x_new


def _legacy_transfer(ctx, x_ot, x_src, mode, method):
    """复刻主流程：4 列逐坐标 min()，可选 Schur 补。"""
    col = ctx["col"]
    base = (x_ot + 0.10 * x_src) / 1.10
    x_des = base.copy()
    for d in BASE_DEFECTS:
        j = col[d]
        b = _bound(d, x_src[j], mode)
        if np.isfinite(b):
            x_des[j] = min(x_des[j], b)
    if method == "ot_cond":
        x = _cond_adjust(ctx, base, x_des,
                         np.array([col[d] for d in BASE_DEFECTS]))
    else:
        x = x_des
    return np.clip(x, ctx["xmin"], ctx["xmax"])


def _eval_transfer(ctx, i, x_src, x, x_ot, dm, zin, method, mode,
                   feasible, sens_before, sens_before_leg):
    data = ctx["data"]
    col = ctx["col"]
    n = len(ctx["X"])
    zns = R._raw_to_ns(x[None, :], ctx["raw_tables"], n)[0]
    zl = ((zns - ctx["mu_ns"]) / ctx["sd_ns"]) @ ctx["Vt"][:R.LATENT_K].T
    sen = _readout(ctx, x)
    sen_leg = _legacy_readout(ctx, x)
    rec = {
        "beer": int(data.X.iloc[i]["beer"]), "method": method, "mode": mode,
        "feasible": bool(feasible),
        "style_shift": float((zl - zin) @ dm / (dm @ dm)),
        "qp_cost": float(np.linalg.norm((x - x_ot) / ctx["xsd"]) /
                         (np.linalg.norm((x_ot - x_src) / ctx["xsd"]) + 1e-12)),
        "maha": ctx["maha"](zl),
    }
    for dim in SENS_CONSTR:
        j = data.targets.index(dim)
        rec[f"sens_{dim}"] = float(sen[j])
        rec[f"delta_{dim}"] = float(sen[j] - sens_before[j])
        rec[f"legacy_delta_{dim}"] = float(sen_leg[j] - sens_before_leg[j])
        rec[f"worsened_clean_{dim}"] = int(sen[j] > sens_before[j] + 1e-6)
        rec[f"worsened_{dim}"] = int(
            sen_leg[j] > sens_before_leg[j] + 1e-6)
    for d in EXT_DEFECTS:
        j = col[d]
        val, before = float(x[j]), float(x_src[j])
        thr = _thr_feature(d)
        rec[f"x_{d}"] = val
        rec[f"above_{d}"] = int(val > thr + 1e-6) if thr is not None else -1
        rec[f"new_above_{d}"] = int(thr is not None and val > thr + 1e-6
                                    and before <= thr + 1e-6)
        rec[f"worsened_{d}"] = int(thr is not None
                                   and val > max(before, thr) + 1e-6)
        rec[f"increased_{d}"] = int(val > before + 1e-6)
    return rec


def _run_pair(ctx, src_mask, tgt_mask, methods, max_src=None):
    Z = ctx["Z"]
    mp, Sp = R._moments(Z, src_mask, ctx["Spool"])
    mt, St = R._moments(Z, tgt_mask, ctx["Spool"])
    dm = mt - mp
    X_clean, col, xmed = ctx["X_clean"], ctx["col"], ctx["xmed"]
    C_ext = [col[d] for d in EXT_DEFECTS]
    rows = []
    src_idx = np.where(src_mask)[0]
    if max_src is not None and len(src_idx) > max_src:
        take = np.linspace(0, len(src_idx) - 1, max_src).round().astype(int)
        src_idx = src_idx[take]
    for i in src_idx:
        zi = Z[i]
        zot = R._gaussian_ot(mp, Sp, mt, St, zi)
        x_ot = R._ns_to_raw(
            ((zot @ ctx["Vt"][:R.LATENT_K]) * ctx["sd_ns"] + ctx["mu_ns"])[None, :],
            ctx["raw_tables"])[0]
        x_src = np.where(np.isfinite(X_clean[i]), X_clean[i], xmed)
        for d in EXT_DEFECTS:
            j = col[d]
            if not np.isfinite(X_clean[i, j]):
                x_src[j] = np.nanmin(X_clean[:, j])
        sens_before = _readout(ctx, x_src)
        sens_before_leg = _legacy_readout(ctx, x_src)
        base_leg = np.array([
            sens_before_leg[ctx["data"].targets.index(dim)]
            for dim in SENS_CONSTR])
        for method, mode in methods:
            if method == "none":
                x = np.clip((x_ot + 0.10 * x_src) / 1.10,
                            ctx["xmin"], ctx["xmax"])
                feas = True
            elif method in ("ot_proj", "ot_cond"):
                x = _legacy_transfer(ctx, x_ot, x_src, mode, method)
                feas = True
            else:
                bd = {col[d]: _bound(d, x_src[col[d]], mode)
                      for d in EXT_DEFECTS}
                x, feas, _ = _solve_qp(
                    ctx, x_ot, x_src, bd, sensory_w=ctx["W_sens"],
                    sensory_base=base_leg, sensory_c=ctx["W_sens_c"])
                if x is None:
                    continue
            rows.append(_eval_transfer(
                ctx, i, x_src, x, x_ot, dm, zi, method, mode, feas,
                sens_before, sens_before_leg))
    return rows


def _summarise(sub):
    out = {"n": len(sub), "feasible_sensory": int(sub.feasible.sum())}
    out["style_shift_median"] = float(sub.style_shift.median())
    out["qp_cost_median"] = float(sub.qp_cost.median())
    for d in EXT_DEFECTS:
        out[f"new_above_{d}"] = int(sub[f"new_above_{d}"].sum())
        out[f"increased_{d}"] = int(sub[f"increased_{d}"].sum())
        out[f"worsened_{d}"] = int(sub[f"worsened_{d}"].sum())
    for dim in SENS_CONSTR:
        out[f"worsened_{dim}"] = int(sub[f"worsened_{dim}"].sum())
        out[f"worsened_clean_{dim}"] = int(
            sub[f"worsened_clean_{dim}"].sum())
        out[f"delta_{dim}_median"] = float(sub[f"delta_{dim}"].median())
        out[f"legacy_delta_{dim}_median"] = float(
            sub[f"legacy_delta_{dim}"].median())
    return out


def q3_qp(data, pairs=False, pair_src=None) -> dict:
    ctx = _setup_q3(data)
    styles = data.styles
    main_methods = [("none", "none"), ("ot_proj", "loose"),
                    ("ot_proj", "strict"), ("ot_cond", "loose"),
                    ("ot_cond", "strict"), ("qp_full", "loose"),
                    ("qp_full", "strict")]
    rows = _run_pair(ctx, styles == "Pils/Lager", styles == "Tripel",
                     main_methods)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "q3_qp_long.csv", index=False, encoding="utf-8-sig")

    # 与主结果交叉核对：ot_proj/ot_cond 的 4 个缺陷列应逐位一致
    recon_max = None
    try:
        ref = pd.read_csv(R.Q3 / "transfer.csv")
        ref = ref[(np.abs(ref["lam"] - 0.10) < 1e-12)
                  & ref["method"].isin(["ot_proj", "ot_cond"])
                  & ref["constraint"].isin(["loose", "strict"])]
        chk = df[df.method.isin(["ot_proj", "ot_cond"])]
        mg = chk.merge(ref, left_on=["beer", "method", "mode"],
                       right_on=["beer", "method", "constraint"],
                       suffixes=("", "_ref"))
        diffs = [float(np.nanmax(np.abs(mg[f"x_{d}"] - mg[f"x_{d}_ref"])))
                 for d in BASE_DEFECTS]
        recon_max = float(np.nanmax(diffs))
    except Exception as exc:
        print(f"  交叉核对跳过：{exc}")

    summ = []
    for (method, mode), sub in df.groupby(["method", "mode"], sort=False):
        summ.append({"method": method, "mode": mode, **_summarise(sub)})
    sm = pd.DataFrame(summ)
    sm.to_csv(OUT / "q3_qp_summary.csv", index=False, encoding="utf-8-sig")
    print(sm.to_string(index=False))

    res = {"summary": summ, "n_pils": int((styles == "Pils/Lager").sum()),
           "reconstruction_max_abs_diff": recon_max}
    if pairs:
        t0 = time.time()
        prow = []
        style_names = sorted(np.unique(styles))
        methods = [("none", "none"), ("ot_proj", "loose"),
                   ("qp_full", "loose")]
        for s in style_names:
            for t in style_names:
                if s == t:
                    continue
                prow.extend(_run_pair(ctx, styles == s, styles == t, methods,
                                      max_src=pair_src))
            print(f"  全风格对：源风格 {s} 完成（{time.time()-t0:.0f}s）",
                  flush=True)
        pdf = pd.DataFrame(prow)
        pdf.to_csv(OUT / "q3_qp_pairs_long.csv", index=False,
                   encoding="utf-8-sig")
        ps = [{"method": m, "mode": mo, **_summarise(sub)}
              for (m, mo), sub in pdf.groupby(["method", "mode"], sort=False)]
        psm = pd.DataFrame(ps)
        psm.to_csv(OUT / "q3_qp_pairs_summary.csv", index=False,
                   encoding="utf-8-sig")
        print(psm.to_string(index=False))
        res["pairs_summary"] = ps
        res["pairs_runtime_s"] = round(time.time() - t0, 1)
    save_json("q3_qp.json", res)
    return res


# ============================================================ D. Q2

def _q2_setup(data):
    import networkx as nx
    cand = [c for c in data.chem_nd if c not in R.Q2_EXCLUDE]
    Xc = data.X[cand].to_numpy(float)
    names = [data.chem_map[c] for c in cand]
    n = len(names)
    C = pd.read_csv(R.RAW / "2.csv", index_col=0).loc[names, names].to_numpy()
    edges = {}
    for i in range(n):
        for j in np.argsort(-np.abs(C[i]))[:11]:
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges[(a, b)] = float(C[i, j])
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (i, j), w in edges.items():
        G.add_edge(i, j, weight=w, abs_weight=abs(w))
    communities = sorted(nx.community.louvain_communities(
        G, weight="abs_weight", seed=R.SEED, resolution=1.0), key=len,
        reverse=True)
    W = np.zeros((n, n))
    for (i, j), w in edges.items():
        W[i, j] = W[j, i] = abs(float(w))
    return {"cand": cand, "Xc": Xc, "Y": data.Y[data.targets].to_numpy(float),
            "names": names, "n": n, "W": W, "edges": edges, "G": G,
            "communities": communities}


def _q2_selection(data, s):
    """复刻 run_q2 的主口径，返回 sel (50,n,B)。"""
    n = s["n"]
    Xc_qc = np.column_stack([s["Xc"], data.cdr])
    Ain_all = R._impute_scale(Xc_qc, Xc_qc)[0]
    A_adj_all = R._center_by_group(Ain_all, data.styles)
    Y_adj = R._center_by_group(s["Y"], data.styles)
    rng = np.random.default_rng(R.SEED)
    style_idx = {g: np.where(data.styles == g)[0]
                 for g in np.unique(data.styles)}
    reps = [np.concatenate([rng.choice(v, size=len(v), replace=True)
                            for v in style_idx.values()])
            for _ in range(R.Q2_BOOTSTRAP)]
    sample_w = np.where(data.floor_heavy, 0.5, 1.0)
    is_aroma = np.array([R._is_aroma_target(t) for t in data.targets])
    vol_mask = np.r_[np.array([c not in R.Q2_NONVOLATILE for c in s["cand"]]),
                     True]
    full_mask = np.ones(n + 1, dtype=bool)
    masks = [vol_mask if a else full_mask for a in is_aroma]
    idx_ar = np.flatnonzero(is_aroma)
    idx_tp = np.flatnonzero(~is_aroma)
    sel_all = np.zeros((len(data.targets), n + 1, len(reps)), dtype=bool)
    coef_all = np.zeros((len(data.targets), n + 1))
    sel_all[idx_ar], coef_all[idx_ar] = R._lasso_select(
        A_adj_all, Y_adj[:, idx_ar], reps,
        masks=[masks[j] for j in idx_ar], weights=sample_w, positive=True)
    sel_all[idx_tp], coef_all[idx_tp] = R._lasso_select(
        Ain_all, s["Y"][:, idx_tp], reps,
        masks=[full_mask] * len(idx_tp), weights=sample_w, positive=True)
    return (sel_all[:, :n, :], coef_all[:, :n], Ain_all, A_adj_all, Y_adj,
            is_aroma)


def _q2_ops(W, gamma=0.20):
    n = W.shape[0]
    d = W.sum(1)
    dd = np.where(d > 0, d, 1.0)
    P = W / dd[:, None]
    Dm = np.diag(1.0 / np.sqrt(dd))
    L_sym = np.eye(n) - Dm @ W @ Dm
    I = np.eye(n)
    return {
        "step": lambda f: (1 - gamma) * f + gamma * (P @ f),
        "laplacian": lambda f: np.linalg.solve(I + gamma * L_sym, f),
        "rwr": lambda f: (1 - gamma) * np.linalg.solve(I - gamma * P, f),
    }


def _family_match(cfam, tcat):
    a, b = str(cfam).strip().lower(), str(tcat).strip().lower()
    if not a or not b or b in ("others", "nan", "none"):
        return "na"
    return "yes" if (b in a or a in b) else "no"


def q2_operator(data, n_boot=None) -> dict:
    if n_boot is not None:
        R.Q2_BOOTSTRAP = int(n_boot)
    s = _q2_setup(data)
    sel, coef, Ain_all, A_adj_all, Y_adj, is_aroma = _q2_selection(data, s)
    np.savez_compressed(OUT / "q2_sel.npz", sel=sel)
    freq = sel.mean(2).T

    corr_raw = pd.concat([pd.DataFrame(Ain_all[:, :-1], columns=s["cand"]),
                          pd.DataFrame(s["Y"], columns=data.targets)], axis=1
                         ).corr(method="spearman").loc[s["cand"], data.targets
                                                      ].to_numpy()
    corr_within = pd.concat([pd.DataFrame(A_adj_all[:, :-1], columns=s["cand"]),
                             pd.DataFrame(Y_adj, columns=data.targets)], axis=1
                            ).corr(method="spearman").loc[s["cand"], data.targets
                                                         ].to_numpy()
    # 与主流程同口径：香气维度用风格内中心化，11 个口味/口感维度用原始矩阵
    corr = np.where(is_aroma[None, :], np.nan_to_num(corr_within),
                    np.nan_to_num(corr_raw))
    n_cand = s["n"]

    def _eligible(j):
        e = np.flatnonzero((coef[j] > 0) & (corr[:, j] > 0))
        if len(e) < 3:
            e = np.flatnonzero(coef[j] > 0)
        return e if len(e) else np.arange(n_cand)

    def _top3(f, j):
        e = _eligible(j)
        order = [int(i) for i in e[np.lexsort((-np.abs(corr[e, j]), -f[e, j]))]]
        if len(order) < 3:
            order += [int(i) for i in np.argsort(-f[:, j])
                      if int(i) not in order and coef[j, i] > 0]
        if len(order) < 3:
            order += [int(i) for i in np.argsort(-f[:, j]) if int(i) not in order]
        return order[:3]

    rng = np.random.default_rng(R.SEED + 1)
    split_pairs = []
    for _ in range(5):
        p = rng.permutation(R.Q2_BOOTSTRAP)
        split_pairs.append((p[:R.Q2_BOOTSTRAP // 2], p[R.Q2_BOOTSTRAP // 2:]))
    fam = R._family_map(data)
    sn = R.load_tables().sensory_names.drop_duplicates("R_name").set_index("R_name")
    tcat = sn.reindex(data.targets)["Category"].fillna("").astype(str).to_dict()

    ops = _q2_ops(s["W"], R.Q2_GRAPH_GAMMA)
    rows, top3_store = [], {}
    for op_name, op in ops.items():
        score = np.column_stack([op(freq[:, j]) for j in range(freq.shape[1])])
        top3, jac_by_target, sf = [], [], []
        for j in range(freq.shape[1]):
            order = np.array(_top3(score, j))
            top3.append(order[:3])
            sf.append(float(freq[order[:3], j].mean()))
            js = []
            e = _eligible(j)
            for h1, h2 in split_pairs:
                f1 = op(sel[j][:, h1].mean(1))
                f2 = op(sel[j][:, h2].mean(1))
                s1 = set(e[np.argsort(-f1[e])[:3]])
                s2 = set(e[np.argsort(-f2[e])[:3]])
                js.append(len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0)
            jac_by_target.append(float(np.median(js)))
        top3_store[op_name] = top3
        yes = no = 0
        for j, picks in enumerate(top3):
            for ii in picks:
                m = _family_match(fam.get(s["cand"][ii], ""),
                                  tcat.get(data.targets[j], ""))
                yes += int(m == "yes")
                no += int(m == "no")
        rows.append({
            "operator": op_name,
            "median_jaccard": float(np.median(jac_by_target)),
            "mean_jaccard": float(np.mean(jac_by_target)),
            "n_targets_jaccard_ge_0.5": int(sum(v >= 0.5
                                                for v in jac_by_target)),
            "mean_selected_freq": float(np.mean(sf)),
            "family_match_rate": float(yes / max(yes + no, 1)),
            "family_match_yes": yes, "family_match_no": no,
        })
    base = top3_store["step"]
    for row in rows:
        t3 = top3_store[row["operator"]]
        row["top3_jaccard_vs_step"] = float(np.mean([
            len(set(a) & set(b)) / len(set(a) | set(b))
            for a, b in zip(base, t3)]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "q2_operator_summary.csv", index=False, encoding="utf-8-sig")
    t3 = pd.DataFrame({op: ["|".join(s["cand"][int(i)] for i in picks[j])
                            for j in range(len(data.targets))]
                       for op, picks in top3_store.items()})
    t3.insert(0, "target", data.targets)
    t3.to_csv(OUT / "q2_operator_top3.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    save_json("q2_operator.json", {"summary": df.to_dict("records"),
                                   "n_boot": R.Q2_BOOTSTRAP,
                                   "gamma": R.Q2_GRAPH_GAMMA})
    return {"summary": df.to_dict("records")}


# ============================================================ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["q1", "q3", "q2", "all"], default="all")
    ap.add_argument("--pairs", action="store_true",
                    help="Q3 额外跑全部 462 个有序风格对")
    ap.add_argument("--pair-src", type=int, default=2,
                    help="每对风格抽样源样本数（默认 2；0 表示全部）")
    ap.add_argument("--n-boot", type=int, default=None,
                    help="Q2 稳定性选择 Bootstrap 次数（默认 300）")
    args = ap.parse_args()

    t0 = time.time()
    R.set_global_seed(R.SEED)
    data = R.load_data()
    res = {"generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    if args.only in ("q1", "all"):
        print("[A] Q1 插补口径")
        res["q1_lod"] = q1_lod(data)
        print("[B] Q1 感官潜状态")
        res["q1_latent"] = q1_latent(data)
    if args.only in ("q3", "all"):
        print("[C] Q3 联合受控 QP")
        res["q3_qp"] = q3_qp(data, pairs=args.pairs,
                             pair_src=(args.pair_src or None))
    if args.only in ("q2", "all"):
        print("[D] Q2 图传播算子")
        res["q2_operator"] = q2_operator(data, n_boot=args.n_boot)
    res["runtime_s"] = round(time.time() - t0, 1)
    prev = OUT / "report.json"
    if prev.exists() and args.only != "all":
        old = json.loads(prev.read_text(encoding="utf-8"))
        old.update(res)
        res = old
    save_json("report.json", res)
    print(f"完成：{OUT}（{res['runtime_s']}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
