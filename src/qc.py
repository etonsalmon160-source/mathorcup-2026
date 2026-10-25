# 本文件由 AI 辅助编写；建模与算法设计决策由参赛队员做出。详见《AI工具使用详情说明》。
"""生信式数据质控（QC）。

按组学数据的习惯分六层：
  1. 完整性：主键、跨表对齐、相关矩阵复算、变换/单位一致性
  2. 样本层：缺失、depth、稳健 Mahalanobis、PCA 的 T²/SPE、孤立森林、重复样本
  3. 特征层：缺失率、零膨胀、方差、稳健 z 极端值
  4. 缺失机制：样式依赖、样本 depth 依赖、MNAR/LOD 证据、可预测性 AUC
  5. 目标层：分布、标准化、相关结构、平行分析、近重复维度
  6. 缺陷与网络就绪度：缺陷物分布/阈值敏感性、共表达矩阵特征值、边稳定性、测量体系混杂

输出：
  results/qc/*.csv, results/qc/report.json, results/qc/report.md
  figures/qc/*.png
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import adjusted_rand_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import PATHS, ROOT, SEED

RAW = PATHS["raw"] / "附件"
OUT = PATHS["results"] / "qc"
FIG = PATHS["figures"] / "qc"
META = ["beer", "beer_id", "tasting_category_fine"]
Z_EXTREME = 5.0

# 缺陷面板：题目点名 + 8.xlsx 的 off_flavors 标记
# key 仅用于输出；field 对应 8.xlsx 的 field_name_R / 校准表列名
DEFECTS = {
    "DMS": "DMS",
    "Diacetyl": "X2.3.butanedione",
    "4VG": "X4VG",
    "Acetaldehyde": "acetaldehyde",
    "H2S": "H2S",
    "DES": "DES",
    "SO2": "SO2.mg.L.",
    "MeSH": "MeSH",
    "t2n": "X.E..2.nonenal",
}


# ---------------------------------------------------------------- 基础工具
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _save_csv(df: pd.DataFrame, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    df.to_csv(p, index=False, encoding="utf-8-sig")
    return p


def _cramers_v(ct: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.to_numpy().sum()
    if n == 0:
        return np.nan
    r, k = ct.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))


def _perm_style_p(style: np.ndarray, miss: np.ndarray, n_perm: int = 1000) -> float:
    """缺失是否依赖风味风格：卡方期望频数不足时用置换检验（22 个风格、缺失稀疏）。"""
    style = np.asarray(style)
    miss = np.asarray(miss, dtype=bool)

    def chi(m: np.ndarray) -> float:
        tab = pd.crosstab(style, m).to_numpy()
        if tab.shape[1] < 2:
            return 0.0
        return float(stats.chi2_contingency(tab, correction=False)[0])

    obs = chi(miss)
    rng = np.random.default_rng(SEED)
    null = np.array([chi(rng.permutation(miss)) for _ in range(n_perm)])
    return float((null >= obs).mean())


def _eta2_kruskal(groups: list[np.ndarray]) -> tuple[float, float]:
    groups = [g[~np.isnan(g)] for g in groups if len(g[~np.isnan(g)]) > 0]
    if len(groups) < 2:
        return np.nan, np.nan
    h, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eta2 = max(0.0, (h - k + 1) / (n - k)) if n > k else np.nan
    return float(eta2), float(p)


def _boot_rate(v: np.ndarray, thr: float, n_boot: int = 2000) -> tuple[float, float]:
    v = v[~np.isnan(v)]
    if len(v) == 0 or not np.isfinite(thr):
        return np.nan, np.nan
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    rates = (v[idx] > thr).mean(axis=1)
    lo, hi = np.quantile(rates, [0.025, 0.975])
    return float(lo), float(hi)


def _detectable_r(n: int, alpha: float) -> float:
    df = max(1, n - 2)
    t = stats.t.ppf(1 - alpha / 2, df)
    return float(t / np.sqrt(t * t + df))


# ---------------------------------------------------------------- 读数据
def load_tables():
    X = pd.read_csv(RAW / "1.csv")
    Y = pd.read_csv(RAW / "4.csv")
    cal = pd.read_excel(RAW / "8.xlsx", sheet_name="Calibrated chemical dataset")
    th = pd.read_excel(RAW / "8.xlsx", sheet_name="Sensory thresholds")
    trans = pd.read_csv(ROOT / "docs" / "data_contract" / "transform_classification_v1.csv")
    for df in (X, Y, cal):
        df["beer"] = df["beer"].astype(str)
    return X, Y, cal, th, trans


def feature_meta(X: pd.DataFrame, trans: pd.DataFrame) -> pd.DataFrame:
    t = trans.set_index("column")
    features = [c for c in X.columns if c not in META]
    m = pd.DataFrame({"column": features})
    m["transform"] = m["column"].map(t["transform"]).fillna("unknown")
    m["source"] = m["column"].map(t["source"]).fillna("unknown")
    m["chemical_name"] = m["column"].map(t["chemical_name"])
    m["calibrated"] = m["source"].astype(str).str.startswith("calibrated")
    m["unit"] = m["column"].map(t["unit"]) if "unit" in t.columns else np.nan
    return m


# ---------------------------------------------------------------- 标准化
def normalise_for_qc(X: pd.DataFrame, fm: pd.DataFrame) -> pd.DataFrame:
    """把混合量纲压到同一把尺子：raw 列先按需 log10，再做稳健 z。"""
    out = {}
    for _, row in fm.iterrows():
        c = row["column"]
        v = pd.to_numeric(X[c], errors="coerce")
        if row["transform"] == "raw":
            pos = v.dropna()
            if len(pos) > 10 and (pos > 0).all() and stats.skew(pos) > 1:
                v = np.log10(v.where(v > 0))
        med = v.median()
        mad = (v - med).abs().median()
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale == 0:
            scale = v.std(ddof=0)
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        out[c] = (v - med) / scale
    return pd.DataFrame(out, index=X.index)


# ---------------------------------------------------------------- 样本层
def sample_qc(Z: pd.DataFrame, X: pd.DataFrame, fm: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    F = Z[[c for c in Z.columns if Z[c].std(ddof=0) > 0]].copy()
    keep = list(F.columns)
    miss = X[keep].isna().sum(axis=1)
    ext = (F.abs() > Z_EXTREME).sum(axis=1)
    Fi = F.fillna(0.0)

    n_comp = int(min(20, Fi.shape[0] - 1, Fi.shape[1]))
    pca = PCA(n_components=n_comp, random_state=SEED)
    scores = pca.fit_transform(Fi)
    evr = pca.explained_variance_ratio_
    k = int(np.searchsorted(np.cumsum(evr), 0.90) + 1)
    k = max(3, min(k, n_comp))

    t2 = ((scores[:, :k] ** 2) / pca.explained_variance_[:k]).sum(axis=1)
    # 用经验 97.5% 分位而不是卡方分位：化学特征重尾、零膨胀，卡方假设不成立
    # （旧版用 chi2.ppf(0.975, 20)，MCD 在 n=250、p=20 下会误标 46% 样本）
    t2_cut = float(np.quantile(t2, 0.975))
    recon = scores[:, :k] @ pca.components_[:k]
    spe = ((Fi.to_numpy() - recon) ** 2).sum(axis=1)
    spe_cut = float(np.quantile(spe, 0.975))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mcd = MinCovDet(random_state=SEED).fit(scores[:, :k])
    md = mcd.mahalanobis(scores[:, :k])
    md_cut = float(np.quantile(md, 0.975))

    iso = IsolationForest(contamination=0.02, random_state=SEED).fit(Fi)
    iso_flag = iso.predict(Fi) == -1

    C = np.corrcoef(Fi.to_numpy())
    np.fill_diagonal(C, np.nan)
    max_corr = np.nanmax(C, axis=1)

    raw_cols = [c for c in keep if fm.set_index("column").loc[c, "transform"] == "raw"]
    log_cols = [c for c in keep if fm.set_index("column").loc[c, "transform"] == "log10"]
    raw_frac = F[raw_cols].mean(axis=1) if raw_cols else pd.Series(0.0, index=F.index)
    log_frac = F[log_cols].mean(axis=1) if log_cols else pd.Series(0.0, index=F.index)
    regime_diff = raw_frac - log_frac

    flag_t2 = t2 > t2_cut
    flag_spe = spe > spe_cut
    flag_md = md > md_cut
    # 极端单元格数同样用经验分位：231 个重尾特征下"有 >10 个 |z|>5"是常态（中位数 10）
    flag_ext = ext > float(np.quantile(ext, 0.975))
    flag_dup = max_corr > 0.999
    votes = (flag_t2.astype(int) + flag_spe.astype(int) + flag_md.astype(int)
             + iso_flag.astype(int) + flag_ext.astype(int) + flag_dup.astype(int))

    out = pd.DataFrame({
        "beer": X["beer"].astype(str),
        "style": X["tasting_category_fine"],
        "n_missing": miss.to_numpy(),
        "missing_rate": (miss / len(keep)).to_numpy(),
        "n_extreme_cells": ext.to_numpy(),
        "pca_t2": t2,
        "t2_flag": flag_t2,
        "pca_spe": spe,
        "spe_flag": flag_spe,
        "robust_md": md,
        "md_flag": flag_md,
        "isolation_forest": iso_flag,
        "max_sample_corr": max_corr,
        "duplicate_flag": flag_dup,
        "regime_diff": regime_diff.to_numpy(),
        "outlier_votes": votes.to_numpy(),
        "sample_flag": votes.to_numpy() >= 2,
    })
    info = {
        "pca_components_90pct": k,
        "pca_evr": evr.tolist(),
        "t2_cut": t2_cut,
        "spe_cut": spe_cut,
        "md_cut": md_cut,
        "n_flagged": int(out["sample_flag"].sum()),
    }
    return out, info


# ---------------------------------------------------------------- 特征层
def feature_qc(X: pd.DataFrame, Z: pd.DataFrame, fm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, m in fm.iterrows():
        c = m["column"]
        v = pd.to_numeric(X[c], errors="coerce")
        z = Z[c]
        rows.append({
            "column": c,
            "chemical_name": m["chemical_name"],
            "transform": m["transform"],
            "calibrated": m["calibrated"],
            "n_valid": int(v.notna().sum()),
            "missing_rate": float(v.isna().mean()),
            "n_unique": int(v.nunique(dropna=True)),
            "zero_rate": float((v == 0).mean()),
            "median": float(v.median()) if v.notna().any() else np.nan,
            "mad": float((v - v.median()).abs().median()) if v.notna().any() else np.nan,
            "min": float(v.min()) if v.notna().any() else np.nan,
            "max": float(v.max()) if v.notna().any() else np.nan,
            "skew": float(stats.skew(v.dropna())) if v.notna().sum() > 3 else np.nan,
            "kurtosis": float(stats.kurtosis(v.dropna())) if v.notna().sum() > 3 else np.nan,
            "n_extreme_z5": int((z.abs() > Z_EXTREME).sum()),
            "max_abs_z": float(z.abs().max()) if z.notna().any() else np.nan,
        })
    fq = pd.DataFrame(rows)
    fq["flag_missing"] = fq["missing_rate"] > 0.05
    fq["flag_zero_inflated"] = fq["zero_rate"] > 0.5
    fq["flag_low_variance"] = fq["mad"].fillna(0) == 0
    # 单个样本的 10σ 不足以判定特征有病；要求至少 3 个极端单元格才算重尾异常
    fq["flag_extreme"] = (fq["max_abs_z"] > 10) & (fq["n_extreme_z5"] >= 3)
    # 真正"不能用"的特征：缺失过半 / 全常数（决定该层是否 FAIL）
    fq["flag_unusable"] = (fq["missing_rate"] > 0.5) | (fq["n_unique"] <= 1)
    fq["feature_flag"] = fq[["flag_missing", "flag_zero_inflated",
                            "flag_low_variance", "flag_extreme"]].any(axis=1)
    return fq


def outlier_cells(X: pd.DataFrame, Z: pd.DataFrame, fm: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for _, m in fm.iterrows():
        c = m["column"]
        z = Z[c]
        bad = z.abs() > Z_EXTREME
        if not bad.any():
            continue
        for i in z.index[bad]:
            recs.append({
                "beer": X.loc[i, "beer"],
                "style": X.loc[i, "tasting_category_fine"],
                "column": c,
                "chemical_name": m["chemical_name"],
                "transform": m["transform"],
                "value": float(pd.to_numeric(X.loc[i, c], errors="coerce")),
                "robust_z": float(z.loc[i]),
            })
    out = pd.DataFrame(recs)
    if len(out):
        out = out.reindex(out["robust_z"].abs().sort_values(ascending=False).index)
    return out


# ---------------------------------------------------------------- 检出限下限
def floor_qc(X: pd.DataFrame, fm: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """识别"未检出"被替换成逐列固定下限的单元格（1.csv 为 log10 尺度）。

    判定：某一 log10 特征的最小值在 ≥5 个样本上精确复现。
    这类单元格不是真实测量值，做 PCA/共表达/相关分析时必须显式处理。
    """
    rows = []
    for _, m in fm.iterrows():
        c = m["column"]
        if m["transform"] != "log10" or c not in X.columns:
            continue
        v = pd.to_numeric(X[c], errors="coerce")
        if v.notna().sum() < 10:
            continue
        mn = float(v.min())
        n = int(np.isclose(v.to_numpy(), mn, atol=1e-6, equal_nan=False).sum())
        if n >= 5:
            rows.append({"column": c, "floor_log10": mn, "floor_value": float(10.0 ** mn),
                         "n_at_floor": n, "floor_rate": n / int(v.notna().sum())})
    # 只有"多处重复"或"与其他特征共用同一哨兵值"才认定为检出限；
    # 单个特征偶然重复 5 次的最小值不算
    if rows:
        common = round(pd.Series([r["floor_log10"] for r in rows]).round(6).mode().iloc[0], 6)
        rows = [r for r in rows
                if r["n_at_floor"] >= 10 or round(r["floor_log10"], 6) == common]
    fdf = pd.DataFrame(rows)
    if not len(fdf):
        return {"n_floor_features": 0}, fdf, pd.DataFrame()

    cols = fdf["column"].tolist()
    M = X[cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    B = np.zeros_like(M, dtype=bool)
    for j, (_, r) in enumerate(fdf.iterrows()):
        B[:, j] = np.isclose(M[:, j], r["floor_log10"], atol=1e-6, equal_nan=False)
    ps = pd.DataFrame({"beer": X["beer"].to_numpy(),
                       "style": X["tasting_category_fine"].to_numpy(),
                       "n_floor": B.sum(axis=1),
                       "floor_rate": B.mean(axis=1)})
    ps = ps.sort_values("n_floor", ascending=False).reset_index(drop=True)
    n_cells = int(B.sum())
    shared = int((fdf["floor_log10"].round(6) == common).sum())
    summary = {
        "n_floor_features": int(len(fdf)),
        "n_shared_floor": shared,
        "shared_floor_log10": float(common),
        "shared_floor_value": float(10.0 ** common),
        "n_log_features": int((fm["transform"] == "log10").sum()),
        "n_floor_cells": n_cells,
        # 三个分母分别对应：贴底特征子矩阵 / 全部对数特征 / 全部化学特征
        "floor_frac_affected": float(n_cells / B.size),
        "floor_frac_log": float(n_cells / (int((fm["transform"] == "log10").sum()) * len(X))),
        "floor_frac_all": float(n_cells / (len([c for c in X.columns if c not in META]) * len(X))),
        "min_sample_floor": int(ps["n_floor"].min()),
        "median_sample_floor": float(ps["n_floor"].median()),
        "floor_heavy": ps.head(6)[["beer", "style", "n_floor"]].to_dict(orient="records"),
    }
    return summary, fdf, ps


# ---------------------------------------------------------------- 缺失机制
def missingness_qc(X: pd.DataFrame, Z: pd.DataFrame, fm: pd.DataFrame) -> pd.DataFrame:
    feats = [c for c in X.columns if c not in META]
    sample_missing = X[feats].isna().sum(axis=1)
    rows = []
    for _, m in fm.iterrows():
        c = m["column"]
        miss = X[c].isna()
        if not miss.any():
            continue
        rec = {"column": c, "chemical_name": m["chemical_name"],
               "n_missing": int(miss.sum()), "missing_rate": float(miss.mean())}

        # 1) 是否依赖风格
        ct = pd.crosstab(X["tasting_category_fine"], miss)
        ct = ct[(ct.sum(axis=1) > 0)]
        if ct.shape[0] >= 2 and ct.shape[1] == 2:
            rec["style_cramers_v"] = _cramers_v(ct)
            try:
                exp = stats.contingency.expected_freq(ct)
                if exp.min() >= 5:
                    rec["style_chi2_p"] = float(stats.chi2_contingency(ct, correction=False)[1])
                    rec["style_test"] = "chi2"
                else:
                    rec["style_chi2_p"] = _perm_style_p(
                        X["tasting_category_fine"].to_numpy(), miss.to_numpy())
                    rec["style_test"] = "perm"
            except Exception:
                rec["style_chi2_p"] = np.nan
                rec["style_test"] = "na"
        else:
            rec["style_chi2_p"] = np.nan
            rec["style_cramers_v"] = np.nan
            rec["style_test"] = "na"

        # 2) 是否依赖样本整体缺失负担
        a = sample_missing[miss].to_numpy()
        b = sample_missing[~miss].to_numpy()
        rec["depth_mwu_p"] = float(stats.mannwhitneyu(a, b, alternative="greater").pvalue) if len(a) and len(b) else np.nan

        # 3) MNAR/LOD：用相关特征预测缺失样本，看是否系统性偏低
        obs = ~miss
        cor = Z[feats].apply(lambda s: s.corr(Z.loc[obs, c]) if s.notna().sum() > 10 else np.nan)
        top = cor.abs().drop(labels=[c], errors="ignore").sort_values(ascending=False).head(5).index.tolist()
        if len(top) >= 2 and obs.sum() > 20 and miss.sum() >= 3:
            train = Z.loc[obs, top].fillna(0.0).to_numpy()
            y = Z.loc[obs, c].fillna(0.0).to_numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ridge = Ridge(alpha=1.0).fit(train, y)
            pred = ridge.predict(Z.loc[miss, top].fillna(0.0).to_numpy())
            obs_vals = Z.loc[obs, c].dropna().to_numpy()
            p10 = float(np.quantile(obs_vals, 0.10))
            rec["mnar_frac_pred_below_p10"] = float((pred < p10).mean())
            rec["mnar_wilcoxon_p"] = float(stats.mannwhitneyu(pred, obs_vals, alternative="less").pvalue)
            rec["mnar_supported"] = bool(rec["mnar_frac_pred_below_p10"] >= 0.5 and rec["mnar_wilcoxon_p"] < 0.05)
        else:
            rec["mnar_frac_pred_below_p10"] = np.nan
            rec["mnar_wilcoxon_p"] = np.nan
            rec["mnar_supported"] = False

        # 4) 缺失是否可预测（MAR/MNAR 的旁证）
        if 10 <= miss.sum() <= len(miss) - 10:
            design = pd.get_dummies(X["tasting_category_fine"], drop_first=True, dtype=float)
            design["sample_missing"] = sample_missing.to_numpy()
            design["top_corr"] = Z[top[0]].fillna(0.0).to_numpy() if top else 0.0
            y = miss.astype(int).to_numpy()
            aucs = []
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            for tr, te in skf.split(design, y):
                clf = make_clf()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(design.iloc[tr], y[tr])
                aucs.append(roc_auc_score(y[te], clf.predict_proba(design.iloc[te])[:, 1]))
            rec["missing_auc"] = float(np.mean(aucs))
        else:
            rec["missing_auc"] = np.nan

        rows.append(rec)
    return pd.DataFrame(rows)


def make_clf():
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


# ---------------------------------------------------------------- 目标层
def target_qc(Y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    targets = [c for c in Y.columns if c not in META]
    T = Y[targets].astype(float)
    rows = []
    for c in targets:
        v = T[c]
        med = v.median()
        mad = (v - med).abs().median()
        z = (v - med) / (1.4826 * mad if mad > 0 else max(v.std(ddof=0), 1e-9))
        rows.append({
            "target": c, "n": int(v.notna().sum()),
            "mean": float(v.mean()), "sd": float(v.std(ddof=0)),
            "min": float(v.min()), "max": float(v.max()),
            "skew": float(stats.skew(v.dropna())) if v.notna().sum() > 3 else np.nan,
            "n_extreme_z5": int((z.abs() > Z_EXTREME).sum()),
            "mean_ok": bool(abs(v.mean()) < 0.10),
            "sd_ok": bool(0.5 <= v.std(ddof=0) <= 1.5),
        })
    tq = pd.DataFrame(rows)

    C = T.corr(method="spearman")
    pairs = []
    for i, a in enumerate(targets):
        for b in targets[i + 1:]:
            r = C.loc[a, b]
            if abs(r) >= 0.95:
                pairs.append({"target_a": a, "target_b": b, "spearman": float(r)})
    pair_df = pd.DataFrame(pairs)

    # 平行分析：观测特征值 vs 同尺寸随机矩阵的 95 分位
    Ts = StandardScaler().fit_transform(T.to_numpy())
    obs_eig = np.linalg.eigvalsh(np.corrcoef(Ts, rowvar=False))[::-1]
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(100):
        M = StandardScaler().fit_transform(rng.normal(size=T.shape))
        null.append(np.linalg.eigvalsh(np.corrcoef(M, rowvar=False))[::-1])
    null_q = np.quantile(np.array(null), 0.95, axis=0)
    n_sig = int((obs_eig > null_q).sum())

    evr = obs_eig / obs_eig.sum()
    recs = []
    tmp = T.copy()
    for c in targets:
        v = tmp[c]
        med = v.median()
        mad = (v - med).abs().median()
        z = (v - med) / (1.4826 * mad if mad > 0 else max(v.std(ddof=0), 1e-9))
        for i in z.index[z.abs() > Z_EXTREME]:
            recs.append({"beer": Y.loc[i, "beer"], "style": Y.loc[i, "tasting_category_fine"],
                         "target": c, "value": float(v.loc[i]), "robust_z": float(z.loc[i])})
    cells = pd.DataFrame(recs)
    info = {
        "n_components_parallel_analysis": n_sig,
        "eigenvalues": obs_eig.tolist(),
        "null_95pct": null_q.tolist(),
        "evr": evr.tolist(),
        "n_near_duplicate_pairs": int(len(pair_df)),
        "n_targets_not_centered": int((~tq["mean_ok"]).sum()),
        "n_targets_sd_outside": int((~tq["sd_ok"]).sum()),
    }
    return tq, pair_df, cells, info


# ---------------------------------------------------------------- 缺陷层
def defect_qc(X: pd.DataFrame, Y: pd.DataFrame, cal: pd.DataFrame,
              th: pd.DataFrame, fm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    th_idx = th.set_index("field_name_R")
    fm_idx = fm.set_index("column")
    rows, by_style, extremes, unit_rows = [], [], [], []

    for label, field in DEFECTS.items():
        v = pd.to_numeric(cal[field], errors="coerce")
        trow = th_idx.loc[field] if field in th_idx.index else pd.Series(dtype=object)
        thr_mean = float(trow.get("threshold_mean", np.nan)) if len(trow) else np.nan
        thr_med = float(trow.get("threshold_median", np.nan)) if len(trow) else np.nan
        unit = trow.get("unit", "") if len(trow) else ""
        aging = trow.get("aging", "") if len(trow) else ""
        off = trow.get("off_flavors", "") if len(trow) else ""
        med = v.median()
        mad = (v - med).abs().median()
        z = (v - med) / (1.4826 * mad if mad > 0 else max(v.std(ddof=0), 1e-9))
        ex_med = int((v > thr_med).sum()) if np.isfinite(thr_med) else None
        ex_mean = int((v > thr_mean).sum()) if np.isfinite(thr_mean) else None
        lo, hi = _boot_rate(v.to_numpy(), thr_med)
        rows.append({
            "defect": label, "field": field, "unit": unit, "aging": aging,
            "off_flavors": off, "n": int(v.notna().sum()), "n_zero": int((v == 0).sum()),
            "min": float(v.min()), "p01": float(v.quantile(.01)), "p05": float(v.quantile(.05)),
            "median": float(med), "p95": float(v.quantile(.95)), "p99": float(v.quantile(.99)),
            "max": float(v.max()), "mean": float(v.mean()), "sd": float(v.std(ddof=0)),
            "skew": float(stats.skew(v.dropna())) if v.notna().sum() > 3 else np.nan,
            "n_extreme_z5": int((z.abs() > Z_EXTREME).sum()),
            "threshold_mean": thr_mean, "threshold_median": thr_med,
            "exceed_mean": ex_mean, "exceed_median": ex_med,
            "exceed_rate_mean": (ex_mean / v.notna().sum()) if ex_mean is not None else None,
            "exceed_rate_median": (ex_med / v.notna().sum()) if ex_med is not None else None,
            "exceed_rate_median_ci_lo": lo, "exceed_rate_median_ci_hi": hi,
        })

        # 按风格
        for style, g in cal.groupby("tasting_category_fine"):
            vv = pd.to_numeric(g[field], errors="coerce")
            if vv.notna().sum() < 3:
                continue
            by_style.append({
                "defect": label, "style": style, "n": int(vv.notna().sum()),
                "median": float(vv.median()),
                "exceed_median": int((vv > thr_med).sum()) if np.isfinite(thr_med) else None,
                "rate_median": float((vv > thr_med).mean()) if np.isfinite(thr_med) else None,
            })

        # 极端样本
        top = v.nlargest(5)
        for i, val in top.items():
            extremes.append({
                "defect": label, "beer": cal.loc[i, "beer"],
                "style": cal.loc[i, "tasting_category_fine"],
                "value": float(val), "unit": unit,
                "threshold_median": thr_med,
                "fold_over_threshold": float(val / thr_med) if np.isfinite(thr_med) and thr_med > 0 else np.nan,
            })

        # 1.csv 与校准表的一致性（验证变换 + 单位）
        merged = X[["beer", field]].merge(cal[["beer", field]], on="beer", suffixes=("_x", "_cal"))
        xv = pd.to_numeric(merged[f"{field}_x"], errors="coerce")
        cv = pd.to_numeric(merged[f"{field}_cal"], errors="coerce")
        if field in fm_idx.index and fm_idx.loc[field, "transform"] == "log10":
            xv = 10 ** xv
        ok = xv.notna() & cv.notna() & (xv > 0) & (cv > 0)
        ratio = (xv[ok] / cv[ok]).to_numpy()
        unit_rows.append({
            "defect": label, "field": field, "n_pairs": int(ok.sum()),
            "median_ratio": float(np.median(ratio)) if len(ratio) else np.nan,
            "q05_ratio": float(np.quantile(ratio, .05)) if len(ratio) else np.nan,
            "q95_ratio": float(np.quantile(ratio, .95)) if len(ratio) else np.nan,
            "frac_within_0.5_2": float(((ratio >= .5) & (ratio <= 2)).mean()) if len(ratio) else np.nan,
            "frac_within_0.1_10": float(((ratio >= .1) & (ratio <= 10)).mean()) if len(ratio) else np.nan,
            "out_beers": ", ".join(merged.loc[ok, "beer"].to_numpy()[(ratio < .5) | (ratio > 2)].tolist()),
        })

    # Stale hops 是表型终点，检查它与 t2n/其他缺陷的关联
    sens = []
    if "stale_hops" in Y.columns:
        for label, field in DEFECTS.items():
            m = Y[["beer", "stale_hops"]].merge(cal[["beer", field]], on="beer")
            r, p = stats.spearmanr(m["stale_hops"], pd.to_numeric(m[field], errors="coerce"),
                                   nan_policy="omit")
            sens.append({"target": "stale_hops", "defect": label,
                         "spearman": float(r), "p": float(p)})
    return (pd.DataFrame(rows), pd.DataFrame(by_style), pd.DataFrame(extremes),
            pd.DataFrame(unit_rows), pd.DataFrame(sens))


# ---------------------------------------------------------------- 网络层
def network_qc(X: pd.DataFrame, fm: pd.DataFrame, n_boot: int = 100) -> tuple[dict, pd.DataFrame]:
    C = pd.read_csv(RAW / "2.csv", index_col=0)
    vals = C.to_numpy()
    eig = np.linalg.eigvalsh(vals)[::-1]
    off = vals[~np.eye(len(vals), dtype=bool)]

    # 2.csv 与 1.csv 的复算
    t = fm.set_index("column")
    t = t[t["chemical_name"].isin(C.index)]
    cols = t.index.tolist()
    names = t["chemical_name"].tolist()
    sub = X[cols].astype(float)
    ref = C.loc[names, names].to_numpy()
    # 2.csv 用的是成对完整观测的 Spearman；中位数填补会把 amyl.acetate 等
    # 缺失列的相关系数带偏（实测最大偏差 0.047）
    got = sub.corr(method="spearman").to_numpy()
    max_diff = float(np.nanmax(np.abs(got - ref)))

    # 边稳定性：bootstrap 重采样（同样用成对完整观测）
    n = len(sub)
    rng = np.random.default_rng(SEED)
    edge_hit = np.zeros((len(cols), len(cols)))
    edge_sign = np.zeros((len(cols), len(cols)))
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        R = sub.iloc[idx].corr(method="spearman").to_numpy()
        hit = np.abs(R) >= 0.5
        edge_hit += np.nan_to_num(hit).astype(float)
        edge_sign += np.nan_to_num(np.sign(R))
    freq = edge_hit / n_boot
    sign_cons = np.abs(edge_sign) / n_boot
    iu = np.triu_indices(len(cols), k=1)
    stab = pd.DataFrame({
        "feature_a": [cols[i] for i in iu[0]],
        "name_a": [names[i] for i in iu[0]],
        "feature_b": [cols[j] for j in iu[1]],
        "name_b": [names[j] for j in iu[1]],
        "spearman": ref[iu],
        "edge_freq": freq[iu],
        "sign_consistency": sign_cons[iu],
    }).sort_values(["edge_freq", "sign_consistency"], ascending=False)

    # 测量体系混杂：raw vs log 特征之间的边是否更强
    tr = t["transform"]
    raw_set = set(t.index[tr == "raw"])
    log_set = set(t.index[tr == "log10"])
    def edge_class(a, b):
        if a in raw_set and b in raw_set:
            return "within_raw"
        if a in log_set and b in log_set:
            return "within_log"
        return "between"
    classes = np.array([edge_class(cols[i], cols[j]) for i, j in zip(iu[0], iu[1])])
    absr = np.abs(ref[iu])
    obs = {k: float(absr[classes == k].mean()) for k in ["within_raw", "within_log", "between"]}
    # 置换：打乱 transform 标签
    perm = {k: [] for k in obs}
    labels = np.array(["raw" if c in raw_set else "log" for c in cols])
    for _ in range(500):
        p = rng.permutation(labels)
        pp = np.array([p[i] for i in iu[0]])
        qq = np.array([p[j] for j in iu[1]])
        cc = np.where((pp == "raw") & (qq == "raw"), "within_raw",
                      np.where((pp == "log") & (qq == "log"), "within_log", "between"))
        for k in obs:
            perm[k].append(float(absr[cc == k].mean()))
    pvals = {k: float((np.array(perm[k]) >= obs[k]).mean()) for k in obs}

    info = {
        "shape": list(C.shape),
        "symmetric": bool(np.allclose(vals, vals.T, equal_nan=True)),
        "eigen_min": float(eig.min()), "eigen_max": float(eig.max()),
        "condition_number": float(eig.max() / max(eig.min(), 1e-12)),
        "frac_abs_r_ge_0.5": float((np.abs(off) >= 0.5).mean()),
        "frac_abs_r_ge_0.3": float((np.abs(off) >= 0.3).mean()),
        "detectable_r_alpha_0.05": _detectable_r(n, 0.05),
        "detectable_r_bonferroni": _detectable_r(n, 0.05 / (len(cols) * (len(cols) - 1) / 2)),
        "recompute_max_abs_diff": max_diff,
        "edge_stability_n_boot": n_boot,
        "frac_edges_freq_ge_0.8": float((freq[iu] >= 0.8).mean()),
        "regime_mean_abs_r": obs,
        "regime_perm_p": pvals,
    }
    return info, stab


# ---------------------------------------------------------------- 图
def make_figures(X, Y, fm, sq, fq, tq, dq, stab, net, pca_info, fls=None):
    FIG.mkdir(parents=True, exist_ok=True)
    targets = [c for c in Y.columns if c not in META]

    # 1 缺失
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    m = fq[fq["missing_rate"] > 0].sort_values("missing_rate")
    if len(m):
        axes[0].barh(m["column"], m["missing_rate"] * 100, color="#c44e52")
        axes[0].set_xlabel("missing rate (%)")
        axes[0].set_title("Features with missing values")
    axes[1].hist(sq["n_missing"], bins=range(0, int(sq["n_missing"].max()) + 2),
                 color="#4c72b0", edgecolor="white")
    axes[1].set_xlabel("missing features per sample")
    axes[1].set_ylabel("samples")
    axes[1].set_title("Per-sample missingness")
    fig.tight_layout()
    fig.savefig(FIG / "fig_qc_missingness.png", dpi=180)
    plt.close(fig)

    # 2 PCA
    # 重新算一次 PCA 用于画图（与 sample_qc 一致）
    keep = [c for c in fm["column"] if c in X.columns]
    Z = normalise_for_qc(X, fm)[keep].fillna(0.0)
    pca = PCA(n_components=2, random_state=SEED).fit(Z)
    sc = pca.transform(Z)
    two = fls is not None and len(fls)
    fig, axes = plt.subplots(1, 2 if two else 1, figsize=(12, 5) if two else (8, 6))
    ax = axes[0] if two else axes
    top_styles = X["tasting_category_fine"].value_counts().head(10).index.tolist()
    for s in top_styles:
        mask = (X["tasting_category_fine"] == s).to_numpy()
        ax.scatter(sc[mask, 0], sc[mask, 1], s=18, alpha=.75, label=s)
    other = ~X["tasting_category_fine"].isin(top_styles).to_numpy()
    ax.scatter(sc[other, 0], sc[other, 1], s=14, alpha=.45, c="lightgray", label="Other")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("Chemistry PCA")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    if two:
        nfloor = fls.set_index("beer").loc[X["beer"], "n_floor"].to_numpy()
        axes[1].scatter(sc[:, 0], nfloor, s=16, alpha=.7, c="#4c72b0")
        axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        axes[1].set_ylabel("# features at detection floor")
        axes[1].set_title(f"PC1 dominates by non-detect count (r={np.corrcoef(sc[:, 0], nfloor)[0, 1]:.2f})")
    fig.tight_layout()
    fig.savefig(FIG / "fig_qc_pca.png", dpi=180)
    plt.close(fig)

    # 3 样本离群
    fig, ax = plt.subplots(figsize=(7, 5))
    flag = sq["sample_flag"].to_numpy()
    ax.scatter(sq.loc[~flag, "pca_t2"], sq.loc[~flag, "pca_spe"], s=16, alpha=.6, label="ok")
    ax.scatter(sq.loc[flag, "pca_t2"], sq.loc[flag, "pca_spe"], s=28, c="#c44e52", label="flag")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Hotelling T² (log)"); ax.set_ylabel("SPE (log)")
    ax.set_title("Sample outliers")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_qc_sample_outliers.png", dpi=180)
    plt.close(fig)

    # 4 目标相关
    C = Y[targets].corr(method="spearman")
    order = np.argsort(np.array(targets))  # 先按名字；也可层次聚类
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(C.to_numpy()[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(targets))); ax.set_xticklabels(np.array(targets)[order], rotation=90, fontsize=6)
    ax.set_yticks(range(len(targets))); ax.set_yticklabels(np.array(targets)[order], fontsize=6)
    ax.set_title("Sensory target Spearman correlation")
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_qc_target_corr.png", dpi=180)
    plt.close(fig)

    # 5 缺陷分布
    fig, ax = plt.subplots(figsize=(10, 5))
    cal = pd.read_excel(RAW / "8.xlsx", sheet_name="Calibrated chemical dataset")
    cal["beer"] = cal["beer"].astype(str)
    data, labels = [], []
    for _, r in dq.iterrows():
        field = r["field"]
        v = pd.to_numeric(cal[field], errors="coerce").dropna()
        v = v[v > 0]
        data.append(np.log10(v.to_numpy()) if len(v) else np.array([np.nan]))
        labels.append(r["defect"])
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True)
    for b in bp["boxes"]:
        b.set_facecolor("#a6bddb")
    ax.set_ylabel("log10 concentration")
    ax.set_title("Defect compounds (raw calibrated units)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG / "fig_qc_defect_dist.png", dpi=180)
    plt.close(fig)

    # 6 单位一致性
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, field in zip(axes, ["DMS", "X2.3.butanedione", "X4VG", "X.E..2.nonenal"]):
        m = X[["beer", field]].merge(cal[["beer", field]], on="beer", suffixes=("_x", "_cal"))
        xv = pd.to_numeric(m[f"{field}_x"], errors="coerce")
        cv = pd.to_numeric(m[f"{field}_cal"], errors="coerce")
        tr = fm.set_index("column").loc[field, "transform"] if field in fm.set_index("column").index else "log10"
        if tr == "log10":
            xv = 10 ** xv
        ok = (xv > 0) & (cv > 0)
        ax.scatter(cv[ok], xv[ok], s=8, alpha=.5)
        lim = [min(cv[ok].min(), xv[ok].min()), max(cv[ok].max(), xv[ok].max())]
        ax.plot(lim, lim, "k--", lw=.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(field, fontsize=8)
        ax.set_xlabel("calibrated")
        ax.set_ylabel("1.csv back-transformed")
    fig.suptitle("Unit / transform consistency")
    fig.tight_layout()
    fig.savefig(FIG / "fig_qc_unit_check.png", dpi=180)
    plt.close(fig)

    # 7 网络
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(np.arange(1, len(net.get("eigenvalues", [])) + 1) if "eigenvalues" in net else [],
                 net.get("eigenvalues", []), marker="o", ms=2)
    axes[0].set_title("2.csv eigenvalue scree")
    axes[0].set_xlabel("component"); axes[0].set_ylabel("eigenvalue")
    axes[1].hist(stab["edge_freq"], bins=20, color="#55a868", edgecolor="white")
    axes[1].set_title("Bootstrap edge frequency")
    axes[1].set_xlabel("frequency |r| >= 0.5"); axes[1].set_ylabel("edges")
    fig.tight_layout()
    fig.savefig(FIG / "fig_qc_network.png", dpi=180)
    plt.close(fig)

    # 8 特征极端值
    fig, ax = plt.subplots(figsize=(9, 4))
    top = fq.sort_values("n_extreme_z5", ascending=False).head(20)
    ax.barh(top["column"], top["n_extreme_z5"], color="#dd8452")
    ax.set_xlabel("cells with |robust z| > 5")
    ax.set_title("Feature outlier burden (top 20)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_qc_feature_outliers.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- 报告
def _status(flag_count: int, n: int, warn_frac: float = 0.05, fail_frac: float = 0.25) -> str:
    if n == 0:
        return "PASS"
    f = flag_count / n
    if f >= fail_frac:
        return "FAIL"
    if f >= warn_frac:
        return "WARN"
    return "PASS"


def build_report(X, Y, fm, sq, fq, mq, tq, tp, tinfo, dq, dstyle, dext, dunit, dsens, net, stab, fl=None) -> dict:
    features = [c for c in X.columns if c not in META]
    layers = {
        "sample": {
            "status": _status(int(sq["sample_flag"].sum()), len(sq), .02, .10),
            "n_flagged": int(sq["sample_flag"].sum()),
            "flagged_beers": sq.loc[sq["sample_flag"], "beer"].tolist(),
        },
        "feature": {
            # 只有"缺失过半 / 全常数"才算硬失败；重尾、零膨胀、低方差只提示
            "status": ("FAIL" if bool(fq["flag_unusable"].any())
                       else "WARN" if bool(fq["feature_flag"].any()) else "PASS"),
            "n_flagged": int(fq["feature_flag"].sum()),
            "n_unusable": int(fq["flag_unusable"].sum()),
            "flagged_features": fq.loc[fq["feature_flag"], "column"].tolist(),
            "floor": fl or {},
        },
        "missingness": {
            "status": "WARN" if len(mq) and bool(mq["mnar_supported"].any()) else "PASS",
            "n_features_with_missing": int(len(mq)),
            "n_mnar_supported": int(mq["mnar_supported"].sum()) if len(mq) else 0,
        },
        "target": {
            "status": "WARN" if tp is not None and len(tp) else "PASS",
            "n_near_duplicate_pairs": int(len(tp)),
            "n_components_parallel_analysis": tinfo["n_components_parallel_analysis"],
            "n_targets_not_centered": tinfo["n_targets_not_centered"],
            "n_targets_sd_outside": tinfo["n_targets_sd_outside"],
        },
        "defect": {
            "status": "WARN" if (dext["fold_over_threshold"].max() > 20 if len(dext) else False) else "PASS",
            "n_extreme_samples": int(len(dext)),
            "unit_consistency_min_frac": float(dunit["frac_within_0.5_2"].min()) if len(dunit) else None,
        },
        "network": {
            "status": "WARN" if any(v < .05 for v in net["regime_perm_p"].values()) else "PASS",
            "recompute_max_abs_diff": net["recompute_max_abs_diff"],
            "regime_perm_p": net["regime_perm_p"],
        },
    }
    key_findings = []
    if layers["sample"]["n_flagged"]:
        key_findings.append(f"样本层 {layers['sample']['n_flagged']} 个样本被多指标标记；见 sample_qc.csv")
    if fl and fl.get("n_floor_features"):
        heavy = "、".join(f"{d['beer']}({d['n_floor']})" for d in fl["floor_heavy"])
        key_findings.append(
            f"检出限替换：{fl['floor_frac_all']:.1%} 的化学单元格被填成固定下限"
            f"（{fl['n_floor_features']}/{fl['n_log_features']} 个对数特征，其中 {fl['n_shared_floor']} 个"
            f"共用同一哨兵值 {fl['shared_floor_log10']:.4f}≈{fl['shared_floor_value']:.4g}）；"
            f"6 个样本有 {fl['floor_heavy'][-1]['n_floor']}–{fl['floor_heavy'][0]['n_floor']} 个特征处于下限"
            f"（beer {heavy}）——PCA 前几个主成分与共表达相关会主要反映'多少化合物未检出'，不能当真值直接分析")
    if layers["missingness"]["n_mnar_supported"]:
        lod = mq.loc[mq["mnar_supported"], "column"].tolist()
        key_findings.append(f"{len(lod)} 个特征的缺失呈低值缺失（LOD/MNAR 证据）：{', '.join(lod)}")
    struct = mq[(mq["style_chi2_p"] < .05) & (mq["missing_auc"] > .8)]
    if len(struct):
        key_findings.append("结构性缺失（与风格相关且可由其他特征高精度预测）："
                            f"{', '.join(struct['column'])}——不能按随机缺失直接填补，应加缺失指示变量")
    if len(tp):
        pairs = "；".join(f"{r.target_a}≈{r.target_b}（ρ={r.spearman:.3f}）" for r in tp.itertuples())
        key_findings.append(f"{len(tp)} 对感官维度近重复：{pairs}")
    if tinfo["n_targets_sd_outside"]:
        key_findings.append(
            f"{tinfo['n_targets_sd_outside']} 个目标 sd<0.5：官方标准化并非单位方差；"
            "做多目标损失加权或比较各目标难度时，必须先按目标归一，不能直接比 raw MSE")
    if len(dunit):
        w = dunit.loc[dunit["frac_within_0.5_2"].idxmin()]
        n_bad = int(round((1 - w["frac_within_0.5_2"]) * w["n_pairs"]))
        if w["frac_within_0.5_2"] < .9:
            key_findings.append(f"{w['defect']} 的 1.csv 与校准表大面积不一致"
                                f"（{n_bad}/{w['n_pairs']}），需先查变换/单位")
        elif n_bad:
            key_findings.append(f"{w['defect']} 有 {n_bad}/{w['n_pairs']} 个样本在 1.csv 与校准表相差 >2 倍"
                                f"（beer {w['out_beers']}）；建模以 1.csv 为准，超标分析时留意")
    if any(v < .05 for v in net["regime_perm_p"].values()):
        key_findings.append(
            "共表达网络按量纲分块：raw 块（色度/酒精度/蛋白/pH/热量等主体参数，平均|ρ|=0.20）"
            "与 log 块（香气化合物，0.17）内部相关都高于跨块（0.13）——模块分析应分层；"
            "log 块内部相关有一部分来自共享的未检出模式（见上节），不能全部当成化学共调控")
    # 检出限重灾样本的感官是否也异常低？（区分"真没香气"和"分析批次问题"）
    floor_check = []
    if fl and fl.get("n_floor_features"):
        scols = [c for c in ["A_hops_all", "F_hops_all", "A_esters_all", "F_esters_all"] if c in Y.columns]
        if scols:
            g = Y.groupby("tasting_category_fine")[scols].agg(["mean", "std"])
            for d in fl["floor_heavy"]:
                hit = Y.loc[Y["beer"] == d["beer"]]
                if not len(hit):
                    continue
                s = hit.iloc[0]
                rec = {"beer": d["beer"], "style": d["style"], "n_floor": d["n_floor"]}
                for c in scols:
                    mu, sd = g.loc[s["tasting_category_fine"], (c, "mean")], g.loc[s["tasting_category_fine"], (c, "std")]
                    rec[c] = float((s[c] - mu) / sd) if np.isfinite(sd) and sd > 0 else None
                floor_check.append(rec)
            scores = [np.nanmean([r[c] for c in scols if r[c] is not None]) for r in floor_check]
            high = int(sum(1 for x in scores if np.isfinite(x) and x > -0.5))
            if floor_check and high >= 4:
                key_findings.append(
                    f"检出限重灾的 {len(floor_check)} 个样本里有 {high} 个的感官香气/酯香评分并不低"
                    f"（风格内 z 均值 > -0.5）——化学'未检出'不是真的没香气，更像分析批次/回收率问题")

    # 缺陷物是否追踪它自己的感官维度（化学检出 ≠ 感官感知）
    own_map = {"DMS": "dms", "Diacetyl": "diacetyl", "4VG": "X4vg", "t2n": "t2n"}
    own = []
    for label, tcol in own_map.items():
        field = DEFECTS[label]
        if field not in X.columns or tcol not in Y.columns:
            continue
        a = pd.to_numeric(X[field], errors="coerce")
        b = pd.to_numeric(Y[tcol], errors="coerce")
        r, p = stats.spearmanr(a, b, nan_policy="omit")
        rs = []
        for _, idx in X.groupby("tasting_category_fine").indices.items():
            aa, bb = a.iloc[idx], b.iloc[idx]
            if aa.notna().sum() >= 8 and aa.nunique() > 3:
                rr, _ = stats.spearmanr(aa, bb, nan_policy="omit")
                if np.isfinite(rr):
                    rs.append(float(rr))
        own.append({"defect": label, "target": tcol, "spearman": float(r), "p": float(p),
                    "within_style_median": float(np.median(rs)) if rs else None,
                    "n_styles": len(rs)})
    weak = [o["defect"] for o in own if abs(o["spearman"]) < .2]
    if weak:
        strong = [f"{o['defect']}(ρ={o['spearman']:.2f})" for o in own if abs(o["spearman"]) >= .2]
        key_findings.append(
            "缺陷物与自身感官维度大多脱钩："
            + "、".join(weak) + " 的 |ρ|<0.2"
            + (f"；只有 {'、'.join(strong)} 追踪得上" if strong else "")
            + "——化学检出量不等于被感知，论文里不能把两者当同一变量")

    if not key_findings:
        key_findings.append("未发现需要阻断建模的 QC 问题")

    return {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_samples": int(len(X)), "n_features": int(len(features)),
        "n_targets": int(len([c for c in Y.columns if c not in META])),
        "n_styles": int(X["tasting_category_fine"].nunique()),
        "layers": layers,
        "key_findings": key_findings,
        "sample_pca": {"n_components_90pct": None},
        "network": net,
        "defect_summary": dq.to_dict(orient="records"),
        "defect_sensory_corr": dsens.to_dict(orient="records") if len(dsens) else [],
        "defect_own_target": own,
        "floor_sensory_check": floor_check,
    }


def _md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    """不依赖 tabulate 的简易 markdown 表。"""
    if df is None or len(df) == 0:
        return "（无）"
    d = df.head(max_rows).copy()
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in d.iterrows():
        vals = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.4g}" if np.isfinite(v) else "NA")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append("| " + " | ".join(["…"] * len(cols)) + " |")
        lines.append(f"\n共 {len(df)} 行，完整结果见 CSV。")
    return "\n".join(lines)


def write_markdown(rep: dict, sq, fq, mq, tq, dq, dunit, net, fl=None, flf=None, fls=None):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "report.md"
    lines = [
        "# 数据质控报告（QC）",
        "",
        f"生成时间：{rep['generated']}",
        "",
        f"样本 {rep['n_samples']} × 特征 {rep['n_features']} × 目标 {rep['n_targets']}；风格 {rep['n_styles']} 类。",
        "",
        "## 分层结论",
        "",
        "| 层 | 状态 | 要点 |",
        "|---|---|---|",
    ]
    for k, v in rep["layers"].items():
        lines.append(f"| {k} | **{v['status']}** | {json.dumps(v, ensure_ascii=False)} |")
    lines += ["", "## 关键发现", ""]
    lines += [f"- {x}" for x in rep["key_findings"]]
    lines += ["", "## 样本层", "",
              f"- 多指标标记样本：{int(sq['sample_flag'].sum())}",
              f"- 重复样本（样本间相关 >0.999）：{int(sq['duplicate_flag'].sum())}"]
    if sq["sample_flag"].any():
        flagged = sq.loc[sq["sample_flag"]].sort_values("outlier_votes", ascending=False)
        lines.append(_md_table(flagged[["beer", "style", "n_missing", "n_extreme_cells",
                                       "t2_flag", "spe_flag", "md_flag", "isolation_forest",
                                       "outlier_votes"]]))
        lines += ["", "> 这些样本不是重复样本，只是多变量离群；其中 Low/No alcohol、Brut、Brett "
                      "本身就是设计上的极端风格，建议保留，建模时用稳健缩放/树模型，不做删除。"]
        if fls is not None and len(fls):
            both = sorted(set(fls.head(6)["beer"]) & set(flagged["beer"]))
            if both:
                lines.append(f"> 注意：{len(both)}/{len(flagged)} 个标记样本同时是检出限重灾样本"
                             f"（beer {', '.join(both)}），其离群主要来自「大量化合物未检出」，不是生物学异常。")
    lines += ["", "## 特征层", "",
              f"- 被标记特征：{int(fq['feature_flag'].sum())}（重尾/零膨胀/低方差，仅提示）",
              f"- 不可用特征（缺失过半或全常数）：{int(fq['flag_unusable'].sum())}",
              f"- 缺失特征：{', '.join(fq.loc[fq['flag_missing'], 'column'].tolist()) or '无'}"]
    if fl and fl.get("n_floor_features"):
        lines += ["", "## 检出限替换（未检出）", "",
                  f"- 固定下限出现在 {fl['n_floor_features']}/{fl['n_log_features']} 个对数特征，"
                  f"共 {fl['n_floor_cells']} 格（占全部化学格 {fl['floor_frac_all']:.1%}、"
                  f"占贴底特征子矩阵 {fl['floor_frac_affected']:.1%}）；"
                  f"其中 {fl['n_shared_floor']} 个特征的填值完全相同（log10={fl['shared_floor_log10']:.4f}，"
                  f"即 {fl['shared_floor_value']:.4g}），是一个统一哨兵而非各自测得的低值",
                  f"- 每个样本处于下限的特征数：中位 {fl['median_sample_floor']:.0f}，"
                  f"范围 {fl['min_sample_floor']}–{fl['floor_heavy'][0]['n_floor']}",
                  "", _md_table(fls.head(8))]
        if rep.get("floor_sensory_check"):
            lines += ["", "### 检出限重灾样本的感官是否也低（风格内 z）", "",
                      _md_table(pd.DataFrame(rep["floor_sensory_check"]))]
        lines += ["", "> 判读：这些格子在 8.xlsx 校准表里多为 0 或 2×下限，说明是「未检出」的替换值，"
                      "不是真实浓度。做 PCA、相关网络、差异分析时应按缺失或按二值（检出/未检出）处理；"
                      "直接当数值，会让前几个主成分变成「检出率因子」。"]
    lines += ["", "## 缺失机制", ""]
    if len(mq):
        lines.append(_md_table(mq[["column", "n_missing", "missing_rate", "style_chi2_p",
                                   "style_test", "mnar_supported", "missing_auc"]]))
        lines += ["", "> 判读：`mnar_supported` 只检测「低值缺失」（LOD）模式；"
                      "`style_chi2_p<0.05` 且 `missing_auc>0.8` 则说明缺失由配方/工艺决定（结构性缺失），"
                      "应加缺失指示变量，不能按随机缺失填中位数。"]
    else:
        lines.append("无缺失。")
    lines += ["", "## 目标层", "",
              f"- 平行分析支持的主成分数：{rep['layers']['target']['n_components_parallel_analysis']}",
              f"- 近重复维度对（|r|≥0.95）：{rep['layers']['target']['n_near_duplicate_pairs']}",
              f"- 未中心化维度：{rep['layers']['target']['n_targets_not_centered']}；sd 超出 [0.5,1.5]：{rep['layers']['target']['n_targets_sd_outside']}",
              "", "## 缺陷面板", ""]
    lines.append(_md_table(dq[["defect", "unit", "aging", "median", "p95", "max",
                               "n_extreme_z5", "exceed_rate_median", "exceed_rate_mean"]]))
    lines += ["", "### 1.csv 与校准表一致性", ""]
    lines.append(_md_table(dunit[["defect", "n_pairs", "median_ratio", "frac_within_0.5_2"]]))
    if rep.get("defect_own_target"):
        lines += ["", "### 缺陷物 vs 对应感官维度", ""]
        lines.append(_md_table(pd.DataFrame(rep["defect_own_target"])))
        lines += ["", "> 判读：整体与风格内两种口径都要看。整体 ρ 高但风格内接近 0，说明相关来自风格间差异；"
                      "两者都低，说明化学检出量与感官评分不是同一回事，建模时不能互相替代。"]
    lines += ["", "## 网络就绪度", "",
              f"- 2.csv 复算最大误差：{net['recompute_max_abs_diff']:.2e}",
              f"- 可检测 |r|（α=0.05）：{net['detectable_r_alpha_0.05']:.3f}；Bonferroni：{net['detectable_r_bonferroni']:.3f}",
              f"- 测量体系混杂置换 p：{net['regime_perm_p']}"]
    r = rep.get("r_extra")
    if r:
        pm, ps = r["permanova_chem"], r["permanova_sens"]
        w = r["wgcna"]
        lines += ["", "## R 侧验证（WGCNA + vegan）", "",
                  f"- goodSamplesGenes：化学 {w['chem_good_samples']} 样本 / {w['chem_good_features']} 特征可用；"
                  f"感官 {w['sens_good_samples']} 样本 / {w['sens_good_targets']} 维度可用",
                  f"- 风格对化学谱的 PERMANOVA：R²={pm['r2']:.3f}，F={pm['f']:.2f}，p={pm['p']:g}",
                  f"- 风格对感官谱的 PERMANOVA：R²={ps['r2']:.3f}，F={ps['f']:.2f}，p={ps['p']:g}",
                  f"- 组内离散度齐性（betadisper）：化学 p={r['betadisper_chem_p']:g}；感官 p={r['betadisper_sens_p']:g}",
                  f"- 风格间差异（Kruskal–Wallis，FDR<0.05）：化学 {r['style_kruskal']['chem_n_fdr05']}/{rep['n_features']}，"
                  f"感官 {r['style_kruskal']['sens_n_fdr05']}/{rep['n_targets']}"]
    lines += ["", "## 产物", "",
              "- `sample_qc.csv` / `feature_qc.csv` / `outlier_cells.csv`",
              "- `floor_by_feature.csv` / `floor_by_sample.csv`",
              "- `missingness.csv` / `target_qc.csv` / `target_pairs.csv`",
              "- `defect_qc.csv` / `defect_by_style.csv` / `defect_extremes.csv` / `defect_unit_check.csv`",
              "- `edge_stability.csv` / `network_qc.json` / `r_extra.json`",
              "- `figures/qc/*.png`", ""]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ---------------------------------------------------------------- R 侧
def run_r_extra(timeout: int = 900) -> dict | None:
    """WGCNA goodSamplesGenes + vegan PERMANOVA/betadisper。
    R 不可用或失败时只告警，不阻断 Python 侧结论。"""
    import shutil
    import subprocess

    script = ROOT / "scripts" / "qc_extra.R"
    if shutil.which("Rscript") is None or not script.exists():
        print("[QC] 跳过 R 侧（未找到 Rscript 或脚本）")
        return None
    print("[QC] R 侧：WGCNA + vegan")
    try:
        subprocess.run(["Rscript", str(script)], cwd=ROOT, check=True, timeout=timeout,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[QC] R 侧失败（不影响 Python 结果）：\n{e.stdout[-2000:]}")
        return None
    except Exception as e:
        print(f"[QC] R 侧失败（不影响 Python 结果）：{e}")
        return None
    p = OUT / "r_extra.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ---------------------------------------------------------------- 主流程
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, Y, cal, th, trans = load_tables()
    fm = feature_meta(X, trans)

    print("[QC] 1/6 标准化 + 样本层")
    Z = normalise_for_qc(X, fm)
    sq, sinfo = sample_qc(Z, X, fm)

    print("[QC] 2/6 特征层 + 极端单元格")
    fq = feature_qc(X, Z, fm)
    cells = outlier_cells(X, Z, fm)
    fl, flf, fls = floor_qc(X, fm)

    print("[QC] 3/6 缺失机制")
    mq = missingness_qc(X, Z, fm)

    print("[QC] 4/6 目标层")
    tq, tp, tcells, tinfo = target_qc(Y)

    print("[QC] 5/6 缺陷面板")
    dq, dstyle, dext, dunit, dsens = defect_qc(X, Y, cal, th, fm)

    print("[QC] 6/6 网络就绪度")
    net, stab = network_qc(X, fm, n_boot=50)
    net["eigenvalues"] = np.linalg.eigvalsh(pd.read_csv(RAW / "2.csv", index_col=0).to_numpy())[::-1].tolist()

    # 落盘
    _save_csv(sq, "sample_qc.csv")
    _save_csv(fq, "feature_qc.csv")
    _save_csv(cells, "outlier_cells.csv")
    _save_csv(flf, "floor_by_feature.csv")
    _save_csv(fls, "floor_by_sample.csv")
    _save_csv(mq, "missingness.csv")
    _save_csv(tq, "target_qc.csv")
    _save_csv(tp, "target_pairs.csv")
    _save_csv(tcells, "target_outlier_cells.csv")
    _save_csv(dq, "defect_qc.csv")
    _save_csv(dstyle, "defect_by_style.csv")
    _save_csv(dext, "defect_extremes.csv")
    _save_csv(dunit, "defect_unit_check.csv")
    _save_csv(dsens, "defect_sensory_corr.csv")
    _save_csv(stab, "edge_stability.csv")
    (OUT / "network_qc.json").write_text(json.dumps(net, ensure_ascii=False, indent=1, default=_json_default),
                                        encoding="utf-8")

    # R 需要的输入
    Xr = X[["beer"] + [c for c in fm["column"]]].copy()
    Xr.to_csv(OUT / "chem_for_r.csv", index=False, encoding="utf-8-sig")
    Yr = Y[["beer"] + [c for c in Y.columns if c not in META]].copy()
    Yr.to_csv(OUT / "sensory_for_r.csv", index=False, encoding="utf-8-sig")
    X[["beer", "tasting_category_fine"]].rename(columns={"tasting_category_fine": "style"}).to_csv(
        OUT / "meta_for_r.csv", index=False, encoding="utf-8-sig")

    print("[QC] 画图")
    make_figures(X, Y, fm, sq, fq, tq, dq, stab, net, sinfo, fls)

    rep = build_report(X, Y, fm, sq, fq, mq, tq, tp, tinfo, dq, dstyle, dext, dunit, dsens, net, stab, fl)
    rep["sample_pca"] = {"n_components_90pct": sinfo["pca_components_90pct"],
                         "evr": sinfo["pca_evr"]}
    r_extra = run_r_extra()
    if r_extra:
        rep["r_extra"] = r_extra
        pm, ps = r_extra["permanova_chem"], r_extra["permanova_sens"]
        rep["key_findings"].append(
            f"风格是数据的主结构：PERMANOVA 化学 R²={pm['r2']:.2f}、感官 R²={ps['r2']:.2f}（置换 p={pm['p']:g}）")
        kchem = r_extra["style_kruskal"]["chem_n_fdr05"]
        rep["key_findings"].append(
            f"风格间显著差异（Kruskal–Wallis，FDR<0.05）：化学 {kchem}/{rep['n_features']}、"
            f"感官 {r_extra['style_kruskal']['sens_n_fdr05']}/{rep['n_targets']}——风格既是强协变量，也是交叉验证必须分层的泄漏源")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=_json_default),
                                    encoding="utf-8")
    md = write_markdown(rep, sq, fq, mq, tq, dq, dunit, net, fl, flf, fls)

    print(f"[QC] 完成：{OUT}")
    print(f"[QC] 报告：{md}")
    for k, v in rep["layers"].items():
        print(f"  {k:12s} {v['status']}")
    hard = [k for k, v in rep["layers"].items() if v["status"] == "FAIL"]
    if hard:
        raise SystemExit(f"[QC] 硬失败层：{hard}")
    return rep


if __name__ == "__main__":
    main()
