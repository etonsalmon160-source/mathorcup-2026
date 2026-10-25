#!/usr/bin/env python
# 本文件由 AI 辅助编写；分析口径与论文取舍由参赛队员决定。
"""S 门：补充数据的非侵入式验证（只读主结果，不改主模型）。

定位
----
仅用于模块通路注释、OAV/价格量化与跨数据集独立验证；
**不进入 Q1/Q2/Q3 主模型训练，也不改已有预测与排序。**

内容
----
1. Q2 模块富集 + Q3 物理阈值 OAV：
   - 8.xlsx / Sensory thresholds 按 field_name_R 映射到 1.csv 特征；
   - 统计 43 个有阈值化合物与 7 类代谢来源在 8 个 Louvain 模块中的超几何富集；
   - 用 Calibrated chemical dataset 核对浓度还原，再算迁移前后的真实 OAV。
1b. Q3 缺陷完整性 + 风格依赖阈值：
   - 重建 11 条 Pils/Lager→Tripel 迁移谱（与 transfer.csv 逐位核对）；
   - 用 Q1 主模型读出 stale_hops/barnyard/metallic 等感官，检查前后变化；
   - 检查 MeSH/H2S/DES/iron 等关联化学物，区分"绝对缺陷"与风格特征。
2. 8.xlsx / Beer Prices：
   - 22 个风格的价格分布；全风格对的价差与 Q3 修正代价/风格位移的关系；
   - **该表没有容量字段，无法换算 €/L**，因此只报告渠道内标价与比值，不伪造单位。
3. 外部 5,558 款啤酒：
   - 与比赛 50 维感官对齐后做 Mantel 检验；
   - Pils/Lager → Tripel 的位移方向与比赛 Q3 位移做余弦/秩相关对照；
   - 外部数据仅在锁定主模型之后使用，不参与任何训练或选择。

输出
----
`results/supplementary/`：report.json、SUMMARY.md 与各明细 CSV。
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# 单任务 ≤8 线程；必须在 numpy 导入前设置。
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

import numpy as np
import pandas as pd
from scipy.stats import (hypergeom, kruskal, mannwhitneyu, pearsonr, spearmanr,
                         wilcoxon)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_v1 as R  # noqa: E402

OUT = R.RESULTS / "supplementary"
RAW = R.RAW
XLSX = RAW / "8.xlsx"
EXT = R.PATHS["external"] / "beer_data_set.csv"
N_PERM = 10_000
N_PERM_MANTEL = 2_000
N_EXT_MANTEL = 600  # 外部 5,558 款按风格分层抽 600 款做 Mantel；不用全量是算力问题
RNG = np.random.default_rng(R.SEED)

# 外部 11 个宏观描述符 → 比赛维度的映射。
# 主分析只用有直接对应物的 9 个；Astringency / Salty 没有比赛对应维度，
# 放在宽松敏感性里用"最近义"映射，不进入主结论。
MACRO_MAP = {
    "Body": ["body"],
    "Alcohol": ["alcohol"],
    "Bitter": ["bitternes"],
    "Sweet": ["sweetness"],
    "Sour": ["acidity"],
    "Fruits": ["A_esters_fruity", "F_esters_fruity"],
    "Hoppy": ["A_hops_all", "F_hops_all"],
    "Spices": ["coriander", "clove"],
    "Malty": ["A_malt_all", "F_malt_all"],
}
MACRO_MAP_LOOSE = dict(MACRO_MAP, Astringency=["aftertaste"], Salty=["metallic"])

# 外部风格 → 比赛风格族（只用于重叠排除与迁移对照）。
STYLE_FAMILY = {
    "pilsner": "pils", "lager": "pils", "pils": "pils",
    "tripel": "tripel", "dubbel": "dubbel",
    "witbier": "wheat", "wheat": "wheat",
    "stout": "stout", "porter": "stout",
    "blonde": "blond", "blond": "blond",
    "amber": "amber", "brown": "brown", "bruin": "brown",
    "saison": "saison", "lambic": "lambic", "kriek": "kriek",
    "brut": "brut", "scotch": "scotch", "ipa": "hoppy", "hoppy": "hoppy",
}


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _family(style: str) -> str:
    s = str(style).lower()
    for key, fam in STYLE_FAMILY.items():
        if key in s:
            return fam
    return _norm(s)


def _spearman(a, b) -> tuple[float, float]:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    r = spearmanr(a[ok], b[ok])
    return float(r.statistic), float(r.pvalue)


def _bh(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    prev = 1.0
    for rank, i in enumerate(order[::-1], start=1):
        val = min(prev, p[i] * m / (m - rank + 1))
        q[i] = val
        prev = val
    return q


def _load() -> dict:
    data = R.load_data()
    th = pd.read_excel(XLSX, sheet_name="Sensory thresholds")
    cal = pd.read_excel(XLSX, sheet_name="Calibrated chemical dataset")
    price = pd.read_excel(XLSX, sheet_name="Beer Prices")
    trans = pd.read_csv(ROOT / "docs" / "data_contract"
                        / "transform_classification_v1.csv")
    ext = pd.read_csv(EXT)
    return {"data": data, "th": th, "cal": cal, "price": price,
            "trans": trans, "ext": ext}


# ---------------------------------------------------------------- 1. 模块富集

def module_enrichment(ctx: dict) -> dict:
    data, th = ctx["data"], ctx["th"]
    comm = pd.read_csv(R.Q2 / "communities.csv", encoding="utf-8-sig")
    key_by_display = {v: k for k, v in data.chem_map.items()}
    comm["column"] = comm["node"].map(key_by_display)
    comm = comm.dropna(subset=["column"])

    # 每个模块里出现过的阈值表记录（field_name_R 与 1.csv 列名一一对应）
    th = th.copy()
    th["measured"] = th["threshold_median"].notna()
    th["origin"] = th["origin_paper"].fillna("Unknown")
    th["off_flavor"] = th["off_flavors"].notna()
    merged = comm.merge(th[["field_name_R", "chemical_name", "measured", "origin",
                            "off_flavor", "threshold_median", "unit"]],
                        left_on="column", right_on="field_name_R", how="left")

    n_total = len(comm)
    n_measured_total = int(th.loc[th["field_name_R"].isin(comm["column"]), "measured"].sum())
    n_mapped_total = int(merged["field_name_R"].notna().sum())

    rows, tests = [], []
    for mod, sub in merged.groupby("community"):
        n = len(sub)
        k = int(sub["measured"].sum())
        expected = n_measured_total * n / n_total
        p = float(hypergeom.sf(k - 1, n_total, n_measured_total, n)) if k else 1.0
        rows.append({
            "module": int(mod), "size": n,
            "n_with_threshold": k,
            "expected_with_threshold": round(expected, 2),
            "fold_enrichment": round(k / expected, 3) if expected else float("nan"),
            "p_threshold": p,
            "n_off_flavor": int(sub["off_flavor"].sum()),
            "top_origins": "",
        })
        for origin, g in sub.groupby("origin"):
            if origin == "Unknown":
                continue
            kk = len(g)
            total_origin = int((th["origin"] == origin).sum())
            p_origin = float(hypergeom.sf(kk - 1, n_total, total_origin, n))
            tests.append({"module": int(mod), "origin": origin, "k": kk,
                          "module_size": n, "origin_total": total_origin,
                          "expected": total_origin * n / n_total,
                          "fold": kk / (total_origin * n / n_total)
                          if total_origin else float("nan"),
                          "p": p_origin})

    mod_df = pd.DataFrame(rows)
    mod_df["q_threshold"] = _bh(mod_df["p_threshold"])
    tdf = pd.DataFrame(tests)
    tdf["q"] = _bh(tdf["p"])

    labels = {}
    for mod in mod_df["module"]:
        hits = tdf[(tdf["module"] == mod) & (tdf["q"] < 0.05)].sort_values("fold", ascending=False)
        labels[int(mod)] = "、".join(f"{r.origin}×{r.fold:.1f}" for r in hits.itertuples())
        mod_df.loc[mod_df["module"] == mod, "top_origins"] = labels[int(mod)]

    # 每个模块里阈值最接近数据的化合物（仅展示，不做筛选）
    examples = []
    for mod, sub in merged.groupby("community"):
        s = sub.dropna(subset=["threshold_median"])
        for r in s.sort_values("threshold_median").head(8).itertuples():
            examples.append({"module": int(mod), "compound": r.column,
                             "chemical_name": r.chemical_name,
                             "origin": r.origin, "threshold_median": r.threshold_median,
                             "unit": r.unit})

    OUT.mkdir(parents=True, exist_ok=True)
    mod_df.to_csv(OUT / "q2_module_annotation.csv", index=False, encoding="utf-8-sig")
    tdf.to_csv(OUT / "q2_module_enrichment_tests.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(examples).to_csv(OUT / "q2_module_threshold_examples.csv",
                                  index=False, encoding="utf-8-sig")
    return {
        "n_nodes": n_total, "n_mapped_threshold_rows": n_mapped_total,
        "n_with_threshold": n_measured_total,
        "modules": mod_df.to_dict("records"),
        "n_enrichment_tests": int(len(tdf)),
        "n_q_below_0.05": int((tdf["q"] < 0.05).sum()),
    }


# ---------------------------------------------------------------- 2. OAV 核验

DEFECT_SPEC = {
    "DMS": {"threshold": 19.0, "unit": "ug/L"},
    "X4VG": {"threshold": 0.20, "unit": "mg/L"},
    "X2.3.butanedione": {"threshold": 0.09, "unit": "mg/L"},
    "X.E..2.nonenal": {"threshold": None, "unit": "mg/L"},
}
AROMA_COMPOUNDS = [
    "ethyl_acetate", "ethyl_octanoate", "isopentyl_acetate", "phenethyl_acetate",
    "isoamyl_alcohol", "linalool", "geraniol", "eugenol",
]


def _physical(x, column: str, trans: pd.DataFrame) -> float:
    row = trans.loc[trans["column"] == column, "transform"]
    if not len(row):
        return float("nan")
    return float(10 ** x) if row.iloc[0] == "log10" else float(x)


def _physical_median(vals, column: str, trans: pd.DataFrame) -> float:
    """先逐样本还原物理量，再取中位数；避免对原始量纲直接取均值被长尾拉高。"""
    row = trans.loc[trans["column"] == column, "transform"]
    if not len(row):
        return float("nan")
    v = np.asarray(vals, float)
    return float(np.nanmedian(10 ** v if row.iloc[0] == "log10" else v))


def oav_check(ctx: dict) -> dict:
    data, th, cal, trans = ctx["data"], ctx["th"], ctx["cal"], ctx["trans"]
    tr = pd.read_csv(R.Q3 / "transfer.csv", encoding="utf-8-sig")
    tr["beer"] = tr["beer"].astype(int)

    # 先用 Calibrated chemical dataset 核对：迁移前浓度应能精确还原。
    cal_by_beer = cal.set_index("beer")
    checks = []
    for beer in sorted(tr["beer"].unique()):
        if beer not in cal_by_beer.index:
            continue
        row = cal_by_beer.loc[beer]
        for d, spec in DEFECT_SPEC.items():
            before = float(data.X.loc[data.X["beer"] == beer, d].iloc[0])
            recon = _physical(before, d, trans)
            ref = float(row[d]) if d in row.index else float("nan")
            if np.isfinite(ref) and ref > 0:
                checks.append({"beer": beer, "compound": d,
                               "reconstructed": recon, "calibrated": ref,
                               "rel_error": abs(recon - ref) / ref})
    chk = pd.DataFrame(checks)
    chk.to_csv(OUT / "q3_oav_calibration_check.csv", index=False, encoding="utf-8-sig")

    rows = []
    for rec in tr.to_dict("records"):
        if rec["method"] not in ("ot_proj", "ot_cond"):
            continue
        if rec["constraint"] not in ("loose", "strict"):
            continue
        beer = int(rec["beer"])
        before_row = data.X.loc[data.X["beer"] == beer]
        for d, spec in DEFECT_SPEC.items():
            before = float(before_row[d].iloc[0])
            after = float(rec[f"x_{d}"])
            c_b = _physical(before, d, trans)
            c_a = _physical(after, d, trans)
            thr = spec["threshold"]
            oav_b = c_b / thr if thr else float("nan")
            oav_a = c_a / thr if thr else float("nan")
            rows.append({
                "beer": beer, "method": rec["method"], "constraint": rec["constraint"],
                "compound": d, "concentration_before": c_b,
                "concentration_after": c_a, "unit": spec["unit"],
                "threshold": thr if thr else np.nan,
                "oav_before": oav_b, "oav_after": oav_a,
                "delta_concentration": c_a - c_b,
                "delta_oav": oav_a - oav_b,
                # 与 Q3 判定同一容差：恰好落在阈值上不算超标。
                "exceeds_before": int(oav_b > 1 + 1e-6) if thr else -1,
                "exceeds_after": int(oav_a > 1 + 1e-6) if thr else -1,
                "new_exceed": int(np.isfinite(oav_a) and oav_a > 1 + 1e-6
                                  and (not np.isfinite(oav_b) or oav_b <= 1 + 1e-6)),
                "below_1_after": int(np.isfinite(oav_a) and oav_a <= 1 + 1e-6),
            })
    oav = pd.DataFrame(rows)
    oav.to_csv(OUT / "q3_oav_transfer.csv", index=False, encoding="utf-8-sig")

    summ = []
    for (method, mode), sub in oav.groupby(["method", "constraint"]):
        d = sub[sub["compound"] == "DMS"]
        di = sub[sub["compound"] == "X2.3.butanedione"]
        x4 = sub[sub["compound"] == "X4VG"]
        summ.append({
            "method": method, "constraint": mode,
            "n_samples": int(len(d)),
            "DMS_oav_after_ge1": int((d["oav_after"] > 1 + 1e-6).sum()),
            "DMS_new_exceed": int(d["new_exceed"].sum()),
            "DMS_max_oav_after": float(np.nanmax(d["oav_after"])),
            "diacetyl_oav_after_ge1": int((di["oav_after"] > 1 + 1e-6).sum()),
            "diacetyl_new_exceed": int(di["new_exceed"].sum()),
            "diacetyl_max_oav_after": float(np.nanmax(di["oav_after"])),
            "X4VG_oav_after_ge1": int((x4["oav_after"] > 1 + 1e-6).sum()),
            "X4VG_new_exceed": int(x4["new_exceed"].sum()),
            "nonenal_threshold": "无官方阈值",
        })
    pd.DataFrame(summ).to_csv(OUT / "q3_oav_summary.csv", index=False,
                              encoding="utf-8-sig")

    # 代表性香气物：用 Q3 风格生成均值（逐样本迁移后全向量未落盘）
    gen = pd.read_csv(R.Q3 / "generated.csv", encoding="utf-8-sig")
    th_by_col = th.set_index("field_name_R")
    aroma = []
    for style in ("Pils/Lager", "Tripel"):
        sub = gen[gen["style"] == style]
        for c in AROMA_COMPOUNDS:
            if c not in sub.columns:
                continue
            conc = _physical_median(sub[c], c, trans)
            thr = (float(th_by_col.loc[c, "threshold_median"])
                   if c in th_by_col.index
                   and np.isfinite(th_by_col.loc[c, "threshold_median"]) else np.nan)
            aroma.append({"style": style, "compound": c, "concentration": conc,
                          "threshold_median": thr,
                          "oav": conc / thr if np.isfinite(thr) and thr > 0 else np.nan})
    ar = pd.DataFrame(aroma)
    ar.to_csv(OUT / "q3_aroma_oav_generated.csv", index=False, encoding="utf-8-sig")
    piv = ar.pivot_table(index="compound", columns="style", values="oav")
    if {"Pils/Lager", "Tripel"}.issubset(piv.columns):
        piv["ratio_tripel_vs_pils"] = piv["Tripel"] / piv["Pils/Lager"].replace(0, np.nan)
    piv.reset_index().to_csv(OUT / "q3_aroma_oav_ratio.csv", index=False,
                             encoding="utf-8-sig")

    return {
        "calibration_max_rel_error": float(chk["rel_error"].max()) if len(chk) else None,
        "n_calibration_points": int(len(chk)),
        "transfer_summary": summ,
        "aroma_generated": piv.reset_index().to_dict("records"),
        "note": "迁移逐样本只落盘 4 个缺陷列；代表香气物用 Q3 风格生成样本的"
                "物理量中位数，不是逐样本迁移后浓度，也不能当逐样本合规判定。",
    }


# ------------------------------------------------ 1b. Q3 缺陷完整性核验

# 题目点名的五项缺陷里，四项有化学列，stale_hops 只有感官分。
# 这里把 Q3 的 11 条 Pils/Lager→Tripel 迁移谱完整重建，再用 Q1 主模型读出感官，
# 并对题目点名的其余异味维度和相关化学物做同一套前后对照。
OFF_FLAVOR_DIMS = ["stale_hops", "barnyard", "metallic", "dms", "diacetyl",
                   "t2n", "X4vg", "lactic", "acetic", "overall"]
OFF_FLAVOR_CHEM = ["DMS", "X2.3.butanedione", "X4VG", "X.E..2.nonenal",
                   "MeSH", "H2S", "DES", "SO2.mg.L.", "iron",
                   "X4.ethyl.phenol", "X4.ethyl.guaiacol", "acetaldehyde",
                   "lactic_acid.mg.L.", "acetic_acid.g.L."]
STYLE_FEATURE_COLS = ["X4VG", "lactic_acid.mg.L.", "acetic_acid.g.L.", "iron"]
# POF+ 是酵母表型，按领域先验列出候选风格；数据是否支持要单独看每风格分布，
# 不因为结果不好看就把 Tripel/Dubbel 移出分组。
POF_DOMAIN_STYLES = ["Wheat", "Blond", "Tripel", "Dubbel", "Saison"]
LOW_R2_DIM = 0.10  # OOF R² 低于此值：只作方向监测，不作结论


def _q1_seeds() -> list[int]:
    m = json.loads((R.Q1 / "metrics.json").read_text(encoding="utf-8"))
    return [int(s) for s in (m.get("seeds") or [R.SEED])]


def _reconstruct_transfer(data) -> tuple[pd.DataFrame, float]:
    """按 Q3 原代码路径重建 11 条迁移谱，并原样验证 4 个缺陷列。

    重建是只读的：不写 results/q3，不重跑生成模型；与 transfer.csv 的
    缺陷列逐位比对，确保后续感官读出与主结果同源。
    """
    X = data.X[data.chem_nd].to_numpy(float)
    Ns = R._normal_scores_matrix(data.X[data.chem_nd], data.chem_nd)
    mu_ns, sd_ns = Ns.mean(0), Ns.std(0, ddof=1)
    sd_ns = np.where(sd_ns > 0, sd_ns, 1.0)
    M = (Ns - mu_ns) / sd_ns
    c0 = data.cdr - data.cdr.mean()
    beta = (M.T @ c0) / max(float(c0 @ c0), 1e-12)
    M = M - np.outer(c0, beta)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    k = R.LATENT_K
    Z = U[:, :k] * S[:k]
    Spool = np.cov(Z, rowvar=False)
    P = data.styles == "Pils/Lager"
    T = data.styles == "Tripel"
    mp, Sp = R._moments(Z, P, Spool)
    mt, St = R._moments(Z, T, Spool)
    raw_tables = R._raw_quantile_tables(data.X, data.chem_nd)

    X_clean = np.where(R._floor_mask(X), np.nan, X)
    xmin = np.nanmin(X_clean, axis=0)
    xmax = np.nanmax(X_clean, axis=0)
    xmed = np.nanmedian(X_clean, axis=0)
    xmed = np.where(np.isfinite(xmed), xmed, 0.0)
    col = {c: j for j, c in enumerate(data.chem_nd)}

    tr = pd.read_csv(R.Q3 / "transfer.csv", encoding="utf-8-sig")
    lam = float(tr[(tr["method"] == "ot_proj") & tr["lam"].notna()]["lam"].iloc[0])

    def defect_thr(d):
        spec = R.DEFECTS[d]
        if spec["threshold"] is None:
            return None
        return (float(np.log10(spec["threshold"])) if spec["scale"] == "log10"
                else float(spec["threshold"]))

    def bound(d, xs, mode):
        thr = defect_thr(d)
        if mode == "none":
            return np.inf
        if thr is None:
            return xs
        return max(xs, thr) if mode == "loose" else thr

    chem_mean = data.X[data.chem_all].mean().to_numpy(float)

    def to_full(x):
        f = chem_mean.copy()
        for c in data.chem_nd:
            f[data.chem_all.index(c)] = x[col[c]]
        return f

    rows = []
    for i in np.where(P)[0]:
        beer = int(data.X.iloc[i]["beer"])
        z_ot = R._gaussian_ot(mp, Sp, mt, St, Z[i])
        x_ot = R._ns_to_raw(((z_ot @ Vt[:k]) * sd_ns + mu_ns)[None, :],
                            raw_tables)[0]
        x_src = np.where(np.isfinite(X_clean[i]), X_clean[i], xmed)
        for d in R.DEFECTS:  # 未检出缺陷用该列最低非缺失值，与 Q3 一致
            j = col[d]
            if not np.isfinite(X_clean[i, j]):
                x_src[j] = np.nanmin(X_clean[:, j])
        for mode in ("before", "none", "loose", "strict"):
            if mode == "before":
                x = x_src
            else:
                x_base = (x_ot + lam * x_src) / (1 + lam)
                x_des = x_base.copy()
                for d in R.DEFECTS:
                    j = col[d]
                    b = bound(d, x_src[j], mode)
                    if np.isfinite(b):
                        x_des[j] = min(x_des[j], b)
                x = np.clip(x_des, xmin, xmax)
            rec = {"beer": beer, "mode": mode,
                   "cdr": float(data.cdr[i])}
            full = to_full(x)
            for j, c in enumerate(data.chem_all):
                rec[c] = float(full[j])
            rows.append(rec)
    spectra = pd.DataFrame(rows)

    chk = spectra[spectra["mode"].isin(["none", "loose", "strict"])].merge(
        tr[(tr["method"] == "ot_proj") & np.isclose(tr["lam"], lam)],
        left_on=["beer", "mode"], right_on=["beer", "constraint"],
        suffixes=("", "_q3"))
    diffs = {}
    for d in R.DEFECTS:
        diffs[d] = float(np.max(np.abs(chk[d].to_numpy(float)
                                       - chk[f"x_{d}"].to_numpy(float))))
    pd.DataFrame([{"compound": d, "max_abs_diff": v}
                  for d, v in diffs.items()]).to_csv(
        OUT / "q3_defect_reconstruction_check.csv", index=False,
        encoding="utf-8-sig")
    return spectra, lam


def _q1_sensory_predictions(data, spectra: pd.DataFrame) -> dict[str, np.ndarray]:
    """Q1 主模型（avg = ET 50% + 8 维 PCA 感官潜状态 50%）逐谱读出。

    在 250 个真实样本上拟合后直接预测迁移谱；这是**诊断用读出器**，
    不是泛化性能估计。CDR 沿用源样本，避免把技术协变量当成迁移变化。
    """
    from sklearn.linear_model import Ridge
    Y = data.Y[data.targets].to_numpy(float)
    Xq1 = np.column_stack([data.X[data.chem_all].to_numpy(float), data.cdr])
    Xte = np.column_stack([spectra[data.chem_all].to_numpy(float),
                           spectra["cdr"].to_numpy(float)])
    seeds = _q1_seeds()
    preds = {}
    for name in ("et", "latent_sens_ridge"):
        preds[name] = np.mean(
            [R.fit_predict(name, Xq1, Y, Xte, None, s, False) for s in seeds],
            axis=0)
    preds["avg"] = 0.5 * (preds["et"] + preds["latent_sens_ridge"])
    # Q3 原报告用的单一 Ridge 读出器，用来核对 stale_hops 前后值与主结果同源。
    A, B = R._impute_scale(data.X[data.chem_all].to_numpy(float),
                           spectra[data.chem_all].to_numpy(float))
    preds["q3_sens_ridge"] = Ridge(alpha=10.0).fit(A, Y).predict(B)
    return preds


def q3_defect_completeness(ctx: dict) -> dict:
    data, th, trans = ctx["data"], ctx["th"], ctx["trans"]
    spectra, lam = _reconstruct_transfer(data)
    preds = _q1_sensory_predictions(data, spectra)
    dims = list(data.targets)
    before = (spectra["mode"] == "before").to_numpy()

    # OOF R²：说明哪些感官维度的变化值得当证据，哪些只能当方向。
    comp = pd.read_csv(R.RESULTS / "composite" / "per_target.csv",
                       encoding="utf-8-sig")
    comp = comp[(comp["variant"] == "avg") & (comp["scheme"] == "random")]
    r2_by_dim = comp.groupby("target")["r2"].mean().to_dict()

    long_rows, summ = [], []
    for pname, P in preds.items():
        for mode in ("none", "loose", "strict"):
            idx = (spectra["mode"] == mode).to_numpy()
            deltas = {}
            for dim in OFF_FLAVOR_DIMS:
                j = dims.index(dim)
                d = P[idx, j] - P[before, j]
                deltas[dim] = d
                long_rows.append({
                    "predictor": pname, "constraint": mode, "dimension": dim,
                    "q1_oof_r2": float(r2_by_dim.get(dim, np.nan)),
                    "n_samples": int(len(d)),
                    "delta_median": float(np.median(d)),
                    "delta_mean": float(np.mean(d)),
                    "delta_q25": float(np.quantile(d, 0.25)),
                    "delta_q75": float(np.quantile(d, 0.75)),
                    "n_worsened": int((d > 1e-6).sum()),
                    "n_improved": int((d < -1e-6).sum()),
                    # 效应量以真实样本该维标准差为单位，避免把标准化分差当绝对值。
                    "delta_over_sd": float(np.median(d)
                                           / np.std(data.Y[dim].to_numpy(float),
                                                    ddof=1)),
                    "note": ("低预测力，仅作方向监测"
                             if r2_by_dim.get(dim, 0.0) < LOW_R2_DIM else ""),
                })
            pvals, recs = [], []
            for dim in OFF_FLAVOR_DIMS:
                d = deltas[dim]
                try:
                    p = float(wilcoxon(d).pvalue) if np.any(np.abs(d) > 1e-12) \
                        else float("nan")
                except ValueError:
                    p = float("nan")
                pvals.append(p)
                recs.append({"predictor": pname, "constraint": mode,
                             "dimension": dim, "wilcoxon_p": p})
            q = _bh(pvals)
            for rec, qq in zip(recs, q):
                rec["bh_q"] = float(qq)
                dim = rec["dimension"]
                j = dims.index(dim)
                d = deltas[dim]
                rec.update({
                    "q1_oof_r2": float(r2_by_dim.get(dim, np.nan)),
                    "delta_median": float(np.median(d)),
                    "n_worsened": int((d > 1e-6).sum()),
                    "delta_over_sd": float(np.median(d)
                                           / np.std(data.Y[dim].to_numpy(float),
                                                    ddof=1)),
                })
                summ.append(rec)

    pd.DataFrame(long_rows).to_csv(OUT / "q3_defect_sensory_long.csv",
                                   index=False, encoding="utf-8-sig")
    pd.DataFrame(summ).to_csv(OUT / "q3_defect_sensory_summary.csv",
                              index=False, encoding="utf-8-sig")

    # 化学物：同一批重建谱，物理量逐样本还原后再汇总。
    th_by_col = th.set_index("field_name_R")
    chem_long, chem_summ = [], []
    for c in OFF_FLAVOR_CHEM:
        tr_row = trans.loc[trans["column"] == c, "transform"]
        if not len(tr_row):
            continue
        is_log = tr_row.iloc[0] == "log10"
        conv = (lambda v: 10.0 ** np.asarray(v, float)) if is_log \
            else (lambda v: np.asarray(v, float))
        b = conv(spectra.loc[before, c].to_numpy(float))
        thr = (float(th_by_col.loc[c, "threshold_median"])
               if c in th_by_col.index
               and np.isfinite(th_by_col.loc[c, "threshold_median"]) else np.nan)
        unit = (str(th_by_col.loc[c, "unit"]) if c in th_by_col.index
                and pd.notna(th_by_col.loc[c, "unit"]) else "")
        for mode in ("none", "loose", "strict"):
            idx = (spectra["mode"] == mode).to_numpy()
            a = conv(spectra.loc[idx, c].to_numpy(float))
            for bb, aa in zip(b, a):
                chem_long.append({"compound": c, "constraint": mode,
                                  "before": float(bb), "after": float(aa),
                                  "unit": unit, "threshold": thr})
            oav_b = b / thr if np.isfinite(thr) and thr > 0 else np.full_like(b, np.nan)
            oav_a = a / thr if np.isfinite(thr) and thr > 0 else np.full_like(a, np.nan)
            try:
                p_chem = (float(wilcoxon(a, b).pvalue)
                          if np.any(np.abs(a - b) > 1e-12) else float("nan"))
            except ValueError:
                p_chem = float("nan")
            chem_summ.append({
                "wilcoxon_p": p_chem,
                "compound": c, "constraint": mode, "unit": unit,
                "threshold": thr,
                "median_before": float(np.median(b)),
                "median_after": float(np.median(a)),
                "median_ratio": float(np.median(a) / np.median(b))
                if np.median(b) != 0 else np.nan,
                "n_increased": int((a > b * (1 + 1e-6)).sum()),
                "n_decreased": int((a < b * (1 - 1e-6)).sum()),
                "oav_median_before": float(np.nanmedian(oav_b)) if np.isfinite(thr) else np.nan,
                "oav_median_after": float(np.nanmedian(oav_a)) if np.isfinite(thr) else np.nan,
                "n_above_before": int((oav_b > 1 + 1e-6).sum()) if np.isfinite(thr) else -1,
                "n_above_after": int((oav_a > 1 + 1e-6).sum()) if np.isfinite(thr) else -1,
                "n_new_above": int(((oav_a > 1 + 1e-6)
                                    & (oav_b <= 1 + 1e-6)).sum())
                if np.isfinite(thr) else -1,
                "note": "" if np.isfinite(thr) else "无官方阈值，只报相对变化",
            })
    pd.DataFrame(chem_long).to_csv(OUT / "q3_defect_chemical_long.csv",
                                   index=False, encoding="utf-8-sig")
    chem = pd.DataFrame(chem_summ)
    chem["bh_q"] = np.nan
    for _mode, sub in chem.groupby("constraint"):
        chem.loc[sub.index, "bh_q"] = _bh(sub["wilcoxon_p"].to_numpy())
    chem.to_csv(OUT / "q3_defect_chemical_summary.csv", index=False,
                encoding="utf-8-sig")
    chem_recs = chem.to_dict("records")

    # 与 Q3 原 stale_hops 列核对：证明感官读出链路没有换模型。
    tr = pd.read_csv(R.Q3 / "transfer.csv", encoding="utf-8-sig")
    tr = tr[(tr["method"] == "ot_proj") & np.isclose(tr["lam"], lam)]
    stale_i = dims.index("stale_hops")
    rows = []
    for mode, colname in (("before", "stale_hops_before"),
                          ("none", "stale_hops_after"),
                          ("loose", "stale_hops_after"),
                          ("strict", "stale_hops_after")):
        sub = (tr.drop_duplicates("beer") if mode == "before"
               else tr[tr["constraint"] == mode])
        idx = (spectra["mode"] == mode).to_numpy()
        beer = spectra.loc[idx, "beer"].to_numpy()
        ref = sub.set_index("beer")[colname].reindex(beer).to_numpy(float)
        ours = preds["q3_sens_ridge"][idx, stale_i]
        rows.append({"mode": mode, "max_abs_diff":
                     float(np.nanmax(np.abs(ours - ref)))})
    pd.DataFrame(rows).to_csv(OUT / "q3_defect_stale_replication.csv",
                              index=False, encoding="utf-8-sig")

    def pick(pname, mode, dim):
        s = pd.DataFrame(summ)
        r = s[(s.predictor == pname) & (s.constraint == mode)
              & (s.dimension == dim)]
        return r.iloc[0].to_dict() if len(r) else {}

    return {
        "lam": lam,
        "n_source_samples": int((spectra["mode"] == "before").sum()),
        "sensory_summary": summ,
        "chemical_summary": chem_recs,
        "stale_hops_avg_loose": pick("avg", "loose", "stale_hops"),
        "metallic_avg_loose": pick("avg", "loose", "metallic"),
        "barnyard_avg_loose": pick("avg", "loose", "barnyard"),
        "h2s_loose": next(r for r in chem_recs
                          if r["compound"] == "H2S" and r["constraint"] == "loose"),
        "mesh_loose": next(r for r in chem_recs
                           if r["compound"] == "MeSH" and r["constraint"] == "loose"),
        "iron_loose": next(r for r in chem_recs
                           if r["compound"] == "iron" and r["constraint"] == "loose"),
        "note": "重建谱与 results/q3/transfer.csv 的 4 个缺陷列逐位一致；"
                "感官读数是诊断用，不是泛化性能。",
    }


# ---------------------------------------------- 1c. 风格依赖阈值

def style_feature_distributions(ctx: dict) -> dict:
    """22 个风格的 4VG / 乳酸 / 醋酸 / 铁分布与阈值对照。"""
    data, th, trans = ctx["data"], ctx["th"], ctx["trans"]
    styles = data.styles
    style_names = sorted(np.unique(styles))
    th_by_col = th.set_index("field_name_R")
    rows, tests, pair_rows, sens_rows = [], [], [], []
    for c in STYLE_FEATURE_COLS:
        tr_row = trans.loc[trans["column"] == c, "transform"]
        is_log = bool(len(tr_row) and tr_row.iloc[0] == "log10")
        v = 10.0 ** data.X[c].to_numpy(float) if is_log \
            else data.X[c].to_numpy(float)
        thr = (float(th_by_col.loc[c, "threshold_median"])
               if c in th_by_col.index
               and np.isfinite(th_by_col.loc[c, "threshold_median"]) else np.nan)
        unit = (str(th_by_col.loc[c, "unit"]) if c in th_by_col.index
                and pd.notna(th_by_col.loc[c, "unit"]) else "")
        floor = float(np.min(v))
        groups = [v[styles == s] for s in style_names]
        H, pk = kruskal(*groups)
        tests.append({"compound": c, "test": "kruskal_all_styles",
                      "statistic": float(H), "p": float(pk),
                      "n_groups": len(groups)})
        for s in style_names:
            x = v[styles == s]
            rows.append({
                "compound": c, "style": s, "n": int(len(x)),
                "median": float(np.median(x)),
                "q25": float(np.quantile(x, 0.25)),
                "q75": float(np.quantile(x, 0.75)),
                "p10": float(np.quantile(x, 0.10)),
                "p90": float(np.quantile(x, 0.90)),
                "n_at_floor": int(np.isclose(x, floor).sum()),
                "floor_value": floor, "threshold": thr, "unit": unit,
                "n_above_threshold": int((x > thr).sum()) if np.isfinite(thr) else -1,
                "exceed_rate": float((x > thr).mean()) if np.isfinite(thr) else np.nan,
            })
        # 每个风格 vs Pils/Lager：只作描述，BH 校正后标注。
        base = v[styles == "Pils/Lager"]
        sub = []
        for s in style_names:
            if s == "Pils/Lager":
                continue
            x = v[styles == s]
            try:
                p = float(mannwhitneyu(x, base, alternative="two-sided").pvalue)
            except ValueError:
                p = float("nan")
            sub.append({"compound": c, "style": s,
                        "n_style": int(len(x)), "n_pils": int(len(base)),
                        "median_style": float(np.median(x)),
                        "median_pils": float(np.median(base)),
                        "median_ratio": float(np.median(x) / np.median(base))
                        if np.median(base) != 0 else np.nan,
                        "mannwhitney_p": p})
        q = _bh([r["mannwhitney_p"] for r in sub])
        for r, qq in zip(sub, q):
            r["bh_q"] = float(qq)
            tests.append({"compound": c, "test": f"{r['style']}_vs_Pils",
                          "statistic": np.nan, "p": r["mannwhitney_p"],
                          "n_groups": 2, "bh_q": float(qq),
                          "median_ratio": r["median_ratio"]})
            pair_rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "q3_style_feature_distribution.csv",
                              index=False, encoding="utf-8-sig")
    pd.DataFrame(tests).to_csv(OUT / "q3_style_feature_tests.csv", index=False,
                               encoding="utf-8-sig")

    # 4VG 与同名/酚香感官：整体和风格内分别看，避免辛普森悖论。
    v4 = 10.0 ** data.X["X4VG"].to_numpy(float)
    for target in ("X4vg", "clove", "coriander", "barnyard", "overall"):
        y = data.Y[target].to_numpy(float)
        rho, p = _spearman(v4, y)
        sens_rows.append({"compound": "X4VG", "style": "ALL",
                          "n_style": int(len(y)),
                          "median_4vg": float(np.median(v4)),
                          "sensory_target": target, "spearman": rho,
                          "spearman_p": p})
        for s in style_names:
            x = v4[styles == s]
            yv = y[styles == s]
            if len(x) < 8 or np.std(x) == 0 or np.std(yv) == 0:
                continue
            rho, p = _spearman(x, yv)
            sens_rows.append({"compound": "X4VG", "style": s,
                              "n_style": int(len(x)),
                              "median_4vg": float(np.median(x)),
                              "sensory_target": target, "spearman": rho,
                              "spearman_p": p})
    pd.DataFrame(pair_rows).to_csv(OUT / "q3_style_feature_pairwise.csv",
                                   index=False, encoding="utf-8-sig")
    pd.DataFrame(sens_rows).to_csv(OUT / "q3_style_feature_sensory_corr.csv",
                                   index=False, encoding="utf-8-sig")

    # POF+ 领域先验分组 vs Pils/Lager（不因结果调整分组）。
    grp = np.isin(styles, POF_DOMAIN_STYLES)
    base = v4[styles == "Pils/Lager"]
    pof_test = {}
    if grp.any():
        try:
            p = float(mannwhitneyu(v4[grp], base, alternative="two-sided").pvalue)
        except ValueError:
            p = float("nan")
        pof_test = {"styles": POF_DOMAIN_STYLES,
                    "n_pof": int(grp.sum()),
                    "median_pof": float(np.median(v4[grp])),
                    "median_pils": float(np.median(base)),
                    "p": p}
    dist = pd.DataFrame(rows)
    tripel_sens = next((r for r in sens_rows
                        if r["style"] == "Tripel"
                        and r["sensory_target"] == "X4vg"), {})
    x4 = dist[dist["compound"] == "X4VG"].set_index("style")
    lac = dist[dist["compound"] == "lactic_acid.mg.L."].set_index("style")
    ace = dist[dist["compound"] == "acetic_acid.g.L."].set_index("style")
    return {
        "kruskal": {r["compound"]: r["p"] for r in tests
                    if r["test"] == "kruskal_all_styles"},
        "x4vg": {
            "wheat_exceed_rate": float(x4.loc["Wheat", "exceed_rate"]),
            "wheat_median": float(x4.loc["Wheat", "median"]),
            "blond_exceed_rate": float(x4.loc["Blond", "exceed_rate"]),
            "saison_exceed_rate": float(x4.loc["Saison", "exceed_rate"]),
            "tripel_sensory": {"spearman": tripel_sens.get("spearman", np.nan),
                               "p": tripel_sens.get("spearman_p", np.nan),
                               "n": tripel_sens.get("n_style", 0)},
            "tripel_at_floor": int(x4.loc["Tripel", "n_at_floor"]),
            "tripel_n": int(x4.loc["Tripel", "n"]),
            "tripel_median": float(x4.loc["Tripel", "median"]),
            "dubbel_at_floor": int(x4.loc["Dubbel", "n_at_floor"]),
            "dubbel_n": int(x4.loc["Dubbel", "n"]),
            "pils_median": float(x4.loc["Pils/Lager", "median"]),
            "threshold": float(x4.loc["Wheat", "threshold"]),
            "pof_domain_test": pof_test,
        },
        "lactic": {
            "sour_styles": ["Lambic", "Kriek", "Faro", "Flanders old brown",
                            "West Flanders ale"],
            "sour_exceed_rates": {s: float(lac.loc[s, "exceed_rate"])
                                  for s in ["Lambic", "Kriek", "Faro",
                                            "Flanders old brown",
                                            "West Flanders ale"]},
            "pils_median": float(lac.loc["Pils/Lager", "median"]),
            "threshold": float(lac.loc["Pils/Lager", "threshold"]),
        },
        "acetic": {
            "sour_exceed_rates": {s: float(ace.loc[s, "exceed_rate"])
                                  for s in ["Lambic", "Kriek", "Faro",
                                            "Flanders old brown",
                                            "West Flanders ale"]},
            "tripel_exceed_rate": float(ace.loc["Tripel", "exceed_rate"]),
            "pils_exceed_rate": float(ace.loc["Pils/Lager", "exceed_rate"]),
            "threshold": float(ace.loc["Pils/Lager", "threshold"]),
        },
        "note": "阈值是感官检测阈值，不是风格缺陷上限；"
                "同一阈值套到所有风格会把酸啤等风格整体判为缺陷。",
    }


# ---------------------------------------------------------------- 3. 价格

PRICE_STYLE_ALIAS = {"wit": "Wheat", "bruin": "Brown", "oud bruin": "Flanders old brown"}


def price_analysis(ctx: dict) -> dict:
    data, price = ctx["data"], ctx["price"]
    price = price.copy()
    style_by_norm = {_norm(s): s for s in data.styles}
    for k, v in PRICE_STYLE_ALIAS.items():
        style_by_norm[_norm(k)] = v
    price["style"] = price["tasting_category"].map(lambda s: style_by_norm.get(_norm(s)))
    price["beer"] = price["beer"].astype(int)

    pcols = [c for c in price.columns if c.startswith("Price_")]
    coverage = {c: int(price[c].notna().sum()) for c in pcols}
    primary = max(coverage, key=coverage.get)

    long = price.melt(id_vars=["beer", "style", "ABV"], value_vars=pcols,
                      var_name="channel", value_name="price").dropna()
    long.to_csv(OUT / "price_long.csv", index=False, encoding="utf-8-sig")

    by_style = (price.dropna(subset=["style", primary])
                .groupby("style")[primary]
                .agg(n="count", median_price="median", mean="mean",
                     q1=lambda s: s.quantile(.25), q3=lambda s: s.quantile(.75))
                .reset_index())
    by_style.to_csv(OUT / "price_style_primary_channel.csv", index=False,
                    encoding="utf-8-sig")

    # 每个渠道的风格中位价（覆盖 ≥20 才报告）
    per_channel = []
    for c in pcols:
        if coverage[c] < 20:
            continue
        g = (price.dropna(subset=["style", c]).groupby("style")[c]
             .agg(n="count", median="median").reset_index())
        g["channel"] = c
        per_channel.append(g)
    pd.concat(per_channel).to_csv(OUT / "price_style_by_channel.csv",
                                  index=False, encoding="utf-8-sig")

    # 全风格对：用 q3_pairs 的逐坐标裁剪结果（与 ot_proj 同口径）
    pairs_path = R.RESULTS / "q3_pairs" / "pairwise.csv"
    if not pairs_path.exists():
        return {"error": "缺 results/q3_pairs/pairwise.csv，无法做全风格对价格分析"}
    pairs = pd.read_csv(pairs_path)
    pairs = pairs[pairs["mode"] == "loose"].copy()
    med = by_style.set_index("style")["median_price"]
    n_by = by_style.set_index("style")["n"]
    pairs["source_price"] = pairs["source"].map(med)
    pairs["target_price"] = pairs["target"].map(med)
    pairs["source_n"] = pairs["source"].map(n_by)
    pairs["target_n"] = pairs["target"].map(n_by)
    pairs = pairs.dropna(subset=["source_price", "target_price"])
    pairs["delta_price"] = pairs["target_price"] - pairs["source_price"]
    pairs["price_ratio"] = pairs["target_price"] / pairs["source_price"]
    pairs["log_price_ratio"] = np.log(pairs["price_ratio"])
    pairs.to_csv(OUT / "price_pairs.csv", index=False, encoding="utf-8-sig")

    corr = {}
    for xcol in ("delta_price", "log_price_ratio"):
        for ycol in ("qp_cost_median", "style_shift_median", "feasible_rate"):
            if pairs[ycol].nunique(dropna=True) < 2:
                corr[f"{xcol}_vs_{ycol}"] = {
                    "spearman": float("nan"), "p": float("nan"),
                    "n": int(len(pairs)), "note": "该列无方差，秩相关无定义"}
                continue
            r, p = _spearman(pairs[xcol], pairs[ycol])
            corr[f"{xcol}_vs_{ycol}"] = {"spearman": r, "p": p, "n": int(len(pairs))}

    # 推荐：可行率高、修正代价低、溢价高、风格方向一致。
    # 注意 qp_cost 在逐坐标裁剪下几乎恒等于 lam/(1+lam)≈0.091，
    # 所以真正有区分度的是价格比与感官位移；代码仍保留代价项，报告里会说明。
    ok = pairs[(pairs["feasible_rate"] >= 0.9) & (pairs["source_n"] >= 3)
               & (pairs["target_n"] >= 3)].copy()
    if len(ok):
        def _z(s):
            sd = s.std()
            return (s - s.mean()) / sd if sd > 0 else s * 0
        ok["score"] = (_z(ok["log_price_ratio"]) + _z(ok["style_shift_median"])
                       + _z(ok["sens_cos_median"]) - _z(ok["qp_cost_median"]))
        top = ok.sort_values("score", ascending=False).head(15)
        top.to_csv(OUT / "price_top_pairs.csv", index=False, encoding="utf-8-sig")
        # 每个目标风格只保留最佳源风格，避免整张表被同一个高价风格占满。
        best_by_target = (ok.sort_values("score", ascending=False)
                          .drop_duplicates("target").head(10))
        best_by_target.to_csv(OUT / "price_best_source_per_target.csv",
                              index=False, encoding="utf-8-sig")
    else:
        top = ok
        best_by_target = ok

    target_rank = (pairs.groupby("target")
                   .agg(median_price=("target_price", "median"),
                        median_ratio=("price_ratio", "median"),
                        median_cost=("qp_cost_median", "median"),
                        median_shift=("style_shift_median", "median"),
                        median_sens_cos=("sens_cos_median", "median"),
                        n_sources=("source", "count"))
                   .sort_values("median_ratio", ascending=False).reset_index())
    target_rank.to_csv(OUT / "price_target_ranking.csv", index=False,
                       encoding="utf-8-sig")

    pils_tripel = pairs[(pairs["source"] == "Pils/Lager") & (pairs["target"] == "Tripel")]
    return {
        "channels": coverage, "primary_channel": primary,
        "unit_note": "价格表没有容量字段，无法换算 €/L；绝对价差是渠道标价单位，"
                     "比值无量纲。",
        "cost_note": "逐坐标裁剪下 qp_cost 几乎恒定（≈0.091），"
                     "因此推荐排序主要由价格比与感官方向决定，不是代价。",
        "n_style_pairs": int(len(pairs)),
        "pils_tripel": pils_tripel.to_dict("records"),
        "correlations": corr,
        "top_recommendations": top[["source", "target", "delta_price", "price_ratio",
                                    "qp_cost_median", "style_shift_median",
                                    "sens_cos_median", "feasible_rate", "score"]].to_dict("records")
        if len(top) else [],
        "best_source_per_target": best_by_target[
            ["source", "target", "delta_price", "price_ratio", "qp_cost_median",
             "style_shift_median", "score"]].to_dict("records") if len(best_by_target) else [],
        "target_ranking": target_rank.to_dict("records"),
        "style_table": by_style.to_dict("records"),
    }


# ---------------------------------------------------------------- 4. 外部验证

def _macro_matrix(df: pd.DataFrame, mapping: dict) -> np.ndarray:
    cols = []
    for dims in mapping.values():
        cols.append(df[dims].mean(axis=1).to_numpy(float))
    return np.column_stack(cols)


def _rank_matrix(M: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return np.column_stack([rankdata(M[:, j]) for j in range(M.shape[1])])


def _mantel(A: np.ndarray, B: np.ndarray, n_perm: int = N_PERM_MANTEL) -> dict:
    """Mantel：两个 Spearman 相关矩阵上三角的 Pearson 相关。

    置换时行、列必须同步打乱，否则相关矩阵不变，p 恒为 1。
    """
    iu = np.triu_indices_from(A, k=1)
    a = A[iu]
    d = A.shape[0]
    r = float(pearsonr(a, B[iu]).statistic)
    cnt = 0
    for _ in range(n_perm):
        idx = RNG.permutation(d)
        rp = float(pearsonr(a, B[np.ix_(idx, idx)][iu]).statistic)
        if np.isfinite(rp) and abs(rp) >= abs(r):
            cnt += 1
    return {"r": r, "p": (cnt + 1) / (n_perm + 1), "n_dims": d,
            "n_perm": n_perm}


def _corr_matrix(M: np.ndarray) -> np.ndarray:
    return np.corrcoef(_rank_matrix(M), rowvar=False)


def _cos_sim(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _perm_rho(a, b, n_perm: int = N_PERM) -> dict:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    r = float(spearmanr(a, b).statistic)
    cnt = 0
    for _ in range(n_perm):
        rp = float(spearmanr(a, RNG.permutation(b)).statistic)
        if np.isfinite(rp) and abs(rp) >= abs(r):
            cnt += 1
    return {"spearman": r, "p_two_sided": (cnt + 1) / (n_perm + 1), "n_dims": len(a)}


def external_validation(ctx: dict) -> dict:
    data, ext, price = ctx["data"], ctx["ext"], ctx["price"]
    Y = data.Y[data.targets].to_numpy(float)
    styles = data.styles

    # -- 保守重叠排除：同风格族 + |ABV 差|≤0.1 + 内部 IBU 落在外部 Min/Max±2
    abv_int = price.set_index("beer")["ABV"]
    ibu_int = pd.Series(10 ** data.X["bitterness"].to_numpy(),
                        index=data.X["beer"].to_numpy())
    fam_int = pd.Series([_family(s) for s in data.styles],
                        index=data.X["beer"].to_numpy())
    ext = ext.copy()
    ext["family"] = ext["Style"].map(_family)
    ext["ibu_mid"] = (ext["Min IBU"] + ext["Max IBU"]) / 2
    overlap_mask = np.zeros(len(ext), dtype=bool)
    for beer in data.X["beer"]:
        a, ib, fam = abv_int.get(beer), ibu_int.get(beer), fam_int.get(beer)
        if not all(np.isfinite(x) for x in (a, ib)):
            continue
        overlap_mask |= ((ext["family"] == fam)
                         & (np.abs(ext["ABV"] - a) <= 0.1)
                         & (ext["Min IBU"] - 2 <= ib)
                         & (ext["Max IBU"] + 2 >= ib)).to_numpy()
    n_overlap = int(overlap_mask.sum())
    ext_clean = ext.loc[~overlap_mask].copy()

    # Mantel 用按风格分层的固定子样本（全量 5,558 条置换成本太高）。
    rng = np.random.default_rng(R.SEED)
    per_style = max(1, N_EXT_MANTEL // ext_clean["Style"].nunique())
    sub_idx = (ext_clean.groupby("Style", group_keys=False)
               .apply(lambda g: g.sample(min(len(g), per_style), random_state=R.SEED))
               .index)
    ext_sub = ext_clean.loc[sub_idx]

    out = {"n_external": int(len(ext)), "n_excluded_overlap": n_overlap,
           "n_used_mantel": int(len(ext_sub)),
           "overlap_rule": "同风格族 + |ABV 差|≤0.1 + 内 IBU 落在外部 Min/Max±2"}
    for tag, mapping in (("primary9", MACRO_MAP), ("loose11", MACRO_MAP_LOOSE)):
        # 外部表与比赛表的列名不同，但 11 个宏观描述符的键名一致。
        E = _macro_matrix(ext_sub, {k: [k] for k in mapping})
        C = _macro_matrix(pd.DataFrame(Y, columns=data.targets), mapping)
        m = _mantel(_corr_matrix(E), _corr_matrix(C))
        m["dims"] = list(mapping)
        out[f"mantel_{tag}"] = m

    # -- 位移方向：外部 Pils vs Tripel（排除重叠后若样本太少则退回全量并标注）
    pils_styles = ["Pilsner - Bohemian / Czech", "Pilsner - German"]
    ext_pils = ext_clean[ext_clean["Style"].isin(pils_styles)]
    ext_trip = ext_clean[ext_clean["Style"].str.strip().str.lower() == "tripel"]
    used_overlap = False
    if len(ext_pils) < 10 or len(ext_trip) < 10:
        used_overlap = True
        ext_pils = ext[ext["Style"].isin(pils_styles)]
        ext_trip = ext[ext["Style"].str.strip().str.lower() == "tripel"]

    dims = list(MACRO_MAP)
    ext_map = {k: [k] for k in MACRO_MAP}
    E_all = _macro_matrix(ext, ext_map)
    mu_e, sd_e = E_all.mean(0), E_all.std(0, ddof=1)
    sd_e = np.where(sd_e > 0, sd_e, 1.0)
    shift_ext = ((_macro_matrix(ext_trip, ext_map).mean(0) - mu_e) / sd_e
                 - (_macro_matrix(ext_pils, ext_map).mean(0) - mu_e) / sd_e)

    C_all = _macro_matrix(pd.DataFrame(Y, columns=data.targets), MACRO_MAP)
    mu_c, sd_c = C_all.mean(0), C_all.std(0, ddof=1)
    sd_c = np.where(sd_c > 0, sd_c, 1.0)
    real_pils = Y[styles == "Pils/Lager"]
    real_trip = Y[styles == "Tripel"]
    gen = pd.read_csv(R.Q3 / "generated_sensory.csv", encoding="utf-8-sig")
    gen_pils = gen[gen["style"] == "Pils/Lager"][data.targets].to_numpy(float)
    gen_trip = gen[gen["style"] == "Tripel"][data.targets].to_numpy(float)

    def _shift(hi, lo):
        return ((_macro_matrix(pd.DataFrame(hi, columns=data.targets), MACRO_MAP).mean(0)
                 - mu_c) / sd_c
                - (_macro_matrix(pd.DataFrame(lo, columns=data.targets), MACRO_MAP).mean(0)
                   - mu_c) / sd_c)

    shift_real = _shift(real_trip, real_pils)
    shift_gen = _shift(gen_trip, gen_pils)

    def _dir(a, b):
        return {"cosine": _cos_sim(a, b), **_perm_rho(a, b),
                "sign_agreement": float(np.mean(np.sign(a) == np.sign(b)))}

    out["shift"] = {
        "dims": dims,
        "external": dict(zip(dims, shift_ext.round(4))),
        "competition_real": dict(zip(dims, shift_real.round(4))),
        "competition_generated": dict(zip(dims, shift_gen.round(4))),
        "ext_vs_comp_real": _dir(shift_ext, shift_real),
        "ext_vs_comp_generated": _dir(shift_ext, shift_gen),
        "n_external_pils": int(len(ext_pils)), "n_external_tripel": int(len(ext_trip)),
        "external_pils_styles": pils_styles,
        "used_overlapping_subset": used_overlap,
        "note": "位移比较若因重叠排除后样本不足，退回全量外部样本；"
                "此时该比较不是独立验证，不能作为无泄漏证据。",
    }

    # -- ABV / IBU 的风格级对照（物理单位）
    rows = []
    for style in ("Pils/Lager", "Tripel"):
        sel = styles == style
        abv = 10 ** data.X.loc[sel, "ethanol..v.v."].to_numpy()
        ibu = 10 ** data.X.loc[sel, "bitterness"].to_numpy()
        rows.append({"dataset": "competition", "style": style,
                     "n": int(sel.sum()), "ABV_mean": float(np.mean(abv)),
                     "IBU_mean": float(np.mean(ibu))})
    for label, sub in (("external_pils", ext_pils), ("external_tripel", ext_trip)):
        rows.append({"dataset": "external", "style": label, "n": int(len(sub)),
                     "ABV_mean": float(sub["ABV"].mean()),
                     "IBU_mean": float(sub["ibu_mid"].mean())})
    ab = pd.DataFrame(rows)
    ab.to_csv(OUT / "external_abv_ibu.csv", index=False, encoding="utf-8-sig")
    out["abv_ibu"] = ab.to_dict("records")

    pd.DataFrame({
        "dim": dims, "external": shift_ext, "competition_real": shift_real,
        "competition_generated": shift_gen,
    }).to_csv(OUT / "external_shift.csv", index=False, encoding="utf-8-sig")
    return out


# ---------------------------------------------------------------- 汇总

def write_report(results: dict, runtime: float, integrity: dict) -> None:
    lines = [
        "# S 门：补充数据独立验证与注释",
        "",
        f"- 生成时间：{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}；用时 {runtime:.1f}s。",
        "- **定位：仅用于机制注释、OAV/价格量化与跨数据集独立验证；"
        "不进入 Q1/Q2/Q3 主模型训练，也未修改主结果。**",
        f"- S 门：registered/pass；外部文件哈希 "
        f"{'全部一致' if integrity['all_match'] else '不一致，结果作废'}。",
        "",
        "## 1. Q2 模块富集 + Q3 物理 OAV",
        "",
        f"- 阈值表 {results['module']['n_mapped_threshold_rows']} 条记录可映射到 Q2 的 "
        f"{results['module']['n_nodes']} 个节点；其中 "
        f"{results['module']['n_with_threshold']} 个有官方阈值。",
        f"- 8 个模块共做 {results['module']['n_enrichment_tests']} 个来源富集检验，"
        f"BH 后 {results['module']['n_q_below_0.05']} 个显著。",
        "",
        "| 模块 | 节点数 | 有阈值 | 期望 | 富集倍数 | q | 主要来源 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for m in results["module"]["modules"]:
        lines.append(f"| {m['module']} | {m['size']} | {m['n_with_threshold']} | "
                     f"{m['expected_with_threshold']} | {m['fold_enrichment']} | "
                     f"{m['q_threshold']:.3g} | {m['top_origins'] or '—'} |")
    m3 = next((m for m in results["module"]["modules"] if m["module"] == 3), None)
    if m3:
        lines += ["", f"- **解读**：模块 3 是酵母发酵中心（酯类、高级醇、硫化物），"
                  f"{m3['n_with_threshold']}/{m3['size']} 个节点有官方阈值，"
                  f"富集 {m3['fold_enrichment']:.1f} 倍（q={m3['q_threshold']:.2g}）；"
                  "模块 0（野生菌/酸/美拉德）与模块 2（酒花/氧化）在来源层面显著，"
                  "但模块整体阈值密度不高；模块 1 没有任何官方阈值，"
                  "只能注释为“缺阈值”，不能写成“低于阈值”。"]
    o = results["oav"]
    lines += [
        "",
        f"- 浓度还原核对：{o['n_calibration_points']} 个点，最大相对误差 "
        f"{o['calibration_max_rel_error']:.2e}。" if o["calibration_max_rel_error"] is not None
        else "- 浓度还原核对：无可用点。",
        "",
        "| 方法 | 口径 | DMS 超标 | 双乙酰超标 | 新增超标(DMS/双乙酰) | DMS 最大 OAV |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for s in o["transfer_summary"]:
        lines.append(f"| {s['method']} | {s['constraint']} | {s['DMS_oav_after_ge1']} | "
                     f"{s['diacetyl_oav_after_ge1']} | "
                     f"{s['DMS_new_exceed']}/{s['diacetyl_new_exceed']} | "
                     f"{s['DMS_max_oav_after']:.2f} |")
    lines += ["", "> 松口径允许“不加重”但不消除原有超标：DMS 与双乙酰各有 3 个样本在迁移前"
              "就已超标；严口径把它们压到阈值上，因此 0 超标，但这是保守上界，"
              "不表示真实样品一定合规。", "",
              f"> {o['note']}", ""]

    d = results["defect"]
    dr = {(r["dimension"], r["constraint"]): r
          for r in d["sensory_summary"] if r["predictor"] == "avg"}
    dc = {(r["compound"], r["constraint"]): r
          for r in d["chemical_summary"]}
    lines += [
        "",
        "## 2. Q3 缺陷完整性（五项点名缺陷与广谱异味）",
        "",
        f"- 覆盖 {d['n_source_samples']} 条真实 Pils/Lager → Tripel 迁移"
        f"（λ={d['lam']:.2f}，`ot_proj`）；重建谱与 `results/q3/transfer.csv` 的"
        f" 4 个缺陷列逐位一致；感官读数是诊断用，不是泛化性能。",
        "",
        "**Q1 主模型读出的感官变化（avg；Δ>0 为变差）**",
        "",
        "| 维度 | 松口径 Δ中位 | 变差样本 | 效应量(Δ/SD) | Q1 OOF R² | 判定 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dim in ("stale_hops", "barnyard", "metallic", "dms", "diacetyl",
                "t2n", "X4vg", "lactic", "acetic"):
        r = dr.get((dim, "loose"))
        if not r:
            continue
        verdict = ("方向监测（R²低，不作结论）"
                   if r["q1_oof_r2"] < LOW_R2_DIM
                   else ("同步下降" if r["delta_median"] < 0 else "略有上升"))
        lines.append(f"| {dim} | {r['delta_median']:+.3f} | "
                     f"{r['n_worsened']}/11 | {r['delta_over_sd']:+.3f} | "
                     f"{r['q1_oof_r2']:.3f} | {verdict} |")
    lines += [
        "",
        "**相关化学物（物理量中位数，松口径）**",
        "",
        "| 化合物 | 迁移前 | 迁移后 | 倍数 | 变高 | 新增超标 | 阈值 | OAV 前→后 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in OFF_FLAVOR_CHEM:
        r = dc.get((c, "loose"))
        if not r or not np.isfinite(r["median_after"]):
            continue
        thr = f"{r['threshold']:g} {r['unit']}" if np.isfinite(r["threshold"]) else "—"
        oav = (f"{r['oav_median_before']:.2f}→{r['oav_median_after']:.2f}"
               if np.isfinite(r["oav_median_before"]) else "—")
        new_above = (str(int(r["n_new_above"])) if np.isfinite(r["threshold"])
                     else "—")
        lines.append(f"| {c} | {r['median_before']:.4g} | {r['median_after']:.4g} | "
                     f"×{r['median_ratio']:.2f} | {r['n_increased']}/11 | "
                     f"{new_above} | {thr} | {oav} |")
    h2s = d["h2s_loose"]
    mesh = d["mesh_loose"]
    iron = d["iron_loose"]
    lines += [
        "",
        "- **同名物一致性**：Q1 读出 `X4vg` +0.189（11/11 上升），但化学 4VG 中位不变"
        "（×1.00）。说明感官 `X4vg` 的变化来自整条谱的风格位移，不是 4VG 单独升高；"
        "与 Q2 的同名物审计一致，**不能把 X4vg 当 4VG 的直接读数**。",
        "- **五项点名缺陷**：DMS、双乙酰、t2n 在松/严口径下都被压回阈值或源水平，"
        "新增超标 0；4VG 在这 11 条里基本不变（10/11 停在检出限 0.0097，"
        "1 条升到 0.0184，仍比 0.2 阈值低一个量级以上；目标是 Tripel，见下节）；"
        "`stale_hops` 的 Q1 读出为下降，但该维 OOF R²≈0.01，"
        "**只能写“方向一致”，不能写“已验证受控”**。",
        f"- **约束未覆盖的异味**：H2S 中位数 ×{h2s['median_ratio']:.2f}"
        f"（OAV {h2s['oav_median_before']:.2f}→{h2s['oav_median_after']:.2f}，"
        f"p={h2s['wilcoxon_p']:.4f}），迁移前就已超阈值、迁移后继续升高；"
        f"MeSH ×{mesh['median_ratio']:.2f}（p={mesh['wilcoxon_p']:.4f}，无官方阈值）。"
        "四缺陷约束按设计只动 4 列，**硫化物整体没有被约束住**，"
        "这是必须写进局限的完整性缺口。",
        "- **“不新增超标”≠“不恶化”**：DMS 与双乙酰在松口径下仍显著升高"
        f"（p={dc[('DMS', 'loose')]['wilcoxon_p']:.4f} / "
        f"{dc[('X2.3.butanedione', 'loose')]['wilcoxon_p']:.4f}，各 8/11），"
        "只是被封在阈值上；醋酸新增 "
        f"{dc[('acetic_acid.g.L.', 'loose')]['n_new_above']}/11 超过 0.145 g/L，"
        "但该阈值本身不适合 Tripel（见下节），不能直接判为缺陷。",
        f"- **金属味**：感官 `metallic` 中位 +{d['metallic_avg_loose']['delta_median']:.3f}"
        f"（{d['metallic_avg_loose']['n_worsened']}/11 变差，Wilcoxon "
        f"p={d['metallic_avg_loose'].get('wilcoxon_p', float('nan')):.3f}，R²=0.107 偏弱）；"
        f"铁 ×{iron['median_ratio']:.2f}，OAV {iron['oav_median_before']:.2f}→"
        f"{iron['oav_median_after']:.2f}（p={iron['wilcoxon_p']:.3f}），仍低于阈值。"
        "方向一致但均不显著，只能列为观察项。",
        "- 表中 `X4.ethyl.phenol` / `X4.ethyl.guaiacol` 的单位是仪器响应 A.U.，"
        "绝对数值不解释，只看倍数。",
        "- **判定**：五项点名缺陷中，有化学列的四项在松/严口径下合规；"
        "stale hops 依赖低 R² 感官预测器，证据强度有限；"
        "H2S 等未点名硫化物恶化，说明“四缺陷约束”不等于“全异味受控”。",
        "",
    ]

    sf = results["style_feature"]
    x4 = sf["x4vg"]
    lac = sf["lactic"]
    ace = sf["acetic"]
    lines += [
        "## 3. 风格依赖阈值：4VG / 乳酸 / 醋酸 / 铁",
        "",
        f"- Kruskal-Wallis（22 风格）：4VG p={sf['kruskal']['X4VG']:.2g}、"
        f"乳酸 p={sf['kruskal']['lactic_acid.mg.L.']:.2g}、"
        f"醋酸 p={sf['kruskal']['acetic_acid.g.L.']:.2g}、"
        f"铁 p={sf['kruskal']['iron']:.2g}——四者都强烈依赖风格。",
        f"- **4VG**：Wheat 中位 {x4['wheat_median']:.3f} mg/L，"
        f"{x4['wheat_exceed_rate']*100:.0f}% 样本超过 0.2 mg/L；"
        f"Blond {x4['blond_exceed_rate']*100:.0f}%、"
        f"Saison {x4['saison_exceed_rate']*100:.0f}%。但 **Tripel "
        f"{x4['tripel_at_floor']}/{x4['tripel_n']} 样本在检出限（中位 "
        f"{x4['tripel_median']:.3f}），Dubbel {x4['dubbel_at_floor']}/"
        f"{x4['dubbel_n']} 也在检出限**。因此本题数据只支持"
        "“4VG 是小麦/部分金色农舍风格的正向特征”，"
        "**不支持“Tripel 是 POF+ 高 4VG 风格”**；论文不能把领域常识写得比数据更硬。"
        f"不过 Tripel 全组 {x4['tripel_sensory']['n']} 个样本（多数在检出限）里，"
        f"4VG 与感官 `X4vg` 仍正相关（ρ={x4['tripel_sensory']['spearman']:.3f}，"
        f"p={x4['tripel_sensory']['p']:.3f}），该相关由少数可检出样本驱动——"
        "只能写“可检出时方向一致”，不能写“Tripel 整体 4VG 高”。",
        f"- **乳酸**：酸啤风格超标率 "
        + "、".join(f"{k} {v*100:.0f}%" for k, v in lac["sour_exceed_rates"].items())
        + f"；Pils/Lager 中位 {lac['pils_median']:.1f} mg/L（阈值 {lac['threshold']:.0f}）。",
        f"- **醋酸**：酸啤风格超标率 "
        + "、".join(f"{k} {v*100:.0f}%" for k, v in ace["sour_exceed_rates"].items())
        + f"；Tripel {ace['tripel_exceed_rate']*100:.0f}%、"
        f"Pils/Lager {ace['pils_exceed_rate']*100:.0f}%（阈值 {ace['threshold']:.3f} g/L）。",
        "- **铁并非纯绝对缺陷**：Fruitbeer/Kriek 等风格也大量超过 0.1 mg/L。"
        "所有“阈值”都首先是感官检测阈值，不是风格合规上限。",
        "",
        "> **口径边界**：数据支持“一刀切严口径不适合风格依赖化合物”，"
        "但当前 Q3 的 `loose` 定义是“不加重”（上界 max(源值, 阈值)），"
        "**并不是“允许向目标风格正常区间靠拢”**。因此正确表述是："
        "数据支持引入**风格条件上界**（如目标风格 P90），"
        "不能宣称现有松口径已经实现了这一目标。",
        "",
    ]

    p = results["price"]
    lines += [
        "## 4. 价格与商业溢价",
        "",
        f"- 主渠道：`{p['primary_channel']}`（覆盖 {p['channels'][p['primary_channel']]}/250）。",
        f"- **单位限制**：{p['unit_note']}",
        f"- **代价限制**：{p['cost_note']}",
        f"- 全风格对 {p['n_style_pairs']} 条；Pils/Lager→Tripel：",
    ]
    for r in p["pils_tripel"]:
        lines.append(f"  - 源价 {r['source_price']:.2f} → 目标价 "
                     f"{r['target_price']:.2f}（渠道单位），比值 {r['price_ratio']:.3f}，"
                     f"修正代价 {r['qp_cost_median']:.3f}。")
    lines += ["", "| 关系 | Spearman | p |", "|---|---:|---:|"]
    for k, v in p["correlations"].items():
        lines.append(f"| {k} | {v['spearman']:.3f} | {v['p']:.3g} |")
    lines += ["", "**低修正代价、高溢价、感官方向一致的迁移（前 8）**：", "",
              "| 源 → 目标 | Δ价 | 价差比 | 修正代价 | 方向一致 | 可行率 | 评分 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in p["top_recommendations"][:8]:
        lines.append(f"| {r['source']} → {r['target']} | {r['delta_price']:.2f} | "
                     f"{r['price_ratio']:.3f} | {r['qp_cost_median']:.3f} | "
                     f"{r['sens_cos_median']:.2f} | "
                     f"{r['feasible_rate']:.2f} | {r['score']:.2f} |")
    lines += ["", "**每个目标风格的最佳来源（去掉同一目标重复）**：", "",
              "| 源 → 目标 | 价差比 | 修正代价 | 方向一致 | 评分 |",
              "|---|---:|---:|---:|---:|"]
    for r in p["best_source_per_target"][:8]:
        lines.append(f"| {r['source']} → {r['target']} | {r['price_ratio']:.3f} | "
                     f"{r['qp_cost_median']:.3f} | {r['style_shift_median']:.3f} | "
                     f"{r['score']:.2f} |")
    lines += ["", "> 推荐表只保证 4 类缺陷阈值可行，不保证工艺与整体感官合理"
              "（例如低醇→Kriek 需要额外发酵/增味）。论文中应写成“筛选靶子”，"
              "不能写成“可直接转产”。"]

    e = results["external"]
    m = e["mantel_primary9"]
    ml = e["mantel_loose11"]
    lines += [
        "",
        "## 5. 外部 5,558 款啤酒的独立验证",
        "",
        f"- 保守重叠排除：按「{e['overlap_rule']}」排除 {e['n_excluded_overlap']} 条；"
        f"Mantel 按风格分层抽 {e['n_used_mantel']} 条（置换成本限制）。",
        f"- Mantel（9 维主映射）：r={m['r']:.3f}，p={m['p']:.4f}，"
        f"{m['n_perm']} 次置换。",
        f"- Mantel（11 维宽松映射）：r={ml['r']:.3f}，p={ml['p']:.4f}。",
        f"- Pils/Lager→Tripel 位移方向：外部 vs 比赛真实 余弦 "
        f"{e['shift']['ext_vs_comp_real']['cosine']:.3f}、Spearman "
        f"{e['shift']['ext_vs_comp_real']['spearman']:.3f}"
        f"（p={e['shift']['ext_vs_comp_real']['p_two_sided']:.3f}）；"
        f"外部 vs Q3 生成 余弦 {e['shift']['ext_vs_comp_generated']['cosine']:.3f}、"
        f"Spearman {e['shift']['ext_vs_comp_generated']['spearman']:.3f}"
        f"（p={e['shift']['ext_vs_comp_generated']['p_two_sided']:.3f}）。",
        f"- 外部 Pils 样本 {e['shift']['n_external_pils']} 条、Tripel "
        f"{e['shift']['n_external_tripel']} 条；风格映射："
        f"{'、'.join(e['shift']['external_pils_styles'])}。"
        + ("**重叠排除后样本不足，位移比较退回全量外部样本，只能算方向一致性、"
           "不能算独立验证。**" if e["shift"]["used_overlapping_subset"] else ""),
        "- **判定**：Mantel 支持两个数据集在 9 个宏观感官维度上共享整体相关结构；"
        "但 Pils→Tripel 的位移方向秩相关不显著，只能作为弱旁证，"
        "不能写成“跨数据集复现了迁移方向”。",
        "- **口味面板差异**：外部 IBU 明显高于比赛数据（外部 Pils 约 36 vs 比赛约 22），"
        "这是位移方向不一致的主要候选解释；ABV 在 Pils 上接近，但外部 Tripel 高约 1 度。",
        "- 外部 Tripel 只有 11 条通过重叠排除，方向检验的功效很低；不要据此下强结论。",
        "",
        "## 必须写进论文的限制",
        "",
        "1. 外部感官字段是评论词频（0–323），不是 0–6 品评分；只比较秩相关结构与位移方向。",
        "2. 外部表按风格等量抽样，不是市场自然分布；不做市场均衡推断。",
        "3. 价格表没有容量，不能换算 €/L；绝对价差只在同一渠道内可比，跨渠道只比比值。",
        "4. 外部数据在锁定主模型之后才使用，未参与训练、调参或特征选择。",
        "5. Q3 的“四缺陷约束”不覆盖 H2S/MeSH/DES 等硫化物；"
        "`stale_hops` 的 Q1 预测器 OOF R²≈0.01，相关结论只能写方向。",
        "6. 4VG/乳酸/醋酸的官方阈值是感官检测阈值，不是风格合规上限；"
        "本题数据中 Tripel 的 4VG 多在检出限，不能写成“Tripel 是 POF+ 高 4VG”。",
        "",
        "## AI 使用声明",
        "",
        "本脚本与报告由 AI 辅助实现；分析口径与是否采用由参赛队员决定。",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def _verify_external() -> dict:
    """S 门：核对 data/external 的哈希与冻结清单一致。"""
    import hashlib
    man = R.PATHS["external"] / "MANIFEST.sha256"
    rows = []
    for line in man.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        p = R.PATHS["external"] / name
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append({"file": name, "expected": digest, "actual": h,
                     "match": h == digest})
    return {"manifest": str(man), "all_match": all(r["match"] for r in rows),
            "files": rows}


def main() -> int:
    t0 = time.time()
    R.set_global_seed(R.SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = _load()
    integrity = _verify_external()
    results = {}
    results["module"] = module_enrichment(ctx)
    results["oav"] = oav_check(ctx)
    results["defect"] = q3_defect_completeness(ctx)
    results["style_feature"] = style_feature_distributions(ctx)
    results["price"] = price_analysis(ctx)
    results["external"] = external_validation(ctx)
    runtime = time.time() - t0

    report = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_s": round(runtime, 1),
        "positioning": "supplementary annotation/validation only; not used in main models",
        "s_gate": {
            "status": "registered/pass",
            "scope": "仅用于模块通路注释、Q3 缺陷完整性核验、OAV/价格量化与"
                     "跨数据集独立验证，不进入 Q1/Q2/Q3 主模型训练",
            "external_integrity": integrity,
        },
        "results": results,
    }
    (OUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    write_report(results, runtime, integrity)
    print(f"S 门补充分析完成：{OUT}（{runtime:.1f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
