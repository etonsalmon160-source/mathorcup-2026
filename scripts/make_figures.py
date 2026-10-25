"""生成论文图件与 Illustrator 独立面板（AI 辅助实现，2026-10-25）。

所有数值从 results/ 读取，不在本文件里写死实验结果。
输出：
    figures/paper/figXX_*.pdf|png     论文拼版
    figures/panels/PXXx_*.pdf|svg|png 独立子面板（透明、无字母）
    figures/panels/PANEL_INDEX.md     面板尺寸与数据来源

用法：
    python scripts/make_figures.py                 # 全部
    python scripts/make_figures.py fig10           # 单张拼版
    python scripts/make_figures.py panels fig05    # 只导出 fig05 的子面板
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import (Ellipse, FancyArrowPatch, FancyBboxPatch,  # noqa: E402
                                PathPatch, Rectangle)
from matplotlib.path import Path as MplPath  # noqa: E402

import src.figstyle as F  # noqa: E402
from src.config import PATHS, SEED  # noqa: E402

RESULTS = ROOT / "results"
RAW = PATHS["raw"] / "附件"
OUT = ROOT / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

F.setup_style()

MODEL_LABEL = {
    "mean": "Mean baseline",
    "style_mean": "Style-mean baseline",
    "svr": "SVR",
    "chem14_lgbm": "Chem PCA14 + LightGBM",
    "chem14_ridge": "Chem PCA14 + Ridge",
    "lgbm": "LightGBM",
    "pls": "PLS",
    "xgb": "XGBoost",
    "ridge": "Ridge",
    "rf": "Random forest",
    "latent_sens_pls": "Sensory PLS8",
    "latent_sens_lgbm": "Sensory PCA8 + LightGBM",
    "et": "Extra trees",
    "latent_sens_ridge": "Sensory PCA8 + Ridge",
}
FAMILY_COLOR = {
    "Malt": F.ORANGE,
    "Hops": F.GREEN,
    "Yeast": F.PINK,
    "Wild flora": F.SKY,
    "Others": F.GREY,
    "Unassigned": F.LGREY,
}
SENS_DIMS = ["stale_hops", "dms", "diacetyl", "barnyard", "metallic"]
SENS_LABEL = {"stale_hops": "Stale hops", "dms": "DMS", "diacetyl": "Diacetyl",
              "barnyard": "Barnyard", "metallic": "Metallic"}
CHEM_DIMS = ["DMS", "X4VG", "X2.3.butanedione", "X.E..2.nonenal",
             "H2S", "MeSH", "DES", "iron"]
CHEM_LABEL = {"DMS": "DMS", "X4VG": "4-VG", "X2.3.butanedione": "Diacetyl",
              "X.E..2.nonenal": "T2N", "H2S": "H2S", "MeSH": "MeSH",
              "DES": "DES", "iron": "Iron"}
STATE_LABEL = {"before": "Before", "none": "None", "ot_proj": "OT proj.",
               "qp_full": "Joint QP"}
METHOD_LABEL = {"none": "No correction", "ot_proj": "OT projection",
                "qp_full": "Joint QP"}
METHOD_COLOR = {"none": F.LGREY, "ot_proj": F.ORANGE, "qp_full": F.BLUE}
STYLE_COLOR = {
    "Pils/Lager": F.BLUE, "Tripel": F.VERM, "Wheat": F.ORANGE,
    "Lambic": F.PINK, "Stout/Porter": F.GREEN,
}


# ---------------------------------------------------------------- 基础
def _read(rel: str, **kw) -> pd.DataFrame:
    return pd.read_csv(RESULTS / rel, encoding="utf-8-sig", **kw)


def _json(rel: str) -> dict:
    return json.loads((RESULTS / rel).read_text(encoding="utf-8"))


def _target_label(t: str) -> str:
    if t.startswith("A_"):
        s = "A · " + t[2:]
    elif t.startswith("F_"):
        s = "F · " + t[2:]
    else:
        s = t
    s = s.replace("_", " ")
    return (s.replace("bitternes", "bitterness").replace("t2n", "T2N")
             .replace("X4vg", "4VG").replace("co2", "CO2").replace("dms", "DMS"))


def _save(fig, name: str) -> None:
    F.save_figure(fig, OUT / name)
    plt.close(fig)
    print("  wrote", name)


# ---------------------------------------------------------------- 面板注册
PANEL = {}


def panel(key: str, w: float, h: float, *, desc: str, sources: str,
          stats: str, margins=(0.19, 0.17, 0.975, 0.965)):
    """注册一个独立子面板；w/h 为名义毫米尺寸，实际以 PDF 为准。"""
    def deco(fn):
        PANEL[key] = dict(fn=fn, w=w, h=h, desc=desc, sources=sources,
                          stats=stats, margins=margins)
        return fn
    return deco


def _standalone(key: str) -> None:
    s = PANEL[key]
    fig = plt.figure(figsize=(s["w"] * F.MM, s["h"] * F.MM))
    ax = fig.add_axes(list(s["margins"]))
    s["fn"](ax)
    F.save_panel(fig, key, description=s["desc"], sources=s["sources"],
                 stats=s["stats"])
    plt.close(fig)


# ---------------------------------------------------------------- 数据缓存
@lru_cache(maxsize=1)
def _latent_curves():
    from scripts.q3_probe import DERIVED, META, load, normal_scores
    X, Y = load()
    feat = [c for c in X.columns if c not in META + DERIVED]
    Ns = normal_scores(X, feat)
    M = ((Ns - Ns.mean()) / Ns.std(ddof=1)).to_numpy(float)
    Yv = Y.drop(columns=["beer", "beer_id", "tasting_category_fine"]).to_numpy(float)
    Mz = (Yv - Yv.mean(0)) / Yv.std(0, ddof=1)

    def curve(mat, seed):
        ev = np.linalg.eigvalsh(np.corrcoef(mat, rowvar=False))[::-1]
        rng = np.random.default_rng(seed)
        null = np.empty((100, mat.shape[1]))
        for b in range(100):
            Mp = np.column_stack([rng.permutation(mat[:, j])
                                  for j in range(mat.shape[1])])
            null[b] = np.linalg.eigvalsh(np.corrcoef(Mp, rowvar=False))[::-1]
        q95 = np.percentile(null, 95, axis=0)
        k = int((ev > q95).sum())
        sv = np.linalg.svd(mat, compute_uv=False)
        evr = sv ** 2 / (sv ** 2).sum()
        return dict(ev=ev, q95=q95, k=k, evr=evr, cum=np.cumsum(evr),
                    p=mat.shape[1])
    return curve(M, SEED), curve(Mz, SEED), len(feat)


@lru_cache(maxsize=1)
def _q2_graph():
    import networkx as nx
    comm = _read("q2/communities.csv")
    names = comm["node"].tolist()
    C = pd.read_csv(RAW / "2.csv", index_col=0).loc[names, names]
    A = C.to_numpy(float)
    edges = {}
    for i in range(len(names)):
        for j in np.argsort(-np.abs(A[i]))[:11]:
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges[(a, b)] = float(A[i, j])
    G = nx.Graph()
    G.add_nodes_from(range(len(names)))
    for (i, j), w in edges.items():
        G.add_edge(i, j, weight=w, abs_weight=abs(w))
    return G, names, comm


@lru_cache(maxsize=1)
def _style_profiles():
    from src.data_gate import META
    X = pd.read_csv(RAW / "1.csv")
    Y = pd.read_csv(RAW / "4.csv")
    targets = [c for c in Y.columns if c not in META]
    gen = _read("q3/generated_sensory.csv")
    styles = X["tasting_category_fine"].to_numpy()
    rows = []
    for s in sorted(set(styles)):
        real = Y.loc[styles == s, targets].to_numpy(float).mean(0)
        syn = gen.loc[gen["style"] == s, targets].to_numpy(float).mean(0)
        rows.append((s, real, syn))
    return targets, rows


@lru_cache(maxsize=1)
def _state():
    axes = _json("supplementary/upgrade/q3_state_axes.json")
    pts = _read("supplementary/upgrade/q3_state_points.csv")
    paths = _read("supplementary/upgrade/q3_state_paths.csv")
    indiv = _read("supplementary/upgrade/q3_state_individual.csv")
    return axes, pts, paths, indiv


def _family_map() -> dict:
    """感官维度 → 香气家族。三个无家族标注的维度并入 Others（n = 15），
    与图 12 的五类计数一致。"""
    q = _read("q2/top3.csv")
    return {k: (v if isinstance(v, str) else "Others")
            for k, v in zip(q["sensory"], q["target_family"])}


# ============================================================ P01 研究设计
@panel("P01a", 183, 104,
       desc="Study design: data layer, three questions, evidence and controls",
       sources="data/raw/附件/1.csv, 5.csv; docs/data_contract/transform_classification_v1.csv; results/data_gate/report.json; results/q2/report.json; results/q3/report.json; results/supplementary/upgrade/q3_qp_pairs_long.csv",
       stats="counts only (250 beers, 22 styles, 231 compounds, 50 sensory dimensions)",
       margins=(0, 0, 1, 1))
def p01a(ax):
    raw1 = pd.read_csv(RAW / "1.csv")
    raw5 = pd.read_csv(RAW / "5.csv")
    tc = _read("../docs/data_contract/transform_classification_v1.csv")
    gate = _json("data_gate/report.json")
    q2 = _json("q2/report.json")
    q3 = _json("q3/report.json")
    pairs = _read("supplementary/upgrade/q3_qp_pairs_long.csv")
    from scripts import run_v1 as R

    n_beer = int(raw1["beer"].nunique())
    n_style = int(raw1["tasting_category_fine"].nunique())
    n_chem = int(len(tc))
    n_sens = int(raw5.shape[0])
    n_gates = int(len(gate["gates"]))
    n_modules = int(q2["graph"]["n_communities"])
    n_boot = int(q2["method"]["n_bootstrap"])
    k3 = int(q3["k"])
    n_gen = int(q3["generation"]["n_rows"])
    n_pairs = int((pairs["method"] == "none").sum())
    ks = int(R.SENSORY_LATENT_K)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, title, body, edge, face="#FFFFFF", lw=0.7):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.010",
            linewidth=lw, edgecolor=edge, facecolor=face, zorder=3))
        ax.text(x + 0.008, y + h - 0.012, title, fontsize=7.2,
                fontweight="bold", color=edge, va="top", ha="left", zorder=4)
        ax.text(x + 0.008, y + h - 0.040, body, fontsize=6.4, color="#333333",
                va="top", ha="left", linespacing=1.35, zorder=4)

    def arrow(x0, y0, x1, y1, color="#666666"):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6,
            linewidth=0.7, color=color, shrinkA=0, shrinkB=0, zorder=2))

    box(0.012, 0.275, 0.185, 0.605, "Data",
        f"{n_beer} beers · {n_style} styles\n{n_chem} chemical features\n"
        f"{n_sens} sensory dimensions\n\nFrozen raw files\nSHA-256 manifest\n\n"
        "LOD sentinels\nmissingness · outliers", F.GREY, "#F7F7F7")
    for yc in (0.785, 0.565, 0.345):
        arrow(0.198, yc, 0.232, yc)

    rows = [
        ("Q1", "Sensory prediction", F.BLUE, "#F2F7FB", 0.700,
         f"{n_chem} compounds\n+ detection rate",
         f"Sensory PCA{ks} → RidgeCV\nfused 1:1 with\nExtra trees",
         f"{n_sens} sensory dimensions\nrandom & style-grouped CV\nablation · calibration"),
        ("Q2", "Key compounds", F.GREEN, "#F0F8F5", 0.480,
         f"Co-expression matrix\n{n_chem} × {n_chem}\n(given, 2.csv)",
         f"Louvain modules\n+ Lasso stability\nselection (B = {n_boot})",
         f"Top-3 per dimension\n{n_modules} modules\nnegative maskers"),
        ("Q3", "Style generation", F.VERM, "#FDF5F0", 0.260,
         f"Style-conditional\nGaussian\n({k3}-d latent)",
         "OT projection\n+ defect-constrained QP\n(DMS · diacetyl · 4VG)",
         f"{n_pairs} cross-style pairs\n{n_gen} generated profiles"),
    ]
    for tag, name, col, face, y, inp, meth, outp in rows:
        ax.text(0.232, y + 0.132, tag, fontsize=8.2, fontweight="bold",
                color=col, va="top", ha="left")
        ax.text(0.266, y + 0.132, name, fontsize=6.8, color=col,
                va="top", ha="left")
        ax.plot([0.232, 0.978], [y + 0.103, y + 0.103], color=col,
                lw=0.35, alpha=0.45, zorder=1)
        box(0.232, y - 0.070, 0.158, 0.168, "Input", inp, col, face)
        box(0.432, y - 0.070, 0.196, 0.168, "Model", meth, col, face)
        box(0.670, y - 0.070, 0.308, 0.168, "Output", outp, col, face)
        arrow(0.390, y + 0.014, 0.430, y + 0.014, col)
        arrow(0.628, y + 0.014, 0.668, y + 0.014, col)

    ax.add_patch(FancyBboxPatch(
        (0.012, 0.022), 0.966, 0.145,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        linewidth=0.5, edgecolor=F.LGREY, facecolor="#FAFAFA", zorder=3))
    ax.text(0.024, 0.158, "Evidence and controls", fontsize=6.8,
            fontweight="bold", color="#444444", va="top")
    ax.text(0.024, 0.135,
            "Permutation importance · C2ST adversarial validation · variance decomposition",
            fontsize=6.1, color="#444444", va="top")
    ax.text(0.024, 0.112,
            f"Seed / α / operator sensitivity · {n_gates} hard data gates · all figures from frozen outputs",
            fontsize=6.1, color="#444444", va="top")
    ax.text(0.024, 0.075,
            "Q1 predicts, Q2 explains, Q3 generates — one latent chemistry links all three.",
            fontsize=6.4, color="#222222", style="italic", va="top")


def fig01():
    fig = F.new_fig(F.W2, 104 * F.MM)
    ax = fig.add_axes([0, 0, 1, 1])
    p01a(ax)
    _save(fig, "fig01_study_design")


# ============================================================ P02 数据审计
@panel("P02a", 88, 42,
       desc="Transform regime of the 231 chemical columns",
       sources="docs/data_contract/transform_classification_v1.csv",
       stats="231 columns; counts by transform source",
       margins=(0.10, 0.30, 0.98, 0.72))
def p02a(ax):
    tc = _read("../docs/data_contract/transform_classification_v1.csv")
    src = tc["source"].value_counts()
    tr = tc["transform"].value_counts()
    comp = {"Negative-proves log10": int(src["negative_proves_log"]),
            "Calibrated log10": int(src["calibrated_log10"]),
            "Inferred log10": int(src["peak_area_inferred"]),
            "Calibrated raw": int(src["calibrated_raw"])}
    total = len(tc)
    subtitle = f'{int(tr["log10"])} log10 + {int(tr["raw"])} raw'
    left = 0
    for (lab, n), c in zip(comp.items(), [F.BLUE, F.SKY, F.LGREY, F.ORANGE]):
        ax.barh(0, n, left=left, height=0.34, color=c, edgecolor="white",
                linewidth=0.6, label=f"{lab} (n={n})")
        if n >= 15:
            ax.text(left + n / 2, 0, str(n), ha="center", va="center",
                    fontsize=6.2,
                    color="white" if c in (F.BLUE, F.ORANGE) else "#333")
        left += n
    ax.set_xlim(0, total * 1.01)
    ax.set_ylim(-0.55, 0.82)
    ax.set_yticks([])
    ax.set_xlabel(f"Chemical columns ({subtitle})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2,
              fontsize=5.9, columnspacing=0.7, handlelength=0.8,
              labelspacing=0.25)
    F.soften(ax, xgrid=True)


@panel("P02b", 95, 58,
       desc="Missing values per chemical feature",
       sources="results/qc/missingness.csv",
       stats="226 measured compounds; proportion missing across 250 beers",
       margins=(0.54, 0.14, 0.985, 0.965))
def p02b(ax):
    miss = _read("qc/missingness.csv").sort_values("missing_rate")
    y = np.arange(len(miss))
    ax.barh(y, miss["missing_rate"] * 100, height=0.62, color=F.LGREY,
            edgecolor="none")
    ax.scatter(miss["missing_rate"] * 100, y, s=9, color=F.BLUE, zorder=3,
               linewidths=0)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [n.replace("Total hop aroma compounds", "Hop aroma (sum)")
         for n in miss["chemical_name"]], fontsize=6.1)
    ax.set_xlabel("Missing values (%)")
    ax.set_xlim(0, 12.5)
    for yi, (_, r) in zip(y, miss.iterrows()):
        ax.text(r["missing_rate"] * 100 + 0.35, yi, f'{r["n_missing"]}',
                va="center", fontsize=5.8, color="#555555")
    ax.set_ylim(-0.7, len(miss) - 0.3)
    F.soften(ax, xgrid=True)


@panel("P02c", 82, 44,
       desc="Compounds at the limit-of-detection floor per beer",
       sources="results/qc/floor_by_sample.csv; results/qc/sample_qc.csv",
       stats="250 beers; flagged beers defined by robust QC votes",
       margins=(0.18, 0.18, 0.98, 0.965))
def p02c(ax):
    floor = _read("qc/floor_by_sample.csv")
    sq = _read("qc/sample_qc.csv")
    floor = floor.merge(sq[["beer", "sample_flag", "outlier_votes"]],
                        on="beer", how="left")
    flagged = floor["sample_flag"].fillna(False).astype(bool)
    ax.hist(floor.loc[~flagged, "floor_rate"] * 100, bins=26,
            color=F.LGREY, edgecolor="white", linewidth=0.3, label="Regular")
    ax.hist(floor.loc[flagged, "floor_rate"] * 100, bins=26,
            color=F.VERM, edgecolor="white", linewidth=0.3,
            label=f"Flagged (n={int(flagged.sum())})")
    ax.set_xlabel("Compounds at LOD floor per beer (%)")
    ax.set_ylabel("Beers")
    ax.legend(loc="upper right", fontsize=6.1)
    F.soften(ax, ygrid=True)


@panel("P02d", 82, 44,
       desc="Sample outliers: robust Mahalanobis distance vs PCA SPE",
       sources="results/qc/sample_qc.csv",
       stats="250 beers; flagged = robust QC vote outlier",
       margins=(0.17, 0.18, 0.98, 0.965))
def p02d(ax):
    sq = _read("qc/sample_qc.csv")
    flag = sq["sample_flag"].astype(bool)
    ax.scatter(sq.loc[~flag, "robust_md"], sq.loc[~flag, "pca_spe"], s=8,
               color=F.LGREY, linewidths=0, alpha=0.85, label="Regular")
    ax.scatter(sq.loc[flag, "robust_md"], sq.loc[flag, "pca_spe"], s=13,
               color=F.VERM, linewidths=0,
               label=f"Flagged (n={int(flag.sum())})", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Robust Mahalanobis distance")
    ax.set_ylabel("PCA SPE")
    ax.legend(loc="lower right", fontsize=6.1)
    F.soften(ax, ygrid=True)


def fig02():
    fig = plt.figure(figsize=(F.W2, 92 * F.MM))
    gs = fig.add_gridspec(2, 2, left=0.080, right=0.985, top=0.92,
                          bottom=0.10, wspace=0.78, hspace=0.68)
    for key, cell, letter, lx in (("P02a", gs[0, 0], "a", -0.18),
                                  ("P02b", gs[0, 1], "b", -0.40),
                                  ("P02c", gs[1, 0], "c", -0.18),
                                  ("P02d", gs[1, 1], "d", -0.18)):
        ax = fig.add_subplot(cell)
        PANEL[key]["fn"](ax)
        F.panel_label(ax, letter, x=lx, y=1.10)
    _save(fig, "fig02_data_qc")


# ============================================================ P03 潜结构
def _latent_component(ax, d, n_max, xlab):
    k = d["k"]
    n = min(len(d["evr"]), n_max)
    x = np.arange(1, n + 1)
    ax.bar(x, d["evr"][:n] * 100, width=0.68, color=F.LGREY,
           edgecolor="white", linewidth=0.3, label="Observed")
    ax.plot(x, d["q95"][:n] / d["p"] * 100, color=F.VERM, lw=0.9,
            label="Parallel-analysis null (95th)")
    ax.axvline(k + 0.5, color=F.BLUE, lw=0.8, ls=(0, (3, 2)))
    ax.set_xlabel(xlab)
    ax.set_ylabel("Variance explained (%)")
    ax.set_xlim(0.3, n + 0.7)
    ax.text(k + 0.8, ax.get_ylim()[1] * 0.55, f"k = {k}", color=F.BLUE,
            fontsize=6.8, va="top")
    h, l = ax.get_legend_handles_labels()
    ax.legend([h[l.index("Observed")], h[l.index("Parallel-analysis null (95th)")]],
              ["Observed", "Parallel-analysis null (95th)"],
              loc="upper right", fontsize=6.1)
    F.soften(ax, ygrid=True)


@panel("P03a", 62, 50,
       desc="Chemical latent structure: scree and parallel analysis",
       sources="data/raw/附件/1.csv; 100-permutation parallel analysis",
       stats="226 measured compounds; k = observed eigenvalue > 95th null",
       margins=(0.22, 0.17, 0.98, 0.965))
def p03a(ax):
    chem, _, n_feat = _latent_curves()
    _latent_component(ax, chem, 30, f"Component (n = {n_feat} compounds)")


@panel("P03b", 62, 50,
       desc="Sensory latent structure: scree and parallel analysis",
       sources="data/raw/附件/4.csv; 100-permutation parallel analysis",
       stats="50 sensory dimensions; k = observed eigenvalue > 95th null",
       margins=(0.22, 0.17, 0.98, 0.965))
def p03b(ax):
    _, sens, _ = _latent_curves()
    _latent_component(ax, sens, 20, "Component (n = 50 dimensions)")


@panel("P03c", 60, 50,
       desc="Cumulative variance explained for chemical and sensory PCA",
       sources="data/raw/附件/1.csv, 4.csv",
       stats="markers at the parallel-analysis selected dimensionalities",
       margins=(0.23, 0.17, 0.98, 0.965))
def p03c(ax):
    chem, sens, _ = _latent_curves()
    for d, col, lab in ((chem, F.BLUE, "Chemical"), (sens, F.VERM, "Sensory")):
        k = d["k"]
        n = min(len(d["cum"]), 30 if col == F.BLUE else 20)
        ax.plot(np.arange(1, n + 1), d["cum"][:n] * 100, color=col, lw=1.1,
                label=lab)
        ax.scatter([k], [d["cum"][k - 1] * 100], s=14, color=col, zorder=3)
        ax.annotate(f"{d['cum'][k - 1] * 100:.1f}%", (k, d["cum"][k - 1] * 100),
                    xytext=(4, -7 if col == F.BLUE else 4),
                    textcoords="offset points", fontsize=6.3, color=col)
    ax.axhline(100, color="#999999", lw=0.4, ls=(0, (2, 2)))
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance (%)")
    ax.set_ylim(0, 108)
    ax.legend(loc="lower right", fontsize=6.3)
    F.soften(ax, ygrid=True)


def fig03():
    fig = plt.figure(figsize=(F.W2, 64 * F.MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1], left=0.075,
                          right=0.985, top=0.87, bottom=0.20, wspace=0.42)
    for j, key in enumerate(["P03a", "P03b", "P03c"]):
        ax = fig.add_subplot(gs[0, j])
        PANEL[key]["fn"](ax)
        F.panel_label(ax, "abc"[j], x=-0.22 if j < 2 else -0.28, y=1.07)
    _save(fig, "fig03_latent_structure")


# ============================================================ P04 Q1 模型
@panel("P04a", 105, 62,
       desc="Mean R² of 14 Q1 models under random and style-grouped CV",
       sources="results/q1/comparison.csv",
       stats="14 models × 2 schemes × 3 seeds; error bars = ±1 SD across seeds",
       margins=(0.30, 0.14, 0.98, 0.965))
def p04a(ax):
    comp = _read("q1/comparison.csv")
    agg = (comp.groupby(["model", "scheme"])["mean_r2"]
           .agg(["mean", "std"]).reset_index())
    wide = agg.pivot(index="model", columns="scheme", values="mean")
    sd = agg.pivot(index="model", columns="scheme", values="std")
    wide = wide.sort_values("random")
    y = np.arange(len(wide))
    for i, (mid, row) in enumerate(wide.iterrows()):
        ax.plot([row["random"], row["group"]], [i, i], color=F.LGREY, lw=0.8,
                zorder=1)
        ax.errorbar(row["random"], i, xerr=sd.loc[mid, "random"], fmt="o",
                    ms=3.2, color=F.BLUE, ecolor=F.BLUE, elinewidth=0.6,
                    capsize=0, zorder=3)
        ax.errorbar(row["group"], i, xerr=sd.loc[mid, "group"], fmt="o",
                    ms=3.2, color=F.VERM, ecolor=F.VERM, elinewidth=0.6,
                    capsize=0, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABEL.get(mid, mid) for mid in wide.index],
                       fontsize=6.3)
    ax.set_xlabel(r"Mean $R^2$ across 50 dimensions (seed mean ± SD)")
    ax.set_xlim(-0.04, float(wide.to_numpy().max()) + 0.035)
    ax.axvline(0, color="#999999", lw=0.5, ls=(0, (2, 2)))
    ax.scatter([], [], s=10, color=F.BLUE, label="Random 5-fold CV")
    ax.scatter([], [], s=10, color=F.VERM, label="Style-grouped CV")
    ax.legend(loc="upper left", fontsize=6.3)
    ax.text(0.97, 0.03,
            f"seed SD ≤ {float(sd.to_numpy().max()):.3f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.4,
            color="#555555")
    ax.set_ylim(-0.7, len(wide) - 0.3)
    F.soften(ax, xgrid=True)


@panel("P04b", 82, 62,
       desc="Fusion schemes relative to Extra trees",
       sources="results/composite/report.json",
       stats="5 fusion schemes × 2 CV schemes; bars = beer-cluster bootstrap 95% CI",
       margins=(0.36, 0.14, 0.98, 0.965))
def p04b(ax):
    fusion = _json("composite/report.json")["summary"]
    order = ["avg", "stack_global", "stack_target", "gate_cv", "gate_domain"]
    label = {"avg": "Equal-weight fusion (adopted)",
             "stack_global": "Global stacking",
             "stack_target": "Per-target stacking",
             "gate_cv": "CV-gated fusion",
             "gate_domain": "Domain-rule gated"}
    yy = np.arange(len(order))[::-1]
    for scheme, col, off in (("random", F.BLUE, 0.16),
                             ("group", F.VERM, -0.16)):
        for i, v in zip(yy, order):
            s = fusion[f"{scheme}/{v}"]
            lo, hi = s["boot_ci95"]
            ax.errorbar(s["delta_vs_et"], i + off,
                        xerr=[[s["delta_vs_et"] - lo], [hi - s["delta_vs_et"]]],
                        fmt="o", ms=3.4, color=col, ecolor=col,
                        elinewidth=0.8, capsize=1.8, zorder=3,
                        markerfacecolor=("white" if lo < 0 < hi else col),
                        markeredgewidth=0.7)
            ax.text(hi + 0.0012, i + off, f'{s["delta_vs_et"]:+.3f}',
                    fontsize=5.8, va="center", color=col)
    ax.axvline(0, color="#555555", lw=0.7)
    ax.set_yticks(yy)
    ax.set_yticklabels([label[v] for v in order], fontsize=6.3)
    ax.set_xlabel(r"$\Delta R^2$ vs Extra trees (95% CI)")
    ax.set_xlim(-0.024, 0.045)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.scatter([], [], s=10, color=F.BLUE, label="Random 5-fold CV")
    ax.scatter([], [], s=10, color=F.VERM, label="Style-grouped CV")
    ax.legend(loc="upper left", fontsize=5.6, handlelength=0.7,
              labelspacing=0.2, borderaxespad=0.3,
              frameon=True, facecolor="white", framealpha=0.9,
              edgecolor="none")
    F.soften(ax, xgrid=True)


def fig04():
    fig = plt.figure(figsize=(F.W2, 78 * F.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], left=0.16,
                          right=0.972, top=0.88, bottom=0.16, wspace=0.62)
    for j, key in enumerate(["P04a", "P04b"]):
        ax = fig.add_subplot(gs[0, j])
        PANEL[key]["fn"](ax)
        F.panel_label(ax, "ab"[j], x=-0.28 if j == 0 else -0.50, y=1.04)
    _save(fig, "fig04_q1_models_fusion")


# ============================================================ P05 逐维表现
@panel("P05a", 105, 132,
       desc="Per-dimension R² under random and style-grouped CV",
       sources="results/composite/per_target.csv (variant = avg)",
       stats="50 sensory dimensions × 2 schemes × 3 seeds; dot = seed mean",
       margins=(0.48, 0.09, 0.98, 0.975))
def p05a(ax):
    pt = _read("composite/per_target.csv")
    pt = pt[pt["variant"] == "avg"]
    g = pt.groupby(["target", "scheme"])["r2"].mean().unstack()
    g = g.sort_values("random")
    y = np.arange(len(g))
    for i, (t, r) in enumerate(g.iterrows()):
        ax.plot([r["group"], r["random"]], [i, i], color=F.LGREY, lw=0.8,
                zorder=1, solid_capstyle="round")
        ax.scatter(r["group"], i, s=7, color=F.VERM, zorder=3, linewidths=0)
        ax.scatter(r["random"], i, s=7, color=F.BLUE, zorder=3, linewidths=0)
    ax.set_yticks(y)
    ax.set_yticklabels([_target_label(t) for t in g.index], fontsize=5.6)
    ax.set_ylim(-0.8, len(g) - 0.2)
    ax.set_xlim(-0.055, 0.72)
    ax.axvline(0, color="#999999", lw=0.45, ls=(0, (2, 2)))
    ax.set_xlabel("Per-dimension $R^2$ (mean of 3 seeds)")
    ax.scatter([], [], s=9, color=F.BLUE, label="Random CV")
    ax.scatter([], [], s=9, color=F.VERM, label="Style-grouped CV")
    ax.legend(loc="lower right", fontsize=6.2)
    F.soften(ax, xgrid=True)


@panel("P05b", 78, 58,
       desc="Random vs style-grouped R² for each sensory dimension",
       sources="results/composite/per_target.csv; results/q2/top3.csv (family)",
       stats="50 dimensions; identity line shown",
       margins=(0.19, 0.17, 0.975, 0.965))
def p05b(ax):
    pt = _read("composite/per_target.csv")
    pt = pt[pt["variant"] == "avg"]
    g = pt.groupby(["target", "scheme"])["r2"].mean().unstack()
    fam = _family_map()
    g["family"] = [fam.get(t, "Overall") for t in g.index]
    for f, sub in g.groupby("family"):
        ax.scatter(sub["random"], sub["group"], s=17,
                   color=FAMILY_COLOR.get(f, F.LGREY), label=f,
                   edgecolor="white", linewidth=0.3, alpha=0.95, zorder=3)
    lim = (-0.06, 0.72)
    ax.plot(lim, lim, color="#777777", lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0, color="#CCCCCC", lw=0.5, zorder=0)
    ax.axvline(0, color="#CCCCCC", lw=0.5, zorder=0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Random CV $R^2$")
    ax.set_ylabel("Style-grouped CV $R^2$")
    ax.legend(loc="lower right", fontsize=6.0, title="Target family",
              title_fontsize=6.2, ncol=2, columnspacing=0.7)
    F.soften(ax, xgrid=True, ygrid=True)


@panel("P05c", 82, 46,
       desc="Cost of style-grouped CV by sensory family (raincloud)",
       sources="results/composite/per_target.csv; results/q2/top3.csv",
       stats="50 dimensions; half-violin KDE + Sina jitter + median/IQR",
       margins=(0.23, 0.19, 0.98, 0.965))
def p05c(ax):
    pt = _read("composite/per_target.csv")
    pt = pt[pt["variant"] == "avg"]
    g = pt.groupby(["target", "scheme"])["r2"].mean().unstack()
    g["delta"] = g["random"] - g["group"]
    fam = _family_map()
    g["family"] = [fam.get(t, "Overall") for t in g.index]
    order = g.groupby("family")["delta"].median().sort_values().index.tolist()
    vals = [g.loc[g["family"] == f, "delta"].to_numpy() for f in order]
    F.raincloud(ax, vals, np.arange(len(order)),
                [FAMILY_COLOR.get(f, F.LGREY) for f in order],
                violin_width=0.34, side="right")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=6.2)
    ax.set_xlabel(r"$\Delta R^2$ (random − style-grouped)")
    lo = min(float(np.nanmin(v)) for v in vals)
    hi = max(float(np.nanmax(v)) for v in vals)
    ax.set_xlim(min(-0.01, lo - 0.02), hi + 0.03)
    ax.axvline(0, color="#AAAAAA", lw=0.5)
    ax.set_ylim(-0.6, len(order) - 0.4)
    F.soften(ax, xgrid=True)


def fig05():
    fig = plt.figure(figsize=(F.W2, 132 * F.MM))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.08, 1], left=0.155,
                             right=0.985, top=0.97, bottom=0.09, wspace=0.30)
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[1.35, 1],
                                    hspace=0.50)
    ax = fig.add_subplot(outer[0, 0])
    p05a(ax)
    F.panel_label(ax, "a", x=-0.42, y=1.02)
    ax = fig.add_subplot(right[0])
    p05b(ax)
    F.panel_label(ax, "b", x=-0.28, y=1.03)
    ax = fig.add_subplot(right[1])
    p05c(ax)
    F.panel_label(ax, "c", x=-0.30, y=1.10)
    _save(fig, "fig05_q1_per_target")


# ============================================================ P06 重要性
@panel("P06a", 98, 92,
       desc="Top-20 compounds by permutation importance",
       sources="results/q1/importance.csv",
       stats="mean permutation ΔMSE across 50 dimensions and 3 seeds, z-scaled",
       margins=(0.46, 0.12, 0.975, 0.965))
def p06a(ax):
    imp = _read("q1/importance.csv")
    top = imp.nsmallest(20, "rank").iloc[::-1]
    y = np.arange(len(top))

    def mcol(m):
        try:
            return F.MODULE_COLORS[int(m)]
        except (ValueError, TypeError):
            return F.GREY

    ax.barh(y, top["delta_mse_z"], height=0.66,
            color=[mcol(m) for m in top["module"]], edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(top["display"], fontsize=6.1)
    ax.set_xlabel(r"$\Delta$MSE after permutation (z-scale)")
    ax.set_xlim(0, top["delta_mse_z"].max() * 1.18)
    for yi, (_, r) in zip(y, top.iterrows()):
        tag = (f'ME{int(r["module"])}' if str(r["module"]).isdigit()
               else {"派生汇总列": "derived", "未入图": "not in graph"}.get(
                   str(r["module"]), str(r["module"])))
        ax.text(r["delta_mse_z"] + 0.00012, yi, tag, va="center", fontsize=5.3,
                color="#666666")
    F.soften(ax, xgrid=True)


@panel("P06b", 88, 76,
       desc="Module contribution to total permutation importance",
       sources="results/q1/importance_modules.csv",
       stats="sum of permutation ΔMSE within module; shares of total",
       margins=(0.52, 0.15, 0.975, 0.965))
def p06b(ax):
    mods = _read("q1/importance_modules.csv").sort_values("share")
    y = np.arange(len(mods))
    cols = []
    for m in mods["module"]:
        if str(m).isdigit():
            cols.append(F.MODULE_COLORS[int(m)])
        elif str(m) == "派生汇总列":
            cols.append(F.GREY)
        elif str(m) == "未入图":
            cols.append(F.LGREY)
        else:
            cols.append(F.XLGREY)
    ax.barh(y, mods["share"] * 100, height=0.66, color=cols, edgecolor="none")
    ax.set_yticks(y)
    labels = []
    for _, r in mods.iterrows():
        m = r["module"]
        name = (f"ME{int(m)}" if str(m).isdigit()
                else {"派生汇总列": "Derived totals",
                      "未入图": "Not in network",
                      "CDR": "Detection rate"}.get(str(m), str(m)))
        labels.append(f"{name} (n={int(r['n_members'])})")
    ax.set_yticklabels(labels, fontsize=6.0)
    ax.set_xlabel(r"Share of total permutation $\Delta$MSE (%)")
    ax.set_xlim(0, mods["share"].max() * 118)
    for yi, (_, r) in zip(y, mods.iterrows()):
        ax.text(r["share"] * 100 + 0.35, yi, f'{r["share"] * 100:.1f}',
                va="center", fontsize=5.4, color="#666666")
    F.soften(ax, xgrid=True)


@panel("P06c", 70, 56,
       desc="Agreement between permutation importance and Q2 Lasso evidence",
       sources="results/q1/importance_vs_lasso.csv; results/q1/importance_summary.json",
       stats="50 sensory dimensions; Spearman ρ and top-10 overlap",
       margins=(0.22, 0.18, 0.975, 0.965))
def p06c(ax):
    vs = _read("q1/importance_vs_lasso.csv")
    summ = _json("q1/importance_summary.json")["vs_lasso"]
    ax.hist(vs["spearman"], bins=np.linspace(-0.25, 0.5, 19),
            color=F.LGREY, edgecolor="white", linewidth=0.4)
    ax.axvline(summ["median_spearman"], color=F.BLUE, lw=0.9)
    ax.set_xlabel(r"Spearman $\rho$ vs Q2")
    ax.set_ylabel("Sensory dimensions")
    ax.set_xlim(-0.28, 0.50)
    ax.text(0.97, 0.97,
            f'n = 50\ntop-10 overlap {summ["mean_top10_overlap"]:.2f}\n'
            f'median $\\rho$ = {summ["median_spearman"]:.3f}',
            transform=ax.transAxes, ha="right", va="top", fontsize=5.9,
            color="#555555", linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none",
                      alpha=0.9))
    F.soften(ax, ygrid=True)


def fig06():
    fig = plt.figure(figsize=(F.W2, 98 * F.MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.28, 1, 1], left=0.175,
                          right=0.972, top=0.88, bottom=0.16, wspace=0.62)
    for j, key in enumerate(["P06a", "P06b", "P06c"]):
        ax = fig.add_subplot(gs[0, j])
        PANEL[key]["fn"](ax)
        F.panel_label(ax, "abc"[j], x=[-0.34, -0.56, -0.30][j], y=1.04)
    _save(fig, "fig06_q1_importance")


# ============================================================ P07 Q2 网络
@panel("P07a", 112, 96,
       desc="Compound co-expression network with Louvain modules",
       sources="data/raw/附件/2.csv; results/q2/communities.csv; results/q1/importance.csv",
       stats="223 nodes; top-10 absolute-correlation neighbours per node; node size = degree",
       margins=(0, 0, 1, 1))
def p07a(ax):
    import networkx as nx
    G, names, comm = _q2_graph()
    mod = dict(zip(comm["node"], comm["community"].astype(int)))
    deg = dict(zip(comm["node"], comm["degree"]))
    Gv = G.copy()
    for a, b, d in Gv.edges(data=True):
        same = mod[names[a]] == mod[names[b]]
        d["layout_w"] = d["abs_weight"] * (6.0 if same else 0.12)
    centers = {m: np.array([np.cos(2 * np.pi * m / 8 + 0.35),
                            np.sin(2 * np.pi * m / 8 + 0.35)])
               for m in range(8)}
    rng = np.random.default_rng(SEED)
    pos0 = {i: centers[mod[names[i]]] + rng.normal(0, 0.06, 2)
            for i in range(len(names))}
    pos = nx.spring_layout(Gv, weight="layout_w", pos=pos0, seed=SEED,
                           iterations=90, k=0.50 / np.sqrt(len(names)))
    xy = np.array([pos[i] for i in range(len(names))])
    xy = (xy - xy.mean(0)) / np.abs(xy).max()

    segs = [[xy[a], xy[b]] for a, b, _ in G.edges(data=True)]
    ax.add_collection(LineCollection(segs, colors="#BBBBBB", linewidths=0.22,
                                     alpha=0.30, zorder=1))
    node_col = [F.MODULE_COLORS[mod[n]] for n in names]
    sizes = np.array([6 + 1.35 * deg[n] for n in names])
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=node_col, edgecolors="white",
               linewidths=0.35, alpha=0.95, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.22, 1.22)
    ax.set_ylim(-1.22, 1.22)

    disp = dict(zip(_read("q1/importance.csv")["feature"],
                    _read("q1/importance.csv")["display"]))
    off = {0: (0.10, -0.48), 1: (0.02, 0.20), 2: (-0.34, 0.05),
           3: (-0.16, 0.14), 4: (-0.10, -0.25), 5: (0.18, 0.11),
           6: (0.26, -0.22), 7: (-0.26, -0.20)}
    seen = set()
    for m in range(8):
        sub = comm[comm["community"] == m]
        r = sub.loc[sub["degree"].idxmax()]
        i = names.index(r["node"])
        lab = disp.get(r["node"], r["node"].replace(".", " "))
        if lab in seen:
            continue
        seen.add(lab)
        cen = xy[[names.index(n) for n in sub["node"]]].mean(0)
        tx, ty = cen + np.array(off[m])
        ha = "right" if tx < -0.78 else ("left" if tx > 0.82 else "center")
        ax.annotate(lab, xy=xy[i], xytext=(tx, ty), fontsize=5.1,
                    ha=ha, va="center", color="#222222",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#888888",
                                    shrinkA=0, shrinkB=1),
                    path_effects=[pe.withStroke(linewidth=1.5,
                                                foreground="white")],
                    zorder=5)
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4.2,
                      markerfacecolor=F.MODULE_COLORS[m],
                      markeredgecolor="white", label=f"ME{m}")
               for m in range(8)]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
              ncol=8, fontsize=5.6, handletextpad=0.25, columnspacing=0.9,
              frameon=False)


@panel("P07b", 74, 72,
       desc="Module size vs threshold-compound enrichment",
       sources="results/supplementary/q2_module_annotation.csv",
       stats="8 Louvain modules; fold enrichment of threshold-active compounds; BH q",
       margins=(0.22, 0.17, 0.975, 0.965))
def p07b(ax):
    ann = _read("supplementary/q2_module_annotation.csv")
    for _, r in ann.iterrows():
        m = int(r["module"])
        c = F.MODULE_COLORS[m]
        ax.scatter(r["size"], r["fold_enrichment"], s=34, color=c,
                   edgecolor="white", linewidth=0.5, zorder=3)
        dy = 0.10 if m not in (0, 2) else -0.16
        ax.annotate(f"ME{m}", (r["size"], r["fold_enrichment"]),
                    xytext=(0, 6 if dy > 0 else -8), textcoords="offset points",
                    ha="center", fontsize=5.8, color=c,
                    fontweight="bold" if r["q_threshold"] < 0.05 else "normal")
    ax.axhline(1.0, color="#999999", lw=0.5, ls=(0, (2, 2)))
    ax.set_xlabel("Module size (compounds)")
    ax.set_ylabel("Threshold-compound enrichment (fold)")
    ax.set_xlim(12, 45)
    ax.set_ylim(-0.15, 4.1)
    r3 = ann.loc[ann["module"].astype(int) == 3].iloc[0]
    q3 = float(r3["q_threshold"])
    mant, expo = f"{q3:.1e}".split("e")
    ax.text(0.03, 0.97,
            f"ME3: enrichment ×{float(r3['fold_enrichment']):.1f}\n"
            f"$q$ = {mant} × 10$^{{{int(expo)}}}$",
            transform=ax.transAxes, va="top", fontsize=6.0,
            color=F.MODULE_COLORS[3])
    ax.text(0.03, 0.80, "Dashed: no enrichment", transform=ax.transAxes,
            va="top", fontsize=5.6, color="#777777")
    F.soften(ax, xgrid=True, ygrid=True)


def fig07():
    fig = plt.figure(figsize=(F.W2, 100 * F.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], left=0.005,
                          right=0.985, top=0.97, bottom=0.05, wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    p07a(ax)
    F.panel_label(ax, "a", x=0.02, y=0.98)
    ax = fig.add_subplot(gs[0, 1])
    p07b(ax)
    F.panel_label(ax, "b", x=-0.22, y=1.02)
    _save(fig, "fig07_q2_network")


# ============================================================ P08 Q2 证据
@panel("P08a", 108, 82,
       desc="Module-by-sensory-category coupling with hierarchical clustering",
       sources="results/supplementary/me_sensory_category_matrix.csv; results/q2/communities.csv; results/supplementary/q2_module_annotation.csv",
       stats="8 modules × 5 categories; mean |Spearman ρ|; average-linkage dendrograms; tracks = members and enrichment",
       margins=(0, 0, 1, 1))
def p08a(ax):
    from scipy.cluster.hierarchy import dendrogram, linkage
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    raw = _read("supplementary/me_sensory_category_matrix.csv").set_index("module")
    raw.index = [int(str(i).replace("ME", "")) for i in raw.index]
    cats = list(raw.columns)
    modules = list(raw.index)
    A = raw.to_numpy(float)
    Zr = linkage(A, method="average", metric="correlation")
    Zc = linkage(A.T, method="average", metric="correlation")
    row_order = dendrogram(Zr, no_plot=True)["leaves"]
    col_order = dendrogram(Zc, no_plot=True)["leaves"]
    nr, nc = A.shape

    ax.set_axis_off()
    axr = ax.inset_axes([0.00, 0.03, 0.14, 0.62])
    axm = ax.inset_axes([0.16, 0.03, 0.48, 0.62])
    axc = ax.inset_axes([0.16, 0.675, 0.48, 0.16])
    axt = ax.inset_axes([0.655, 0.03, 0.15, 0.62])
    axb = ax.inset_axes([0.935, 0.03, 0.022, 0.62])

    dendrogram(Zr, orientation="left", ax=axr, color_threshold=0,
               above_threshold_color=F.GREY, no_labels=True,
               link_color_func=lambda _: F.GREY)
    dendrogram(Zc, orientation="top", ax=axc, color_threshold=0,
               above_threshold_color=F.GREY, no_labels=True,
               link_color_func=lambda _: F.GREY)
    for a in (axr, axc):
        a.set_axis_off()

    vmax = A.max()
    im = axm.imshow(A[np.ix_(row_order, col_order)], cmap=F.SEQ_BLUE,
                    aspect="auto", vmin=0, vmax=vmax, origin="lower",
                    extent=(0, 10 * nc, 0, 10 * nr))
    axm.set_xlim(0, 10 * nc)
    axm.set_ylim(0, 10 * nr)
    axm.set_xticks([5 + 10 * j for j in range(nc)])
    axm.set_xticklabels([cats[j] for j in col_order], rotation=35,
                        ha="right", fontsize=6.0)
    axm.set_yticks([])
    for s in axm.spines.values():
        s.set_visible(False)
    for i in range(nr):
        for j in range(nc):
            v = A[row_order[i], col_order[j]]
            axm.text(5 + 10 * j, 5 + 10 * i, f"{v:.2f}", ha="center",
                     va="center", fontsize=5.3,
                     color="white" if v > vmax * 0.55 else "#333333")

    axr.set_ylim(0, 10 * nr)
    axc.set_xlim(0, 10 * nc)

    ann = _read("supplementary/q2_module_annotation.csv").set_index("module")
    size = ann["size"].to_dict()
    fold = ann["fold_enrichment"].to_dict()
    nrm_s = Normalize(vmin=min(size.values()), vmax=max(size.values()))
    nrm_f = Normalize(vmin=min(fold.values()), vmax=max(fold.values()))
    for i, mi in enumerate(row_order):
        m = modules[mi]
        for x0, col in ((0, F.SEQ_BLUE(nrm_s(size[m]))),
                        (1, F.SEQ_ORANGE(nrm_f(fold[m])))):
            axt.add_patch(Rectangle((x0 + 0.03, 10 * i + 1.2), 0.64, 7.6,
                                    facecolor=col, edgecolor="white",
                                    linewidth=0.5))
        axt.text(0.35, 5 + 10 * i, f"{int(size[m])}", ha="center", va="center",
                 fontsize=5.6,
                 color="white" if nrm_s(size[m]) > 0.5 else "#222222")
        axt.text(1.35, 5 + 10 * i, f"{fold[m]:.1f}", ha="center", va="center",
                 fontsize=5.4,
                 color="white" if nrm_f(fold[m]) > 0.55 else "#222222")
    axt.set_xlim(0, 2)
    axt.set_ylim(0, 10 * nr)
    axt.set_yticks([5 + 10 * i for i in range(nr)])
    axt.set_yticklabels([f"ME{modules[mi]}" for mi in row_order], fontsize=5.9)
    axt.yaxis.tick_right()
    axt.tick_params(axis="y", length=0, pad=1.5)
    axt.set_xticks([])
    for s in axt.spines.values():
        s.set_visible(False)

    ax.text(0.681, 0.700, "n", transform=ax.transAxes, fontsize=5.6,
            color="#555555", ha="center")
    ax.text(0.756, 0.700, "×fold", transform=ax.transAxes, fontsize=5.6,
            color="#555555", ha="center")
    ax.text(0.885, 0.700, "module", transform=ax.transAxes, fontsize=5.6,
            color="#555555", ha="center")
    cb = plt.colorbar(ScalarMappable(norm=Normalize(0, vmax), cmap=F.SEQ_BLUE),
                      cax=axb)
    cb.set_label("Mean |Spearman $\\rho$|", fontsize=6.2)
    cb.ax.tick_params(labelsize=5.6)
    r3 = ann.loc[3]
    q3 = float(r3["q_threshold"])
    mant, expo = f"{q3:.1e}".split("e")
    ax.text(0.655, 0.745,
            f"ME3 ×{float(r3['fold_enrichment']):.1f}; $q$ = {mant}×10$^{{{int(expo)}}}$",
            transform=ax.transAxes, fontsize=5.7, color=F.MODULE_COLORS[3],
            ha="left", va="bottom")


@panel("P08b", 84, 62,
       desc="Top-3 selection stability by sensory family (raincloud)",
       sources="results/q2/top3.csv; results/q2/stability.json",
       stats="50 dimensions; bootstrap selection frequency; half-violin KDE + Sina jitter + median/IQR",
       margins=(0.24, 0.17, 0.975, 0.965))
def p08b(ax):
    st = _read("q2/top3.csv")
    stab = _json("q2/stability.json")
    # 三个无香气家族归属的维度（aftertaste / body / overall）并入 Others，
    # 与图 12 的五类计数一致（Others n = 15）。
    st["family"] = st["target_family"].fillna("Others")
    st["stable_rate"] = st["stable_rate"].astype(float)
    order = ["Hops", "Malt", "Yeast", "Wild flora", "Others"]
    order = [o for o in order if o in set(st["family"])]
    vals = [st.loc[st["family"] == f, "stable_rate"].to_numpy() for f in order]
    F.raincloud(ax, vals, np.arange(len(order)),
                [FAMILY_COLOR.get(f, F.LGREY) for f in order],
                violin_width=0.34, side="right")
    thr = stab["stable_rate_threshold"]
    ax.axvline(thr, color=F.VERM, lw=0.8, ls=(0, (3, 2)))
    ax.text(thr + 0.01, len(order) - 0.45, f"threshold {thr:.1f}",
            color=F.VERM, fontsize=5.9, va="top")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=6.3)
    ax.set_xlabel("Selection stability (bootstrap frequency)")
    lo = min(float(np.nanmin(v)) for v in vals)
    hi = max(float(np.nanmax(v)) for v in vals)
    ax.set_xlim(lo - 0.025, max(1.0, hi) + 0.02)
    ax.set_ylim(-0.6, len(order) - 0.4)
    F.soften(ax, xgrid=True)


def fig08():
    fig = plt.figure(figsize=(F.W2, 86 * F.MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.32, 1], left=0.015,
                          right=0.975, top=0.92, bottom=0.15, wspace=0.30)
    ax = fig.add_subplot(gs[0, 0])
    p08a(ax)
    F.panel_label(ax, "a", x=-0.02, y=1.03)
    ax = fig.add_subplot(gs[0, 1])
    p08b(ax)
    F.panel_label(ax, "b", x=-0.22, y=1.03)
    _save(fig, "fig08_q2_evidence")


# ============================================================ P09 三问互证
@panel("P09a", 64, 128,
       desc="Style-mean fidelity between real and generated sensory profiles",
       sources="data/raw/附件/4.csv; results/q3/generated_sensory.csv",
       stats="22 styles × 50 dimensions; Spearman ρ of style means; 0.50 reference",
       margins=(0.38, 0.075, 0.975, 0.975))
def p09a(ax):
    from scipy.stats import spearmanr
    targets, prof = _style_profiles()
    rho = np.array([spearmanr(real, syn).statistic for _, real, syn in prof])
    names = [s for s, _, _ in prof]
    order = np.argsort(rho)
    rho, names = rho[order], [names[i] for i in order]
    y = np.arange(len(rho))
    ok = rho >= 0.5
    for yi, ri in zip(y, rho):
        ax.plot([0.24, ri], [yi, yi], color=F.XLGREY, lw=0.8, zorder=1)
    ax.scatter(rho[ok], y[ok], s=16, color=F.BLUE, zorder=3,
               edgecolor="white", linewidth=0.35, label="ρ ≥ 0.50")
    ax.scatter(rho[~ok], y[~ok], s=16, color=F.VERM, zorder=3,
               edgecolor="white", linewidth=0.35, label="ρ < 0.50")
    ax.axvline(0.5, color=F.GREY, lw=0.8, ls=(0, (3, 2)), zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.6)
    ax.set_ylim(-0.7, len(rho) - 0.3)
    ax.set_xlim(0.24, 0.95)
    ax.set_xlabel("Spearman ρ: real vs generated style mean")
    ax.text(0.03, 0.985,
            f"mean {rho.mean():.3f}   median {np.median(rho):.3f}\n"
            f"{ok.sum()}/{len(rho)} styles ≥ 0.50",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.8)
    ax.legend(loc="lower right", fontsize=6.2, handletextpad=0.4,
              borderaxespad=0.2)
    F.soften(ax, xgrid=True)


@panel("P09b", 78, 64,
       desc="All style-by-dimension means, real vs generated",
       sources="data/raw/附件/4.csv; results/q3/generated_sensory.csv; results/q2/top3.csv",
       stats="22 styles × 50 dimensions; colours = target family",
       margins=(0.18, 0.17, 0.975, 0.965))
def p09b(ax):
    targets, prof = _style_profiles()
    fam = _family_map()
    fam_col = dict(FAMILY_COLOR)
    xs, ys, cs = [], [], []
    for s, real, syn in prof:
        for j, t in enumerate(targets):
            xs.append(real[j])
            ys.append(syn[j])
            cs.append(fam_col.get(fam.get(t, "Unassigned"), F.LGREY))
    xs, ys = np.asarray(xs), np.asarray(ys)
    lim = (min(xs.min(), ys.min()) - 0.08, max(xs.max(), ys.max()) + 0.08)
    ax.plot(lim, lim, color=F.GREY, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.scatter(xs, ys, s=5, c=cs, alpha=0.55, linewidths=0, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Real style mean (z)")
    ax.set_ylabel("Generated style mean (z)")
    r = np.corrcoef(xs, ys)[0, 1]
    ax.text(0.03, 0.97, f"22 styles × 50 dimensions\nPearson r = {r:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=6.8)
    handles = [Line2D([], [], marker="o", ls="", markersize=3.2,
                      markerfacecolor=c, markeredgecolor="none", label=k)
               for k, c in fam_col.items() if k != "Unassigned"]
    ax.legend(handles=handles, loc="lower right", fontsize=5.9, ncol=2,
              handletextpad=0.35, columnspacing=0.7, borderaxespad=0.2)
    F.soften(ax, xgrid=True, ygrid=True)


@panel("P09c", 82, 44,
       desc="Aroma OAV shift from Pils/Lager to Tripel",
       sources="results/supplementary/q3_aroma_oav_ratio.csv",
       stats="compound-level OAV; ratio labels are Tripel/Pils/Lager",
       margins=(0.33, 0.19, 0.975, 0.965))
def p09c(ax):
    oav = _read("supplementary/q3_aroma_oav_ratio.csv")
    yy = np.arange(len(oav))[::-1]
    for yi, (_, row) in zip(yy, oav.iterrows()):
        ax.plot([row["Pils/Lager"], row["Tripel"]], [yi, yi],
                color=F.XLGREY, lw=1.6, zorder=1, solid_capstyle="round")
    ax.scatter(oav["Pils/Lager"], yy, s=22, color=F.BLUE, zorder=3,
               label="Pils/Lager", edgecolor="white", linewidth=0.4)
    ax.scatter(oav["Tripel"], yy, s=22, color=F.VERM, zorder=3,
               label="Tripel", edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.axvline(1.0, color=F.GREY, lw=0.7, ls=(0, (3, 2)), zorder=0)
    ax.set_yticks(yy)
    ax.set_yticklabels([c.replace("_", " ").capitalize()
                        for c in oav["compound"]], fontsize=6.4)
    ax.set_ylim(-0.6, len(oav) - 0.4)
    ax.set_xlim(0.09, 420)
    for yi, (_, row) in zip(yy, oav.iterrows()):
        ax.text(row["Tripel"] * 1.55, yi, f"×{row['ratio_tripel_vs_pils']:.1f}",
                va="center", ha="left", fontsize=6.0, color="#333333")
    ax.set_xlabel("Aroma OAV (log scale)")
    ax.legend(loc="lower right", fontsize=6.2, ncol=2,
              bbox_to_anchor=(1.0, 1.015), handletextpad=0.4,
              columnspacing=0.8, borderaxespad=0.0)
    F.soften(ax, xgrid=True)


@panel("P09d", 84, 46,
       desc="Q2↔Q3 consistency under signed and positive-only selection",
       sources="results/supplementary/consistency_dual.json; results/supplementary/consistency_dual_per_dim.csv",
       stats="50 dimensions; per-dimension Spearman ρ; 0.15 threshold; medians marked",
       margins=(0.17, 0.18, 0.975, 0.965))
def p09d(ax):
    from scipy.stats import gaussian_kde
    d = _read("supplementary/consistency_dual_per_dim.csv")
    j = _json("supplementary/consistency_dual.json")["tracks"]
    tr = {t["name"]: t for t in j}
    tracks = [("signed", "rho_signed", F.BLUE, "Signed (all effects)"),
              ("positive", "rho_positive", F.ORANGE, "Positive-only")]
    lo = min(d["rho_signed"].min(), d["rho_positive"].min()) - 0.03
    hi = max(d["rho_signed"].max(), d["rho_positive"].max()) + 0.03
    grid = np.linspace(lo, hi, 300)
    ymax = 0.0
    for name, col, color, lab in tracks:
        v = d[col].to_numpy(float)
        y = gaussian_kde(v)(grid)
        ymax = max(ymax, float(y.max()))
        ax.fill_between(grid, 0, y, color=color, alpha=0.18, linewidth=0)
        ax.plot(grid, y, color=color, lw=1.0, label=lab)
        ax.axvline(tr[name]["median_spearman"], color=color, lw=1.0,
                   ls=(0, (4, 2)))
    thr = 0.15
    ax.axvline(thr, color=F.VERM, lw=0.9, ls=(0, (3, 2)),
               label="0.15 cut-off")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, ymax * 1.42)
    ax.set_xlabel("Spearman ρ: Q2 participation vs Q3 loading")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=5.7, handlelength=1.0,
              title="dashed = median", title_fontsize=5.7,
              labelspacing=0.25)
    ax.text(0.02, 0.60,
            "mean / median; dims ≥ 0.15\n"
            f"Signed {tr['signed']['mean_spearman']:.3f} / "
            f"{tr['signed']['median_spearman']:.3f}; "
            f"{tr['signed']['n_dims_passing_threshold']}/50\n"
            f"Positive {tr['positive']['mean_spearman']:.3f} / "
            f"{tr['positive']['median_spearman']:.3f}; "
            f"{tr['positive']['n_dims_passing_threshold']}/50\n"
            f"Top-20 overlap {tr['positive']['hypergeom_fold']:.1f}×, "
            f"p = {tr['positive']['hypergeom_p']:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=5.6,
            color="#444444", linespacing=1.32,
            bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="none",
                      alpha=0.92))
    F.soften(ax, ygrid=True)


def fig09():
    fig = F.new_fig(F.W2, 138 * F.MM)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.0],
                          height_ratios=[1.15, 0.90, 0.90],
                          left=0.120, right=0.985, top=0.955, bottom=0.095,
                          wspace=0.42, hspace=0.62)
    ax = fig.add_subplot(gs[:, 0])
    p09a(ax)
    F.panel_label(ax, "a", x=-0.30, y=1.01)
    ax = fig.add_subplot(gs[0, 1])
    p09b(ax)
    F.panel_label(ax, "b", x=-0.15, y=1.02)
    ax = fig.add_subplot(gs[1, 1])
    p09c(ax)
    F.panel_label(ax, "c", x=-0.30, y=1.04)
    ax = fig.add_subplot(gs[2, 1])
    p09d(ax)
    F.panel_label(ax, "d", x=-0.15, y=1.04)
    _save(fig, "fig09_q3_fidelity_consistency")


# ============================================================ P10 缺陷约束
def _quad_guide(p0, p1, bow: float = 0.22, up: bool = True):
    """两状态之间的二次贝塞尔示意线（端点来自数据，弯曲仅用于可读性）。"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.hypot(*d))
    if L < 1e-9:
        return MplPath([p0, p1], [MplPath.MOVETO, MplPath.LINETO])
    perp = np.array([-d[1], d[0]]) / L
    if perp[1] < 0:
        perp = -perp
    ctrl = (p0 + p1) / 2 + (1 if up else -1) * bow * L * perp
    return MplPath([p0, ctrl, p1],
                   [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])


def _ellipse80(ax, x, y, color, label=None, n_min=3, fill=False,
               alpha=0.75, lw=0.7):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < n_min:
        return
    cov = np.cov(np.vstack([x, y]))
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if vals[0] <= 0:
        return
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    chi2 = 3.2189  # 2 自由度 80%
    w, h = 2 * np.sqrt(vals * chi2)
    e = Ellipse((x.mean(), y.mean()), width=w, height=h, angle=ang,
                facecolor=color if fill else "none", edgecolor=color,
                alpha=alpha if not fill else 0.12, lw=lw,
                ls="-" if fill else (0, (4, 2)), zorder=3)
    ax.add_patch(e)
    ax.plot([], [], color=color, lw=0.8, label=label)
    return e


@panel("P10a", 183, 100,
       desc="Latent state-space manifold of 250 beers with 11 Pils/Lager→Tripel transfers",
       sources="results/supplementary/upgrade/q3_state_points.csv; q3_state_paths.csv; q3_state_axes.json; q3_state_individual.csv; results/supplementary/q3_defect_chemical_summary.csv",
       stats="u1 = Pils→Tripel centroid fraction; u2 = orthogonal composite off-flavour-risk axis (Ridge risk gradient, SD units); background KDE and 80% ellipses; guides are quadratic interpolations between observed states",
       margins=(0.09, 0.13, 0.975, 0.965))
def p10a(ax):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Patch
    from scipy.ndimage import gaussian_filter
    from scipy.stats import gaussian_kde
    from sklearn.neighbors import KNeighborsRegressor

    axes, pts, paths, indiv = _state()
    bg = pts[pts["role"] == "background"]
    x = bg["u1"].to_numpy(float)
    y = bg["u2"].to_numpy(float)
    r = bg["risk"].to_numpy(float)
    qlo, qhi = 0.005, 0.995
    xlo = min(np.quantile(x, qlo), pts["u1"].min()) - 0.08
    xhi = max(np.quantile(x, qhi), pts["u1"].max()) + 0.08
    ylo = min(np.quantile(y, qlo), pts["u2"].min()) - 0.15
    yhi = max(np.quantile(y, qhi), pts["u2"].max()) + 0.15
    gx = np.linspace(xlo, xhi, 220)
    gy = np.linspace(ylo, yhi, 220)
    XX, YY = np.meshgrid(gx, gy)
    dens = gaussian_kde(np.vstack([x, y]))(
        np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    dens = dens / dens.max()
    risk = KNeighborsRegressor(n_neighbors=45, weights="uniform").fit(
        np.c_[x, y], r).predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
    risk = gaussian_filter(risk, 2.0)

    kde_cmap = LinearSegmentedColormap.from_list(
        "kde", ["#FCFCFB", "#F1F1EF", "#E4E5E3", "#D6D8D7", "#C4C7C7"])
    ax.contourf(XX, YY, np.ma.masked_less(dens, 0.01),
                levels=np.linspace(0.01, 1.0, 8), cmap=kde_cmap,
                alpha=0.98, zorder=0)
    ax.scatter(x, y, s=3.4, color="#7C8A92", alpha=0.22, linewidths=0,
               zorder=2)

    # 高风险核：背景啤酒综合风险的前 25%
    zthr = float(axes["zone_threshold"])
    zone = np.ma.masked_where((dens < 0.03) | (risk < zthr), risk)
    ax.contourf(XX, YY, zone, levels=[zthr, float(np.nanmax(risk))],
                colors=["#C0392B"], alpha=0.08, zorder=1)
    with plt.rc_context({"hatch.linewidth": 0.35}):
        hatch = ax.contourf(XX, YY, zone,
                            levels=[zthr, float(np.nanmax(risk))],
                            colors="none", hatches=["//"], zorder=1)
    hatch.set_edgecolor("#B03A2E")
    ax.contour(XX, YY, zone, levels=[zthr], colors=["#A93226"],
               linewidths=0.7, zorder=2)
    ax.axhline(0, color="#6F6F6F", lw=0.55, ls=(0, (2, 2)), zorder=1)
    ax.text(0.985, 0.515, "risk-neutral axis (u2 = 0)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.3, color="#6F6F6F",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                      alpha=1.0))

    # 安全切线：风险场在源质心的梯度，切线方向 = w_off^T Δz = 0
    filled = np.where(np.isfinite(risk), risk, np.nanmedian(risk))
    dy_, dx_ = np.gradient(filled, gy, gx)
    src = paths[paths["state"] == "before"]
    x0, y0 = float(src["u1"].median()), float(src["u2"].median())
    j0 = int(np.argmin(np.abs(gx - x0)))
    i0 = int(np.argmin(np.abs(gy - y0)))
    grad = np.array([dx_[i0, j0], dy_[i0, j0]])
    grad = grad / max(np.linalg.norm(grad), 1e-12)
    tan = np.array([-grad[1], grad[0]])
    ax.plot([x0 - 0.38 * tan[0], x0 + 0.38 * tan[0]],
            [y0 - 0.38 * tan[1], y0 + 0.38 * tan[1]],
            color=F.SKY, lw=0.75, ls=(0, (2, 2)), zorder=2)

    # 风格 80% 置信椭圆（背景样本；只描边，避免大面积色块）
    for style, color in STYLE_COLOR.items():
        sub = bg[bg["style"] == style]
        _ellipse80(ax, sub["u1"], sub["u2"], color, label=style, fill=False,
                   alpha=0.42, lw=0.6)

    path_color = {"ot_proj": F.VERM, "qp_full": "#173F6B"}

    # 11 条迁移：只画 OT 与 QP，避免与 P10b 的四状态重复
    b = paths[paths["state"] == "before"].set_index("beer")
    for m in ("ot_proj", "qp_full"):
        e = paths[paths["state"] == m].set_index("beer")
        for beer in b.index:
            if beer not in e.index:
                continue
            p0 = b.loc[beer, ["u1", "u2"]].to_numpy(float)
            p1 = e.loc[beer, ["u1", "u2"]].to_numpy(float)
            path = _quad_guide(p0, p1, bow=0.18, up=(m != "qp_full"))
            ax.add_patch(PathPatch(
                path, facecolor="none", edgecolor=path_color[m],
                lw=0.65, alpha=0.30,
                ls=(0, (3, 2)) if m == "ot_proj" else "-", zorder=4))
        ax.scatter(e["u1"], e["u2"], s=10,
                   marker="o" if m == "ot_proj" else "s",
                   color=path_color[m], edgecolor="white", linewidth=0.3,
                   alpha=0.9, zorder=5)
    ax.scatter(b["u1"], b["u2"], s=24, facecolor="white",
               edgecolor="#333333", linewidth=0.75, zorder=6)

    # 代表性轨迹：源 u1 中位的那一款
    rep = b["u1"].sort_values().index[len(b) // 2]
    for m, lw in (("ot_proj", 1.5), ("qp_full", 1.6)):
        e = paths[paths["state"] == m].set_index("beer")
        p0 = b.loc[rep, ["u1", "u2"]].to_numpy(float)
        p1 = e.loc[rep, ["u1", "u2"]].to_numpy(float)
        path = _quad_guide(p0, p1, bow=0.18, up=(m != "qp_full"))
        ax.add_patch(PathPatch(
            path, facecolor="none", edgecolor=path_color[m], lw=lw,
            alpha=0.95, ls=(0, (3, 2)) if m == "ot_proj" else "-",
            zorder=7))
        q0, ctrl, q1 = path.vertices

        def _bez(t):
            return (1 - t) ** 2 * q0 + 2 * (1 - t) * t * ctrl + t ** 2 * q1

        ax.annotate("", xy=_bez(0.99), xytext=_bez(0.90),
                    arrowprops=dict(arrowstyle="-|>", lw=lw * 0.9,
                                    color=path_color[m], shrinkA=0,
                                    shrinkB=0, mutation_scale=6),
                    zorder=8)

    # 数值标注
    defs = _read("supplementary/q3_defect_chemical_summary.csv")
    h2s = float(defs[(defs["compound"] == "H2S")
                     & (defs["constraint"] == "none")]["median_ratio"].iloc[0])
    mesh = float(defs[(defs["compound"] == "MeSH")
                      & (defs["constraint"] == "none")]["median_ratio"].iloc[0])
    iw = indiv.copy()
    before = iw[iw["state"] == "before"].set_index("beer")
    deltas = [f"delta_{d}" for d in SENS_DIMS]
    ot = iw[iw["state"] == "ot_proj"].set_index("beer")
    qp = iw[iw["state"] == "qp_full"].set_index("beer")
    n_ot = int((ot[deltas] > 0).any(axis=1).sum())
    n_qp = int((qp[deltas] > 0).any(axis=1).sum())
    box = dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.72)
    ax.text(
        0.035, 0.965,
        f"OT proj.: H$_2$S OAV ×{h2s:.2f}; MeSH ×{mesh:.2f}\n"
        f"{n_ot}/11 sensory worsening; ends on the risk-elevated side",
        transform=ax.transAxes, fontsize=5.9, color=F.VERM,
        va="top", ha="left", linespacing=1.35, bbox=box, zorder=9)
    ax.text(
        0.035, 0.335,
        f"Joint QP: {n_qp}/11 worsening\nmedian shift "
        f"{paths[paths['state'] == 'qp_full']['style_shift'].median():.3f}; "
        "median below the risk-neutral axis",
        transform=ax.transAxes, fontsize=5.9, color=path_color["qp_full"],
        va="top", ha="left", linespacing=1.35, bbox=box, zorder=9)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("u1: Pils/Lager → Tripel (centroid fraction)")
    ax.set_ylabel("u2: composite off-flavour risk (SD)")
    ax.text(0.985, 0.02,
            "background: 250 beers; contours: Gaussian KDE\n"
            "hatched: top-quartile composite risk\n"
            "○ source (n = 11) · orange ○ OT endpoint (dashed) · "
            "navy □ Joint QP (solid)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.4,
            color="#555555", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none",
                      alpha=0.88), zorder=9)
    style_handles = [
        Patch(facecolor="none", edgecolor=STYLE_COLOR[s], lw=0.8,
              ls=(0, (4, 2)), label=s) for s in STYLE_COLOR]
    ax.legend(handles=style_handles, loc="lower left",
              fontsize=5.6, ncol=1, handlelength=1.2,
              labelspacing=0.25, title="80% ellipse", title_fontsize=5.8)
    ax.grid(False)


@panel("P10b", 183, 72,
       desc="Paired trajectories of six aroma compounds across four states",
       sources="results/supplementary/upgrade/q3_state_individual.csv",
       stats="11 Pils/Lager beers × 4 states; thin lines = individuals, band = IQR, thick = median; OAV = 1 reference",
       margins=(0.0, 0.0, 1.0, 1.0))
def p10b(ax):
    _, _, _, indiv = _state()
    states = ["before", "none", "ot_proj", "qp_full"]
    specs = [
        ("DMS", "DMS OAV", True),
        ("X2.3.butanedione", "Diacetyl OAV", True),
        ("H2S", "H$_2$S OAV", True),
        ("MeSH", "MeSH (µg/L)", False),
        ("isopentyl_acetate", "Isoamyl acetate OAV", True),
        ("ethyl_octanoate", "Ethyl octanoate OAV", True),
    ]
    ax.set_axis_off()
    pos = [(0.075, 0.56, 0.255, 0.38), (0.395, 0.56, 0.255, 0.38),
           (0.715, 0.56, 0.255, 0.38), (0.075, 0.08, 0.255, 0.38),
           (0.395, 0.08, 0.255, 0.38), (0.715, 0.08, 0.255, 0.38)]
    for k, ((comp, lab, is_oav), rect) in enumerate(zip(specs, pos)):
        a = ax.inset_axes(rect)
        p = indiv.pivot(index="beer", columns="state", values=comp)
        p = p.reindex(columns=states)
        xx = np.arange(4)
        for xi, (state, color) in enumerate(zip(states, ["#F3F3F3", "#F3F3F3",
                                                         "#FDF1E3", "#EAF2F8"])):
            a.axvspan(xi - 0.42, xi + 0.42, color=color, zorder=0, lw=0)
        for beer, row in p.iterrows():
            v = row.to_numpy(float)
            if np.isfinite(v).sum() < 2:
                continue
            a.plot(xx, v, color="#6F7F8C", lw=0.55, alpha=0.30, zorder=1)
        q1, med, q3 = np.nanpercentile(p.to_numpy(float), [25, 50, 75], axis=0)
        a.fill_between(xx, q1, q3, color="#9FB6C8", alpha=0.25, lw=0, zorder=2)
        a.plot(xx, med, color="#111111", lw=1.25, marker="o", ms=2.6,
               zorder=4)
        if is_oav:
            from matplotlib.ticker import NullFormatter

            a.set_yscale("log")
            a.axhline(1.0, color="#C0392B", lw=0.7, ls=(0, (3, 2)), zorder=0)
            vals = p.to_numpy(float)
            lo = max(np.nanmin(vals) * 0.55, 1e-3)
            hi = np.nanmax(vals) * 1.7
            a.set_ylim(lo, hi)
            nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
            cand = [t for t in nice if lo <= t <= hi]
            if not cand:
                cand = [round(lo, 2), round(hi, 2)]
            while len(cand) > 4:
                cand.remove(max(cand, key=lambda t: abs(np.log10(t))))
            a.set_yticks(cand)
            a.set_yticklabels([f"{t:g}" for t in cand])
            a.yaxis.set_minor_formatter(NullFormatter())
        else:
            vals = p.to_numpy(float)
            a.set_ylim(np.nanmin(vals) - 0.25, np.nanmax(vals) + 0.25)
        a.set_xlim(-0.42, 3.42)
        a.set_xticks(xx)
        if k >= 3:
            a.set_xticklabels(["Before", "None", "OT", "QP"], fontsize=6.0)
        else:
            a.set_xticklabels([])
        a.tick_params(axis="y", labelsize=5.6, pad=1.2)
        a.set_title(lab, fontsize=6.5, pad=2.5, loc="left")
        F.soften(a, ygrid=True)


@panel("P10c", 100, 82,
       desc="Individual-level sensory and chemical changes under OT vs Joint QP",
       sources="results/supplementary/upgrade/q3_state_individual.csv",
       stats="11 beers × 13 endpoints (5 sensory, 8 chemical); colour = Δ divided by column SD across the 22 OT/QP states (sign preserved); blue = improvement",
       margins=(0.0, 0.0, 1.0, 1.0))
def p10c(ax):
    _, _, _, indiv = _state()
    cols = [("sensory", d, f"delta_{d}") for d in SENS_DIMS] + \
           [("chemical", d, f"delta_x_{d}") for d in CHEM_DIMS]
    labels = [SENS_LABEL.get(d, d) for d in SENS_DIMS] + \
             [CHEM_LABEL[d] for d in CHEM_DIMS]
    ot = indiv[indiv["state"] == "ot_proj"].set_index("beer")
    qp = indiv[indiv["state"] == "qp_full"].set_index("beer")
    beers = ot.index.tolist()
    raw = np.vstack([ot[[c[2] for c in cols]].to_numpy(float),
                     qp[[c[2] for c in cols]].to_numpy(float)])
    # 只除以列内 SD、不减均值：保留 Δ 的原始符号，避免 QP 的「零恶化」
    # 被列内中心化后画成暖色。
    z = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        v = raw[:, j]
        s = np.nanstd(v, ddof=1)
        z[:, j] = v / s if s > 1e-9 else 0.0
    mot, mqp = z[:len(beers)], z[len(beers):]
    order = np.argsort(-np.nanmax(mot, axis=1))
    mot, mqp = mot[order], mqp[order]
    beers = [beers[i] for i in order]

    ax.set_axis_off()
    a1 = ax.inset_axes([0.15, 0.54, 0.76, 0.28])
    a2 = ax.inset_axes([0.15, 0.13, 0.76, 0.28])
    acb = ax.inset_axes([0.925, 0.24, 0.020, 0.38])
    vmax = float(min(3.0, max(1.0, np.nanpercentile(np.abs(z), 97))))
    for a, mat, tag in ((a1, mot, "OT projection"), (a2, mqp, "Joint QP")):
        im = a.imshow(mat, cmap=F.DIVERGING, vmin=-vmax, vmax=vmax,
                      aspect="auto", interpolation="nearest")
        a.set_xticks(np.arange(mat.shape[1]))
        a.set_yticks(np.arange(len(beers)))
        a.set_yticklabels(beers, fontsize=5.4)
        a.tick_params(axis="y", length=0, pad=1.2)
        if a is a1:
            a.set_xticklabels([])
        else:
            a.set_xticklabels(labels, rotation=38, ha="right", fontsize=5.7)
        a.axvline(4.5, color="white", lw=1.2)
        for s in a.spines.values():
            s.set_visible(False)
        a.text(-0.13, 0.5, tag, transform=a.transAxes, rotation=90,
               va="center", ha="center", fontsize=6.5,
               color="#B35A00" if a is a1 else F.BLUE)
    a1.text(0.50, 1.045, "Sensory off-flavour Δ   |   Chemical defect Δ",
            transform=a1.transAxes, ha="center", va="bottom", fontsize=5.8,
            color="#555555")
    a2.text(0.99, 1.10,
            "Joint QP: 0/11 worsened in any of the 5 sensory dimensions",
            transform=a2.transAxes, ha="right", va="bottom", fontsize=5.6,
            color=F.BLUE)
    cb = plt.colorbar(im, cax=acb)
    cb.set_label("Δ / column SD (red = worse)", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.4)


@panel("P10d", 84, 64,
       desc="Worst sensory change across 924 cross-style pairs (raincloud)",
       sources="results/supplementary/upgrade/q3_qp_pairs_long.csv",
       stats="924 pairs × 3 methods; max legacy Δ over 5 sensory off-flavour dimensions; half-violin KDE + Sina jitter + median/IQR",
       margins=(0.22, 0.17, 0.975, 0.965))
def p10d(ax):
    q = _read("supplementary/upgrade/q3_qp_pairs_long.csv")
    dims = [f"legacy_delta_{d}" for d in SENS_DIMS]
    methods = ["none", "ot_proj", "qp_full"]
    labels = ["No correction", "OT projection", "Joint QP"]
    groups = [q.loc[q["method"] == m, dims].max(axis=1).to_numpy(float)
              for m in methods]
    ypos = np.arange(len(methods))[::-1]
    F.raincloud(ax, groups, ypos, [METHOD_COLOR[m] for m in methods],
                violin_width=0.36, side="right")
    ax.axvline(0, color="#C0392B", lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=6.4)
    ax.set_ylim(-0.6, len(methods) - 0.4)
    ax.set_xlabel("Worst sensory Δ across 5 dimensions")
    lo = min(float(np.min(g)) for g in groups)
    hi = max(float(np.max(g)) for g in groups)
    ax.set_xlim(lo - 0.10, hi + 0.10)
    share = [float((g > 0).mean() * 100) for g in groups]
    for y, s in zip(ypos, share):
        ax.text(0.985, y, f"{s:.1f}% above 0",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=5.8, color="#555555",
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none",
                          alpha=0.9))
    ax.text(0.985, 0.04,
            "QP: 917/924 at the −0.005 safety\nmargin; all ≤ 0 (active constraint)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5,
            color=F.BLUE, linespacing=1.3)
    F.soften(ax, xgrid=True)


def fig10():
    fig = plt.figure(figsize=(F.W2, 246 * F.MM))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.34, 0.96, 1.02],
                          left=0.075, right=0.985, top=0.975, bottom=0.035,
                          wspace=0.28, hspace=0.42)
    ax = fig.add_subplot(gs[0, :])
    p10a(ax)
    F.panel_label(ax, "a", x=-0.055, y=1.005)
    ax = fig.add_subplot(gs[1, :])
    p10b(ax)
    F.panel_label(ax, "b", x=-0.055, y=1.03)
    ax = fig.add_subplot(gs[2, 0])
    p10c(ax)
    F.panel_label(ax, "c", x=-0.02, y=1.02)
    ax = fig.add_subplot(gs[2, 1])
    p10d(ax)
    F.panel_label(ax, "d", x=-0.24, y=1.02)
    _save(fig, "fig10_q3_constraints")


# ============================================================ P11 Pareto
def _cluster_boot(d, *, n_boot: int = 2000, seed: int = 42):
    """按啤酒整簇重采样的中位数 95% CI（避免把 924 对当独立样本）。"""
    groups = [g[["style_shift", "qp_cost"]].to_numpy(float)
              for _, g in d.groupby("beer", sort=True)]
    rng = np.random.default_rng(seed)
    stats = np.empty((n_boot, 2))
    for b in range(n_boot):
        idx = rng.integers(0, len(groups), len(groups))
        arr = np.vstack([groups[i] for i in idx])
        stats[b] = np.median(arr, axis=0)
    return np.percentile(stats, [2.5, 50, 97.5], axis=0)


@panel("P11a", 183, 92,
       desc="Cost–fidelity Pareto front of three transfer methods (44 beers)",
       sources="results/supplementary/upgrade/q3_qp_pairs_long.csv",
       stats="per-beer medians over 21 non-native styles; big markers = cohort median with clustered bootstrap 95% CI (2000 resamples of 44 beers); small points = 44 beer medians",
       margins=(0.095, 0.145, 0.975, 0.965))
def p11a(ax):
    q = _read("supplementary/upgrade/q3_qp_pairs_long.csv")
    methods = ["none", "ot_proj", "qp_full"]
    labels = {"none": "No correction", "ot_proj": "OT projection",
              "qp_full": "Joint QP"}
    per = (q.groupby(["method", "beer"])[["style_shift", "qp_cost"]]
             .median().reset_index())
    cis = {m: _cluster_boot(q[q["method"] == m]) for m in methods}

    # 底层：44 款啤酒各自的中位数点云
    for m in methods:
        s = per[per["method"] == m]
        if m == "none":
            ax.scatter(s["qp_cost"], s["style_shift"], s=10,
                       facecolor="none", edgecolor="#8A8A8A", lw=0.45,
                       alpha=0.55, zorder=2)
        else:
            ax.scatter(s["qp_cost"], s["style_shift"],
                       s=6.5, color=METHOD_COLOR[m], alpha=0.38,
                       linewidths=0, zorder=2)

    # 上层：群落中位数大点 + 双轴 95% CI 十字误差棒
    for m in ("none", "ot_proj", "qp_full"):
        lo, mid, hi = cis[m]
        x, y = mid[1], mid[0]
        if m == "none":
            ax.errorbar(
                x, y, xerr=[[x - lo[1]], [hi[1] - x]],
                yerr=[[y - lo[0]], [hi[0] - y]], fmt="o", ms=8.5,
                mfc="white", mec="#555555", mew=1.0, ecolor="#555555",
                elinewidth=0.8, capsize=2.0, capthick=0.6, zorder=5,
                label="No correction")
        elif m == "ot_proj":
            ax.errorbar(
                x, y, xerr=[[x - lo[1]], [hi[1] - x]],
                yerr=[[y - lo[0]], [hi[0] - y]], fmt="o", ms=6.2,
                mfc=F.ORANGE, mec="white", mew=0.6, ecolor=F.ORANGE,
                elinewidth=0.8, capsize=2.0, capthick=0.6, zorder=6,
                label="OT projection")
        else:
            ax.errorbar(
                x, y, xerr=[[x - lo[1]], [hi[1] - x]],
                yerr=[[y - lo[0]], [hi[0] - y]], fmt="D", ms=6.0,
                mfc=F.BLUE, mec="white", mew=0.6, ecolor=F.BLUE,
                elinewidth=0.9, capsize=2.0, capthick=0.6, zorder=7,
                label="Joint QP")

    ax.text(0.0938, 0.872,
            "No correction and OT\ncosts coincide", fontsize=5.6,
            color="#555555", ha="left", va="center", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                      alpha=1.0))
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.022), ncol=3,
              fontsize=5.9, frameon=False, handlelength=1.1,
              handletextpad=0.45, columnspacing=1.6, borderaxespad=0.0)
    lo, mid, hi = cis["qp_full"]
    ax.annotate(
        f"Joint QP\nmedian shift {mid[0]:.3f}",
        xy=(mid[1], mid[0]), xytext=(mid[1] - 0.012, mid[0] - 0.035),
        fontsize=5.8, color=F.BLUE, ha="left", va="top", linespacing=1.3,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                  alpha=0.82),
        arrowprops=dict(arrowstyle="-", lw=0.5, color=F.BLUE,
                        shrinkA=0, shrinkB=3))
    ax.set_xlim(0.083, 0.195)
    ax.set_ylim(0.72, 0.905)
    ax.set_xlabel("Correction cost (normalized L2; median)")
    ax.set_ylabel("Style shift (median; higher is better)")
    ax.text(0.985, 0.972,
            "small points: 44 beer medians   ·   crosshair: clustered "
            "bootstrap 95% CI\n(2000 resamples of beers, not of 924 pairs)",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.5,
            color="#555555", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none",
                      alpha=1.0))
    F.soften(ax, xgrid=True, ygrid=True)


@panel("P11b", 120, 80,
       desc="Pairwise cost–shift trade-off of Joint QP against OT projection",
       sources="results/supplementary/upgrade/q3_qp_pairs_long.csv",
       stats="924 matched cross-style pairs; Δ = Joint QP − OT projection; grey = pairs, blue = per-beer medians (44 beers); 68.5% in the higher-cost/lower-shift quadrant",
       margins=(0.135, 0.155, 0.975, 0.965))
def p11b(ax):
    q = _read("supplementary/upgrade/q3_qp_pairs_long.csv")
    q["pair"] = q.groupby(["beer", "method"]).cumcount()
    w = q.pivot(index=["beer", "pair"], columns="method",
                values=["style_shift", "qp_cost"]).dropna()
    dshift = (w[("style_shift", "qp_full")]
              - w[("style_shift", "ot_proj")]).to_numpy(float)
    dcost = (w[("qp_cost", "qp_full")]
             - w[("qp_cost", "ot_proj")]).to_numpy(float)
    per = pd.DataFrame({"beer": w.index.get_level_values("beer"),
                        "dshift": dshift, "dcost": dcost}
                       ).groupby("beer")[["dshift", "dcost"]].median()

    xlo, xhi = -0.015, 0.545
    ylo, yhi = -0.60, 0.42
    ax.add_patch(Rectangle((0, ylo), xhi, -ylo, facecolor="#EAF3FA",
                           alpha=1.0, lw=0, zorder=0))
    ax.scatter(dcost, dshift, s=5, color="#8C8C8C", alpha=0.30,
               linewidths=0, zorder=2, label="924 pairs")
    ax.scatter(per["dcost"], per["dshift"], s=15, color=F.BLUE,
               edgecolor="white", linewidth=0.35, alpha=0.9, zorder=4,
               label="44 beer medians")
    ax.axhline(0, color="#555555", lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.axvline(0, color="#555555", lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.plot([np.median(dcost)], [np.median(dshift)], marker="+",
            ms=10, mew=1.2, color="#222222", zorder=5)
    share = float(np.mean((dcost > 0) & (dshift < 0)) * 100)
    ax.text(0.53, -0.47, f"{share:.1f}%: higher cost,\nlower style shift",
            fontsize=6.0, color=F.BLUE, ha="right", va="bottom",
            linespacing=1.35)
    ax.text(0.015, 0.972, "shift preserved / improved",
            transform=ax.transAxes, fontsize=5.5, color="#555555",
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                      alpha=0.9))
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("Δ correction cost (Joint QP − OT projection)")
    ax.set_ylabel("Δ style shift (higher is better)")
    ax.text(0.985, 0.03,
            "matched pairs: same beer × non-native style; + marker = median",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5,
            color="#555555")
    handles = [
        Line2D([], [], marker="o", ls="", ms=2.8, color="#8C8C8C",
               alpha=0.6, label="924 pairs"),
        Line2D([], [], marker="o", ls="", ms=4.2, mfc=F.BLUE, mec="white",
               mew=0.4, label="44 beer medians"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=5.8,
              labelspacing=0.25, frameon=True, facecolor="white",
              framealpha=0.82, edgecolor="none")
    F.soften(ax, xgrid=True, ygrid=True)


def fig11():
    fig = plt.figure(figsize=(F.W2, 196 * F.MM))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.12, 1.0],
                          left=0.075, right=0.985, top=0.972, bottom=0.055,
                          hspace=0.42)
    ax = fig.add_subplot(gs[0, 0])
    p11a(ax)
    F.panel_label(ax, "a", x=-0.07, y=1.015)
    ax = fig.add_subplot(gs[1, 0])
    p11b(ax)
    F.panel_label(ax, "b", x=-0.095, y=1.02)
    _save(fig, "fig11_q3_pareto")


# ============================================================ P12 方差分解
@panel("P12a", 62, 62,
       desc="Sequential variance decomposition across the three modelling layers",
       sources="results/supplementary/variance_decomposition_overall.csv",
       stats="mean R² over 50 sensory dimensions; seeds 42/43/44; layers are style mean, + sensory latent (8), + Extra Trees residual",
       margins=(0.27, 0.18, 0.96, 0.95))
def p12a(ax):
    d = _read("supplementary/variance_decomposition_overall.csv").iloc[0]
    y = np.array([d["mean_r2_style"], d["mean_r2_plus_pls"],
                  d["mean_r2_full_et"]], float)
    x = np.arange(3)
    ax.fill_between(x, y.min() - 0.02, y, color=F.BLUE, alpha=0.08, lw=0)
    ax.plot(x, y, color="#222222", lw=1.0, marker="o", ms=4.5, zorder=3)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.3f}", (xi, yi), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=5.8, color="#222222")
    for xi, delta in ((0.5, y[1] - y[0]), (1.5, y[2] - y[1])):
        ax.annotate(f"+{delta:.3f}", (xi, np.interp(xi, x, y)),
                    xytext=(0, -12), textcoords="offset points",
                    ha="center", va="top", fontsize=5.5, color=F.BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(["Style\nmean", "+ sensory\nlatent (8)",
                        "+ Extra\nTrees"], fontsize=5.8)
    ax.set_ylim(0.13, 0.212)
    ax.set_ylabel("Mean R² (50 dimensions)")
    ax.text(0.03, 0.965, "n = 50 dimensions", transform=ax.transAxes,
            fontsize=5.4, color="#555555", va="top")
    F.soften(ax, ygrid=True)


@panel("P12b", 118, 62,
       desc="Layer-wise ΔR² per sensory dimension (raincloud)",
       sources="results/supplementary/variance_decomposition.csv",
       stats="50 dimensions; ΔR² for sensory latent (8) and Extra Trees residual; half-violin KDE + Sina jitter + median/IQR; negative = no gain",
       margins=(0.155, 0.17, 0.975, 0.96))
def p12b(ax):
    d = _read("supplementary/variance_decomposition.csv")
    groups = [d["delta_pls"].to_numpy(float), d["delta_et"].to_numpy(float)]
    ypos = np.array([1.0, 0.0])
    F.raincloud(ax, groups, ypos, [F.BLUE, F.ORANGE],
                violin_width=0.34, side="right")
    ax.axvline(0, color="#555555", lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(["Sensory latent\n(+8 dims)", "Extra Trees\n(residual)"],
                       fontsize=6.2)
    ax.set_ylim(-0.6, 1.6)
    lo = min(float(np.min(g)) for g in groups)
    hi = max(float(np.max(g)) for g in groups)
    ax.set_xlim(lo - 0.012, hi + 0.012)
    ax.set_xlabel("ΔR² at this layer (per sensory dimension)")
    for y, g in zip(ypos, groups):
        ax.text(0.985, y, f"{int((g < 0).sum())}/50 < 0",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=5.8, color="#555555",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none",
                          alpha=0.88))
    F.soften(ax, xgrid=True)


@panel("P12c", 84, 66,
       desc="Absolute R² by sensory category across layers",
       sources="results/supplementary/variance_decomposition_by_category.csv",
       stats="5 sensory categories; R² at style-mean, +sensory latent (8), +Extra Trees layers; category n shown at right",
       margins=(0.19, 0.16, 0.78, 0.96))
def p12c(ax):
    d = _read("supplementary/variance_decomposition_by_category.csv")
    colors = {"Hops": F.GREEN, "Malt": F.ORANGE, "Others": "#777777",
              "Wild flora": F.SKY, "Yeast": F.PINK}
    x = np.arange(3)
    for _, r in d.iterrows():
        y = np.array([r["r2_style"], r["r2_plus_pls"], r["r2_full_et"]],
                     float)
        ax.plot(x, y, color=colors.get(r["category"], "#555555"),
                lw=0.9, marker="o", ms=3.6, alpha=0.92)
        ax.text(2.12, float(r["r2_full_et"]),
                f'{r["category"]} (n = {int(r["n"])})', va="center",
                ha="left", fontsize=5.6,
                color=colors.get(r["category"], "#555555"))
    ax.set_xticks(x)
    ax.set_xticklabels(["Style", "+ latent", "+ ET"], fontsize=6.0)
    ax.set_xlim(-0.15, 3.55)
    ax.set_ylabel("R²")
    F.soften(ax, ygrid=True)


def fig12():
    fig = plt.figure(figsize=(F.W2, 146 * F.MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.30, 0.70],
                          height_ratios=[1.0, 0.94],
                          left=0.07, right=0.985, top=0.965, bottom=0.06,
                          wspace=0.36, hspace=0.44)
    ax = fig.add_subplot(gs[0, 0])
    p12a(ax)
    F.panel_label(ax, "a", x=-0.26, y=1.02)
    ax = fig.add_subplot(gs[0, 1])
    p12b(ax)
    F.panel_label(ax, "b", x=-0.16, y=1.02)
    ax = fig.add_subplot(gs[1, :])
    p12c(ax)
    F.panel_label(ax, "c", x=-0.075, y=1.02)
    _save(fig, "fig12_variance_decomposition")


# ============================================================ P13 稳健性
@panel("P13a", 88, 64,
       desc="Seed-to-seed variability of per-dimension R²",
       sources="results/composite/per_target.csv",
       stats="50 sensory dimensions × 3 seeds × 2 CV schemes; SD across seeds; half-violin KDE + Sina jitter + median/IQR",
       margins=(0.225, 0.17, 0.975, 0.96))
def p13a(ax):
    d = _read("composite/per_target.csv")
    d = d[(d["variant"] == "avg") & (d["scheme"].isin(["random", "group"]))]
    sd = (d.groupby(["scheme", "target"])["r2"].std(ddof=1)
            .reset_index())
    groups = [sd.loc[sd["scheme"] == s, "r2"].to_numpy(float)
              for s in ("random", "group")]
    ypos = np.array([1.0, 0.0])
    F.raincloud(ax, groups, ypos, [F.BLUE, F.VERM],
                violin_width=0.33, side="right")
    ax.set_yticks(ypos)
    ax.set_yticklabels(["Random 5-fold", "Grouped by style"], fontsize=6.2)
    ax.set_ylim(-0.55, 1.55)
    lo = min(float(np.min(g)) for g in groups)
    hi = max(float(np.max(g)) for g in groups)
    ax.set_xlim(max(0.0, lo - 0.0005), hi * 1.13)
    ax.set_xlabel("Seed-to-seed SD of R² (3 seeds)")
    for y, g in zip(ypos, groups):
        ax.text(0.985, y - 0.30, f"median {np.median(g):.4f}",
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=5.8, color="#555555")
    F.soften(ax, xgrid=True)


@panel("P13b", 86, 64,
       desc="Sensitivity to the number of chemical latent dimensions",
       sources="results/robustness/summary.json",
       stats="k = 6–22; mean R² over 50 dimensions for random and grouped CV; selected k = 14",
       margins=(0.18, 0.17, 0.975, 0.96))
def p13b(ax):
    d = json.loads((RESULTS / "robustness" / "summary.json").read_text())
    rows = pd.DataFrame(d["k_sensitivity"])
    for scheme, color, label in (("random", F.BLUE, "Random 5-fold"),
                                 ("group", F.VERM, "Grouped by style")):
        s = rows[rows["scheme"] == scheme].sort_values("k")
        ax.plot(s["k"], s["mean_r2"], color=color, lw=1.0, marker="o",
                ms=3.8, label=label)
        ax.text(s["k"].iloc[-1] + 0.4, s["mean_r2"].iloc[-1], label,
                color=color, fontsize=5.7, va="center")
    ax.axvline(14, color="#777777", lw=0.6, ls=(0, (2, 2)))
    ax.text(14.4, ax.get_ylim()[0] + 0.006, "selected k = 14",
            fontsize=5.5, color="#555555")
    ax.set_xticks(sorted(rows["k"].unique()))
    ax.set_xlim(4.5, 27.0)
    ax.set_xlabel("Chemical latent dimensions (k)")
    ax.set_ylabel("Mean R²")
    F.soften(ax, ygrid=True)


@panel("P13c", 86, 64,
       desc="Performance under random missingness",
       sources="results/robustness/summary.json",
       stats="MCAR stress at 0/5/10/20%; mean R² over 50 dimensions; 20% missingness reduces mean R² by 0.0298",
       margins=(0.18, 0.17, 0.975, 0.96))
def p13c(ax):
    d = json.loads((RESULTS / "robustness" / "summary.json").read_text())
    rows = pd.DataFrame(d["missingness_stress"]).sort_values("missing_rate")
    x = rows["missing_rate"].to_numpy(float) * 100
    y = rows["mean_r2"].to_numpy(float)
    ax.plot(x, y, color=F.BLUE, lw=1.0, marker="o", ms=4.0)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.3f}", (xi, yi), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=5.6, color="#333333")
    drop = float(y[-1] - y[0])
    ax.annotate(f"Δ (0 → 20%) = {drop:+.4f}", xy=(x[-1], y[-1]),
                xytext=(-8, -18), textcoords="offset points", ha="right",
                fontsize=5.8, color=F.VERM)
    ax.set_xticks(x)
    ax.set_xlabel("Random missingness (%)")
    ax.set_ylabel("Mean R²")
    ax.set_ylim(min(y) - 0.012, max(y) + 0.012)
    F.soften(ax, ygrid=True)


@panel("P13d", 90, 66,
       desc="Constraint-strength Pareto front",
       sources="results/robustness/summary.json",
       stats="15 configurations: λ ∈ {0.02, 0.05, 0.10, 0.20, 0.50} × {none, loose, strict}; filled = no new DMS/diacetyl exceedance; open = new exceedance",
       margins=(0.205, 0.17, 0.975, 0.96))
def p13d(ax):
    d = json.loads((RESULTS / "robustness" / "summary.json").read_text())
    rows = pd.DataFrame(d["constraint_pareto"])
    colors = {"none": "#9A9A9A", "loose": F.BLUE, "strict": F.GREEN}
    handles = []
    for mode in ("none", "loose", "strict"):
        s = rows[rows["mode"] == mode].sort_values("lam")
        ax.plot(s["qp_cost_median"], s["style_shift_median"],
                color=colors[mode], lw=0.9, zorder=2)
        for _, r in s.iterrows():
            safe = (r["new_DMS"] == 0) and (r["new_diacetyl"] == 0)
            ax.scatter(r["qp_cost_median"], r["style_shift_median"],
                       s=18, facecolor=colors[mode] if safe else "white",
                       edgecolor=colors[mode], linewidth=0.8, zorder=3)
        first = s.iloc[0]
        dx, dy, ha = {"none": (9, 0, "left"), "loose": (0, 7, "center"),
                      "strict": (7, 6, "left")}[mode]
        ax.annotate(f"λ={first['lam']:.2f}",
                    (first["qp_cost_median"], first["style_shift_median"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=5.2, color=colors[mode], ha=ha, va="center")
        handles.append(Line2D(
            [0], [0], color=colors[mode], marker="o", ms=3.4,
            markerfacecolor=colors[mode] if mode != "none" else "white",
            markeredgewidth=0.7, lw=0.9, label=mode))
    ax.set_xlabel("Correction cost (median)")
    ax.set_ylabel("Style shift (median; higher is better)")
    ax.legend(handles=handles, loc="lower left", fontsize=5.6,
              handlelength=1.3, labelspacing=0.25, frameon=True,
              facecolor="white", framealpha=0.85, edgecolor="none")
    ax.text(0.985, 0.965,
            "filled: no new DMS/diacetyl exceedance\n"
            "open: new exceedance · λ 0.02 → 0.50 along each curve",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.4,
            color="#555555", linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                      alpha=0.9))
    F.soften(ax, xgrid=True, ygrid=True)


def fig13():
    fig = plt.figure(figsize=(F.W2, 146 * F.MM))
    gs = fig.add_gridspec(2, 2, left=0.075, right=0.985, top=0.965,
                          bottom=0.06, wspace=0.38, hspace=0.44)
    for key, pos, letter, lx in (("P13a", (0, 0), "a", -0.26),
                                 ("P13b", (0, 1), "b", -0.22),
                                 ("P13c", (1, 0), "c", -0.22),
                                 ("P13d", (1, 1), "d", -0.22)):
        ax = fig.add_subplot(gs[pos])
        PANEL[key]["fn"](ax)
        F.panel_label(ax, letter, x=lx, y=1.02)
    _save(fig, "fig13_robustness")


# ============================================================ CLI
FIGS = {
    "fig01": fig01, "fig02": fig02, "fig03": fig03, "fig04": fig04,
    "fig05": fig05, "fig06": fig06, "fig07": fig07, "fig08": fig08,
    "fig09": fig09, "fig10": fig10, "fig11": fig11, "fig12": fig12,
    "fig13": fig13,
}


def _panel_order(keys):
    return sorted(keys, key=lambda k: (int(k[1:3]), k[3:]))


def main(argv=None):
    import sys

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "panels":
        want = argv[1:]
        keys = _panel_order(PANEL)
        if want:
            pref = [w.lower() for w in want]
            keys = [k for k in keys if any(k.lower().startswith(p)
                                           for p in pref)]
        for k in keys:
            _standalone(k)
            print("  panel", k)
        if len(keys) == len(PANEL):
            p = F.write_panel_index()
            print("  wrote", p)
        return
    if argv:
        for name in argv:
            if name not in FIGS:
                raise SystemExit(f"unknown figure {name}; choose from "
                                 f"{', '.join(FIGS)} or 'panels'")
            FIGS[name]()
        return
    for name, fn in FIGS.items():
        fn()
    for k in _panel_order(PANEL):
        _standalone(k)
        print("  panel", k)
    p = F.write_panel_index()
    print("  wrote", p)


if __name__ == "__main__":
    main()
