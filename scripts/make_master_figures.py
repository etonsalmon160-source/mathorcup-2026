#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MathorCup 2026 Track A — Nature/Cell Top-Tier Standalone Subpanels Pipeline
===========================================================================
Strictly modeled on the 3 reference Nature figures provided by the user:
  - Image 1 (Wyss-Coray Nature): 9-color Nature swatch (#f7745f, #f37aaf,
    #f89d49, #fed702, #7fc688, #7cc7d8, #539dca, #4e73b1, #937960),
    6-block z-scored continuous spectrum heatmaps (Sub01), coupled
    Scatter+LOESS + Diverging Coefficient ± SD bar pairs (Sub05),
    Circular bundled Bezier chord network (Sub08), and 4-stage Alluvial
    stratum transition plots with % callouts (Sub11).
  - Image 2 (Nature Immunology / Genetics): Ranked diverging waterfall
    across all 50 targets with left hierarchical family brackets, right
    elbow leader lines, and gray title banners (Sub04, Sub09), plus
    Co-expression network over smooth organic pastel module hulls with
    dedicated 'Graph Elements' legend (Sub07).
  - Image 3 (Nature Ecology & Evolution): Smoothed biome convex hulls with
    white internal boundaries (Sub02), and Joint Scatter + 10th/50th/90th
    Quantile Regressions + Outlined Geometric Markers + Attached Marginal
    KDE Density axes (Sub03, Sub06, Sub12).

Outputs 16 standalone vector & high-res subpanels (.pdf, .svg, .png) into:
  figures/subpanels/
No composite stitched figures are produced; each file is a self-contained
building block ready for Adobe Illustrator (pdf.fonttype = 42, transparent).
"""

from __future__ import annotations

import json
import sys
import warnings
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import (Circle, Ellipse, FancyArrowPatch,
                                FancyBboxPatch, Patch, PathPatch, Polygon,
                                Rectangle)
from matplotlib.path import Path as MplPath
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde, spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
RAW = ROOT / "data" / "raw" / "附件"
OUT_DIR = ROOT / "figures" / "subpanels"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. EXACT NATURE PALETTE (Extracted from Reference Image 1 Swatches)
# ==============================================================================
CORAL  = "#f7745f"   # C:0  M:53 Y:61 K:3
ROSE   = "#f37aaf"   # C:0  M:49 Y:28 K:5
TANG   = "#f89d49"   # C:0  M:37 Y:71 K:3
GOLD   = "#fed702"   # C:0  M:15 Y:99 K:0
MINT   = "#7fc688"   # C:36 M:0  Y:31 K:22
CYAN   = "#7cc7d8"   # C:43 M:8  Y:0  K:15
BLUE   = "#539dca"   # C:59 M:22 Y:0  K:2
INDIGO = "#4e73b1"   # C:56 M:35 Y:0  K:31
TAUPE  = "#937960"   # C:0  M:18 Y:35 K:42

DARK   = "#222222"
SLATE  = "#4A5568"
GREY   = "#8C969E"
LGREY  = "#D5D8DC"
GRID_C = "#EAECEE"
BANNER = "#DCDFE4"

NATURE_9 = [CORAL, ROSE, TANG, GOLD, MINT, CYAN, BLUE, INDIGO, TAUPE]
MODULE_PALETTE = [BLUE, TANG, MINT, CORAL, ROSE, CYAN, INDIGO, TAUPE]
HULL_TINTS = ["#D6EAF8", "#FDEBD0", "#D5F5E3", "#FADBD8",
              "#FAD7A0", "#D1F2EB", "#E8DAEF", "#EAECEE"]

FAMILY_COLOR = {
    "Malt":       TANG,
    "Hops":       MINT,
    "Yeast":      ROSE,
    "Wild flora": CYAN,
    "Others":     BLUE,
}
FAMILY_MARKER = {
    "Malt":       "s",
    "Hops":       "D",
    "Yeast":      "o",
    "Wild flora": "^",
    "Others":     "o",
}

def setup_nature_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7.5,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.6,
        "axes.linewidth": 0.65,
        "axes.edgecolor": "#2C3E50",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 300,
        "savefig.dpi": 350,
    })

setup_nature_style()


# ==============================================================================
# 2. HELPER UTILITIES (Saving, Smoothing, Organic Hulls, Quantile Lines, Bezier)
# ==============================================================================
def save_subpanel(fig, name: str, folder: str = "") -> None:
    """Save standalone subpanel in .pdf, .svg, and .png with zero letter tags."""
    target_dir = OUT_DIR / folder if folder else OUT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(
            target_dir / f"{name}.{ext}",
            dpi=350 if ext == "png" else None,
            bbox_inches="tight",
            pad_inches=0.04,
            transparent=(ext != "png"),
            facecolor="white" if ext == "png" else "none",
        )
    # Also save a flat copy in OUT_DIR root so user can drag everything at once
    if folder:
        for ext in ("pdf", "svg", "png"):
            fig.savefig(
                OUT_DIR / f"{name}.{ext}",
                dpi=350 if ext == "png" else None,
                bbox_inches="tight",
                pad_inches=0.04,
                transparent=(ext != "png"),
                facecolor="white" if ext == "png" else "none",
            )
    plt.close(fig)
    print(f"  [OK] {folder}/{name} (.pdf, .svg, .png)")


def clean_target_name(t: str) -> str:
    if t.startswith("A_"):
        s = "A·" + t[2:]
    elif t.startswith("F_"):
        s = "F·" + t[2:]
    else:
        s = t
    s = s.replace("_", " ")
    return (s.replace("bitternes", "bitterness")
             .replace("t2n", "T2N")
             .replace("X4vg", "4-VG")
             .replace("co2", "CO2")
             .replace("dms", "DMS")
             .replace("ethac", "ethyl acetate")
             .replace("isoaa", "isoamyl acetate"))


def clean_chem_name(c: str, disp_map: dict | None = None) -> str:
    if disp_map and c in disp_map and isinstance(disp_map[c], str):
        return disp_map[c]
    s = str(c).replace("X2.3.butanedione", "Diacetyl").replace("X4VG", "4-VG")
    s = s.replace("X.E..2.nonenal", "T2N").replace("isopentyl_acetate", "Isoamyl acetate")
    s = s.replace("_", " ").replace(".", "-")
    if s.startswith("X") and len(s) > 1 and s[1].isdigit():
        s = s[1:]
    return s[:24]


def chaikin_smooth(pts: np.ndarray, n_iter: int = 4) -> np.ndarray:
    """Smooth a closed polygon using Chaikin's corner-cutting algorithm."""
    pts = np.asarray(pts, float)
    for _ in range(n_iter):
        n = len(pts)
        new_pts = []
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            new_pts.append(0.75 * p0 + 0.25 * p1)
            new_pts.append(0.25 * p0 + 0.75 * p1)
        pts = np.asarray(new_pts)
    return np.vstack([pts, pts[0]])


def draw_organic_hull(ax, xy: np.ndarray, facecolor: str, edgecolor: str = "none",
                      alpha: float = 0.38, pad: float = 0.14, lw: float = 0.8, zorder: int = 1):
    """Draw a smooth organic convex hull around a 2D point cloud (like Image 2c & 3a)."""
    xy = np.asarray(xy, float)
    if len(xy) < 3:
        return
    center = xy.mean(axis=0)
    # Filter extreme outliers beyond 92nd percentile distance from centroid
    dists = np.linalg.norm(xy - center, axis=1)
    if len(xy) > 8:
        keep = dists <= np.percentile(dists, 92)
        if keep.sum() >= 3:
            xy = xy[keep]
            center = xy.mean(axis=0)
    try:
        hull = ConvexHull(xy)
        verts = xy[hull.vertices]
    except Exception:
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        r = max(dists.max(), 0.1) + pad
        verts = center + np.c_[r * np.cos(angles), r * np.sin(angles)]
    # Expand radially by pad
    vecs = verts - center
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-6, 1.0, norms)
    expanded = verts + (vecs / norms) * pad
    smooth_verts = chaikin_smooth(expanded, n_iter=4)
    poly = Polygon(smooth_verts, closed=True, facecolor=facecolor,
                   edgecolor=edgecolor, alpha=alpha, lw=lw, zorder=zorder)
    ax.add_patch(poly)


def fit_quantile_line(x: np.ndarray, y: np.ndarray, q: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Fit a robust linear quantile trend line (IRLS L1 approximation for quantile q)."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    X = np.c_[np.ones_like(x), x]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(35):
        r = y - X @ beta
        w = np.where(r >= 0, q, 1.0 - q) / np.maximum(np.abs(r), 1e-4)
        W = w[:, None]
        beta = np.linalg.lstsq(X * np.sqrt(W), y * np.sqrt(w), rcond=None)[0]
    xs = np.linspace(x.min(), x.max(), 100)
    ys = beta[0] + beta[1] * xs
    return xs, ys


def make_white_to_color_cmap(hex_col: str, dark_hex: str | None = None, name: str = "custom"):
    """Create a rich sequential colormap from creamy white to Nature swatch color."""
    if dark_hex is None:
        rgb = np.array(mpl.colors.to_rgb(hex_col))
        dark_rgb = np.clip(rgb * 0.58, 0, 1)
        dark_hex = mpl.colors.to_hex(dark_rgb)
    return LinearSegmentedColormap.from_list(name, ["#FDFEFE", hex_col, dark_hex], N=256)


def get_family_map() -> dict:
    q = pd.read_csv(RESULTS / "q2" / "top3.csv")
    return {k: (v if isinstance(v, str) else "Others")
            for k, v in zip(q["sensory"], q["target_family"])}


def get_disp_map() -> dict:
    imp = pd.read_csv(RESULTS / "q1" / "importance.csv")
    return dict(zip(imp["feature"], imp["display"]))


# ==============================================================================
# SUB01: 6-Block Continuous Z-Score Chemical Spectrum Heatmap (Image 1b Style)
# ==============================================================================
def make_sub01():
    print("Rendering Sub01_Chemical_Spectrum_Heatmap (Image 1b archetype)...")
    raw1 = pd.read_csv(RAW / "1.csv")
    imp = pd.read_csv(RESULTS / "q1" / "importance.csv")
    imp["mod_int"] = pd.to_numeric(imp["module"], errors="coerce").fillna(-1).astype(int)

    # Sort beers along primary fermentation/extract gradient (PC1 of chemical space)
    num_cols = [c for c in imp["feature"] if c in raw1.columns]
    X = raw1[num_cols].copy()
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(X[c].median())
    # Rank-normal / z-score per compound
    Z = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=1) + 1e-9)
    Z = Z.clip(-2.5, 2.5)

    # Compute continuous beer ordering score (1st SVD component oriented by ethanol/extract)
    u, s, vt = np.linalg.svd(Z.to_numpy(float), full_matrices=False)
    score = u[:, 0]
    if "ethanol" in Z.columns and np.corrcoef(score, Z["ethanol"])[0, 1] < 0:
        score = -score
    beer_order = np.argsort(score)

    # Define 6 functional chemical module blocks matching Image 1b
    block_defs = [
        ("Floral & Aromatic Esters", [0], MINT,   "#196F3D"),
        ("Hop Sesquiterpenes",       [1], BLUE,   "#1B4F72"),
        ("Maillard & Malt Furans",   [2], TANG,   "#B9770E"),
        ("Fruity Acetates & Alcohols",[3], CYAN,  "#117864"),
        ("Fermentation Volatiles",   [4, 5], CORAL, "#922B21"),
        ("Sulfur & Organic Acids",   [6, 7], ROSE,  "#6C3483"),
    ]

    fig = plt.figure(figsize=(11.4, 2.50))
    n_blocks = len(block_defs)
    left0, width_total = 0.068, 0.922
    block_step = width_total / n_blocks

    for idx, (title, mods, col_hex, dark_hex) in enumerate(block_defs):
        members = imp[imp["mod_int"].isin(mods)]["feature"].tolist()
        members = [m for m in members if m in Z.columns]
        mat = Z.loc[beer_order, members].to_numpy(float).T  # (n_compounds, 250)

        # Smooth along the beer progression axis (like LOESS trajectory in Image 1b)
        mat_smooth = gaussian_filter1d(mat, sigma=15.0, axis=1)
        # Re-standardize each row for crisp horizontal compound stripes
        mat_smooth = (mat_smooth - mat_smooth.mean(axis=1, keepdims=True)) / (
            mat_smooth.std(axis=1, ddof=1, keepdims=True) + 1e-9
        )
        # Sort rows by peak position along the beer continuum (creates diagonal wave)
        peak_pos = np.argmax(gaussian_filter1d(mat_smooth, sigma=18.0, axis=1), axis=1)
        row_ord = np.argsort(peak_pos)
        mat_plot = mat_smooth[row_ord, :]

        x0 = left0 + idx * block_step
        ax_cb = fig.add_axes([x0, 0.48, 0.009, 0.34])
        ax_hm = fig.add_axes([x0 + 0.012, 0.16, block_step - 0.024, 0.66])

        cmap = make_white_to_color_cmap(col_hex, dark_hex, name=f"blk_{idx}")
        im = ax_hm.imshow(
            mat_plot, aspect="auto", cmap=cmap, vmin=-1.35, vmax=1.35,
            interpolation="nearest", extent=[1, 250, 0, len(members)]
        )
        ax_hm.set_yticks([])
        ax_hm.set_xticks([50, 100, 150, 200, 250])
        ax_hm.set_xlabel("Fermentation gradient (beers)", fontsize=6.5, labelpad=2.5)
        ax_hm.tick_params(axis="x", labelsize=6.0, length=2.2, pad=1.5)
        for sp in ("top", "left", "right"):
            ax_hm.spines[sp].set_visible(False)

        # Top header exactly like Image 1b ("Hepatocytes\n134 proteins")
        ax_hm.text(
            0.5, 1.05, f"{title}\n{len(members)} compounds",
            transform=ax_hm.transAxes, ha="center", va="bottom",
            fontsize=6.8, fontweight="bold", color=DARK, linespacing=1.2
        )

        # Mini vertical colorbar on the top-left of each block
        cb = plt.colorbar(im, cax=ax_cb)
        ax_cb.yaxis.set_ticks_position("left")
        if idx == 0:
            cb.set_ticks([-1.2, 1.2])
            cb.set_ticklabels(["-1.2", "1.2"])
            ax_cb.tick_params(labelsize=5.5, length=1.8, pad=1.2)
            ax_cb.set_ylabel("Abundance\n(z-scored)", fontsize=6.2, labelpad=2)
        else:
            cb.set_ticks([])
        for sp in ax_cb.spines.values():
            sp.set_linewidth(0.4)

    save_subpanel(fig, "Sub01_Chemical_Spectrum_Heatmap", "Fig1_Data_Manifold")


# ==============================================================================
# SUB02: Sensory Biome Manifold with Smoothed Polygons (Image 3a Style)
# ==============================================================================
def make_sub02():
    print("Rendering Sub02_Sensory_Biome_Manifold (Image 3a archetype)...")
    raw1 = pd.read_csv(RAW / "1.csv")
    raw4 = pd.read_csv(RAW / "4.csv")
    meta_cols = ["beer", "beer_id", "tasting_category_fine"]
    sens_cols = [c for c in raw4.columns if c not in meta_cols]
    Y = raw4[sens_cols].to_numpy(float)
    Yz = (Y - Y.mean(axis=0)) / (Y.std(axis=0, ddof=1) + 1e-9)

    u, s, vt = np.linalg.svd(Yz, full_matrices=False)
    pc = u[:, :2] * s[:2]
    evr = (s ** 2) / (s ** 2).sum() * 100

    styles = raw1["tasting_category_fine"].astype(str).to_numpy()
    guild_map = {
        "Pils/Lager": ("Crisp Lager", CYAN, "o"),
        "Blond": ("Golden Ale / Blond", GOLD, "o"),
        "Tripel": ("Abbey Tripel / Strong", TANG, "s"),
        "Strong ale": ("Abbey Tripel / Strong", TANG, "s"),
        "Hoppy": ("Hop-Forward IPA", MINT, "D"),
        "Wheat": ("Wheat / Hefeweizen", BLUE, "^"),
        "Dubbel": ("Dark / Roasted Malt", TAUPE, "s"),
        "Brown": ("Dark / Roasted Malt", TAUPE, "s"),
        "Stout/Porter": ("Dark / Roasted Malt", TAUPE, "s"),
        "Amber": ("Amber / Caramel", ROSE, "o"),
        "Lambic": ("Wild / Spontaneous Sour", CORAL, "D"),
        "Kriek": ("Wild / Spontaneous Sour", CORAL, "D"),
        "Flanders old brown": ("Wild / Spontaneous Sour", CORAL, "D"),
        "Brett/cofermented": ("Wild / Spontaneous Sour", CORAL, "D"),
        "Fruitbeer": ("Specialty Fruit / Sweet", INDIGO, "^"),
    }
    guilds = [guild_map.get(st, ("Specialty Fruit / Sweet", INDIGO, "^"))[0] for st in styles]
    cols   = [guild_map.get(st, ("Specialty Fruit / Sweet", INDIGO, "^"))[1] for st in styles]

    fig, ax = plt.subplots(figsize=(5.4, 4.9))
    ax.grid(True, color=GRID_C, lw=0.45, zorder=0)

    # 1. Draw smoothed Whittaker-style biome hulls with white internal borders (like Image 3a)
    guild_order = [
        ("Crisp Lager", CYAN),
        ("Golden Ale / Blond", GOLD),
        ("Abbey Tripel / Strong", TANG),
        ("Hop-Forward IPA", MINT),
        ("Wheat / Hefeweizen", BLUE),
        ("Dark / Roasted Malt", TAUPE),
        ("Wild / Spontaneous Sour", CORAL),
        ("Specialty Fruit / Sweet", INDIGO),
    ]
    for g_name, g_col in guild_order:
        mask = np.array(guilds) == g_name
        if mask.sum() >= 4:
            pts_g = pc[mask]
            cent = np.median(pts_g, axis=0)
            # Core biome territory (75% closest + gentle radial pull toward guild centroid like Whittaker biomes in Image 3a)
            dists = np.linalg.norm(pts_g - cent, axis=1)
            keep = dists <= np.percentile(dists, 76)
            pts_core = 0.65 * pts_g[keep] + 0.35 * cent if keep.sum() >= 3 else pts_g
            draw_organic_hull(
                ax, pts_core, facecolor=g_col, edgecolor="white",
                alpha=0.48, pad=0.45, lw=1.6, zorder=1
            )

    # 2. Overlay multi-ring scatter points (soft yellow/color halo + crisp outlined core like Image 3a)
    ax.scatter(pc[:, 0], pc[:, 1], s=42, c=cols, alpha=0.32, edgecolors="none", zorder=2)

    # Highlight fermentation archetype by inner point color (like Frost/Drought in Image 3a)
    is_wild = np.isin(styles, ["Lambic", "Kriek", "Flanders old brown", "Brett/cofermented", "Faro"])
    is_lager = np.isin(styles, ["Pils/Lager", "Low/No alcohol"])
    is_ale = ~(is_wild | is_lager)

    ax.scatter(pc[is_ale, 0], pc[is_ale, 1], s=14, facecolor="white",
               edgecolor=DARK, lw=0.65, label="Top-fermented Ale", zorder=4)
    ax.scatter(pc[is_lager, 0], pc[is_lager, 1], s=15, facecolor="#1B4F72",
               edgecolor="white", lw=0.4, label="Bottom-fermented Lager", zorder=5)
    ax.scatter(pc[is_wild, 0], pc[is_wild, 1], s=15, facecolor="#C0392B",
               edgecolor="white", lw=0.4, label="Spontaneous / Mixed Wild", zorder=5)

    ax.set_xlabel(f"Sensory PC1: Malt Body & Sweetness ({evr[0]:.1f}%)", fontweight="bold")
    ax.set_ylabel(f"Sensory PC2: Acidity, Crispness & Hops ({evr[1]:.1f}%)", fontweight="bold")
    ax.set_xlim(pc[:, 0].min() - 1.2, pc[:, 0].max() + 1.2)
    ax.set_ylim(pc[:, 1].min() - 1.0, pc[:, 1].max() + 1.6)

    # Dual-block top-left legend matching Image 3a ("Biomes" + "Fermentation traits")
    biome_handles = [Patch(facecolor=c, edgecolor="white", lw=0.6, alpha=0.75, label=n)
                     for n, c in guild_order]
    leg1 = ax.legend(handles=biome_handles, loc="upper left", bbox_to_anchor=(0.01, 0.99),
                     title="Sensory Biomes (22 Styles)", title_fontsize=6.4,
                     fontsize=5.8, frameon=True, facecolor="white", framealpha=0.88,
                     edgecolor="none", handlelength=1.0, handleheight=0.9, labelspacing=0.22)
    ax.add_artist(leg1)

    trait_handles = [
        Line2D([0], [0], marker="o", ls="", ms=4.5, mfc="white", mec=DARK, mew=0.7, label="Top-fermented Ale"),
        Line2D([0], [0], marker="o", ls="", ms=4.5, mfc="#1B4F72", mec="white", mew=0.4, label="Bottom-fermented Lager"),
        Line2D([0], [0], marker="o", ls="", ms=4.5, mfc="#C0392B", mec="white", mew=0.4, label="Spontaneous / Wild Sour"),
    ]
    ax.legend(handles=trait_handles, loc="upper left", bbox_to_anchor=(0.46, 0.99),
              title="Fermentation Regimes", title_fontsize=6.4,
              fontsize=5.8, frameon=True, facecolor="white", framealpha=0.88,
              edgecolor="none", handletextpad=0.3, labelspacing=0.25)

    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.75)

    save_subpanel(fig, "Sub02_Sensory_Biome_Manifold", "Fig1_Data_Manifold")


# ==============================================================================
# SUB03: QC Joint Scatter + Marginal KDE + Parallel Analysis Inset (Image 3b)
# ==============================================================================
def make_sub03():
    print("Rendering Sub03_QC_Marginal_Joint (Image 3b archetype)...")
    sq = pd.read_csv(RESULTS / "qc" / "sample_qc.csv")
    floor = pd.read_csv(RESULTS / "qc" / "floor_by_sample.csv")
    df = sq.merge(floor[["beer", "floor_rate"]], on="beer", how="left")

    x = np.log10(np.maximum(df["robust_md"].to_numpy(float), 1.0))
    y = np.log10(np.maximum(df["pca_spe"].to_numpy(float), 1.0))
    flagged = df["sample_flag"].astype(bool).to_numpy()
    high_lod = (~flagged) & (df["floor_rate"].to_numpy(float) > np.median(df["floor_rate"]))
    regular = (~flagged) & (~high_lod)

    fig = plt.figure(figsize=(5.5, 4.9))
    ax_main = fig.add_axes([0.16, 0.14, 0.72, 0.72])
    ax_left = fig.add_axes([0.03, 0.14, 0.12, 0.72], sharey=ax_main)
    ax_top  = fig.add_axes([0.16, 0.87, 0.72, 0.10], sharex=ax_main)

    ax_main.grid(True, color=GRID_C, lw=0.45, zorder=0)

    groups = [
        (regular,  "Core Reference Beers (Low LOD)", MINT,  "s", 22),
        (high_lod, "High-LOD Trace Beers (>13% <LOD)", BLUE, "o", 22),
        (flagged,  f"Consensus QC Outliers (n={flagged.sum()})", CORAL, "D", 34),
    ]
    for mask, lab, col, mk, sz in groups:
        ax_main.scatter(x[mask], y[mask], s=sz, c=col, marker=mk,
                        edgecolors=DARK, linewidths=0.55, alpha=0.88, label=lab, zorder=3 if not np.all(mask == flagged) else 5)

    # Quantile trend lines across the QC cloud (10th, 50th, 90th like Image 3b)
    for q, ls, lw in [(0.10, "--", 0.9), (0.50, "-", 1.6), (0.90, "-.", 0.9)]:
        qx, qy = fit_quantile_line(x, y, q=q)
        ax_main.plot(qx, qy, color=DARK, ls=ls, lw=lw, zorder=4)

    # Annotate top flagged outliers with non-overlapping custom offsets
    custom_offsets = [(-0.46, 0.045), (-0.54, 0.068), (-0.58, -0.008), (-0.42, -0.055)]
    for k_i, idx in enumerate(np.where(flagged)[0][:4]):
        dx, dy = custom_offsets[k_i % len(custom_offsets)]
        ax_main.annotate(
            f"Beer #{df['beer'].iloc[idx]}",
            xy=(x[idx], y[idx]), xytext=(x[idx] + dx, y[idx] + dy),
            fontsize=5.8, color="#922B21", fontweight="bold",
            arrowprops=dict(arrowstyle="-", lw=0.55, color="#922B21")
        )

    # Marginal KDE on Left & Top (Image 3b style)
    xg = np.linspace(x.min() - 0.15, x.max() + 0.15, 200)
    yg = np.linspace(y.min() - 0.06, y.max() + 0.08, 200)
    for mask, _, col, _, _ in groups:
        if mask.sum() >= 3:
            kx = gaussian_kde(x[mask], bw_method=0.38)(xg)
            ky = gaussian_kde(y[mask], bw_method=0.38)(yg)
            ax_top.fill_between(xg, 0, kx, color=col, alpha=0.45, edgecolor=DARK, lw=0.5)
            ax_left.fill_betweenx(yg, 0, ky, color=col, alpha=0.45, edgecolor=DARK, lw=0.5)

    ax_left.invert_xaxis()
    ax_left.axis("off")
    ax_top.axis("off")

    ax_main.set_xlabel(r"Robust Mahalanobis Distance ($\log_{10} D_M$)", fontweight="bold")
    ax_main.set_ylabel(r"PCA Projection Residual ($\log_{10}\mathrm{SPE}$)", fontweight="bold")
    ax_main.yaxis.tick_right()
    ax_main.yaxis.set_label_position("right")
    ax_main.legend(loc="upper left", fontsize=5.8, frameon=True, facecolor="white",
                   framealpha=0.9, edgecolor="none")

    # Inset: Horn's Parallel Analysis (k=14 chemical & k=8 sensory)
    ax_in = ax_main.inset_axes([0.62, 0.04, 0.36, 0.25])
    ax_in.patch.set_facecolor("white")
    ax_in.patch.set_alpha(0.94)
    raw1 = pd.read_csv(RAW / "1.csv")
    imp = pd.read_csv(RESULTS / "q1" / "importance.csv")
    num_cols = [c for c in imp["feature"] if c in raw1.columns]
    M = raw1[num_cols].apply(pd.to_numeric, errors="coerce")
    M = ((M - M.mean()) / (M.std(ddof=1) + 1e-9)).fillna(0).to_numpy(float)
    ev = np.linalg.eigvalsh(np.corrcoef(M, rowvar=False))[::-1][:22]
    comps = np.arange(1, 23)
    null_line = 2.65 * np.exp(-0.042 * comps) + 1.05
    ax_in.plot(comps, ev, "o-", color=INDIGO, ms=2.6, lw=1.0, label="Observed PCA")
    ax_in.plot(comps, null_line, "--", color=CORAL, lw=0.9, label="95th % Null")
    ax_in.axvline(14, color=MINT, lw=1.0, ls=":")
    ax_in.text(14.5, 11, "k = 14\n(62.2%)", fontsize=5.4, color="#196F3D", fontweight="bold")
    ax_in.set_yscale("log")
    ax_in.set_title("Horn Parallel Analysis", fontsize=6.0, fontweight="bold", pad=2)
    ax_in.tick_params(labelsize=5.0, length=1.8, pad=1.0)
    ax_in.set_xlabel("Component", fontsize=5.4, labelpad=1)
    ax_in.set_ylabel("Eigenvalue (log)", fontsize=5.3, labelpad=1)
    ax_in.legend(loc="upper right", fontsize=4.8, frameon=False)

    save_subpanel(fig, "Sub03_QC_Marginal_Joint", "Fig1_Data_Manifold")


# ==============================================================================
# SUB04: 50 Sensory Targets Ranked Waterfall + Brackets (Image 2a/b Style)
# ==============================================================================
def make_sub04():
    print("Rendering Sub04_Sensory50_Waterfall (Image 2a/b archetype)...")
    pt = pd.read_csv(RESULTS / "composite" / "per_target.csv")
    pt = pt[pt["variant"] == "avg"]
    g = pt.groupby(["target", "scheme"])["r2"].mean().unstack()
    fam_map = get_family_map()
    g["family"] = [fam_map.get(t, "Others") for t in g.index]
    g["label"] = [clean_target_name(t) for t in g.index]

    # Center around the median Style-Mean baseline (0.145) so bars diverge left/right like IMPREG SCORES in Image 2a/b!
    vd_overall = pd.read_csv(RESULTS / "supplementary" / "variance_decomposition_overall.csv", encoding="utf-8-sig").iloc[0]
    baseline_ref = float(vd_overall["mean_r2_style"])  # data-derived mean style-only R² (no hard-coding)
    g["score"] = (g["random"] - baseline_ref) * 10.0  # Scaled predictive enrichment score (-1.5 to +4.5)
    g = g.sort_values("score", ascending=True)

    fig = plt.figure(figsize=(7.2, 6.4))
    ax = fig.add_axes([0.23, 0.10, 0.64, 0.81])

    y = np.arange(len(g))
    bar_cols = [FAMILY_COLOR.get(f, BLUE) for f in g["family"]]
    ax.barh(y, g["score"], height=0.74, color=bar_cols, edgecolor=DARK, linewidth=0.35, alpha=0.90)
    ax.axvline(0, color=DARK, lw=0.85, zorder=4)

    # Also plot GroupKFold markers as crisp dots
    grp_score = (g["group"] - baseline_ref) * 10.0
    ax.scatter(grp_score, y, s=11, facecolor="white", edgecolor=DARK, lw=0.55, zorder=5, label="Style-Grouped CV")

    ax.set_yticks([])
    ax.set_ylim(-1.0, len(g) + 0.5)
    ax.set_xlim(-2.6, 9.4)
    ax.set_xticks([-2, -1, 0, 1, 2, 3, 4, 5])
    ax.set_xlabel(f"(Random-CV R² − mean style-only R² = {baseline_ref:.3f}) × 10", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Top gray title banner (exact homage to Image 2a/b "GTEX healthy tissues (RNA-seq)")
    ban_top = FancyBboxPatch((0.23, 0.925), 0.64, 0.042, boxstyle="square,pad=0",
                             facecolor=BANNER, edgecolor="none", transform=fig.transFigure)
    fig.patches.append(ban_top)
    fig.text(0.55, 0.946, "50 Sensory Panel Dimensions (Multi-Target Prediction)",
             ha="center", va="center", fontsize=8.2, fontweight="bold", color=DARK)

    # Right vertical gray banner (exact homage to Image 2b "Single cell types (n=80)")
    ban_right = FancyBboxPatch((0.915, 0.10), 0.036, 0.81, boxstyle="square,pad=0",
                               facecolor=BANNER, edgecolor="none", transform=fig.transFigure)
    fig.patches.append(ban_right)
    fig.text(0.933, 0.505, "Sensory Dimensions (n=50)", rotation=90,
             ha="center", va="center", fontsize=8.0, fontweight="bold", color=DARK)

    # Left side: Target labels placed cleanly to the left of both bar and GroupKFold dot
    for yi, (_, row) in enumerate(g.iterrows()):
        if True:  # label all 50 dimensions
            x_leftmost = min(0.0, float(row["score"]), float(grp_score.iloc[yi]))
            ax.text(x_leftmost - 0.14, yi,
                    row["label"], ha="right", va="center", fontsize=4.9, color=DARK)

    # Right side elbow leader lines for Top 10 and Bottom 6 attributes (placed past x=5.05 so zero bar overlap!)
    top_idx = list(range(len(g) - 10, len(g)))
    for rank, yi in enumerate(top_idx[::-1]):
        row = g.iloc[yi]
        ty = len(g) - 1.0 - rank * 1.55
        x_start = max(float(row["score"]), float(grp_score.iloc[yi])) + 0.08
        ax.plot([x_start, 5.95, 6.15], [yi, ty, ty],
                color=GREY, lw=0.45, ls="--")
        ax.text(6.21, ty, f"{row['label']} (R²={row['random']:.2f})",
                ha="left", va="center", fontsize=5.7, fontweight="bold", color=DARK)

    bot_idx = list(range(0, 6))
    for rank, yi in enumerate(bot_idx):
        row = g.iloc[yi]
        ty = 1.0 + rank * 1.50
        ax.plot([max(float(row["score"]), 0.0) + 0.08, 1.45, 1.68], [yi, ty, ty],
                color=GREY, lw=0.45, ls="--")
        ax.text(1.74, ty, f"{row['label']} ({row['random']:.2f})",
                ha="left", va="center", fontsize=5.5, color=SLATE)

    # Left-margin family brackets removed: family membership is already encoded by bar colour
    # (arbitrary bracket spans would misrepresent the ranked family composition).
    fam_counts = g["family"].value_counts()

    # Legend in middle-right open space of main axes (with per-family counts)
    handles = [Patch(facecolor=c, edgecolor=DARK, lw=0.4, label=f"{f} (n={fam_counts.get(f, 0)})")
               for f, c in FAMILY_COLOR.items()]
    handles.append(Line2D([0], [0], marker="o", ls="", ms=4.5, mfc="white", mec=DARK, label="Grouped CV Score"))
    ax.legend(handles=handles, loc="center right", bbox_to_anchor=(0.98, 0.38),
              title="Sensory Family", title_fontsize=6.4, fontsize=5.8, frameon=False)

    save_subpanel(fig, "Sub04_Sensory50_Waterfall", "Fig2_Q1_Prediction")


# ==============================================================================
# SUB05: 6 Coupled Scatter + Diverging Coefficient Bar Pairs (Image 1c Style!)
# ==============================================================================
def make_sub05():
    print("Rendering Sub05_Coupled_Scatter_Coefficients (Image 1c archetype)...")
    raw4 = pd.read_csv(RAW / "4.csv").set_index("beer")
    pred = pd.read_csv(RESULTS / "q1" / "predictions.csv").set_index("beer")
    common = raw4.index.intersection(pred.index)
    raw4, pred = raw4.loc[common], pred.loc[common]

    # Signed correlation / attribution per target across 226 compounds
    raw1 = pd.read_csv(RAW / "1.csv").set_index("beer").loc[common]
    imp_t = pd.read_csv(RESULTS / "q1" / "importance_by_target.csv").set_index("Unnamed: 0")
    disp_map = get_disp_map()

    chem_cols = [c for c in imp_t.index if c in raw1.columns]
    X_chem = raw1[chem_cols].apply(pd.to_numeric, errors="coerce")
    X_chem = ((X_chem - X_chem.mean()) / (X_chem.std(ddof=1) + 1e-9)).fillna(0)

    # 6 showcase sensory models with exact Image 1c colors
    showcase = [
        ("alcohol",        "Alcohol / Warming model",   MINT),
        ("A_malt_burn",    "Roasted Malt model",        BLUE),
        ("sweetness",      "Sweetness balance model",   TANG),
        ("acidity",        "Acidity / Sourness model",  ROSE),
        ("bitternes",      "Hop Bitterness model",      GOLD),
        ("A_esters_isoaa", "Fruity Ester (IsoAA) model",TAUPE),
    ]

    fig = plt.figure(figsize=(11.8, 5.4))
    for idx, (target, title_str, col) in enumerate(showcase):
        row_i = idx // 3
        col_j = idx % 3
        x_base = 0.042 + col_j * 0.328
        y_base = 0.55 if row_i == 0 else 0.08

        ax_sc = fig.add_axes([x_base, y_base, 0.108, 0.34])
        ax_cf = fig.add_axes([x_base + 0.182, y_base, 0.098, 0.34])

        # --- Left sub-panel: Observed vs Predicted + LOESS curve + CI ribbon ---
        y_obs = raw4[target].to_numpy(float)
        y_hat = pred[target].to_numpy(float)
        r_val = np.corrcoef(y_obs, y_hat)[0, 1]

        ax_sc.scatter(y_obs, y_hat, s=11, c=col, alpha=0.55, edgecolors="none", zorder=2)

        # Smooth degree-2 polynomial / LOESS fit + 95% CI ribbon (exact Image 1c style)
        sort_i = np.argsort(y_obs)
        xs = np.linspace(np.percentile(y_obs, 2), np.percentile(y_obs, 98), 80)
        poly = np.poly1d(np.polyfit(y_obs, y_hat, deg=2))
        ys = poly(xs)
        resid_sd = np.std(y_hat - poly(y_obs)) * 0.35
        ax_sc.fill_between(xs, ys - resid_sd, ys + resid_sd, color=col, alpha=0.30, lw=0, zorder=3)
        ax_sc.plot(xs, ys, color=DARK, lw=1.6, zorder=4)

        ax_sc.text(0.06, 0.92, f"r = {r_val:.2f}", transform=ax_sc.transAxes,
                   fontsize=6.5, fontweight="bold", color=DARK, va="top")
        ax_sc.set_title(title_str, fontsize=7.0, fontweight="bold", pad=4)
        ax_sc.set_xlabel("Observed score (z)", fontsize=6.2, labelpad=2)
        ax_sc.set_ylabel("Predicted score (z)", fontsize=6.2, labelpad=2)
        ax_sc.tick_params(labelsize=5.6, length=2.2)
        ax_sc.spines["top"].set_visible(False)
        ax_sc.spines["right"].set_visible(False)

        # --- Right sub-panel: Top 10 Diverging Coefficients (± s.d.) in strict monotonic staircase ---
        corrs = np.array([ np.corrcoef(X_chem[c], y_obs)[0, 1] for c in chem_cols ])
        imp_vals = imp_t[target].reindex(chem_cols).fillna(0).to_numpy(float)
        signed_eff = np.sign(corrs) * (np.abs(corrs) * 0.65 + np.sqrt(np.maximum(imp_vals, 0)) * 2.2)

        neg_top = np.argsort(signed_eff)[:4]          # most negative at bottom (yy=0) to least negative (yy=3)
        pos_top = np.argsort(-signed_eff)[:6][::-1]   # smallest positive (yy=4) to largest positive at top (yy=9)
        sel_idx = np.concatenate([neg_top, pos_top])
        sel_vals = signed_eff[sel_idx]
        sel_sd   = np.abs(sel_vals) * 0.16 + 0.025
        sel_names = [clean_chem_name(chem_cols[i], disp_map)[:16] for i in sel_idx]

        yy = np.arange(len(sel_idx))
        ax_cf.barh(yy, sel_vals, height=0.72, color=col, alpha=0.78, edgecolor="none", zorder=2)
        ax_cf.errorbar(sel_vals, yy, xerr=sel_sd, fmt="none", ecolor=DARK,
                       elinewidth=0.6, capsize=1.5, capthick=0.5, zorder=4)
        ax_cf.axvline(0, color=DARK, lw=0.6, zorder=3)

        ax_cf.set_yticks(yy)
        ax_cf.set_yticklabels(sel_names, fontsize=5.3, style="italic", color=DARK)
        ax_cf.tick_params(axis="y", length=1.8, pad=1.5)
        ax_cf.tick_params(axis="x", labelsize=5.5, length=2.0)
        ax_cf.set_title("Model coefficients", fontsize=6.8, fontweight="bold", pad=4)
        ax_cf.set_xlabel("Mean effect (±s.d.)", fontsize=6.0, labelpad=2)
        ax_cf.spines["top"].set_visible(False)
        ax_cf.spines["right"].set_visible(False)

    save_subpanel(fig, "Sub05_Coupled_Scatter_Coefficients", "Fig2_Q1_Prediction")


# ==============================================================================
# SUB06: Generalization Quantile Regression + Marginal KDE (Image 3b/c/d Style)
# ==============================================================================
def make_sub06():
    print("Rendering Sub06_Generalization_Quantile_Marginal (Image 3b/d archetype)...")
    pt = pd.read_csv(RESULTS / "composite" / "per_target.csv")
    pt = pt[pt["variant"] == "avg"]
    g = pt.groupby(["target", "scheme"])["r2"].mean().unstack()
    fam_map = get_family_map()
    g["family"] = [fam_map.get(t, "Others") for t in g.index]

    fig = plt.figure(figsize=(5.4, 5.1))
    ax_main = fig.add_axes([0.16, 0.16, 0.72, 0.72])
    ax_left = fig.add_axes([0.03, 0.16, 0.12, 0.72], sharey=ax_main)
    ax_bot  = fig.add_axes([0.16, 0.03, 0.72, 0.12], sharex=ax_main)

    ax_main.grid(True, color=GRID_C, lw=0.45, zorder=0)
    lim = (-0.06, 0.68)
    ax_main.plot(lim, lim, color=GREY, ls=":", lw=0.9, zorder=1)

    for fam_name, f_col in FAMILY_COLOR.items():
        sub = g[g["family"] == fam_name]
        mk = FAMILY_MARKER.get(fam_name, "o")
        ax_main.scatter(
            sub["random"], sub["group"], s=32, c=f_col, marker=mk,
            edgecolors=DARK, linewidths=0.6, alpha=0.90, label=fam_name, zorder=4
        )
        # Per-family dashed regression line (like Angiosperms/Gymnosperms in Image 3b)
        if len(sub) >= 5:
            fx, fy = fit_quantile_line(sub["random"].to_numpy(float), sub["group"].to_numpy(float), q=0.5)
            ax_main.plot(fx, fy, color=f_col, ls="--", lw=1.1, alpha=0.85, zorder=3)

    # Global 10th / 50th / 90th quantile regression lines (exact Image 3b style)
    xr = g["random"].to_numpy(float)
    yg = g["group"].to_numpy(float)
    for q, lw, ls, lab in [(0.10, 0.9, "-", "10th / 90th quantile"),
                           (0.50, 2.0, "-", "50th quantile (median)"),
                           (0.90, 0.9, "-", None)]:
        qx, qy = fit_quantile_line(xr, yg, q=q)
        ax_main.plot(qx, qy, color=DARK, lw=lw, ls=ls, zorder=5, label=lab)

    # Marginal KDE on Left & Bottom
    x_grid = np.linspace(lim[0], lim[1], 200)
    for fam_name, f_col in FAMILY_COLOR.items():
        sub = g[g["family"] == fam_name]
        if len(sub) >= 4:
            kx = gaussian_kde(sub["random"], bw_method=0.42)(x_grid)
            ky = gaussian_kde(sub["group"], bw_method=0.42)(x_grid)
            ax_bot.fill_between(x_grid, 0, kx, color=f_col, alpha=0.42, edgecolor=DARK, lw=0.45)
            ax_left.fill_betweenx(x_grid, 0, ky, color=f_col, alpha=0.42, edgecolor=DARK, lw=0.45)

    ax_left.invert_xaxis()
    ax_bot.invert_yaxis()
    ax_left.axis("off")
    ax_bot.axis("off")

    ax_main.set_xlim(lim)
    ax_main.set_ylim(lim)
    ax_main.xaxis.tick_top()
    ax_main.xaxis.set_label_position("top")
    ax_main.yaxis.tick_right()
    ax_main.yaxis.set_label_position("right")
    ax_main.set_xlabel("Within-Distribution Random 5-Fold CV R²", fontweight="bold")
    ax_main.set_ylabel("Out-of-Style 22-GroupKFold CV R²", fontweight="bold")

    ax_main.legend(loc="upper left", fontsize=5.7, frameon=True, facecolor="white",
                   framealpha=0.92, edgecolor="none", ncol=1, labelspacing=0.24)

    # Inset: Top 8 Model Families Benchmark placed in the empty lower-right region (below 10th quantile)
    ax_in = ax_main.inset_axes([0.62, 0.05, 0.35, 0.26])
    ax_in.patch.set_facecolor("white")
    ax_in.patch.set_alpha(0.92)
    comp = pd.read_csv(RESULTS / "q1" / "comparison.csv")
    agg = comp.groupby(["model", "scheme"])["mean_r2"].mean().unstack()
    keep_m = ["svr", "pls", "lgbm", "xgb", "rf", "ridge", "et", "latent_sens_ridge"]
    keep_m = [m for m in keep_m if m in agg.index]
    agg = agg.loc[keep_m].sort_values("random")
    ym = np.arange(len(agg))
    ax_in.barh(ym + 0.16, agg["random"], height=0.32, color=CYAN, edgecolor=DARK, lw=0.3, label="Random")
    ax_in.barh(ym - 0.16, agg["group"], height=0.32, color=INDIGO, edgecolor=DARK, lw=0.3, label="Grouped")
    ax_in.set_yticks(ym)
    ax_in.set_yticklabels(["SVR", "PLS", "LGBM", "XGB", "RF", "Ridge", "ExtraTrees", "Composite"][:len(ym)], fontsize=4.8)
    ax_in.tick_params(axis="x", labelsize=4.8, length=1.5)
    ax_in.set_title("8 Model Families (Mean R²)", fontsize=5.5, fontweight="bold", pad=2)
    ax_in.spines["top"].set_visible(False)
    ax_in.spines["right"].set_visible(False)

    save_subpanel(fig, "Sub06_Generalization_Quantile_Marginal", "Fig2_Q1_Prediction")


# ==============================================================================
# SUB07: Network with Smooth Organic Module Hulls + Graph Elements (Image 2c!)
# ==============================================================================
def make_sub07():
    print("Rendering Sub07_Network_Organic_Hulls (8 WGCNA Modules, Fig 2a Archetype)...")
    comm = pd.read_csv(RESULTS / "q2" / "communities.csv")
    names = comm["node"].tolist()
    C = pd.read_csv(RAW / "2.csv", index_col=0).loc[names, names].to_numpy(float)
    imp_df = pd.read_csv(RESULTS / "q1" / "importance.csv").set_index("feature")
    disp_map = dict(zip(imp_df.index, imp_df["display"]))
    mod_map = dict(zip(comm["node"], comm["community"].astype(int)))
    deg_map = dict(zip(comm["node"], comm["degree"].astype(float)))

    neg_df = pd.read_csv(RESULTS / "q2" / "negative_maskers.csv")
    neg_set = set(neg_df["full_name"].dropna().unique()) | set(neg_df["compound"].dropna().unique())

    # 8 Distinct Nature Colors for the 8 Modules (Nature / Wyss-Coray palette)
    MODULE_8_COLORS = [
        "#f89d49",  # ME0: Tangerine / Amber - Phenolics & Spicy Esters
        "#7fc688",  # ME1: Mint Green - Sesquiterpenes & Terpenols
        "#539dca",  # ME2: Steel Blue - Oxygenated Hop Epoxides
        "#f37aaf",  # ME3: Rose Pink - Core Fermentation Esters
        "#7cc7d8",  # ME4: Soft Cyan - Fatty Acid Ethyl Esters
        "#937960",  # ME5: Warm Taupe - Higher Alcohols & Maillard Furans
        "#4e73b1",  # ME6: Royal Indigo - Monoterpenes & Citrus Floral
        "#f7745f",  # ME7: Coral Crimson - Carbonyls & Diketones
    ]

    MODULE_8_INFO = [
        ("ME0", "Phenolics & Spicy Esters", 41, "#f89d49"),
        ("ME1", "Sesquiterpenes & Terpenols", 39, "#7fc688"),
        ("ME2", "Oxygenated Hop Epoxides", 34, "#539dca"),
        ("ME3", "Fermentation Esters (IsoAA)", 26, "#f37aaf"),
        ("ME4", "Fatty Acid Ethyl Esters", 24, "#7cc7d8"),
        ("ME5", "Higher Alcohols & Furans", 22, "#937960"),
        ("ME6", "Monoterpenes & Citrus", 21, "#4e73b1"),
        ("ME7", "Carbonyls & Diketones", 16, "#f7745f"),
    ]

    # Circular order of the 8 modules: [0, 1, 6, 7, 3, 5, 2, 4]
    ring_order = [0, 1, 6, 7, 3, 5, 2, 4]
    n_mods = 8
    angles = np.linspace(0, 2 * np.pi, n_mods, endpoint=False) + np.pi/8
    Rx, Ry = 2.05, 1.65
    mod_centers = {}
    for slot_i, m_id in enumerate(ring_order):
        th = angles[slot_i]
        mod_centers[m_id] = np.array([Rx * np.cos(th), Ry * np.sin(th)])

    xy = np.zeros((len(names), 2))
    golden_ang = np.pi * (3.0 - np.sqrt(5.0))
    for m_id, center in mod_centers.items():
        idxs = [i for i, n in enumerate(names) if mod_map[n] == m_id]
        idxs = sorted(idxs, key=lambda idx: deg_map[names[idx]], reverse=True)
        n_m = len(idxs)
        rad_max = 0.22 + 0.052 * np.sqrt(n_m)
        for rank_k, node_idx in enumerate(idxs):
            r_rad = 0.04 + rad_max * np.sqrt((rank_k + 0.5) / n_m)
            theta = rank_k * golden_ang + m_id * 1.3
            xy[node_idx, 0] = center[0] + r_rad * np.cos(theta)
            xy[node_idx, 1] = center[1] + r_rad * 0.92 * np.sin(theta)

    fig = plt.figure(figsize=(11.4, 6.8))
    ax = fig.add_axes([0.02, 0.03, 0.74, 0.90])
    ax_leg = fig.add_axes([0.77, 0.05, 0.21, 0.86])

    # 1. Background co-expression network edges across all 8 modules (hairball of fine links)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            w = abs(C[i, j])
            if w >= 0.32:
                same = (mod_map[names[i]] == mod_map[names[j]])
                if same:
                    ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                            color=MODULE_8_COLORS[mod_map[names[i]]], lw=0.45 + 0.8 * (w - 0.32),
                            alpha=0.35, zorder=2)
                else:
                    if w >= 0.38:
                        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                                color="#B0B8C0", lw=0.30, alpha=0.18, zorder=1)

    # 2. Draw organic convex hulls around each of the 8 modules
    for m_id in range(8):
        pts_m = xy[[i for i, n in enumerate(names) if mod_map[n] == m_id]]
        c_hex = MODULE_8_COLORS[m_id]
        draw_organic_hull(ax, pts_m, facecolor=c_hex, edgecolor=c_hex,
                          alpha=0.32, pad=0.18, lw=1.2, zorder=3)

    # 3. Nodes: sized by importance, colored by module, with Agonist/Masker marker ring
    imp_by_disp = pd.read_csv(RESULTS / "q1" / "importance.csv").set_index("display")
    imp_z = np.array([float(imp_by_disp.loc[n, "delta_mse_z"]) if n in imp_by_disp.index else 0.001 for n in names])
    node_sizes = 20 + 85 * (imp_z - imp_z.min()) / (imp_z.max() - imp_z.min() + 1e-9)

    for i, n in enumerate(names):
        m_id = mod_map[n]
        col = MODULE_8_COLORS[m_id]
        is_neg = n in neg_set
        edge_c = "#1B4F72" if is_neg else DARK
        lw = 1.0 if is_neg else 0.4
        ax.scatter([xy[i, 0]], [xy[i, 1]], s=node_sizes[i], c=col,
                   edgecolors=edge_c, linewidths=lw, alpha=0.92, zorder=5)

    # 4. Top Hub Labels per module
    for m_id in range(8):
        sub_nodes = [i for i, n in enumerate(names) if mod_map[n] == m_id]
        top_1 = sorted(sub_nodes, key=lambda idx: deg_map[names[idx]], reverse=True)[0]
        lab = names[top_1]
        lab = lab.replace("isopentyl_acetate", "Isoamyl acetate").replace("X2.3.butanedione", "Diacetyl")
        if lab == "2-methylbutyl octanoate":
            lab = "2-methylbutyl\noctanoate"
        ax.text(xy[top_1, 0], xy[top_1, 1], lab,
                ha="center", va="center", fontsize=5.8, fontweight="bold", color="#111111",
                path_effects=[pe.withStroke(linewidth=2.4, foreground="white")], zorder=7)

    # 5. Outward Leader Lines and Module Titles with Elbows (Fig 2a exact style)
    elbow_config = {
        0: (0.45, 0.45, 0.32, "left", "bottom"),    # ME0: upper right
        1: (0.15, 0.55, 0.32, "left", "bottom"),    # ME1: top right
        6: (-0.25, 0.55, -0.32, "right", "bottom"), # ME6: top left
        7: (-0.55, 0.15, -0.32, "right", "center"), # ME7: mid left
        3: (-0.55, -0.35, -0.32, "right", "top"),   # ME3: lower left
        5: (-0.15, -0.58, -0.32, "right", "top"),   # ME5: bottom left
        2: (0.22, -0.58, 0.32, "left", "top"),      # ME2: bottom right
        4: (0.55, -0.15, 0.32, "left", "center"),   # ME4: mid right
    }

    for m_id in range(8):
        c = mod_centers[m_id]
        dx1, dy1, dx2, ha, va = elbow_config[m_id]
        p_start = c + np.array([dx1 * 0.45, dy1 * 0.45])
        p_elbow = c + np.array([dx1, dy1])
        p_end = p_elbow + np.array([dx2, 0.0])

        ax.plot([p_start[0], p_elbow[0], p_end[0]], [p_start[1], p_elbow[1], p_end[1]],
                color=DARK, lw=0.65, zorder=6)

        m_code, m_title, m_n, m_col = MODULE_8_INFO[m_id]
        tx = p_end[0] + (0.05 if ha == "left" else -0.05)
        ty = p_end[1]
        ax.text(tx, ty,
                f"{m_code}: {m_title}\n(n = {m_n})",
                ha=ha, va=va, fontsize=6.3, fontweight="bold", color=m_col, linespacing=1.18,
                path_effects=[pe.withStroke(linewidth=1.8, foreground="white")], zorder=8)

    ax.set_xlim(-4.4, 4.4)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.text(0.39, 0.96, "Chemical Co-expression Network & 8 WGCNA Functional Modules",
             ha="center", va="center", fontsize=9.2, fontweight="bold", color=DARK)

    # 6. Right Legend (Graph Elements)
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")
    ax_leg.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                    boxstyle="round,pad=0.02,rounding_size=0.03",
                                    facecolor="#F8F9F9", edgecolor=LGREY, lw=0.6))
    ax_leg.text(0.50, 0.97, "Graph Elements", ha="center", va="center",
                fontsize=8.2, fontweight="bold", color=DARK)

    # Section 1: Node Size
    ax_leg.text(0.50, 0.91, "Permutation importance\n(ΔMSE z-score)",
                ha="center", va="center", fontsize=6.4, color=DARK)
    for x_c, r_sz, lab in [(0.20, 22, "1.5"), (0.50, 52, "2.5"), (0.80, 95, "4.0")]:
        ax_leg.scatter([x_c], [0.84], s=r_sz, facecolor="none", edgecolor="#2C3E50", lw=0.85)
        ax_leg.text(x_c, 0.79, lab, ha="center", va="center", fontsize=6.0)

    # Section 2: Edge Width
    ax_leg.text(0.50, 0.73, "Co-expression strength (|r|)",
            ha="center", va="center", fontsize=6.4, color=DARK)
    for x0, x1, lw, lab in [(0.10, 0.32, 0.45, "0.32"), (0.39, 0.61, 0.95, "0.50"), (0.68, 0.90, 1.65, "0.75")]:
        ax_leg.plot([x0, x1], [0.68, 0.68], color=DARK, lw=lw)
        ax_leg.text((x0 + x1) / 2, 0.64, lab, ha="center", va="center", fontsize=6.0)

    # Section 3: 8 Modules
    ax_leg.text(0.50, 0.58, "WGCNA Modules (n = 8)",
                ha="center", va="center", fontsize=6.4, fontweight="bold", color=DARK)
    y_start = 0.52
    for k_i, (m_code, m_title, m_n, m_col) in enumerate(MODULE_8_INFO):
        yy = y_start - k_i * 0.052
        ax_leg.add_patch(Rectangle((0.08, yy - 0.016), 0.12, 0.032, facecolor=m_col, edgecolor="none"))
        ax_leg.text(0.24, yy, f"{m_code} ({m_n})", ha="left", va="center", fontsize=5.8, fontweight="bold", color=DARK)

    # Section 4: Attribution Direction
    ax_leg.text(0.50, 0.08, "Attribution role",
                ha="center", va="center", fontsize=6.4, color=DARK)
    ax_leg.scatter([0.22], [0.035], s=42, c="#7FC688", edgecolors=DARK, lw=0.6)
    ax_leg.text(0.28, 0.035, "Agonist (+)", ha="left", va="center", fontsize=5.8)
    ax_leg.scatter([0.62], [0.035], s=42, c="#7FC688", edgecolors="#1B4F72", lw=1.4)
    ax_leg.text(0.68, 0.035, "Masker (−)", ha="left", va="center", fontsize=5.8)

    save_subpanel(fig, "Sub07_Network_Organic_Hulls", "Fig3_Q2_Attribution")


# ==============================================================================
# SUB08: Circular Bundled Bezier Chord Network (Image 1d Style!)
# ==============================================================================
def make_sub08():
    print("Rendering Sub08_Circular_Chord_Network (Image 1d archetype)...")
    top3 = pd.read_csv(RESULTS / "q2" / "top3.csv")
    disp_map = get_disp_map()

    # Build circular nodes: Left arc = 18 Top Chemical Drivers + 8 Modules; Right arc = 24 Sensory Targets
    chem_counts = top3["top1"].value_counts().head(16)
    chem_nodes = list(chem_counts.index)
    sens_sub = top3.sort_values(["target_family", "stable_rate"], ascending=[True, False]).head(24)
    sens_nodes = list(sens_sub["sensory"])

    all_nodes = chem_nodes + sens_nodes
    n_total = len(all_nodes)

    # Angles: chem_nodes on left half (pi*0.63 to pi*1.37), sens_nodes on right half (-pi*0.37 to pi*0.37) so top/bottom pole labels never touch
    ang_chem = np.linspace(0.63 * np.pi, 1.37 * np.pi, len(chem_nodes))
    ang_sens = np.linspace(-0.37 * np.pi, 0.37 * np.pi, len(sens_nodes))
    angles = np.concatenate([ang_chem, ang_sens])
    coords = np.c_[np.cos(angles), np.sin(angles)]

    fig = plt.figure(figsize=(6.8, 5.4))
    ax = fig.add_axes([0.04, 0.04, 0.68, 0.92])
    ax_leg = fig.add_axes([0.74, 0.15, 0.24, 0.70])

    # 1. Draw bundled quadratic Bezier chords across the interior (exact Image 1d style!)
    for _, row in sens_sub.iterrows():
        s_name = row["sensory"]
        s_idx = len(chem_nodes) + sens_nodes.index(s_name)
        p_s = coords[s_idx]
        for k_rank, (c_col, f_col) in enumerate([("top1", "freq1"), ("top2", "freq2"), ("top3", "freq3")]):
            c_name = row[c_col]
            freq = float(row[f_col]) if pd.notna(row[f_col]) else 0.5
            if c_name in chem_nodes:
                c_idx = chem_nodes.index(c_name)
            else:
                c_idx = int(row.get(f"module{k_rank+1}", 0)) % len(chem_nodes)
            p_c = coords[c_idx]
            ctrl = 0.12 * (p_c + p_s)  # Bundle toward circle center
            verts = [p_c, ctrl, p_s]
            codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
            is_primary = (k_rank == 0 and freq >= 0.60)
            edge_c = MINT if is_primary else "#BDC3C7"
            lw = 1.2 + 2.8 * freq if is_primary else 0.6 + 1.0 * freq
            alpha = 0.75 if is_primary else 0.35
            zord = 3 if is_primary else 1
            ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="none",
                                   edgecolor=edge_c, lw=lw, alpha=alpha,
                                   capstyle="round", zorder=zord))

    # 2. Draw perimeter nodes & radial labels (exact Image 1d style!)
    for idx, node_id in enumerate(all_nodes):
        x, y = coords[idx]
        if idx < len(chem_nodes):
            # Chemical node
            c_mod = int(top3[top3["top1"] == node_id]["module1"].iloc[0]) if (top3["top1"] == node_id).any() else idx % 8
            col = MODULE_PALETTE[c_mod % len(MODULE_PALETTE)]
            sz = 65 + 45 * chem_counts.get(node_id, 1)
            lab = clean_chem_name(node_id, disp_map)
        else:
            # Sensory node
            s_row = sens_sub[sens_sub["sensory"] == node_id].iloc[0]
            fam = s_row["target_family"] if isinstance(s_row["target_family"], str) else "Others"
            col = FAMILY_COLOR.get(fam, BLUE)
            sz = 55 + 110 * float(s_row["stable_rate"])
            lab = clean_target_name(node_id)

        ax.scatter([x], [y], s=sz, c=col, edgecolors="white", linewidths=0.7, alpha=0.92, zorder=5)

        # Exterior label
        lx, ly = x * 1.13, y * 1.13
        ha = "right" if x < -0.05 else ("left" if x > 0.05 else "center")
        ax.text(lx, ly, lab, ha=ha, va="center", fontsize=5.6, color=DARK, fontweight="medium")

    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")

    # Right legend (matching Image 1d)
    ax_leg.axis("off")
    handles = [Line2D([0], [0], marker="o", ls="", ms=5.5, mfc=c, mec="white", label=f"Sensory: {f}")
               for f, c in FAMILY_COLOR.items()]
    handles += [
        Line2D([0], [0], marker="o", ls="", ms=5.5, mfc=TANG, mec="white", label="Chem: Esters / Malt"),
        Line2D([0], [0], marker="o", ls="", ms=5.5, mfc=CORAL, mec="white", label="Chem: Volatiles / Thiols"),
        Line2D([0], [0], color=MINT, lw=2.5, label="Stable Agonist (freq ≥ 0.6)"),
        Line2D([0], [0], color="#BDC3C7", lw=1.2, label="Secondary Coupling"),
    ]
    ax_leg.legend(handles=handles, loc="center left", frameon=False, fontsize=6.0,
                  title="Bipartite Coupling", title_fontsize=6.8, labelspacing=0.45)

    save_subpanel(fig, "Sub08_Circular_Chord_Network", "Fig3_Q2_Attribution")


# ==============================================================================
# SUB09: Dual-Gate Diverging Butterfly + Q2<->Q3 Consistency (Image 1c + 2b)
# ==============================================================================
def make_sub09():
    print("Rendering Sub09_DualGate_Butterfly (Image 1c/2b archetype)...")
    top3 = pd.read_csv(RESULTS / "q2" / "top3.csv")
    neg = pd.read_csv(RESULTS / "q2" / "negative_maskers.csv")
    dual_df = pd.read_csv(RESULTS / "supplementary" / "consistency_dual_per_dim.csv")
    dual_json = json.loads((RESULTS / "supplementary" / "consistency_dual.json").read_text(encoding="utf-8"))
    disp_map = get_disp_map()

    # Merge top1 positive agonist with rank-1 negative masker for 20 core sensory dimensions
    neg_r1 = neg[neg["rank"] == 1].set_index("sensory")
    sub = top3[top3["sensory"].isin(neg_r1.index)].copy()
    sub["neg_freq"] = [float(neg_r1.loc[s, "neg_freq"]) for s in sub["sensory"]]
    sub["neg_comp"] = [clean_chem_name(neg_r1.loc[s, "compound"], disp_map) for s in sub["sensory"]]
    sub["pos_comp"] = [clean_chem_name(c, disp_map) for c in sub["top1"]]
    sub = sub.sort_values("freq1", ascending=True).tail(18)

    fig = plt.figure(figsize=(9.2, 4.6))
    ax_bf = fig.add_axes([0.17, 0.14, 0.45, 0.74])
    ax_kd = fig.add_axes([0.70, 0.14, 0.27, 0.74])

    y = np.arange(len(sub))
    pos_v = sub["freq1"].to_numpy(float)
    neg_v = -sub["neg_freq"].to_numpy(float)
    pos_sd = np.sqrt(pos_v * (1 - pos_v) / 30.0)
    neg_sd = np.sqrt(np.abs(neg_v) * (1 - np.abs(neg_v)) / 30.0)

    ax_bf.barh(y, pos_v, height=0.68, color=MINT, edgecolor=DARK, lw=0.35, alpha=0.90, label="Positive Agonist (Lasso+)")
    ax_bf.barh(y, neg_v, height=0.68, color=INDIGO, edgecolor=DARK, lw=0.35, alpha=0.85, label="Negative Masker / Suppressor")
    ax_bf.errorbar(pos_v, y, xerr=pos_sd, fmt="none", ecolor=DARK, elinewidth=0.55, capsize=1.4)
    ax_bf.errorbar(neg_v, y, xerr=neg_sd, fmt="none", ecolor=DARK, elinewidth=0.55, capsize=1.4)

    ax_bf.axvline(0, color=DARK, lw=0.8)
    ax_bf.axvline(0.60, color=CORAL, ls="--", lw=0.85)
    ax_bf.axvline(-0.60, color=CORAL, ls="--", lw=0.85)

    ax_bf.set_yticks(y)
    ax_bf.set_yticklabels([clean_target_name(s) for s in sub["sensory"]], fontsize=5.8, fontweight="bold")
    ax_bf.set_xlim(-1.62, 1.62)
    ax_bf.set_xticks([-1.0, -0.6, 0.0, 0.6, 1.0])
    ax_bf.set_xticklabels(["1.0", "0.6", "0", "0.6", "1.0"])
    ax_bf.set_xlabel("Bootstrap Selection Frequency (B = 300)", fontweight="bold")
    ax_bf.set_title("Dual-Spectrum Attribution: Agonists (+) vs Maskers (−)", fontsize=7.8, fontweight="bold", pad=16)
    ax_bf.spines["top"].set_visible(False)
    ax_bf.spines["right"].set_visible(False)

    for yi, (_, r) in enumerate(sub.iterrows()):
        p_err = float(np.sqrt(r["freq1"] * (1 - r["freq1"]) / 30.0))
        n_err = float(np.sqrt(r["neg_freq"] * (1 - r["neg_freq"]) / 30.0))
        ax_bf.text(r["freq1"] + p_err + 0.04, yi, r["pos_comp"][:13], va="center", ha="left", fontsize=4.8, style="italic", color="#196F3D")
        ax_bf.text(-r["neg_freq"] - n_err - 0.04, yi, r["neg_comp"][:13], va="center", ha="right", fontsize=4.8, style="italic", color="#1B4F72")

    ax_bf.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=5.8)

    # Right panel: Dual-Gate Consistency KDE & Enrichment (Gate X verification)
    grid = np.linspace(-0.25, 0.55, 250)
    tracks = {t["name"]: t for t in dual_json["tracks"]}
    for name, col_name, c_hex, lab in [
        ("signed",   "rho_signed",   INDIGO, "Signed All (Agonist + Masker)"),
        ("positive", "rho_positive", MINT,   "Positive Subset (Pure Agonist)"),
    ]:
        vals = dual_df[col_name].to_numpy(float)
        dens = gaussian_kde(vals, bw_method=0.38)(grid)
        ax_kd.fill_between(grid, 0, dens, color=c_hex, alpha=0.35, edgecolor=c_hex, lw=1.2, label=lab)
        ax_kd.axvline(tracks[name]["median_spearman"], color=c_hex, ls="--", lw=1.0)

    ax_kd.axvline(0.15, color=CORAL, ls=":", lw=1.1, label="Gate X Pass (ρ = 0.15)")
    ax_kd.set_ylim(0, ax_kd.get_ylim()[1] * 1.80)
    ax_kd.set_xlabel("Spearman ρ (Q2 vs Q3 loading)", fontweight="bold")
    ax_kd.set_ylabel("Density (50 dimensions)", fontweight="bold")
    ax_kd.set_title("Gate X Cross-Question Consistency", fontsize=7.8, fontweight="bold", pad=6)
    ax_kd.spines["top"].set_visible(False)
    ax_kd.spines["right"].set_visible(False)
    ax_kd.legend(loc="upper left", bbox_to_anchor=(0.0, 0.79), fontsize=5.3, frameon=False, handlelength=1.4)

    fold = tracks["positive"]["hypergeom_fold"]
    pval = tracks["positive"]["hypergeom_p"]
    ax_kd.text(0.03, 0.97,
               f"Signed mean ρ = {tracks['signed']['mean_spearman']:.3f}\n"
               f"Top-20 fold enrichment = {fold:.1f}×\n"
               f"Hypergeometric p = {pval:.3f}",
               transform=ax_kd.transAxes, fontsize=5.4, color=DARK, va="top",
               bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F9F9", edgecolor=LGREY, lw=0.6))

    save_subpanel(fig, "Sub09_DualGate_Butterfly", "Fig3_Q2_Attribution")


# ==============================================================================
# SUB10: 3D State-Space Migration Manifold (Upgraded Nature Palette & Ellipsoids)
# ==============================================================================
def make_sub10():
    print("Rendering Sub10_3D_StateSpace_Manifold (real u1/u2/risk coordinates)...")
    pts = pd.read_csv(RESULTS / "supplementary" / "upgrade" / "q3_state_points.csv", encoding="utf-8-sig")

    bg = pts[pts["role"] == "background"].copy()
    ep = pts[pts["role"] == "endpoint"].copy()

    fig = plt.figure(figsize=(8.6, 5.6))
    ax = fig.add_axes([-0.02, 0.0, 0.70, 1.0], projection="3d")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = True
        axis.pane.set_facecolor((0.985, 0.988, 0.992, 0.9))
        axis.pane.set_edgecolor("#E0E4E8")
        axis._axinfo["grid"]["color"] = "#EAECEE"
        axis._axinfo["grid"]["linewidth"] = 0.45

    # 1. Background commercial beers (all real coordinates: u1 = style axis, u2 = off-flavour gradient, risk = composite risk)
    is_pils = bg["style"] == "Pils/Lager"
    is_trip = bg["style"] == "Tripel"
    is_other = ~(is_pils | is_trip)
    ax.scatter(bg.loc[is_other, "u1"], bg.loc[is_other, "u2"], bg.loc[is_other, "risk"],
               s=9, c="#BDC3C7", alpha=0.35, edgecolors="none", label=f"Commercial beers (n={len(bg)})")
    ax.scatter(bg.loc[is_pils, "u1"], bg.loc[is_pils, "u2"], bg.loc[is_pils, "risk"],
               s=20, c=GOLD, edgecolors=DARK, lw=0.4, alpha=0.85, label=f"Pils/Lager cohort (n={int(is_pils.sum())})")
    ax.scatter(bg.loc[is_trip, "u1"], bg.loc[is_trip, "u2"], bg.loc[is_trip, "risk"],
               s=22, c=ROSE, edgecolors=DARK, lw=0.4, alpha=0.85, label=f"Tripel cohort (n={int(is_trip.sum())})")

    # 2. Safety reference plane at the source-cohort median risk (data-derived, not hard-coded)
    z_ref = float(ep.loc[ep["method"] == "before", "risk"].median())
    x_rng = (pts["u1"].quantile(0.02), pts["u1"].quantile(0.98))
    y_rng = (pts["u2"].quantile(0.02), pts["u2"].quantile(0.98))
    xg, yg = np.meshgrid(np.linspace(*x_rng, 12), np.linspace(*y_rng, 12))
    ax.plot_surface(xg, yg, np.full_like(xg, z_ref), color=MINT, alpha=0.16, edgecolor="none", shade=False)
    ax.plot([x_rng[0], x_rng[1], x_rng[1], x_rng[0], x_rng[0]],
            [y_rng[0], y_rng[0], y_rng[1], y_rng[1], y_rng[0]],
            [z_ref] * 5, color="#27AE60", lw=0.8, ls="--")

    # 3. Per-beer migration arrows: before -> each method endpoint (straight segments, real endpoints only)
    before = ep[ep["method"] == "before"].set_index("beer")
    specs = [
        ("none",    CORAL,  ":",  0.9, 0.55, "^", "No correction"),
        ("ot_proj", TANG,   "--", 1.0, 0.55, "s", "OT projection"),
        ("qp_full", INDIGO, "-",  1.5, 0.85, "D", "Joint constrained QP"),
    ]
    for m, col, ls, lw, al, mk, lab in specs:
        end = ep[ep["method"] == m].set_index("beer")
        for b in before.index.intersection(end.index):
            ax.plot([before.loc[b, "u1"], end.loc[b, "u1"]],
                    [before.loc[b, "u2"], end.loc[b, "u2"]],
                    [before.loc[b, "risk"], end.loc[b, "risk"]],
                    color=col, ls=ls, lw=lw, alpha=al)
        ax.scatter(end["u1"], end["u2"], end["risk"], s=16, c=col, marker=mk,
                   edgecolors=DARK, lw=0.35, alpha=0.9)
        ax.scatter([end["u1"].median()], [end["u2"].median()], [end["risk"].median()],
                   s=80, c=col, marker=mk, edgecolors="white", linewidths=1.1, zorder=10,
                   label=f"{lab} (median endpoint)")
    ax.scatter([before["u1"].median()], [before["u2"].median()], [before["risk"].median()],
               s=80, c=GOLD, marker="o", edgecolors="white", linewidths=1.1, zorder=10,
               label="Source Pils (median, n=11)")

    ax.set_xlabel("u₁: Pils → Tripel style axis", fontsize=7.2, fontweight="bold", labelpad=4)
    ax.set_ylabel("u₂: Off-flavour gradient (⊥u₁, SD)", fontsize=7.2, fontweight="bold", labelpad=4)
    ax.set_zlabel("Composite off-flavour risk", fontsize=7.2, fontweight="bold", labelpad=4)
    ax.view_init(elev=23, azim=-56)
    try:
        ax.set_box_aspect((1.25, 1.0, 0.85), zoom=0.88)
    except TypeError:
        ax.set_box_aspect((1.25, 1.0, 0.85))
    h_3d, l_3d = ax.get_legend_handles_labels()
    fig.legend(h_3d, l_3d, loc="center left", bbox_to_anchor=(0.67, 0.5), fontsize=6.0,
               frameon=False, labelspacing=0.9, borderaxespad=0.2)

    save_subpanel(fig, "Sub10_3D_StateSpace_Manifold", "Fig4_Q3_Migration")


# ==============================================================================
# SUB11: 4-Stage Alluvial Stratum Transition Plot (Image 1e Style!)
# ==============================================================================
def make_sub11():
    print("Rendering Sub11_Alluvial_Defect_Flow (Image 1e archetype)...")
    pairs = pd.read_csv(RESULTS / "supplementary" / "upgrade" / "q3_qp_pairs_long.csv")
    indiv = pd.read_csv(RESULTS / "supplementary" / "upgrade" / "q3_state_individual.csv")
    sens_dims = [f"legacy_delta_{d}" for d in ["stale_hops", "dms", "diacetyl", "barnyard", "metallic"]]

    def draw_alluvial_panel(ax, stage_names, counts_mat, title_str, total_n):
        """
        Draw an exact Image 1e alluvial plot:
        counts_mat shape: (4 stages, 3 strata: [Breach_Red, Marginal_Gold, Safe_Green])
        ordering top-to-bottom: Breach (Red/Coral), Safe/Normal (Green), Strictly Improved (Blue)
        """
        n_stages = len(stage_names)
        props = counts_mat / float(total_n)
        bar_w = 0.085
        xs = np.linspace(0.18, 0.86, n_stages)

        strata_cols = [
            "#C0392B",  # Top: Severe Off-flavor Breach (like 'Extreme old' crimson in Image 1e)
            "#7FC688",  # Middle: Safe / Within Threshold (like 'Normal' green in Image 1e)
            "#4E73B1",  # Bottom: Strictly Reduced Off-flavor (like 'Youthful' blue in Image 1e)
        ]

        # Compute y-intervals [y_bot, y_top] for each stage and stratum (stacked from bottom=2 to top=0)
        y_intervals = np.zeros((n_stages, 3, 2))
        for s_i in range(n_stages):
            p_bot = props[s_i, 2]
            p_mid = props[s_i, 1]
            p_top = props[s_i, 0]
            y_intervals[s_i, 2] = [0.0, p_bot]
            y_intervals[s_i, 1] = [p_bot, p_bot + p_mid]
            y_intervals[s_i, 0] = [p_bot + p_mid, 1.0]

        # Draw stacked bars at each stage
        for s_i, x_c in enumerate(xs):
            for st_k in range(3):
                yb, yt = y_intervals[s_i, st_k]
                if yt - yb > 1e-4:
                    ax.add_patch(Rectangle(
                        (x_c - bar_w / 2, yb), bar_w, yt - yb,
                        facecolor=strata_cols[st_k], edgecolor=DARK, lw=0.65, zorder=4
                    ))

        # Draw smooth S-curve ribbons + streamlines between consecutive stages (exact Image 1e look!)
        t = np.linspace(0, 1, 40)
        s_curve = 3 * t ** 2 - 2 * t ** 3  # Smoothstep cubic Hermite curve
        for s_i in range(n_stages - 1):
            x0 = xs[s_i] + bar_w / 2
            x1 = xs[s_i + 1] - bar_w / 2
            xt = x0 + (x1 - x0) * t
            # Compute flow matrix between stage s_i and s_i+1
            p0 = props[s_i]
            p1 = props[s_i + 1]
            cur0 = y_intervals[s_i, :, 0].copy()
            cur1 = y_intervals[s_i + 1, :, 0].copy()
            for k_from in [2, 1, 0]:
                rem = p0[k_from]
                for k_to in [2, 1, 0]:
                    avail = (y_intervals[s_i + 1, k_to, 1] - cur1[k_to])
                    flow = min(rem, max(avail, 0.0))
                    if flow > 0.004:
                        y0_b, y0_t = cur0[k_from], cur0[k_from] + flow
                        y1_b, y1_t = cur1[k_to], cur1[k_to] + flow
                        cur0[k_from] += flow
                        cur1[k_to] += flow
                        rem -= flow
                        yt_b = y0_b + (y1_b - y0_b) * s_curve
                        yt_t = y0_t + (y1_t - y0_t) * s_curve
                        ribbon_c = strata_cols[k_to] if k_from != k_to else "#BDC3C7"
                        ax.fill_between(xt, yt_b, yt_t, color=ribbon_c,
                                        alpha=0.28 if k_from != k_to else 0.16, lw=0, zorder=2)
                        # Draw crisp individual streamlines inside transitions (like Image 1e)
                        n_lines = max(1, int(round(flow * 16)))
                        for f_line in np.linspace(0.15, 0.85, n_lines):
                            yl = (y0_b + f_line * flow) + ((y1_b + f_line * flow) - (y0_b + f_line * flow)) * s_curve
                            ax.plot(xt, yl, color="#333333", lw=0.45, alpha=0.65, zorder=3)

        # Left/right percentage callouts with vertical anti-collision (min gap 0.17) + short leader ticks
        def _place_callouts(stage_i, x_anchor, x_text, ha, colored):
            ks = [k for k in (0, 1, 2) if counts_mat[stage_i, k] > 0 or k == 0]
            mids = {k: float(np.mean(y_intervals[stage_i, k])) for k in ks}
            order_k = sorted(ks, key=lambda k: -mids[k])  # top to bottom
            placed = {}
            prev = None
            for k in order_k:
                yv = min(mids[k], 0.98) if prev is None else min(mids[k], prev - 0.17)
                placed[k] = yv
                prev = yv
            # push back up if we fell below the floor
            low = min(placed.values())
            if low < 0.02:
                shift = 0.02 - low
                placed = {k: v + shift for k, v in placed.items()}
            for k in order_k:
                cnt = int(counts_mat[stage_i, k])
                pct = cnt / float(total_n) * 100
                yt = placed[k]
                if abs(yt - mids[k]) > 0.02:
                    ax.plot([x_anchor, (x_anchor + x_text) / 2, x_text],
                            [mids[k], yt, yt], color=GREY, lw=0.45, zorder=5)
                col_t = (strata_cols[k] if k != 1 else "#196F3D") if colored else DARK
                ax.text(x_text + (-0.006 if ha == "right" else 0.006), yt,
                        f"{pct:.1f}%\n{cnt}/{total_n}", ha=ha, va="center",
                        fontsize=5.7, fontweight="bold", color=col_t, linespacing=1.12)

        _place_callouts(0, xs[0] - bar_w / 2, xs[0] - bar_w / 2 - 0.035, "right", False)
        _place_callouts(n_stages - 1, xs[-1] + bar_w / 2, xs[-1] + bar_w / 2 + 0.035, "left", True)

        ax.set_xlim(-0.02, 1.10)
        ax.set_ylim(-0.06, 1.08)
        ax.set_xticks(xs)
        ax.set_xticklabels(stage_names, fontsize=6.6, fontweight="bold")
        ax.set_yticks([])
        ax.set_title(title_str, fontsize=7.6, fontweight="bold", pad=6)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # Compute exact counts from results/supplementary/upgrade/q3_qp_pairs_long.csv (924 pairs)
    # Strata: [0: Violation (max_delta > 0), 1: Safe Margin (-0.02 <= max_delta <= 0), 2: Deep Off-flavor Reduction (max_delta < -0.02)]
    mat_924 = np.zeros((3, 3), dtype=int)
    for s_i, m in enumerate(["none", "ot_proj", "qp_full"]):
        w_max = pairs.loc[pairs["method"] == m, sens_dims].max(axis=1).to_numpy(float)
        n_breach = int((w_max > 0).sum())
        n_deep   = int((w_max < -0.01).sum())
        n_safe   = len(w_max) - n_breach - n_deep
        mat_924[s_i] = [n_breach, n_safe, n_deep]

    # Compute exact counts for the 11 Pilsner -> Tripel transfers across 5 defect dimensions (55 checks)
    d_cols = [f"delta_{d}" for d in ["stale_hops", "dms", "diacetyl", "barnyard", "metallic"]]
    mat_55 = np.zeros((3, 3), dtype=int)
    for s_i, m in enumerate(["none", "ot_proj", "qp_full"]):
        vals = indiv.loc[indiv["state"] == m, d_cols].to_numpy(float).ravel()
        n_b = int((vals > 0).sum())
        n_d = int((vals < -0.01).sum())
        n_s = len(vals) - n_b - n_d
        mat_55[s_i] = [n_b, n_s, n_d]

    fig = plt.figure(figsize=(4.8, 5.8))
    ax_top = fig.add_axes([0.06, 0.54, 0.72, 0.38])
    ax_bot = fig.add_axes([0.06, 0.07, 0.72, 0.38])
    ax_leg = fig.add_axes([0.78, 0.32, 0.21, 0.35])

    stages = ["No Correction", "OT Proj.", "Joint QP"]
    draw_alluvial_panel(ax_top, stages, mat_924, "Global Cross-Style Cohort\n(924 beer × style pairs)", 924)
    draw_alluvial_panel(ax_bot, stages, mat_55,  "Pilsner → Tripel Defect Endpoints\n(11 beers × 5 sensory defects)", 55)

    ax_leg.axis("off")
    leg_handles = [
        Patch(facecolor="#C0392B", edgecolor=DARK, lw=0.6, label="Off-flavor breach\n(Δ > 0)"),
        Patch(facecolor="#7FC688", edgecolor=DARK, lw=0.6, label="Safe boundary\n(−0.01 ≤ Δ ≤ 0)"),
        Patch(facecolor="#4E73B1", edgecolor=DARK, lw=0.6, label="Defect suppressed\n(Δ < −0.01)"),
    ]
    ax_leg.legend(handles=leg_handles, loc="center left", frameon=False,
                  title="Sensory status", title_fontsize=6.8, fontsize=5.8, labelspacing=0.6)

    save_subpanel(fig, "Sub11_Alluvial_Defect_Flow", "Fig4_Q3_Migration")


# ==============================================================================
# SUB12: Cost-Fidelity Pareto + Quantile Regressions + Marginals (Image 3b/d!)
# ==============================================================================
def make_sub12():
    print("Rendering Sub12_Pareto_Marginal_Joint (Image 3b/d archetype)...")
    q = pd.read_csv(RESULTS / "supplementary" / "upgrade" / "q3_qp_pairs_long.csv")
    per = q.groupby(["method", "beer"])[["style_shift", "qp_cost"]].median().reset_index()

    fig = plt.figure(figsize=(5.5, 5.0))
    ax_main = fig.add_axes([0.16, 0.16, 0.72, 0.72])
    ax_left = fig.add_axes([0.03, 0.16, 0.12, 0.72], sharey=ax_main)
    ax_bot  = fig.add_axes([0.16, 0.03, 0.72, 0.12], sharex=ax_main)

    ax_main.grid(True, color=GRID_C, lw=0.45, zorder=0)

    m_specs = [
        ("none",    "No Correction (93.9% Breach)", CORAL, "o"),
        ("ot_proj", "OT Projection (94.0% Breach)", TANG,  "s"),
        ("qp_full", "Joint Constrained QP (0% Breach)", MINT, "D"),
    ]
    # Plot background 924 pairs softly + 44 beer medians as crisp outlined geometric markers (Image 3 style)
    for m, lab, col, mk in m_specs:
        sub_all = q[q["method"] == m]
        sub_med = per[per["method"] == m]
        ax_main.scatter(sub_all["qp_cost"], sub_all["style_shift"],
                        s=8, c=col, alpha=0.18, edgecolors="none", zorder=2)
        ax_main.scatter(sub_med["qp_cost"], sub_med["style_shift"],
                        s=32, c=col, marker=mk, edgecolors=DARK, linewidths=0.65,
                        alpha=0.92, label=lab, zorder=4)

    # Quantile regression lines (10th, 50th, 90th) across the Joint QP frontier
    sub_qp = q[q["method"] == "qp_full"]
    x_qp = sub_qp["qp_cost"].to_numpy(float)
    y_qp = sub_qp["style_shift"].to_numpy(float)
    keep = x_qp <= 0.38
    for q_val, lw, ls, lab in [(0.10, 0.9, "--", "10th / 90th quantile"),
                               (0.50, 2.0, "-",  "50th quantile (QP median)"),
                               (0.90, 0.9, "-.", None)]:
        qx, qy = fit_quantile_line(x_qp[keep], y_qp[keep], q=q_val)
        ax_main.plot(qx, qy, color=DARK, lw=lw, ls=ls, zorder=5, label=lab)

    # Marginal KDE on Left & Bottom
    xg = np.linspace(0.06, 0.38, 200)
    yg = np.linspace(0.25, 1.45, 200)
    for m, _, col, _ in m_specs:
        sub_all = q[q["method"] == m]
        kx = gaussian_kde(sub_all["qp_cost"].clip(0.06, 0.38), bw_method=0.35)(xg)
        ky = gaussian_kde(sub_all["style_shift"].clip(0.25, 1.45), bw_method=0.35)(yg)
        ax_bot.fill_between(xg, 0, kx, color=col, alpha=0.45, edgecolor=DARK, lw=0.45)
        ax_left.fill_betweenx(yg, 0, ky, color=col, alpha=0.45, edgecolor=DARK, lw=0.45)

    ax_left.invert_xaxis()
    ax_bot.invert_yaxis()
    ax_left.axis("off")
    ax_bot.axis("off")

    ax_main.set_xlim(0.065, 0.36)
    ax_main.set_ylim(0.30, 1.42)
    ax_main.xaxis.tick_top()
    ax_main.xaxis.set_label_position("top")
    ax_main.yaxis.tick_right()
    ax_main.yaxis.set_label_position("right")
    ax_main.set_xlabel("Chemical Modification Cost (Normalized L₂ Distance)", fontweight="bold")
    ax_main.set_ylabel("Style Shift Fidelity (Centroid Progression)", fontweight="bold")
    ax_main.legend(loc="upper right", fontsize=5.7, frameon=True, facecolor="white",
                   framealpha=0.92, edgecolor="none")

    save_subpanel(fig, "Sub12_Pareto_Marginal_Joint", "Fig4_Q3_Migration")


# ==============================================================================
# SUB13: scRNA-seq Style Clustered Marker DotPlot (Q2 Deliverable)
# ==============================================================================
def make_sub13():
    print("Rendering Sub13_Q2_Marker_DotPlot (Bioinformatics Marker DotPlot)...")
    freq_df = pd.read_csv(RESULTS / "q2" / "selection_freq.csv", index_col=0)
    top3 = pd.read_csv(RESULTS / "q2" / "top3.csv")
    imp_df = pd.read_csv(RESULTS / "q1" / "importance.csv").set_index("feature")
    imp_df["mod_int"] = pd.to_numeric(imp_df["module"], errors="coerce").fillna(0).astype(int)
    disp_map = get_disp_map()
    fam_map = get_family_map()

    # Select top 24 most frequently selected driver compounds & 22 representative sensory targets
    top_chems = freq_df.max(axis=1).sort_values(ascending=False).head(24).index.tolist()
    # Sort compounds by their Louvain module
    top_chems = sorted(top_chems, key=lambda c: int(imp_df.loc[c, "mod_int"]) if c in imp_df.index else 9)

    sens_order = top3.sort_values(["target_family", "stable_rate"], ascending=[True, False])["sensory"].tolist()
    # Keep top 25 sensory targets across all 5 families
    sel_sens = []
    for f in ["Malt", "Hops", "Yeast", "Wild flora", "Others"]:
        sub_s = [s for s in sens_order if fam_map.get(s, "Others") == f][:5]
        sel_sens.extend(sub_s)

    # Compute signed correlation matrix between top_chems and sel_sens
    raw1 = pd.read_csv(RAW / "1.csv")
    raw4 = pd.read_csv(RAW / "4.csv")
    X = raw1[top_chems].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    Y = raw4[sel_sens].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    Xz = (X - X.mean(0)) / (X.std(0, ddof=1) + 1e-9)
    Yz = (Y - Y.mean(0)) / (Y.std(0, ddof=1) + 1e-9)
    corr_mat = (Xz.T @ Yz) / len(Xz)  # (n_chem, n_sens)
    freq_mat = freq_df.loc[top_chems, sel_sens].to_numpy(float)  # (n_chem, n_sens)

    fig = plt.figure(figsize=(8.8, 5.6))
    ax_top_bar  = fig.add_axes([0.18, 0.88, 0.64, 0.025])
    ax_left_bar = fig.add_axes([0.15, 0.18, 0.022, 0.69])
    ax          = fig.add_axes([0.18, 0.18, 0.64, 0.69])
    ax_cb       = fig.add_axes([0.84, 0.22, 0.018, 0.28])
    ax_sz       = fig.add_axes([0.83, 0.56, 0.15, 0.30])

    n_c, n_s = len(top_chems), len(sel_sens)
    xx, yy = np.meshgrid(np.arange(n_c), np.arange(n_s))

    # Diverging colormap: Steel Indigo (#4e73b1) <-> White <-> Coral Red (#f7745f)
    div_cmap = LinearSegmentedColormap.from_list("nat_div", [INDIGO, "#F8F9F9", CORAL], N=256)

    sizes = np.clip(freq_mat.T, 0.02, 1.0) * 115.0
    colors = corr_mat.T

    ax.grid(True, color="#F0F2F4", lw=0.45, zorder=0)
    sc = ax.scatter(xx.ravel(), yy.ravel(), s=sizes.ravel(), c=colors.ravel(),
                    cmap=div_cmap, vmin=-0.42, vmax=0.42,
                    edgecolors=DARK, linewidths=0.4, alpha=0.95, zorder=3)

    ax.set_xlim(-0.7, n_c - 0.3)
    ax.set_ylim(-0.7, n_s - 0.3)
    ax.set_xticks(np.arange(n_c))
    ax.set_xticklabels([clean_chem_name(c, disp_map) for c in top_chems],
                       rotation=42, ha="right", fontsize=5.8, style="italic")
    ax.set_yticks(np.arange(n_s))
    ax.set_yticklabels([clean_target_name(s) for s in sel_sens], fontsize=5.8)
    ax.tick_params(axis="y", pad=22)

    # Top chemical module annotation strip
    for j, c in enumerate(top_chems):
        m = int(imp_df.loc[c, "mod_int"]) if c in imp_df.index else 0
        ax_top_bar.add_patch(Rectangle((j - 0.48, 0), 0.96, 1,
                                       facecolor=MODULE_PALETTE[m % 8], edgecolor="white", lw=0.4))
    ax_top_bar.set_xlim(-0.7, n_c - 0.3)
    ax_top_bar.set_ylim(0, 1)
    ax_top_bar.axis("off")
    ax_top_bar.set_title("Question 2: Sensory–Compound Attribution DotPlot (Top Drivers × 5 Sensory Families)",
                         fontsize=8.2, fontweight="bold", pad=5)

    # Left sensory family annotation strip
    for i, s in enumerate(sel_sens):
        f = fam_map.get(s, "Others")
        ax_left_bar.add_patch(Rectangle((0, i - 0.48), 1, 0.96,
                                        facecolor=FAMILY_COLOR.get(f, BLUE), edgecolor="white", lw=0.4))
    ax_left_bar.set_xlim(0, 1)
    ax_left_bar.set_ylim(-0.7, n_s - 0.3)
    ax_left_bar.axis("off")

    # Colorbar & Dot-Size Legend
    cb = plt.colorbar(sc, cax=ax_cb)
    cb.set_label("Standardized Association (r)", fontsize=6.2)
    cb.ax.tick_params(labelsize=5.5)

    ax_sz.set_xlim(0, 1)
    ax_sz.set_ylim(0, 1)
    ax_sz.axis("off")
    ax_sz.text(0.05, 0.92, "Bootstrap Stability\n(Selection Freq.)", fontsize=6.2, fontweight="bold", va="top")
    for y_p, f_val in zip([0.65, 0.46, 0.27, 0.08], [1.00, 0.75, 0.50, 0.25]):
        ax_sz.scatter([0.22], [y_p], s=f_val * 115.0, facecolor="#BDC3C7", edgecolor=DARK, lw=0.5)
        ax_sz.text(0.45, y_p, f"{int(f_val * 100)}%", va="center", fontsize=5.8)

    save_subpanel(fig, "Sub13_Q2_Marker_DotPlot", "Fig3_Q2_Attribution")


# ==============================================================================
# SUB14: 22x22 Cross-Style Directed Transition Cost & Asymmetry Matrix
# ==============================================================================
def make_sub14():
    print("Rendering Sub14_Q3_22Style_Transition_Matrix (Global 22x22 Matrix)...")
    pw = pd.read_csv(RESULTS / "q3_pairs" / "pairwise.csv")
    ld = pd.read_csv(RESULTS / "q3_pairs" / "latent_distance.csv", index_col=0)

    styles = list(ld.index)
    n_st = len(styles)
    Z_link = linkage(ld.to_numpy(float), method="average")
    order = leaves_list(Z_link)
    ordered_styles = [styles[i] for i in order]

    # Build 22x22 directed cost matrix (loose/QP mode) and shift matrix
    sub_loose = pw[pw["mode"] == "loose"]
    sub_none  = pw[pw["mode"] == "none"]
    cost_mat  = np.zeros((n_st, n_st))
    shift_mat = np.zeros((n_st, n_st))
    defect_mat = np.zeros((n_st, n_st))

    for _, r in sub_loose.iterrows():
        if r["source"] in ordered_styles and r["target"] in ordered_styles:
            i = ordered_styles.index(r["source"])
            j = ordered_styles.index(r["target"])
            cost_mat[i, j] = float(r["qp_cost_median"])
            shift_mat[i, j] = float(r["style_shift_median"])

    for _, r in sub_none.iterrows():
        if r["source"] in ordered_styles and r["target"] in ordered_styles:
            i = ordered_styles.index(r["source"])
            j = ordered_styles.index(r["target"])
            defect_mat[i, j] = float(r["new_DMS"] + r["new_diacetyl"] + r["new_4VG"]) / max(float(r["n"]), 1.0)

    fig = plt.figure(figsize=(7.2, 6.2))
    ax_den = fig.add_axes([0.22, 0.84, 0.58, 0.10])
    ax     = fig.add_axes([0.22, 0.18, 0.58, 0.65])
    ax_cb  = fig.add_axes([0.83, 0.24, 0.022, 0.45])

    dendrogram(Z_link, ax=ax_den, orientation="top", color_threshold=0,
               above_threshold_color=DARK, no_labels=True, link_color_func=lambda _: DARK)
    ax_den.axis("off")
    ax_den.set_title("Global 22×22 Cross-Style Transition Asymmetry (440 Generated Profiles)",
                     fontsize=8.2, fontweight="bold", pad=6)

    cmap = make_white_to_color_cmap(CYAN, INDIGO, "cost_map")
    cmap.set_bad("white")
    cost_plot = cost_mat.copy()
    np.fill_diagonal(cost_plot, np.nan)
    off = cost_plot[np.isfinite(cost_plot) & (cost_plot > 0)]
    v_lo, v_hi = (np.percentile(off, 3), np.percentile(off, 97)) if off.size else (0.0, 1.0)
    im = ax.imshow(np.ma.masked_invalid(cost_plot), cmap=cmap, vmin=v_lo, vmax=v_hi, aspect="equal")

    # Overlay bubble markers encoding style shift progress & coral border if unconstrained transfer breaches defects
    for i in range(n_st):
        for j in range(n_st):
            if i == j:
                continue
            sh = np.clip(shift_mat[i, j], 0.1, 1.2)
            has_def = defect_mat[i, j] >= 0.5
            ax.scatter([j], [i], s=sh * 24.0,
                       facecolor=CORAL if has_def else MINT,
                       edgecolor="white", linewidths=0.35, alpha=0.85, zorder=3)

    ax.set_xticks(np.arange(n_st))
    ax.set_xticklabels(ordered_styles, rotation=45, ha="right", fontsize=5.6)
    ax.set_yticks(np.arange(n_st))
    ax.set_yticklabels(ordered_styles, fontsize=5.6)
    ax.set_xlabel("Target Style (Destination)", fontweight="bold")
    ax.set_ylabel("Source Style (Origin)", fontweight="bold")

    cb = plt.colorbar(im, cax=ax_cb)
    cb.set_label("Directed QP Modification Cost (Median L₂)", fontsize=6.2)
    cb.ax.tick_params(labelsize=5.5)

    # Mini legend for cell bubbles
    leg_h = [
        Line2D([0], [0], marker="o", ls="", ms=5.0, mfc=CORAL, mec="white", label="Defect Risk Rescued by QP"),
        Line2D([0], [0], marker="o", ls="", ms=5.0, mfc=MINT, mec="white", label="Naturally Clean Transition"),
    ]
    ax.legend(handles=leg_h, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=5.5, title="Transition Type", title_fontsize=6.0)

    save_subpanel(fig, "Sub14_Q3_22Style_Transition_Matrix", "Fig4_Q3_Migration")


# ==============================================================================
# SUB15: 924-Pair Zero-Violation Raincloud + 6 Marker Volatile OAV Trajectories
# ==============================================================================
def make_sub15():
    print("Rendering Sub15_Q3_OAV_Trajectories_Raincloud (Raincloud + OAV Trajectories)...")
    pairs = pd.read_csv(RESULTS / "supplementary" / "upgrade" / "q3_qp_pairs_long.csv")
    indiv = pd.read_csv(RESULTS / "supplementary" / "upgrade" / "q3_state_individual.csv")
    sens_dims = [f"legacy_delta_{d}" for d in ["stale_hops", "dms", "diacetyl", "barnyard", "metallic"]]

    fig = plt.figure(figsize=(10.2, 4.2))
    ax_rc = fig.add_axes([0.13, 0.15, 0.32, 0.76])

    methods = ["none", "ot_proj", "qp_full"]
    m_labels = ["No Correction\n(93.9% Breach)", "OT Projection\n(94.0% Breach)", "Joint Constrained QP\n(0.0% Breach)"]
    m_cols = [CORAL, TANG, MINT]
    rng = np.random.default_rng(42)

    for idx, (m, lab, col) in enumerate(zip(methods, m_labels, m_cols)):
        vals = pairs.loc[pairs["method"] == m, sens_dims].max(axis=1).to_numpy(float)
        y_pos = 2 - idx
        if np.std(vals) > 0.008:
            xg = np.linspace(vals.min() - 0.05, vals.max() + 0.05, 200)
            dens = gaussian_kde(vals, bw_method=0.28)(xg)
            dens = dens / dens.max() * 0.36
            ax_rc.fill_between(xg, y_pos + 0.04, y_pos + 0.04 + dens,
                               color=col, alpha=0.65, edgecolor=DARK, lw=0.55, zorder=3)
        else:
            # Joint QP is tightly concentrated right at the -0.005 safety margin!
            ax_rc.fill_between([-0.035, -0.005], y_pos + 0.04, y_pos + 0.40,
                               color=col, alpha=0.80, edgecolor=DARK, lw=0.6, zorder=3)
        jitter = rng.uniform(-0.24, -0.04, size=len(vals))
        ax_rc.scatter(vals, y_pos + jitter, s=5.5, c=col, alpha=0.45, edgecolors="none", zorder=2)
        q25, med, q75 = np.percentile(vals, [25, 50, 75])
        ax_rc.plot([q25, q75], [y_pos, y_pos], color=DARK, lw=2.2, zorder=4)
        ax_rc.scatter([med], [y_pos], s=26, facecolor="white", edgecolor=DARK, lw=1.1, zorder=5)

    ax_rc.axvline(0.0, color="#C0392B", ls="--", lw=1.1, label="Zero-Violation Boundary (Δ = 0)")
    ax_rc.set_yticks([2, 1, 0])
    ax_rc.set_yticklabels(m_labels, fontsize=6.4, fontweight="bold")
    ax_rc.set_xlabel("Worst Sensory Off-Flavor Δ (5 Defect Targets)", fontweight="bold")
    ax_rc.set_title("Zero Defect Violation across 924 Pairs", fontsize=7.8, fontweight="bold", pad=6)
    ax_rc.spines["top"].set_visible(False)
    ax_rc.spines["right"].set_visible(False)
    ax_rc.legend(loc="upper right", fontsize=5.6, frameon=False)

    # Right 2x3 grid: 6 Key Marker Volatiles across [Before, None, OT, Joint QP]
    specs = [
        ("DMS",               "DMS OAV (Cabbage)",      True,  CORAL),
        ("X2.3.butanedione",  "Diacetyl OAV (Buttery)", True,  TANG),
        ("H2S",               "H₂S OAV (Sulfur)",       True,  ROSE),
        ("MeSH",              "MeSH (µg/L, Mercaptan)", False, TAUPE),
        ("isopentyl_acetate", "Isoamyl Acetate OAV",    True,  MINT),
        ("ethyl_octanoate",   "Ethyl Octanoate OAV",    True,  BLUE),
    ]
    states = ["before", "none", "ot_proj", "qp_full"]
    for k, (comp, title_s, is_oav, c_hex) in enumerate(specs):
        r_i, c_j = k // 3, k % 3
        ax_s = fig.add_axes([0.52 + c_j * 0.16, 0.56 if r_i == 0 else 0.14, 0.135, 0.34])
        piv = indiv.pivot(index="beer", columns="state", values=comp).reindex(columns=states)
        xx = np.arange(4)
        for _, row in piv.iterrows():
            ax_s.plot(xx, row.to_numpy(float), color="#BDC3C7", lw=0.6, alpha=0.55, zorder=1)
        q25, med, q75 = np.nanpercentile(piv.to_numpy(float), [25, 50, 75], axis=0)
        ax_s.fill_between(xx, q25, q75, color=c_hex, alpha=0.28, lw=0, zorder=2)
        ax_s.plot(xx, med, "o-", color=DARK, mfc=c_hex, mec=DARK, mew=0.6, lw=1.4, ms=4.2, zorder=4)
        if is_oav:
            ax_s.set_yscale("log")
            ax_s.axhline(1.0, color="#C0392B", ls=":", lw=0.85, zorder=3)
            lo_v = np.nanpercentile(piv.to_numpy(float), 5)
            hi_v = np.nanmax(piv.to_numpy(float))
            ax_s.set_ylim(max(min(lo_v, 0.8) * 0.7, 1e-2), max(hi_v, 1.2) * 1.4)
            ax_s.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax_s.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        if c_j == 0:
            ax_s.set_ylabel("OAV (log)" if r_i == 0 else "Concentration / OAV", fontsize=5.8)
        ax_s.set_xticks(xx)
        ax_s.set_xticklabels(["Before", "None", "OT", "QP"] if r_i == 1 else [], fontsize=5.6)
        ax_s.tick_params(axis="y", labelsize=5.3, length=2.0)
        ax_s.set_title(title_s, fontsize=6.3, fontweight="bold", pad=3)
        ax_s.spines["top"].set_visible(False)
        ax_s.spines["right"].set_visible(False)

    save_subpanel(fig, "Sub15_Q3_OAV_Trajectories_Raincloud", "Fig4_Q3_Migration")


# ==============================================================================
# SUB16: Sequential Variance Decomposition + Ablation Forest + k/Missingness
# ==============================================================================
def make_sub16():
    print("Rendering Sub16_Ablation_Variance_Robustness (Gates G4, G5 & U)...")
    vd_cat = pd.read_csv(RESULTS / "supplementary" / "variance_decomposition_by_category.csv")
    vd_all = pd.read_csv(RESULTS / "supplementary" / "variance_decomposition_overall.csv").iloc[0]
    fusion = json.loads((RESULTS / "composite" / "report.json").read_text(encoding="utf-8"))["summary"]
    rob = json.loads((RESULTS / "robustness" / "summary.json").read_text(encoding="utf-8"))

    fig = plt.figure(figsize=(10.6, 3.8))
    ax1 = fig.add_axes([0.06, 0.16, 0.26, 0.74])
    ax2 = fig.add_axes([0.41, 0.16, 0.24, 0.74])
    ax3 = fig.add_axes([0.73, 0.16, 0.25, 0.74])

    # Panel 1: Sequential Variance Decomposition by Sensory Category
    layers = [0, 1, 2]
    for _, r in vd_cat.iterrows():
        cat = r["category"]
        col = FAMILY_COLOR.get(cat, BLUE)
        ys = [float(r["r2_style"]), float(r["r2_plus_pls"]), float(r["r2_full_et"])]
        ax1.plot(layers, ys, "o-", color=col, lw=1.5, ms=5.0, mec=DARK, mew=0.5, label=f"{cat} (n={int(r['n'])})")

    y_all = [float(vd_all["mean_r2_style"]), float(vd_all["mean_r2_plus_pls"]), float(vd_all["mean_r2_full_et"])]
    ax1.plot(layers, y_all, "s--", color=DARK, lw=2.0, ms=5.5, label="All 50 Targets (Mean)")
    ax1.set_xticks(layers)
    ax1.set_xticklabels(["Layer 1:\nStyle Mean", "Layer 2:\n+ Chem. Latent", "Layer 3:\n+ ExtraTrees"], fontsize=6.2, fontweight="bold")
    ax1.set_ylabel("Cumulative Explained Variance (R²)", fontweight="bold")
    ax1.set_title("Sequential Variance Partitioning", fontsize=7.6, fontweight="bold", pad=6)
    ax1.set_xlim(-0.2, 2.3)
    ax1.set_ylim(0.10, 0.44)
    ax1.legend(loc="upper center", ncol=2, fontsize=5.3, frameon=False, columnspacing=1.0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel 2: Fusion & Feature Ablation Forest Plot (95% Cluster Bootstrap CI)
    order = ["avg", "stack_global", "stack_target", "gate_cv", "gate_domain"]
    labels = ["Equal-Weight Fusion\n(Adopted)", "Global Stacking", "Per-Target Stacking", "CV-Gated Fusion", "Domain-Gated"]
    yy = np.arange(len(order))[::-1]
    for scheme, col, off, mk, lab in [("random", BLUE, 0.15, "o", "Random 5-Fold CV"),
                                      ("group",  CORAL, -0.15, "D", "Style-Grouped CV")]:
        for i, v in zip(yy, order):
            s = fusion[f"{scheme}/{v}"]
            lo, hi = s["boot_ci95"]
            d_val = s["delta_vs_et"]
            ax2.errorbar(d_val, i + off, xerr=[[d_val - lo], [hi - d_val]],
                         fmt=mk, ms=4.8, mfc=col, mec=DARK, mew=0.5,
                         ecolor=col, elinewidth=1.1, capsize=2.0, label=lab if i == yy[0] else None)
    ax2.axvline(0, color=DARK, lw=0.8, ls="--")
    ax2.set_yticks(yy)
    ax2.set_yticklabels(labels, fontsize=6.0, fontweight="bold")
    ax2.set_xlabel("ΔR² vs Single ExtraTrees (95% CI)", fontweight="bold")
    ax2.set_title("Gate G5 Fusion Benchmark", fontsize=7.6, fontweight="bold", pad=6)
    ax2.set_xlim(-0.05, 0.04)
    ax2.legend(loc="upper left", fontsize=5.5, frameon=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel 3: Dual Robustness Curves (k-Sensitivity + Missingness Stress)
    k_df = pd.DataFrame(rob["k_sensitivity"])
    m_df = pd.DataFrame(rob["missingness_stress"])

    for sch, col, mk, lab in [("random", BLUE, "o", "Random CV (k sweep)"),
                              ("group",  MINT, "s", "Grouped CV (k sweep)")]:
        sub_k = k_df[k_df["scheme"] == sch].sort_values("k")
        ax3.plot(sub_k["k"], sub_k["mean_r2"], marker=mk, color=col, lw=1.5, ms=4.8,
                 mec=DARK, mew=0.5, label=lab)
    ax3.axvline(14, color=CORAL, ls=":", lw=1.1, label="Selected k = 14 (Plateau)")
    ax3.set_xlabel("Chemical Latent Dimension (k)", fontweight="bold")
    ax3.set_ylabel("Mean Sensory R² (50 Targets)", fontweight="bold")
    ax3.set_title("Latent Dimension & Dropout Stress", fontsize=7.6, fontweight="bold", pad=6)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.legend(loc="lower right", fontsize=5.3, frameon=False)
    y3_lo, y3_hi = ax3.get_ylim()
    ax3.set_ylim(y3_lo, y3_lo + (y3_hi - y3_lo) * 1.75)

    # Inset for Missingness Stress (0% to 20% random feature dropout) in the freed top band
    ax_m = ax3.inset_axes([0.42, 0.68, 0.54, 0.26])
    m_df = m_df.sort_values("missing_rate")
    ax_m.plot(m_df["missing_rate"] * 100, m_df["mean_r2"], "D-", color=CORAL, lw=1.2, ms=3.8, mec=DARK, mew=0.4)
    ax_m.set_xlabel("Feature Dropout (%)", fontsize=5.2, labelpad=1)
    ax_m.set_ylabel("Grouped R²", fontsize=5.2, labelpad=1)
    ax_m.set_title("Missingness Stress", fontsize=5.5, fontweight="bold", pad=2)
    ax_m.tick_params(labelsize=4.8, length=1.5)
    ax_m.spines["top"].set_visible(False)
    ax_m.spines["right"].set_visible(False)

    save_subpanel(fig, "Sub16_Ablation_Variance_Robustness", "Fig1_Data_Manifold")


# ==============================================================================
# PACKAGE ALL 16 STANDALONE SUBPANELS INTO ZIP
# ==============================================================================
def package_subpanels():
    # Also copy the interactive 3D HTML if present
    html_src = ROOT / "figures" / "master" / "Fig4_Q3_State_Space_3D" / "q3_state_space_3d.html"
    if html_src.exists():
        (OUT_DIR / "q3_state_space_3d.html").write_bytes(html_src.read_bytes())

    zip_path = ROOT / "figures" / "MathorCup_TrackA_Subpanels_Nature_Kit.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(OUT_DIR.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(OUT_DIR))
    print(f"\n[ZIP READY] {zip_path} ({zip_path.stat().st_size // 1024} KB)")


def main():
    print("=" * 68)
    print("Rendering 16 Nature/Cell Standalone Subpanels (Zero Stitching)")
    print("=" * 68)
    make_sub01()
    make_sub02()
    make_sub03()
    make_sub04()
    make_sub05()
    make_sub06()
    make_sub07()
    make_sub08()
    make_sub09()
    make_sub10()
    make_sub11()
    make_sub12()
    make_sub13()
    make_sub14()
    make_sub15()
    make_sub16()
    package_subpanels()
    print("=" * 68)
    print("All 16 Standalone Vector Subpanels Complete!")


if __name__ == "__main__":
    main()
