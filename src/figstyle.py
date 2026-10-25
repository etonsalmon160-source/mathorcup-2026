"""论文图件统一风格（AI 辅助生成，2026-10-24）。

设计口径：色盲友好（Okabe–Ito / Paul Tol 衍生）、无图内标题、面板小写字母、
四位以下小数、Nimbus Sans（Helvetica 度量兼容）、矢量 PDF + 600 dpi PNG。
图注由作者撰写，本模块只负责版式。所有数据由调用方从 results/ 读取，
这里不放任何实验数值。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: F401

# ---------------------------------------------------------------- 尺寸
MM = 1 / 25.4
W1 = 89 * MM      # 单栏
W15 = 120 * MM     # 1.5 栏
W2 = 183 * MM      # 双栏

# ---------------------------------------------------------------- 颜色
# Okabe–Ito（色盲安全）
BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"
PINK = "#CC79A7"
ORANGE = "#E69F00"
SKY = "#56B4E9"
YELLOW = "#F0E442"
BLACK = "#000000"
GREY = "#6E6E6E"
LGREY = "#BDBDBD"
XLGREY = "#E6E6E6"

# Paul Tol muted（用于 8 个模块，避免纯饱和色堆叠）
MODULE_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # pink
    "#E69F00",  # orange
    "#56B4E9",  # sky
    "#332288",  # indigo
    "#882255",  # wine
]

METHOD_COLORS = {
    "none": LGREY,
    "ot_proj": ORANGE,
    "ot_cond": SKY,
    "qp_full": BLUE,
    "naive": GREY,
    "source": BLACK,
}

# 顺序型：蓝→浅灰（避免彩虹色）
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
SEQ_ORANGE = LinearSegmentedColormap.from_list(
    "seq_orange", ["#FFF5EB", "#FDD0A2", "#FD8D3C", "#D94801", "#7F2704"])
DIVERGING = "RdBu_r"


def setup_style() -> None:
    """全局 rcParams。重复调用安全。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Nimbus Sans", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7,
        "figure.titlesize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.minor.size": 1.2,
        "ytick.minor.size": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.0,
        "lines.markersize": 3.2,
        "lines.markeredgewidth": 0.4,
        "patch.linewidth": 0.5,
        "grid.color": "#E8E8E8",
        "grid.linewidth": 0.4,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "legend.borderaxespad": 0.3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": True,
        "figure.autolayout": False,
    })


def new_fig(width: float = W2, height: float = 2.0, **kwargs):
    return plt.figure(figsize=(width, height), **kwargs)


def panel_label(ax, letter: str, *, x: float = -0.09, y: float = 1.06,
                size: float = 9) -> None:
    """面板字母：粗体小写，贴左上角。"""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="bottom", ha="left", color="#111111")


def thin_axes(ax, *, grid: bool = False) -> None:
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)
    if grid:
        ax.grid(True, axis="y", zorder=0)
        ax.set_axisbelow(True)


def save_figure(fig, stem: str | Path, *, dpi: int = 600,
                tight: bool = True) -> list[Path]:
    """同时写 PDF（排版）与 PNG（审查）。stem 不带后缀。"""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    kw = dict(bbox_inches="tight" if tight else None, pad_inches=0.015,
              facecolor="white")
    if kw["bbox_inches"] is None:
        kw.pop("bbox_inches")
    p = stem.with_suffix(".pdf")
    fig.savefig(p, **kw)
    out.append(p)
    q = stem.with_suffix(".png")
    fig.savefig(q, dpi=dpi, **kw)
    out.append(q)
    return out


def soften(ax, *, xgrid: bool = False, ygrid: bool = False) -> None:
    """去掉上/右框线、加淡网格。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=("y" if ygrid and not xgrid else
                        "x" if xgrid and not ygrid else "both"),
            color="#ECECEC", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- 独立面板
PANEL_DIR = Path(__file__).resolve().parents[1] / "figures" / "panels"
PANEL_RECORDS: list[dict] = []


def _pdf_mm(pdf_path: Path) -> tuple[float, float]:
    """从 PDF MediaBox 读实际物理尺寸（mm）。失败时退回 0。"""
    try:
        raw = pdf_path.read_bytes()
        m = re.search(rb"/MediaBox\s*\[\s*([-\d.]+)\s+([-\d.]+)\s+"
                      rb"([-\d.]+)\s+([-\d.]+)", raw)
        if not m:
            return 0.0, 0.0
        x0, y0, x1, y1 = (float(v) for v in m.groups())
        return (x1 - x0) / 72.0 * 25.4, (y1 - y0) / 72.0 * 25.4
    except Exception:
        return 0.0, 0.0


def save_panel(fig, key: str, *, description: str = "", sources: str = "",
               stats: str = "", dpi: int = 600) -> list[Path]:
    """导出 Illustrator 用单面板：透明背景、无面板字母、PDF+SVG+PNG。

    面板字母由拼版时在 Illustrator 里添加，这里一律不加，避免二次编辑时
    与正文编号错位。`pdf.fonttype=42` / `svg.fonttype='none'` 已在
    `setup_style()` 固定，SVG 里文字保持可编辑。
    """
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = PANEL_DIR / key
    kw = dict(bbox_inches="tight", pad_inches=0.02, transparent=True,
              facecolor="none", edgecolor="none")
    out = []
    for ext in ("pdf", "svg", "png"):
        p = stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=dpi, **kw)
        out.append(p)
    w_mm, h_mm = _pdf_mm(stem.with_suffix(".pdf"))
    PANEL_RECORDS.append({
        "key": key, "width_mm": w_mm, "height_mm": h_mm,
        "description": description, "sources": sources, "stats": stats,
    })
    return out


def write_panel_index(path: Path | None = None) -> Path:
    path = PANEL_DIR / "PANEL_INDEX.md" if path is None else path
    lines = [
        "# 独立子面板索引（figures/panels）",
        "",
        "每个子面板同时提供 PDF / SVG / PNG 三种格式，供 Adobe Illustrator",
        "拼版。全部为**透明背景**、**不含 a/b/c/d 面板字母**；字体以",
        "`pdf.fonttype=42`、`svg.fonttype='none'` 嵌入，SVG 文字可直接编辑。",
        "面板字母请在 Illustrator 中按最终版式添加。",
        "",
        "物理尺寸为 bbox-tight 后的实际画布尺寸（mm），不是名义 figsize。",
        "",
        "| 面板 | 宽 × 高 (mm) | 内容 | 数据来源 | 统计量 |",
        "|---|---:|---|---|---|",
    ]
    for r in PANEL_RECORDS:
        lines.append(
            f'| {r["key"]} | {r["width_mm"]:.1f} × {r["height_mm"]:.1f} | '
            f'{r["description"]} | {r["sources"]} | {r["stats"]} |')
    lines += [
        "",
        "## 使用提醒",
        "",
        "- 同一图号的子面板在 `figures/paper/figXX_*.pdf` 中已按论文版式",
        "  拼好；这里的单件是给 Illustrator 重排用的，内容与拼版完全同源。",
        "- 透明背景在 PNG 预览里会显示为棋盘格；置入 Illustrator 后即为透明。",
        "- 所有数值从 `results/` 读取，没有手写实验结果。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------- 雨云图
def _kde(values: np.ndarray, grid: np.ndarray, bw: float | None = None):
    from scipy.stats import gaussian_kde

    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 3 or np.allclose(v, v[0]):
        return np.zeros_like(grid)
    kde = gaussian_kde(v, bw_method=bw)
    return kde(grid)


def half_violin(ax, values, position: float, color: str, *,
                width: float = 0.34, side: str = "right",
                alpha: float = 0.55, bw: float | None = None,
                span: tuple[float, float] | None = None) -> None:
    """水平半小提琴（KDE），画在 y = position 的一侧。"""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return
    lo, hi = (np.min(v), np.max(v)) if span is None else span
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    grid = np.linspace(lo - pad, hi + pad, 220)
    d = _kde(v, grid, bw)
    if d.max() <= 0:
        return
    d = d / d.max() * width
    sign = 1.0 if side == "right" else -1.0
    ax.fill_between(grid, position, position + sign * d, color=color,
                    alpha=alpha, linewidth=0, zorder=2)


def sina_jitter(ax, values, position: float, color: str, *, rng=None,
                width: float = 0.10, size: float = 3.0,
                alpha: float = 0.85) -> None:
    """Sina 风格 jitter：点的横向偏移按局部密度缩放。"""
    rng = np.random.default_rng(42) if rng is None else rng
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return
    grid = np.linspace(np.min(v), np.max(v), 120)
    d = _kde(v, grid)
    if d.max() > 0:
        d = d / d.max()
        jit = np.interp(v, grid, d)
    else:
        jit = np.ones_like(v)
    ax.scatter(v, position + rng.uniform(-1, 1, len(v)) * width * (0.35 + jit),
               s=size, color=color, alpha=alpha, linewidths=0, zorder=3)


def iqr_crossbar(ax, values, position: float, color: str = "#222222", *,
                 half_height: float = 0.10, lw: float = 1.0) -> None:
    """中位数 + IQR 横杆（水平）。"""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([q1, q3], [position, position], color=color, lw=lw, zorder=4,
            solid_capstyle="butt")
    ax.plot([med, med], [position - half_height, position + half_height],
            color=color, lw=lw + 0.2, zorder=5)


def raincloud(ax, groups, positions, colors, *, violin_width: float = 0.34,
              jitter: float = 0.075, bw: float | None = None,
              span: tuple[float, float] | None = None, side: str = "right",
              rng=None) -> None:
    """一组水平的 half-violin + Sina jitter + IQR crossbar。"""
    rng = np.random.default_rng(42) if rng is None else rng
    for vals, pos, col in zip(groups, positions, colors):
        half_violin(ax, vals, pos, col, width=violin_width, side=side,
                    bw=bw, span=span)
        sina_jitter(ax, vals, pos + (-0.10 if side == "right" else 0.10),
                    col, rng=rng, width=jitter)
        iqr_crossbar(ax, vals, pos, color="#222222")
