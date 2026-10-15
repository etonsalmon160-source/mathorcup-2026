"""出图。

    python -m src.plots --config src/configs/baseline.yaml

图直接进论文，所以：300 dpi、字号放大、中文别乱码。图存到 figures/（这个目录入库）。
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import PATHS, get_logger, load_config

log = get_logger("plots")

# 常见的几款中文字体，按优先级试。本机目前只有 Droid Sans Fallback。
CJK_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC", "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei", "Microsoft YaHei", "SimHei", "PingFang SC",
    "Droid Sans Fallback", "AR PL UMing CN",
]


def setup_font() -> None:
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_CANDIDATES:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            log.info("绘图字体：%s", name)
            break
    else:
        log.warning("没找到中文字体，图中的中文会变方框。装一个：sudo apt install fonts-noto-cjk")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300,
                         "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12})


def _save(fig, name: str, sub: str) -> None:
    """图放在 figures/<实验名>/ 下，不同实验的图不会互相覆盖。"""
    d = PATHS["figures"] / sub
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("图 -> %s", path)


def plot_importance(out_dir, sub: str, top_n: int = 30) -> None:
    p = out_dir / "feature_importance.csv"
    if not p.exists():
        return
    imp = pd.read_csv(p).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.32)))
    ax.barh(imp["特征"], imp["重要度"], color="#4C72B0")
    ax.set_xlabel("重要度（增益）")
    ax.set_title(f"特征重要度 Top {len(imp)}")
    _save(fig, "01_特征重要度.png", sub)


def plot_classification(out_dir, sub: str, fig_name: str) -> None:
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

    p = out_dir / "oof_predictions.csv"
    if not p.exists():
        return
    oof = pd.read_csv(p)

    cm = confusion_matrix(oof["y_true"], oof["oof_pred"])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xlabel("预测"); ax.set_ylabel("真实"); ax.set_title("交叉验证混淆矩阵")
    ax.set_xticks(range(cm.shape[0])); ax.set_yticks(range(cm.shape[0]))
    fig.colorbar(im, ax=ax)
    _save(fig, fig_name, sub)

    if "proba_0" not in oof.columns or oof["proba_0"].isna().any():
        return
    classes = _classes(out_dir)
    if oof.filter(like="proba_").shape[1] == 2:
        pos = oof["proba_1"]
        auc = roc_auc_score(oof["y_true"], pos)
        fpr, tpr, _ = roc_curve(oof["y_true"], pos)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot(fpr, tpr, lw=2, color="#C44E52", label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set_xlabel("假正率"); ax.set_ylabel("真正率")
        ax.set_title("交叉验证 ROC 曲线"); ax.legend(loc="lower right")
        _save(fig, "03_ROC曲线.png", sub)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for k, lab in enumerate(classes):
            ax.hist(oof.loc[oof["y_true"] == k, "proba_1"], bins=40, alpha=0.6,
                    label=f"真实：{lab}", density=True)
        ax.axvline(0.5, ls="--", color="black", lw=1, label="默认阈值 0.5")
        ax.set_xlabel("预测为正类的概率"); ax.set_ylabel("密度")
        ax.set_title("预测概率分布（阈值该定在哪，看图说话）")
        ax.legend()
        _save(fig, "04_预测概率分布.png", sub)


def plot_regression(out_dir, sub: str, fig_name: str) -> None:
    p = out_dir / "oof_predictions.csv"
    if not p.exists():
        return
    oof = pd.read_csv(p)
    y, pred = oof["y_true"].to_numpy(), oof["oof_pred"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(y, pred, s=8, alpha=0.4, color="#4C72B0")
    lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    ax.plot(lim, lim, "--", color="#C44E52", lw=1.5, label="理想线 y = ŷ")
    ax.set_xlabel("真实值"); ax.set_ylabel("预测值"); ax.set_title("交叉验证：预测 vs 真实")
    ax.legend()
    _save(fig, fig_name, sub)

    resid = pred - y
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].hist(resid, bins=50, color="#55A868")
    axes[0].axvline(0, ls="--", color="black", lw=1)
    axes[0].set_xlabel("残差（预测 − 真实）"); axes[0].set_ylabel("频数"); axes[0].set_title("残差分布")
    axes[1].scatter(pred, resid, s=8, alpha=0.4, color="#4C72B0")
    axes[1].axhline(0, ls="--", color="black", lw=1)
    axes[1].set_xlabel("预测值"); axes[1].set_ylabel("残差"); axes[1].set_title("残差 vs 预测")
    _save(fig, "03_残差诊断.png", sub)


def _classes(out_dir) -> list:
    meta = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    return meta.get("classes") or [0, 1]


def main() -> None:
    ap = argparse.ArgumentParser(description="出图")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    setup_font()
    cfg = load_config(args.config)
    out_dir = PATHS["results"] / cfg["name"]
    meta_path = out_dir / "metrics.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"找不到 {meta_path}。先跑 src.train。")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    sub = cfg["name"]
    plot_importance(out_dir, sub)
    if meta["task"] == "classification":
        plot_classification(out_dir, sub, "02_混淆矩阵.png")
    else:
        plot_regression(out_dir, sub, "02_预测vs真实.png")
    print(f"\n图已存到 {PATHS['figures']}/")


if __name__ == "__main__":
    main()
