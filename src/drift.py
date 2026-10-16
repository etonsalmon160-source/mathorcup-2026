"""分布漂移的正式统计检验（零依赖，只用 numpy/scipy）。

    python -m src.drift            # 跑示例

## 为什么自己写而不是装库

候选库都有代价：
- **Alibi Detect**：统计检验最全，但拉 TensorFlow 或 PyTorch（很重），
  且许可证已从 Apache 改成 **BSL —— 不再是开源**，论文里不能称它 open-source。
- **Evidently**：面向生产监控，带 UI，且**没有方法学论文**，引用只能算"引工具"。
- **NannyML**：会降级 lightgbm/xgboost（见下）。

**这些检验的数学本身很简单**，自己写三十行，好处是：
① 零依赖、不会跟竞赛环境的版本打架；
② 逻辑透明，**能在论文里讲清楚**（比"我们调了个库"有力得多）；
③ 每一步都能对着原论文核对。

## 三个检验

| 检验 | 测什么 | 出处 |
|---|---|---|
| **MMD**（最大均值差异） | 用核方法测**两个多元分布**是否相同 | Gretton et al., *A Kernel Two-Sample Test*, JMLR 2012 |
| **逐特征 KS** | 每个特征单独测，**定位是哪些字段在漂移** | 经典 |
| **C2ST** | 分类器两样本检验 | 见 `src/validate.py`，Lopez-Paz & Oquab, ICLR 2017 |

> MMD 是**多元**的（把整条样本当整体看），KS 是**逐特征**的（适合定位）。
> 两者互补：MMD 说"整体有没有变"，KS 说"哪里变了"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import get_logger

log = get_logger("drift")


# ---------------------------------------------------------------------------
def _rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0))


def _median_heuristic(X: np.ndarray, Y: np.ndarray, rng, max_n: int = 500) -> float:
    """带宽用"中位数启发式"（Gretton 2012 推荐）：取样本间距离的中位数。

    比手选带宽稳，也是标准做法。样本大时子采样估计，省时间。
    """
    Z = np.vstack([X, Y])
    if len(Z) > max_n:
        Z = Z[rng.choice(len(Z), max_n, replace=False)]
    d = np.sqrt(np.maximum(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1), 0))
    med = np.median(d[d > 0]) if (d > 0).any() else 1.0
    return 1.0 / (2.0 * med ** 2 + 1e-12)


def mmd_test(X: np.ndarray, Y: np.ndarray, *, n_perm: int = 200,
             seed: int = 42) -> dict:
    """MMD 两样本检验：**两个多元分布是否相同**。

    统计量 MMD² = E[k(x,x')] + E[k(y,y')] − 2·E[k(x,y)]，
    零假设下应当接近 0。用置换检验给 p 值。

    ⚠️ 置换 p 值有分辨率下限 1/(n_perm+1)。要更小的 p 就加大 n_perm。
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    rng = np.random.default_rng(seed)
    gamma = _median_heuristic(X, Y, rng)

    def stat(A, B):
        Kxx = _rbf_kernel(A, A, gamma)
        Kyy = _rbf_kernel(B, B, gamma)
        Kxy = _rbf_kernel(A, B, gamma)
        n, m = len(A), len(B)
        # 对角元要排除，否则自己跟自己比会引入偏置
        np.fill_diagonal(Kxx, 0.0)
        np.fill_diagonal(Kyy, 0.0)
        return (Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
                - 2 * Kxy.mean())

    obs = float(stat(X, Y))

    pool = np.vstack([X, Y])
    n = len(X)
    perm = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(pool))
        perm[i] = stat(pool[idx[:n]], pool[idx[n:]])

    p = float((perm >= obs).sum() + 1) / (n_perm + 1)
    return {
        "mmd2": obs,
        "p_value": round(p, 4),
        "n_permutations": n_perm,
        "n_x": len(X), "n_y": len(Y),
        "verdict": "两边分布**不同**（p < 0.05）" if p < 0.05
                   else "不能拒绝（两边同分布）",
        "note": "置换 p 值下限为 1/(n_perm+1)；要更小的 p 请加大 n_perm。",
    }


# ---------------------------------------------------------------------------
def per_feature_ks(X: np.ndarray, Y: np.ndarray, names: list[str],
                   *, alpha: float = 0.05, correct: str = "bh") -> pd.DataFrame:
    """逐特征 KS 检验 + **多重检验校正**，用来定位**是哪些字段在漂移**。

    ⚠️ 逐特征做几十上百次检验，不做校正的话假阳性会淹掉真信号。
    默认用 Benjamini-Hochberg 控制 FDR（你熟悉的那个）。
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    rows = []
    for j, nm in enumerate(names):
        a, b = X[:, j], Y[:, j]
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if len(a) < 5 or len(b) < 5:
            rows.append({"特征": nm, "KS统计量": np.nan, "p值": np.nan, "校正后p": np.nan})
            continue
        s, p = stats.ks_2samp(a, b)
        rows.append({"特征": nm, "KS统计量": float(s), "p值": float(p)})

    df = pd.DataFrame(rows)
    ok = df["p值"].notna()
    if ok.sum() and correct:
        df.loc[ok, "校正后p"] = stats.false_discovery_control(
            df.loc[ok, "p值"].to_numpy(), method=correct)
    df["是否漂移"] = df.get("校正后p", df["p值"]) < alpha
    return df.sort_values("KS统计量", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
def demo() -> None:
    from src.validate import _c2st_test  # 复用已有的 C2ST

    rng = np.random.default_rng(0)
    print("=" * 64)
    print("分布漂移检测：MMD（整体）+ 逐特征 KS（定位）+ C2ST（分类器）")
    print("=" * 64)

    names = [f"特征{i+1}" for i in range(6)]
    X = rng.normal(0, 1, (600, 6))

    for tag, Y in [
        ("同分布", rng.normal(0, 1, (600, 6))),
        ("整体平移 + 单特征漂移", np.column_stack([
            rng.normal(0, 1, 600), rng.normal(2.0, 1, 600),   # 第 2 个特征明显漂移
            rng.normal(0, 1, 600), rng.normal(0, 1, 600),
            rng.normal(0, 1, 600), rng.normal(0, 1, 600)])),
    ]:
        print(f"\n=== {tag} ===")
        m = mmd_test(X, Y, n_perm=100)
        print(f"  MMD² = {m['mmd2']:+.5f}   p = {m['p_value']:.4f}   → {m['verdict']}")

        ks = per_feature_ks(X, Y, names)
        drifting = ks[ks["是否漂移"]]["特征"].tolist()
        print(f"  逐特征 KS 判为漂移的：{drifting if drifting else '（无）'}")
        print("  前 3 名：")
        for _, r in ks.head(3).iterrows():
            print(f"    {r['特征']:<8} KS={r['KS统计量']:.4f}  校正后p={r['校正后p']:.3g}")

    print("\n" + "=" * 64)
    print("MMD 回答「整体变没变」，KS 回答「哪里变了」，C2ST 回答「能不能分得开」")
    print("=" * 64)


if __name__ == "__main__":
    demo()
