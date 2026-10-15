"""评价类赛题起手式：AHP + 熵权法 + TOPSIS。

    python -m src.evaluate          # 跑一个完整示例，看每个函数怎么用

## 评价类题的套路（几乎每年都有人用）

1. 建**决策矩阵**：m 个对象（城市/方案/供应商…）× n 个指标
2. **指标正向化**：把"越小越好"的指标翻过来（成本、污染、风险…）
3. **标准化**：不同指标量纲不同，必须无量纲化
4. **定权重**：主观（AHP）／客观（熵权法）／两者**组合**
5. **TOPSIS 排序**：算每个对象离"理想解"有多近
6. **灵敏度分析**：权重动一动，排名还稳不稳

## 两个最容易被忽略、也最容易被扣分的点

- 🔴 **AHP 必须做一致性检验**（CR < 0.1）。大多数队伍算完权重就交，
  **没有 CR 的 AHP 是没做完的 AHP**。
- 🔴 **指标正向化不能漏**。指标里混着"越大越好"和"越小越好"时，
  不先统一方向，算出来的排名是错的。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import get_logger

log = get_logger("evaluate")

# AHP 一致性检验用的随机一致性指标（Saaty 标准表，n = 1..10）
RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


# ---------------------------------------------------------------------------
# 第一步：指标正向化 + 标准化
# ---------------------------------------------------------------------------
def normalize_matrix(X: np.ndarray, directions: list[str] | None = None) -> np.ndarray:
    """把决策矩阵统一成"越大越好"且无量纲。

    directions: 每个指标的极性，`"+"` 越大越好（效益型），`"−"` 越小越好（成本型）。
                **不许省** —— 混着极性不处理，排名一定是错的。
    标准化用极差法（min-max），落在 [0,1]。
    """
    X = np.asarray(X, dtype=float)
    m, n = X.shape
    if directions is None:
        directions = ["+"] * n
    if len(directions) != n:
        raise ValueError(f"directions 有 {len(directions)} 个，指标有 {n} 个，对不上")

    out = np.empty_like(X)
    for j in range(n):
        col = X[:, j]
        lo, hi = col.min(), col.max()
        rng = hi - lo
        if rng == 0:                       # 这一列全是同一个值，无信息
            out[:, j] = 0.0
            log.warning("第 %d 个指标取值全部相同，已置零（它对本评价没有区分力）", j + 1)
            continue
        z = (col - lo) / rng               # 正指标：越大越好
        out[:, j] = z if directions[j] in ("+", "benefit") else 1.0 - z
    return out


# ---------------------------------------------------------------------------
# 第二步：定权重
# ---------------------------------------------------------------------------
def ahp_weights(pairwise: np.ndarray) -> tuple[np.ndarray, float]:
    """层次分析法（主观赋权）。返回 (权重, 一致性比率 CR)。

    pairwise：**正互反矩阵**，a_ij 表示"指标 i 比指标 j 重要多少倍"，
    取 1/9 ~ 9。必须满足 a_ij = 1/a_ji。

    🔴 **CR < 0.1 才算通过。** CR ≥ 0.1 说明你的两两比较自相矛盾
    （比如"A 比 B 重要、B 比 C 重要、C 又比 A 重要"），必须回去改判断矩阵。
    """
    A = np.asarray(pairwise, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("两两比较矩阵必须是方阵")
    if not np.allclose(A * A.T, np.ones_like(A), atol=1e-6):
        raise ValueError("两两比较矩阵不满足正互反性（要求 a_ij = 1/a_ji）")

    eigvals, eigvecs = np.linalg.eig(A)
    k = int(np.argmax(eigvals.real))
    lam_max = float(eigvals.real[k])
    w = np.abs(eigvecs[:, k].real)
    w = w / w.sum()

    CI = (lam_max - n) / (n - 1) if n > 1 else 0.0
    RI = RI_TABLE.get(n)
    if RI is None:
        log.warning("n=%d 超出 RI 表范围，无法做一致性检验", n)
        return w, float("nan")
    CR = CI / RI if RI > 0 else 0.0
    if CR >= 0.1:
        log.warning("⚠ AHP 一致性检验不过（CR=%.4f ≥ 0.1），判断矩阵自相矛盾，必须重做", CR)
    return w, CR


def entropy_weights(X: np.ndarray) -> np.ndarray:
    """熵权法（客观赋权）。输入应当是**已正向化**的矩阵。

    思想：某个指标在各对象间差别越大，它承载的信息越多，权重越高；
    如果所有对象在该指标上几乎一样，那它没有区分力，权重趋近 0。
    """
    P = np.asarray(X, dtype=float)
    m, n = P.shape
    col_sum = P.sum(axis=0)
    col_sum[col_sum == 0] = 1e-12                      # 防除零
    P = P / col_sum

    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(P > 0, P * np.log(P), 0.0)
    e = -terms.sum(axis=0) / np.log(m)                 # 熵，∈ [0,1]
    d = 1.0 - e                                        # 差异系数
    if d.sum() == 0:
        log.warning("所有指标的差异系数都是 0，退化为等权")
        return np.full(n, 1.0 / n)
    return d / d.sum()


def combine_weights(*weight_sets: np.ndarray, alphas: list[float] | None = None) -> np.ndarray:
    """组合赋权：把主观权重和客观权重按比例混合。

    这是"主客观结合"的标准做法 —— 论文里既体现业务判断（AHP），
    又不完全靠拍脑袋（熵权法），比单用一种更有说服力。
    """
    ws = [np.asarray(w, dtype=float) for w in weight_sets]
    k = len(ws)
    if alphas is None:
        alphas = [1.0 / k] * k
    if len(alphas) != k or abs(sum(alphas) - 1.0) > 1e-9:
        raise ValueError("alphas 个数要对上，且和为 1")
    out = sum(a * w for a, w in zip(alphas, ws))
    return out / out.sum()


# ---------------------------------------------------------------------------
# 第三步：TOPSIS 排序
# ---------------------------------------------------------------------------
def topsis(X: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TOPSIS。返回 (贴近度, 排名)。**X 必须是已经正向化过的矩阵。**

    贴近度越大越优；排名从 1 开始。
    """
    Z = np.asarray(X, dtype=float)
    w = np.asarray(weights, dtype=float) / np.asarray(weights, dtype=float).sum()

    # 向量归一化（TOPSIS 的标配）
    norm = np.sqrt((Z ** 2).sum(axis=0))
    norm[norm == 0] = 1e-12
    V = Z / norm * w

    ideal_best = V.max(axis=0)
    ideal_worst = V.min(axis=0)
    d_best = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))
    closeness = d_worst / (d_best + d_worst + 1e-12)

    ranking = pd.Series(-closeness).rank(method="min").astype(int).to_numpy()
    return closeness, ranking


# ---------------------------------------------------------------------------
# 第四步：灵敏度分析（**这一节能拿分，很多队伍不做**）
# ---------------------------------------------------------------------------
def weight_sensitivity(X, weights, names=None, *, n_perturb=200, seed=42) -> dict:
    """扰动权重，看排名稳不稳。

    做法：每次给权重加一点随机扰动，重算排名，看与基准排名的
    Spearman 秩相关的分布。**秩相关普遍很高 ⇒ 结论稳健**，可以写进论文。
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    base_w = np.asarray(weights, dtype=float)
    _, base_rank = topsis(X, base_w)

    corrs = []
    for _ in range(n_perturb):
        noise = rng.normal(0, 0.1, size=base_w.shape)
        w = np.clip(base_w + noise, 1e-6, None)
        _, r = topsis(X, w / w.sum())
        corrs.append(float(spearmanr(base_rank, r).statistic))
    corrs = np.array(corrs)
    return {
        "秩相关中位数": float(np.median(corrs)),
        "秩相关5%分位": float(np.percentile(corrs, 5)),
        "排名完全不变的次数": int((corrs > 0.9999).sum()),
        "扰动次数": n_perturb,
        "结论": "稳健" if np.percentile(corrs, 5) > 0.9 else "权重敏感，需在论文里说明",
    }


# ---------------------------------------------------------------------------
def demo() -> None:
    """完整示例：5 个城市 × 4 个指标（含 2 个成本型指标）。"""
    cities = ["甲", "乙", "丙", "丁", "戊"]
    indicators = ["经济水平", "环境质量", "生活成本", "通勤时间"]
    X = np.array([
        [85, 70, 60, 45],
        [72, 88, 45, 30],
        [90, 55, 75, 60],
        [65, 92, 38, 25],
        [78, 76, 55, 40],
    ], dtype=float)
    directions = ["+", "+", "−", "−"]        # 生活成本、通勤时间是"越小越好"

    print("=" * 62)
    print("评价类起手式示例：5 城 × 4 指标")
    print("=" * 62)

    Xn = normalize_matrix(X, directions)
    print(f"\n[1] 正向化 + 标准化后（成本型指标已翻转）：\n{np.round(Xn, 3)}")

    # 主观：AHP
    A = np.array([
        [1, 2, 4, 3],
        [1 / 2, 1, 3, 2],
        [1 / 4, 1 / 3, 1, 1 / 2],
        [1 / 3, 1 / 2, 2, 1],
    ], dtype=float)
    w_ahp, CR = ahp_weights(A)
    print(f"\n[2] AHP 主观权重：{np.round(w_ahp, 4)}")
    print(f"    一致性比率 CR = {CR:.4f}  →  {'✓ 通过（<0.1）' if CR < 0.1 else '✗ 不通过，需重做判断矩阵'}")

    # 客观：熵权法
    w_ent = entropy_weights(Xn)
    print(f"\n[3] 熵权法客观权重：{np.round(w_ent, 4)}")

    # 组合
    w_mix = combine_weights(w_ahp, w_ent, alphas=[0.5, 0.5])
    print(f"\n[4] 组合权重（主客观各半）：{np.round(w_mix, 4)}")

    # TOPSIS
    c_mix, r_mix = topsis(Xn, w_mix)
    df = pd.DataFrame({
        "城市": cities, "贴近度": np.round(c_mix, 4), "排名": r_mix,
    }).sort_values("排名")
    print(f"\n[5] TOPSIS 排序结果：")
    print(df.to_string(index=False))

    # 灵敏度
    s = weight_sensitivity(Xn, w_mix)
    print(f"\n[6] 灵敏度分析（权重扰动 {s['扰动次数']} 次）：")
    for k, v in s.items():
        print(f"    {k}：{v}")

    print("\n" + "=" * 62)
    print("论文里就这么写：正向化 → AHP(附 CR) → 熵权 → 组合 → TOPSIS → 灵敏度")
    print("=" * 62)


if __name__ == "__main__":
    demo()
