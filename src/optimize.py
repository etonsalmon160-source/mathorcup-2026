"""优化类赛题起手式：线性/整数规划 + 指派问题 + 灵敏度分析。

    python -m src.optimize         # 跑四个完整示例，看每个函数怎么用

## 优化类题的套路

1. **定决策变量**：你要决定的是什么？（生产多少、派谁去、从哪运到哪）
2. **写目标函数**：最大化利润 / 最小化成本 —— 注意统一成"最小化"再交给求解器
3. **列约束**：资源上限、需求下限、整数要求
4. **求解**
5. 🔴 **灵敏度分析**：约束右端项动一动，最优值怎么变？**这一步几乎所有优化题都要求，
   而且最能拿分** —— 它回答的是"老板最该去争取哪个资源"。

## scipy 已经装好，不用额外依赖

`scipy.optimize` 里 `linprog`（线性）、`milp`（整数/混合整数）、
`linear_sum_assignment`（指派问题）都在，背后是 HiGHS 求解器。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, linear_sum_assignment, milp

from src.config import get_logger

log = get_logger("optimize")


# ---------------------------------------------------------------------------
def solve_lp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None,
             bounds=None, maximize=False) -> dict:
    """线性规划。**注意 scipy 只做最小化**，求最大就取负号，我们替你处理好。

    返回 dict：最优解 x、最优值 objective、是否成功、对偶值（shadow price）。
    """
    c = np.asarray(c, dtype=float)
    if maximize:
        c = -c
    res = linprog(
        c,
        A_ub=None if A_ub is None else np.asarray(A_ub, float),
        b_ub=None if b_ub is None else np.asarray(b_ub, float),
        A_eq=None if A_eq is None else np.asarray(A_eq, float),
        b_eq=None if b_eq is None else np.asarray(b_eq, float),
        bounds=bounds if bounds is not None else (0, None),
        method="highs",
    )
    if not res.success:
        log.warning("线性规划未求解成功：%s", res.message)
    obj = -res.fun if maximize else res.fun
    return {
        "x": res.x, "objective": float(obj), "success": bool(res.success),
        "message": res.message,
        # 对偶值 = 该约束放松一个单位，目标能改善多少（经济学上就是"影子价格"）
        # ⚠ 最大化时我们给 c 取了负号，对偶值也跟着反了，这里要翻回来，
        #   否则会出现"影子价格是负的、但利润却在涨"这种自相矛盾的输出。
        "shadow_price": None if res.ineqlin is None else (
            -np.asarray(res.ineqlin.marginals) if maximize
            else np.asarray(res.ineqlin.marginals)),
    }


def solve_milp(c, constraints, integrality, bounds, maximize=False) -> dict:
    """整数 / 混合整数规划。`integrality` 里 1 表示该变量必须取整。

    例：生产件数、派车数量、开关决策（0/1）。
    """
    c = np.asarray(c, dtype=float)
    if maximize:
        c = -c
    res = milp(c=c, constraints=constraints, integrality=np.asarray(integrality),
               bounds=bounds)
    if not res.success:
        log.warning("整数规划未求解成功：%s", res.message)
    obj = -res.fun if (maximize and res.fun is not None) else res.fun
    return {"x": res.x, "objective": float(obj), "success": bool(res.success),
            "message": res.message}


def assign(cost: np.ndarray, maximize=False) -> dict:
    """指派问题（谁去做哪件事最省）。匈牙利算法，O(n³)。

    也适用于"任务分配""人员排班""设备对订单"这类一对一指派。
    """
    C = np.asarray(cost, dtype=float)
    rows, cols = linear_sum_assignment(-C if maximize else C)
    total = float(C[rows, cols].sum())
    return {"配对": list(zip(rows.tolist(), cols.tolist())),
            "总代价": total, "成功": True}


# ---------------------------------------------------------------------------
def rhs_sensitivity(c, A_ub, b_ub, bounds, idx, deltas=None, maximize=True) -> "np.ndarray":
    """**灵敏度分析**：某个约束的右端项变化时，最优目标值怎么变。

    `idx` 是要扰动的那个约束的下标。返回 (扰动值, 最优目标值) 的两列数组。

    论文里画成图：横轴资源量、纵轴最优利润 —— **斜率就是影子价格**，
    告诉你"再争取一个单位的这个资源，值多少钱"。老板最关心这个。
    """
    A_ub = np.asarray(A_ub, float)
    b_ub = np.asarray(b_ub, float).copy()
    base = b_ub[idx]

    # 基准点的**局部**影子价格：从求解器直接读，不要用整条曲线的平均斜率 ——
    # 这条曲线是分段线性的，跑远了斜率就变了（别的约束变成瓶颈）。
    if deltas is None:
        deltas = np.linspace(-0.5 * base, 0.5 * base, 21)

    out = []
    for d in deltas:
        b = b_ub.copy()
        b[idx] = base + d
        r = solve_lp(c, A_ub, b, bounds=bounds, maximize=maximize)
        out.append((base + d, r["objective"]))
    return np.array(out)


# ---------------------------------------------------------------------------
def demo() -> None:
    print("=" * 66)
    print("优化类起手式示例")
    print("=" * 66)

    # ---- 示例 1：线性规划（经典生产计划）----
    # 产品甲/乙，利润 40/30；消耗原料(≤100)、工时(≤120)、市场上限(甲≤30)
    print("\n[1] 线性规划：生产计划")
    print("    产品甲利润 40/件、乙 30/件")
    print("    约束：原料 2x甲+1x乙 ≤ 100；工时 1x甲+2x乙 ≤ 120；甲 ≤ 30")
    r = solve_lp(c=[40, 30], A_ub=[[2, 1], [1, 2], [1, 0]], b_ub=[100, 120, 30],
                 maximize=True)
    _A, _b = [[2, 1], [1, 2], [1, 0]], [100, 120, 30]
    print(f"    → 最优：甲 {r['x'][0]:.1f} 件、乙 {r['x'][1]:.1f} 件，"
          f"最大利润 {r['objective']:.1f}")
    if r["shadow_price"] is not None:
        for i, sp in enumerate(r["shadow_price"]):
            if abs(sp) > 1e-9:
                print(f"    约束 {i+1} 的影子价格 = {sp:.2f}（放松 1 单位，利润增这么多）")

    # ---- 示例 2：整数规划（同样的题，但件数必须是整数）----
    print("\n[2] 整数规划：件数必须是整数")
    cons = LinearConstraint([[2, 1], [1, 2], [1, 0]], -np.inf, [100, 120, 30])
    r2 = solve_milp(c=[40, 30], constraints=cons, integrality=[1, 1],
                    bounds=Bounds(0, np.inf), maximize=True)
    print(f"    → 最优：甲 {r2['x'][0]:.0f} 件、乙 {r2['x'][1]:.0f} 件，"
          f"最大利润 {r2['objective']:.0f}")
    print("    （整数解 ≤ 线性解，这是正常的——加了约束）")

    # ---- 示例 3：指派问题 ----
    print("\n[3] 指派问题：4 个人做 4 项任务的耗时（分钟）")
    cost = np.array([[10, 15, 9, 12],
                     [14, 11, 13, 10],
                     [9, 13, 16, 14],
                     [12, 10, 11, 15]], dtype=float)
    print(f"    耗时矩阵：\n{cost}")
    a = assign(cost)
    total_naive = cost[np.arange(4), np.arange(4)].sum()   # 对角安排的对照
    print(f"    → 最优配对：{a['配对']}")
    print(f"      最优总耗时 {a['总代价']:.0f} 分钟")
    print(f"      （对角线那种顺手安排要 {total_naive:.0f} 分钟，省 {total_naive - a['总代价']:.0f}）")

    # ---- 示例 4：灵敏度分析 ----
    print("\n[4] 灵敏度分析：原料量变化时，最大利润怎么变")
    s = rhs_sensitivity([40, 30], _A, _b, bounds=(0, None), idx=0, maximize=True)
    print("    原料量 → 最大利润：")
    for b, obj in s[::5]:
        print(f"      {b:7.1f}  →  {obj:8.1f}")
    sp0 = r["shadow_price"][0]
    print(f"    基准点的影子价格 = {sp0:.2f}（从求解器直接读）")
    print("    ⚠ 这条曲线是**分段线性**的：原料加到一定程度后，工时就成了新瓶颈，")
    print("      曲线变平、影子价格降到 0。所以别用整条线的平均斜率当影子价格。")
    print("    论文里画这条曲线 + 标注分段拐点，说明「再进一吨原料值多少钱、到多少就不再值」")

    print("\n" + "=" * 66)
    print("论文里就这么写：定变量 → 目标 → 约束 → 求解 → 灵敏度（影子价格）")
    print("=" * 66)


if __name__ == "__main__":
    demo()
