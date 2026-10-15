"""共形风险控制：给决策加一层**分布无关的有限样本保证**。

    python -m src.conformal --config src/configs/baseline.yaml --budget 0.10

## 为什么需要它

平时定阈值靠"调一个看效果"，但那个阈值**没有任何保证**。
共形风险控制（Conformal Risk Control, ICLR 2024）能给出一句话的保证：

> **在验证集上校准出来的阈值 λ，使得在新样本上「期望损失 ≤ α」成立。**

不依赖任何分布假设，且是**有限样本**成立的（样本越少越显示它的价值）。

## 实现来源（两套独立实现）

1. **官方实现**：`get_lhat` 逐字照抄自论文作者自己的仓库
   `github.com/aangelopoulos/conformal-risk` 的 `core/get_lhat.py`（MIT 协议）。
   官方注释说这个算法"含函数头一共 5 行"。
2. **独立对照**：用 **MAPIE**（scikit-learn 风格的第三方实现）跑同一件事，
   两者结果应当一致。**两套独立实现互相印证，而不是自己验自己**——
   这是抄 `~/luad_v2/docs/scientific_rigor_and_audit.md` 里"避免同源即自证"的准则。

## 两种阈值口径，别混为一谈（重要）

| 口径 | 有保证吗 | 说明 |
|---|---|---|
| **漏判率（FNR）受控** | ✅ **有**：E[FNR] ≤ α | 风险对阈值单调，官方 `get_lhat` 直接适用 |
| **期望误判代价最小** | ⚠️ **没有**：这是插值估计 | 代价对阈值是 U 形，**不单调**，用不了共形保证；只能给 bootstrap 置信区间 |

**论文里必须写清这个区别。** 把第二种说成"有保证"是夸大。

## 产出

`results/<名>/conformal.json` —— 阈值、实测风险、保证是否成立、两套实现是否一致。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import PATHS, SEED, get_logger, load_config, run_dir, set_global_seed

log = get_logger("conformal")


# ---------------------------------------------------------------------------
# 官方实现：逐字照抄自 aangelopoulos/conformal-risk (MIT) 的 core/get_lhat.py
# ---------------------------------------------------------------------------
def get_lhat(calib_loss_table, lambdas, alpha, B=1):
    """Gets the value of lambda hat that controls the marginal risk for a
    monotone risk function. The calib loss table should be ordered from
    small loss to large loss."""
    n = calib_loss_table.shape[0]
    rhat = calib_loss_table.mean(axis=0)
    lhat_idx = max(np.argmax(((n / (n + 1)) * rhat + B / (n + 1)) >= alpha) - 1, 0)
    return lambdas[lhat_idx]


# ---------------------------------------------------------------------------
def fnr_loss_table(proba: np.ndarray, y: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """构造"漏判率"的校准损失表，形状 (n 正类样本, n 阈值)。

    🔴 **必须只在正类样本上校准**（Mondrian / 类条件共形）。

    一开始我用的是"全部样本"，损失 = 1{真阳性且判为阴性}，那算出来的是
    **FN / 总样本数**，而不是**漏判率 FNR = FN / 正类数**。两回事：
    正类占 25% 时，两者差 4 倍。用它校准出来的阈值当然不满足 FNR 的保证。

    文献里（cost-sensitive conformal 的基准论文）明确记录过这个坑：
    边缘共形在稀有类上的覆盖率可以低到 0.5%，**类条件共形才是对的**。

    阈值越大 → 判正的越少 → 漏判越多 ⇒ 该风险对阈值**单调递增**，符合共形要求。
    """
    pos = y == 1
    if pos.sum() == 0:
        raise ValueError("校准集里没有正类样本，无法按类条件校准")
    pred_pos = proba[pos][:, None] >= lambdas[None, :]      # (n_pos, K)
    return (~pred_pos).astype(float)                        # 正类被漏掉的指示


def cost_at(proba, y, lam, cost_fp, cost_fn) -> float:
    """某个阈值下的平均误判代价（插值估计，**没有共形保证**）。"""
    pred = (proba >= lam).astype(int)
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return (fp * cost_fp + fn * cost_fn) / len(y)


def main() -> None:
    ap = argparse.ArgumentParser(description="共形风险控制：给决策加保证")
    ap.add_argument("--config", required=True)
    ap.add_argument("--budget", type=float, default=0.10,
                    help="漏判率预算 α：保证在新样本上 E[漏判率] ≤ α（默认 0.10）")
    ap.add_argument("--cost-fp", type=float, default=1.0, help="误判为正的代价")
    ap.add_argument("--cost-fn", type=float, default=5.0, help="漏判的代价（通常更高）")
    ap.add_argument("--bootstrap", type=int, default=2000, help="代价的 bootstrap 次数")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(SEED)
    out_dir = run_dir(cfg)
    oof_path = out_dir / "oof_predictions.csv"
    if not oof_path.exists():
        raise FileNotFoundError(f"缺 {oof_path}。先跑 make train（需要 OOF 预测来校准）")

    oof = pd.read_csv(oof_path)
    if "proba_1" not in oof.columns:
        raise ValueError("这是回归任务或多分类，共形风险控制目前只支持二分类")
    proba = oof["proba_1"].to_numpy()
    y = oof["y_true"].to_numpy().astype(int)

    # 把 OOF 预测对半劈：一半校准（定阈值），一半验证（看保证是否真的成立）
    # **不能拿同一批数据既校准又验证** —— 那就成了自证。
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    half = len(y) // 2
    cal, val = idx[:half], idx[half:]

    lambdas = np.round(np.arange(0.001, 1.0, 0.001), 4)
    alpha = args.budget

    # ---- 官方实现：漏判率受控 ----
    loss_tab = fnr_loss_table(proba[cal], y[cal], lambdas)
    lam_official = float(get_lhat(loss_tab, lambdas, alpha))

    # 校准集上的 FNR：同样只在正类上算，与校准口径一致
    fnr_cal = float(loss_tab[:, np.where(lambdas == lam_official)[0][0]].mean())
    pred_val = (proba[val] >= lam_official).astype(int)
    fnr_val = float(((pred_val == 0) & (y[val] == 1)).sum() / max((y[val] == 1).sum(), 1))

    # ---- 独立对照：MAPIE ----
    # MAPIE 没有 false_negative_rate 这个风险名，但 recall = 1 − FNR，
    # 所以「控制 recall ≥ 1−α」与「控制漏判率 ≤ α」是同一件事。
    lam_mapie = None
    try:
        from mapie.risk_control import BinaryClassificationController

        # MAPIE 的 predict_function 要返回 sklearn 那样的**两列**概率矩阵，
        # 它内部取 [:, 1] 当正类概率。
        def _two_col(X):
            p = np.asarray(X).ravel()
            return np.column_stack([1.0 - p, p])

        ctrl = BinaryClassificationController(
            predict_function=_two_col, risk="recall",
            target_level=1 - alpha, list_predict_params=lambdas,
        )
        ctrl.calibrate(proba[cal].reshape(-1, 1), y[cal])
        pred = ctrl.predict(proba[val].reshape(-1, 1))
        lam_mapie = float(np.asarray(pred[1]).ravel()[0]) if isinstance(pred, tuple) \
            else float(np.asarray(pred).ravel()[0])
        log.info("MAPIE 独立实现给出阈值 λ = %.3f", lam_mapie)
    except Exception as e:                                  # noqa: BLE001
        log.warning("MAPIE 对照未跑通（不影响官方实现的结果）：%s", str(e)[:150])

    # ---- 代价最小阈值（插值，无保证）+ bootstrap 区间 ----
    cost_grid = np.array([cost_at(proba[cal], y[cal], l, args.cost_fp, args.cost_fn)
                          for l in lambdas])
    lam_cost = float(lambdas[int(np.argmin(cost_grid))])
    boot = []
    for _ in range(args.bootstrap):
        s = rng.integers(0, len(val), len(val))
        boot.append(cost_at(proba[val][s], y[val][s], lam_cost, args.cost_fp, args.cost_fn))
    lo, hi = np.percentile(boot, [2.5, 97.5])

    result = {
        "alpha_budget": alpha,
        "cost_fp": args.cost_fp, "cost_fn": args.cost_fn,
        "n_calibration": len(cal), "n_validation": len(val),
        "threshold_fnr_controlled": lam_official,
        "fnr_on_calibration": round(fnr_cal, 4),
        "fnr_on_validation": round(fnr_val, 4),
        "guarantee_holds": bool(fnr_val <= alpha),
        "mapie_threshold": lam_mapie,
        # 两套实现选的是**解集里不同的点**（官方取满足漏判率约束下最保守的，
        # MAPIE 取满足 recall 约束下最宽松的），比"阈值是否相等"没有意义。
        # 有意义的对照是：**独立实现算出来的阈值，是否同样满足保证**。
        "mapie_fnr_on_validation": (None if lam_mapie is None else round(float(
            ((proba[val] < lam_mapie) & (y[val] == 1)).sum() / max((y[val] == 1).sum(), 1)
        ), 4)),
        "mapie_also_satisfies_guarantee": (None if lam_mapie is None else bool(
            ((proba[val] < lam_mapie) & (y[val] == 1)).sum() / max((y[val] == 1).sum(), 1) <= alpha
        )),
        "threshold_cost_optimal": lam_cost,
        "cost_ci95_on_validation": [round(float(lo), 5), round(float(hi), 5)],
        "note": "threshold_fnr_controlled 有分布无关的有限样本保证；"
                "threshold_cost_optimal 只是插值估计，代价对阈值不单调，用不了共形保证。",
    }
    (out_dir / "conformal.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 人看的结论 ----
    print(f"\n=== 共形风险控制（预算 α = {alpha}）===")
    print(f"校准集 {len(cal)} 条 / 验证集 {len(val)} 条，阈值网格 {len(lambdas)} 个")
    print(f"\n【有保证的口径】漏判率受控")
    print(f"  校准出的阈值 λ* = {lam_official:.3f}")
    print(f"  校准集漏判率 = {fnr_cal:.4f}")
    print(f"  验证集漏判率 = {fnr_val:.4f}   ← 保证要求 ≤ {alpha}")
    print(f"  {'✓ 保证成立' if fnr_val <= alpha else '✗ 保证未成立（样本太小或偏移过大，如实报告）'}")
    if lam_mapie is not None:
        mapie_fnr = ((proba[val] < lam_mapie) & (y[val] == 1)).sum() / max((y[val] == 1).sum(), 1)
        print(f"  MAPIE 独立实现给出 λ = {lam_mapie:.3f}，其验证集漏判率 = {mapie_fnr:.4f}")
        print("    （两套实现选的是**解集里不同的点**：官方取最保守的、MAPIE 取最宽松的，")
        print("      所以阈值本就不该相等。有意义的是下一条：）")
        print(f"    {'✓ 独立实现同样满足保证' if mapie_fnr <= alpha else '✗ 独立实现不满足保证，需查'}")
    else:
        print("  （MAPIE 对照未跑通）")
    print(f"\n【无保证的口径】期望代价最小")
    print(f"  阈值 λ = {lam_cost:.3f}，验证集期望代价 95% 区间 [{lo:.4f}, {hi:.4f}]")
    print("  ⚠ 这个口径是插值估计，**没有**共形保证 —— 论文里不要把它写成有保证的。")
    print(f"\n产物 -> {out_dir / 'conformal.json'}")


if __name__ == "__main__":
    main()
