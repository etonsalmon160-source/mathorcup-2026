"""复合架构评估：树模型 + 感官潜状态，按维度融合。

AI 辅助声明：本脚本由 AI 辅助编写。融合方案、领域分流规则与采纳标准
由参赛队员决定；AI 只负责实现与跑数。

评估的复合方式
--------------
1. avg            两模型 50/50 平均（不看结果，无拟合）
2. stack_global   外层训练集内用内层 OOF 拟合一个全局混合系数 alpha
3. stack_target   每个感官维度各拟合一个 alpha（向 0.5 收缩 50%）
4. gate_domain    按预先写死的领域规则分流：
                   有单一主导化学测量的维度走树模型，其余走潜状态模型
5. gate_cv        每个维度在外层训练集内用内层 CV 选表现更好的基模型
6. oracle         每个维度直接选外层测试集上更好的基模型（只作上界参考，不可采纳）

诚实性说明
----------
领域规则是在看到 v1 逐维结果之后写的，因此它只是一个**假设**，
不是预注册。真正的无泄漏对照是 gate_cv（选模也发生在外层训练集内）。
所有复合方式都走同一套外层划分，与 v1 的 ET / 潜状态结果可比。

采纳标准（跑之前定好，不许改）
------------------------------
随机 CV 比 ET 低不超过 0.002，且分组 CV 比 ET 高至少 0.005，
且分组 CV 的配对 bootstrap 95% 区间不含 0。
不满足就只用复合架构做论文叙事，主模型仍用 ET。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v1 as R  # noqa: E402

OUT = R.RESULTS / "composite"

BASES = ["et", "latent_sens_ridge"]

# 领域规则：有单一主导化学测量、且感知-化学近似单调的维度走树模型。
# 依据是 1.csv 里能一一对应的测量项，不依据 v1 的 R²。
DOMAIN_DIRECT = [
    "acidity",    # pH + 有机酸
    "alcohol",    # 乙醇
    "lactic",     # 乳酸
    "acetic",     # 乙酸
    "bitternes",  # 异 α-酸
    "sweetness",  # 糖
    "X4vg",       # 4-乙烯基愈创木酚
    "diacetyl",   # 双乙酰
    "dms",        # DMS
    "t2n",        # E-2-壬烯醛
    "co2",        # CO2
]

ADOPT_RANDOM_TOL = 0.002
ADOPT_GROUP_GAIN = 0.005


def folds(scheme: str, n: int, groups: np.ndarray, seed: int):
    from sklearn.model_selection import GroupKFold, KFold
    if scheme == "random":
        return list(KFold(n_splits=5, shuffle=True, random_state=seed).split(np.arange(n)))
    return list(GroupKFold(n_splits=5).split(np.arange(n), groups=groups))


def fit_base(name: str, Xtr, Ytr, Xte, seed: int):
    return R.fit_predict(name, Xtr, Ytr, Xte, None, seed, quick=False)


def inner_oof(X, Y, styles, scheme, seed):
    """在外层训练集内部生成两个基模型的 OOF 预测。"""
    out = {b: np.full_like(Y, np.nan, dtype=float) for b in BASES}
    for tr, te in folds(scheme, len(X), styles, seed + 1):
        for b in BASES:
            out[b][te] = fit_base(b, X[tr], Y[tr], X[te], seed)
    return out


def r2_cols(y, p):
    ss_res = ((y - p) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    ss_tot = np.where(ss_tot > 1e-12, ss_tot, np.nan)
    return 1.0 - ss_res / ss_tot


def fit_alpha(p1, p2, y):
    """最小二乘拟合 p1 的权重，裁剪到 [0,1] 并向 0.5 收缩一半。"""
    d = (p1 - p2).ravel()
    den = float((d * d).sum())
    if den < 1e-12:
        return 0.5
    a = float(((d * (y - p2).ravel()).sum()) / den)
    a = 0.5 + 0.5 * (min(max(a, 0.0), 1.0) - 0.5)
    return a


def run_scheme(X, Y, styles, scheme, seed, targets):
    n, m = Y.shape
    outer = folds(scheme, n, styles, R.SEED)
    oof = {b: np.full_like(Y, np.nan, dtype=float) for b in BASES}
    for tr, te in outer:
        for b in BASES:
            oof[b][te] = fit_base(b, X[tr], Y[tr], X[te], seed)

    preds = {k: np.full_like(Y, np.nan, dtype=float)
             for k in ("avg", "stack_global", "stack_target", "gate_domain", "gate_cv")}
    domain_vec = np.array([t in DOMAIN_DIRECT for t in targets])
    gate_choice = np.zeros(m, dtype=int)   # 1 = 潜状态

    for tr, te in outer:
        inner = inner_oof(X[tr], Y[tr], styles[tr], scheme, seed)
        pe, pl = inner["et"], inner["latent_sens_ridge"]

        a_g = fit_alpha(pe, pl, Y[tr])
        preds["stack_global"][te] = a_g * oof["et"][te] + (1 - a_g) * oof["latent_sens_ridge"][te]

        for j in range(m):
            a_j = fit_alpha(pe[:, j], pl[:, j], Y[tr][:, j])
            preds["stack_target"][te, j] = (a_j * oof["et"][te, j]
                                            + (1 - a_j) * oof["latent_sens_ridge"][te, j])
            r_e = r2_cols(Y[tr][:, [j]], pe[:, [j]])[0]
            r_l = r2_cols(Y[tr][:, [j]], pl[:, [j]])[0]
            if np.nan_to_num(r_l, nan=-9) > np.nan_to_num(r_e, nan=-9):
                gate_choice[j] += 1
                preds["gate_cv"][te, j] = oof["latent_sens_ridge"][te, j]
            else:
                preds["gate_cv"][te, j] = oof["et"][te, j]

    preds["avg"] = 0.5 * oof["et"] + 0.5 * oof["latent_sens_ridge"]
    preds["gate_domain"] = np.where(domain_vec, oof["et"], oof["latent_sens_ridge"])
    preds.update(oof)
    return preds, oof, gate_choice


def bootstrap_delta(y, pa, pb, n_boot=1000, seed=0):
    """按啤酒重抽样的配对 bootstrap；返回 mean R² 差的 95% 区间。"""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        ra = np.nanmean(r2_cols(y[idx], pa[idx]))
        rb = np.nanmean(r2_cols(y[idx], pb[idx]))
        diffs[b] = ra - rb
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(R.SEEDS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(OUT / "run.log", encoding="utf-8")])
    log = logging.getLogger("composite")
    R.set_global_seed(R.SEED)
    data = R.load_data()
    # 与 Q1 主流程一致：231 个化学特征 + 样本检出率协变量。
    X = np.column_stack([data.X[data.chem_all].to_numpy(float), data.cdr])
    Y = data.Y[data.targets].to_numpy(float)
    styles = data.styles
    targets = data.targets
    log.info("数据：%d×%d → %d 维感官，%d 风格", *X.shape, Y.shape[1], len(set(styles)))

    rows, per_rows = [], []
    store: dict[tuple[str, str, int], np.ndarray] = {}
    choices: dict[str, np.ndarray] = {}
    t0 = time.time()

    for scheme in ("random", "group"):
        for seed in args.seeds:
            ts = time.time()
            preds, oof, gate_choice = run_scheme(X, Y, styles, scheme, seed, targets)
            choices[f"{scheme}_{seed}"] = gate_choice
            for name, p in preds.items():
                r2 = r2_cols(Y, p)
                rho = R._spearman_cols(Y, p)
                rows.append({"variant": name, "scheme": scheme, "seed": seed,
                             "mean_r2": float(np.nanmean(r2)),
                             "median_r2": float(np.nanmedian(r2)),
                             "mean_spearman": float(np.nanmean(rho)),
                             "runtime_s": round(time.time() - ts, 1)})
                for j, t in enumerate(targets):
                    per_rows.append({"variant": name, "scheme": scheme, "seed": seed,
                                     "target": t, "r2": float(r2[j])})
                store[(name, scheme, seed)] = p
            log.info("%-6s seed=%d  ET=%.4f latent=%.4f avg=%.4f gate_cv=%.4f  (%.1fs)",
                     scheme, seed, np.nanmean(r2_cols(Y, preds["et"])),
                     np.nanmean(r2_cols(Y, preds["latent_sens_ridge"])),
                     np.nanmean(r2_cols(Y, preds["avg"])),
                     np.nanmean(r2_cols(Y, preds["gate_cv"])), time.time() - ts)

    met = pd.DataFrame(rows)
    met.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    per = pd.DataFrame(per_rows)
    per.to_csv(OUT / "per_target.csv", index=False, encoding="utf-8-sig")

    seeds = args.seeds
    variants = sorted({k[0] for k in store})
    avg_preds = {(v, sc): np.mean([store[(v, sc, sd)] for sd in seeds], axis=0)
                 for v in variants for sc in ("random", "group")}
    summary, decision_rows = {}, []
    for scheme in ("random", "group"):
        base = avg_preds[("et", scheme)]
        for b in BASES:
            summary[f"{scheme}/{b}"] = {"mean_r2": float(np.nanmean(r2_cols(Y, avg_preds[(b, scheme)])))}
        for v in ("avg", "stack_global", "stack_target", "gate_domain", "gate_cv"):
            p = avg_preds[(v, scheme)]
            r2 = r2_cols(Y, p)
            r2b = r2_cols(Y, base)
            lo, hi = bootstrap_delta(Y, p, base, seed=17)
            from scipy.stats import wilcoxon
            try:
                pval = float(wilcoxon(r2, r2b).pvalue)
            except ValueError:
                pval = 1.0
            summary[f"{scheme}/{v}"] = {
                "mean_r2": float(np.nanmean(r2)),
                "delta_vs_et": float(np.nanmean(r2) - np.nanmean(r2b)),
                "boot_ci95": [lo, hi], "wilcoxon_p": pval,
                "n_targets_better": int(np.sum(r2 > r2b)),
            }
            decision_rows.append({"scheme": scheme, "variant": v,
                                  "mean_r2": float(np.nanmean(r2)),
                                  "delta_vs_et": float(np.nanmean(r2) - np.nanmean(r2b)),
                                  "ci_low": lo, "ci_high": hi, "wilcoxon_p": pval,
                                  "n_better": int(np.sum(r2 > r2b))})

        # oracle 仅作上界参考
        r2e = r2_cols(Y, avg_preds[("et", scheme)])
        r2l = r2_cols(Y, avg_preds[("latent_sens_ridge", scheme)])
        orc = np.where(r2e >= r2l, avg_preds[("et", scheme)],
                       avg_preds[("latent_sens_ridge", scheme)])
        summary[f"{scheme}/oracle_reference"] = {
            "mean_r2": float(np.nanmean(r2_cols(Y, orc))), "note": "只作上界，不可采纳"}

    # 按预先定好的标准选一个复合方案
    best_v, best_gain, adopt = None, -9.9, False
    for v in ("avg", "stack_global", "stack_target", "gate_domain", "gate_cv"):
        r = summary[f"random/{v}"]
        g = summary[f"group/{v}"]
        ok = (r["delta_vs_et"] >= -ADOPT_RANDOM_TOL
              and g["delta_vs_et"] >= ADOPT_GROUP_GAIN
              and g["boot_ci95"][0] > 0)
        if ok and g["delta_vs_et"] > best_gain:
            best_v, best_gain, adopt = v, g["delta_vs_et"], True

    choice_counts = {k: int(np.sum(v > 0)) for k, v in choices.items()}
    report = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds, "bases": BASES, "domain_direct": DOMAIN_DIRECT,
        "thresholds": {"random_tol": ADOPT_RANDOM_TOL, "group_gain": ADOPT_GROUP_GAIN},
        "summary": summary, "decision_rows": decision_rows,
        "gate_cv_latent_chosen_per_fold": choice_counts,
        "gate_cv_total_folds": 5 * len(seeds),
        "adopt": adopt, "adopted_variant": best_v,
        "caveat": ("领域规则在看到 v1 逐维结果后制定，只能算假设；"
                   "无泄漏对照是 gate_cv。oracle 只是上界。"),
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    # G5 融合门：必须同时赢过两个单模型，否则应退回单模型。
    fused = bool(adopt and all(
        summary[f"{sc}/{best_v}"]["mean_r2"] > max(summary[f"{sc}/{b}"]["mean_r2"] for b in BASES)
        for sc in ("random", "group"))) if best_v else False
    (OUT / "fusion.json").write_text(json.dumps({
        "fused_beats_best_single": fused,
        "adopted_variant": best_v,
        "single_models": {f"{sc}/{b}": summary[f"{sc}/{b}"]["mean_r2"]
                          for sc in ("random", "group") for b in BASES},
        "fused": {sc: summary[f"{sc}/{best_v}"]["mean_r2"]
                  for sc in ("random", "group")} if best_v else {},
        "note": "按随机 CV 与分组 CV 同时比较；最终采纳由队员确认。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 中文简报
    lines = ["# 复合架构评估", "",
             f"- 生成：{report['generated']}；种子：{seeds}",
             f"- 基模型：Extra Trees + 8 维 PCA 感官潜状态",
             f"- 采纳标准：随机 CV 不比 ET 低超过 {ADOPT_RANDOM_TOL}；"
             f"分组 CV 比 ET 高至少 {ADOPT_GROUP_GAIN}；配对 bootstrap 95% 区间不含 0。",
             f"- **结论：{'采纳 ' + best_v if adopt else '不采纳复合，主模型仍用 ET'}**", "",
             "| 方案 | 划分 | 平均 R² | 相对 ET | 95% 区间 | 逐维胜出 | Wilcoxon p |",
             "|---|---|---:|---:|---|---:|---:|"]
    for r in decision_rows:
        ci = f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]"
        lines.append(f"| {r['variant']} | {r['scheme']} | {r['mean_r2']:.4f} | "
                     f"{r['delta_vs_et']:+.4f} | {ci} | {r['n_better']}/50 | {r['wilcoxon_p']:.3f} |")
    for scheme in ("random", "group"):
        o = summary[f"{scheme}/oracle_reference"]["mean_r2"]
        lines.append(f"| oracle（仅上界） | {scheme} | {o:.4f} | — | — | — | — |")
    lines += ["", "## 逐维路由", "",
              f"- gate_cv 在多数外层折选择潜状态模型的维度数："
              f"{ {k: v for k, v in choice_counts.items()} }（满分 5 折 × 种子）",
              f"- 领域规则指定的「直接测量」维度：{'、'.join(DOMAIN_DIRECT)}", "",
              "> 读法：如果 gate_cv 与 ET 持平，说明逐维路由没有真实增益，"
              "领域规则再合理也只是叙事；如果 gate_cv 稳定更好，再讨论用领域规则近似它。",
              "", "本文件由 AI 辅助生成；方案取舍由参赛队员决定。"]
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    np.savez_compressed(OUT / "oof_predictions.npz",
                        **{f"{k[0]}_{k[1]}_{k[2]}": v for k, v in store.items()})
    log.info("完成，用时 %.1fs；采纳=%s（%s）", time.time() - t0, adopt, best_v)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
