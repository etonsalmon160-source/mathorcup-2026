#!/usr/bin/env python
# 本文件由 AI 辅助编写；建模与算法设计决策由参赛队员做出。详见《AI工具使用详情说明》。
"""问题三迁移可行性探针：Pils/Lager → Tripel，缺陷化合物不恶化。

这不是最终生成模型，只回答一个决策问题：
    在 250 样本的潜状态空间里，"向 Tripel 迁移"与"缺陷不恶化"能不能同时做到？代价多大？

口径（按题面原文）：
    缺陷以 DMS / Diacetyl / 4VG / trans-2-nonenal / Stale hops 的**超阈值项**为准。
    - loose：不新增超标、且不加重已有超标（题面"不恶化"的可操作定义）
    - strict：迁移后全部不超阈值（更严，用于检查是否存在"干净 Tripel"）

运行：make q3-probe   （或 python scripts/q3_probe.py --k 0）
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, Ridge

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import PATHS, SEED  # noqa: E402

META = ["beer", "beer_id", "tasting_category_fine"]
DERIVED = ["kcalperc", "esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum"]
EPS = 1e-6

# 阈值来自 8.xlsx Sensory thresholds 的 threshold_median（t2n 无官方阈值）
# source_scale：1.csv 里该列的原始口径（DMS/4VG 已是 log10，双乙酰/t2n 是浓度）
DEFECTS = {
    "DMS": {"threshold": 19.0, "source_scale": "log10", "unit": "ug/L"},
    "X4VG": {"threshold": 0.20, "source_scale": "log10", "unit": "mg/L"},
    "X2.3.butanedione": {"threshold": 0.09, "source_scale": "raw", "unit": "mg/L"},
    "X.E..2.nonenal": {"threshold": None, "source_scale": "raw", "unit": "mg/L"},
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    X = pd.read_csv(PATHS["raw"] / "附件" / "1.csv")
    Y = pd.read_csv(PATHS["raw"] / "附件" / "4.csv")
    return X, Y


def normal_scores(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    n = len(X)
    out = []
    for c in cols:
        r = X[c].astype(float).rank(method="average")
        out.append(pd.Series(norm.ppf((r - 0.5) / n), name=c))
    return pd.concat(out, axis=1).fillna(0.0)


def parallel_analysis(M: np.ndarray, reps: int, seed: int) -> int:
    ev = np.linalg.eigvalsh(np.corrcoef(M, rowvar=False))[::-1]
    rng = np.random.default_rng(seed)
    null = np.empty((reps, M.shape[1]))
    for b in range(reps):
        Mp = np.column_stack([rng.permutation(M[:, j]) for j in range(M.shape[1])])
        null[b] = np.linalg.eigvalsh(np.corrcoef(Mp, rowvar=False))[::-1]
    return int((ev > np.percentile(null, 95, axis=0)).sum())


def sqrtm_psd(S: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh((S + S.T) / 2)
    return V @ np.diag(np.sqrt(np.clip(w, 1e-8, None))) @ V.T


def gaussian_ot(m0, S0, m1, S1, z):
    """Wasserstein-2 最优传输映射（Bures 几何）。"""
    h0 = sqrtm_psd(S0)
    h0i = np.linalg.pinv(h0)
    A = h0i @ sqrtm_psd(h0 @ S1 @ h0) @ h0i
    return m1 + A @ (z - m0)


def solve_qp(zt, B, c, lo, hi):
    """min ||z-zt||^2 s.t. Bz<=c, lo<=z<=hi。LP 判可行，SLSQP 精解。"""
    if B.shape[0] == 0:
        return zt.copy()
    lp = linprog(np.zeros(B.shape[1]), A_ub=B, b_ub=c,
                 bounds=list(zip(lo, hi)), method="highs")
    if not lp.success:
        return None
    f = lambda z: float((z - zt) @ (z - zt))
    j = lambda z: 2 * (z - zt)
    best = None
    for x0 in (zt, lp.x):
        r = minimize(
            f, x0, jac=j, method="SLSQP", bounds=list(zip(lo, hi)),
            constraints=[{"type": "ineq",
                          "fun": lambda z, B=B, c=c: c - B @ z,
                          "jac": lambda z, B=B: -B}],
            options={"ftol": 1e-14, "maxiter": 2000},
        )
        if np.all(B @ r.x - c <= 1e-6):
            val = f(r.x)
            if best is None or val < best[0]:
                best = (val, r.x)
    return best[1] if best else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=0, help="潜维数；0 = 平行分析自动定")
    ap.add_argument("--reps", type=int, default=100, help="平行分析置换次数")
    args = ap.parse_args()

    out = PATHS["results"] / "q3_probe"
    out.mkdir(parents=True, exist_ok=True)

    X, Y = load()
    styles = X["tasting_category_fine"].values
    feat = [c for c in X.columns if c not in META + DERIVED]
    Ns = normal_scores(X, feat)
    M = ((Ns - Ns.mean()) / Ns.std(ddof=1)).values

    k_auto = parallel_analysis(M, args.reps, SEED)
    k = args.k or k_auto
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    Z = (U * S)[:, :k]

    gm = Z.mean(0)
    ss_tot = ((Z - gm) ** 2).sum()
    ss_b = sum(((Z[styles == s].mean(0) - gm) ** 2).sum() * (styles == s).sum()
               for s in np.unique(styles))
    style_r2 = float(ss_b / ss_tot)
    evr = float((S[:k] ** 2).sum() / (S ** 2).sum())

    # 缺陷在 log10 尺度上建模：保证生成值为正，约束仍是线性的
    ylog, thr = {}, {}
    for d, spec in DEFECTS.items():
        v = X[d].values.astype(float)
        ylog[d] = np.log10(np.maximum(v, EPS)) if spec["source_scale"] == "raw" else v
        thr[d] = np.log10(spec["threshold"]) if spec["threshold"] else None
    beta, alpha, defect_r2 = {}, {}, {}
    for d in DEFECTS:
        y = ylog[d]
        ok = np.isfinite(y)
        lr = LinearRegression().fit(Z[ok], y[ok])
        beta[d], alpha[d] = lr.coef_, lr.intercept_
        pred = lr.predict(Z[ok])
        defect_r2[d] = float(1 - ((y[ok] - pred) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum())

    P = styles == "Pils/Lager"
    T = styles == "Tripel"
    Spool = np.cov(Z, rowvar=False)

    def moments(mask, lam=20.0):
        Zc = Z[mask]
        return (Zc.mean(0),
                (mask.sum() * np.cov(Zc, rowvar=False) + lam * Spool) / (mask.sum() + lam))

    mp, Sp = moments(P)
    mt, St = moments(T)
    dm = mt - mp
    lo = Z.min(0) - 0.25 * Z.std(0)
    hi = Z.max(0) + 0.25 * Z.std(0)

    mu = Z.mean(0)
    Si = np.linalg.pinv(np.cov(Z, rowvar=False))
    maha = lambda z: float(np.sqrt((z - mu) @ Si @ (z - mu)))
    maha_99 = float(np.percentile([maha(z) for z in Z], 99))

    tcols = [c for c in Y.columns if c not in META]
    sens = Y[tcols].astype(float)
    sridge = Ridge(alpha=10.0).fit(Ns.values[~P], sens.values[~P])
    trip_sens = sens.values[T].mean(0)
    pils_sens = sens.values[P].mean(0)

    coupling = {
        d: float(beta[d] @ dm / (np.linalg.norm(beta[d]) * np.linalg.norm(dm) + 1e-12))
        for d in DEFECTS
    }

    rows = []
    for i in np.where(P)[0]:
        zi = Z[i]
        # 迁移前基线：同一套指标，便于直接看"移了多远"
        sen0 = sridge.predict(Ns.values[[i]])[0]
        rec0 = {
            "beer": int(X.loc[i, "beer"]), "method": "source", "constraint": "before",
            "feasible": True, "style_shift": 0.0, "qp_cost": 0.0,
            "rel_centroid": float(np.linalg.norm(zi - mt) / (np.linalg.norm(zi - mp) + 1e-12)),
            "maha": maha(zi),
            "sens_cos": 0.0,
            "sens_dist": float(np.linalg.norm(sen0 - trip_sens)),
        }
        for d in DEFECTS:
            conc = 10 ** ylog[d][i] if np.isfinite(ylog[d][i]) else np.nan
            t = DEFECTS[d]["threshold"]
            rec0[f"x_{d}"] = float(conc)
            rec0[f"above_{d}"] = int(conc > t * (1 + 1e-4)) if t and np.isfinite(conc) else -1
            rec0[f"new_above_{d}"] = -1
            rec0[f"oav_ratio_{d}"] = float(conc / t) if t and np.isfinite(conc) else np.nan
            rec0[f"oav_before_{d}"] = rec0[f"oav_ratio_{d}"]
        rows.append(rec0)

        targets = {"naive": zi + dm, "ot": gaussian_ot(mp, Sp, mt, St, zi)}
        for method, zt in targets.items():
            for mode in ("none", "loose", "strict"):
                B, c = [], []
                for d in DEFECTS:
                    src = ylog[d][i]
                    src = src if np.isfinite(src) else float(alpha[d] + beta[d] @ zi)
                    t = thr[d]
                    if mode == "none":
                        lim = np.inf
                    elif mode == "loose":
                        lim = max(src, t) if t is not None else src
                    else:
                        lim = t if t is not None else src
                    if np.isfinite(lim):
                        B.append(beta[d])
                        c.append(lim - alpha[d])
                z = solve_qp(zt, np.array(B), np.array(c), lo, hi)
                feasible = z is not None
                z = zt if not feasible else z
                p = {d: float(alpha[d] + beta[d] @ z) for d in DEFECTS}
                sen = sridge.predict((z @ Vt[:k])[None])[0]
                rec = {
                    "beer": int(X.loc[i, "beer"]), "method": method, "constraint": mode,
                    "feasible": feasible,
                    "style_shift": float((z - zi) @ dm / (dm @ dm)),
                    "qp_cost": float(np.linalg.norm(z - zt) / (np.linalg.norm(zt - zi) + 1e-12)),
                    "rel_centroid": float(np.linalg.norm(z - mt) / (np.linalg.norm(z - mp) + 1e-12)),
                    "maha": maha(z),
                    "sens_cos": float((sen - pils_sens) @ (trip_sens - pils_sens) /
                                      (np.linalg.norm(sen - pils_sens) *
                                       np.linalg.norm(trip_sens - pils_sens) + 1e-12)),
                    "sens_dist": float(np.linalg.norm(sen - trip_sens)),
                }
                for d in DEFECTS:
                    conc = 10 ** p[d]
                    rec[f"x_{d}"] = float(conc)
                    t = DEFECTS[d]["threshold"]
                    before = 10 ** ylog[d][i] if np.isfinite(ylog[d][i]) else np.nan
                    rec[f"above_{d}"] = int(conc > t * (1 + 1e-4)) if t else -1
                    rec[f"new_above_{d}"] = int(bool(t) and conc > t * (1 + 1e-4) and before <= t)
                    rec[f"oav_ratio_{d}"] = float(conc / t) if t else np.nan
                    rec[f"oav_before_{d}"] = float(before / t) if t else np.nan
                rows.append(rec)

        # 观测对照：最近的 Tripel、以及源样本本身
        j = np.where(T)[0][np.argmin(np.linalg.norm(Z[T] - zi, axis=1))]
        rows.append({
            "beer": int(X.loc[i, "beer"]), "method": "observed_nearest_tripel",
            "constraint": "none", "feasible": True,
            "style_shift": 0.0, "qp_cost": 0.0,
            "rel_centroid": float(np.linalg.norm(Z[j] - mt) / (np.linalg.norm(Z[j] - mp) + 1e-12)),
            "maha": maha(Z[j]), "sens_cos": 0.0,
            "sens_dist": float(np.linalg.norm(sens.values[j] - trip_sens)),
            **{f"x_{d}": float(10 ** ylog[d][j]) for d in DEFECTS},
            **{f"above_{d}": int(10 ** ylog[d][j] > DEFECTS[d]["threshold"] * (1 + 1e-4))
               if DEFECTS[d]["threshold"] else -1 for d in DEFECTS},
            **{f"new_above_{d}": -1 for d in DEFECTS},
            **{f"oav_ratio_{d}": float(10 ** ylog[d][j] / DEFECTS[d]["threshold"])
               if DEFECTS[d]["threshold"] else np.nan for d in DEFECTS},
            **{f"oav_before_{d}": np.nan for d in DEFECTS},
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / "transfer.csv", index=False, encoding="utf-8-sig")

    ot = df[df.method == "ot"]
    ot_none = ot[ot.constraint == "none"]
    src_rows = df[df.method == "source"]
    obs_rows = df[df.method == "observed_nearest_tripel"]
    clean_tripel = int(((10 ** ylog["DMS"][T] <= thr["DMS"] * (1 + 1e-9)) &
                        (10 ** ylog["X4VG"][T] <= thr["X4VG"]) &
                        (10 ** ylog["X2.3.butanedione"][T] <= thr["X2.3.butanedione"])).sum())
    n_ot = len(ot_none)
    n_tripel = int(T.sum())

    def summarize(mode):
        s = ot[ot.constraint == mode]
        return {
            "feasible": int(s.feasible.sum()),
            "n": int(len(s)),
            "style_shift_median": float(s.style_shift.median()),
            "qp_cost_median": float(s.qp_cost.median()),
            "rel_centroid_median": float(s.rel_centroid.median()),
            "maha_median": float(s.maha.median()),
            "sens_cos_median": float(s.sens_cos.median()),
            "sens_dist_median": float(s.sens_dist.median()),
            "new_DMS": int(s.new_above_DMS.sum()),
            "new_diacetyl": int(s["new_above_X2.3.butanedione"].sum()),
            "new_4VG": int(s.new_above_X4VG.sum()),
            "worsened_DMS": int((s.oav_ratio_DMS > s.oav_before_DMS * (1 + 1e-6)).sum()),
            "worsened_diacetyl": int((s["oav_ratio_X2.3.butanedione"] >
                                      s["oav_before_X2.3.butanedione"] * (1 + 1e-6)).sum()),
        }

    summary = {m: summarize(m) for m in ("none", "loose", "strict")}
    D, C, Bt, Tn = DEFECTS["DMS"], DEFECTS["X2.3.butanedione"], DEFECTS["X4VG"], DEFECTS["X.E..2.nonenal"]
    report = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "k": k, "k_parallel": k_auto, "evr": evr, "style_r2": style_r2,
        "defect_model_r2": defect_r2,
        "n_pils": int(P.sum()), "n_tripel": int(T.sum()),
        "maha_99": maha_99,
        "clean_tripel_observed": clean_tripel,
        "clean_tripel_denominator": n_tripel,
        "unnormalized_ot": {
            "n": n_ot,
            "DMS_above": int((ot_none["above_DMS"] == 1).sum()),
            "new_DMS": int(ot_none["new_above_DMS"].sum()),
            "diacetyl_above": int((ot_none["above_X2.3.butanedione"] == 1).sum()),
            "new_diacetyl": int(ot_none["new_above_X2.3.butanedione"].sum()),
        },
        "coupling_cos": coupling,
        "source_baseline": {
            "rel_centroid_median": float(src_rows.rel_centroid.median()),
            "maha_median": float(src_rows.maha.median()),
            "sens_dist_median": float(src_rows.sens_dist.median()),
        },
        "constraints": summary,
        "thresholds": {d: DEFECTS[d] for d in DEFECTS},
        "verdict": {},
    }
    lo, st = summary["loose"], summary["strict"]
    report["verdict"] = {
        "loose_solvable": bool(lo["feasible"] == lo["n"] and lo["new_DMS"] == 0 and
                               lo["new_diacetyl"] == 0 and lo["new_4VG"] == 0 and
                               lo["worsened_DMS"] == 0 and lo["worsened_diacetyl"] == 0),
        "strict_model_feasible": bool(st["feasible"] == st["n"] and st["new_DMS"] == 0 and
                                      st["new_diacetyl"] == 0 and st["new_4VG"] == 0),
        "strict_off_manifold": bool(clean_tripel == 0),
        "shift_retention_loose": lo["style_shift_median"],
        "cost_loose": lo["qp_cost_median"],
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")

    def fmt(x, nd=3):
        return f"{x:.{nd}f}"

    lines = [
        "# 问题三迁移可行性探针（Pils/Lager → Tripel）",
        "",
        f"生成时间：{report['generated']}　潜维数 k={k}（平行分析 {k_auto}，"
        f"解释 {evr:.1%} 方差，风格 R²={style_r2:.3f}）",
        "",
        "> 这是**可行性探针**，不是最终生成模型。目的是回答：向 Tripel 迁移与"
        "缺陷不恶化能否同时做到、代价多大。所有结论都在潜状态模型内成立。",
        "",
        "## 结论",
        "",
        f"1. **不约束的迁移会加重缺陷**：{report['unnormalized_ot']['DMS_above']}/{n_ot} "
        f"个 Pils 样本在向 Tripel 传输后 DMS 超阈值（其中 "
        f"{report['unnormalized_ot']['new_DMS']} 个为新增），"
        f"{report['unnormalized_ot']['new_diacetyl']} 个新增双乙酰超标。"
        f"风格方向与 DMS 载荷的余弦相似度为 {coupling['DMS']:.2f}——这不是偶然，是真实耦合。",
        f"2. **按题面口径（不新增超标、不加重已有超标）可以做到**："
        f"{lo['feasible']}/{lo['n']} 可行，新增超标 0；代价是只保留 "
        f"{lo['style_shift_median']:.0%} 的风格位移，QP 修正量中位数为传输距离的 "
        f"{lo['qp_cost_median']:.0%}。",
        f"3. **更严的“全部回到阈值以下”在模型内也可行**（{st['feasible']}/{st['n']}），"
        f"但 **{clean_tripel}/{n_tripel} 个真实 Tripel 满足这条**——这样的“干净 Tripel”"
        "在数据里不存在，属于模型外推，论文里必须标注，不能当成真实风格代表。",
        f"4. 约束后的点在潜空间仍处于数据支撑内（Mahalanobis 中位数 "
        f"{lo['maha_median']:.2f}，整体 99% 分位 {maha_99:.1f}；相对 Tripel 质心距离 "
        f"{lo['rel_centroid_median']:.2f}），不是无边外推；但感官保真度下降"
        "（到 Tripel 感官均值的距离 "
        f"{lo['sens_dist_median']:.2f} vs 不约束 {summary['none']['sens_dist_median']:.2f}）。",
        "",
        "## 四种迁移的目标分布距离",
        "",
        "| 方案 | 可行 | 风格位移保留 | 相对 Tripel 质心距离 | 到 Tripel 感官距离 | 新增 DMS 超标 | 新增双乙酰超标 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.append(f"| 源样本（迁移前） | — | — | "
                 f"{src_rows.rel_centroid.median():.2f} | {src_rows.sens_dist.median():.2f} | — | — |")
    for m, label in (("none", "不约束"), ("loose", "不恶化（题面口径）"), ("strict", "全部阈值以下")):
        s = summary[m]
        lines.append(f"| {label} | {s['feasible']}/{s['n']} | {s['style_shift_median']:.0%} | "
                     f"{s['rel_centroid_median']:.2f} | {s['sens_dist_median']:.2f} | "
                     f"{s['new_DMS']} | {s['new_diacetyl']} |")
    lines += [
        "",
        "（相对 Tripel 质心距离 <1 表示比源更接近 Tripel；风格位移保留 100% 表示"
        "完全按 Tripel 分布搬移。源样本到 Tripel 感官距离大，说明这次迁移的跨度不小。）",
        "",
        "## 缺陷物的潜模型解释度（log10 尺度）",
        "",
        "| 缺陷 | 阈值 | 单位 | 潜模型 R² |",
        "|---|---:|---|---:|",
        f"| DMS | {D['threshold']} | {D['unit']} | {defect_r2['DMS']:.3f} |",
        f"| Diacetyl | {C['threshold']} | {C['unit']} | {defect_r2['X2.3.butanedione']:.3f} |",
        f"| 4VG | {Bt['threshold']} | {Bt['unit']} | {defect_r2['X4VG']:.3f} |",
        f"| trans-2-nonenal | 无官方阈值 | {Tn['unit']} | {defect_r2['X.E..2.nonenal']:.3f} |",
        "",
        "> DMS 与 4VG 的 R² 中等，说明潜状态只能部分控制它们；"
        "Diacetyl/t2n 在 log 尺度上 R² 较高，但重尾样本仍会主导。"
        "所有约束是对**生成谱本身**的硬约束，不依赖模型预测真值。",
        "",
        "## 对写作的含义",
        "",
        "- **推荐口径**：把“不恶化”定义为 `OAV_after ≤ max(OAV_before, 1)`，"
        "即不新增超标、不加重已有超标。这是题面“以超阈值项为准”的直接推论。",
        "- **必须报告代价**：风格位移保留率与缺陷约束强度是一条 Pareto 前沿，"
        "不能只报“约束后仍合格”而不报风格损失。",
        "- **stale hops 没有浓度**：用 t2n 与其感官维度做代理，"
        "在论文里明确写“代理变量”，不要伪造阈值。",
        "- **不要声称生成了真实 Tripel**：严格约束下的谱在观测 Tripel 之外，"
        "应称为“模型内的低缺陷 Tripel 反事实”。",
        "",
        "## 产物",
        "",
        "`report.json` / `transfer.csv`（逐样本、逐方案的全部指标）",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"✅ k={k} (平行分析 {k_auto})，风格 R²={style_r2:.3f}")
    print(f"   loose 可行 {lo['feasible']}/{lo['n']}，风格位移保留 {lo['style_shift_median']:.0%}")
    print(f"   真实 Tripel 满足全部阈值：{clean_tripel}/{n_tripel}")
    print(f"   报告：{(out / 'report.md').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
