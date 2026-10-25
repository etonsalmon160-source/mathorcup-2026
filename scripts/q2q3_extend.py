"""Q2 稳定性敏感性 + Q3 全风格对迁移矩阵。

AI 辅助声明：本脚本由 AI 辅助编写；指标口径与论文取舍由参赛队员决定。

两个问题
--------
Q2：首版中位 Jaccard 正好 0.500，贴着门控线。这里回答两件事：
    1. 换 Lasso 正则强度（0.04–0.10），Top-3 与稳定性是否稳定；
    2. 把 Bootstrap 从 100 次加到 300 次，中位 Jaccard 落在哪。
    只做敏感性报告，**不改主口径**。

Q3：首版只验证了 Pils/Lager → Tripel 一条路径。这里把 22×21 个有向风格对
    全部跑一遍，看在统一口径（松/严）下哪些迁移可行、代价多大，
    并检验「风格越远越难」是否成立。

资源：单进程，BLAS 限 4 线程，内存 < 2 GB；可与 8 线程的复合架构评估并行。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v1 as R  # noqa: E402

Q2_OUT = R.RESULTS / "q2_sensitivity"
Q3_OUT = R.RESULTS / "q3_pairs"


# ------------------------------------------------------------------ Q2

def q2_sensitivity(data):
    from scipy.stats import spearmanr  # noqa: F401  (保持与主流程同依赖)
    Q2_OUT.mkdir(parents=True, exist_ok=True)
    cand = [c for c in data.chem_nd if c not in R.Q2_EXCLUDE]
    Xc = data.X[cand].to_numpy(float)
    Y = data.Y[data.targets].to_numpy(float)
    A = R._impute_scale(Xc, Xc)[0]
    rng = np.random.default_rng(R.SEED)
    style_idx = {s: np.where(data.styles == s)[0] for s in np.unique(data.styles)}
    n_boot = 100
    reps = [np.concatenate([rng.choice(v, size=len(v), replace=True)
                            for v in style_idx.values()]) for _ in range(n_boot)]

    def top3(freq):
        return [set(np.argsort(-freq[:, j])[:3]) for j in range(freq.shape[1])]

    def stability(sel, freq):
        med, unstable, rates = [], 0, []
        for j in range(freq.shape[1]):
            js = []
            for _ in range(5):
                perm = rng.permutation(n_boot)
                h1, h2 = perm[:n_boot // 2], perm[n_boot // 2:]
                s1 = set(np.argsort(-sel[j][:, h1].mean(1))[:3])
                s2 = set(np.argsort(-sel[j][:, h2].mean(1))[:3])
                js.append(len(s1 & s2) / len(s1 | s2))
            med.append(float(np.median(js)))
            sr = float(np.mean(freq[np.argsort(-freq[:, j])[:3], j]))
            rates.append(sr)
            unstable += int(sr < 0.6)
        return med, unstable, rates

    rows, per_dim, tops = [], [], {}
    t0 = time.time()
    for alpha in (0.04, 0.06, 0.08, 0.10):
        sel, coef = R._lasso_select(A, Y, reps, alpha=alpha)
        freq = sel.mean(2).T
        med, unstable, rates = stability(sel, freq)
        tops[alpha] = top3(freq)
        rows.append({"alpha": alpha, "n_boot": n_boot,
                     "median_jaccard": float(np.median(med)),
                     "mean_jaccard": float(np.mean(med)),
                     "n_unstable": unstable,
                     "mean_stable_rate": float(np.mean(rates))})
        for j, t in enumerate(data.targets):
            per_dim.append({"alpha": alpha, "target": t,
                            "jaccard_median": med[j], "stable_rate": rates[j]})
        print(f"Q2 alpha={alpha:.2f} 中位Jaccard={np.median(med):.3f} "
              f"不稳定={unstable} ({time.time()-t0:.0f}s)")

    # Bootstrap 加大到 300，只跑主口径
    n_boot_big = 300
    reps_big = [np.concatenate([rng.choice(v, size=len(v), replace=True)
                                for v in style_idx.values()])
                for _ in range(n_boot_big)]
    sel_b, _ = R._lasso_select(A, Y, reps_big, alpha=R.Q2_ALPHA)
    freq_b = sel_b.mean(2).T
    js_big = []
    for j in range(freq_b.shape[1]):
        for _ in range(5):
            perm = rng.permutation(n_boot_big)
            h1, h2 = perm[:n_boot_big // 2], perm[n_boot_big // 2:]
            s1 = set(np.argsort(-sel_b[j][:, h1].mean(1))[:3])
            s2 = set(np.argsort(-sel_b[j][:, h2].mean(1))[:3])
            js_big.append(len(s1 & s2) / len(s1 | s2))
    big = {"alpha": R.Q2_ALPHA, "n_boot": n_boot_big,
           "median_jaccard": float(np.median(js_big)),
           "mean_jaccard": float(np.mean(js_big))}

    # Top-3 在不同正则强度下的重合度
    overlap = []
    base = tops[R.Q2_ALPHA]
    for alpha in (0.04, 0.08, 0.10):
        ov = [len(base[j] & tops[alpha][j]) / len(base[j] | tops[alpha][j])
              for j in range(len(base))]
        overlap.append({"alpha": alpha, "top3_jaccard_vs_main": float(np.mean(ov))})

    pd.DataFrame(rows).to_csv(Q2_OUT / "alpha_sensitivity.csv", index=False,
                              encoding="utf-8-sig")
    pd.DataFrame(per_dim).to_csv(Q2_OUT / "per_dimension.csv", index=False,
                                 encoding="utf-8-sig")
    pd.DataFrame(overlap).to_csv(Q2_OUT / "top3_overlap.csv", index=False,
                                 encoding="utf-8-sig")
    (Q2_OUT / "big_bootstrap.json").write_text(
        json.dumps(big, ensure_ascii=False, indent=2), encoding="utf-8")
    out = {"alpha_sensitivity": rows, "big_bootstrap": big,
           "top3_overlap": overlap, "runtime_s": round(time.time() - t0, 1)}
    (Q2_OUT / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    return out


# ------------------------------------------------------------------ Q3

def q3_pairs(data):
    Q3_OUT.mkdir(parents=True, exist_ok=True)
    from sklearn.linear_model import Ridge

    Xdf = data.X[data.chem_nd]
    X = Xdf.to_numpy(float)
    Y = data.Y[data.targets].to_numpy(float)
    styles = data.styles
    style_names = sorted(np.unique(styles))
    n, p = X.shape
    k = R.LATENT_K

    Ns = R._normal_scores_matrix(Xdf, data.chem_nd)
    mu_ns, sd_ns = Ns.mean(0), Ns.std(0, ddof=1)
    sd_ns = np.where(sd_ns > 0, sd_ns, 1.0)
    M = (Ns - mu_ns) / sd_ns
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    Z = U[:, :k] * S[:k]
    Spool = np.cov(Z, rowvar=False)
    mu = Z.mean(0)
    Si = np.linalg.pinv(np.cov(Z, rowvar=False))
    maha = lambda z: float(np.sqrt((z - mu) @ Si @ (z - mu)))
    raw_tables = R._raw_quantile_tables(data.X, data.chem_nd)
    xmin, xmax = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
    xsd = np.nanstd(X, axis=0, ddof=1)
    xsd = np.where(xsd > 0, xsd, 1.0)
    col = {c: j for j, c in enumerate(data.chem_nd)}

    Xall = data.X[data.chem_all].to_numpy(float)
    A_all = R._impute_scale(Xall, Xall)[0]
    sens = Ridge(alpha=10.0).fit(A_all, Y)

    def defect_thr(d):
        spec = R.DEFECTS[d]
        thr = spec["threshold"]
        return None if thr is None else (
            float(np.log10(thr)) if spec["scale"] == "log10" else float(thr))

    def bound(d, xs, mode):
        thr = defect_thr(d)
        if mode == "none" or thr is None:
            return np.inf if mode == "none" else xs
        return max(xs, thr) if mode == "loose" else thr

    style_moments = {s: R._moments(Z, styles == s, Spool) for s in style_names}
    rows = []
    t0 = time.time()
    for si, s in enumerate(style_names):
        mp, Sp = style_moments[s]
        src_idx = np.where(styles == s)[0]
        for ti, t in enumerate(style_names):
            if si == ti:
                continue
            mt, St = style_moments[t]
            dm = mt - mp
            tgt_mean = Y[styles == t].mean(0)
            src_mean = Y[styles == s].mean(0)
            recs = {m: [] for m in ("none", "loose", "strict")}
            for i in src_idx:
                zi = Z[i]
                zot = R._gaussian_ot(mp, Sp, mt, St, zi)
                x_ot = R._ns_to_raw(((zot @ Vt[:k]) * sd_ns + mu_ns)[None, :],
                                    raw_tables)[0]
                x_src = X[i].copy()
                for mode in recs:
                    x = (x_ot + 0.10 * x_src) / 1.10
                    for d in R.DEFECTS:
                        j = col[d]
                        xs = x_src[j]
                        if not np.isfinite(xs):
                            thr = R.DEFECTS[d]["threshold"]
                            xs = thr if thr is not None else np.nanmedian(X[:, j])
                        b = bound(d, xs, mode)
                        if np.isfinite(b):
                            x[j] = min(x[j], b)
                    x = np.clip(x, xmin, xmax)
                    zns = R._raw_to_ns(x[None, :], raw_tables, n)[0]
                    zl = ((zns - mu_ns) / sd_ns) @ Vt[:k].T
                    x_full = np.full(len(data.chem_all), np.nan)
                    for jj, c in enumerate(data.chem_all):
                        x_full[jj] = x[col[c]] if c in col else float(data.X[c].mean())
                    sen = sens.predict(R._impute_scale(Xall, x_full[None, :])[1])[0]
                    ok = True
                    new = {}
                    for d in R.DEFECTS:
                        j = col[d]
                        val, before = float(x[j]), float(x_src[j])
                        thr = defect_thr(d)
                        # 阈值可能为负（log10 特征），容差必须加性。
                        isnew = int(thr is not None and val > thr + 1e-6
                                    and before <= thr + 1e-6)
                        new[d] = isnew
                        if mode == "strict" and thr is not None and val > thr + 1e-6:
                            ok = False
                        if mode == "loose" and isnew:
                            ok = False
                        if mode in ("loose", "strict") and d == "X.E..2.nonenal" \
                                and val > before + 1e-6:
                            ok = False
                    recs[mode].append({
                        "feasible": float(ok),
                        "style_shift": float((zl - zi) @ dm / (dm @ dm)),
                        "qp_cost": float(np.linalg.norm((x - x_ot) / xsd) /
                                         (np.linalg.norm((x_ot - x_src) / xsd) + 1e-12)),
                        "maha": maha(zl),
                        "sens_cos": float((sen - src_mean) @ (tgt_mean - src_mean) /
                                          (np.linalg.norm(sen - src_mean) *
                                           np.linalg.norm(tgt_mean - src_mean) + 1e-12)),
                        "sens_dist": float(np.linalg.norm(sen - tgt_mean)),
                        "new_DMS": new["DMS"],
                        "new_diacetyl": new["X2.3.butanedione"],
                        "new_4VG": new["X4VG"],
                    })
            for mode in recs:
                d = pd.DataFrame(recs[mode])
                rows.append({
                    "source": s, "target": t, "mode": mode, "n": len(d),
                    "feasible_rate": float(d.feasible.mean()),
                    "style_shift_median": float(d.style_shift.median()),
                    "qp_cost_median": float(d.qp_cost.median()),
                    "maha_median": float(d.maha.median()),
                    "sens_cos_median": float(d.sens_cos.median()),
                    "new_DMS": int(d.new_DMS.sum()),
                    "new_diacetyl": int(d.new_diacetyl.sum()),
                    "new_4VG": int(d.new_4VG.sum()),
                })
        print(f"Q3 源风格 {s} 完成 ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(Q3_OUT / "pairwise.csv", index=False, encoding="utf-8-sig")

    from scipy.stats import spearmanr
    dist = pd.DataFrame(index=style_names, columns=style_names, dtype=float)
    for s in style_names:
        mp = style_moments[s][0]
        for t in style_names:
            dist.loc[s, t] = float(np.linalg.norm(style_moments[t][0] - mp))
    dist.to_csv(Q3_OUT / "latent_distance.csv", encoding="utf-8-sig")

    summ = {}
    for mode in ("none", "loose", "strict"):
        sub = df[df["mode"] == mode]
        summ[mode] = {
            "feasible_rate_overall": float(np.average(sub.feasible_rate, weights=sub.n)),
            "n_pairs_all_feasible": int((sub.feasible_rate == 1).sum()),
            "n_pairs_none_feasible": int((sub.feasible_rate == 0).sum()),
            "style_shift_median": float(sub.style_shift_median.median()),
            "qp_cost_median": float(sub.qp_cost_median.median()),
            "new_DMS_total": int(sub.new_DMS.sum()),
            "new_diacetyl_total": int(sub.new_diacetyl.sum()),
            "new_4VG_total": int(sub.new_4VG.sum()),
            "sens_cos_median": float(sub.sens_cos_median.median()),
            "maha_median": float(sub.maha_median.median()),
        }
    sub = df[df["mode"] == "loose"].copy()
    sub["latent_distance"] = [dist.loc[a, b] for a, b in zip(sub.source, sub.target)]
    rho_cost = spearmanr(sub.latent_distance, sub.qp_cost_median).statistic
    rho_shift = spearmanr(sub.latent_distance, sub.style_shift_median).statistic
    per_target = (df[df["mode"] == "loose"]
                  .groupby("target")["feasible_rate"].mean().sort_values())
    per_source = (df[df["mode"] == "loose"]
                  .groupby("source")["feasible_rate"].mean().sort_values())
    per_target.to_csv(Q3_OUT / "per_target_style.csv", encoding="utf-8-sig")
    per_source.to_csv(Q3_OUT / "per_source_style.csv", encoding="utf-8-sig")

    report = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
              "n_style_pairs": int(len(df) / 3), "modes": summ,
              "distance_vs_cost_spearman": float(rho_cost),
              "distance_vs_shift_spearman": float(rho_shift),
              "hardest_targets": per_target.head(5).to_dict(),
              "easiest_targets": per_target.tail(5).to_dict(),
              "runtime_s": round(time.time() - t0, 1)}
    (Q3_OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    return report


def main() -> int:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    R.set_global_seed(R.SEED)
    data = R.load_data()
    q2 = q2_sensitivity(data)
    q3 = q3_pairs(data)
    print(json.dumps({"q2": q2, "q3": q3}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
