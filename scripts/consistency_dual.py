#!/usr/bin/env python
# 本文件由 AI 辅助编写；统计口径与采用与否由参赛队员决定。
"""Q2↔Q3 一致性的双轨复算（只写 results/supplementary/）。

主结果 results/q2/evidence_scores.csv 是「正向 Lasso」的入选频率。为了判断
一致性结论是否依赖“只保留正向效应”这一选择，这里在完全相同的 Bootstrap
分层、样本权重、候选池与图平滑算子上，再跑一遍有符号 Lasso（positive=False），
两种参与度分别与 Q3 的化合物空间载荷做逐维 Spearman。

输出
----
results/supplementary/consistency_dual.json
results/supplementary/consistency_dual_per_dim.csv
results/supplementary/consistency_dual_freq.npz  （有符号入选频率，50×226）

不修改 results/q1–q3 或 results/composite。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_v1 as R  # noqa: E402
import upgrade_probe as U  # noqa: E402

OUT = R.RESULTS / "supplementary"
OUT.mkdir(parents=True, exist_ok=True)
TOP_K = 20


def _signed_freq(data, s, cache: Path | None = None):
    """与 Q2 主流程同一套重采样，只把 positive=True 改成 False。"""
    if cache is not None and cache.exists():
        z = np.load(cache)
        return z["freq"]
    n = s["n"]
    Xc_qc = np.column_stack([s["Xc"], data.cdr])
    Ain_all = R._impute_scale(Xc_qc, Xc_qc)[0]
    A_adj_all = R._center_by_group(Ain_all, data.styles)
    Y_adj = R._center_by_group(s["Y"], data.styles)
    rng = np.random.default_rng(R.SEED)
    style_idx = {g: np.where(data.styles == g)[0]
                 for g in np.unique(data.styles)}
    reps = [np.concatenate([rng.choice(v, size=len(v), replace=True)
                            for v in style_idx.values()])
            for _ in range(R.Q2_BOOTSTRAP)]
    sample_w = np.where(data.floor_heavy, 0.5, 1.0)
    is_aroma = np.array([R._is_aroma_target(t) for t in data.targets])
    vol_mask = np.r_[np.array([c not in R.Q2_NONVOLATILE for c in s["cand"]]),
                     True]
    full_mask = np.ones(n + 1, dtype=bool)
    masks = [vol_mask if a else full_mask for a in is_aroma]
    idx_ar = np.flatnonzero(is_aroma)
    idx_tp = np.flatnonzero(~is_aroma)
    sel = np.zeros((len(data.targets), n + 1, len(reps)), dtype=bool)
    sel[idx_ar], _ = R._lasso_select(
        A_adj_all, Y_adj[:, idx_ar], reps,
        masks=[masks[j] for j in idx_ar], weights=sample_w, positive=False)
    sel[idx_tp], _ = R._lasso_select(
        Ain_all, s["Y"][:, idx_tp], reps,
        masks=[full_mask] * len(idx_tp), weights=sample_w, positive=False)
    freq = sel.mean(2)[:, :n]            # 50 × 223，按 cand 顺序
    if cache is not None:
        np.savez_compressed(cache, freq=freq)
    return freq


def _q3_loading(data):
    """与 run_consistency 完全相同的 Q3 载荷（风格内中心化 + CDR 残差化）。"""
    Ns = R._normal_scores_matrix(data.X[data.chem_nd], data.chem_nd)
    mu, sd = Ns.mean(0), Ns.std(0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    M = (Ns - mu) / sd
    c0 = data.cdr - data.cdr.mean()
    beta = (M.T @ c0) / max(float(c0 @ c0), 1e-12)
    M = M - np.outer(c0, beta)
    Y = data.Y[data.targets].to_numpy(float)
    Yc = Y.copy()
    for st in np.unique(data.styles):
        m = data.styles == st
        M[m] -= M[m].mean(0)
        Yc[m] -= Yc[m].mean(0)
    U_, S_, Vt = np.linalg.svd(M, full_matrices=False)
    Z = U_[:, :R.LATENT_K] * S_[:R.LATENT_K]
    coef = Ridge(alpha=10.0).fit(Z, Yc).coef_
    return np.abs(coef @ Vt[:R.LATENT_K])


def _track(name, freq50, W3, n_chem):
    rhos, ov, obs_hits = [], [], []
    for j in range(freq50.shape[0]):
        r = spearmanr(freq50[j], W3[j]).statistic
        rhos.append(float(r) if np.isfinite(r) else 0.0)
        a = set(np.argsort(-freq50[j])[:TOP_K])
        b = set(np.argsort(-W3[j])[:TOP_K])
        ov.append(len(a & b) / TOP_K)
        obs_hits.append(len(a & b))
    rhos = np.asarray(rhos)
    ov = np.asarray(ov)
    obs = float(np.mean(obs_hits))
    exp = TOP_K * TOP_K / n_chem
    fold = obs / exp
    p = float(hypergeom.sf(int(round(obs)) - 1, n_chem, TOP_K, TOP_K))
    return {
        "name": name,
        "n_dims": int(len(rhos)),
        "mean_spearman": float(np.nanmean(rhos)),
        "median_spearman": float(np.nanmedian(rhos)),
        "mean_top20_overlap": float(np.mean(ov)),
        "mean_hits_of_20": obs,
        "expected_hits_of_20": exp,
        "hypergeom_fold": fold,
        "hypergeom_p": p,
        "n_dims_passing_threshold": int((rhos >= 0.15).sum()),
        "passed": bool(np.nanmean(rhos) >= 0.15),
        "_rhos": rhos,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true",
                    help="忽略缓存，重新跑有符号 Lasso（约 5–10 min）")
    args = ap.parse_args()
    t0 = time.time()
    R.set_global_seed(R.SEED)
    data = R.load_data()
    s = U._q2_setup(data)
    cache = OUT / "consistency_dual_freq.npz"
    if args.recompute and cache.exists():
        cache.unlink()
    signed = _signed_freq(data, s, cache)          # 50 × 223
    n_chem = len(data.chem_nd)
    signed_full = np.zeros((len(data.targets), n_chem))
    pos = {c: i for i, c in enumerate(data.chem_nd)}
    for i, c in enumerate(s["cand"]):
        signed_full[:, pos[c]] = signed[:, i]
    pos_freq = pd.read_csv(R.Q2 / "evidence_scores.csv",
                           index_col=0)[list(data.chem_nd)].to_numpy(float)
    W3 = _q3_loading(data)

    tracks = [
        _track("signed", signed_full, W3, n_chem),
        _track("positive", pos_freq, W3, n_chem),
    ]
    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(R.SEED),
        "n_bootstrap": int(R.Q2_BOOTSTRAP),
        "n_compounds": int(n_chem),
        "top_k": TOP_K,
        "threshold": 0.15,
        "tracks": [{k: v for k, v in t.items() if not k.startswith("_")}
                   for t in tracks],
        "note": ("signed = 有符号 Lasso 的入选频率（正负效应均计）；"
                 "positive = Q2 主结果的正向 Lasso 入选频率。"
                 "两者使用完全相同的分层 Bootstrap、样本权重与候选池；"
                 "图平滑未在此表施加（与 evidence_scores.csv 同口径）。"),
    }
    (OUT / "consistency_dual.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    per_dim = pd.DataFrame({
        "sensory": data.targets,
        "rho_signed": tracks[0]["_rhos"],
        "rho_positive": tracks[1]["_rhos"],
        "passes_signed": tracks[0]["_rhos"] >= 0.15,
        "passes_positive": tracks[1]["_rhos"] >= 0.15,
    })
    per_dim.to_csv(OUT / "consistency_dual_per_dim.csv", index=False,
                   encoding="utf-8-sig")
    for t in tracks:
        print(f'{t["name"]:8s} mean {t["mean_spearman"]:.4f}  '
              f'median {t["median_spearman"]:.4f}  '
              f'overlap {t["mean_top20_overlap"]:.3f}  '
              f'fold {t["hypergeom_fold"]:.2f}  p {t["hypergeom_p"]:.3g}  '
              f'passed={t["passed"]}')
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
