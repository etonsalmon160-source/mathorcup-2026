#!/usr/bin/env python
# 本文件由 AI 辅助编写；潜状态轴的定义与采用由参赛队员决定。
"""重建 Q3 的二维潜状态图与 11 例 Pils/Lager→Tripel 迁移轨迹。

只写 results/supplementary/upgrade/，不改任何主结果。

坐标口径
--------
u1 : Pils/Lager 质心 → Tripel 质心的单位轴；坐标缩放到 Pils = 0、Tripel = 1。
u2 : 异味风险轴。风险分数由 DMS、双乙酰、H2S 的 log10 OAV、MeSH 的 z 分数
     与五个感官异味维度的 z 分数等权合成；用 Ridge 回归到 14 维潜状态得到
     风险梯度 β，再把 β 对 u1 正交化并标准化。u2 > 0 表示风险升高。
     该轴是**数据驱动的近似**，不是因果方向；图中只用于展示迁移把样本带向
     高风险侧还是低风险侧。

输出
----
q3_state_axes.json
q3_state_points.csv        250 款背景啤酒 + 11 个源样本 + 33 个端点
q3_state_paths.csv         11 条迁移的状态序列（含 14 维潜坐标）
q3_state_individual.csv    11 个样本 × 4 状态 × 6 个化合物的 OAV/浓度
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_v1 as R  # noqa: E402
import upgrade_probe as U  # noqa: E402

OUT = R.RESULTS / "supplementary" / "upgrade"
OUT.mkdir(parents=True, exist_ok=True)
SENS = U.SENS_CONSTR
AROMA = ["DMS", "X2.3.butanedione", "H2S", "MeSH",
         "isopentyl_acetate", "ethyl_octanoate"]
# 阈值口径与 upgrade_probe.EXT_DEFECTS 一致；香气阈值来自 3.xlsx
THR = {
    "DMS": (19.0, "log10"),
    "X2.3.butanedione": (0.09, "raw"),
    "H2S": (3.5, "log10"),
    "MeSH": (None, "log10"),
    "isopentyl_acetate": (0.603333, "log10"),
    "ethyl_octanoate": (0.700000, "log10"),
}


def _z(v):
    v = np.asarray(v, float)
    m = np.nanmedian(v)
    s = np.nanstd(v, ddof=1)
    s = s if s > 1e-12 else 1.0
    return np.clip((v - m) / s, -3, 3)


def _oav(x, name):
    thr, scale = THR[name]
    x = np.asarray(x, float)
    if thr is None:
        return x
    if scale == "log10":
        return 10.0 ** (x - np.log10(thr))
    return x / thr


def _risk_table(data):
    """逐啤酒的异味风险分量。缺失用中位数补，避免个别空值拖垮合成。"""
    def col(c):
        v = data.X[c].to_numpy(float)
        return np.where(np.isfinite(v), v, np.nanmedian(v))

    dms = col("DMS") - np.log10(19.0)
    dia = np.log10(np.maximum(col("X2.3.butanedione"), 1e-6) / 0.09)
    h2s = col("H2S") - np.log10(3.5)
    mesh = _z(col("MeSH"))
    sens = np.mean([_z(data.Y[t].to_numpy(float)) for t in SENS], axis=0)
    risk = np.mean([_z(dms), _z(dia), _z(h2s), mesh, sens], axis=0)
    return risk, {"DMS_logOAV": dms, "diacetyl_logOAV": dia,
                  "H2S_logOAV": h2s, "MeSH_z": mesh, "sensory_z": sens}


def main() -> None:
    R.set_global_seed(R.SEED)
    data = R.load_data()
    ctx = U._setup_q3(data)
    Z = ctx["Z"]
    k = R.LATENT_K
    styles = data.styles
    pils = styles == "Pils/Lager"
    trip = styles == "Tripel"

    # ---- u1: Pils → Tripel
    p0 = Z[pils].mean(0)
    p1 = Z[trip].mean(0)
    u1 = p1 - p0
    u1_scale = float(u1 @ u1)
    u1 = u1 / np.sqrt(u1_scale)

    # ---- u2: 异味风险轴（对 u1 正交化）
    risk, comps = _risk_table(data)
    beta = Ridge(alpha=10.0).fit(Z, risk).coef_
    u2 = beta - (beta @ u1) * u1
    u2 = u2 / np.linalg.norm(u2)
    q1 = (Z - p0) @ u1 / np.sqrt(u1_scale)

    def coords(z):
        z = np.atleast_2d(z)
        c1 = (z - p0) @ u1 / np.sqrt(u1_scale)
        c2 = (z - p0) @ u2
        return c1, c2

    c1, c2 = coords(Z)
    if np.corrcoef(c2, risk)[0, 1] < 0:
        u2 = -u2
        c1, c2 = coords(Z)
    u2_sd = float(np.std(c2, ddof=1))
    u2 = u2 / u2_sd

    def uv(z):
        z = np.atleast_2d(z)
        return ((z - p0) @ u1 / np.sqrt(u1_scale),
                (z - p0) @ u2)

    c1, c2 = uv(Z)

    # ---- 11 例迁移（复刻 upgrade_probe._run_pair 的三种状态）
    mp, Sp = R._moments(Z, pils, ctx["Spool"])
    mt, St = R._moments(Z, trip, ctx["Spool"])
    dm = mt - mp
    X_clean, col, xmed = ctx["X_clean"], ctx["col"], ctx["xmed"]
    src_idx = np.where(pils)[0]
    points, paths, indiv = [], [], []

    for i in range(len(Z)):
        points.append({
            "role": "background", "beer": int(data.X.iloc[i]["beer"]),
            "style": styles[i], "method": "observed", "order": 0,
            "u1": float(c1[i]), "u2": float(c2[i]),
            "risk": float(risk[i]),
        })

    for pos_i, i in enumerate(src_idx):
        beer = int(data.X.iloc[i]["beer"])
        zi = Z[i]
        zot = R._gaussian_ot(mp, Sp, mt, St, zi)
        x_ot = R._ns_to_raw(
            ((zot @ ctx["Vt"][:k]) * ctx["sd_ns"] + ctx["mu_ns"])[None, :],
            ctx["raw_tables"])[0]
        x_src = np.where(np.isfinite(X_clean[i]), X_clean[i], xmed)
        for d in U.EXT_DEFECTS:
            j = col[d]
            if not np.isfinite(X_clean[i, j]):
                x_src[j] = np.nanmin(X_clean[:, j])
        sens_leg = U._legacy_readout(ctx, x_src)
        base_leg = np.array([sens_leg[data.targets.index(dim)]
                             for dim in SENS])

        states = {"before": (zi, x_src)}
        for method, mode in (("none", "none"), ("ot_proj", "loose"),
                             ("qp_full", "loose")):
            if method == "none":
                x = np.clip((x_ot + 0.10 * x_src) / 1.10,
                            ctx["xmin"], ctx["xmax"])
            elif method == "ot_proj":
                x = U._legacy_transfer(ctx, x_ot, x_src, mode, method)
            else:
                bd = {col[d]: U._bound(d, x_src[col[d]], mode)
                      for d in U.EXT_DEFECTS}
                x, _, _ = U._solve_qp(
                    ctx, x_ot, x_src, bd, sensory_w=ctx["W_sens"],
                    sensory_base=base_leg, sensory_c=ctx["W_sens_c"])
                if x is None:
                    raise RuntimeError(f"QP infeasible for beer {beer}")
            zns = R._raw_to_ns(x[None, :], ctx["raw_tables"], len(Z))[0]
            zl = ((zns - ctx["mu_ns"]) / ctx["sd_ns"]) @ ctx["Vt"][:k].T
            states[method] = (zl, x)

        for order, (state, (zl, x)) in enumerate(states.items()):
            a, b = uv(zl)
            r_pred = float(Ridge(alpha=10.0).fit(Z, risk).predict(zl[None, :])[0])
            points.append({
                "role": "endpoint", "beer": beer,
                "style": styles[i], "method": state, "order": order,
                "u1": float(a[0]), "u2": float(b[0]), "risk": r_pred,
            })
            paths.append({
                "beer": beer, "state": state, "order": order,
                "u1": float(a[0]), "u2": float(b[0]),
                "style_shift": float((zl - zi) @ dm / (dm @ dm)),
                **{f"z{j + 1}": float(v) for j, v in enumerate(zl)},
            })
            row = {"beer": beer, "source_index": pos_i, "state": state,
                   "u1": float(a[0]), "u2": float(b[0]),
                   "style_shift": float((zl - zi) @ dm / (dm @ dm))}
            for name in AROMA:
                if name == "MeSH":       # 无阈值，特征本身即 log10(µg/L)
                    row[name] = float(10.0 ** x[col[name]])
                    row[f"{name}_log10"] = float(x[col[name]])
                else:
                    o = _oav(x[col[name]], name)
                    row[name] = float(o)
                    row[f"{name}_log10"] = float(np.log10(max(o, 1e-9)))
            for dim in SENS:
                j = data.targets.index(dim)
                sen = U._legacy_readout(ctx, x)[j]
                before = sens_leg[j]
                row[f"delta_{dim}"] = float(sen - before)
            for d in U.EXT_DEFECTS:
                row[f"delta_x_{d}"] = float(x[col[d]] - x_src[col[d]])
                row[f"before_x_{d}"] = float(x_src[col[d]])
            indiv.append(row)

    axes = {
        "latent_k": int(k),
        "u1": {
            "definition": "Pils/Lager centroid -> Tripel centroid",
            "vector": u1.tolist(),
            "pils_centroid": p0.tolist(),
            "tripel_centroid": p1.tolist(),
        },
        "u2": {
            "definition": ("Ridge gradient of composite off-flavour risk, "
                           "orthogonalised to u1, scaled to 1 SD"),
            "vector": u2.tolist(),
            "risk_components": ["DMS log10 OAV", "diacetyl log10 OAV",
                                "H2S log10 OAV", "MeSH z",
                                "5 sensory off-flavour dims z"],
            "risk_beta": beta.tolist(),
            "zone_quantile": 0.75,
        },
        "risk_quantiles": {
            str(q): float(np.quantile(risk, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        },
        "zone_threshold": float(np.quantile(risk, 0.75)),
        "n_background": int(len(Z)),
        "n_source": int(pils.sum()),
    }
    (OUT / "q3_state_axes.json").write_text(
        json.dumps(axes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(points).to_csv(OUT / "q3_state_points.csv", index=False,
                                encoding="utf-8-sig")
    pd.DataFrame(paths).to_csv(OUT / "q3_state_paths.csv", index=False,
                               encoding="utf-8-sig")
    pd.DataFrame(indiv).to_csv(OUT / "q3_state_individual.csv", index=False,
                               encoding="utf-8-sig")
    print(f"wrote {len(points)} points, {len(paths)} path states, "
          f"{len(indiv)} individual-state rows")


if __name__ == "__main__":
    main()
