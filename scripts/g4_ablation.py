"""G4：消融实验。砍掉一组处理/特征，看主指标掉多少。

AI 辅助声明：本脚本由 AI 辅助编写；消融分组与"负贡献即下线"的判定标准
由参赛队员确认，AI 只负责实现与跑数。

分组
----
- 完整：哨兵值先置缺失 + 中位数插补 + 231 化学特征 + 样本检出率协变量
- 去掉检出率协变量
- 去掉派生汇总列（kcalperc / esters_sum / aroma_hops_sum / acids_sum / sulfur_sum）
- 哨兵当真实值（不屏蔽）
- 审计前口径（哨兵当真实值且无检出率协变量）

主模型固定为 Extra Trees（与 Q1 主模型一致），随机 5 折与按风格 5 折各跑 3 个种子。
`与完整模型的差` = 该变体平均 R² − 完整模型平均 R²；正数表示砍掉后变好（应下线）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_v1 as R  # noqa: E402

OUT = R.Q1
SEEDS = R.SEEDS
HARM_MAX = 0.002


def _plain_scale(Xtr, Xte):
    """对照口径：不识别哨兵，直接中位数插补 + 标准化。"""
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sc = StandardScaler()
    return sc.fit_transform(imp.fit_transform(Xtr)), sc.transform(imp.transform(Xte))


def _x_full(data):
    return np.column_stack([data.X[data.chem_all].to_numpy(float), data.cdr])


def _x_no_derived(data):
    cols = [c for c in data.chem_all if c not in R.DERIVED]
    return np.column_stack([data.X[cols].to_numpy(float), data.cdr])


VARIANTS = {
    "完整（哨兵屏蔽 + 检出率协变量）": ("qc", _x_full),
    "去掉检出率协变量": ("qc", lambda d: d.X[d.chem_all].to_numpy(float)),
    "去掉派生汇总列": ("qc", _x_no_derived),
    "哨兵当真实值（不屏蔽）": ("plain", _x_full),
    "审计前口径（哨兵当真实值、无检出率）": ("plain",
                                            lambda d: d.X[d.chem_all].to_numpy(float)),
}
# 对照基线不是"可砍掉的特征组"，只用来量 QC 的代价，不参与下线判定。
COMPARATORS = {"审计前口径（哨兵当真实值、无检出率）"}
N_BOOT = 2000


def main() -> int:
    t0 = time.time()
    R.set_global_seed(R.SEED)
    data = R.load_data()
    Y = data.Y[data.targets].to_numpy(float)
    styles = data.styles
    n = len(Y)

    rows = []
    dim_r2 = {}
    for name, (mode, builder) in VARIANTS.items():
        X = builder(data)
        for scheme in ("random", "group"):
            scores, per_dim = [], []
            for seed in SEEDS:
                splits = (list(KFold(5, shuffle=True, random_state=R.SEED).split(np.arange(n)))
                          if scheme == "random"
                          else list(GroupKFold(5).split(np.arange(n), groups=styles)))
                oof = np.full_like(Y, np.nan)
                for tr, te in splits:
                    if mode == "qc":
                        A, B = R._impute_scale(X[tr], X[te])
                    else:
                        A, B = _plain_scale(X[tr], X[te])
                    oof[te] = R.fit_predict("et", A, Y[tr], B, styles[te], seed, quick=False)
                r2, _, _ = R._target_metrics(Y, oof)
                scores.append(float(np.mean(r2)))
                per_dim.append(r2)
            rows.append({"特征组": name, "处理口径": mode, "划分": scheme,
                         "mean_r2": float(np.mean(scores)),
                         "seed_std": float(np.std(scores, ddof=1)),
                         "n_seeds": len(SEEDS)})
            dim_r2[(name, scheme)] = np.mean(per_dim, axis=0)

    df = pd.DataFrame(rows)
    full_name = next(n for n in VARIANTS if n.startswith("完整"))
    full = df[df["特征组"] == full_name].set_index("划分")["mean_r2"]
    df["与完整模型的差"] = df.apply(
        lambda r: r["mean_r2"] - full[r["划分"]], axis=1)

    # 50 个维度上的配对 bootstrap：差异是否只是维度间噪声。
    n_targets = Y.shape[1]
    rng = np.random.default_rng(R.SEED)
    boot_idx = rng.integers(0, n_targets, size=(N_BOOT, n_targets))
    lo, hi = [], []
    for _, r in df.iterrows():
        if r["特征组"] == full_name:
            lo.append(0.0)
            hi.append(0.0)
            continue
        d = dim_r2[(r["特征组"], r["划分"])] - dim_r2[(full_name, r["划分"])]
        boot = d[boot_idx].mean(axis=1)
        b0, b1 = np.percentile(boot, [2.5, 97.5])
        lo.append(float(b0))
        hi.append(float(b1))
    df["差_95CI下"] = lo
    df["差_95CI上"] = hi
    df["参与判定"] = np.where(df["特征组"].isin(COMPARATORS), "否", "是")
    df["判定"] = np.select(
        [~df["参与判定"].eq("是"),
         df["与完整模型的差"] > HARM_MAX],
        ["对照基线（仅参考）", "负贡献，应下线"],
        default="有贡献或中性")

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ablation.csv", index=False, encoding="utf-8-sig")

    print(df.to_string(index=False))
    bad = df[(df["参与判定"] == "是") & (df["与完整模型的差"] > HARM_MAX)]
    print(f"\nG4：{'FAIL，以下处理有负贡献：' + '、'.join(bad['特征组']) if len(bad) else 'PASS，无负贡献处理'}")
    print(f"用时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
