"""G3：没有官方测试集，用内部留出折做对抗验证与漂移检验。

AI 辅助声明：本脚本由 AI 辅助编写；检验口径（随机折判泄漏、风格折判域偏移）
由参赛队员确认，AI 只负责实现与跑数。

两种划分的用途不同
------------------
1. 随机折、按风格分层：两边本应同分布。AUC 接近 0.5、MMD/C2ST 不显著，
   才说明本地 CV 的训练/留出之间没有泄漏。G3 门看这一项。
2. 按风格分组留出：两边必然可分（风格化学差异是真实信号，不是泄漏）。
   只作域偏移诊断写进 drift.json，**不能用它判 G3 失败**。

产物
----
results/g3/adversarial.json        G3 判定用（auc 必须 < 0.60）
results/g3/adversarial_features.csv 随机折里最漂移的特征
results/g3/drift.json              风格留出折 AUC/MMD、逐特征 KS
results/g3/SUMMARY.md              人看的结论
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import run_v1 as R  # noqa: E402
from src.drift import mmd_test, per_feature_ks  # noqa: E402
from src.validate import _c2st_test  # noqa: E402

OUT = R.RESULTS / "g3"
AUC_MAX = 0.60
N_ESTIMATORS = 300
N_PERMUTATIONS = 200


def _classifier() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS, min_samples_leaf=3, max_features="sqrt",
        n_jobs=R.N_JOBS, random_state=R.SEED)


def _oof_auc(X: np.ndarray, y: np.ndarray, splits) -> tuple[float, np.ndarray, np.ndarray]:
    clf = _classifier()
    proba = np.full(len(y), np.nan)
    for tr, te in splits:
        clf.fit(X[tr], y[tr])
        proba[te] = clf.predict_proba(X[te])[:, 1]
    imp = np.zeros(X.shape[1])
    for tr, _ in splits:
        clf.fit(X[tr], y[tr])
        imp += clf.feature_importances_
    return float(roc_auc_score(y, proba)), imp / len(splits)


def _random_half_labels(styles: np.ndarray, seed: int) -> np.ndarray:
    """按风格分层随机分两半，保证两边的风格构成一致。"""
    rng = np.random.default_rng(seed)
    y = np.zeros(len(styles), dtype=int)
    for s in np.unique(styles):
        idx = np.where(styles == s)[0]
        take = rng.choice(idx, len(idx) // 2, replace=False)
        y[take] = 1
    return y


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R.set_global_seed(R.SEED)
    data = R.load_data()
    names = list(data.chem_all) + ["cdr"]
    Xraw = np.column_stack([data.X[data.chem_all].to_numpy(float), data.cdr])
    X = R._impute_scale(Xraw, Xraw)[0]
    styles = data.styles
    n = len(X)

    # ---------- 1) 随机折：判泄漏，G3 用 ----------
    y_rand = _random_half_labels(styles, R.SEED)
    rand_splits = list(StratifiedKFold(5, shuffle=True, random_state=R.SEED).split(
        np.arange(n), y_rand))
    auc_rand, imp = _oof_auc(X, y_rand, rand_splits)
    c2st = _c2st_test(pd.DataFrame(X, columns=names), y_rand, n_perm=N_PERMUTATIONS)
    a, b = X[y_rand == 0], X[y_rand == 1]
    mmd_rand = mmd_test(a, b, n_perm=N_PERMUTATIONS, seed=R.SEED)
    ks = per_feature_ks(a, b, names)
    ks.to_csv(OUT / "adversarial_features.csv", index=False, encoding="utf-8-sig")

    adversarial = {
        "auc": round(auc_rand, 4),
        "c2st": c2st,
        "mmd": mmd_rand,
        "threshold": AUC_MAX,
        "ok": bool(auc_rand < AUC_MAX),
        "n": n,
        "split": "random stratified by style",
        "top_drifting_features": ks.head(10)["特征"].tolist(),
        "note": "没有官方测试集，按 docs/门控系统.md G3 用内部随机留出折判泄漏；"
                "风格留出折的 AUC 另存 drift.json，只作域偏移诊断。",
    }
    (OUT / "adversarial.json").write_text(
        json.dumps(adversarial, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 2) 风格留出：域偏移诊断，不算 G3 失败 ----------
    # 随机选一半风格作为一组、另一半作为另一组，再用分层 CV 判可分性。
    # 不能用 GroupKFold 的折号当标签：那样训练折里只有一个类，分类器学不到决策边界。
    rng = np.random.default_rng(R.SEED)
    style_list = np.unique(styles)
    held = set(rng.choice(style_list, len(style_list) // 2, replace=False))
    y_style = np.array([s in held for s in styles], dtype=int)
    style_splits = list(StratifiedKFold(5, shuffle=True, random_state=R.SEED).split(
        np.arange(n), y_style))
    auc_style, _ = _oof_auc(X, y_style, style_splits)
    drift = {
        "style_subset_auc": round(auc_style, 4),
        "interpretation": "随机一半风格 vs 另一半：AUC 高是预期结果，"
                          "风格化学差异是真实信号。该项不参与 G3 判定。",
        "random_holdout_auc": round(auc_rand, 4),
        "c2st_p_permutation": c2st["p_permutation"],
        "mmd_p_value": mmd_rand["p_value"],
        "n_features_ks_drift_bh_0.05": int(ks["是否漂移"].sum()),
        "top_ks_features": ks.head(10)[["特征", "KS统计量", "校正后p"]].to_dict("records"),
    }
    (OUT / "drift.json").write_text(
        json.dumps(drift, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# G3 对抗验证与漂移检验", "",
        f"- 随机留出（按风格分层）：AUC = **{auc_rand:.4f}**（门控上限 {AUC_MAX}）→ "
        f"{'PASS' if auc_rand < AUC_MAX else 'FAIL'}",
        f"- C2ST 置换 p = {c2st['p_permutation']:.4g}；MMD p = {mmd_rand['p_value']:.4g}",
        f"- 逐特征 KS（BH 校正 0.05）判为漂移：{int(ks['是否漂移'].sum())}/{len(ks)} 列",
        f"- 一半风格 vs 另一半 AUC = {auc_style:.4f}（域偏移诊断，不参与 G3 判定）", "",
        "> 读法：随机折两边同分布才说明本地 CV 没有泄漏；"
        "风格折可分是 Q3 要解决的域偏移，不是数据错误。",
        "", f"用时 {time.time() - t0:.1f}s。本文件由 AI 辅助生成，口径由参赛队员确认。",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
