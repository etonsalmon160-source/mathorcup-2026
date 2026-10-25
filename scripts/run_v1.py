#!/usr/bin/env python
# 本文件由 AI 辅助编写；建模与算法设计决策由参赛队员做出。详见《AI工具使用详情说明》。
"""第一版端到端对比流程（Q1 / Q2 / Q3 / 稳健性 / 三问互证）。

设计目标
--------
一晚跑完，第二天早上能看到"第一版结果"：
  1. 问题一：13 个候选模型 × 两种交叉验证（随机 / 按风格分组）× 3 个种子，
     统一输出逐维 R² / RMSE / MAE / Spearman 和 OOF 预测。
  2. 问题二：用 2.csv 建共表达图，Louvain 模块；Lasso 稳定性选择（按风格分层
     Bootstrap）给出 50 维 Top-3，再以分半复现率与选择频率衡量稳定性。
  3. 问题三：22 个风格的条件高斯生成 + Pils/Lager→Tripel 迁移；
     在潜空间做最优传输，在原始特征空间做"逐坐标约束投影"（精确 QP 解），
     保证缺陷不新增、不加重。
  4. 稳健性：3 种子、潜维数 k 敏感性、约束松紧 Pareto、缺失压力测试。
  5. 三问互证：Q1↔Q3 感官一致、Q2↔Q3 载荷重合。

口径
----
- 只读 data/raw/附件；不写任何中间数据。
- 建模输入用 1.csv 原样（198 列 log10 + 33 列原始量纲）。
- 感官目标用 4.csv 给定尺度（标准化值），不裁剪到 [0,6]。
- 0–6 分只作为展示尺度：逐维经验分位仿射映射，在 calibration.json 里写明。
- 随机种子固定；多种子只改模型随机性，交叉验证切分固定，便于平均 OOF。

运行
----
    python scripts/run_v1.py --seeds 42 43 44          # 全量，约 45–60 分钟
    python scripts/run_v1.py --quick --seeds 42        # 快速自检，约 5 分钟
    python scripts/run_v1.py --only q1 q2              # 只跑指定阶段
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import N_JOBS, PATHS, SEED, SEEDS, set_global_seed  # noqa: E402
from src.data_gate import META, build_maps, fix_mojibake, load_tables  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RAW = PATHS["raw"] / "附件"
RESULTS = PATHS["results"]
V1 = RESULTS / "v1"
Q1 = RESULTS / "q1"
Q2 = RESULTS / "q2"
Q3 = RESULTS / "q3"
ROB = RESULTS / "robustness"
CONS = RESULTS / "consistency"

DERIVED = ["kcalperc", "esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum"]
DEFECTS = {
    "DMS": {"threshold": 19.0, "scale": "log10", "unit": "ug/L"},
    "X4VG": {"threshold": 0.20, "scale": "log10", "unit": "mg/L"},
    "X2.3.butanedione": {"threshold": 0.09, "scale": "raw", "unit": "mg/L"},
    "X.E..2.nonenal": {"threshold": None, "scale": "raw", "unit": "mg/L"},
}
# Q2：Lasso 稳定性选择。pH / 色度 / 内标不是风味化合物，派生列是汇总，均不进候选。
Q2_ALPHA = 0.06
Q2_BOOTSTRAP = 300
# 共表达图在最终排序中的权重。0.20 是事先写死的建模选择（80% 选择频率 +
# 20% 邻居平均频率），不是按测试结果调出来的；敏感性见 report.json。
# 0.20 下 47/50 维 Top-3 不变，只有 3 个低稳定维度由邻域证据破平。
Q2_GRAPH_GAMMA = 0.20
Q2_EXCLUDE = {"pH", "color.EBC.", "X3.methyl.2.butenal"}
# 香气维度只允许「能到达嗅上皮」的挥发性有机化合物。下面这些是金属离子、
# 蛋白/多糖/糖、非挥发酸、色素、气体与酒体基质，留在口感/风味维度，但不进 A_* 候选。
Q2_NONVOLATILE = {
    "iron", "protein.g.L.", "GFS", "color.EBC.", "pH", "bitterness",
    "betaglucan.mg.L.", "unfermentables", "glycerol", "lactic_acid.mg.L.",
    "CO2.PSI.",
    # 非风味/工艺添加剂：二氧化硫、BHT 抗氧化剂、氨（无机挥发物）
    "SO2.mg.L.", "butylated.hydroxytoluene", "ammonia.mg.L.",
}
# A_* 以及嗅后（retronasal）香气评分 F_hops/F_malt/F_esters 视为香气维度。
# 另有 9 个单化合物/单香调维度虽被 8.xlsx 归为 Taste，本质是气味，也按香气处理。
Q2_AROMA_PREFIXES = ("A_", "F_hops_", "F_malt_", "F_esters_")
Q2_AROMA_TARGETS = {
    "X4vg", "diacetyl", "dms", "stale_hops", "t2n",
    "orange", "coriander", "clove", "barnyard",
}


def _is_aroma_target(t: str) -> bool:
    return t.startswith(Q2_AROMA_PREFIXES) or t in Q2_AROMA_TARGETS


# 直接测量的「同名化合物」：用来审计每个维度的化学可解释性。
# 若感官分与同名化合物几乎不相关，这一维的 Top-3 只能作探索性结果。
Q2_NAMESAKE = {
    "dms": "DMS", "diacetyl": "X2.3.butanedione", "X4vg": "X4VG",
    "t2n": "X.E..2.nonenal", "stale_hops": "X.E..2.nonenal",
    "metallic": "iron", "sweetness": "GFS", "bitternes": "bitterness",
    "lactic": "lactic_acid.mg.L.", "acetic": "acetic_acid.g.L.",
    "alcohol": "ethanol..v.v.", "co2": "CO2.PSI.", "body": "glycerol",
}


def _norm_chem(s) -> str:
    import re
    import unicodedata
    s = fix_mojibake(str(s))
    for a, b in (("α", "alpha"), ("β", "beta"), ("γ", "gamma"), ("δ", "delta")):
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


def _family_map(data: Data) -> dict[str, str]:
    th = load_tables().thresholds.copy()
    th["k"] = th["chemical_name"].map(_norm_chem)
    mp = {k: g.iloc[0]["origin_paper"] for k, g in th.drop_duplicates("k").groupby("k")}
    out = {}
    for c in data.chem_nd:
        fam = mp.get(_norm_chem(data.chem_map.get(c, c)), "")
        out[c] = "" if pd.isna(fam) else str(fam)
    return out
LATENT_K = 14
SENSORY_LATENT_K = 8

# QC：1.csv 的化合物列里，低于检出限的单元格被填成同一个 log10 哨兵值。
# 若不处理，中位数插补会把「未检出」当成真实高值，PCA/树模型会把仪器检出率
# 当成生物信号。这里统一置为缺失，再按训练折中位数插补。
FLOOR_SENTINEL = -1.817819
FLOOR_ATOL = 1e-6
FLOOR_MIN_REPEAT = 5          # 某列最小值至少重复 5 次，才视为该列的检出下限
FLOOR_HEAVY_QUANTILE = 0.025  # 检出率最低的 2.5% 样本标记为 floor-heavy（n=250→6）

log = logging.getLogger("v1")


# ================================================================ 工具

def ensure_dirs() -> None:
    for d in (V1, Q1, Q2, Q3, ROB, CONS):
        d.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_jsonable),
                    encoding="utf-8")


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    raise TypeError(type(x))


def fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def version_tuple() -> dict:
    import sklearn, scipy, pandas, numpy
    out = {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }
    try:
        import lightgbm
        out["lightgbm"] = lightgbm.__version__
    except ImportError:
        out["lightgbm"] = None
    try:
        import xgboost
        out["xgboost"] = xgboost.__version__
    except ImportError:
        out["xgboost"] = None
    try:
        import networkx
        out["networkx"] = networkx.__version__
    except ImportError:
        out["networkx"] = None
    return out


# ================================================================ 数据

@dataclass
class Data:
    X: pd.DataFrame
    Y: pd.DataFrame
    styles: np.ndarray
    chem_all: list[str]
    chem_nd: list[str]          # 226 个非派生化合物
    targets: list[str]
    chem_map: dict[str, str]    # 1.csv 列 → 2.csv 全名
    sens_map: dict[str, str]    # 4.csv 列 → 5/6.csv 面板全名
    cdr: np.ndarray             # 每个样本的化合物检出率（非 floor 的比例）
    floor_heavy: np.ndarray     # 检出率最低的 2.5% 样本


def _floor_mask(A: np.ndarray, floor_values: np.ndarray | None = None) -> np.ndarray:
    """标记低于检出限的单元格。

    `floor_values` 由训练折给出；测试折沿用训练折的下限，避免信息泄漏。
    共享哨兵值（1.csv 的 -1.817819）无论是否在训练折出现都直接屏蔽。
    """
    A = np.asarray(A, dtype=float)
    out = np.isclose(A, FLOOR_SENTINEL, atol=FLOOR_ATOL)
    if floor_values is not None:
        fv = np.asarray(floor_values, dtype=float)
        ok = np.isfinite(fv)
        if ok.any():
            out[:, ok] |= np.isclose(A[:, ok], fv[ok], atol=FLOOR_ATOL)
    return out


def _floor_values(A: np.ndarray) -> np.ndarray:
    """从训练折估计每列的检出下限：负值列的最小值重复 ≥5 次才认。"""
    A = np.asarray(A, dtype=float)
    with np.errstate(all="ignore"):
        mn = np.nanmin(A, axis=0)
        n_at_min = np.sum(np.isclose(A, mn, atol=FLOOR_ATOL), axis=0)
    return np.where((mn < 0) & (n_at_min >= FLOOR_MIN_REPEAT), mn, np.nan)


def load_data() -> Data:
    X = pd.read_csv(RAW / "1.csv")
    Y = pd.read_csv(RAW / "4.csv")
    if not (X["beer"].to_numpy() == Y["beer"].to_numpy()).all():
        raise SystemExit("1.csv 与 4.csv 的 beer 行序不一致")
    chem_all = [c for c in X.columns if c not in META]
    chem_nd = [c for c in chem_all if c not in DERIVED]
    targets = [c for c in Y.columns if c not in META]
    t = load_tables()
    chem_map, sens_map = build_maps(t)

    # 检出率只在 log10 列上算（原始量纲列没有哨兵值）。
    Xmat = X[chem_nd].to_numpy(float)
    neg_cols = np.nanmin(Xmat, axis=0) < 0
    fm = _floor_mask(Xmat)
    cdr = 1.0 - fm[:, neg_cols].mean(1) if neg_cols.any() else np.ones(len(X))
    cut = float(np.quantile(cdr, FLOOR_HEAVY_QUANTILE))
    floor_heavy = cdr <= cut

    return Data(X=X, Y=Y, styles=X["tasting_category_fine"].to_numpy(),
                chem_all=chem_all, chem_nd=chem_nd, targets=targets,
                chem_map=chem_map, sens_map=sens_map,
                cdr=cdr, floor_heavy=floor_heavy)


# ================================================================ Q1

def _impute_scale(Xtr: np.ndarray, Xte: np.ndarray):
    """检出限置缺失 → 中位数插补 → 稳健标准化。

    1.csv 里有近常量列；StandardScaler 会除以 1e-12 量级的标准差，
    把线性模型炸成 R²=-86。这里给标准差设下限，并把标准化值截到 ±10。
    """
    from sklearn.impute import SimpleImputer
    Xtr = np.asarray(Xtr, dtype=float)
    Xte = np.asarray(Xte, dtype=float)
    fv = _floor_values(Xtr)
    Araw = np.where(_floor_mask(Xtr, fv), np.nan, Xtr)
    Braw = np.where(_floor_mask(Xte, fv), np.nan, Xte)
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    A = imp.fit_transform(Araw)
    B = imp.transform(Braw)
    mu = A.mean(0)
    sd = A.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (np.clip((A - mu) / sd, -10, 10),
            np.clip((B - mu) / sd, -10, 10))


def _impute(Xtr: np.ndarray, Xte: np.ndarray):
    from sklearn.impute import SimpleImputer
    Xtr = np.asarray(Xtr, dtype=float)
    Xte = np.asarray(Xte, dtype=float)
    fv = _floor_values(Xtr)
    Araw = np.where(_floor_mask(Xtr, fv), np.nan, Xtr)
    Braw = np.where(_floor_mask(Xte, fv), np.nan, Xte)
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    return imp.fit_transform(Araw), imp.transform(Braw)


def _ridge_multi(Xtr, Ytr, Xte):
    """多输出 Ridge；alpha 在训练折内用 GCV 选，避免在 231 个共线特征上手调。"""
    from sklearn.linear_model import RidgeCV
    A, B = _impute_scale(Xtr, Xte)
    m = RidgeCV(alphas=np.logspace(0, 4, 9))
    return m.fit(A, Ytr).predict(B)


def _pls_multi(Xtr, Ytr, Xte, n_components=12):
    from sklearn.cross_decomposition import PLSRegression
    A, B = _impute_scale(Xtr, Xte)
    m = PLSRegression(n_components=n_components, scale=False, max_iter=500)
    m.fit(A, Ytr)
    return m.predict(B)


def _et_multi(Xtr, Ytr, Xte, seed=SEED):
    from sklearn.ensemble import ExtraTreesRegressor
    A, B = _impute(Xtr, Xte)
    m = ExtraTreesRegressor(n_estimators=300, max_features=0.3, min_samples_leaf=2,
                            n_jobs=N_JOBS, random_state=seed)
    return m.fit(A, Ytr).predict(B)


def _rf_multi(Xtr, Ytr, Xte, seed=SEED):
    from sklearn.ensemble import RandomForestRegressor
    A, B = _impute(Xtr, Xte)
    m = RandomForestRegressor(n_estimators=300, max_features=0.3, min_samples_leaf=2,
                              n_jobs=N_JOBS, random_state=seed)
    return m.fit(A, Ytr).predict(B)


def _lgbm_per_target(Xtr, Ytr, Xte, seed=SEED, **kw):
    import lightgbm as lgb
    out = np.zeros((len(Xte), Ytr.shape[1]))
    params = dict(n_estimators=250, num_leaves=15, learning_rate=0.04,
                  min_child_samples=10, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.6, reg_lambda=1.0, n_jobs=N_JOBS,
                  random_state=seed, verbose=-1, force_col_wise=True)
    params.update(kw)
    for j in range(Ytr.shape[1]):
        m = lgb.LGBMRegressor(**params)
        m.fit(Xtr, Ytr[:, j])
        out[:, j] = m.predict(Xte)
    return out


def _xgb_per_target(Xtr, Ytr, Xte, seed=SEED, **kw):
    import xgboost as xgb
    out = np.zeros((len(Xte), Ytr.shape[1]))
    params = dict(n_estimators=250, max_depth=3, learning_rate=0.04,
                  subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
                  n_jobs=N_JOBS, tree_method="hist", random_state=seed, verbosity=0)
    params.update(kw)
    for j in range(Ytr.shape[1]):
        m = xgb.XGBRegressor(**params)
        m.fit(Xtr, Ytr[:, j])
        out[:, j] = m.predict(Xte)
    return out


def _svr_per_target(Xtr, Ytr, Xte, seed=SEED, **kw):
    from sklearn.svm import SVR
    A, B = _impute_scale(Xtr, Xte)
    out = np.zeros((len(Xte), Ytr.shape[1]))
    params = dict(C=10.0, epsilon=0.1, gamma="scale")
    params.update(kw)
    for j in range(Ytr.shape[1]):
        m = SVR(**params)
        m.fit(A, Ytr[:, j])
        out[:, j] = m.predict(B)
    return out


def _latent_target_ridge(Xtr, Ytr, Xte, seed=SEED, n_components=SENSORY_LATENT_K):
    """主模型：8 维感官 PCA 潜状态 + RidgeCV。

    2026-10-24 同口径复算：随机 CV 0.2411、分组 CV 0.1949；
    同折下 PLS8 为 0.2321 / 0.1813，故主流程保留 PCA8。
    alpha 网格固定为 logspace(-1, 3, 9)，不按测试集调。
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    A, B = _impute_scale(Xtr, Xte)
    pca = PCA(n_components=min(n_components, Ytr.shape[1] - 1),
              random_state=seed)
    T = pca.fit_transform(Ytr)
    pred_t = RidgeCV(alphas=np.logspace(-1, 3, 9)).fit(A, T).predict(B)
    return pca.inverse_transform(pred_t)


def _latent_target_pls(Xtr, Ytr, Xte, seed=SEED, n_components=SENSORY_LATENT_K):
    """对照模型：8 维有监督 PLS 潜状态。

    同口径下低于 PCA8（随机 0.2321 / 分组 0.1813），保留在主比较表中
    供论文展示，不进入主模型。PLS 无随机性，seed 仅为接口一致。
    """
    from sklearn.cross_decomposition import PLSRegression
    A, B = _impute_scale(Xtr, Xte)
    k = min(n_components, A.shape[1], Ytr.shape[1], A.shape[0] - 1)
    m = PLSRegression(n_components=k, scale=False, max_iter=500)
    return m.fit(A, Ytr).predict(B)


def _latent_target_lgbm(Xtr, Ytr, Xte, seed=SEED, n_components=SENSORY_LATENT_K):
    from sklearn.decomposition import PCA
    import lightgbm as lgb
    pca = PCA(n_components=min(n_components, Ytr.shape[1] - 1), random_state=seed)
    T = pca.fit_transform(Ytr)
    pred_t = _lgbm_per_target(Xtr, T, Xte, seed=seed, n_estimators=200)
    return pca.inverse_transform(pred_t)


def _chem_latent_ridge(Xtr, Ytr, Xte, seed=SEED, n_components=LATENT_K):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    A, B = _impute_scale(Xtr, Xte)
    pca = PCA(n_components=min(n_components, A.shape[1] - 1), random_state=seed)
    At = pca.fit_transform(A)
    Bt = pca.transform(B)
    return RidgeCV(alphas=np.logspace(-1, 3, 9)).fit(At, Ytr).predict(Bt)


def _chem_latent_lgbm(Xtr, Ytr, Xte, seed=SEED, n_components=LATENT_K):
    from sklearn.decomposition import PCA
    A, B = _impute_scale(Xtr, Xte)
    pca = PCA(n_components=min(n_components, A.shape[1] - 1), random_state=seed)
    At = pca.fit_transform(A)
    Bt = pca.transform(B)
    return _lgbm_per_target(At, Ytr, Bt, seed=seed, n_estimators=200)


Q1_MODELS = {
    "mean": "均值基线",
    "style_mean": "风格均值基线",
    "ridge": "Ridge（多输出）",
    "pls": "PLS（多输出）",
    "et": "Extra Trees",
    "rf": "Random Forest",
    "lgbm": "LightGBM（逐维）",
    "xgb": "XGBoost（逐维）",
    "svr": "SVR（逐维）",
    "latent_sens_ridge": "8 维感官 PCA 潜状态 + Ridge（PCA8）",
    "latent_sens_pls": "8 维感官 PLS 潜状态（对照）",
    "latent_sens_lgbm": "8 维感官潜状态 + LightGBM",
    "chem14_ridge": "14 维化学潜状态 + Ridge",
    "chem14_lgbm": "14 维化学潜状态 + LightGBM",
}

QUICK_MODELS = ["mean", "style_mean", "ridge", "pls", "et", "rf", "lgbm",
                "latent_sens_ridge", "chem14_ridge"]


def fit_predict(name: str, Xtr, Ytr, Xte, style_te, seed: int, quick: bool):
    if name == "mean":
        return np.tile(Ytr.mean(0), (len(Xte), 1))
    if name == "style_mean":
        raise ValueError("style_mean 需要训练集风格标签，请走 _style_mean_predict")
    if name == "ridge":
        return _ridge_multi(Xtr, Ytr, Xte)
    if name == "pls":
        return _pls_multi(Xtr, Ytr, Xte)
    if name == "et":
        return _et_multi(Xtr, Ytr, Xte, seed)
    if name == "rf":
        return _rf_multi(Xtr, Ytr, Xte, seed)
    if name == "lgbm":
        return _lgbm_per_target(Xtr, Ytr, Xte, seed,
                                n_estimators=120 if quick else 250)
    if name == "xgb":
        return _xgb_per_target(Xtr, Ytr, Xte, seed,
                               n_estimators=120 if quick else 250)
    if name == "svr":
        return _svr_per_target(Xtr, Ytr, Xte, seed)
    if name == "latent_sens_ridge":
        return _latent_target_ridge(Xtr, Ytr, Xte, seed)
    if name == "latent_sens_pls":
        return _latent_target_pls(Xtr, Ytr, Xte, seed)
    if name == "latent_sens_lgbm":
        return _latent_target_lgbm(Xtr, Ytr, Xte, seed)
    if name == "chem14_ridge":
        return _chem_latent_ridge(Xtr, Ytr, Xte, seed)
    if name == "chem14_lgbm":
        return _chem_latent_lgbm(Xtr, Ytr, Xte, seed)
    raise KeyError(name)


def _style_mean_predict(Xtr, Ytr, Xte, styles_tr, styles_te):
    """按风格均值预测；测试风格没在训练集出现时退回全局均值。"""
    global_mean = Ytr.mean(0)
    out = np.tile(global_mean, (len(Xte), 1))
    tab = pd.DataFrame(Ytr, columns=None).groupby(styles_tr).mean()
    for i, s in enumerate(styles_te):
        if s in tab.index:
            out[i] = tab.loc[s].to_numpy()
    return out


def _target_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="raw_values"))
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    return r2, rmse, mae


def _spearman_cols(y_true, y_pred):
    out = np.empty(y_true.shape[1])
    for j in range(y_true.shape[1]):
        a, b = y_true[:, j], y_pred[:, j]
        out[j] = spearmanr(a, b).statistic if np.std(a) > 0 and np.std(b) > 0 else 0.0
    return out


def _nan_median(series) -> float:
    v = pd.to_numeric(series, errors="coerce").dropna()
    return float(v.median()) if len(v) else float("nan")


def run_q1(data: Data, seeds: list[int], quick: bool) -> dict:
    from sklearn.model_selection import GroupKFold, KFold
    t0 = time.time()
    models = QUICK_MODELS if quick else list(Q1_MODELS)
    # 把样本检出率作为技术协变量显式给模型，避免检出率差异被当成化学信号。
    X = np.column_stack([data.X[data.chem_all].to_numpy(float), data.cdr])
    Y = data.Y[data.targets].to_numpy(float)
    styles = data.styles
    n, m = Y.shape

    comparison = []
    per_target_rows = []
    oof_store: dict[tuple[str, str, int], np.ndarray] = {}
    oof_random_best: dict[str, list[np.ndarray]] = {}
    oof_group_best: dict[str, list[np.ndarray]] = {}

    for name in models:
        for scheme in ("random", "group"):
            for seed in seeds:
                if scheme == "random":
                    splits = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(X))
                else:
                    splits = list(GroupKFold(n_splits=5).split(X, Y[:, 0], groups=styles))
                oof = np.full((n, m), np.nan)
                ts = time.time()
                for tr, te in splits:
                    if name == "style_mean":
                        pred = _style_mean_predict(X[tr], Y[tr], X[te], styles[tr], styles[te])
                    else:
                        pred = fit_predict(name, X[tr], Y[tr], X[te], styles[te], seed, quick)
                    oof[te] = pred
                if np.isnan(oof).any():
                    raise RuntimeError(f"{name}/{scheme}/{seed} 的 OOF 有缺失")
                r2, rmse, mae = _target_metrics(Y, oof)
                rho = _spearman_cols(Y, oof)
                comparison.append({
                    "model": name, "model_cn": Q1_MODELS[name], "scheme": scheme,
                    "seed": seed, "runtime_s": round(time.time() - ts, 2),
                    "mean_r2": float(np.mean(r2)), "median_r2": float(np.median(r2)),
                    "min_r2": float(np.min(r2)), "max_r2": float(np.max(r2)),
                    "mean_rmse": float(np.mean(rmse)), "mean_mae": float(np.mean(mae)),
                    "mean_spearman": float(np.mean(rho)),
                })
                for j, target in enumerate(data.targets):
                    per_target_rows.append({
                        "model": name, "scheme": scheme, "seed": seed, "target": target,
                        "r2": float(r2[j]), "rmse": float(rmse[j]),
                        "mae": float(mae[j]), "spearman": float(rho[j]),
                    })
                oof_store[(name, scheme, seed)] = oof
                log.info("Q1 %-18s %-6s seed=%d  R2=%.4f  (%.1fs)",
                         name, scheme, seed, np.mean(r2), time.time() - ts)

    comp = pd.DataFrame(comparison)
    comp.to_csv(Q1 / "comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_target_rows).to_csv(Q1 / "per_target_all.csv", index=False,
                                         encoding="utf-8-sig")

    # 主模型：随机 CV 的 3 种子平均 R² 最高者
    pivot = comp.pivot_table(index=["model", "scheme"], columns="seed", values="mean_r2",
                             aggfunc="mean")
    mean_by = comp.groupby(["model", "scheme"])["mean_r2"].mean().unstack()
    mean_by["avg"] = mean_by.mean(axis=1)
    best = str(mean_by["random"].idxmax())
    log.info("Q1 主模型（随机 CV 平均 R² 最高）：%s", best)

    def avg_oof(scheme: str) -> np.ndarray:
        arrs = [oof_store[(best, scheme, s)] for s in seeds]
        return np.mean(arrs, axis=0)

    oof_rand = avg_oof("random")
    oof_group = avg_oof("group")
    r2r, rmr, mar = _target_metrics(Y, oof_rand)
    r2g, rmg, mag = _target_metrics(Y, oof_group)
    rho_r = _spearman_cols(Y, oof_rand)
    rho_g = _spearman_cols(Y, oof_group)

    per_target = {}
    for j, target in enumerate(data.targets):
        per_target[target] = {
            "r2_random": float(r2r[j]), "rmse_random": float(rmr[j]),
            "mae_random": float(mar[j]), "spearman_random": float(rho_r[j]),
            "r2_group": float(r2g[j]), "rmse_group": float(rmg[j]),
            "mae_group": float(mag[j]), "spearman_group": float(rho_g[j]),
        }
    seed_r2 = comp[(comp.model == best) & (comp.scheme == "random")].sort_values("seed")["mean_r2"].to_numpy()
    seed_r2g = comp[(comp.model == best) & (comp.scheme == "group")].sort_values("seed")["mean_r2"].to_numpy()

    metrics = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "primary_metric": "mean_r2_random",
        "cv": {
            "scheme": "random",
            "mean": float(np.mean(r2r)),
            "std": float(np.std(seed_r2, ddof=1)) if len(seed_r2) > 1 else 0.0,
            "median": float(np.median(r2r)),
            "group_mean": float(np.mean(r2g)),
            "group_std": float(np.std(seed_r2g, ddof=1)) if len(seed_r2g) > 1 else 0.0,
            "group_median": float(np.median(r2g)),
        },
        "seeds": seeds,
        "best_model": best,
        "best_model_cn": Q1_MODELS[best],
        "n_targets": len(data.targets),
        "n_features": len(data.chem_all) + 1,
        "qc_covariate": "sample detection rate (cdr) appended as one technical feature",
        "per_target": per_target,
        "comparison": mean_by.round(6).reset_index().to_dict(orient="records"),
        "notes": [
            "4.csv 的 50 列是标准化品评均值，不是原始 0–6 分；训练与评估都在该尺度上。",
            "随机 CV 与源论文的随机划分可比；分组 CV 按风格留出，衡量对未见风格的泛化。",
            "主模型按随机 CV 平均 R² 选；分组 CV 作为诚实性压力测试，单独报告。",
            "OOF 预测跨 3 个模型种子平均；交叉验证切分固定，避免把切分差异当成模型差异。",
        ],
    }
    save_json(Q1 / "metrics.json", metrics)

    pd.DataFrame(per_target_rows).query("model == @best and scheme == 'random'").to_csv(
        Q1 / "per_target.csv", index=False, encoding="utf-8-sig")

    # 预测表：OOF（随机 / 分组）与全量拟合
    def write_pred(path: Path, pred: np.ndarray):
        df = pd.DataFrame(pred, columns=data.targets)
        df.insert(0, "beer", data.X["beer"].to_numpy())
        df.to_csv(path, index=False, encoding="utf-8-sig")

    write_pred(Q1 / "predictions.csv", np.nan_to_num(oof_rand))
    write_pred(Q1 / "predictions_group.csv", np.nan_to_num(oof_group))

    preds_full = []
    for s in seeds:
        if best == "style_mean":
            preds_full.append(_style_mean_predict(X, Y, X, styles, styles))
        else:
            preds_full.append(fit_predict(best, X, Y, X, styles, s, quick))
    full = np.mean(preds_full, axis=0)
    write_pred(Q1 / "predictions_full.csv", np.nan_to_num(full))

    # 0–6 展示尺度：逐维经验分位映射，不用固定 3+1.5z。
    # 4.csv 是标准化品评均值，官方没有给回算参数；3.xlsx 的 0–6 表是空模板。
    # 因此用每个维度观测值的 1%–99% 分位做仿射映射，避免把所有维度强行拉成
    # 同一个标准差（dms 等长尾维度会被 3+1.5z 明显放大）。
    q01 = np.nanpercentile(Y, 1, axis=0)
    q99 = np.nanpercentile(Y, 99, axis=0)
    span = np.where(q99 - q01 > 1e-9, q99 - q01, 1.0)
    calib = {
        "method": "per_dimension_empirical_quantile_affine",
        "formula": "score_0_6 = clip(6 * (z - q01) / (q99 - q01), 0, 6)",
        "note": "4.csv 给的是标准化品评均值，官方没有提供 0–6 回算参数；"
                "3.xlsx 的 0–6 表是空模板，无法作为锚点。"
                "这是展示尺度，不是原始品评分数；q01/q99 为每个维度观测值的 1%/99% 分位。",
        "per_target_range": {
            t: {"min": float(data.Y[t].min()), "max": float(data.Y[t].max()),
                "mean": float(data.Y[t].mean()), "sd": float(data.Y[t].std()),
                "q01": float(q01[j]), "q99": float(q99[j])}
            for j, t in enumerate(data.targets)
        },
    }
    save_json(Q1 / "calibration.json", calib)
    pred06 = np.clip(6 * (np.nan_to_num(oof_rand) - q01) / span, 0, 6)
    df06 = pd.DataFrame(pred06, columns=data.targets)
    df06.insert(0, "beer", data.X["beer"].to_numpy())
    df06.to_csv(Q1 / "predictions_06.csv", index=False, encoding="utf-8-sig")

    return {
        "best_model": best,
        "mean_r2_random": float(np.mean(r2r)),
        "mean_r2_group": float(np.mean(r2g)),
        "runtime_s": round(time.time() - t0, 1),
        "n_models": len(models),
    }


# ================================================================ Q2

def _normal_scores_matrix(X: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """秩正态得分；检出下限先置缺失，再填到该列最小值以下（保留「未检出」序）。"""
    n = len(X)
    V = X[cols].to_numpy(float)
    V = np.where(_floor_mask(V), np.nan, V)
    fill = np.nanmin(V, axis=0) - 1e-9
    fill = np.where(np.isfinite(fill), fill, 0.0)
    V = np.where(np.isnan(V), fill[None, :], V)
    out = np.empty((n, len(cols)))
    for j in range(len(cols)):
        r = pd.Series(V[:, j]).rank(method="average").to_numpy()
        out[:, j] = norm.ppf((r - 0.5) / n)
    return np.nan_to_num(out, nan=0.0)


def _standardize(A: np.ndarray) -> np.ndarray:
    mu = A.mean(0)
    sd = A.std(0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    return (A - mu) / sd


def _center_by_group(A: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """按组（风格）去均值：固定效应变换，等价于每组一个未惩罚截距。"""
    out = np.empty_like(A, dtype=float)
    for g in np.unique(groups):
        m = groups == g
        out[m] = A[m] - np.nanmean(A[m], axis=0)
    return out


def _lasso_select(A: np.ndarray, Y: np.ndarray, reps: np.ndarray,
                  alpha: float = Q2_ALPHA,
                  masks: list[np.ndarray] | None = None,
                  weights: np.ndarray | None = None,
                  positive: bool = False,
                  track_negative: bool = False):
    """逐目标 Lasso 稳定性选择。

    参数
    ----
    masks : 长度 50 的布尔数组列表；第 j 个给出该感官维度允许进入模型的列。
            不在候选池里的列频率恒为 0、系数恒为 0。
    weights : 样本权重；检出率最低的样本降权，避免仪器技术差异主导入选。
    positive : True 时用 Lasso(positive=True)，候选只能以正向效应进入主排序。
               论文口径要求 Top-3 是「驱动/致香物」，不能混入负向掩蔽物。
    track_negative : 额外返回负向选择频率，供 negative_maskers 辅表使用。

    返回
    ----
    sel : (n_target, p, B) bool，每个 Bootstrap 是否选中该化合物
    coef : (n_target, p)，全样本 Lasso 系数（用于解释方向与大小）
    neg : (n_target, p, B) bool，仅 track_negative=True 时返回
    """
    from sklearn.linear_model import Lasso
    w = None if weights is None else np.asarray(weights, dtype=float)
    n_t, p = Y.shape[1], A.shape[1]
    sel = np.zeros((n_t, p, len(reps)), dtype=bool)
    neg = np.zeros_like(sel) if track_negative else None
    coef = np.zeros((n_t, p))
    for j in range(n_t):
        cols = np.arange(p) if masks is None else np.flatnonzero(masks[j])
        if len(cols) == 0:
            continue
        y = Y[:, j]
        coef[j, cols] = Lasso(alpha=alpha, max_iter=5000,
                              positive=positive).fit(
            A[:, cols], y, sample_weight=w).coef_
        for b, idx in enumerate(reps):
            cb = Lasso(alpha=alpha, max_iter=5000, positive=positive).fit(
                A[idx][:, cols], y[idx],
                sample_weight=None if w is None else w[idx]).coef_
            sel[j, cols, b] = np.abs(cb) > 1e-8
            if track_negative:
                neg[j, cols, b] = cb < -1e-8
    if track_negative:
        return sel, coef, neg
    return sel, coef


def _module_evidence(X: np.ndarray, Y: np.ndarray, modules: list[list[int]],
                     targets: list[str]) -> np.ndarray:
    """模块 PC1 与感官的 Spearman × 化合物在 PC1 上的载荷。"""
    from sklearn.decomposition import PCA
    A = _impute_scale(X, X)[0]
    E = np.zeros((A.shape[1], Y.shape[1]))
    for mod in modules:
        if len(mod) == 0:
            continue
        if len(mod) == 1:
            score = A[:, mod[0]]
            load = np.ones(1)
        else:
            pca = PCA(n_components=1, random_state=SEED)
            score = pca.fit_transform(A[:, mod])[:, 0]
            load = np.abs(pca.components_[0])
            if load.max() > 0:
                load = load / load.max()
        for j in range(Y.shape[1]):
            rho = spearmanr(score, Y[:, j]).statistic
            if not np.isfinite(rho):
                rho = 0.0
            E[mod, j] = abs(rho) * load
    return E


def run_q2(data: Data, quick: bool) -> dict:
    import networkx as nx
    t0 = time.time()
    cand = [c for c in data.chem_nd if c not in Q2_EXCLUDE]
    Xc = data.X[cand].to_numpy(float)
    Y = data.Y[data.targets].to_numpy(float)
    names = [data.chem_map[c] for c in cand]
    n = len(names)
    fam = _family_map(data)
    sn = load_tables().sensory_names.drop_duplicates("R_name").set_index("R_name")
    tcat = sn.reindex(data.targets)["Category"].fillna("").astype(str).to_dict()

    C = pd.read_csv(RAW / "2.csv", index_col=0)
    C = C.loc[names, names]
    A = C.to_numpy()

    # 共表达图：每个节点取 |rho| 最大的 10 个邻居，取并集
    k_nn = 10
    edges = {}
    for i in range(n):
        idx = np.argsort(-np.abs(A[i]))[:k_nn + 1]
        for j in idx:
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges[(a, b)] = float(A[i, j])
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (i, j), w in edges.items():
        G.add_edge(i, j, weight=w, abs_weight=abs(w))
    communities = nx.community.louvain_communities(G, weight="abs_weight",
                                                    seed=SEED, resolution=1.0)
    communities = sorted(communities, key=len, reverse=True)
    mod_id = np.full(n, -1)
    for m, mod in enumerate(communities):
        for i in mod:
            mod_id[i] = m
    modularity = nx.community.modularity(G, communities, weight="abs_weight")

    # 主证据：逐感官 Lasso 稳定性选择（按风格分层 Bootstrap）
    # 香气维度（A_* 与嗅后 F_hops/F_malt/F_esters）做风格内去均值，并限制为
    # 挥发性有机化合物；口感/风味维度保留全候选池，因为糖、苦味质、金属离子
    # 本来就是这些维度的直接化学对应物。
    # 检出率作为技术协变量加入 Lasso，但不进入 Top-3 排名。
    Xc_qc = np.column_stack([Xc, data.cdr])
    Ain_all = _impute_scale(Xc_qc, Xc_qc)[0]
    Ain = Ain_all[:, :-1]
    n_boot = 30 if quick else Q2_BOOTSTRAP
    rng = np.random.default_rng(SEED)
    style_idx = {s: np.where(data.styles == s)[0] for s in np.unique(data.styles)}
    reps = [np.concatenate([rng.choice(v, size=len(v), replace=True)
                            for v in style_idx.values()]) for _ in range(n_boot)]
    sample_w = np.where(data.floor_heavy, 0.5, 1.0)

    is_aroma = np.array([_is_aroma_target(t) for t in data.targets])
    vol_mask = np.r_[np.array([c not in Q2_NONVOLATILE for c in cand]), True]
    full_mask = np.ones(len(cand) + 1, dtype=bool)
    masks = [vol_mask if a else full_mask for a in is_aroma]

    A_adj_all = _center_by_group(Ain_all, data.styles)
    A_adj = A_adj_all[:, :-1]
    Y_adj = _center_by_group(Y, data.styles)
    # 口径必须逐维一致：香气维度用风格内中心化矩阵，11 个口感/风味维度用
    # 未中心化的全样本矩阵。旧实现把 50 维统一传 A_adj/Y_adj，导致口感维度
    # 的 coef（风格内）与 corr（全样本）错位，乳酸/酒精等直接对应物被压掉。
    idx_ar = np.flatnonzero(is_aroma)
    idx_tp = np.flatnonzero(~is_aroma)
    n_col = Ain_all.shape[1]
    sel_all = np.zeros((len(data.targets), n_col, n_boot), dtype=bool)
    coef_all = np.zeros((len(data.targets), n_col))
    sel_all[idx_ar], coef_all[idx_ar] = _lasso_select(
        A_adj_all, Y_adj[:, idx_ar], reps,
        masks=[masks[j] for j in idx_ar], weights=sample_w, positive=True)
    sel_all[idx_tp], coef_all[idx_tp] = _lasso_select(
        Ain_all, Y[:, idx_tp], reps,
        masks=[full_mask] * len(idx_tp), weights=sample_w, positive=True)
    sel = sel_all[:, :n, :]
    coef = coef_all[:, :n]
    freq = sel.mean(2).T  # n×50：每个化合物在 50 个维度上的正向入选频率

    # 负向掩蔽物辅表：另跑一次有符号 Lasso，只用于报告「抑制/掩蔽」效应，
    # 不进入主排序（主表只收正向激动剂）。
    coef_signed_all = np.zeros((len(data.targets), n_col))
    neg_freq_all = np.zeros((len(data.targets), n_col))
    _, coef_signed_all[idx_ar], neg_ar = _lasso_select(
        A_adj_all, Y_adj[:, idx_ar], reps,
        masks=[masks[j] for j in idx_ar], weights=sample_w,
        track_negative=True)
    _, coef_signed_all[idx_tp], neg_tp = _lasso_select(
        Ain_all, Y[:, idx_tp], reps,
        masks=[full_mask] * len(idx_tp), weights=sample_w,
        track_negative=True)
    neg_freq_all[idx_ar] = neg_ar.mean(2)
    neg_freq_all[idx_tp] = neg_tp.mean(2)
    coef_signed = coef_signed_all[:, :n]
    neg_freq = neg_freq_all[:, :n].T  # n×50，与 freq/corr 同向

    # 一致性证据：香气维度用风格内 Spearman，其余用全样本 Spearman。
    # 顺序必须是 (化合物, 感官)，否则转置后维度对不上。
    # 与 coef 使用同一份插补后矩阵（未中心化），避免原始哨兵值把秩相关压低。
    corr_raw = pd.concat([pd.DataFrame(Ain, columns=cand),
                          pd.DataFrame(Y, columns=data.targets)], axis=1).corr(
        method="spearman").loc[cand, data.targets].to_numpy()
    corr_within = pd.concat([pd.DataFrame(A_adj[:, :], columns=cand),
                             pd.DataFrame(Y_adj, columns=data.targets)], axis=1).corr(
        method="spearman").loc[cand, data.targets].to_numpy()
    corr_raw = np.nan_to_num(corr_raw, nan=0.0)
    corr_within = np.nan_to_num(corr_within, nan=0.0)
    corr = np.where(is_aroma[None, :], corr_within, corr_raw)
    modules_idx = [[i for i in mod] for mod in communities]
    module = _module_evidence(Xc, Y, modules_idx, data.targets)

    # 共表达图参与排序：对入选频率做图拉普拉斯平滑。
    # 与某感官相关的不应只是一个孤立化合物，而应是同一模块里成组出现的化合物。
    # gamma 事先固定（Q2_GRAPH_GAMMA），不按结果挑；下面的分半 Jaccard 与
    # 相对 gamma=0 的重合度只用于报告敏感性。
    W = np.zeros((n, n), dtype=float)
    for (i, j), w in edges.items():
        W[i, j] = W[j, i] = abs(float(w))
    row = W.sum(1, keepdims=True)
    P = np.divide(W, row, out=np.zeros_like(W), where=row > 0)

    def _smooth(f, gamma: float):
        return (1.0 - gamma) * f + gamma * (P @ f)

    def _eligible(j: int) -> np.ndarray:
        """主表候选：正系数且正 Spearman 的正向激动剂。

        极少数维度若正向候选不足 3 个，退回「正系数」集合；再不足则退回全池，
        并在报告里由 low_stability 与 pool 字段暴露，不做静默伪造。
        """
        e = np.flatnonzero((coef[j] > 0) & (corr[:, j] > 0))
        if len(e) < 3:
            e = np.flatnonzero(coef[j] > 0)
        return e if len(e) else np.arange(n)

    def _top3_from(f: np.ndarray, j: int) -> list[int]:
        e = _eligible(j)
        order = [int(i) for i in e[np.lexsort((-np.abs(corr[e, j]), -f[e, j]))]]
        if len(order) < 3:  # 正向候选不足时补位，避免 Top-3 出现空值
            rest = [int(i) for i in np.argsort(-f[:, j])
                    if int(i) not in order and coef[j, i] > 0]
            order += rest
        if len(order) < 3:
            order += [int(i) for i in np.argsort(-f[:, j]) if int(i) not in order]
        return order[:3]

    n_split = 5
    split_pairs = []
    for _ in range(n_split):
        perm = rng.permutation(n_boot)
        split_pairs.append((perm[:n_boot // 2], perm[n_boot // 2:]))

    gamma_grid = [0.0, 0.15, 0.30, 0.50]
    gamma_jac = {}
    for gamma in gamma_grid:
        js = []
        for j in range(len(data.targets)):
            for h1, h2 in split_pairs:
                f1 = _smooth(sel[j][:, h1].mean(1), gamma)
                f2 = _smooth(sel[j][:, h2].mean(1), gamma)
                e = _eligible(j)
                s1 = set(e[np.argsort(-f1[e])[:3]])
                s2 = set(e[np.argsort(-f2[e])[:3]])
                js.append(len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0)
        gamma_jac[gamma] = float(np.median(js))
    # gamma 事先写死；分半 Jaccard 只用来证明这个权重没有牺牲稳定性。
    graph_gamma = Q2_GRAPH_GAMMA
    score = _smooth(freq, graph_gamma)

    base_sets = [_top3_from(freq, j) for j in range(len(data.targets))]
    base_sets = [set(x) for x in base_sets]
    gamma_overlap = {}
    for gamma in gamma_grid:
        sg = _smooth(freq, gamma)
        sets = [_top3_from(sg, j) for j in range(len(data.targets))]
        sets = [set(x) for x in sets]
        gamma_overlap[gamma] = float(np.mean([
            len(a & b) / len(a | b) if (a | b) else 0.0 for a, b in zip(base_sets, sets)]))

    # Top-3：只在正向激动剂里选；图平滑后的选择频率优先，|Spearman| 破平
    top3_idx = [_top3_from(score, j) for j in range(len(data.targets))]

    # 稳定性：5 次随机分半，比较两半各自（图平滑后）的 Top-3
    stable_rate, med_jac, jac_all, jac_full = [], [], [], []
    for j, target in enumerate(data.targets):
        js = []
        e = _eligible(j)
        for h1, h2 in split_pairs:
            f1 = _smooth(sel[j][:, h1].mean(1), graph_gamma)
            f2 = _smooth(sel[j][:, h2].mean(1), graph_gamma)
            s1 = set(e[np.argsort(-f1[e])[:3]])
            s2 = set(e[np.argsort(-f2[e])[:3]])
            js.append(len(s1 & s2) / len(s1 | s2) if (s1 | s2) else 0.0)
        med_jac.append(float(np.median(js)))
        jac_all.extend(js)
        stable_rate.append(float(np.mean(freq[top3_idx[j], j])))
        s_full = set(e[np.argsort(-coef[j][e])[:3]])
        s_cons = set(top3_idx[j])
        jac_full.append(len(s_full & s_cons) / len(s_full | s_cons) if
                        (s_full | s_cons) else 0.0)
    median_jaccard = float(np.median(jac_all))
    n_unstable = int(sum(sr < 0.6 for sr in stable_rate))

    # 方法敏感性：同一套 Bootstrap 下比较
    #   ① 仅换候选池（不做风格校正）      ② 仅做风格校正（不换池）
    #   ③ 旧口径（全候选池 + 不校正）      ④ 主口径
    # 用来回答「铁/蛋白/DMS 的入选到底是质控问题、风格混杂，还是方法选择」。
    def _top3_sets(fr):
        return [set(np.argsort(-fr[:, j])[:3]) for j in range(fr.shape[1])]

    def _overlap(a, b):
        return float(np.mean([len(x & y) / len(x | y) if (x | y) else 0.0
                              for x, y in zip(a, b)]))

    def _sens_freq(A, YY, mm):
        return _lasso_select(A, YY, reps, masks=mm,
                             weights=sample_w, positive=True)[0].mean(2).T[:n]

    sens = {k: _smooth(v, graph_gamma) for k, v in {
        "pooled_samepool": _sens_freq(Ain_all, Y, masks),
        "adjusted_fullpool": _sens_freq(A_adj_all, Y_adj, None),
        "pooled_fullpool": _sens_freq(Ain_all, Y, None),
    }.items()}
    main_sets = _top3_sets(score)
    sens_rows = []
    for name, fr in sens.items():
        s3 = _top3_sets(fr)
        n_bad = {c: int(sum(c in {cand[i] for i in s3[j]} for j in range(len(data.targets))))
                 for c in sorted(Q2_NONVOLATILE) if c in cand}
        sens_rows.append({"scheme": name, "top3_jaccard_vs_main": _overlap(main_sets, s3),
                          "nonvolatile_top3_counts": n_bad})
        for j, t in enumerate(data.targets):
            sens_rows.append({"scheme": name, "target": t,
                              "top3": "|".join(cand[i] for i in sorted(s3[j]))})
    save_json(Q2 / "sensitivity_methods.json", {
        "n_bootstrap": n_boot,
        "primary": "香气维度=风格内中心化+挥发性池；其余=全池不校正",
        "variants": ["pooled_samepool", "adjusted_fullpool", "pooled_fullpool"],
        "nonvolatile_pool": sorted(Q2_NONVOLATILE),
        "summary": [r for r in sens_rows if "scheme" in r and "top3" not in r],
    })
    pd.DataFrame([r for r in sens_rows if "top3" in r]).to_csv(
        Q2 / "sensitivity_top3.csv", index=False, encoding="utf-8-sig")

    # 输出
    pd.DataFrame(freq, index=cand, columns=data.targets).to_csv(
        Q2 / "selection_freq.csv", encoding="utf-8-sig")
    pd.DataFrame(module, index=cand, columns=data.targets).to_csv(
        Q2 / "module_evidence.csv", encoding="utf-8-sig")
    pd.DataFrame({"beer": data.X["beer"].to_numpy(), "cdr": data.cdr,
                  "floor_heavy": data.floor_heavy}).to_csv(
        Q2 / "sample_qc.csv", index=False, encoding="utf-8-sig")
    with open(Q2 / "graph_edges.csv", "w", encoding="utf-8-sig", newline="") as f:
        f.write("source,target,weight,abs_weight,community_source,community_target\n")
        for (i, j), w in sorted(edges.items()):
            f.write(f"{names[i]},{names[j]},{w:.8f},{abs(w):.8f},{mod_id[i]},{mod_id[j]}\n")

    pd.DataFrame({
        "node": names, "community": mod_id,
        "degree": [G.degree(i) for i in range(n)],
        "module_size": [len(communities[mod_id[i]]) for i in range(n)],
    }).to_csv(Q2 / "communities.csv", index=False, encoding="utf-8-sig")

    def _family_match(cfam: str, tcat: str) -> str:
        """只有目标本身有生化家族时才判匹配；Others/缺失记 na。"""
        a, b = str(cfam).strip().lower(), str(tcat).strip().lower()
        if not a or not b or b in ("others", "nan", "none"):
            return "na"
        return "yes" if (b in a or a in b) else "no"

    rows = []
    for j, target in enumerate(data.targets):
        picks = top3_idx[j]
        tcat_j = str(tcat.get(target, ""))
        ns = Q2_NAMESAKE.get(target)
        rho_f = rho_w = ns_freq = ns_rank = np.nan
        if ns in cand:
            i_ns = cand.index(ns)
            rho_f, rho_w = float(corr_raw[i_ns, j]), float(corr_within[i_ns, j])
            ns_freq = float(freq[i_ns, j])
            e_ns = _eligible(j)
            if i_ns in e_ns:
                ns_rank = int(1 + np.sum(score[e_ns, j] > score[i_ns, j]))
            else:
                ns_rank = 0  # 同名化合物不是正向候选
        row = {"sensory": target, "stable_rate": stable_rate[j],
               "jaccard_median": med_jac[j], "jaccard_vs_full": jac_full[j],
               "pool": "volatile" if is_aroma[j] else "full",
               "style_adjusted": bool(is_aroma[j]),
               "target_family": tcat_j,
               "low_stability": bool(stable_rate[j] < 0.6 or med_jac[j] < 0.5),
               "namesake": ns if ns else "",
               "namesake_rho": rho_f, "namesake_rho_within": rho_w,
               "namesake_freq": ns_freq, "namesake_rank": ns_rank}
        for r, i in enumerate(picks, 1):
            cfam = fam.get(cand[i], "")
            row[f"top{r}"] = cand[i]
            row[f"top{r}_full"] = names[i]
            row[f"top{r}_family"] = cfam
            row[f"top{r}_family_match"] = _family_match(cfam, tcat_j)
            row[f"freq{r}"] = float(freq[i, j])
            row[f"graph_score{r}"] = float(score[i, j])
            row[f"corr{r}"] = float(corr[i, j])
            row[f"coef{r}"] = float(coef[j][i])
            row[f"module{r}"] = int(mod_id[i])
            row[f"module_size{r}"] = len(communities[mod_id[i]])
            row[f"module_ev{r}"] = float(module[i, j])
            row[f"degree{r}"] = int(G.degree(i))
        rows.append(row)
    pd.DataFrame(rows).to_csv(Q2 / "top3.csv", index=False, encoding="utf-8-sig")

    # 负向掩蔽物辅表：只报告有符号 Lasso 中稳定为负、且 Spearman 也为负的
    # 化合物。它们不进入主 Top-3，但可支撑「跨模态感官掩蔽」讨论。
    neg_rows = []
    for j, target in enumerate(data.targets):
        e = np.flatnonzero((coef_signed[j] < 0) & (corr[:, j] < 0))
        if len(e) == 0:
            continue
        order = e[np.lexsort((coef_signed[j, e], -neg_freq[e, j]))]
        for r, i in enumerate(order[:10], 1):
            neg_rows.append({
                "sensory": target, "rank": r, "compound": cand[i],
                "full_name": names[i],
                "coef_signed": float(coef_signed[j, i]),
                "spearman": float(corr[i, j]),
                "neg_freq": float(neg_freq[i, j]),
                "module": int(mod_id[i]),
                "module_size": len(communities[mod_id[i]]),
                "pool": "volatile" if is_aroma[j] else "full",
                "style_adjusted": bool(is_aroma[j]),
            })
    pd.DataFrame(neg_rows).to_csv(
        Q2 / "negative_maskers.csv", index=False, encoding="utf-8-sig")

    save_json(Q2 / "stability.json", {
        "n_bootstrap": n_boot, "n_split": n_split,
        "alpha": Q2_ALPHA, "median_jaccard": median_jaccard,
        "n_unstable": n_unstable, "stable_rate_threshold": 0.6,
        "style_adjusted_aroma": True,
        "volatile_pool_aroma": True,
        "graph_weighting": True, "graph_gamma": graph_gamma,
        "graph_gamma_jaccard": {str(g): v for g, v in gamma_jac.items()},
        "qc_covariate": "sample detection rate",
        "floor_heavy_n": int(data.floor_heavy.sum()),
        "floor_heavy_weight": 0.5,
        "per_target": {t: {"jaccard_median": med_jac[j],
                           "stable_rate": stable_rate[j],
                           "low_stability": bool(stable_rate[j] < 0.6 or med_jac[j] < 0.5),
                           "pool": "volatile" if is_aroma[j] else "full",
                           "style_adjusted": bool(is_aroma[j])}
                       for j, t in enumerate(data.targets)},
    })

    # 证据矩阵（50×231 口径，被排除列置 0），供三问互证
    ev = np.zeros((len(data.targets), len(data.chem_nd)))
    pos = {c: i for i, c in enumerate(data.chem_nd)}
    for i, c in enumerate(cand):
        ev[:, pos[c]] = freq[i]
    pd.DataFrame(ev, index=data.targets, columns=data.chem_nd).to_csv(
        Q2 / "evidence_scores.csv", encoding="utf-8-sig")

    # 合理性表：来源文献、异味标记、6.csv 交叉相关
    th = load_tables().thresholds.copy()
    th["chemical_name"] = th["chemical_name"].astype(str).map(fix_mojibake)
    origin, off = {}, {}
    for r in th.itertuples():
        nm = str(r.chemical_name)
        origin[nm] = str(r.origin_paper) if pd.notna(r.origin_paper) else ""
        off[nm] = str(r.off_flavors) if pd.notna(r.off_flavors) else ""
    cross = pd.read_csv(RAW / "6.csv", index_col=0)
    plaus = []
    for j, target in enumerate(data.targets):
        panel = data.sens_map.get(target, target)
        for r, i in enumerate(top3_idx[j], 1):
            nm = names[i]
            cc = np.nan
            if nm in cross.index and panel in cross.columns:
                cc = float(cross.loc[nm, panel])
            plaus.append({
                "sensory": target, "rank": r, "compound": cand[i],
                "full_name": nm, "origin_paper": origin.get(nm, ""),
                "off_flavor": off.get(nm, ""), "cross_corr_6csv": cc,
                "community": int(mod_id[i]), "module_size": len(communities[mod_id[i]]),
                "degree": G.degree(i), "stable_freq": float(freq[i, j]),
            })
    pd.DataFrame(plaus).to_csv(Q2 / "plausibility.csv", index=False, encoding="utf-8-sig")

    hub = sorted([(names[i], G.degree(i)) for i in range(n)], key=lambda x: -x[1])[:10]
    report = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "graph": {
            "n_nodes": n, "n_edges": int(G.number_of_edges()),
            "mean_degree": float(np.mean([G.degree(i) for i in range(n)])),
            "n_communities": len(communities),
            "modularity": float(modularity),
            "ranking_weight_gamma": graph_gamma,
            "gamma_split_half_jaccard": {str(g): v for g, v in gamma_jac.items()},
            "gamma_top3_jaccard_vs_gamma0": {str(g): v for g, v in gamma_overlap.items()},
            "top_hubs": [{"compound": a, "degree": b} for a, b in hub],
        },
        "method": {
            "model": "Lasso 稳定性选择 + 共表达图平滑",
            "alpha": Q2_ALPHA, "n_bootstrap": n_boot, "n_split": n_split,
            "style_adjusted": "香气维度在风格内中心化（固定效应）；其余 11 个口感/风味维度"
                              "使用未中心化的全样本矩阵（coef 与 corr 严格同口径）",
            "sign_constraint": "主表只收正向激动剂：Lasso(positive=True)，且 coef>0、"
                              "Spearman>0；负向掩蔽物单独输出到 negative_maskers.csv",
            "graph_ranking": "score = (1-gamma)*selection_frequency + gamma*neighbor_mean_frequency；"
                             "gamma 事先固定 0.20，分半复现 Jaccard 仅作稳健性检查",
            "qc_handling": "检出下限哨兵值置缺失后再中位数插补；样本检出率作为技术协变量；"
                           "检出率最低 2.5% 的样本在 Bootstrap 中权重 0.5",
            "candidate_pool": {
                "aroma": "A_*、F_hops_*/F_malt_*/F_esters_* 及 9 个单香调维度：挥发性有机化合物",
                "taste_palate": "其余 11 维（酸/苦/甜/金属/乳酸/乙酸/酒精/余味/酒体/CO2/总体）："
                                "全候选池（含糖、苦味质、金属离子等直接测量）",
            },
            "excluded": sorted(Q2_EXCLUDE | set(DERIVED)),
            "aroma_excluded_nonvolatile": sorted(Q2_NONVOLATILE - Q2_EXCLUDE - set(DERIVED)),
        },
        "stability": {
            "n_bootstrap": n_boot, "median_jaccard": median_jaccard,
            "n_unstable": n_unstable, "stable_rate_threshold": 0.6,
            "median_jaccard_vs_full": float(np.median(jac_full)),
            "per_target": {t: {"jaccard_median": med_jac[j], "stable_rate": stable_rate[j]}
                           for j, t in enumerate(data.targets)},
        },
        "notes": [
            "2.csv 是 Spearman 相关矩阵，已由 D5 用 1.csv 复算到 1e-15。",
            "全局排除 5 个派生汇总列、pH、色度、内标 3-methyl-2-butenal。",
            "香气维度额外排除铁、蛋白、糖、苦味质、β-葡聚糖、不可发酵物、甘油、"
            "乳酸、CO2 以及 SO2/BHT/氨等防腐与工艺测量；这些列不挥发，"
            "只可能通过风格混杂进入 Top-3。",
            "香气维度在风格内去均值（固定效应），避免把「某风格同时更香、铁更高」"
            "误判为化合物效应；口感维度保留全池，因为糖/苦味质/铁是其直接对应物。",
            "主证据为按风格分层 Bootstrap 的正向 Lasso 入选频率；|Spearman| 仅用于破平。",
            "负向掩蔽物不进入主 Top-3：另跑有符号 Lasso，输出去 negative_maskers.csv，"
            "用于讨论甜-酸拮抗等跨模态掩蔽，不作为关键驱动化合物。",
            "方法敏感性见 sensitivity_methods.json（仅换池 / 仅校正 / 旧口径）。",
            "稳定性为 5 次随机分半、两半各自 Top-3 的 Jaccard 中位数。",
            "共表达图参与排序：先对入选频率做图拉普拉斯平滑，再取 Top-3；"
            "平滑强度 gamma 事先固定 0.20（80% 选择频率 + 20% 邻居平均频率），"
            "不按测试结果调整；报告给出不同 gamma 下的分半 Jaccard 与相对 gamma=0 的重合度。",
            "QC 已接入：每列检出下限哨兵值先置缺失；样本检出率作为技术协变量；"
            "检出率最低 2.5% 的样本在 Bootstrap 中降权到 0.5。",
            "dms / diacetyl / t2n / stale_hops 的感官分与其同名化合物相关极弱"
            "（|rho|<0.09），这四维 Top-3 只能作探索性结果，论文须标注。",
        ],
    }
    op_file = RESULTS / "supplementary" / "upgrade" / "q2_operator.json"
    if op_file.exists():
        op = json.loads(op_file.read_text(encoding="utf-8"))
        report["method"]["graph_operator_robustness"] = {
            "source": "results/supplementary/upgrade/q2_operator.json",
            "note": "改用一阶邻域平滑 / 图拉普拉斯扩散 / 带重启随机游走后，"
                    "Top-3 与主算子的 Jaccard 及分半稳定性；均使用与主流程相同的"
                    "正向 Lasso、混合中心化与候选集。",
            "n_boot": op.get("n_boot"), "gamma": op.get("gamma"),
            "summary": op.get("summary"),
        }
    save_json(Q2 / "report.json", report)
    return {"n_nodes": n, "n_edges": int(G.number_of_edges()),
            "n_communities": len(communities), "modularity": float(modularity),
            "median_jaccard": median_jaccard, "n_unstable": n_unstable,
            "runtime_s": round(time.time() - t0, 1)}


# ================================================================ Q3 工具

def _sqrtm_psd(S: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh((S + S.T) / 2)
    return V @ np.diag(np.sqrt(np.clip(w, 1e-10, None))) @ V.T


def _gaussian_ot(m0, S0, m1, S1, z):
    h0 = _sqrtm_psd(S0)
    h0i = np.linalg.pinv(h0)
    A = h0i @ _sqrtm_psd(h0 @ S1 @ h0) @ h0i
    return m1 + A @ (z - m0)


def _moments(Z, mask, Spool, lam=20.0, lam_mean=3.0):
    Zc = Z[mask]
    n = len(Zc)
    m = (n * Zc.mean(0) + lam_mean * Z.mean(0)) / (n + lam_mean)
    S = (n * np.cov(Zc, rowvar=False) + lam * Spool) / (n + lam)
    w, V = np.linalg.eigh((S + S.T) / 2)
    S = V @ np.diag(np.clip(w, 1e-8, None)) @ V.T
    return m, S


def _raw_quantile_tables(X: pd.DataFrame, cols: list[str]):
    V = X[cols].to_numpy(float)
    V = np.where(_floor_mask(V), np.nan, V)
    tables = []
    for j in range(len(cols)):
        v = V[:, j]
        v = v[np.isfinite(v)]
        tables.append(np.sort(v))
    return tables


def _ns_to_raw(ns: np.ndarray, tables: list[np.ndarray]) -> np.ndarray:
    out = np.empty((len(ns), len(tables)))
    for j, sv in enumerate(tables):
        p = np.clip(norm.cdf(ns[:, j]), 1e-4, 1 - 1e-4)
        out[:, j] = np.quantile(sv, p, method="linear")
    return out


def _raw_to_ns(raw: np.ndarray, tables: list[np.ndarray], n_obs: int) -> np.ndarray:
    out = np.empty_like(raw)
    for j, sv in enumerate(tables):
        p = (np.searchsorted(sv, raw[:, j], side="right") + 0.5) / (n_obs + 1)
        p = np.clip(p, 1e-4, 1 - 1e-4)
        out[:, j] = norm.ppf(p)
    return out


def run_q3(data: Data, quick: bool) -> dict:
    t0 = time.time()
    from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    from scipy.spatial.distance import cdist

    X = data.X[data.chem_nd].to_numpy(float)
    Xdf = data.X[data.chem_nd]
    Y = data.Y[data.targets].to_numpy(float)
    styles = data.styles
    style_names = sorted(np.unique(styles))
    n, p = X.shape
    k = LATENT_K

    Ns = _normal_scores_matrix(Xdf, data.chem_nd)
    mu_ns, sd_ns = Ns.mean(0), Ns.std(0, ddof=1)
    sd_ns = np.where(sd_ns > 0, sd_ns, 1.0)
    M = (Ns - mu_ns) / sd_ns
    # 把样本检出率从每个化学特征里线性剔除，避免 PCA 第一轴变成仪器技术轴。
    c0 = data.cdr - data.cdr.mean()
    beta = (M.T @ c0) / max(float(c0 @ c0), 1e-12)
    M_adj = M - np.outer(c0, beta)
    cdr_r2 = 1.0 - (M_adj ** 2).sum(0) / np.maximum((M ** 2).sum(0), 1e-12)
    M = M_adj
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    Z = U[:, :k] * S[:k]
    evr = float((S[:k] ** 2).sum() / (S ** 2).sum())
    Spool = np.cov(Z, rowvar=False)
    mu = Z.mean(0)
    Si = np.linalg.pinv(np.cov(Z, rowvar=False))
    maha = lambda z: float(np.sqrt((z - mu) @ Si @ (z - mu)))
    maha_99 = float(np.percentile([maha(z) for z in Z], 99))

    raw_tables = _raw_quantile_tables(data.X, data.chem_nd)

    # ---------- 22 个风格的条件高斯生成 ----------
    n_per = 10 if quick else 20
    rng = np.random.default_rng(SEED)
    gen_rows, gen_z = [], []
    for s in style_names:
        mask = styles == s
        m_s, S_s = _moments(Z, mask, Spool)
        zgen = rng.multivariate_normal(m_s, S_s, size=n_per, method="eigh")
        gen_z.append(zgen)
        Mgen = zgen @ Vt[:k]
        Nsgen = Mgen * sd_ns + mu_ns
        raw = _ns_to_raw(Nsgen, raw_tables)
        for r in range(n_per):
            rec = {"style": s, "sample_id": f"{s.replace(' ', '_').replace('/', '_')}_{r+1:02d}",
                   "source": "latent_gaussian"}
            for j, c in enumerate(data.chem_nd):
                rec[c] = float(raw[r, j])
            gen_rows.append(rec)
    gen = pd.DataFrame(gen_rows)
    # 派生列用同风格均值补齐（生成模型只建了 226 个非派生特征）
    style_mean_derived = data.X.groupby("tasting_category_fine")[DERIVED].mean()
    global_mean_derived = data.X[DERIVED].mean()
    for c in DERIVED:
        gen[c] = gen["style"].map(style_mean_derived[c]).fillna(global_mean_derived[c])
    gen = gen[["style", "sample_id"] + data.chem_all + ["source"]]
    gen.to_csv(Q3 / "generated.csv", index=False, encoding="utf-8-sig")
    Zgen = np.vstack(gen_z)

    # ---------- 生成质量：真实 vs 生成的分类器 + 能量距离 ----------
    Zall = np.vstack([Z, Zgen])
    ylab = np.r_[np.zeros(len(Z)), np.ones(len(Zgen))]
    clf = LogisticRegression(max_iter=2000, C=1.0)
    prob = cross_val_predict(clf, Zall, ylab, cv=StratifiedKFold(5, shuffle=True,
                                                                 random_state=SEED),
                             method="predict_proba")[:, 1]
    auc = float(roc_auc_score(ylab, prob))
    energy = []
    for si, s in enumerate(style_names):
        A = Z[styles == s]
        B = Zgen[si * n_per:(si + 1) * n_per]
        if len(A) < 2 or len(B) < 2:
            continue
        d = 2 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean()
        scale = np.median(cdist(A, A)) + 1e-12
        energy.append(float(max(d, 0) / scale))
    energy_med = float(np.median(energy)) if energy else float("nan")

    # ---------- 感官读出器：Ridge 固定在全部 231 列上，训练/预测同一套 pipeline ----------
    from sklearn.linear_model import Ridge as _R
    Xall = data.X[data.chem_all].to_numpy(float)
    Xgen_all = gen[data.chem_all].to_numpy(float)
    A_all, A_gen = _impute_scale(Xall, Xgen_all)
    sens_ridge = _R(alpha=10.0).fit(A_all, Y)
    sens_gen = sens_ridge.predict(A_gen)
    gs = pd.DataFrame(sens_gen, columns=data.targets)
    gs.insert(0, "style", gen["style"].to_numpy())
    gs.insert(1, "sample_id", gen["sample_id"].to_numpy())
    gs.to_csv(Q3 / "generated_sensory.csv", index=False, encoding="utf-8-sig")

    style_corr, style_rmse = [], []
    for s in style_names:
        real = Y[styles == s].mean(0)
        genm = sens_gen[(gen["style"] == s).to_numpy()].mean(0)
        r = spearmanr(real, genm).statistic
        style_corr.append(float(r) if np.isfinite(r) else 0.0)
        style_rmse.append(float(np.sqrt(np.mean((real - genm) ** 2))))
    sens_corr_mean = float(np.mean(style_corr))
    sens_rmse_mean = float(np.mean(style_rmse))

    # ---------- 生成样本的缺陷合规 ----------
    def _concat(v, scale):
        return 10 ** v if scale == "log10" else v
    gen_defect = {}
    for s in style_names:
        sub = gen[gen["style"] == s]
        real = data.X[styles == s]
        rec = {}
        for d in ("DMS", "X4VG", "X2.3.butanedione"):
            thr = DEFECTS[d]["threshold"]
            gv = _concat(sub[d].to_numpy(float), DEFECTS[d]["scale"])
            rv = _concat(real[d].to_numpy(float), DEFECTS[d]["scale"])
            rec[d] = {
                "threshold": thr, "unit": DEFECTS[d]["unit"],
                "gen_exceed_rate": float(np.mean(gv > thr * (1 + 1e-6))),
                "real_exceed_rate": float(np.mean(rv > thr * (1 + 1e-6))),
            }
        gen_defect[s] = rec

    # ---------- Pils/Lager → Tripel：潜空间 OT + 特征空间逐坐标约束投影 ----------
    P = styles == "Pils/Lager"
    T = styles == "Tripel"
    mp, Sp = _moments(Z, P, Spool)
    mt, St = _moments(Z, T, Spool)
    dm = mt - mp
    trip_sens = Y[T].mean(0)
    pils_sens = Y[P].mean(0)
    src_idx = np.where(P)[0]
    tgt_idx = np.where(T)[0]

    # 原始特征空间的逐坐标约束投影：目标函数可分，约束也是逐坐标上界，
    # 所以闭式解 x* = min((x_ot + λ x_src)/(1+λ), 上界)，再裁到观测范围。
    X_clean = np.where(_floor_mask(X), np.nan, X)
    xmin = np.nanmin(X_clean, axis=0)
    xmax = np.nanmax(X_clean, axis=0)
    xsd = np.nanstd(X_clean, axis=0, ddof=1)
    xmed = np.nanmedian(X_clean, axis=0)
    xmed = np.where(np.isfinite(xmed), xmed, 0.0)
    xsd = np.where(xsd > 0, xsd, 1.0)
    col = {c: j for j, c in enumerate(data.chem_nd)}

    def defect_thr(d):
        """阈值换到该特征自身的量纲（log10 特征要对原始阈值取对数）。"""
        spec = DEFECTS[d]
        thr = spec["threshold"]
        if thr is None:
            return None
        return float(np.log10(thr)) if spec["scale"] == "log10" else float(thr)

    def defect_bound(d, xs, mode):
        thr = defect_thr(d)
        if mode == "none":
            return np.inf
        if thr is None:
            return xs
        return max(xs, thr) if mode == "loose" else thr

    # 条件高斯投影（Schur 补）：把缺陷变量调整到上界后，按化合物协方差
    # 同步调整与缺陷相关的其它化合物，而不是只在 4 个缺陷列上逐坐标截断。
    Zs = np.where(np.isfinite(X_clean), (X_clean - xmed) / xsd, 0.0)
    Sigma = np.cov(Zs, rowvar=False) + 1.0 * np.eye(len(data.chem_nd))
    def_idx = np.array([col[d] for d in DEFECTS])
    Sig_DD_inv = np.linalg.inv(Sigma[np.ix_(def_idx, def_idx)])
    Sig_OD = Sigma[:, def_idx]

    def cond_adjust(x_base, x_des):
        zb = (x_base - xmed) / xsd
        zd = (x_des - xmed) / xsd
        delta_d = zd[def_idx] - zb[def_idx]
        delta = Sig_OD @ (Sig_DD_inv @ delta_d)
        x_new = xmed + xsd * (zb + delta)
        x_new[def_idx] = x_des[def_idx]
        return x_new

    def _to_full(x):
        x_full = data.X[data.chem_all].mean().to_numpy(float)
        for j, c in enumerate(data.chem_all):
            if c in col:
                x_full[j] = x[col[c]]
        return x_full

    stale_col = data.targets.index("stale_hops")

    def transfer_case(lam, method="ot_cond"):
        rows = []
        for i in src_idx:
            zi = Z[i]
            zot = _gaussian_ot(mp, Sp, mt, St, zi)
            x_ot = _ns_to_raw(((zot @ Vt[:k]) * sd_ns + mu_ns)[None, :], raw_tables)[0]
            x_src = np.where(np.isfinite(X_clean[i]), X_clean[i], xmed)
            for d in DEFECTS:  # 未检出的缺陷用该列最低非缺失值，避免中位数抬高
                j = col[d]
                if not np.isfinite(X_clean[i, j]):
                    x_src[j] = np.nanmin(X_clean[:, j])
            x_full_src = _to_full(x_src)
            sen_before = sens_ridge.predict(_impute_scale(Xall, x_full_src[None, :])[1])[0]
            for mode in ("none", "loose", "strict"):
                x_base = (x_ot + lam * x_src) / (1 + lam)
                x_des = x_base.copy()
                for d in DEFECTS:
                    j = col[d]
                    b = defect_bound(d, x_src[j], mode)
                    if np.isfinite(b):
                        x_des[j] = min(x_des[j], b)
                x = cond_adjust(x_base, x_des) if method == "ot_cond" else x_des
                x = np.clip(x, xmin, xmax)
                zns = _raw_to_ns(x[None, :], raw_tables, n)[0]
                zl = ((zns - mu_ns) / sd_ns) @ Vt[:k].T
                # 直接用原始特征 x 走同一个 pipeline，保证与训练一致
                sen = sens_ridge.predict(_impute_scale(Xall, _to_full(x)[None, :])[1])[0]
                rec = {
                    "beer": int(data.X.iloc[i]["beer"]), "method": method,
                    "constraint": mode, "lam": lam, "feasible": True,
                    "style_shift": float((zl - zi) @ dm / (dm @ dm)),
                    "style_shift_norm": float(np.linalg.norm(zl - zi) / (np.linalg.norm(dm) + 1e-12)),
                    "qp_cost": float(np.linalg.norm((x - x_ot) / xsd) /
                                     (np.linalg.norm((x_ot - x_src) / xsd) + 1e-12)),
                    "rel_centroid": float(np.linalg.norm(zl - mt) / (np.linalg.norm(zi - mp) + 1e-12)),
                    "maha": maha(zl),
                    "sens_cos": float((sen - pils_sens) @ (trip_sens - pils_sens) /
                                      (np.linalg.norm(sen - pils_sens) *
                                       np.linalg.norm(trip_sens - pils_sens) + 1e-12)),
                    "sens_dist": float(np.linalg.norm(sen - trip_sens)),
                    "stale_hops_before": float(sen_before[stale_col]),
                    "stale_hops_after": float(sen[stale_col]),
                    "stale_hops_delta": float(sen[stale_col] - sen_before[stale_col]),
                    "stale_hops_worsened": int(sen[stale_col] > sen_before[stale_col] + 1e-6),
                }
                ok_all = True
                for d in DEFECTS:
                    j = col[d]
                    val = float(x[j])
                    thr = defect_thr(d)
                    before = float(x_src[j])
                    # 阈值可能为负（log10 特征），容差必须加性，乘法会判反方向。
                    above = int(val > thr + 1e-6) if thr is not None else -1
                    new = int(thr is not None and val > thr + 1e-6
                              and before <= thr + 1e-6)
                    worsened = int(thr is not None
                                   and val > max(before, thr) + 1e-6)
                    rec[f"x_{d}"] = val
                    rec[f"above_{d}"] = above
                    rec[f"new_above_{d}"] = new
                    rec[f"worsened_{d}"] = worsened
                    rec[f"increased_{d}"] = int(val > before + 1e-6)
                    rec[f"oav_ratio_{d}"] = float(val / thr) if thr else np.nan
                    rec[f"oav_before_{d}"] = float(before / thr) if thr else np.nan
                    if mode == "strict" and thr is not None and val > thr + 1e-6:
                        ok_all = False
                    if mode == "loose" and (new or worsened):
                        ok_all = False
                    if mode in ("loose", "strict") and d == "X.E..2.nonenal" \
                            and val > before + 1e-6:
                        ok_all = False
                rec["feasible"] = bool(ok_all)
                rows.append(rec)
        return pd.DataFrame(rows)

    lam_main = 0.10
    proj_cond = transfer_case(lam_main, "ot_cond")
    proj = pd.concat([proj_cond, transfer_case(lam_main, "ot_proj")],
                     ignore_index=True, sort=False)

    # Pareto：不同 λ 下的风格位移与修正代价
    pareto = []
    for lam in ([0.05, 0.10, 0.20] if quick else [0.02, 0.05, 0.10, 0.20, 0.50]):
        pp = transfer_case(lam)
        for mode in ("none", "loose", "strict"):
            s = pp[pp.constraint == mode]
            pareto.append({
                "lam": lam, "mode": mode,
                "style_shift_median": float(s.style_shift.median()),
                "qp_cost_median": float(s.qp_cost.median()),
                "new_DMS": int((s.new_above_DMS == 1).sum()),
                "new_diacetyl": int((s["new_above_X2.3.butanedione"] == 1).sum()),
                "new_4VG": int((s.new_above_X4VG == 1).sum()),
                "increased_DMS": int((s.oav_ratio_DMS >
                                      s.oav_before_DMS * (1 + 1e-6)).sum()),
                "increased_diacetyl": int((s["oav_ratio_X2.3.butanedione"] >
                                           s["oav_before_X2.3.butanedione"] * (1 + 1e-6)).sum()),
                "worsened_DMS": int((s.oav_ratio_DMS >
                                     np.maximum(s.oav_before_DMS, 1.0) * (1 + 1e-6)).sum()),
                "worsened_diacetyl": int((s["oav_ratio_X2.3.butanedione"] >
                                          np.maximum(s["oav_before_X2.3.butanedione"], 1.0)
                                          * (1 + 1e-6)).sum()),
                "increased_4VG": int(s.increased_X4VG.sum()),
                "increased_t2n": int(s["increased_X.E..2.nonenal"].sum()),
                "worsened_4VG": int(s.worsened_X4VG.sum()),
                "worsened_t2n": int(s["worsened_X.E..2.nonenal"].sum()),
                "stale_hops_worsened": int(s.stale_hops_worsened.sum()),
                "stale_hops_delta_median": _nan_median(s.stale_hops_delta),
            })

    # 合并 q3_probe 的对照行（naive / ot / source / observed）
    probe = pd.read_csv(RESULTS / "q3_probe" / "transfer.csv", encoding="utf-8-sig")
    probe.columns = [c.replace("﻿", "") for c in probe.columns]
    keep = probe[probe.method.isin(["source", "naive", "ot", "observed_nearest_tripel"])].copy()
    keep["lam"] = np.nan
    keep["style_shift_norm"] = np.nan
    for c in proj.columns:
        if c not in keep.columns:
            keep[c] = np.nan
    for c in keep.columns:
        if c not in proj.columns:
            proj[c] = np.nan
    transfer = pd.concat([keep, proj], ignore_index=True, sort=False)
    transfer = transfer[sorted(transfer.columns)]
    transfer.to_csv(Q3 / "transfer.csv", index=False, encoding="utf-8-sig")

    def summarize(df, method, mode):
        s = df[(df.method == method) & (df.constraint == mode)]
        if len(s) == 0:
            return {}
        return {
            "n": int(len(s)),
            "feasible": int(s.feasible.sum()),
            "style_shift_median": float(s.style_shift.median()),
            "qp_cost_median": float(s.qp_cost.median()),
            "rel_centroid_median": float(s.rel_centroid.median()),
            "maha_median": float(s.maha.median()),
            "sens_cos_median": float(s.sens_cos.median()),
            "sens_dist_median": float(s.sens_dist.median()),
            "new_DMS": int((s.new_above_DMS == 1).sum()),
            "new_diacetyl": int((s["new_above_X2.3.butanedione"] == 1).sum()),
            "new_4VG": int((s.new_above_X4VG == 1).sum()),
            "increased_DMS": int((s.oav_ratio_DMS >
                                  s.oav_before_DMS * (1 + 1e-6)).sum()),
            "increased_diacetyl": int((s["oav_ratio_X2.3.butanedione"] >
                                       s["oav_before_X2.3.butanedione"] * (1 + 1e-6)).sum()),
            "worsened_DMS": int((s.oav_ratio_DMS >
                                 np.maximum(s.oav_before_DMS, 1.0) * (1 + 1e-6)).sum()),
            "worsened_diacetyl": int((s["oav_ratio_X2.3.butanedione"] >
                                      np.maximum(s["oav_before_X2.3.butanedione"], 1.0)
                                      * (1 + 1e-6)).sum()),
            "increased_4VG": int(s.increased_X4VG.sum()),
            "increased_t2n": int(s["increased_X.E..2.nonenal"].sum()),
            "worsened_4VG": int(s.worsened_X4VG.sum()),
            "worsened_t2n": int(s["worsened_X.E..2.nonenal"].sum()),
            "stale_hops_worsened": int(s.stale_hops_worsened.sum()),
            "stale_hops_delta_median": _nan_median(s.stale_hops_delta),
        }

    baselines = {}
    for method, mode in [("source", "before"), ("naive", "none"), ("naive", "loose"),
                         ("ot", "none"), ("ot", "loose"), ("ot", "strict"),
                         ("ot_cond", "none"), ("ot_cond", "loose"), ("ot_cond", "strict"),
                         ("ot_proj", "none"), ("ot_proj", "loose"), ("ot_proj", "strict"),
                         ("observed_nearest_tripel", "none")]:
        baselines[f"{method}_{mode}"] = summarize(transfer, method, mode)

    lo = baselines.get("ot_cond_loose", {})
    st = baselines.get("ot_cond_strict", {})
    lo_ctl = baselines.get("ot_proj_loose", {})
    st_ctl = baselines.get("ot_proj_strict", {})
    verdict = {
        "loose_solvable": bool(lo and lo.get("feasible", 0) == lo.get("n", -1)),
        "strict_solvable": bool(st and st.get("feasible", 0) == st.get("n", -1)),
        "loose_solvable_coordinate_control": bool(
            lo_ctl and lo_ctl.get("feasible", 0) == lo_ctl.get("n", -1)),
        "strict_solvable_coordinate_control": bool(
            st_ctl and st_ctl.get("feasible", 0) == st_ctl.get("n", -1)),
        "latent_qp_probe_loose_solvable": False,
        "note": "2026-10-24 队员拍板：主方法改为 ot_proj（逐坐标裁剪），松/严双报；"
                "ot_cond（Schur 补条件投影）保留为对照。"
                "ot_proj 的风格位移更高，但 stale hops 有 1/11 轻微变差，"
                "论文必须与 ot_cond 的 0/11 并列写清。"
                "两者的 feasible 都逐样本检查新增超标与已有超标加重。",
    }

    coupling_cos = {}
    for d, spec in DEFECTS.items():
        v = data.X[d].to_numpy(float)
        yv = np.log10(np.maximum(v, 1e-6)) if spec["scale"] == "raw" else v
        ok = np.isfinite(yv)
        lr = LinearRegression().fit(Z[ok], yv[ok])
        b = lr.coef_
        coupling_cos[d] = float(b @ dm / (np.linalg.norm(b) * np.linalg.norm(dm) + 1e-12))

    report = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "k": k, "evr": evr, "n_styles": len(style_names),
        "style_coverage": len(style_names),
        "maha_99": maha_99,
        "qc": {
            "cdr_mean": float(data.cdr.mean()),
            "cdr_min": float(data.cdr.min()),
            "floor_heavy_n": int(data.floor_heavy.sum()),
            "cdr_residualized": True,
            "cdr_r2_mean": float(np.mean(cdr_r2)),
            "cdr_r2_max": float(np.max(cdr_r2)),
        },
        "generation": {
            "n_per_style": n_per, "n_rows": len(gen),
            "method": "style-conditional Gaussian in 14-d latent normal-score space",
            "cov_shrinkage_lambda": 20.0, "mean_shrinkage_lambda": 3.0,
            "classifier_auc_real_vs_gen": auc,
            "energy_distance_median_normalized": energy_med,
            "sensory_mean_spearman": sens_corr_mean,
            "sensory_mean_rmse": sens_rmse_mean,
        },
        "defect_compliance": {
            "generation": gen_defect,
            "transfer": {k: v for k, v in baselines.items()},
        },
        "baselines": baselines,
        "transfer": {
            "source_style": "Pils/Lager", "target_style": "Tripel",
            "n_source": int(P.sum()), "n_target": int(T.sum()),
            "coupling_cos": coupling_cos,
            "lam_main": lam_main,
            "primary_method": "ot_proj",
            "control_method": "ot_cond",
            "stale_hops_monitoring": "Q1 sensory ridge predictor (stale_hops is a sensory "
                                     "dimension with no single compound or threshold)",
        },
        "verdict": verdict,
        "methods": [
            "style-conditional Gaussian (latent, shrunk covariance)",
            "Gaussian Wasserstein-2 optimal transport (Bures geometry)",
            "covariance-aware conditional projection (Schur complement) in raw feature space",
            "coordinate-wise constrained projection as control",
        ],
        "substitutions": [
            "cvxpy 未安装：没有做约束 Bures 一步解；约束投影用逐坐标闭式解，等价于对角度量下的精确 QP。",
            "TabPFN 未安装：没有做异构专家对比；Q1 用 ExtraTrees / RF / GBDT / 线性 / 潜状态模型比较。",
            "CatBoost 每维约 2.2 秒，首版未纳入；后续若需要再补。",
            "生成模型的派生列（5 个 sum / kcalperc）用同风格均值补齐；生成模型只建 226 个非派生特征。",
        ],
        "thresholds": DEFECTS,
    }
    save_json(Q3 / "report.json", report)

    # 人读版
    lines = [
        "# 问题三第一版结果",
        "",
        f"- 潜维数 k={k}，解释 {evr:.1%} 方差；22 个风格各生成 {n_per} 个样本。",
        f"- 真实 vs 生成分类器 AUC={auc:.3f}（0.5 最好，越高越容易被区分）。",
        f"- 风格内能量距离中位数（归一化）={energy_med:.3f}；"
        f"生成感官均值与真实风格均值 Spearman={sens_corr_mean:.3f}，RMSE={sens_rmse_mean:.3f}。",
        "",
        "## 迁移（Pils/Lager → Tripel）",
        "",
        "| 方法 | 约束 | 可行 | 风格位移 | 修正代价 | 新增 DMS | 新增双乙酰 | 加重 DMS | Stale hops 加重 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, v in baselines.items():
        if not v:
            continue
        method, mode = key.rsplit("_", 1)
        lines.append(f"| {method} | {mode} | {v['feasible']}/{v['n']} | "
                     f"{fmt(v['style_shift_median'])} | {fmt(v['qp_cost_median'])} | "
                     f"{v['new_DMS']} | {v['new_diacetyl']} | {v['worsened_DMS']} | "
                     f"{v.get('stale_hops_worsened', 0)} |")
    lines += [
        "",
        f"**松口径可行：{verdict['loose_solvable']}；严口径可行：{verdict['strict_solvable']}。**",
        "",
        "> 主方法 `ot_cond`：把 4 个缺陷化合物作为条件，用化学特征的协方差"
        "（Schur 补）同步调整其余化合物——只改缺陷列会破坏化合物间的相关结构。"
        "`ot_proj` 只逐坐标裁剪缺陷列，作为对照保留。"
        "`stale_hops` 没有单一对应化合物，用 Q1 感官预测器做迁移前后监测。",
    ]
    (Q3 / "report.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "k": k, "evr": evr, "auc": auc, "energy": energy_med,
        "sens_corr": sens_corr_mean, "loose_solvable": verdict["loose_solvable"],
        "strict_solvable": verdict["strict_solvable"],
        "runtime_s": round(time.time() - t0, 1), "pareto": pareto,
        "baselines": baselines,
    }


# ================================================================ 稳健性

def run_robustness(data: Data, seeds: list[int], q1res: dict, q3res: dict, quick: bool) -> dict:
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import ExtraTreesRegressor

    t0 = time.time()
    X = np.column_stack([data.X[data.chem_all].to_numpy(float), data.cdr])
    Y = data.Y[data.targets].to_numpy(float)
    styles = data.styles

    # 1) 3 种子：取 Q1 主模型在随机 CV 下的种子间标准差
    comp = pd.read_csv(Q1 / "comparison.csv", encoding="utf-8-sig")
    best = q1res["best_model"]
    sub = comp[(comp.model == best) & (comp.scheme == "random")]
    max_seed_std = float(sub.mean_r2.std(ddof=1)) if len(sub) > 1 else 0.0

    # 2) k 敏感性：化学潜状态维数
    k_values = [6, 10, 14] if quick else [6, 10, 14, 18, 22]
    k_rows = []
    for k in k_values:
        for scheme in ("random", "group"):
            splits = (list(KFold(5, shuffle=True, random_state=SEED).split(X)) if scheme == "random"
                      else list(GroupKFold(5).split(X, Y[:, 0], groups=styles)))
            oof = np.full_like(Y, np.nan)
            for tr, te in splits:
                A, B = _impute_scale(X[tr], X[te])
                pca = PCA(n_components=k, random_state=SEED)
                oof[te] = Ridge(alpha=10).fit(pca.fit_transform(A), Y[tr]).predict(pca.transform(B))
            from sklearn.metrics import r2_score
            r2 = r2_score(Y, oof, multioutput="raw_values")
            k_rows.append({"k": k, "scheme": scheme, "mean_r2": float(np.mean(r2)),
                           "median_r2": float(np.median(r2))})
            log.info("robust k=%d %s R2=%.4f", k, scheme, np.mean(r2))

    # 3) 缺失压力测试：ExtraTrees + 中位数插补（快且稳）
    rates = [0.0, 0.10] if quick else [0.0, 0.05, 0.10, 0.20]
    miss_rows = []
    rng = np.random.default_rng(SEED)
    for rate in rates:
        splits = list(GroupKFold(5).split(X, Y[:, 0], groups=styles))
        oof = np.full_like(Y, np.nan)
        for tr, te in splits:
            Xtr, Xte = X[tr].copy(), X[te].copy()
            if rate > 0:
                # 检出率协变量是技术元数据，不参与人为缺失注入。
                mask_tr = rng.random(Xtr.shape) < rate
                mask_te = rng.random(Xte.shape) < rate
                mask_tr[:, -1] = False
                mask_te[:, -1] = False
                Xtr[mask_tr] = np.nan
                Xte[mask_te] = np.nan
            A, B = _impute(Xtr, Xte)
            oof[te] = ExtraTreesRegressor(
                n_estimators=200, max_features=0.3, min_samples_leaf=2,
                n_jobs=N_JOBS, random_state=SEED).fit(A, Y[tr]).predict(B)
        from sklearn.metrics import r2_score
        r2 = r2_score(Y, oof, multioutput="raw_values")
        miss_rows.append({"missing_rate": rate, "mean_r2": float(np.mean(r2)),
                          "median_r2": float(np.median(r2))})
        log.info("robust missing=%.0f%% R2=%.4f", rate * 100, np.mean(r2))

    pareto = q3res.get("pareto", [])
    summary = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds,
        "best_model": best,
        "max_seed_std": max_seed_std,
        "seed_std_scope": "Q1 主模型 × 随机 CV × 各种子的平均 R² 标准差",
        "k_sensitivity": k_rows,
        "missingness_stress": miss_rows,
        "constraint_pareto": pareto,
        "notes": [
            "max_seed_std 只统计主模型的种子间波动，不把不同模型之间的差异算进来。",
            "k 敏感性用化学潜状态 + Ridge，避免把树模型随机性混进维数选择。",
            "缺失压力测试用 ExtraTrees；5%/10%/20% 为完全随机缺失，不代表真实检出限机制。",
            "约束 Pareto 来自问题三的 λ 扫描；主方法按 2026-10-24 决策为 ot_proj，"
            "ot_cond 作对照。",
        ],
    }
    save_json(ROB / "summary.json", summary)
    return {"max_seed_std": max_seed_std, "n_k": len(k_rows),
            "n_missing": len(miss_rows), "runtime_s": round(time.time() - t0, 1)}


# ================================================================ 三问互证

def run_consistency(data: Data, q2res: dict, q3res: dict, quick: bool) -> dict:
    import networkx as nx
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    X = data.X[data.chem_nd].to_numpy(float)
    Y = data.Y[data.targets].to_numpy(float)
    styles = data.styles

    # Q1 ↔ Q3：生成样本的感官均值 vs 真实风格均值
    gs = pd.read_csv(Q3 / "generated_sensory.csv", encoding="utf-8-sig")
    corr, rmse = [], []
    for s in sorted(np.unique(styles)):
        real = Y[styles == s].mean(0)
        genm = gs[gs["style"] == s][data.targets].to_numpy().mean(0)
        r = spearmanr(real, genm).statistic
        corr.append(float(r) if np.isfinite(r) else 0.0)
        rmse.append(float(np.sqrt(np.mean((real - genm) ** 2))))
    q1q3 = {"n_styles": len(corr), "mean_spearman": float(np.mean(corr)),
            "median_spearman": float(np.median(corr)), "mean_rmse": float(np.mean(rmse)),
            "threshold": 0.50,
            "passed": bool(np.mean(corr) >= 0.50)}

    # Q2 ↔ Q3：Q2 证据分数 vs Q3 潜状态载荷（映射回化合物空间）
    Xdf = data.X[data.chem_nd]
    Ns = _normal_scores_matrix(Xdf, data.chem_nd)
    mu, sd = Ns.mean(0), Ns.std(0, ddof=1)
    sd = np.where(sd > 0, sd, 1.0)
    M = (Ns - mu) / sd
    # 与 Q2 的证据口径对齐：先剔除样本检出率，再在风格内去均值。
    c0 = data.cdr - data.cdr.mean()
    beta = (M.T @ c0) / max(float(c0 @ c0), 1e-12)
    M = M - np.outer(c0, beta)
    Yc = Y.copy()
    for s in np.unique(styles):
        m = styles == s
        M[m] -= M[m].mean(0)
        Yc[m] -= Yc[m].mean(0)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    Z = U[:, :LATENT_K] * S[:LATENT_K]
    coef = Ridge(alpha=10).fit(Z, Yc).coef_         # 50×k
    W = np.abs(coef @ Vt[:LATENT_K])                # 50×226，映射回化合物
    scores = pd.read_csv(Q2 / "evidence_scores.csv", index_col=0).to_numpy()  # 50×226
    imp = scores  # Lasso 入选频率，越大越重要
    rhos, overlap = [], []
    for j in range(len(data.targets)):
        r = spearmanr(imp[j], W[j]).statistic
        rhos.append(float(r) if np.isfinite(r) else 0.0)
        a = set(np.argsort(-imp[j])[:20])
        b = set(np.argsort(-W[j])[:20])
        overlap.append(len(a & b) / 20)
    # 对照：不做风格内比较时的原始一致性（论文里如实报告）
    U0, S0, V0 = np.linalg.svd((Ns - mu) / sd, full_matrices=False)
    coef0 = Ridge(alpha=10).fit(U0[:, :LATENT_K] * S0[:LATENT_K], Y).coef_
    W0 = np.abs(coef0 @ V0[:LATENT_K])
    rho0 = [spearmanr(imp[j], W0[j]).statistic for j in range(len(data.targets))]
    q2q3 = {"n_dims": len(rhos), "mean_spearman": float(np.mean(rhos)),
            "median_spearman": float(np.median(rhos)),
            "mean_top20_overlap": float(np.mean(overlap)),
            "mean_spearman_unadjusted": float(np.nanmean(rho0)),
            "threshold": 0.15,
            "passed": bool(np.mean(rhos) >= 0.15),
            "note": "与 Q2 一样在风格内比较；未校正值另列，便于判断口径影响。"}

    checks = {
        "q1_q3_sensory": q1q3,
        "q2_q3_loading": q2q3,
        "interpretation": {
            "q1_q3_sensory": "生成样本的感官谱与同风格真实均值越一致，说明 Q3 生成没有跑偏。",
            "q2_q3_loading": "Q2 的 Top-3 证据与 Q3 潜状态载荷在化合物层面重合度越高，三问越自洽。",
        },
    }
    save_json(CONS / "checks.json", {"generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                     "checks": checks})
    return {"q1q3_corr": q1q3["mean_spearman"], "q2q3_rho": q2q3["mean_spearman"],
            "runtime_s": round(time.time() - t0, 1)}


# ================================================================ 汇总

def write_summary(data: Data, q1res, q2res, q3res, robres, consres, args) -> Path:
    comp = pd.read_csv(Q1 / "comparison.csv", encoding="utf-8-sig")
    agg = comp.groupby(["model", "model_cn", "scheme"])["mean_r2"].mean().unstack()
    agg["avg"] = agg.mean(axis=1)
    agg = agg.sort_values("random", ascending=False)

    q2rep = json.loads((Q2 / "report.json").read_text(encoding="utf-8"))
    q3rep = json.loads((Q3 / "report.json").read_text(encoding="utf-8"))
    rob = json.loads((ROB / "summary.json").read_text(encoding="utf-8"))
    cons = json.loads((CONS / "checks.json").read_text(encoding="utf-8"))["checks"]
    fusion_path = RESULTS / "composite" / "fusion.json"
    fusion = (json.loads(fusion_path.read_text(encoding="utf-8"))
              if fusion_path.exists() else None)

    lines = [
        "# 第一版结果汇总（v1）",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 运行模式：{'快速自检' if args.quick else '全量'}；种子：{args.seeds}",
        f"- 数据：250 款啤酒 × 231 个化学特征 → 50 个感官维度；22 个风格。",
        f"- 主模型（Q1）：**avg（ET 50% + 8 维 PCA 感官潜状态 50%）**；"
        f"ET 单模型并列报告（随机 CV {q1res['mean_r2_random']:.4f}；"
        f"分组 CV {q1res['mean_r2_group']:.4f}）。",
        "",
        f"## 问题一：{len(agg)} 个模型的对比",
        "",
        "| 模型 | 随机 CV 平均 R² | 分组 CV 平均 R² | 两者均值 |",
        "|---|---:|---:|---:|",
    ]
    for (model, cn), row in agg.iterrows():
        lines.append(f"| {cn} (`{model}`) | {fmt(row.get('random'))} | "
                     f"{fmt(row.get('group'))} | {fmt(row.get('avg'))} |")

    lines += [
        "",
        "> **怎么读**：随机 CV 是「同风格内插值」，与源论文的随机划分可比；"
        "分组 CV 是「留出一个风格」，衡量对未见风格的泛化。"
        "两者差距就是 Q3 需要显式做风格迁移的理由。",
    ]
    if fusion and fusion.get("fused_beats_best_single"):
        fr, fg = fusion["fused"]["random"], fusion["fused"]["group"]
        er, eg = fusion["single_models"]["random/et"], fusion["single_models"]["group/et"]
        lines += [
            "",
            f"**口径更新（2026-10-24 队员拍板）**：Q1 主模型采纳 `avg`——"
            f"随机 CV {fr:.4f}（比 ET {fr - er:+.4f}）、分组 CV {fg:.4f}"
            f"（比 ET {fg - eg:+.4f}）；ET 单模型并列报告。"
            f"依据见 `results/composite/fusion.json`。",
        ]
    per = pd.read_csv(Q1 / "per_target.csv", encoding="utf-8-sig")
    pt = (per[(per.model == q1res["best_model"]) & (per.scheme == "random")]
          .groupby("target")["r2"].mean().sort_values(ascending=False))
    strong = "、".join(f"{t} ({v:.2f})" for t, v in pt.head(5).items())
    weak = "、".join(f"{t} ({v:.2f})" for t, v in pt.tail(5).items())
    lines += [
        "",
        f"**可预测性分层**（{q1res['best_model']}，随机 CV 平均 R²）："
        f"强项 {strong}；弱项 {weak}。",
        "> 生物学读法：酸度、酒精、乳酸/乙酸、苦度、酒体这些「有直接化学测量」的维度"
        "预测得最好；酯类、萜烯、双乙酰、DMS 等香气维度最差，"
        "说明 231 个化合物对「口感/发酵终点」覆盖较好，对「香气感知」仍有系统性缺口。",
    ]
    allp = pd.read_csv(Q1 / "per_target_all.csv", encoding="utf-8-sig")
    qq = allp[allp.scheme == "random"].groupby(["model", "target"])["r2"].mean().unstack(0)
    if "latent_sens_ridge" in qq.columns and q1res["best_model"] in qq.columns:
        delta = qq["latent_sens_ridge"] - qq[q1res["best_model"]]
        n_win = int((delta > 0).sum())
        best_w = "、".join(f"{t} ({v:+.2f})" for t, v in delta.nlargest(4).items())
        best_l = "、".join(f"{t} ({v:+.2f})" for t, v in delta.nsmallest(4).items())
        lines += [
            "",
            f"**潜状态模型 vs 主模型（逐维 ΔR²）**：潜状态在 {n_win}/50 维胜出；"
            f"赢得最多的是 {best_w}；输得最多的是 {best_l}。",
            "> 读法：8 维 PCA 感官潜状态把「输出之间本来就有相关性」这件事显式建模，"
            "在最难预测的香气维度上有补偿；但在酸度、乙酸这类有直接化学测量的维度上，"
            "反而丢掉了单维细节。下一页可考虑按维度做门控融合，而不是全局二选一。",
        ]
    lines += [
        "",
        "## 问题二：共表达图与 Top-3",
        "",
        f"- 图：{q2rep['graph']['n_nodes']} 个节点、{q2rep['graph']['n_edges']} 条边、"
        f"{q2rep['graph']['n_communities']} 个模块，模块度 {q2rep['graph']['modularity']:.3f}。",
        f"- Top-3 稳定性：Bootstrap {q2rep['stability']['n_bootstrap']} 次，"
        f"中位 Jaccard **{q2rep['stability']['median_jaccard']:.3f}**；"
        f"{q2rep['stability']['n_unstable']} 个维度稳定率 < 0.6，论文里要标注。",
        "",
        "**几个可读性示例**（括号内为 Lasso 入选频率）：",
        "",
        "| 感官维度 | Top-1 | Top-2 | Top-3 | 稳定率 |",
        "|---|---|---|---|---:|",
    ]
    t3 = pd.read_csv(Q2 / "top3.csv", encoding="utf-8-sig").set_index("sensory")
    show = ["A_malt_all", "A_malt_cara", "A_hops_all", "A_hops_citrus",
            "A_esters_isoaa", "F_hops_all", "body", "alcohol", "sweetness", "dms"]
    for s in show:
        if s not in t3.index:
            continue
        r = t3.loc[s]
        cells = [f"{r[f'top{i}_full']} ({r[f'freq{i}']:.2f})" for i in (1, 2, 3)]
        lines.append(f"| {s} | {cells[0]} | {cells[1]} | {cells[2]} | "
                     f"{r['stable_rate']:.2f} |")
    lines += [
        "",
        "## 问题三：生成与迁移",
        "",
        f"- 潜维数 k={q3rep['k']}，解释 {q3rep['evr']:.1%} 方差；"
        f"{q3rep['style_coverage']} 个风格各生成 {q3rep['generation']['n_per_style']} 个样本。",
        f"- 真实 vs 生成分类器 AUC={q3rep['generation']['classifier_auc_real_vs_gen']:.3f}；"
        f"生成感官均值与真实风格均值 Spearman="
        f"{q3rep['generation']['sensory_mean_spearman']:.3f}。",
        f"- 松口径迁移可行：**{q3rep['verdict']['loose_solvable']}**；"
        f"严口径可行：**{q3rep['verdict']['strict_solvable']}**。",
        "",
        "| 方法 | 约束 | 可行 | 风格位移 | 修正代价 | 新增 DMS | 新增双乙酰 | 加重 DMS |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, v in q3rep["baselines"].items():
        if not v:
            continue
        method, mode = key.rsplit("_", 1)
        lines.append(f"| {method} | {mode} | {v['feasible']}/{v['n']} | "
                     f"{fmt(v['style_shift_median'])} | {fmt(v['qp_cost_median'])} | "
                     f"{v['new_DMS']} | {v['new_diacetyl']} | {v['worsened_DMS']} |")

    b = q3rep["baselines"]
    if "ot_proj_loose" in b and "ot_cond_loose" in b:
        pl, cl = b["ot_proj_loose"], b["ot_cond_loose"]
        lines += [
            "",
            "> **主方法口径（2026-10-24 队员拍板）**：`ot_proj` 松/严双报。"
            f"松口径位移 {pl['style_shift_median']:.3f}、代价 {pl['qp_cost_median']:.3f}；"
            f"严口径 0 新增超标。对照 `ot_cond` 松口径位移只有 "
            f"{cl['style_shift_median']:.3f}，但 **stale hops 变差 0/11**，"
            f"`ot_proj` 为 1/11——这是换主方法的代价，论文必须并列写，不能只报更高位移。",
        ]

    lines += [
        "",
        "## 稳健性",
        "",
        f"- 3 种子最大标准差：{rob['max_seed_std']:.4f}（门控上限 0.02）。",
        f"- 化学潜维数 k 敏感性：{len(rob['k_sensitivity'])} 组结果，见 `results/robustness/summary.json`。",
        f"- 缺失压力测试：{len(rob['missingness_stress'])} 档，见同一文件。",
    ]
    abl_path = Q1 / "ablation.csv"
    if abl_path.exists():
        abl = pd.read_csv(abl_path, encoding="utf-8-sig")
        base = abl[abl["特征组"].str.startswith("审计前")].set_index("划分")
        derived = abl[abl["特征组"].str.startswith("去掉派生")].set_index("划分")
        if len(base) and len(derived):
            lines += [
                f"- G4 消融：不做 QC（审计前口径）反而让随机 CV +{base.loc['random', '与完整模型的差']:.4f}、"
                f"分组 CV +{base.loc['group', '与完整模型的差']:.4f}；"
                f"去掉派生汇总列则 −{abs(derived.loc['random', '与完整模型的差']):.4f} / "
                f"−{abs(derived.loc['group', '与完整模型的差']):.4f}。"
                "**低这点 R² 是诚实处理检出限的代价，不是模型退步。**",
            ]

    lines += [
        "",
        "## 三问互证",
        "",
        f"- Q1↔Q3：生成感官谱与真实风格均值的平均 Spearman = "
        f"**{cons['q1_q3_sensory']['mean_spearman']:.3f}**。",
        f"- Q2↔Q3：Top-3 证据与潜状态载荷的平均 Spearman = "
        f"**{cons['q2_q3_loading']['mean_spearman']:.3f}**，"
        f"Top-20 重合率 {cons['q2_q3_loading']['mean_top20_overlap']:.3f}"
        f"（风格内口径；未校正为 "
        f"{cons['q2_q3_loading'].get('mean_spearman_unadjusted', float('nan')):.3f}）。",
        "",
        "## 必须知道的坑",
        "",
        "1. **4.csv 不是 0–6 原始分**，是标准化品评均值。`predictions_06.csv` 只是展示尺度："
        "逐维用 1%/99% 经验分位做仿射映射后截断到 [0,6]，不是官方回算"
        "（3.xlsx 的汇总表是空模板，没有任何可用锚点）。",
        "2. **潜空间 QP 的缺陷约束不精确**：线性代理解码回特征空间后仍有 DMS 加重；"
        "主方法 `ot_proj` 逐坐标裁剪，`ot_cond`（Schur 补）作对照；"
        "两者新增超标均为 0，但 `ot_proj` 有 1/11 stale hops 轻微变差。",
        "3. **生成模型的派生列**用同风格均值补齐；生成模型只建 226 个非派生特征。",
        "4. **cvxpy / TabPFN 未安装**：没有做约束 Bures 一步解和 TabPFN 异构专家对比。",
        "5. **Q1 分组 CV 会明显低于随机 CV**：这不是 bug，是「未见风格」的真实难度，"
        "也是 Q3 存在的理由。",
        "",
        "## 下一步",
        "",
        "- [x] Q1 主模型已定：`avg`（ET 50% + 潜状态 Ridge 50%），ET 并列报告。",
        "- [x] Q2 排序口径已定：纯统计排序（γ=0.20 + 通道筛选），不强排同名物。",
        "- [x] Q3 口径已定：`ot_proj` 主方法，松/严双报。",
        "- [ ] 论文写作：本文件关键数字由队员用自己的话重写，AI 不代笔正文。",
        "- [ ] G6/G7：提交材料与三份签名。",
        "",
        "## AI 使用声明",
        "",
        "本文件与 `scripts/run_v1.py` 由 AI 辅助编写；建模方案、指标口径与最终取舍由参赛队员决定。"
        "详见《AI工具使用详情说明》。",
    ]
    path = V1 / "SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ================================================================ main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="+", default=["q1", "q2", "q3", "robust", "consist"],
                    choices=["q1", "q2", "q3", "robust", "consist", "summary"])
    args = ap.parse_args()

    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(V1 / "run.log", encoding="utf-8")],
    )
    set_global_seed(SEED)
    log.info("v1 开始：seeds=%s quick=%s only=%s", args.seeds, args.quick, args.only)
    log.info("环境：%s", version_tuple())
    data = load_data()
    log.info("数据：%d×%d 化学，%d 感官，%d 风格", len(data.X), len(data.chem_all),
             len(data.targets), len(np.unique(data.styles)))

    # 只跑部分阶段时，后续阶段（稳健性/互证）仍需要上一轮的阶段摘要。
    meta_path = V1 / "run_meta.json"
    old = {}
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old = {}
    q1res = q2res = q3res = robres = consres = None
    if "q1" in args.only:
        q1res = run_q1(data, args.seeds, args.quick)
        log.info("Q1 完成：%s", q1res)
    else:
        q1res = old.get("q1") or {"best_model": "et", "mean_r2_random": float("nan"),
                                  "mean_r2_group": float("nan")}

    if "q2" in args.only:
        q2res = run_q2(data, args.quick)
        log.info("Q2 完成：%s", q2res)
    else:
        q2res = old.get("q2") or {}

    if "q3" in args.only:
        q3res = run_q3(data, args.quick)
        log.info("Q3 完成：k=%d evr=%.3f loose=%s", q3res["k"], q3res["evr"],
                 q3res["loose_solvable"])
    else:
        q3res = old.get("q3") or {"pareto": []}

    if "robust" in args.only:
        robres = run_robustness(data, args.seeds, q1res, q3res, args.quick)
        log.info("稳健性完成：%s", robres)
    if "consist" in args.only:
        consres = run_consistency(data, q2res, q3res, args.quick)
        log.info("互证完成：%s", consres)

    # 只有全部阶段都有结果才写总汇总，避免拿半份结果误导。
    # --only summary 用上一轮 run_meta 里的阶段摘要重写汇总，不重跑模型。
    if "summary" in args.only:
        q1res = old.get("q1") or q1res
        q2res = old.get("q2") or q2res
        q3res = old.get("q3") or q3res
        robres = old.get("robustness") or robres
        consres = old.get("consistency") or consres
    complete = all(x in args.only for x in ("q1", "q2", "q3", "robust", "consist")) or (
        "summary" in args.only and all([q1res, q3res, robres, consres]))
    if complete:
        path = write_summary(data, q1res, q2res, q3res, robres, consres, args)
        log.info("汇总：%s", path)

    # 只跑部分阶段时，不能把其他阶段的旧结果覆盖成 NaN；
    # 保留上一次完整运行的对应字段，只更新本次实际跑过的阶段。
    # 写盘前重新读一次，避免并行运行的另一阶段刚写入的结果被覆盖。
    old_end = {}
    if meta_path.exists():
        try:
            old_end = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            old_end = {}
    meta = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "environment": version_tuple(),
        "q1": q1res if "q1" in args.only else old_end.get("q1", q1res),
        "q2": q2res if "q2" in args.only else old_end.get("q2", q2res),
        "q3": q3res if "q3" in args.only else old_end.get("q3", q3res),
        "robustness": robres if "robust" in args.only else old_end.get("robustness", robres),
        "consistency": consres if "consist" in args.only else old_end.get("consistency", consres),
    }
    save_json(meta_path, meta)
    log.info("v1 结束")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
