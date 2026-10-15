"""模型工厂与训练/预测封装。

三驾马车（LightGBM / XGBoost / CatBoost）接口统一，换模型只改 yaml 里的 `model` 一行。
本机没有 GPU，这三个在 CPU 上就是结构化数据的最优解，别折腾更重的东西。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import N_JOBS, get_logger

log = get_logger("models")

SUPPORTED = ("lightgbm", "xgboost", "catboost")

# 各库自己的模型文件后缀，存/取时要一致
MODEL_SUFFIX = {"lightgbm": ".txt", "xgboost": ".json", "catboost": ".cbm"}

# LightGBM 专有的参数名。别的库不认，见 validate_params。
_LGB_ONLY = {
    "num_leaves", "min_child_samples", "min_split_gain", "bagging_fraction",
    "bagging_freq", "feature_fraction", "min_sum_hessian_in_leaf",
    "lambda_l1", "lambda_l2",
}
PARAM_INCOMPATIBLE = {
    "lightgbm": set(),
    # xgboost 不报错，只是当没看见——最危险的一种
    "xgboost": _LGB_ONLY,
    # catboost 会直接 TypeError
    "catboost": _LGB_ONLY | {"colsample_bytree", "tree_method", "enable_categorical"},
}


def detect_task(y: pd.Series) -> str:
    """判断是分类还是回归。浮点且取值几乎全不同 → 回归；否则分类。"""
    if y.dtype == object or str(y.dtype) in ("category", "bool"):
        return "classification"
    if pd.api.types.is_float_dtype(y) and y.nunique() > max(20, len(y) * 0.05):
        return "regression"
    return "classification"


def validate_params(model_name: str, params: dict) -> None:
    """拦住"参数名不属于这个库"的情况。

    这是个真会咬人的坑：三个库的参数名长得像但不一样，
    **xgboost 会静默忽略它不认识的参数**（模型悄悄变差，你完全看不出来），
    catboost 则直接 TypeError。宁可当场报错，也别跑出一个悄悄变差的结果。
    """
    bad = sorted(set(params) & PARAM_INCOMPATIBLE.get(model_name.lower(), set()))
    if not bad:
        return
    hint = {
        "xgboost": "num_leaves→max_leaves（还要配 grow_policy: lossguide），min_child_samples→min_child_weight",
        "catboost": "num_leaves→depth，colsample_bytree→rsm，n_estimators 用 iterations 更稳",
    }.get(model_name.lower(), "")
    raise ValueError(
        f"配置里的参数 {bad} 不属于 {model_name}。\n"
        f"  对照写法：{hint}\n"
        f"  或者直接改用 src/configs/{model_name}.yaml 那份配置。"
    )


def build_model(model_name: str, params: dict, task: str, seed: int):
    """按名字造一个未训练的模型。"""
    model_name = model_name.lower()
    if model_name not in SUPPORTED:
        raise ValueError(f"不支持的模型 {model_name!r}，可选 {SUPPORTED}")
    validate_params(model_name, params)

    # 线程数统一从 params.n_jobs 取（配置里不写就用默认上限），
    # 再按各库的叫法传进去。三个库的线程参数名不一样。
    base = {k: v for k, v in params.items() if k != "n_jobs"}
    threads = int(params.get("n_jobs") or N_JOBS)

    # 默认值放前面、配置放后面 —— 配置里写了的以配置为准
    if model_name == "lightgbm":
        import lightgbm as lgb

        cls = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
        return cls(**{"verbose": -1, **base, "n_jobs": threads, "random_state": seed})

    if model_name == "xgboost":
        import xgboost as xgb

        cls = xgb.XGBClassifier if task == "classification" else xgb.XGBRegressor
        return cls(**{"enable_categorical": True, "tree_method": "hist",
                      **base, "n_jobs": threads, "random_state": seed})

    import catboost as cb

    cls = cb.CatBoostClassifier if task == "classification" else cb.CatBoostRegressor
    # allow_writing_files=False：不然 catboost 每次训练都在仓库根目录写一个
    # catboost_info/ 疯狂刷文件
    return cls(**{"verbose": 0, "allow_writing_files": False, **base,
                  "thread_count": threads, "random_seed": seed})


def fit_predict(
    model_name: str,
    params: dict,
    task: str,
    seed: int,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray | None]:
    """训一个模型并预测验证集。返回 (预测值, 概率或 None)。

    概率在分类任务里才有——排序、阈值、AUC 都要靠它，别只留硬标签。
    """
    model = build_model(model_name, params, task, seed)
    cat_cols = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"]

    if model_name.lower() == "catboost" and cat_cols:
        # CatBoost 要求显式声明类别列，并让它们以字符串形式出现
        X_tr, X_va = X_tr.copy(), X_va.copy()
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype(str)
            X_va[c] = X_va[c].astype(str)
        model.fit(X_tr, y_tr, cat_features=cat_cols)
    else:
        model.fit(X_tr, y_tr)

    proba = None
    if task == "classification":
        proba = model.predict_proba(X_va)
        pred = model.predict(X_va)
    else:
        pred = model.predict(X_va)

    return np.asarray(pred), proba


def train_full(
    model_name: str, params: dict, task: str, seed: int,
    X: pd.DataFrame, y: pd.Series,
):
    """用全量训练数据训最终模型（用于出测试集预测）。"""
    model = build_model(model_name, params, task, seed)
    cat_cols = [c for c in X.columns if str(X[c].dtype) == "category"]
    if model_name.lower() == "catboost" and cat_cols:
        X = X.copy()
        for c in cat_cols:
            X[c] = X[c].astype(str)
        model.fit(X, y, cat_features=cat_cols)
    else:
        model.fit(X, y)
    return model


def save_model(model, model_name: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_name = model_name.lower()
    if model_name == "lightgbm":
        model.booster_.save_model(str(path))
    else:
        model.save_model(str(path))
    log.info("模型已存 -> %s", path)


def load_model(model_name: str, task: str, path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到模型文件：{path}（先跑 src.train）")
    model_name = model_name.lower()

    if model_name == "lightgbm":
        import lightgbm as lgb

        return lgb.Booster(model_file=str(path))

    if model_name == "xgboost":
        import xgboost as xgb

        cls = xgb.XGBClassifier if task == "classification" else xgb.XGBRegressor
        m = cls()
        m.load_model(str(path))
        return m

    import catboost as cb

    cls = cb.CatBoostClassifier if task == "classification" else cb.CatBoostRegressor
    m = cls()
    m.load_model(str(path))
    return m


def predict_loaded(
    model, model_name: str, task: str, X: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray | None]:
    """用 load_model 拿到的模型做预测。返回 (预测标签/值, 概率或 None)。"""
    X = X.copy()
    if model_name.lower() == "catboost":
        for c in X.columns:
            if str(X[c].dtype) == "category":
                X[c] = X[c].astype(str)

    # LightGBM 存下来的是 Booster，它**没有 predict_proba**，
    # 分类任务下 predict 直接返回概率 —— 必须自己取 argmax，否则会拿概率当标签用。
    if model_name.lower() == "lightgbm":
        raw = np.asarray(model.predict(X))
        if task != "classification":
            return raw, None
        proba = np.column_stack([1.0 - raw, raw]) if raw.ndim == 1 else raw
        return np.argmax(proba, axis=1), proba

    if task == "classification":
        return np.asarray(model.predict(X)), np.asarray(model.predict_proba(X))
    return np.asarray(model.predict(X)), None


def feature_importance(model, model_name: str, feature_names: list[str]) -> pd.DataFrame:
    """统一取特征重要度。论文里那张"关键驱动因子"图就靠它。"""
    model_name = model_name.lower()
    if model_name == "lightgbm":
        imp = model.booster_.feature_importance(importance_type="gain") \
            if hasattr(model, "booster_") else model.feature_importance(importance_type="gain")
    elif model_name == "xgboost":
        imp = model.feature_importances_
    else:
        imp = model.get_feature_importance()

    df = pd.DataFrame({"特征": feature_names, "重要度": imp})
    df["重要度占比"] = df["重要度"] / max(df["重要度"].sum(), 1e-12)
    return df.sort_values("重要度", ascending=False).reset_index(drop=True)
