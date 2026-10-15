"""特征工程。

分两层：
- 通用层（本文件已有）：日期拆解、行内统计、缺失模式、频次编码、类别对齐。
  这些对任何赛题都能用，开赛当天就有特征可用。
- 领域层（`domain_features`）：拿到赛题后，**生信视角的领域特征写在这里**。
  这是你们相对纯 CS 队的护城河，见 docs/角色分工.md。

三条纪律（别破）：
1. **不许用目标列构造任何特征** —— 目标编码必须折内计算，此处不做，别图省事。
2. 只在训练集上决定"丢哪些列"，再套到测试集，避免测试集反过来影响特征选择。
3. 行内统计只用自己的取值，天然无泄漏，放心用。
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import PATHS, get_logger, load_config
from src.data import load_table, split_xy

log = get_logger("features")

NA_TOKEN = "__NA__"


def add_datetime_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """把日期列拆成周期特征。树的模型学不会"2019-01-31 到 2019-02-01 只差一天"，
    必须手工告诉它。凡是"时间"字段都该拆。
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            log.warning("日期列 %s 不存在，跳过", c)
            continue
        dt = pd.to_datetime(out[c], errors="coerce")
        n_bad = int(dt.isna().sum() - out[c].isna().sum())
        if n_bad > len(dt) * 0.05:
            log.warning("%s 有 %d 个值解析不成日期，可能不是日期列，跳过", c, n_bad)
            continue
        out[f"{c}__年"] = dt.dt.year
        out[f"{c}__月"] = dt.dt.month
        out[f"{c}__日"] = dt.dt.day
        out[f"{c}__星期"] = dt.dt.dayofweek
        out[f"{c}__季度"] = dt.dt.quarter
        out[f"{c}__年内第几天"] = dt.dt.dayofyear
        out[f"{c}__第几周"] = dt.dt.isocalendar().week.astype("float")
        out[f"{c}__是否周末"] = (dt.dt.dayofweek >= 5).astype("float")
        out[f"{c}__是否月初"] = (dt.dt.day <= 5).astype("float")
        out[f"{c}__是否月末"] = (dt.dt.day >= 25).astype("float")
        # 时间戳本身也是特征（能表达单调趋势）
        out[f"{c}__时间戳"] = (dt - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
        out = out.drop(columns=[c])
        log.info("日期列 %s 拆出 11 个周期特征", c)
    return out


def add_row_stats(df: pd.DataFrame, cols: list[str], *, tag: str = "行统计") -> pd.DataFrame:
    """行内横向统计：均值/标准差/最大/最小/极差/偏度。

    在生信里这叫"模块评分"（module score）——**一个样本的整体水平往往比单个字段更能说明问题**。
    放到业务数据上同理：一个用户的行为强度、一个门店的综合表现。
    """
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return df
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    out = df.copy()
    out[f"{tag}__均值"] = sub.mean(axis=1)
    out[f"{tag}__标准差"] = sub.std(axis=1)
    out[f"{tag}__最大"] = sub.max(axis=1)
    out[f"{tag}__最小"] = sub.min(axis=1)
    out[f"{tag}__极差"] = out[f"{tag}__最大"] - out[f"{tag}__最小"]
    out[f"{tag}__非零个数"] = (sub != 0).sum(axis=1)
    out[f"{tag}__负值个数"] = (sub < 0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[f"{tag}__变异系数"] = out[f"{tag}__标准差"] / out[f"{tag}__均值"].abs().replace(0, np.nan)
    log.info("行统计基于 %d 个数值列生成 8 个特征", len(cols))
    return out


def add_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """每行缺失了哪些字段、缺了多少。

    生信里"缺失模式本身就是信号"（比如某基因 panel 没测到可能说明样本质量差）。
    业务数据里同理：**一个人填表时跳过哪些栏位，可能比填了什么更能说明问题。**
    """
    out = df.copy()
    na = out.isna()
    out["缺失__总数"] = na.sum(axis=1)
    out["缺失__比例"] = na.mean(axis=1)
    return out


def add_frequency_features(
    df: pd.DataFrame, cols: list[str], *, min_count: int = 2
) -> pd.DataFrame:
    """频次编码：把"取值本身"换成"这个取值出现了多少次"。

    对高基数类别（ID 类、地区、机构）特别有用——既保留了信息又不用展开成几百列。
    注意：这里**不使用目标列**，所以不涉及泄漏。
    """
    out = df.copy()
    made = 0
    for c in cols:
        if c not in out.columns:
            continue
        vc = out[c].value_counts(dropna=True)
        if len(vc) < min_count:
            continue
        out[f"{c}__频次"] = out[c].map(vc).astype("float")
        made += 1
    if made:
        log.info("频次编码生成 %d 个特征", made)
    return out


def domain_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """领域特征入口 —— 开赛后由生信侧在这里补充。

    写什么？先问一句：**这个字段在现实里是怎么产生的？**
    机制驱动出来的特征通常最强，而且能在论文里讲出道理（对标评分第 1、4 轴）。

    举个例子（按赛题换）：
        # 若是"用户行为"数据
        df["行为__人均交互强度"] = df["点击"] / (df["活跃天数"] + 1)
        # 若是"供应链/物流"数据
        df["周转__库销比"] = df["库存"] / (df["销量"] + 1)

    要求：只用本行的取值，不许碰目标列，不许用测试集统计量。
    """
    return df


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame | None, list[str]]:
    """把原始表变成模型能吃的特征矩阵。

    返回 (X_train, X_test, 特征名列表)。X_test 在没有测试表时为 None。
    """
    f = cfg.get("features", {})
    target = cfg["data"]["target"]
    id_col = cfg["data"].get("id_col")

    X_tr, y, _ = split_xy(train, target, id_col)
    if test is None:
        X_te = None
    elif target in test.columns:  # 有些比赛测试集也给了标签（用于本地验证）
        X_te, _, _ = split_xy(test, target, id_col)
    else:
        X_te = test.drop(columns=[id_col], errors="ignore").copy()

    n_tr = len(X_tr)
    combined = pd.concat([X_tr, X_te], axis=0, ignore_index=True) if X_te is not None else X_tr.copy()
    log.info("特征工程输入：训练 %d 行，测试 %s", n_tr, f"{len(X_te)} 行" if X_te is not None else "无")

    if f.get("datetime", True):
        date_cols = f.get("datetime_cols") or [
            c for c in combined.columns if "date" in c.lower() or "time" in c.lower() or "日期" in c
        ]
        combined = add_datetime_features(combined, date_cols)

    if f.get("missing", True):
        combined = add_missing_features(combined)

    num_cols = combined.select_dtypes(include=np.number).columns.tolist()
    if f.get("row_stats", True) and num_cols:
        combined = add_row_stats(combined, num_cols)

    obj_cols = combined.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if f.get("frequency", True) and obj_cols:
        high_card = [c for c in obj_cols if combined[c].nunique() >= f.get("freq_min_unique", 5)]
        combined = add_frequency_features(combined, high_card)

    # 丢列只看训练集，再套到测试集（避免测试集影响特征选择）
    train_part, dropped = _drop_on_train(combined.iloc[:n_tr], cfg)
    combined = combined.drop(columns=dropped, errors="ignore")

    X_tr = combined.iloc[:n_tr].reset_index(drop=True)
    X_te = combined.iloc[n_tr:].reset_index(drop=True) if X_te is not None else None

    if f.get("categorical", True):
        X_tr, X_te = _align_categories(X_tr, X_te)

    names = list(X_tr.columns)
    log.info("最终特征：%d 列（其中类别型 %d 列）",
             len(names), int(sum(str(X_tr[c].dtype) == "category" for c in names)))
    return X_tr, X_te, names


def _drop_on_train(X_tr: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    from src.data import drop_useless_cols

    f = cfg.get("features", {})
    if not f.get("drop_useless", True):
        return X_tr, []
    return drop_useless_cols(
        X_tr,
        max_missing=f.get("max_missing", 0.98),
        drop_constant=f.get("drop_constant", True),
    )


def _align_categories(
    X_tr: pd.DataFrame, X_te: pd.DataFrame | None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """把类别列转成 pandas category，并**让训练集和测试集共用同一套类别取值**。

    这一步不做会出大错：LightGBM 读 category 列时用的是内部编码，
    两边类别集合不一致会导致同一个取值在训练/测试里编码不同，模型直接失效。
    """
    obj_cols = X_tr.select_dtypes(include=["object", "string"]).columns.tolist()
    for c in obj_cols:
        X_tr[c] = X_tr[c].fillna(NA_TOKEN).astype(str)
        if X_te is not None:
            X_te[c] = X_te[c].fillna(NA_TOKEN).astype(str)

        seen = set(X_tr[c].unique())
        categories = pd.Index(sorted(seen))
        if X_te is not None:
            extra = set(X_te[c].unique()) - seen
            if extra:
                log.warning("测试集有训练集没见过的取值，落到 %s：%s", NA_TOKEN, sorted(extra)[:5])
            categories = pd.Index(sorted(seen | extra))

        X_tr[c] = pd.Categorical(X_tr[c], categories=categories)
        if X_te is not None:
            X_te[c] = pd.Categorical(X_te[c], categories=categories)
    return X_tr, X_te


def main() -> None:
    ap = argparse.ArgumentParser(description="构建特征并存到 data/processed/")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    train = load_table(cfg["data"]["train"])
    test = load_table(cfg["data"]["test"]) if cfg["data"].get("test") else None

    X_tr, X_te, names = prepare_features(train, test, cfg)

    PATHS["processed"].mkdir(parents=True, exist_ok=True)
    out = PATHS["processed"] / f"{cfg['name']}_train.parquet"
    X_tr.to_parquet(out, index=False)
    log.info("训练特征 -> %s", out)

    if X_te is not None:
        out = PATHS["processed"] / f"{cfg['name']}_test.parquet"
        X_te.to_parquet(out, index=False)
        log.info("测试特征 -> %s", out)

    print(f"\n特征数 {len(names)}。前 30 个：")
    for c in names[:30]:
        print(f"  {c}")


if __name__ == "__main__":
    main()
