"""数据读入与体检。

开赛拿到数据后，先跑 `python -m src.data --config <配置>` 看一眼数据长什么样，
再决定特征怎么做。别跳过这一步。
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import get_logger, load_config, resolve, set_global_seed, SEED

log = get_logger("data")

_READERS = {
    ".csv": lambda p: pd.read_csv(p),
    ".tsv": lambda p: pd.read_csv(p, sep="\t"),
    ".txt": lambda p: pd.read_csv(p, sep=None, engine="python"),
    ".parquet": lambda p: pd.read_parquet(p),
    ".pq": lambda p: pd.read_parquet(p),
    ".feather": lambda p: pd.read_feather(p),
    ".xlsx": lambda p: pd.read_excel(p),
    ".xls": lambda p: pd.read_excel(p),
    ".json": lambda p: pd.read_json(p),
}


def load_table(path: str) -> pd.DataFrame:
    """按后缀自动选择读法。csv / tsv / parquet / excel / json 都认。"""
    p = resolve(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到数据文件：{p}")
    reader = _READERS.get(p.suffix.lower())
    if reader is None:
        raise ValueError(f"不认识的文件类型：{p.suffix}（支持 {sorted(_READERS)}）")
    df = reader(p)
    log.info("读入 %s：%d 行 × %d 列", p.name, df.shape[0], df.shape[1])
    return df


def drop_useless_cols(
    df: pd.DataFrame,
    *,
    max_missing: float = 0.98,
    drop_constant: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """丢掉几乎没有信息量的列。

    两类：① 缺失率超过 max_missing；② 整列只有一个取值（方差为 0）。
    被丢的列名会打印出来——**这些列本身可能就是业务信息**，别不看就丢。
    """
    dropped: list[str] = []

    miss = df.isna().mean()
    too_missing = miss[miss >= max_missing].index.tolist()
    dropped += too_missing
    if too_missing:
        log.info("缺失率 ≥ %.0f%% 丢掉 %d 列：%s", max_missing * 100, len(too_missing), too_missing[:10])

    if drop_constant:
        rest = df.drop(columns=dropped, errors="ignore")
        nunique = rest.nunique(dropna=True)
        constant = nunique[nunique <= 1].index.tolist()
        dropped += constant
        if constant:
            log.info("取值唯一丢掉 %d 列：%s", len(constant), constant[:10])

    return df.drop(columns=dropped, errors="ignore"), dropped


def split_xy(
    df: pd.DataFrame, target: str, id_col: str | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """切成 X / y / id。返回的 X 已剔除目标列和 ID 列。"""
    if target not in df.columns:
        raise KeyError(f"目标列 {target!r} 不在数据里。现有列：{list(df.columns)[:20]}")

    y = df[target]
    ids = df[id_col] if id_col and id_col in df.columns else None
    drop = [target] + ([id_col] if id_col and id_col in df.columns else [])
    X = df.drop(columns=drop)
    return X, y, ids


def describe(df: pd.DataFrame, target: str | None = None) -> None:
    """打印一份人看得懂的数据体检报告。"""
    print(f"\n=== 数据体检：{df.shape[0]} 行 × {df.shape[1]} 列 ===\n")

    kind = pd.DataFrame({
        "类型": df.dtypes.astype(str),
        "缺失数": df.isna().sum(),
        "缺失率": (df.isna().mean() * 100).round(2),
        "唯一值": df.nunique(dropna=True),
    })
    print(kind.sort_values("缺失率", ascending=False).to_string())
    print()

    num = df.select_dtypes(include=np.number)
    if not num.empty:
        print("--- 数值列分布 ---")
        print(num.describe().T.round(4).to_string())
        print()

    if target and target in df.columns:
        t = df[target]
        print(f"--- 目标列 {target} ---")
        print(f"类型：{t.dtype}，唯一值：{t.nunique()}")
        if t.dtype == object or t.nunique() <= 20:
            vc = t.value_counts(dropna=False)
            print((vc / len(t) * 100).round(3).to_string())
            if t.nunique() == 2:
                r = vc.iloc[0] / vc.iloc[-1]
                flag = "⚠ 类别不平衡，别只看准确率" if r > 4 else "类别大致均衡"
                print(f"→ 正负比 {r:.2f}:1 —— {flag}")
        else:
            print(t.describe().round(4).to_string())
        print()

    print("--- 前 3 行 ---")
    print(df.head(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="数据读入与体检")
    ap.add_argument("--config", required=True, help="实验配置 yaml")
    args = ap.parse_args()

    set_global_seed(SEED)
    cfg = load_config(args.config)
    d = cfg.get("data", {})

    train = load_table(d["train"])
    describe(train, target=d.get("target"))

    if d.get("test"):
        test = load_table(d["test"])
        describe(test)
        only_train = set(train.columns) - set(test.columns)
        only_test = set(test.columns) - set(train.columns)
        print(f"\n--- 训练/测试列差异 ---")
        print(f"只在训练集有的列：{sorted(only_train) or '无'}")
        print(f"只在测试集有的列：{sorted(only_test) or '无'}")


if __name__ == "__main__":
    main()
