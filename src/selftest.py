"""自检：用造假数据把整条流水线跑一遍。

    make smoke        或        python -m src.selftest

不碰 data/raw 里的真实数据。它要证明的是：**代码骨架是通的**。
开赛前每次改完代码跑一次，比赛期间就不用担心"代码本身有 bug"。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import PATHS, ROOT, load_config

MODELS = ["lightgbm", "xgboost", "catboost"]
PREFIX = "_selftest"


def make_synthetic(kind: str, n: int, seed: int, offset: int = 0) -> pd.DataFrame:
    """造一份"长得像真题"的表：数值列 + 低基数类别 + 高基数类别 + 日期 + 缺失值。"""
    rng = np.random.default_rng(seed)

    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(5, 2, n)
    x3 = rng.exponential(2, n)
    x4 = rng.integers(0, 100, n).astype(float)

    城市 = rng.choice(["北京", "上海", "广州", "深圳", "成都"], n)
    门店 = rng.choice([f"S{i:04d}" for i in range(offset, offset + 200)], n)
    日期 = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")

    df = pd.DataFrame({
        "编号": [f"T{i:06d}" for i in range(offset, offset + n)],
        "数值A": x1, "数值B": x2, "数值C": x3, "数值D": x4,
        "城市": 城市, "门店": 门店, "日期": 日期,
        "常量列": "同一个值",          # 应该被自动丢掉
    })

    # 人为制造缺失：真实数据不会这么干净
    for c in ["数值B", "数值C", "城市"]:
        df.loc[rng.random(n) < 0.08, c] = np.nan

    signal = 2.0 * x1 + 0.6 * x2 - 1.2 * x3 + 0.02 * x4 + (城市 == "北京") * 1.5
    if kind == "classification":
        p = 1 / (1 + np.exp(-(signal - signal.mean()) / signal.std()))
        df["标签"] = (p > 0.6).astype(int)
    else:
        df["标签"] = signal + rng.normal(0, 1.0, n)
    return df


def test_name(model: str, kind: str) -> str:
    """产物目录名带上任务类型——否则分类和回归会写进同一个目录、互相覆盖。"""
    return f"{PREFIX}_{kind}_{model}"


# 三个库的参数名不通用，自检必须各给各的 —— 这也是在验证参数守卫别误伤
SELFTEST_PARAMS = {
    "lightgbm": {"n_estimators": 120, "learning_rate": 0.1, "num_leaves": 31},
    "xgboost": {"n_estimators": 120, "learning_rate": 0.1, "max_depth": 5},
    "catboost": {"iterations": 120, "learning_rate": 0.1, "depth": 5},
}


def write_config(path: Path, model: str, kind: str, train: Path, test: Path) -> None:
    cfg = {
        "name": test_name(model, kind),
        "seed": 42,
        "data": {"train": str(train), "test": str(test), "target": "标签", "id_col": "编号"},
        "cv": {"strategy": "stratified" if kind == "classification" else "kfold",
               "n_splits": 3},
        "model": model,
        "params": SELFTEST_PARAMS[model],
        "features": {"datetime": True, "missing": True, "row_stats": True,
                     "frequency": True, "categorical": True, "drop_useless": True},
        "submission": {"pred_col": "prediction"},
    }
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def run_step(module: str, config: Path) -> None:
    cmd = [sys.executable, "-m", module, "--config", str(config)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise SystemExit(f"✗ {module} 失败（退出码 {r.returncode}）")


def check_outputs(name: str, n_test: int, kind: str, deep: bool = False) -> dict:
    """检查产物是否齐全、形状对不对、有没有 NaN。"""
    out_dir = PATHS["results"] / name
    problems = []

    sub_path = out_dir / "submission.csv"
    if not sub_path.exists():
        raise SystemExit(f"✗ 没生成 {sub_path}")
    sub = pd.read_csv(sub_path)
    if len(sub) != n_test:
        problems.append(f"提交行数 {len(sub)} ≠ 测试集 {n_test}")
    if "prediction" not in sub.columns:
        problems.append("提交文件缺 prediction 列")
    if sub["prediction"].isna().any():
        problems.append("提交文件里有空值")
    if kind == "classification" and sub["prediction"].nunique() < 2:
        problems.append("分类任务只预测出一个类别")

    required = ["metrics.json", "oof_predictions.csv", "feature_importance.csv"]
    if deep:   # 对抗验证 / 消融只在一条组合上跑，免得自检太慢
        required += ["adversarial.json", "adversarial_features.csv", "ablation.csv"]
    for f in required:
        if not (out_dir / f).exists():
            problems.append(f"缺产物 {f}")

    figs = sorted((PATHS["figures"] / name).glob("*.png")) if (PATHS["figures"] / name).exists() else []
    if not figs:
        problems.append("没有出图")

    return {"problems": problems, "n_sub": len(sub), "n_figs": len(figs)}


def check_shipped_configs() -> list[tuple[str, str]]:
    """把 src/configs/ 下每份配置的"模型 + 参数"真跑一遍。

    专挡最容易踩的坑：换模型时参数名不通用——xgboost 会静默忽略不认识的参数
    （模型悄悄变差），catboost 直接报错。这里只验参数，不碰配置里的数据路径。
    """
    from src.models import build_model, validate_params

    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "a": rng.normal(size=80), "b": rng.normal(size=80),
        "c": pd.Categorical(rng.choice(["x", "y"], 80)),
    })
    y = (X["a"] + X["b"] > 0).astype(int).to_numpy()

    out = []
    for cfg_path in sorted(PATHS["configs"].glob("*.yaml")):
        cfg = load_config(cfg_path)
        model = cfg.get("model", "lightgbm")
        try:
            m = build_model(model, cfg.get("params") or {}, "classification", 42)
            if model == "catboost":
                Xc = X.copy()
                Xc["c"] = Xc["c"].astype(str)
                m.fit(Xc, y, cat_features=["c"])
            else:
                m.fit(X, y)
            out.append((cfg_path.name, "✓ 能训起来"))
        except Exception as e:
            out.append((cfg_path.name, f"✗ {type(e).__name__}: {str(e).splitlines()[0][:110]}"))

    # 反向检查：故意把 LightGBM 的参数喂给 xgboost，守卫必须拦住
    lgb_params = load_config(PATHS["configs"] / "baseline.yaml").get("params") or {}
    try:
        validate_params("xgboost", lgb_params)
        out.append(("参数守卫", "✗ 没拦住「LightGBM 参数喂给 xgboost」"))
    except ValueError:
        out.append(("参数守卫", "✓ 误用参数会被当场拦下"))
    return out


def check_guards_falsifiable() -> list[tuple[str, str]]:
    """**守卫证伪**：喂故意做坏的输入，确认每道门真的会红。

    > 只"跑通"不算验证 —— **永远通过的守卫是假的**。
    > （这条准则抄自 `~/luad_v2/docs/scientific_rigor_and_audit.md` 的审计流程第 5 条。）

    做法：在 results/ 下造一个假实验目录，塞进各种"故意做坏"的产物，
    看 gate_check 的对应函数**是否真的返回 FAIL**。测完删掉。
    """
    import importlib.util
    import json as _json

    spec = importlib.util.spec_from_file_location("_gc", ROOT / "scripts" / "gate_check.py")
    gc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gc)

    # 注意：不能叫 _ 开头的名字 —— 门的扫描器会跳过它们（那是为排除自检产物），
    # 假数据就会被无视，守卫永远不红。这个坑正是证伪测试第一次跑出来的。
    fake = PATHS["results"] / "zz_guardtest"
    fake.mkdir(parents=True, exist_ok=True)
    out = []

    def probe(name, fn, expect, write):
        write()
        status, detail = fn()
        ok = status == expect
        out.append((name, f"{'✓' if ok else '✗'} 期望 {expect}，实得 {status}"
                         + ("" if ok else f" —— {detail[:60]}")))

    def w(path, obj, csv_rows=None):
        if csv_rows is not None:
            import csv as _csv
            with open(fake / path, "w", newline="", encoding="utf-8") as f:
                wtr = _csv.DictWriter(f, fieldnames=list(csv_rows[0]))
                wtr.writeheader(); wtr.writerows(csv_rows)
        else:
            (fake / path).write_text(_json.dumps(obj, ensure_ascii=False), encoding="utf-8")

    # G3：对抗验证 AUC 超阈值 → 必须 FAIL
    probe("G3 对抗验证超阈值", gc.g3_leakage, gc.FAIL,
          lambda: w("adversarial.json", {"auc": 0.93}))
    # G4：某特征组被砍掉后反而变好 → 必须 FAIL
    probe("G4 特征组负贡献", gc.g4_ablation, gc.FAIL,
          lambda: w("ablation.csv", None, [{"特征组": "行内统计", "与完整模型的差": "0.0150"}]))
    # G5：融合没赢过单模型 → 必须 FAIL
    probe("G5 融合未增益", gc.g5_model, gc.FAIL,
          lambda: (w("ablation.csv", None, [{"特征组": "行内统计", "与完整模型的差": "-0.0100"}]),
                   w("fusion.json", {"fused_beats_best_single": False}),
                   w("metrics.json", {"primary_metric": "f1", "seeds": [1, 2, 3],
                                      "cv": {"f1": 0.9, "f1_std": 0.001}})))
    # G0/G1 的"没填完就拦下"、G2 的"提交文件格式错就拦下"，在真实赛题数据到来后现场验证；
    # 上面三条（G3/G4/G5）是纯数值判据，可以在这里用假输入证伪。
    shutil.rmtree(fake, ignore_errors=True)
    return out


def check_starter_kits() -> list[tuple[str, str]]:
    """评价类 / 优化类起手式的冒烟测试。

    这两块平时用不上（赛题多半是预测题），但**正是"平时用不上"的部分最容易烂**。
    """
    import io
    import contextlib

    out = []
    for name, mod_name in [("评价类（AHP+熵权+TOPSIS）", "src.evaluate"),
                           ("优化类（LP/MILP/指派/灵敏度）", "src.optimize")]:
        try:
            mod = __import__(mod_name, fromlist=["demo"])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                mod.demo()
            out.append((name, "✓ 跑通"))
        except Exception as e:                              # noqa: BLE001
            out.append((name, f"✗ {type(e).__name__}: {str(e)[:90]}"))
    return out


def cleanup(names: list[str]) -> None:
    for name in names:
        shutil.rmtree(PATHS["results"] / name, ignore_errors=True)
        shutil.rmtree(PATHS["figures"] / name, ignore_errors=True)
        for f in PATHS["models"].glob(f"{name}.*"):
            f.unlink(missing_ok=True)
        for f in PATHS["processed"].glob(f"{name}_*.parquet"):
            f.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="用造假数据自检整条流水线")
    ap.add_argument("--keep", action="store_true", help="保留产物，方便看结果")
    args = ap.parse_args()

    names = [test_name(m, k) for k in ("classification", "regression") for m in MODELS]
    cleanup(names)  # 先清掉上次的残留

    tmp = Path(tempfile.mkdtemp(prefix="mathorcup_selftest_"))
    print(f"临时数据目录：{tmp}\n")

    print("===== 守卫证伪：故意做坏的输入，守卫必须变红 =====")
    guard_results = check_guards_falsifiable()
    for name, status in guard_results:
        print(f"  {status:34s} {name}")
    guard_failed = any(s.startswith("✗") for _, s in guard_results)

    print("\n===== 题型起手式（评价类 / 优化类）=====")
    starter_results = check_starter_kits()
    for name, status in starter_results:
        print(f"  {status:30s} {name}")
    starter_failed = any(s.startswith("✗") for _, s in starter_results)

    print("\n===== 检查 src/configs/ 下的配置文件 =====")
    cfg_results = check_shipped_configs()
    for name, status in cfg_results:
        print(f"  {status:44s} {name}")
    cfg_failed = any(s.startswith("✗") for _, s in cfg_results)

    results, failed = [], (cfg_failed or guard_failed or starter_failed)
    for kind in ["classification", "regression"]:
        n_tr, n_te = 900, 300
        tr = make_synthetic(kind, n_tr, seed=1)
        # 测试集故意混入训练集没见过的门店号，验证类别对齐不会崩
        te = make_synthetic(kind, n_te, seed=2, offset=150)
        tr.to_csv(tmp / f"{kind}_train.csv", index=False)
        te.to_csv(tmp / f"{kind}_test.csv", index=False)

        print(f"===== 任务：{kind}（训练 {n_tr} / 测试 {n_te}）=====")
        for model in MODELS:
            cfg = tmp / f"{kind}_{model}.yaml"
            write_config(cfg, model, kind, tmp / f"{kind}_train.csv", tmp / f"{kind}_test.csv")
            name = test_name(model, kind)
            print(f"\n--- {kind} / {model} ---")
            try:
                deep = (model == "lightgbm" and kind == "classification")
                run_step("src.train", cfg)
                run_step("src.predict", cfg)
                run_step("src.plots", cfg)
                if deep:
                    run_step("src.validate", cfg)
                    run_step("src.ablate", cfg)
                info = check_outputs(name, n_te, kind, deep)
            except SystemExit as e:
                print(f"✗ {e}")
                failed = True
                continue

            status = "✓" if not info["problems"] else "✗ " + "；".join(info["problems"])
            failed = failed or bool(info["problems"])
            print(f"  {status}  提交 {info['n_sub']} 行，出图 {info['n_figs']} 张")
            results.append((kind, model, status))

    if not args.keep:
        cleanup(names)
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        print(f"\n产物保留在 results/{PREFIX}_* 和 figures/{PREFIX}_*；临时数据在 {tmp}")

    print("\n" + "=" * 58)
    for name, status in guard_results:
        print(f"  守卫 {name:30s} {status}")
    for name, status in starter_results:
        print(f"  起手式 {name:26s} {status}")
    for name, status in cfg_results:
        print(f"  配置 {name:34s} {status}")
    for kind, model, status in results:
        print(f"  {kind:14s} {model:10s} {status}")
    print("=" * 58)
    if failed:
        raise SystemExit("== 自检未通过，先修上面标 ✗ 的地方 ==")
    print("== 自检通过：整条流水线（训练 → 预测 → 出图）跑得通 ==")


if __name__ == "__main__":
    main()
