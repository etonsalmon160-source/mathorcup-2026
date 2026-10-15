"""全局配置：路径、随机种子、日志。

所有脚本从这里取路径，不要在别处写死绝对路径。见 docs/复现规范.md。
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import yaml

SEED = 42

# 线程数上限。**不要用 -1**（= 用满全部核）：
# 本机 20 核实测，LightGBM 开 n_jobs=-1 会从"0.1 秒"退化到"几分钟卡死"，
# 因为线程互相抢占空转。小数据上 4 线程最快，大数据再加。
# 想改：在配置里写 params.n_jobs，或改这个默认值。
N_JOBS = min(8, os.cpu_count() or 1)

# ---- 路径（相对本文件定位，clone 到哪都能跑）----
ROOT = Path(__file__).resolve().parent.parent
PATHS = {
    "root": ROOT,
    "raw": ROOT / "data" / "raw",
    "interim": ROOT / "data" / "interim",
    "processed": ROOT / "data" / "processed",
    "external": ROOT / "data" / "external",
    "models": ROOT / "models",
    "results": ROOT / "results",
    "figures": ROOT / "figures",
    "configs": ROOT / "src" / "configs",
    "logs": ROOT / "logs",
}


def load_config(path: str | Path) -> dict:
    """读实验配置（src/configs/*.yaml）。相对路径按仓库根解析。"""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"找不到配置文件：{p}")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["name"] = cfg.get("name") or p.stem
    cfg["_config_path"] = str(p)
    return cfg


def resolve(path: str | Path) -> Path:
    """数据路径解析：绝对路径原样返回，相对路径按仓库根解析。"""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def run_dir(cfg: dict) -> Path:
    """本次实验的产物目录 results/<name>/，不存在就建。"""
    d = PATHS["results"] / cfg["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


# 台账列固定，别随任务类型变——否则分类和回归的行会串列
LEDGER_COLUMNS = ["time", "name", "config", "model", "task", "n_seeds", "n_features",
                  "primary_metric", "primary_value", "primary_std", "baseline", "vs_baseline"]


def record_run(
    cfg: dict, task: str, primary_metric: str,
    value: float, std: float, baseline: float, gain: float,
) -> Path:
    """往 results/ledger.csv 追加一行实验台账。见 docs/复现规范.md 第 4 节。"""
    import csv
    from datetime import datetime

    ledger = PATHS["results"] / "ledger.csv"
    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "name": cfg["name"],
        "config": Path(cfg.get("_config_path", "")).name,
        "model": cfg.get("model", ""),
        "task": task,
        "n_seeds": len(cfg.get("seeds", [cfg.get("seed", SEED)])),
        "n_features": cfg.get("_n_features", ""),
        "primary_metric": primary_metric,
        "primary_value": round(value, 6),
        "primary_std": round(std, 6),
        "baseline": round(baseline, 6),
        "vs_baseline": round(gain, 6),
    }

    ledger.parent.mkdir(parents=True, exist_ok=True)
    new = not ledger.exists()
    with open(ledger, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return ledger


def config_hash(cfg: dict) -> str:
    """配置指纹，写进日志方便对账。"""
    import hashlib

    payload = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def set_global_seed(seed: int = SEED) -> None:
    """把种子按到所有随机源上。任何涉及随机的实验，开头先调这个。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """带时间戳的 logger，同时输出到控制台和 logs/<name>.log。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    PATHS["logs"].mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(PATHS["logs"] / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
