"""全局配置：路径、随机种子、日志。

所有脚本从这里取路径，不要在别处写死绝对路径。见 docs/复现规范.md。
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

SEED = 42

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
