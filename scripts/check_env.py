#!/usr/bin/env python
"""环境自检：确认竞赛环境真的可用。跑 `make check`。"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

REQUIRED = [
    "numpy", "pandas", "scipy", "sklearn", "lightgbm", "xgboost",
    "catboost", "optuna", "shap", "statsmodels", "matplotlib",
    "seaborn", "plotly", "pyarrow", "polars", "duckdb", "yaml", "tqdm",
]
OPTIONAL = ["torch", "transformers", "jieba", "umap", "networkx"]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    ok = True

    print(f"Python {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 10):
        print("  ! 建议用 Python >= 3.10，当前偏旧")
        ok = False

    print("\n-- 必需包 --")
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            print(f"  ok   {name:14s} {ver}")
        except ImportError:
            print(f"  MISS {name:14s}  <-- 需要安装")
            ok = False

    print("\n-- 可选包（缺了不影响起步）--")
    for name in OPTIONAL:
        try:
            mod = importlib.import_module(name)
            print(f"  ok   {name:14s} {getattr(mod, '__version__', '?')}")
        except ImportError:
            print(f"  ---- {name:14s} 未安装")

    print("\n-- 目录 --")
    for sub in ["data/raw", "data/interim", "data/processed",
                "src", "results", "figures", "submissions", "paper"]:
        p = root / sub
        print(f"  {'ok  ' if p.exists() else 'MISS'} {sub}")
        if not p.exists():
            ok = False

    sys.path.insert(0, str(root))
    from src.config import SEED, set_global_seed  # noqa: E402

    set_global_seed(SEED)
    import numpy as np

    a, b = np.random.rand(3), np.random.rand(3)
    set_global_seed(SEED)
    c = np.random.rand(3)
    same = bool((a == c).all())
    print(f"\n-- 随机种子 SEED={SEED} 可复现：{'是' if same else '否 !!'} --")
    if not same:
        ok = False

    print("\n" + ("== 自检通过，可以开工 ==" if ok else "== 有问题，见上面标记 =="))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
