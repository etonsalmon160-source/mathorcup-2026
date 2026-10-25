#!/usr/bin/env python
# 本文件由 AI 辅助编写；建模与算法设计决策由参赛队员做出。详见《AI工具使用详情说明》。
"""数据门 CLI。核心逻辑在 src/data_gate.py。

用法：
    python scripts/data_gate.py             跑 D0–D6，失败退出码 1
    python scripts/data_gate.py --bless     首次生成/确认刷新 docs/data_contract/ 契约
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_gate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
