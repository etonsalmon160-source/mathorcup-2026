#!/usr/bin/env python
# 本文件由 AI 辅助编写；建模与算法设计决策由参赛队员做出。详见《AI工具使用详情说明》。
"""数据质控 CLI。核心逻辑在 src/qc.py。

用法：
    python scripts/qc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.qc import main  # noqa: E402

if __name__ == "__main__":
    main()
