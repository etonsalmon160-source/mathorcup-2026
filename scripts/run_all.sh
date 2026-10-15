#!/usr/bin/env bash
# 一键复现：从 data/raw 跑到最终提交文件。
# 用法：bash scripts/run_all.sh [配置名]      默认 baseline
# 见 docs/复现规范.md「一入口原则」。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/eto/miniconda3/envs/mathorcup/bin/python
CONFIG="${1:-src/configs/baseline.yaml}"
cd "$ROOT"

echo ">>> [0/7] 环境自检"
$PY scripts/check_env.py > /dev/null || { echo "环境有问题，先跑 make check 看详情"; exit 1; }

echo ">>> [1/7] 数据体检（只看不写）"
$PY -m src.data --config "$CONFIG"

echo ">>> [2/7] 训练 + 交叉验证"
$PY -m src.train --config "$CONFIG"

echo ">>> [3/7] 对抗验证（训练/测试是否同分布）"
$PY -m src.validate --config "$CONFIG"

echo ">>> [4/7] 特征组消融 + 融合/门控"
$PY -m src.ablate --config "$CONFIG"
$PY -m src.fuse --config "$CONFIG"

echo ">>> [5/7] 共形风险控制（给决策加保证）"
$PY -m src.conformal --config "$CONFIG" --budget "${BUDGET:-0.10}" || \
  echo "  （非二分类任务会跳过，不影响流程）"

echo ">>> [6/7] 出提交文件"
$PY -m src.predict --config "$CONFIG"

echo ">>> [7/7] 出图"
$PY -m src.plots --config "$CONFIG"

echo ">>> 台账 + 过门检查"
$PY - <<'EOF'
from pathlib import Path
from src.config import PATHS
p = PATHS["results"] / "ledger.csv"
print(p.read_text(encoding="utf-8") if p.exists() else "（还没有台账）")
EOF

$PY scripts/gate_check.py

echo ">>> 完成。提交文件在 results/<配置名>/submission.csv"
