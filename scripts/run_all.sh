#!/usr/bin/env bash
# 一键复现：从 data/raw 跑到最终提交文件。
# 开赛拿到数据后，把各步骤填进下面。见 docs/复现规范.md「一入口原则」。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/home/eto/miniconda3/envs/mathorcup/bin/python
cd "$ROOT"

echo ">>> [1/5] 数据加载与清洗"
# $PY -m src.load_data

echo ">>> [2/5] 特征工程"
# $PY -m src.features

echo ">>> [3/5] 训练"
# $PY -m src.train --config src/configs/baseline.yaml

echo ">>> [4/5] 预测并生成提交文件"
# $PY -m src.predict

echo ">>> [5/5] 出图"
# $PY -m src.plots

echo ">>> 完成。提交文件在 results/"
