#!/usr/bin/env bash
# 一键复现：从 data/raw 跑到门控证据与论文所需结果。
# 用法：bash scripts/run_all.sh
# 环境变量：PY（解释器）、SEEDS（随机种子，默认 42 43 44）
# 见 docs/复现规范.md「一入口原则」。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/eto/miniconda3/envs/mathorcup/bin/python}"
SEEDS="${SEEDS:-42 43 44}"
cd "$ROOT"

echo ">>> [0/8] 环境自检"
$PY scripts/check_env.py > /dev/null || { echo "环境有问题，先跑 make check 看详情"; exit 1; }

echo ">>> [1/8] 原始数据冻结校验"
$PY scripts/freeze_data.py --check

echo ">>> [2/8] 数据门 D0–D6"
$PY scripts/data_gate.py

echo ">>> [3/8] 生信式 QC（样本/特征/缺失机制/目标/缺陷/网络就绪度）"
$PY scripts/qc.py

echo ">>> [4/8] 三问主流程（Q1 13 模型 / Q2 / Q3 / 稳健性 / 三问互证）"
$PY scripts/run_v1.py --seeds $SEEDS

echo ">>> [5/8] 复合架构（树模型 + 感官潜状态，嵌套 CV）"
$PY scripts/run_composite.py --seeds $SEEDS

echo ">>> [6/8] G3 对抗验证（内部随机留出 + C2ST/MMD/KS）"
$PY scripts/g3_adversarial.py

echo ">>> [7/8] G4 消融（哨兵处理 / 检出率协变量 / 派生列）"
$PY scripts/g4_ablation.py

echo ">>> [8/8] 门控检查"
$PY scripts/gate_check.py

echo ">>> 完成。主结果在 results/v1/SUMMARY.md，门控证据在 results/ 下各目录。"
echo ">>> 提交前另跑：make pack PROBLEM=A TEAM_ID=<队号>"
