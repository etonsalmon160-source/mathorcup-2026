.PHONY: help check env paper run-all run data train predict plots eda smoke gate validate ablate fuse conformal starters drift clean gpu-setup gpu-check gpu-preflight dl pack freeze freeze-check freeze-external freeze-check-external data-gate qc q3-probe v1 v1-quick composite g3 g4

CONFIG ?= src/configs/baseline.yaml
BUDGET ?= 0.10
SEEDS ?= 42 43 44

PY ?= /home/eto/miniconda3/envs/mathorcup/bin/python
LATEX := /home/eto/.local/bin/xelatex

# GPU：make train GPU=1        树模型改用 GPU（xgboost / catboost）
#      make dl  PY=/home/eto/miniconda3/envs/mathorcup-gpu/bin/python
GPU ?= 0
export MATHORCUP_GPU := $(if $(filter 1 true yes,$(GPU)),1,0)

help:
	@echo "make env       建/更新 conda 环境"
	@echo "make check     环境自检（包版本 + 目录 + 种子）"
	@echo "make freeze    首次冻结原始数据（哈希清单 + zip 锚定）"
	@echo "make freeze-check  校验原始数据没被改"
	@echo "make freeze-external       首次冻结补充数据（非官方，见 docs/补充数据预注册.md）"
	@echo "make freeze-check-external 校验补充数据没被改"
	@echo "make data-gate 数据门 D0–D6（结构/缺失/变换/离群/阈值/自洽/契约）"
	@echo "make qc        生信式 QC（样本/特征/缺失机制/目标/缺陷/网络就绪度）"
	@echo "make run-all   从 raw 数据跑到最终结果"
	@echo "make v1        第一版全流程（Q1 13 模型 / Q2 / Q3 / 稳健性 / 互证）"
	@echo "make v1-quick  第一版快速自检（单种子、少模型，约 2 分钟）"
	@echo "make composite 复合架构（树模型 + 感官潜状态，嵌套 CV）"
	@echo "make g3        G3 证据：内部留出对抗验证 + C2ST/MMD/KS"
	@echo "make g4        G4 证据：预处理/特征组消融"
	@echo "make paper     编译论文 PDF"
	@echo "make eda       启动 JupyterLab"
	@echo "make clean     清理中间产物与 LaTeX 编译垃圾"
	@echo ""
	@echo "GPU:"
	@echo "make gpu-setup     装 GPU 环境（独立 mathorcup-gpu，不碰现有环境）"
	@echo "make gpu-preflight 验安装（没驱动也能跑）"
	@echo "make gpu-check     驱动到位后验 GPU（必须全绿）"
	@echo "make train GPU=1   树模型用 GPU（xgboost / catboost）"
	@echo "make dl            表格深度模型 RealMLP"
	@echo ""
	@echo "提交:"
	@echo "make pack PROBLEM=A TEAM_ID=2600001   打包支撑材料（代码+结果+运行说明）"

env:
	bash scripts/setup_env.sh

check:
	$(PY) scripts/check_env.py

# 原始数据冻结：哈希清单 + 官方 zip 逐字节锚定。清单要入库，数据不入库。
freeze:
	$(PY) scripts/freeze_data.py --init

freeze-check:
	$(PY) scripts/freeze_data.py --check

# 补充数据（data/external/）：非赛题官方，用途分级见 docs/补充数据预注册.md
freeze-external:
	$(PY) scripts/freeze_data.py --external-init

freeze-check-external:
	$(PY) scripts/freeze_data.py --external-check

# 数据门：结构、缺失、变换口径、离群、阈值、相关矩阵自洽、契约漂移
data-gate:
	$(PY) scripts/data_gate.py

# 生信式 QC：Python 六层 + R 侧 WGCNA/vegan
qc:
	$(PY) scripts/qc.py

# 问题三迁移可行性探针（潜状态 + 缺陷约束 QP）
q3-probe:
	$(PY) scripts/q3_probe.py

run-all:
	bash scripts/run_all.sh $(CONFIG)

# 第一版端到端对比：Q1 / Q2 / Q3 / 稳健性 / 三问互证。结果在 results/v1/SUMMARY.md
v1:
	$(PY) scripts/run_v1.py --seeds $(SEEDS)

v1-quick:
	$(PY) scripts/run_v1.py --quick --seeds 42

composite:        # 复合架构：树模型 + 感官潜状态（嵌套 CV，防选模泄漏）
	$(PY) scripts/run_composite.py --seeds $(SEEDS)

g3:               # G3 证据：内部随机留出对抗验证 + C2ST/MMD/KS
	$(PY) scripts/g3_adversarial.py

g4:               # G4 证据：预处理/特征组消融（结果写到 results/q1/ablation.csv）
	$(PY) scripts/g4_ablation.py

importance:       # Q1 主模型置换重要性 + 模块汇总 + 与 Q2 Lasso 频率对照
	$(PY) scripts/q1_importance.py --seeds $(SEEDS)

importance-plot:  # 只重画重要性图，不重算（读 results/q1/importance_by_target.csv）
	$(PY) scripts/q1_importance.py --seeds $(SEEDS) --from-cache

data:            # 先看数据长什么样，再决定特征怎么做
	$(PY) -m src.data --config $(CONFIG)

train:
	$(PY) -m src.train --config $(CONFIG)

predict:
	$(PY) -m src.predict --config $(CONFIG)

plots:
	$(PY) -m src.plots --config $(CONFIG)

# 用造假数据把整条流水线跑一遍，确认代码没问题（不碰真实数据）
smoke:
	$(PY) -m src.selftest

# 七道门自动判 PASS / FAIL，秒级。见 docs/门控系统.md
gate:
	$(PY) scripts/gate_check.py

# 对抗验证：训练集和测试集是不是同一个分布（查泄漏/漂移）
validate:
	$(PY) -m src.validate --config $(CONFIG)

# 特征组消融：砍掉每组看主指标掉多少（G4 的证据）
ablate:
	$(PY) -m src.ablate --config $(CONFIG)

# 融合/门控：多专家加权组合，必须赢过最好单模型才算数（G5 的证据）
fuse:
	$(PY) -m src.fuse --config $(CONFIG)

# 分布漂移检验：MMD（整体有没有变）+ 逐特征 KS（哪里变了）
drift:
	$(PY) -m src.drift

# 题型起手式：万一赛题不是预测题（评价类 / 优化类）
starters:
	$(PY) -m src.evaluate
	$(PY) -m src.optimize

# 共形风险控制：不用 0.5 当阈值，反解出"期望漏判率 ≤ 预算"的阈值（带分布无关保证）
conformal:
	$(PY) -m src.conformal --config $(CONFIG) --budget $(BUDGET)

# GPU：一条命令装好独立环境（不碰 mathorcup）。见 docs/GPU部署清单.md
gpu-setup:
	bash scripts/setup_gpu.sh

# 没驱动也能跑：验安装完整
gpu-preflight:
	$(PY) scripts/gpu_check.py --preflight

# 驱动到位后跑：必须全绿才算真的能用 GPU
gpu-check:
	$(PY) scripts/gpu_check.py

# 表格深度模型（RealMLP），要 mathorcup-gpu 环境：make dl PY=.../mathorcup-gpu/bin/python
dl:
	$(PY) -m src.dl --config $(CONFIG)

# 打包支撑材料（题号+参赛编号+附件.zip），代码 + 结果数据集 + 运行说明
# 例：make pack PROBLEM=A TEAM_ID=2600001
pack:
	$(PY) scripts/pack_submission.py --problem $(PROBLEM) --team-id $(TEAM_ID)

paper:
	cd paper && $(LATEX) -interaction=nonstopmode main.tex
	# 尚无 \cite 时 bibtex 会返回非零，用 || true 容错（不影响正文）
	cd paper && bibtex main || true
	cd paper && $(LATEX) -interaction=nonstopmode main.tex
	cd paper && $(LATEX) -interaction=nonstopmode main.tex
	@echo ">>> 论文：paper/main.pdf"

eda:
	$(PY) -m jupyterlab --notebook-dir=notebooks

clean:
	rm -rf paper/*.aux paper/*.log paper/*.out paper/*.toc paper/*.bbl \
	       paper/*.blg paper/*.fls paper/*.fdb_latexmk paper/*.synctex.gz
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '.ipynb_checkpoints' -type d -prune -exec rm -rf {} +
	@echo ">>> 已清理"
