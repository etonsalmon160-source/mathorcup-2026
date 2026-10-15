.PHONY: help check env paper run-all run data train predict plots eda smoke gate validate ablate fuse conformal starters clean

CONFIG ?= src/configs/baseline.yaml
BUDGET ?= 0.10

PY := /home/eto/miniconda3/envs/mathorcup/bin/python
LATEX := /home/eto/.local/bin/xelatex

help:
	@echo "make env       建/更新 conda 环境"
	@echo "make check     环境自检（包版本 + 目录 + 种子）"
	@echo "make run-all   从 raw 数据跑到最终结果"
	@echo "make paper     编译论文 PDF"
	@echo "make eda       启动 JupyterLab"
	@echo "make clean     清理中间产物与 LaTeX 编译垃圾"

env:
	bash scripts/setup_env.sh

check:
	$(PY) scripts/check_env.py

run-all:
	bash scripts/run_all.sh $(CONFIG)

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

# 题型起手式：万一赛题不是预测题（评价类 / 优化类）
starters:
	$(PY) -m src.evaluate
	$(PY) -m src.optimize

# 共形风险控制：不用 0.5 当阈值，反解出"期望漏判率 ≤ 预算"的阈值（带分布无关保证）
conformal:
	$(PY) -m src.conformal --config $(CONFIG) --budget $(BUDGET)

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
