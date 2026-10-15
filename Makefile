.PHONY: help check env paper paper-quick run-all eda clean

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
	bash scripts/run_all.sh

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
