#!/usr/bin/env bash
# 建/更新竞赛 conda 环境。幂等，可重复跑。
# 用法：bash scripts/setup_env.sh
set -euo pipefail

CONDA=/home/eto/miniconda3/bin/conda
ENV_NAME=mathorcup
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo ">>> 环境名：$ENV_NAME"
echo ">>> 工作区：$ROOT"

# 让本环境内的 pip 默认走清华镜像（裸连 PyPI 装依赖要几十分钟）
PIP_CONF="$HOME/miniconda3/envs/$ENV_NAME/pip.conf"
if [ ! -d "$(dirname "$PIP_CONF")" ]; then mkdir -p "$(dirname "$PIP_CONF")"; fi
cat > "$PIP_CONF" <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
extra-index-url = https://pypi.org/simple
timeout = 60
EOF
echo ">>> pip 镜像已配置：$PIP_CONF"

if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo ">>> 环境已存在，执行 update --prune"
  "$CONDA" env update -f "$ROOT/environment.yml" --prune
else
  echo ">>> 新建环境"
  "$CONDA" env create -f "$ROOT/environment.yml"
fi

echo ">>> 注册 Jupyter kernel"
"$CONDA" run -n "$ENV_NAME" python -m ipykernel install --user \
  --name "$ENV_NAME" --display-name "Python (mathorcup)"

echo ">>> 版本快照 -> $ROOT/docs/环境版本快照.txt"
"$CONDA" run -n "$ENV_NAME" python -m pip freeze > "$ROOT/docs/环境版本快照.txt"

echo ">>> 完成。激活：conda activate $ENV_NAME"
