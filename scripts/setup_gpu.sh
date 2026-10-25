#!/usr/bin/env bash
# 在 GPU 节点上补齐深度学习栈（表格 / 视觉 / 文本三条路）。
#
#   bash scripts/setup_gpu.sh              # 装
#   bash scripts/setup_gpu.sh --check      # 只体检，不装
#
# 原则：
#   1. **绝不碰 mathorcup 环境** —— 全部装进独立的 mathorcup-gpu。
#   2. **幂等** —— 重复跑不会坏，已装好的包跳过。
#   3. **驱动没到位也能先装** —— CUDA 版 torch 在没有驱动的机器上可以 import，
#      只是 is_available() 返回 False。所以现在装好，23 号驱动一上就能直接用。
#
# 为什么不用 conda 装 xgboost：conda 的 xgboost 是 CPU-only 构建
# （build_info()['USE_CUDA'] 为 False）。要 GPU 必须用 pip 的 wheel。
set -euo pipefail

ENV_NAME="${ENV_NAME:-mathorcup-gpu}"
BASE_ENV="${BASE_ENV:-mathorcup}"
CONDA="${CONDA:-/home/eto/miniconda3/bin/conda}"
ENV_DIR="${CONDA%/bin/conda}/envs/${ENV_NAME}"
PY="${ENV_DIR}/bin/python"

# 版本矩阵：torch 与 torchvision 必须成对，且都来自同一个 CUDA 索引。
TORCH_VER="${TORCH_VER:-2.7.1}"
TORCHVISION_VER="${TORCHVISION_VER:-0.22.1}"
PYTABKIT_VER="${PYTABKIT_VER:-1.7.3}"
XGB_VER="${XGB_VER:-3.4.1}"
CAT_VER="${CAT_VER:-1.2.10}"

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

# 所有提示一律走 stderr：pick_cuda_index 的 stdout 被 $(...) 捕获，
# 提示混进去会被当成 CUDA 标签，进而被 bash 当作算术表达式求值。
say()  { printf '\n\033[1m>>> %s\033[0m\n' "$*" >&2; }
warn() { printf '\033[33m  ! %s\033[0m\n' "$*" >&2; }
ok()   { printf '\033[32m  ✓ %s\033[0m\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# 1. 挑 CUDA 版本：由驱动版本决定，不能拍脑袋
# ---------------------------------------------------------------------------
pick_cuda_index() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        warn "没有 nvidia-smi —— 驱动还没装或没挂上卡。"
        warn "先按默认 cu126 装（不装驱动也能 import），23 号驱动到位再跑 --check。"
        echo "cu126"; return
    fi
    local drv
    # 注意：没有驱动时 nvidia-smi 会把"NVIDIA-SMI has failed..."打到 **stdout**，
    # 所以必须校验结果是纯数字，否则会被 bash 当成算术表达式去求值。
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
          | head -1 | cut -d. -f1 | tr -d '[:space:]')
    if ! [[ "${drv:-}" =~ ^[0-9]+$ ]]; then
        warn "nvidia-smi 在，但读不到驱动版本（卡或驱动还没挂上）。先按 cu126 装。"
        echo "cu126"; return
    fi
    if   (( drv >= 525 )); then echo "cu126"
    elif (( drv >= 470 )); then echo "cu118"
    else
        warn "驱动 ${drv} 太旧（<470），CUDA 12/11.8 都跑不了。请让管理员升级驱动。"
        echo "cu118"
    fi
}

CUDA_TAG=$(pick_cuda_index)
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
say "CUDA 轮子索引：${CUDA_TAG}（驱动版本决定，不是随便选的）"

# ---------------------------------------------------------------------------
# 2. 建独立环境
# ---------------------------------------------------------------------------
if [[ -x "$PY" ]]; then
    ok "环境已存在：$ENV_DIR"
elif (( CHECK_ONLY )); then
    echo "环境还没建，先跑 bash scripts/setup_gpu.sh"; exit 1
else
    say "克隆 ${BASE_ENV} -> ${ENV_NAME}（保证 GBDT 版本与竞赛环境完全一致）"
    "$CONDA" create -n "$ENV_NAME" --clone "$BASE_ENV" -y
fi

[[ -x "$PY" ]] || { echo "✗ 找不到 $PY"; exit 1; }
"$PY" -V

(( CHECK_ONLY )) && { exec "$PY" scripts/gpu_check.py --preflight; }

# ---------------------------------------------------------------------------
# 3. 装包
# ---------------------------------------------------------------------------
# pip 的 Tsinghua 镜像对普通包很快，但 torch 的 CUDA 轮子只在官方索引有，
# 所以 torch 单独指定 --index-url，其余走镜像 + PyPI 兜底。
say "1/5 PyTorch ${TORCH_VER} (${CUDA_TAG}) + torchvision ${TORCHVISION_VER}"
"$PY" -m pip install --no-cache-dir \
    "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" \
    --index-url "$TORCH_INDEX" \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

say "2/5 表格深度模型：pytabkit (RealMLP / FT-Transformer)"
"$PY" -m pip install --no-cache-dir "pytabkit==${PYTABKIT_VER}"

say "3/5 树模型换成 pip 的 GPU 构建（conda 那份是 CPU-only）"
# --force-reinstall 是故意的：conda 版文件已经在 site-packages 里，
# 不强制替换的话 pip 会认为"已满足"而跳过，结果是假的 GPU 支持。
"$PY" -m pip install --no-cache-dir --force-reinstall --no-deps \
    "xgboost==${XGB_VER}" "catboost==${CAT_VER}"
# --no-deps 之后把它们自己的依赖补齐（大多数在 clone 里已经有了）
"$PY" -m pip install --no-cache-dir "xgboost==${XGB_VER}" "catboost==${CAT_VER}"

say "4/5 视觉：timm + OpenCV（迁移学习的主力）"
"$PY" -m pip install --no-cache-dir timm opencv-python-headless

say "5/5 文本：transformers + sentence-transformers（把文本变成特征）"
"$PY" -m pip install --no-cache-dir transformers sentence-transformers

# 可选：目标检测。YOLO 是 AGPL-3.0，学术使用没问题，但论文里要注明。
if [[ "${WITH_YOLO:-0}" == "1" ]]; then
    say "可选：ultralytics (YOLO)"
    "$PY" -m pip install --no-cache-dir ultralytics
fi

# 可选：图神经网络。只有赛题是"实体+关系"结构时才需要，且要匹配 CUDA 版本。
if [[ "${WITH_GRAPH:-0}" == "1" ]]; then
    say "可选：PyTorch Geometric"
    "$PY" -m pip install --no-cache-dir torch-geometric
    "$PY" -m pip install --no-cache-dir torch-scatter torch-sparse \
        -f "https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"
fi

# ---------------------------------------------------------------------------
# 4. 体检
# ---------------------------------------------------------------------------
say "体检"
"$PY" scripts/gpu_check.py --preflight

say "完成。环境：$ENV_DIR"
echo "  用法：make dl PY=${PY}"
echo "  23 号驱动到位后：bash scripts/setup_gpu.sh --check   （必须全绿）"
