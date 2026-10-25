#!/usr/bin/env python
"""GPU 环境体检。

    python scripts/gpu_check.py --preflight   # 没驱动也能跑：验安装是否完整
    python scripts/gpu_check.py               # 23 号驱动到位后跑：必须全绿
    python scripts/gpu_check.py --falsify     # 证伪：把 GPU 藏起来，这个体检必须变红

## 为什么需要单独的体检脚本

"装上了"和"真的在用 GPU"是两件事。最典型的坑是 **xgboost 静默跑在 CPU 上**：
参数写对了、程序不报错、结果也能出来，就是慢十倍，而你完全看不出来。
（本项目里 conda 装的 xgboost 就是 CPU-only 构建，`USE_CUDA` 为 False。）

所以每一条都要求**实测**，不接受"import 成功"当证据。

## 判定标准

- **PASS**：实测通过
- **FAIL**：该用的没生效（会让整轮体检返回非零）
- **WARN**：预期内的限制，知道就行（例如 LightGBM 的 pip 轮子从来不带 CUDA）
- **SKIP**：当前模式不检查（preflight 模式跳过所有要 GPU 的项）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
COLOR = {PASS: "\033[32m", FAIL: "\033[31m", WARN: "\033[33m", SKIP: "\033[90m"}
RESET = "\033[0m"

rows: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    rows.append((name, status, detail))
    c = COLOR[status]
    print(f"  {c}{status:<4}{RESET} {name:<38} {detail}")


def section(title: str) -> None:
    print(f"\n\033[1m== {title} ==\033[0m")


# ---------------------------------------------------------------------------
def check_driver(preflight: bool) -> dict:
    """读驱动。没驱动不是错——只是说明现在还用不了 GPU。"""
    info = {"present": False, "driver": None, "name": None, "memory": None}
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        )
    except FileNotFoundError:
        record("NVIDIA 驱动", SKIP if preflight else FAIL,
               "nvidia-smi 不存在 —— 驱动还没装")
        return info
    except Exception as e:                                    # noqa: BLE001
        record("NVIDIA 驱动", FAIL, f"nvidia-smi 调用失败：{e}")
        return info

    if q.returncode != 0 or not q.stdout.strip():
        msg = (q.stderr or q.stdout).strip().splitlines()
        record("NVIDIA 驱动", SKIP if preflight else FAIL,
               (msg[0] if msg else "nvidia-smi 无输出")[:80])
        return info

    parts = [p.strip() for p in q.stdout.strip().splitlines()[0].split(",")]
    if len(parts) >= 3:
        info.update(present=True, name=parts[0], driver=parts[1], memory=parts[2])
        major = int(parts[1].split(".")[0])
        status = PASS if major >= 525 else WARN
        note = "" if major >= 525 else "（<525，只能用 cu118）"
        record("NVIDIA 驱动", status, f"{parts[0]} | {parts[1]} | {parts[2]} MB{note}")
    return info


def check_torch(preflight: bool) -> dict:
    info = {"ok": False, "cuda": False}
    try:
        import torch
    except Exception as e:                                    # noqa: BLE001
        record("torch 导入", FAIL, f"{type(e).__name__}: {str(e)[:70]}")
        return info

    info["ok"] = True
    is_cu = bool(torch.version.cuda)
    record("torch 版本", PASS if is_cu else FAIL,
           f"{torch.__version__} | 内置 CUDA {torch.version.cuda or '无（这是 CPU 版！）'}")

    try:
        import torchvision
        tv_ok = torchvision.__version__.split("+")[0].split(".")[:2]
        t_ok = torch.__version__.split("+")[0].split(".")[:2]
        record("torchvision 配对", PASS if tv_ok == t_ok else WARN,
               f"{torchvision.__version__}（torch {torch.__version__}）")
    except Exception as e:                                    # noqa: BLE001
        record("torchvision", WARN, f"{type(e).__name__}: {str(e)[:60]}")

    avail = bool(torch.cuda.is_available())
    info["cuda"] = avail
    if not avail:
        record("CUDA 可用", SKIP if preflight else FAIL,
               "torch.cuda.is_available() = False"
               + ("（驱动未就位，符合预期）" if preflight else " —— 驱动/CUDA 版本不匹配"))
        return info

    i = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(i)
    record("GPU 设备", PASS,
           f"{p.name} | 算力 {p.major}.{p.minor} | 显存 {p.total_memory/2**30:.1f} GB")
    return info


def check_torch_compute(gpu: bool, preflight: bool) -> None:
    """真算一次矩阵乘，并用显存峰值证明**计算确实发生在 GPU 上**。

    只检查 is_available() 不够 —— 有 CUDA 库但 kernel 起不来时它也可能是 True。
    """
    if not gpu:
        record("GPU 实际计算", SKIP, "没有可用 GPU")
        return
    import torch

    n = 4096
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 2**20

    tf = (2 * n ** 3 * 5) / dt / 1e12
    # 显存峰值必须真的涨起来，否则说明数据被搬回 CPU 算了
    ok = peak > 100
    record("GPU 实际计算", PASS if ok else FAIL,
           f"5×{n}³ 矩阵乘 {dt*1000:.0f} ms ≈ {tf:.1f} TFLOP/s，峰值显存 {peak:.0f} MB")


def check_xgboost(gpu: bool, preflight: bool) -> None:
    """xgboost 是最容易"假装有 GPU"的一个：conda 版编译时就没开 CUDA。"""
    try:
        import xgboost as xgb
    except Exception as e:                                    # noqa: BLE001
        record("xgboost 导入", FAIL, f"{type(e).__name__}: {str(e)[:60]}")
        return

    use_cuda = bool(xgb.build_info().get("USE_CUDA", False))
    record("xgboost 编译选项", PASS if use_cuda else FAIL,
           f"v{xgb.__version__} | USE_CUDA={use_cuda}"
           + ("" if use_cuda else " ← 这是 CPU-only 构建，必须 pip 重装"))

    if not use_cuda:
        return
    if not gpu:
        record("xgboost GPU 训练", SKIP, "没有可用 GPU")
        return

    import numpy as np
    X = np.random.default_rng(0).normal(size=(2000, 20)).astype("float32")
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    try:
        m = xgb.XGBClassifier(n_estimators=20, tree_method="hist",
                              device="cuda", max_depth=4)
        m.fit(X, y)
        record("xgboost GPU 训练", PASS, "device='cuda' 跑通")
    except Exception as e:                                    # noqa: BLE001
        record("xgboost GPU 训练", FAIL, f"{type(e).__name__}: {str(e)[:90]}")


def check_catboost(gpu: bool, preflight: bool) -> None:
    try:
        import catboost as cb
    except Exception as e:                                    # noqa: BLE001
        record("catboost 导入", FAIL, f"{type(e).__name__}: {str(e)[:60]}")
        return
    if not gpu:
        record("catboost GPU 训练", SKIP, "没有可用 GPU")
        return

    import numpy as np
    X = np.random.default_rng(0).normal(size=(2000, 20)).astype("float32")
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    try:
        m = cb.CatBoostClassifier(iterations=20, depth=4, task_type="GPU",
                                  devices="0", verbose=0,
                                  allow_writing_files=False)
        m.fit(X, y)
        record("catboost GPU 训练", PASS, "task_type='GPU' 跑通")
    except Exception as e:                                    # noqa: BLE001
        record("catboost GPU 训练", FAIL, f"{type(e).__name__}: {str(e)[:90]}")


def check_lightgbm() -> None:
    """LightGBM 的 pip 轮子**从来没有** CUDA 支持，这里只如实报告，不算失败。

    想用 GPU 版得自己用 CMake `-DUSE_CUDA=1` 编译，需要 CUDA toolkit + 匹配的 gcc。
    赛期只有 7 天，不值得。加速交给 XGBoost / CatBoost。
    """
    try:
        import lightgbm as lgb
    except Exception as e:                                    # noqa: BLE001
        record("lightgbm 导入", WARN, f"{type(e).__name__}: {str(e)[:60]}")
        return
    try:
        lgb.LGBMClassifier(n_estimators=1, device_type="cuda", verbose=-1)
    except Exception as e:                                    # noqa: BLE001
        msg = str(e)
        if "not enabled" in msg or "CUDA" in msg:
            record("lightgbm CUDA", WARN,
                   f"v{lgb.__version__} 无 CUDA —— 预期内，继续用 CPU（本来就快）")
            return
        record("lightgbm CUDA", WARN, msg[:80])
        return
    record("lightgbm CUDA", PASS, f"v{lgb.__version__} 支持 CUDA（少见）")


def check_pytabkit(gpu: bool, preflight: bool) -> None:
    """RealMLP：表格深度模型里目前最能打的一个，也是唯一值得进融合的。"""
    try:
        import pytabkit
    except Exception as e:                                    # noqa: BLE001
        record("pytabkit 导入", FAIL, f"{type(e).__name__}: {str(e)[:70]}")
        return
    record("pytabkit 版本", PASS, getattr(pytabkit, "__version__", "?"))

    import numpy as np
    try:
        from pytabkit import RealMLP_TD_Classifier
    except Exception as e:                                    # noqa: BLE001
        record("RealMLP 导入", FAIL, f"{type(e).__name__}: {str(e)[:70]}")
        return

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 10))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    device = "cuda" if gpu else "cpu"

    import torch
    if gpu:
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    try:
        m = RealMLP_TD_Classifier(n_epochs=3, random_state=0, device=device,
                                  n_threads=4, verbosity=0)
        m.fit(X[:300], y[:300])
        p = m.predict_proba(X[300:])
        dt = time.perf_counter() - t0
    except Exception as e:                                    # noqa: BLE001
        record("RealMLP 训练", FAIL if gpu else WARN,
               f"{type(e).__name__}: {str(e)[:90]}")
        return

    shape_ok = getattr(p, "shape", (0,))[0] == 100
    if gpu:
        peak = torch.cuda.max_memory_allocated() / 2**20
        # 关键判据：显存峰值涨了，才算真的在 GPU 上训
        record("RealMLP 训练", PASS if (shape_ok and peak > 1) else FAIL,
               f"{device} | {dt:.1f}s | 峰值显存 {peak:.0f} MB | 输出 {getattr(p, 'shape', '?')}")
    else:
        record("RealMLP 训练（CPU）", PASS if shape_ok else FAIL,
               f"{dt:.1f}s | 输出 {getattr(p, 'shape', '?')}（GPU 到位后会自动切到 cuda）")


def check_vision(gpu: bool, preflight: bool) -> None:
    """视觉迁移学习：timm 能建模型就算通。**不下载预训练权重**，免得卡在网络上。"""
    try:
        import timm
        import torch
        m = timm.create_model("resnet18", pretrained=False, num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        if gpu:
            m = m.cuda()
            x = x.cuda()
        with torch.no_grad():
            out = m(x)
        record("timm 视觉", PASS,
               f"v{timm.__version__} | resnet18 前向 {tuple(out.shape)}"
               + ("（GPU）" if gpu else "（CPU）"))
    except Exception as e:                                    # noqa: BLE001
        record("timm 视觉", FAIL, f"{type(e).__name__}: {str(e)[:80]}")

    try:
        import cv2
        record("OpenCV", PASS, cv2.__version__)
    except Exception as e:                                    # noqa: BLE001
        record("OpenCV", WARN, f"{type(e).__name__}: {str(e)[:60]}")


def check_text() -> None:
    try:
        import transformers
        record("transformers", PASS, transformers.__version__ + "（未下载模型，省流量）")
    except Exception as e:                                    # noqa: BLE001
        record("transformers", FAIL, f"{type(e).__name__}: {str(e)[:60]}")
        return
    try:
        import sentence_transformers
        record("sentence-transformers", PASS, sentence_transformers.__version__)
    except Exception as e:                                    # noqa: BLE001
        record("sentence-transformers", WARN, f"{type(e).__name__}: {str(e)[:60]}")


# ---------------------------------------------------------------------------
def falsify() -> int:
    """证伪：把 GPU 藏起来再跑一遍，体检**必须**返回非零。

    永远通过的守卫是假守卫。这条规矩抄自 `~/luad_v2/docs/scientific_rigor_and_audit.md`。
    """
    print("=== 证伪：隐藏 GPU 后重跑体检，必须 FAIL ===")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, __file__], env=env,
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("  ✗ 藏掉 GPU 之后体检居然还通过了 —— 这个体检没有意义，得修。")
        return 1
    print("  ✓ 藏掉 GPU 后体检返回非零，守卫是有效的。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU 环境体检")
    ap.add_argument("--preflight", action="store_true",
                    help="没驱动也能跑：只查安装是否完整")
    ap.add_argument("--falsify", action="store_true",
                    help="证伪：隐藏 GPU 后体检必须变红")
    args = ap.parse_args()

    if args.falsify:
        return falsify()

    print("=" * 72)
    print("GPU 环境体检" + ("（预检模式：不要求 GPU 已就位）" if args.preflight else ""))
    print("=" * 72)

    section("硬件与驱动")
    drv = check_driver(args.preflight)

    section("PyTorch")
    t = check_torch(args.preflight)
    gpu = t["cuda"]

    section("实际计算")
    check_torch_compute(gpu, args.preflight)

    section("梯度提升树的 GPU 支持")
    check_xgboost(gpu, args.preflight)
    check_catboost(gpu, args.preflight)
    check_lightgbm()

    section("表格深度模型")
    check_pytabkit(gpu, args.preflight)

    section("视觉 / 文本")
    check_vision(gpu, args.preflight)
    check_text()

    # ---- 汇总 ----
    n_fail = sum(1 for _, s, _ in rows if s == FAIL)
    n_warn = sum(1 for _, s, _ in rows if s == WARN)
    n_skip = sum(1 for _, s, _ in rows if s == SKIP)
    print("\n" + "=" * 72)
    print(f"PASS {sum(1 for _, s, _ in rows if s == PASS)} | "
          f"FAIL {n_fail} | WARN {n_warn} | SKIP {n_skip}")
    if n_fail:
        print("\033[31m有 FAIL —— 按上面的提示处理，别带着它上场比赛。\033[0m")
        return 1
    if n_skip and not args.preflight:
        print("\033[33m有 SKIP —— GPU 还没完全就位。\033[0m")
        return 1
    if args.preflight and not gpu:
        print("\033[33m安装完整。GPU 未就位，23 号驱动上来后再跑一次（不加 --preflight）。\033[0m")
        return 0
    print("\033[32m全绿。可以放心用 GPU。\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
