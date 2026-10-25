"""打包上传用的「支撑材料」压缩包（题号+参赛编号+附件）。

    python scripts/pack_submission.py --problem A --team-id 2600001
    python scripts/pack_submission.py --problem A --team-id 2600001 --dry-run

产出 submissions/to_upload/A2600001附件.zip，里面是源码 + 结果数据集 + 运行说明。

**不打包** data/raw（组委会提供，体积大）、models/（可重训）、logs/、.git/。
重复跑结果完全一致（固定时间戳、按路径排序），方便赛后对账。
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "submissions" / "to_upload"

INCLUDE_DIRS = ["src", "scripts"]
INCLUDE_FILES = ["Makefile", "environment.yml", "README.md"]
RESULTS_DIR = ROOT / "results"
SKIP_DIR_NAMES = {"__pycache__", ".ipynb_checkpoints", ".git", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo"}

WARN_MB = 50
FAIL_MB = 200

RUNBOOK = """\
# 运行说明

## 1. 运行环境

- Python 3.12
- 依赖见根目录 `environment.yml`
- 建环境：`conda env create -f environment.yml`
- 若已装 conda 环境，可直接用 `make check` 验证环境是否完整

## 2. 目录结构

```
src/        源代码（数据处理、特征、模型、训练、融合）
scripts/    自检与环境脚本
results/    结果数据集（每个实验一个子目录）
Makefile    一键运行入口
```

## 3. 复现步骤

```bash
make check      # 环境自检：包版本、目录、随机种子
make run-all    # 从原始数据跑到最终结果
```

产物写入 `results/<实验名>/`：`submission.csv`（结果数据集）、
`metrics.json`（交叉验证指标）、`oof_predictions.csv`（袋外预测）。

## 4. 数据

赛题原始数据由组委会提供，放在 `data/raw/`，**未随本附件打包**。
目录结构与文件名见 `src/config.py`。

## 5. 随机性与一致性

全局随机种子固定为 42（`src/config.py`）。敏感模型跑 3 个种子，
报告中给出「均值 ± 标准差」。
"""


def _walk(base: Path):
    for p in sorted(base.rglob("*")):
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIR_NAMES for part in rel.parts):
            continue
        if p.is_file() and p.suffix not in SKIP_SUFFIXES:
            yield p


def collect() -> list[Path]:
    files: list[Path] = []
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if base.is_dir():
            files.extend(_walk(base))
    files.extend(ROOT / f for f in INCLUDE_FILES if (ROOT / f).is_file())
    if RESULTS_DIR.is_dir():
        files.extend(p for p in _walk(RESULTS_DIR) if p.suffix in {".csv", ".json"})
    return sorted(set(files))


def main() -> None:
    ap = argparse.ArgumentParser(description="打包支撑材料（题号+参赛编号+附件）")
    ap.add_argument("--problem", required=True, choices=["A", "B"], help="题目编号 A 或 B")
    ap.add_argument("--team-id", required=True, help="参赛编号，如 2600001")
    ap.add_argument("--dry-run", action="store_true", help="只列出文件，不写 zip")
    args = ap.parse_args()

    if not args.team_id.isalnum():
        raise SystemExit("参赛编号只能是字母数字，检查一下有没有多打空格或连字符")

    out = OUT_DIR / f"{args.problem}{args.team_id}附件.zip"
    files = collect()
    if not files:
        raise SystemExit("没找到要打包的文件，确认在仓库根目录下运行")

    total = sum(f.stat().st_size for f in files)
    size_mb = total / 1024 / 1024

    print(f"将打包 {len(files)} 个文件，共 {size_mb:.1f} MB")
    biggest = sorted(files, key=lambda p: p.stat().st_size, reverse=True)[:5]
    for f in biggest:
        print(f"  {f.stat().st_size / 1024 / 1024:7.2f} MB  {f.relative_to(ROOT)}")

    if size_mb > FAIL_MB:
        raise SystemExit(
            f"压缩包 {size_mb:.0f} MB 过大（上限按 {FAIL_MB} MB 估），"
            "把大的结果文件筛一筛，或只留最终版本"
        )
    if size_mb > WARN_MB:
        print(f"  ! 超过 {WARN_MB} MB，上传可能很慢，注意留时间")

    if args.dry_run:
        for f in files:
            print(f"  {f.relative_to(ROOT)}")
        print("\n（--dry-run，没有写文件）")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            info = zipfile.ZipInfo(str(f.relative_to(ROOT)), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, f.read_bytes())
        info = zipfile.ZipInfo("运行说明.md", date_time=(2026, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        zf.writestr(info, RUNBOOK.encode("utf-8"))

    print(f"\n压缩包 -> {out.relative_to(ROOT)}  ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    print("\n提交前还要做的三件事：")
    print("  1. 论文 PDF 命名为 <题号><参赛编号>.pdf（如 A2600001.pdf）")
    print("  2. 承诺书每位队员签名，命名 <题号><参赛编号>承诺书")
    print("  3. 先在 submissions/REGISTRY.md 登记，再上传")


if __name__ == "__main__":
    sys.exit(main())
