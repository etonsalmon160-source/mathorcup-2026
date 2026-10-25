#!/usr/bin/env python
# 本文件由 AI 辅助编写；建模与算法设计决策由参赛队员做出。详见《AI工具使用详情说明》。
"""原始数据冻结 CLI。核心逻辑在 src/freeze.py。

用法：
    python scripts/freeze_data.py --init      首次冻结（核验 zip 成员逐字节一致）
    python scripts/freeze_data.py --check     校验（make freeze-check / run_all 第一步）
    python scripts/freeze_data.py --update --reason "组委会重新发布数据"
    python scripts/freeze_data.py --lock      把权威文件设为只读（chmod 444）

补充数据（data/external/，非官方）用 --external-* 子命令，与上面互不影响：
    python scripts/freeze_data.py --external-init
    python scripts/freeze_data.py --external-check
    python scripts/freeze_data.py --external-update --reason "队员补交新版本"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import freeze  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="原始数据冻结")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--init", action="store_true", help="首次生成哈希清单")
    g.add_argument("--check", action="store_true", help="校验清单")
    g.add_argument("--update", action="store_true", help="更新清单（必须配 --reason）")
    g.add_argument("--lock", action="store_true", help="设为只读")
    g.add_argument("--external-init", action="store_true", help="补充数据首次冻结")
    g.add_argument("--external-check", action="store_true", help="补充数据校验")
    g.add_argument("--external-update", action="store_true",
                   help="补充数据更新清单（必须配 --reason）")
    ap.add_argument("--reason", default="", help="更新原因，写进 PROVENANCE.md")
    ap.add_argument("--force", action="store_true", help="--init 已存在清单时强制覆盖")
    args = ap.parse_args()

    if args.external_init:
        if freeze.EXT_MANIFEST_JSON.exists() and not args.force:
            print("🔴 补充数据清单已存在。若要重新冻结，请用")
            print("   --external-update --reason \"……\"；确需覆盖再加 --force。")
            return 1
        freeze.init_external_manifest()
        return 0
    if args.external_update:
        freeze.update_external_manifest(args.reason)
        return 0
    if args.external_check:
        ok, details = freeze.check_external_manifest()
        if ok:
            print(f"✅ 补充数据冻结校验通过：{details[0]}")
            return 0
        print("🔴 补充数据冻结校验失败：")
        for d in details:
            print(f"   - {d}")
        print("\n   若确属队员补交：--external-update --reason \"……\"，并同步更新")
        print("   docs/补充数据预注册.md 的变更记录。")
        return 1

    if args.init:
        if freeze.MANIFEST_JSON.exists() and not args.force:
            print("🔴 清单已存在。若要重新冻结，请用 --update --reason \"……\"；")
            print("   确需覆盖（例如首次冻结漏了文件）再加 --force。")
            return 1
        freeze.init_manifest()
        return 0
    if args.update:
        freeze.update_manifest(args.reason)
        return 0
    if args.lock:
        freeze.lock_files()
        return 0

    ok, details = freeze.check_manifest()
    if ok:
        print(f"✅ 数据冻结校验通过：{details[0]}")
        return 0
    print("🔴 数据冻结校验失败：")
    for d in details:
        print(f"   - {d}")
    print("\n   若确实是官方更新：python scripts/freeze_data.py --update --reason \"……\"")
    print("   若只是误改：从官方 zip 重新解压覆盖，不要改清单。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
