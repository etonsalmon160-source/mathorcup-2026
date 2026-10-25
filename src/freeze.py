# 本文件由 AI 辅助编写；建模与算法设计决策由参赛队员做出。详见《AI工具使用详情说明》。
"""原始数据冻结：哈希清单 + 来源记录 + 官方 zip 锚定校验。

为什么要锚定 zip：
    只比对清单和文件本身，等于让"改文件 + 重算清单"绕过检查。
    这里把每个解压文件都回溯到官方 zip 的成员内容，改一个字都会被抓到。

权威副本：
    data/raw/附件/            与 zip 成员逐字节一致，纳入清单
    data/raw/*.pdf,*.docx     官方单独发布的文件，纳入清单
    data/raw/track_A/         已核实是 附件/ 的完整副本，**不纳入**清单
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFEST_JSON = RAW / "MANIFEST.json"
MANIFEST_SHA = RAW / "MANIFEST.sha256"
PROVENANCE = RAW / "PROVENANCE.md"

OFFICIAL_GLOBS = ["*.zip", "*.pdf", "*.docx"]
EXTRACTED_DIR = "附件"

# 不纳入清单的目录（非权威副本）
EXCLUDE_DIRS = {"track_A", "__pycache__"}


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def official_zip() -> Path:
    zips = sorted(RAW.glob("*.zip"))
    if len(zips) != 1:
        raise SystemExit(
            f"data/raw 下应有且仅有一个官方 zip，实际 {len(zips)} 个："
            f"{[z.name for z in zips]}"
        )
    return zips[0]


def canonical_files() -> list[Path]:
    """纳入清单的权威文件（相对路径由 _rel 给出）。"""
    files: list[Path] = []
    for pattern in OFFICIAL_GLOBS:
        files += sorted(RAW.glob(pattern))
    extracted = RAW / EXTRACTED_DIR
    if extracted.exists():
        files += sorted(p for p in extracted.rglob("*") if p.is_file())
    return [p for p in files if not any(part in EXCLUDE_DIRS for part in p.parts)]


def _rel(p: Path) -> str:
    return p.relative_to(RAW).as_posix()


def verify_zip_members() -> dict[str, str]:
    """zip 内每个文件都必须与磁盘上的同名文件逐字节一致。返回 {成员路径: sha256}。"""
    z = official_zip()
    members: dict[str, str] = {}
    with zipfile.ZipFile(z) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member = info.filename
            if not (info.flag_bits & 0x800):
                try:
                    member = member.encode('cp437').decode('gbk')
                except Exception:
                    pass
            data = zf.read(info.filename)
            disk = RAW / member
            if not disk.exists():
                raise SystemExit(f"🔴 zip 成员在磁盘上缺失：{member}（应存在于 {disk}）")
            got = sha256_file(disk)
            want = _sha256_bytes(data)
            if got != want:
                raise SystemExit(
                    f"🔴 {member} 与官方 zip 内成员不一致——文件被改过。\n"
                    f"    磁盘 {got}\n    zip  {want}"
                )
            members[member] = want
    return members


def build_manifest() -> dict:
    members = verify_zip_members()
    z = official_zip()
    files = []
    for p in canonical_files():
        rel = _rel(p)
        files.append({
            "path": rel,
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
            "source": f"zip:{rel}" if rel in members else "official-download",
        })
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "official_zip": {"path": _rel(z), "sha256": sha256_file(z), "bytes": z.stat().st_size},
        "extracted_dir": EXTRACTED_DIR,
        "excluded_note": "track_A/ 为 附件/ 的完整副本，非权威，不纳入清单",
        "files": sorted(files, key=lambda r: r["path"]),
    }


def _write_manifest(manifest: dict) -> None:
    MANIFEST_JSON.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [f"{f['sha256']}  {f['path']}" for f in manifest["files"]]
    MANIFEST_SHA.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_provenance(manifest: dict, reason: str, append: bool) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    row = f"| {ts} | {'更新' if append else '首次冻结'} | {reason or '—'} |"
    header = (
        "# 原始数据来源与冻结记录\n\n"
        "- 数据来源：MathorCup 2026 大数据竞赛 赛道 A 官方发布\n"
        "- 获取方式：队员从官方渠道下载后上传至服务器（2026-10-23）\n"
        f"- 官方压缩包：`{manifest['official_zip']['path']}`\n"
        f"- zip SHA-256：`{manifest['official_zip']['sha256']}`\n"
        "- 权威副本：`data/raw/附件/`（与 zip 成员逐字节一致）\n"
        "- 非权威副本：`data/raw/track_A/`（已核对与 `附件/` 一致，不纳入清单）\n"
        "- 冻结工具：`scripts/freeze_data.py`（清单见 `MANIFEST.json` / `MANIFEST.sha256`）\n\n"
        "## 变更记录\n\n"
        "| 时间 | 动作 | 说明 |\n|---|---|---|\n"
    )
    if append and PROVENANCE.exists():
        text = PROVENANCE.read_text(encoding="utf-8")
        if not text.endswith("\n"):
            text += "\n"
        PROVENANCE.write_text(text + row + "\n", encoding="utf-8")
    else:
        PROVENANCE.write_text(header + row + "\n", encoding="utf-8")


def init_manifest(reason: str = "首次冻结（zip 成员逐字节核验通过）") -> None:
    manifest = build_manifest()
    _write_manifest(manifest)
    _write_provenance(manifest, reason, append=False)
    print(f"✅ 已冻结 {len(manifest['files'])} 个文件")
    print(f"   清单：{MANIFEST_JSON.relative_to(ROOT)} / {MANIFEST_SHA.relative_to(ROOT)}")
    print(f"   来源：{PROVENANCE.relative_to(ROOT)}")


def update_manifest(reason: str) -> None:
    if not reason or reason.strip() in ("", "—"):
        raise SystemExit("🔴 --update 必须用 --reason 写清为什么更新（会留下变更记录）")
    init_manifest(reason=reason.strip())


def check_manifest() -> tuple[bool, list[str]]:
    """返回 (是否通过, 明细)。任何不一致都算失败。"""
    problems: list[str] = []
    if not MANIFEST_JSON.exists():
        return False, ["缺少 data/raw/MANIFEST.json —— 先跑 `make freeze`"]

    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    expected = {f["path"]: f["sha256"] for f in manifest["files"]}

    try:
        verify_zip_members()
    except SystemExit as e:
        problems.append(str(e))

    actual_paths = {_rel(p) for p in canonical_files()}
    missing = sorted(set(expected) - actual_paths)
    extra = sorted(actual_paths - set(expected))
    if missing:
        problems.append(f"缺少文件：{missing}")
    if extra:
        problems.append(f"多出未登记文件：{extra}")

    for rel, want in sorted(expected.items()):
        p = RAW / rel
        if not p.exists():
            continue
        got = sha256_file(p)
        if got != want:
            problems.append(f"哈希不符：{rel}\n    清单 {want}\n    实际 {got}")

    if problems:
        return False, problems
    return True, [f"{len(expected)} 个文件哈希全部一致，zip 成员核验通过"]


def lock_files() -> None:
    """把权威文件设为只读（444）。不是安全边界，只是防止 Excel 悄悄覆盖。"""
    n = 0
    for p in canonical_files():
        p.chmod(0o444)
        n += 1
    print(f"🔒 已设为只读：{n} 个文件（chmod 444）")


# ---------------------------------------------------------------------------
# 补充数据冻结（data/external/）
#
# 与官方数据分开管理：这些文件**不是**赛题附件，不允许混进 data/raw 的清单，
# 也不允许用官方 zip 锚定逻辑校验。用途分级见 docs/补充数据预注册.md。
# ---------------------------------------------------------------------------

EXTERNAL = ROOT / "data" / "external"
EXT_MANIFEST_JSON = EXTERNAL / "MANIFEST.json"
EXT_MANIFEST_SHA = EXTERNAL / "MANIFEST.sha256"
EXT_PROVENANCE = EXTERNAL / "PROVENANCE.md"
EXT_NON_DATA = {"MANIFEST.json", "MANIFEST.sha256", "PROVENANCE.md", "README.md"}


def external_files() -> list[Path]:
    """data/external 下纳入清单的文件（排除清单自身与隐藏文件）。"""
    if not EXTERNAL.exists():
        return []
    return sorted(
        p for p in EXTERNAL.rglob("*")
        if p.is_file() and p.name not in EXT_NON_DATA and not p.name.startswith(".")
    )


def _ext_rel(p: Path) -> str:
    return p.relative_to(EXTERNAL).as_posix()


def external_zip() -> Path | None:
    zips = sorted(EXTERNAL.glob("*.zip"))
    if len(zips) > 1:
        raise SystemExit(
            f"data/external 下有多个 zip，无法锚定：{[z.name for z in zips]}"
        )
    return zips[0] if zips else None


def verify_external_zip_members() -> dict[str, str]:
    """补充 zip 的成员也必须与磁盘上的同名文件逐字节一致。"""
    z = external_zip()
    if z is None:
        return {}
    members: dict[str, str] = {}
    with zipfile.ZipFile(z) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            data = zf.read(info.filename)
            disk = EXTERNAL / info.filename
            if not disk.exists():
                raise SystemExit(f"🔴 补充 zip 成员在磁盘上缺失：{info.filename}")
            got = sha256_file(disk)
            want = _sha256_bytes(data)
            if got != want:
                raise SystemExit(
                    f"🔴 {info.filename} 与补充 zip 内成员不一致——文件被改过。\n"
                    f"    磁盘 {got}\n    zip  {want}"
                )
            members[info.filename] = want
    return members


def build_external_manifest() -> dict:
    members = verify_external_zip_members()
    files = []
    for p in external_files():
        rel = _ext_rel(p)
        files.append({
            "path": rel,
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
            "source": f"zip:{rel}" if rel in members else "member-provided",
        })
    z = external_zip()
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "external-supplementary",
        "official": False,
        "policy": "docs/补充数据预注册.md",
        "note": "非赛题官方附件；允许用途与判据见 policy 文件",
        "zip": ({"path": _ext_rel(z), "sha256": sha256_file(z), "bytes": z.stat().st_size}
                if z else None),
        "zip_members": sorted(members),
        "files": sorted(files, key=lambda r: r["path"]),
    }


def _write_external_manifest(manifest: dict) -> None:
    EXT_MANIFEST_JSON.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [f"{f['sha256']}  {f['path']}" for f in manifest["files"]]
    EXT_MANIFEST_SHA.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_external_provenance(manifest: dict, reason: str, append: bool) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    row = f"| {ts} | {'更新' if append else '首次冻结'} | {reason or '—'} |"
    header = (
        "# 补充数据来源与冻结记录\n\n"
        "- 性质：**非赛题官方数据**，为外部补充材料\n"
        "- 使用政策：`docs/补充数据预注册.md`（含允许/条件允许/禁止三类用途）\n"
        f"- 压缩包：`{manifest['zip']['path']}`\n"
        f"- zip SHA-256：`{manifest['zip']['sha256']}`\n"
        "- 冻结工具：`scripts/freeze_data.py --external-check`\n"
        "- 注意：本清单与 `data/raw/MANIFEST.json` 相互独立，不得合并\n\n"
        "## 变更记录\n\n"
        "| 时间 | 动作 | 说明 |\n|---|---|---|\n"
    )
    if append and EXT_PROVENANCE.exists():
        text = EXT_PROVENANCE.read_text(encoding="utf-8")
        if not text.endswith("\n"):
            text += "\n"
        EXT_PROVENANCE.write_text(text + row + "\n", encoding="utf-8")
    else:
        EXT_PROVENANCE.write_text(header + row + "\n", encoding="utf-8")


def init_external_manifest(reason: str = "首次冻结（补充 zip 成员逐字节核验通过）") -> None:
    manifest = build_external_manifest()
    _write_external_manifest(manifest)
    _write_external_provenance(manifest, reason, append=False)
    print(f"✅ 已冻结补充数据 {len(manifest['files'])} 个文件")
    print(f"   清单：{EXT_MANIFEST_JSON.relative_to(ROOT)} / {EXT_MANIFEST_SHA.relative_to(ROOT)}")
    print(f"   来源：{EXT_PROVENANCE.relative_to(ROOT)}")


def update_external_manifest(reason: str) -> None:
    if not reason or reason.strip() in ("", "—"):
        raise SystemExit("🔴 --update 必须用 --reason 写清为什么更新（会留下变更记录）")
    init_external_manifest(reason=reason.strip())


def check_external_manifest() -> tuple[bool, list[str]]:
    problems: list[str] = []
    if not EXT_MANIFEST_JSON.exists():
        return False, ["缺少 data/external/MANIFEST.json —— 先跑 `make freeze-external`"]

    manifest = json.loads(EXT_MANIFEST_JSON.read_text(encoding="utf-8"))
    expected = {f["path"]: f["sha256"] for f in manifest["files"]}

    try:
        verify_external_zip_members()
    except SystemExit as e:
        problems.append(str(e))

    actual_paths = {_ext_rel(p) for p in external_files()}
    missing = sorted(set(expected) - actual_paths)
    extra = sorted(actual_paths - set(expected))
    if missing:
        problems.append(f"缺少文件：{missing}")
    if extra:
        problems.append(f"多出未登记文件：{extra}（先确认用途，再 --external-update 登记）")

    for rel, want in sorted(expected.items()):
        p = EXTERNAL / rel
        if not p.exists():
            continue
        got = sha256_file(p)
        if got != want:
            problems.append(f"哈希不符：{rel}\n    清单 {want}\n    实际 {got}")

    if problems:
        return False, problems
    return True, [f"{len(expected)} 个补充文件哈希全部一致，zip 成员核验通过"]
