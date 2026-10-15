#!/usr/bin/env python
"""自动过门检查 —— `make gate`。

七道门的证据全部自动判 PASS / FAIL / 未开始。比赛里只该在 FAIL 时人工介入。
见 docs/门控系统.md。

每道门检查的是**某个产物文件存在且数值在护栏内**，所以它足够快（秒级），
可以在任何时候跑，不占实验时间。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL, TODO = "PASS", "FAIL", "未开始"

# 护栏默认值（预注册里可覆盖，这里只做兜底）
ADVERSARIAL_AUC_MAX = 0.60
MULTISEED_STD_MAX = 0.02
ABLATION_HARM_MAX = 0.002    # 砍掉某组后主指标上升超过这么多 ⇒ 该组是噪音，必须下线


def _experiments() -> list[Path]:
    """results/ 下的实验目录（排除 _ 开头的自检产物）。"""
    d = ROOT / "results"
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_dir() and not p.name.startswith("_"))


def _has_placeholder(path: Path) -> bool:
    """文档里还留着没填的空，就算没写完。"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return True
    for marker in ("____", "〔", "〕"):
        if marker in text:
            return True
    return False


# ---------------- 各道门 ----------------

def g0_prereg() -> tuple[str, str]:
    p = ROOT / "docs" / "预注册.md"
    if not p.exists():
        return TODO, "缺 docs/预注册.md（开赛前必须写）"
    if _has_placeholder(p):
        return TODO, "docs/预注册.md 还有没填的空"
    return PASS, "预注册已定稿"


def g1_problem() -> tuple[str, str]:
    p = ROOT / "docs" / "问题理解.md"
    if not p.exists():
        return TODO, "缺 docs/问题理解.md"
    if _has_placeholder(p):
        return TODO, "docs/问题理解.md 还有没填的空"
    return PASS, "问题理解、指标、验证方案已定稿"


def g2_end_to_end() -> tuple[str, str]:
    exps = _experiments()
    subs = [e / "submission.csv" for e in exps if (e / "submission.csv").exists()]
    if not subs:
        return TODO, "还没有任何 submission.csv（跑 make run-all）"
    bad = []
    for s in subs:
        try:
            with open(s, encoding="utf-8-sig") as f:
                rows = list(csv.reader(f))
        except OSError as e:
            bad.append(f"{s.parent.name}: 读不了 ({e})")
            continue
        if len(rows) < 2:
            bad.append(f"{s.parent.name}: 只有表头没有数据")
        elif len(set(len(r) for r in rows)) > 1:
            bad.append(f"{s.parent.name}: 各行列数不一致")
    if bad:
        return FAIL, "；".join(bad)
    return PASS, f"{len(subs)} 份提交文件格式正常（行数、列数一致）"


def g3_leakage() -> tuple[str, str]:
    exps = _experiments()
    files = [e / "adversarial.json" for e in exps if (e / "adversarial.json").exists()]
    if not files:
        return TODO, "还没跑对抗验证（python -m src.validate --config ...）"
    worst = []
    for f in files:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            auc = float(r["auc"])
        except (OSError, KeyError, ValueError) as e:
            worst.append(f"{f.parent.name}: 结果读不了 ({e})")
            continue
        if auc >= ADVERSARIAL_AUC_MAX:
            worst.append(f"{f.parent.name}: AUC={auc:.3f} ≥ {ADVERSARIAL_AUC_MAX} ⇒ 训练/测试不同分布")
    if worst:
        return FAIL, "；".join(worst)
    return PASS, f"{len(files)} 份对抗验证均通过（AUC < {ADVERSARIAL_AUC_MAX}）"


def g4_ablation() -> tuple[str, str]:
    files = []
    for e in _experiments():
        f = e / "ablation.csv"
        if f.exists():
            files.append(f)
    if not files:
        return TODO, "还没有消融表（python -m src.ablate --config ...）"
    problems = []
    for f in files:
        with open(f, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                try:
                    drop = float(row["与完整模型的差"])
                except (KeyError, ValueError):
                    continue
                # 正数 = 砍掉后变好 = 这组是噪音
                if drop > ABLATION_HARM_MAX:
                    problems.append(f"{f.parent.name}/{row.get('特征组', '?')}: 砍掉后 +{drop:.4f}")
    if problems:
        return FAIL, "这些特征组是噪音，该下线：" + "；".join(problems)
    return PASS, f"{len(files)} 份消融表均无负贡献特征组"


def g5_model() -> tuple[str, str]:
    exps = _experiments()
    metas = [e / "metrics.json" for e in exps if (e / "metrics.json").exists()]
    if not metas:
        return TODO, "还没训练结果（make train）"
    problems, note = [], []
    for m in metas:
        r = json.loads(m.read_text(encoding="utf-8"))
        pm = r["primary_metric"]
        n_seeds = len(r.get("seeds") or [])
        std = float(r["cv"].get(f"{pm}_std", 0.0))
        if n_seeds < 3:
            note.append(f"{m.parent.name}: 只跑了 {n_seeds} 个种子（要报均值±标准差得 ≥3）")
        elif std > MULTISEED_STD_MAX:
            problems.append(f"{m.parent.name}: 多种子标准差 {std:.4f} > {MULTISEED_STD_MAX}（不稳定）")

    fusions = [e / "fusion.json" for e in exps if (e / "fusion.json").exists()]
    for f in fusions:
        r = json.loads(f.read_text(encoding="utf-8"))
        if not r.get("fused_beats_best_single", False):
            problems.append(f"{f.parent.name}: 融合没赢过最好单模型 ⇒ 应退回单模型")

    if problems:
        return FAIL, "；".join(problems)
    if note:
        return TODO, "；".join(note)
    extra = f"，{len(fusions)} 份融合对比已过" if fusions else "（还没做融合对比）"
    return PASS, f"{len(metas)} 份训练结果稳定{extra}"


def g6_paper() -> tuple[str, str]:
    p = ROOT / "paper" / "main.pdf"
    if not p.exists():
        return TODO, "还没编出 paper/main.pdf（make paper）"
    if _has_placeholder(ROOT / "paper" / "main.tex"):
        return TODO, "paper/main.tex 里还有没填的〔〕"
    return PASS, f"论文已成稿（{p.stat().st_size // 1024} KB）"


def g7_deliver() -> tuple[str, str]:
    arch = ROOT / "submissions" / "archive"
    zips = sorted(arch.glob("*.zip")) if arch.exists() else []
    if not zips:
        return TODO, "submissions/archive/ 里还没有归档的提交物"
    return PASS, f"已归档 {len(zips)} 份提交物（最新 {zips[-1].name}）"


GATES = [
    ("G0", "预注册", g0_prereg),
    ("G1", "问题定义", g1_problem),
    ("G2", "端到端贯通", g2_end_to_end),
    ("G3", "数据可信", g3_leakage),
    ("G4", "特征消融", g4_ablation),
    ("G5", "模型与融合", g5_model),
    ("G6", "论文成稿", g6_paper),
    ("G7", "交付", g7_deliver),
]

ICON = {PASS: "✓", FAIL: "✗", TODO: "·"}


def main() -> int:
    results = [(code, name, *fn()) for code, name, fn in GATES]

    print("\n================== 过门检查 ==================")
    for code, name, status, detail in results:
        print(f" {ICON[status]} {code} {name:<12s} [{status}] {detail}")

    n_pass = sum(1 for *_, s, _ in results if s == PASS)
    n_fail = sum(1 for *_, s, _ in results if s == FAIL)
    print("----------------------------------------------")
    print(f" 通过 {n_pass} / 失败 {n_fail} / 未开始 {len(results) - n_pass - n_fail}")

    if n_fail:
        print(" ⇒ 有门没过，按 docs/门控系统.md 先停下来处理，别往下推。")
    print("==============================================\n")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
