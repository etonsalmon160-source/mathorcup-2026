#!/usr/bin/env python
"""自动过门检查 —— `make gate`。

七道门的证据全部自动判 PASS / FAIL / 未开始。比赛里只该在 FAIL 时人工介入。
见 docs/门控系统.md。

每道门检查的是**某个产物文件存在且数值在护栏内**，所以它足够快（秒级），
可以在任何时候跑，不占实验时间。
"""
from __future__ import annotations

import csv
import re
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
Q2_JACCARD_MIN = 0.5         # 问题二 Top-3 跨重采样中位 Jaccard 下限
Q2_STABLE_RATE = 0.6         # 单维稳定率低于此值必须在论文里标注
Q3_MIN_STYLES = 22           # 问题三至少覆盖的风格数（内部数据共 22 类）


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_columns() -> list[str]:
    """4.csv 里除 3 个元数据列之外的 50 个感官维度。"""
    p = ROOT / "data" / "raw" / "附件" / "4.csv"
    if not p.exists():
        return []
    with open(p, encoding="utf-8-sig") as f:
        header = next(csv.reader(f), [])
    return [c for c in header if c not in ("beer", "beer_id", "tasting_category_fine")]


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


# ---------------- 数据层（先于 G0） ----------------

def r_rules() -> tuple[str, str]:
    """规则与 AI 合规是常驻门：任何阶段都不许失守。"""
    rules = ROOT / "docs" / "赛制与学术风险清单.md"
    log = ROOT / "docs" / "决策记录.md"
    problems = []
    if not rules.exists() or _has_placeholder(rules):
        problems.append("docs/赛制与学术风险清单.md 缺失或没写完")
    if not log.exists():
        problems.append("docs/决策记录.md 缺失（AI 使用说明的素材来源）")
    else:
        n = len(re.findall(r"^### 2026-", log.read_text(encoding="utf-8"), re.M))
        if n == 0:
            problems.append("docs/决策记录.md 还没有一条真实决策记录")
    if problems:
        return FAIL, "；".join(problems)
    return PASS, "规则清单在位，决策记录有据可查（论文阶段还需正文声明 + 代码注释）"


def f_freeze() -> tuple[str, str]:
    from src import freeze
    ok, details = freeze.check_manifest()
    if ok:
        return PASS, details[0]
    return FAIL, "；".join(details[:2])


def s_supplementary() -> tuple[str, str]:
    """补充数据：先预注册、再冻结；用途分级没签完不许用。"""
    from src import freeze
    policy = ROOT / "docs" / "补充数据预注册.md"
    if not policy.exists():
        return TODO, "缺 docs/补充数据预注册.md"
    ok, details = freeze.check_external_manifest()
    if not ok:
        return FAIL, "；".join(details[:2])
    if "〔" in policy.read_text(encoding="utf-8") or "〕" in policy.read_text(encoding="utf-8"):
        return TODO, "补充数据预注册还有没填的空"
    if "____" in policy.read_text(encoding="utf-8"):
        return TODO, "预注册已写、已冻结；待队员签署 A/B/C 三档用途"
    import json
    report = ROOT / "results" / "supplementary" / "report.json"
    if not report.exists():
        return TODO, "用途 B 已登记但尚未执行 results/supplementary/report.json"
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
        passed = payload.get("s_gate", {}).get("status") == "registered/pass"
    except (OSError, ValueError):
        return FAIL, "results/supplementary/report.json 读不了"
    if not passed:
        return FAIL, "S 门补充分析未标记 registered/pass"
    if "待签" in policy.read_text(encoding="utf-8"):
        return PASS, "用途 B 已执行（仅注释/验证）；B/C 签名待 G7 前补"
    return PASS, "补充数据已预注册、已冻结，用途分级已签署"


def d_data_gate() -> tuple[str, str]:
    from src.data_gate import evaluate
    try:
        gates, _, _ = evaluate()
    except SystemExit as e:
        return FAIL, str(e)
    hard_fail = [g for g in gates if g.hard and not g.passed]
    if hard_fail:
        return FAIL, "；".join(f"{g.code} {g.summary}" for g in hard_fail)
    return PASS, f"D0–D6 全过（{len(gates)} 项）"


def q_qc() -> tuple[str, str]:
    """QC 必须在最后一次冻结之后跑；FAIL 层必须处理，WARN 层要进论文的限制说明。"""
    rep = ROOT / "results" / "qc" / "report.json"
    if not rep.exists():
        return TODO, "还没跑质控（make qc）"
    manifests = [
        ROOT / "data" / "raw" / "MANIFEST.json",
        ROOT / "data" / "external" / "MANIFEST.json",
    ]
    stale = [m.parent.name for m in manifests if m.exists() and rep.stat().st_mtime < m.stat().st_mtime]
    if stale:
        return FAIL, f"QC 早于冻结清单（{stale}）——冻结后必须重跑 make qc"

    r = json.loads(rep.read_text(encoding="utf-8"))
    layers = r.get("layers", {})
    if not layers:
        return FAIL, "report.json 没有分层结果，QC 不完整"
    failed = [k for k, v in layers.items() if str(v.get("status", "")).upper() == "FAIL"]
    if failed:
        return FAIL, f"质控有 FAIL 层：{failed}"
    floor = layers.get("feature", {}).get("floor") or {}
    if "floor_frac_all" not in floor:
        return FAIL, "缺少检出限层（floor）——QC 版本过旧，重跑 make qc"

    n_warn = sum(1 for v in layers.values() if str(v.get("status", "")).upper() == "WARN")
    return PASS, (
        f"{len(layers)} 层通过（{n_warn} 层 WARN 需写进论文限制）；"
        f"检出限 {floor['floor_frac_all']:.1%}；晚于冻结清单"
    )


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
    """Track A 没有官方提交文件；端到端 = Q1 预测表 + Q3 生成表都能落盘且形状正确。"""
    pred = ROOT / "results" / "q1" / "predictions.csv"
    gen = ROOT / "results" / "q3" / "generated.csv"
    if not pred.exists():
        return TODO, "还没有 results/q1/predictions.csv（Q1 端到端未贯通）"
    if not gen.exists():
        return TODO, "还没有 results/q3/generated.csv（Q3 端到端未贯通）"

    problems = []
    with open(pred, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 251:
        problems.append(f"predictions.csv 只有 {max(len(rows) - 1, 0)} 行数据，应为 250 行")
    if rows and len(set(len(r) for r in rows)) > 1:
        problems.append("predictions.csv 各行列数不一致")
    header = rows[0] if rows else []
    targets = _target_columns()
    missing = [c for c in targets if c not in header]
    if missing:
        problems.append(f"predictions.csv 缺 {len(missing)} 个感官列（如 {missing[:3]}）")
    bad = sum(1 for r in rows[1:] for v in r if v.strip() == "" or v.strip().lower() in {"nan", "na"})
    if bad:
        problems.append(f"predictions.csv 有 {bad} 个空/NaN 值")

    with open(gen, encoding="utf-8-sig") as f:
        grows = list(csv.reader(f))
    if len(grows) < 2:
        problems.append("generated.csv 只有表头没有数据")
    elif len(set(len(r) for r in grows)) > 1:
        problems.append("generated.csv 各行列数不一致")

    if problems:
        return FAIL, "；".join(problems)
    return PASS, f"Q1 预测表 250×{len(targets)} 无 NaN；Q3 生成表 {len(grows) - 1} 行"


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
                if row.get("参与判定") == "否":      # 对照基线，不是可砍掉的组
                    continue
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


def p1_q1() -> tuple[str, str]:
    """问题一交付：50 维预测 + 逐维指标 + 3 种子 + 0–6 校准。"""
    m = ROOT / "results" / "q1" / "metrics.json"
    cal = ROOT / "results" / "q1" / "calibration.json"
    if not m.exists():
        return TODO, "缺 results/q1/metrics.json（问题一还没定稿）"
    if not cal.exists():
        return TODO, "缺 results/q1/calibration.json（0–6 分级校准未做）"
    r = _json(m)
    per = r.get("per_target") or {}
    if len(per) < 50:
        return FAIL, f"per_target 只有 {len(per)} 维，应为 50 维"
    if not r.get("primary_metric"):
        return FAIL, "metrics.json 没有 primary_metric"
    cv = r.get("cv") or {}
    mean = cv.get("mean", cv.get("primary_mean"))
    if mean is None or not isinstance(mean, (int, float)):
        return FAIL, "metrics.json 的 cv 里没有可用的主指标均值"
    n_seeds = len(r.get("seeds") or [])
    note = f"3 种子" if n_seeds >= 3 else f"仅 {n_seeds} 个种子（G5 会拦）"
    fus = ROOT / "results" / "composite" / "fusion.json"
    if fus.exists():
        f = _json(fus)
        fused = f.get("fused") or {}
        if f.get("adopted_variant") == "avg" and {"random", "group"} <= set(fused):
            return PASS, (
                f"50 维齐全；主模型 avg 随机 {fused['random']:.4f} / "
                f"分组 {fused['group']:.4f}，ET 单模型 {float(mean):.4f}，{note}"
            )
    return PASS, f"50 维齐全，主指标均值 {float(mean):.4f}，{note}"


def p2_q2() -> tuple[str, str]:
    """问题二交付：网络、模块、50 维 Top-3、稳定性。"""
    paths = {
        "图": ROOT / "results" / "q2" / "graph_edges.csv",
        "模块": ROOT / "results" / "q2" / "communities.csv",
        "Top-3": ROOT / "results" / "q2" / "top3.csv",
        "稳定性": ROOT / "results" / "q2" / "stability.json",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        return TODO, "缺 " + "、".join(missing) + "（问题二还没定稿）"

    with open(paths["Top-3"], encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 50:
        return FAIL, f"top3.csv 只有 {len(rows)} 行，应为 50 个感官维度"
    bad_rows = []
    for r in rows:
        picks = [r.get(k, "").strip() for k in ("top1", "top2", "top3")]
        if not all(picks):
            bad_rows.append(r.get("sensory", "?"))
    if bad_rows:
        return FAIL, f"这些维度缺 Top-3：{bad_rows[:5]}"

    st = _json(paths["稳定性"])
    jac = st.get("median_jaccard")
    if jac is None:
        return FAIL, "stability.json 缺 median_jaccard"
    if float(jac) < Q2_JACCARD_MIN:
        return FAIL, f"Top-3 中位 Jaccard {float(jac):.2f} < {Q2_JACCARD_MIN}"
    unstable = int(st.get("n_unstable", 0))
    return PASS, f"50 维 Top-3 齐全；中位 Jaccard {float(jac):.2f}；{unstable} 维稳定率 <{Q2_STABLE_RATE} 需标注"


def p3_q3() -> tuple[str, str]:
    """问题三交付：生成谱、迁移表、约束可行、覆盖足够风格、有基线对比。"""
    rep = ROOT / "results" / "q3" / "report.json"
    gen = ROOT / "results" / "q3" / "generated.csv"
    tr = ROOT / "results" / "q3" / "transfer.csv"
    if not rep.exists():
        return TODO, "缺 results/q3/report.json（问题三还没定稿；探针结果在 results/q3_probe/）"
    if not gen.exists() or not tr.exists():
        return TODO, "缺 results/q3/generated.csv 或 transfer.csv"

    r = _json(rep)
    verdict = r.get("verdict") or {}
    if not verdict.get("loose_solvable"):
        return FAIL, "松口径下迁移不可行或仍有新增/加重超标"
    missing = [k for k in ("baselines", "defect_compliance", "style_coverage") if k not in r]
    if missing:
        return FAIL, "report.json 缺 " + "、".join(missing) + "（不能只报结果不报对照）"

    coverage = int(r.get("style_coverage") or 0)
    if coverage < Q3_MIN_STYLES:
        return FAIL, f"只覆盖 {coverage} 个风格，至少要 {Q3_MIN_STYLES} 个（或显式声明子集）"

    with open(gen, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < coverage:
        return FAIL, "generated.csv 行数少于声明的风格覆盖数"
    return PASS, f"松口径可行；覆盖 {coverage} 个风格；生成 {len(rows)} 行；有基线对照"


def u_robustness() -> tuple[str, str]:
    """稳健性：3 种子、k 敏感性、约束松紧 Pareto、缺失压力测试。"""
    p = ROOT / "results" / "robustness" / "summary.json"
    if not p.exists():
        return TODO, "缺 results/robustness/summary.json（稳健性检查未做）"
    r = _json(p)
    seeds = r.get("seeds") or []
    if len(seeds) < 3:
        return FAIL, f"只跑了 {len(seeds)} 个种子"
    std = float(r.get("max_seed_std", 0.0))
    if std > MULTISEED_STD_MAX:
        return FAIL, f"种子间标准差 {std:.4f} > {MULTISEED_STD_MAX}"
    missing = [k for k in ("k_sensitivity", "constraint_pareto", "missingness_stress") if k not in r]
    if missing:
        return FAIL, "缺 " + "、".join(missing)
    return PASS, f"3 种子（最大 std {std:.4f}）；k 敏感性、约束 Pareto、缺失压力测试齐全"


def x_consistency() -> tuple[str, str]:
    """三问互证：Q1↔Q3 感官一致、Q2↔Q3 载荷重合。"""
    p = ROOT / "results" / "consistency" / "checks.json"
    if not p.exists():
        return TODO, "缺 results/consistency/checks.json（三问互证未做）"
    r = _json(p)
    checks = r.get("checks") or {}
    need = ["q1_q3_sensory", "q2_q3_loading"]
    missing = [k for k in need if k not in checks]
    if missing:
        return FAIL, "缺互证项：" + "、".join(missing)
    failed = [k for k in need if not checks[k].get("passed")]
    if failed:
        return FAIL, "互证未过：" + "、".join(failed)
    return PASS, "Q1↔Q3 感官一致、Q2↔Q3 载荷重合均在容差内"


def g5_model() -> tuple[str, str]:
    exps = _experiments()
    metas = [e / "metrics.json" for e in exps if (e / "metrics.json").exists()]
    if not metas:
        return TODO, "还没训练结果（make train）"
    problems, note = [], []
    for m in metas:
        r = json.loads(m.read_text(encoding="utf-8"))
        pm = r.get("primary_metric")
        if not pm:
            problems.append(f"{m.parent.name}: metrics.json 缺 primary_metric")
            continue
        n_seeds = len(r.get("seeds") or [])
        cv = r.get("cv") or {}
        std = float(cv.get(f"{pm}_std", cv.get("std", 0.0)) or 0.0)
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

    # 论文里引用的编号，bib 里必须真的存在 —— 否则参考文献会缺条目，编译不报错但论文是残缺的
    bib = ROOT / "paper" / "refs.bib"
    keys = set(re.findall(r"^@\w+\{([^,]+),", bib.read_text(encoding="utf-8"), re.M))
    cited = set()
    for tex in list((ROOT / "paper").glob("*.tex")) + list((ROOT / "paper" / "sections").glob("*.tex")):
        cited |= set(re.findall(r"\\cite[a-z]*\{([^}]+)\}", tex.read_text(encoding="utf-8")))
    cited = {k.strip() for grp in cited for k in grp.split(",")}
    missing = sorted(cited - keys)
    if missing:
        return FAIL, f"论文引用了 bib 里没有的编号：{missing}"

    unused = len(keys - cited)
    note = f"，已引用 {len(cited)} 条" + (f"（另有 {unused} 条备用）" if unused else "")
    return PASS, f"论文已成稿（{p.stat().st_size // 1024} KB）{note}"


def g7_deliver() -> tuple[str, str]:
    arch = ROOT / "submissions" / "archive"
    zips = sorted(arch.glob("*.zip")) if arch.exists() else []
    if not zips:
        return TODO, "submissions/archive/ 里还没有归档的提交物"
    return PASS, f"已归档 {len(zips)} 份提交物（最新 {zips[-1].name}）"


GATES = [
    ("R", "规则与 AI 合规", r_rules),
    ("F", "官方数据冻结", f_freeze),
    ("S", "补充数据", s_supplementary),
    ("D", "数据门", d_data_gate),
    ("Q", "质控", q_qc),
    ("G0", "预注册", g0_prereg),
    ("G1", "问题定义", g1_problem),
    ("G2", "端到端贯通", g2_end_to_end),
    ("G3", "数据可信", g3_leakage),
    ("P1", "问题一交付", p1_q1),
    ("G4", "特征消融", g4_ablation),
    ("P2", "问题二交付", p2_q2),
    ("P3", "问题三交付", p3_q3),
    ("G5", "模型与融合", g5_model),
    ("U", "稳健性", u_robustness),
    ("X", "三问互证", x_consistency),
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
