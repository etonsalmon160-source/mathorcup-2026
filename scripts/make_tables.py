"""生成论文表格（AI 辅助实现，2026-10-24）。

所有数值从 results/ 与 data/raw/ 读取，不在本文件里写死实验结果。
输出 tables/paper/：每个表一份 CSV（可直接核对）+ 一份 LaTeX booktabs 骨架；
长表用 longtable。正文与表注由作者撰写。
用法：
    python scripts/make_tables.py            # 全部
    python scripts/make_tables.py T01 S09    # 指定表
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import src.figstyle as _figstyle  # noqa: E402,F401  （确保 matplotlib 后端一致）

RESULTS = ROOT / "results"
RAW = ROOT / "data" / "raw" / "附件"
OUT = ROOT / "tables" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

META_COLS = {"beer", "beer_id", "tasting_category_fine"}


# ---------------------------------------------------------------- 通用
def _read(rel: str, **kw) -> pd.DataFrame:
    return pd.read_csv(RESULTS / rel, encoding="utf-8-sig", **kw)


def _json(rel: str) -> dict:
    return json.loads((RESULTS / rel).read_text(encoding="utf-8"))


def _raw(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW / name, encoding="utf-8-sig")


def f3(x) -> str:
    return "" if pd.isna(x) else f"{float(x):.3f}"


def f4(x) -> str:
    return "" if pd.isna(x) else f"{float(x):.4f}"


def fp(x) -> str:
    if pd.isna(x):
        return ""
    x = float(x)
    return "<0.001" if x < 0.001 else f"{x:.3f}"


def ci(lo, hi) -> str:
    if pd.isna(lo) or pd.isna(hi):
        return ""
    return f"[{float(lo):.3f}, {float(hi):.3f}]"


def pct(n, d) -> str:
    return "" if not d else f"{100 * float(n) / float(d):.2f}%"


def target_label(t: str) -> str:
    if t.startswith("A_"):
        s = "A · " + t[2:]
    elif t.startswith("F_"):
        s = "F · " + t[2:]
    else:
        s = t
    s = s.replace("_", " ")
    s = (s.replace("bitternes", "bitterness").replace("t2n", "T2N")
         .replace("X4vg", "4VG").replace("co2", "CO2").replace("dms", "DMS"))
    if s.startswith("A · "):
        return "A · " + s[4:5].upper() + s[5:]
    if s.startswith("F · "):
        return "F · " + s[4:5].upper() + s[5:]
    return s[:1].upper() + s[1:]


def tex_escape(s) -> str:
    s = str(s)
    repl = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
            ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
            ("}", r"\}"), ("~", r"\textasciitilde{}"),
            ("^", r"\textasciicircum{}"), ("★", r"$\star$"),
            ("→", r"$\rightarrow$"), ("≥", r"$\geq$"),
            ("≤", r"$\leq$"), ("×", r"$\times$")]
    for a, b in repl:
        s = s.replace(a, b)
    return s


def write_table(spec: dict, df: pd.DataFrame, note: str) -> dict:
    key, slug = spec["key"], spec["slug"]
    stem = OUT / f"{key}_{slug}"
    df.to_csv(stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    align = spec["align"]
    assert len(align) == df.shape[1], (key, align, df.shape[1])
    head = " & ".join(tex_escape(c) for c in df.columns) + r" \\"
    body = [" & ".join(tex_escape(v) for v in row) + r" \\"
            for row in df.itertuples(index=False, name=None)]
    cap = tex_escape(spec["title"])
    label = f"tab:{key.lower()}_{slug}"
    size = spec.get("size", "footnotesize")
    colsep = spec.get("colsep", "3.5pt")

    if spec.get("long"):
        tex = [
            r"{%s" % ("\\" + size if size else ""),
            r"\setlength{\tabcolsep}{%s}" % colsep,
            r"\begin{longtable}{%s}" % align,
            r"\caption{%s}" % cap,
            r"\label{%s}\\" % label,
            r"\toprule",
            head,
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            head,
            r"\midrule",
            r"\endhead",
            r"\bottomrule",
            r"\endlastfoot",
            *body,
            r"\end{longtable}",
            r"}",
        ]
    else:
        tex = [
            r"\begin{table}[tbp]",
            r"\centering",
            r"\caption{%s}" % cap,
            r"\label{%s}" % label,
            r"\%s" % size if size else "",
            r"\setlength{\tabcolsep}{%s}" % colsep,
            r"\begin{tabular}{%s}" % align,
            r"\toprule",
            head,
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    tex = [t for t in tex if t != ""]
    if note:
        tex.append("% 注（写入图注/表注，勿直接编译）：" + note.replace("\n", " "))
    stem.with_suffix(".tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"  {key}: {df.shape[0]} 行 × {df.shape[1]} 列")
    return {"key": key, "title": spec["title"], "rows": df.shape[0],
            "cols": df.shape[1], "source": spec["source"],
            "csv": str(stem.with_suffix(".csv").relative_to(ROOT)),
            "tex": str(stem.with_suffix(".tex").relative_to(ROOT)),
            "note": note, "long": bool(spec.get("long"))}


# ---------------------------------------------------------------- T01
def t01():
    x, y = _raw("1.csv"), _raw("4.csv")
    chem = [c for c in x.columns if c not in META_COLS]
    sens = [c for c in y.columns if c not in META_COLS]
    imp = _read("q1/importance.csv")
    derived = set(imp.loc[imp["module"].astype(str).str.contains("派生"), "feature"])
    meas = [c for c in chem if c not in derived]
    miss_c = int(x[chem].isna().sum().sum())
    miss_s = int(y[sens].isna().sum().sum())
    floor = _read("qc/floor_by_feature.csv")
    n_floor = int(floor["n_at_floor"].sum())
    qc = _json("qc/report.json")
    gate = _json("data_gate/report.json")
    counts = y["tasting_category_fine"].value_counts()
    sk = qc["r_extra"]["style_kruskal"]
    n_pass = sum(bool(g["passed"]) for g in gate["gates"])
    n_gate = len(gate["gates"])
    n_flag_s = len(qc["layers"]["sample"]["flagged_beers"])
    n_flag_f = qc["layers"]["feature"]["n_flagged"]
    n_unusable = qc["layers"]["feature"]["n_unusable"]
    tr = gate["transform_counts"]

    rows = [
        ("样本", "啤酒样本 / 风格数", f"{len(x)} / {y['tasting_category_fine'].nunique()}"),
        ("样本", "各风格样本量（最小 / 中位 / 最大）",
         f"{counts.min()} / {int(counts.median())} / {counts.max()}"),
        ("特征", "化学特征列", f"{len(chem)}（实测化合物 {len(meas)} + 派生汇总 {len(derived)}）"),
        ("特征", "感官维度", f"{len(sens)}"),
        ("缺失", "化学缺失单元格", f"{miss_c} 格（{pct(miss_c, len(x) * len(chem))}）"),
        ("缺失", "感官缺失单元格", f"{miss_s} 格（{pct(miss_s, len(y) * len(sens))}）"),
        ("检出限", "下限哨兵值单元格",
         f"{n_floor} 格（{pct(n_floor, len(x) * len(chem))}）；涉及 {len(floor)} 个特征"),
        ("变换", "数值变换列数", f"log10 {tr.get('log10', 0)} 列；原始尺度 {tr.get('raw', 0)} 列"),
        ("风格结构", "Kruskal–Wallis FDR < 0.05",
         f"化学 {sk['chem_n_fdr05']}/{len(chem)}；感官 {sk['sens_n_fdr05']}/{len(sens)}"),
        ("质量标记", "样本 / 特征 / 不可用",
         f"{n_flag_s} / {n_flag_f} / {n_unusable}"),
        ("数据门", "D0–D6 通过数", f"{n_pass}/{n_gate}"),
    ]
    df = pd.DataFrame(rows, columns=["层面", "指标", "数值"])
    note = ("“缺失”按原始 1.csv/4.csv 直接计数；下限哨兵值比例按 250 × 231 化学单元格计。"
            "风格检验为 Kruskal–Wallis 后 BH-FDR；质量标记的定义见 results/qc/report.json。")
    return df, note


# ---------------------------------------------------------------- T02
def t02():
    c = _read("q1/comparison.csv")
    g = (c.groupby(["model", "model_cn", "scheme"])
         .agg(r2=("mean_r2", "mean"), r2_sd=("mean_r2", "std"),
              med=("median_r2", "mean"), rho=("mean_spearman", "mean"),
              runtime=("runtime_s", "mean")).reset_index())
    piv = g.pivot(index=["model", "model_cn"], columns="scheme")
    rows = []
    for (model, cn), r in piv.iterrows():
        rows.append({
            "模型": cn,
            "随机CV R²": f"{r[('r2','random')]:.3f} ± {r[('r2_sd','random')]:.3f}",
            "随机CV 中位R²": f3(r[("med", "random")]),
            "随机CV Spearman": f3(r[("rho", "random")]),
            "分组CV R²": f"{r[('r2','group')]:.3f} ± {r[('r2_sd','group')]:.3f}",
            "分组CV 中位R²": f3(r[("med", "group")]),
            "分组CV Spearman": f3(r[("rho", "group")]),
            "耗时 (s)": f"{r[('runtime','random')]:.1f}",
            "_sort": r[("r2", "random")],
        })
    df = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort")
    df["耗时 (s)"] = pd.to_numeric(df["耗时 (s)"], errors="coerce").round(1)
    note = ("每格为 3 个随机种子（42/43/44）的均值 ± 标准差；随机CV 与分组CV 的划分不同，"
            "两列不能直接做配对检验。R² 为 50 个感官维度的平均 OOF R²。")
    return df, note


# ---------------------------------------------------------------- T03
def t03():
    rep = _json("composite/report.json")
    names = {
        "et": "Extra Trees（单模型）",
        "latent_sens_ridge": "感官 PCA8 + Ridge（单模型）",
        "avg": "等权融合",
        "stack_global": "全局线性堆叠",
        "stack_target": "逐维线性堆叠",
        "gate_domain": "领域规则软门控",
        "gate_cv": "CV 学习软门控",
        "oracle_reference": "Oracle 上界",
    }
    adopted = rep.get("adopted_variant")
    rows = []
    for scheme, zh in [("random", "随机 CV"), ("group", "分组 CV")]:
        for variant in ["et", "latent_sens_ridge", "avg", "stack_global",
                        "stack_target", "gate_domain", "gate_cv",
                        "oracle_reference"]:
            key = f"{scheme}/{variant}"
            if key not in rep["summary"]:
                continue
            v = rep["summary"][key]
            mark = []
            if variant == adopted:
                mark.append("★ 采纳")
            if variant == "oracle_reference":
                mark.append("仅上界")
            rows.append({
                "划分": zh,
                "方案": names.get(variant, variant),
                "种子平均预测 OOF R²": f4(v["mean_r2"]),
                "Δ vs ET": (("+" if v["delta_vs_et"] >= 0 else "") + f4(v["delta_vs_et"])) if "delta_vs_et" in v else "—",
                "95% CI": ci(*(v["boot_ci95"])) if "boot_ci95" in v else "—",
                "Wilcoxon p": fp(v["wilcoxon_p"]) if "wilcoxon_p" in v else "—",
                "优于 ET 维数": f"{v['n_targets_better']}/50" if "n_targets_better" in v else "—",
                "标记": "；".join(mark),
            })
    df = pd.DataFrame(rows)
    note = ("R² 为 3 个种子预测取平均后的 OOF R²；因此单模型数值与表 T02 的逐种子均值略有差异。"
            "Δ、CI、p 均以同划分下的 Extra Trees 为参照；CI 为按啤酒聚类的 Bootstrap 95% 区间，"
            "p 为逐维 Wilcoxon 符号秩检验。Oracle 使用真实标签选方案，只作上界，不可采纳。")
    return df, note


# ---------------------------------------------------------------- T04
def t04():
    a = _read("q1/ablation.csv")
    rows = []
    for _, r in a.iterrows():
        rows.append({
            "特征组": r["特征组"],
            "处理口径": r["处理口径"],
            "划分": "随机 CV" if r["划分"] == "random" else "分组 CV",
            "平均 R²": f4(r["mean_r2"]),
            "种子 SD": f4(r["seed_std"]),
            "Δ vs 完整": "—" if r["特征组"].startswith("完整") else ("+" if r["与完整模型的差"] >= 0 else "") + f4(r["与完整模型的差"]),
            "95% CI": "—" if r["特征组"].startswith("完整") else ci(r["差_95CI下"], r["差_95CI上"]),
            "判定": r["判定"],
        })
    df = pd.DataFrame(rows)
    note = ("Δ = 去掉该特征组后的平均 OOF R² − 完整模型；CI 为 3 个种子 × 50 维的 Bootstrap 区间。"
            "“有贡献或中性”表示 CI 未显示明确损失，不等同于该组必要。")
    return df, note


# ---------------------------------------------------------------- T05
def t05():
    q2 = _json("q2/report.json")
    st = _json("q2/stability.json")
    g = q2["graph"]
    gj = list(st["graph_gamma_jaccard"].values())
    g0 = list(g["gamma_top3_jaccard_vs_gamma0"].items())[-1]
    rows = [
        ("网络", "化合物节点数", f"{g['n_nodes']}"),
        ("网络", "共表达边数", f"{g['n_edges']}"),
        ("网络", "平均度", f"{g['mean_degree']:.2f}"),
        ("网络", "模块数", f"{g['n_communities']}"),
        ("网络", "模块度", f"{g['modularity']:.3f}"),
        ("筛选", "Lasso 稳定性选择 α", f"{st['alpha']:.2f}"),
        ("筛选", "Bootstrap 次数 B", f"{st['n_bootstrap']}"),
        ("筛选", "分半次数", f"{st['n_split']}"),
        ("稳定性", "稳定率阈值", f"{st['stable_rate_threshold']:.2f}"),
        ("稳定性", "低稳定维度数", f"{st['n_unstable']}/50"),
        ("稳定性", "分半 Jaccard 中位数", f"{st['median_jaccard']:.4f}"),
        ("图平滑", "排序权重 γ", f"{st['graph_gamma']:.2f}"),
        ("图平滑", "分半 Jaccard（γ = 0–0.5 范围）", f"{min(gj):.3f}–{max(gj):.3f}"),
        ("图平滑", f"Top-3 Jaccard vs γ = 0（γ = {g0[0]}）", f"{g0[1]:.3f}"),
    ]
    df = pd.DataFrame(rows, columns=["层面", "指标", "数值"])
    note = ("主表只收正向激动剂（Lasso positive、coef > 0、Spearman > 0）；负向掩蔽物另表。"
            "稳定性为 5 次随机分半、两半各自 Top-3 的 Jaccard；γ 事先固定为 0.20，未按测试结果调节。")
    return df, note


# ---------------------------------------------------------------- T06
def t06():
    c = _read("q2/communities.csv")
    ann = _read("supplementary/q2_module_annotation.csv")
    imp = _read("q1/importance_modules.csv")
    rows = []
    for m in sorted(c["community"].unique()):
        sub = c[c["community"] == m]
        hub = sub.sort_values(["degree", "node"], ascending=[False, True]).iloc[0]
        imp_m = imp[imp["module"].astype(str) == str(m)]
        tops = ""
        if len(imp_m):
            tops = "、".join(str(imp_m.iloc[0]["top_compounds"]).split("、")[:3])
        a = ann[ann["module"] == m]
        origin = "—" if not len(a) or pd.isna(a.iloc[0]["top_origins"]) else a.iloc[0]["top_origins"]
        q = np.nan if not len(a) else a.iloc[0]["q_threshold"]
        noff = np.nan if not len(a) else a.iloc[0]["n_off_flavor"]
        rows.append({
            "模块": f"ME{m}",
            "节点数": len(sub),
            "度中心枢纽": f"{hub['node']} ({int(hub['degree'])})",
            "代表性化合物（前 3）": tops,
            "富集来源（倍数）": origin,
            "q 值": f"{q:.2e}" if pd.notna(q) and q < 0.001 else f3(q),
            "异味化合物数": "" if pd.isna(noff) else int(noff),
        })
    df = pd.DataFrame(rows).sort_values("节点数", ascending=False)
    note = ("富集检验只针对化合物来源标签（Hops / Malt / Yeast / Wild flora / Others），"
            "q 为 BH-FDR；ME3 的酵母富集是唯一通过 FDR 的模块。代表性化合物按 Q1 置换重要性排序。")
    return df, note


# ---------------------------------------------------------------- T07
def t07():
    t = _read("q2/top3.csv")
    rows = []
    for _, r in t.iterrows():
        low = bool(r.get("low_stability", False))
        rows.append({
            "感官维度": target_label(r["sensory"]),
            "Top-1": r["top1_full"],
            "Top-2": r["top2_full"],
            "Top-3": r["top3_full"],
            "稳定率": f3(r["stable_rate"]),
            "分半 Jaccard": f3(r["jaccard_median"]),
            "低稳定": "是" if low else "",
        })
    df = pd.DataFrame(rows)
    note = ("Top-3 为正向激动剂；稳定率为 300 次分层 Bootstrap 中进入 Top-3 的频率，"
            "分半 Jaccard 为 5 次随机分半的中位数。“低稳定”= 稳定率 < 0.60，"
            "共 24 维，论文中只能作探索性结论。")
    return df, note


def _any_worsening(long: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["worsened_stale_hops", "worsened_dms",
                        "worsened_diacetyl", "worsened_barnyard",
                        "worsened_metallic"] if c in long.columns]
    return long[cols].fillna(False).astype(bool).any(axis=1)


# ---------------------------------------------------------------- T08
def t08():
    q = _json("supplementary/upgrade/q3_qp.json")
    l11 = _read("supplementary/upgrade/q3_qp_long.csv")
    l924 = _read("supplementary/upgrade/q3_qp_pairs_long.csv")
    method_zh = {"none": "无校正", "ot_proj": "OT 投影",
                 "ot_cond": "OT 条件化", "qp_full": "联合 QP"}
    mode_zh = {"none": "—", "loose": "松", "strict": "严"}
    rows = []
    for label, summary, long in [("11 个 Pils/Lager 样本", q["summary"], l11),
                                 ("924 个风格对", q["pairs_summary"], l924)]:
        for s in summary:
            m, mo = s["method"], s["mode"]
            sub = long[(long["method"] == m) & (long["mode"] == mo)]
            n = int(s["n"])
            any_w = int(_any_worsening(sub).sum()) if len(sub) else 0
            rows.append({
                "数据": label,
                "方法": method_zh.get(m, m),
                "约束": mode_zh.get(mo, mo),
                "n": n,
                "可行率": pct(s["feasible_sensory"], n),
                "风格位移中位数": f3(s["style_shift_median"]),
                "QP 成本中位数": f3(s["qp_cost_median"]),
                "新增 DMS": int(s["new_above_DMS"]),
                "新增双乙酰": int(s["new_above_X2.3.butanedione"]),
                "新增 4VG": int(s["new_above_X4VG"]),
                "任一维劣化": f"{any_w}/{n}（{pct(any_w, n)}）",
            })
    df = pd.DataFrame(rows)
    note = ("“新增”指从阈值以下升到阈值以上；“任一维劣化”是 stale_hops、DMS、diacetyl、"
            "barnyard、metallic 五个感官维度中至少一个变差。风格位移与 QP 成本均为中位数，"
            "越小越接近源风格 / 约束越弱。")
    return df, note


# ---------------------------------------------------------------- T09
def t09():
    l = _read("supplementary/upgrade/q3_qp_pairs_long.csv")
    dims = [("stale_hops", "老化酒花"), ("dms", "DMS"), ("diacetyl", "双乙酰"),
            ("barnyard", "马厩味"), ("metallic", "金属味")]
    method_zh = {"none": "无校正", "ot_proj": "OT 投影", "qp_full": "联合 QP"}
    rows = []
    for m in ["none", "ot_proj", "qp_full"]:
        mo = "none" if m == "none" else "loose"
        sub = l[(l["method"] == m) & (l["mode"] == mo)]
        n = len(sub)
        if not n:
            continue
        for code, zh in dims:
            w = int(sub[f"worsened_{code}"].fillna(False).astype(bool).sum())
            rows.append({
                "方法": method_zh[m],
                "感官维度": zh,
                "劣化 n/N": f"{w}/{n}",
                "劣化率": pct(w, n),
                "Δ 中位数": f3(sub[f"delta_{code}"].median()),
            })
        w_any = int(_any_worsening(sub).sum())
        rows.append({
            "方法": method_zh[m],
            "感官维度": "任一维",
            "劣化 n/N": f"{w_any}/{n}",
            "劣化率": pct(w_any, n),
            "Δ 中位数": "",
        })
    df = pd.DataFrame(rows)
    note = ("Δ = 约束后 − 约束前的风格内标准化感官分；Δ > 0 表示该缺陷维度变差。"
            "五个维度是预注册的监测维度，未使用测试集或生成结果反向挑选。")
    return df, note


# ---------------------------------------------------------------- T10
def t10():
    checks = _json("consistency/checks.json")["checks"]
    q1q3 = checks["q1_q3_sensory"]
    dual = _json("supplementary/consistency_dual.json")
    tr = {t["name"]: t for t in dual["tracks"]}
    thr = dual["threshold"]
    rows = [
        ("Q1 ↔ Q3", "生成 vs 真实风格均值", "平均 Spearman ρ",
         f4(q1q3["mean_spearman"]), f3(q1q3["threshold"]),
         "通过" if q1q3["passed"] else "未通过"),
        ("Q1 ↔ Q3", "生成 vs 真实风格均值", "中位 Spearman ρ",
         f4(q1q3["median_spearman"]), f3(q1q3["threshold"]),
         "通过" if q1q3["passed"] else "未通过"),
        ("Q1 ↔ Q3", "生成 vs 真实风格均值", "平均 RMSE",
         f4(q1q3["mean_rmse"]), "—", "—"),
    ]
    for name, label in (("signed", "有符号 Lasso"), ("positive", "正向 Lasso")):
        t = tr[name]
        rows += [
            ("Q2 ↔ Q3", label, "平均 Spearman ρ", f4(t["mean_spearman"]),
             f3(thr), "通过" if t["passed"] else "未通过"),
            ("Q2 ↔ Q3", label, "中位 Spearman ρ", f4(t["median_spearman"]),
             f3(thr), "通过" if t["passed"] else "未通过"),
            ("Q2 ↔ Q3", label, "维度数 ≥ 阈值",
             f"{t['n_dims_passing_threshold']}/{t['n_dims']}", "—", "—"),
            ("Q2 ↔ Q3", label, "Top-20 重合率 / 富集",
             f"{f3(t['mean_top20_overlap'])} / {f3(t['hypergeom_fold'])}×",
             f"p = {t['hypergeom_p']:.3f}", "—"),
        ]
    df = pd.DataFrame(rows, columns=["检查", "口径", "指标", "数值", "阈值/检验", "结论"])
    note = ("Q2↔Q3 按两种口径并列：有符号 Lasso 把正、负效应都计入入选频率，"
            f"平均 ρ = {tr['signed']['mean_spearman']:.4f}（{tr['signed']['n_dims_passing_threshold']}/50 维"
            f" ≥ {thr:.2f}，通过）；正向 Lasso 即 Q2 主结果，平均 ρ = "
            f"{tr['positive']['mean_spearman']:.4f}（{tr['positive']['n_dims_passing_threshold']}/50 维，未通过）。"
            "两口径的 Spearman 中位数同号，Top-20 重合均显著富集于随机期望（超几何检验 p = "
            f"{tr['signed']['hypergeom_p']:.3f}）。主结果 results/q2/top3.csv 仍是纯正向激动剂，"
            "未因本表改动。Q1↔Q3 取自 results/consistency/checks.json；Q2↔Q3 取自 "
            "results/supplementary/consistency_dual.json（同 Bootstrap 分层、权重、候选池与图平滑）。"
            "checks.json 中旧的 Q2↔Q3 数值（平均 0.1723）早于 Q2 重跑，已不可复现，不再引用。"
            "此表不修改 results/q1–q3 或 results/composite。")
    return df, note


# ---------------------------------------------------------------- T11
def t11():
    ov = _read("supplementary/variance_decomposition_overall.csv").iloc[0]
    cat = _read("supplementary/variance_decomposition_by_category.csv")
    rows = [{
        "层级": "总体",
        "n 维": int(ov["n_targets"]),
        "风格 R²": f4(ov["mean_r2_style"]),
        "+ 潜状态 R²": f4(ov["mean_r2_plus_pls"]),
        "+ ET R²": f4(ov["mean_r2_full_et"]),
        "Δ 潜状态": f4(ov["mean_delta_pls"]),
        "Δ ET": f4(ov["mean_delta_et"]),
        "风格占比": "—",
        "潜状态占比": "—",
        "ET 占比": "—",
        "负增量维数（潜/ET）": f"{int(ov['n_delta_pls_negative'])} / {int(ov['n_delta_et_negative'])}",
    }]
    for _, r in cat.iterrows():
        rows.append({
            "层级": r["category"],
            "n 维": int(r["n"]),
            "风格 R²": f4(r["r2_style"]),
            "+ 潜状态 R²": f4(r["r2_plus_pls"]),
            "+ ET R²": f4(r["r2_full_et"]),
            "Δ 潜状态": f4(r["delta_pls"]),
            "Δ ET": f4(r["delta_et"]),
            "风格占比": f3(r["share_style"]),
            "潜状态占比": f3(r["share_pls"]),
            "ET 占比": f3(r["share_et"]),
            "负增量维数（潜/ET）": "—",
        })
    df = pd.DataFrame(rows)
    note = ("序贯残差分解：先只用风格均值，再加入 8 维感官潜状态，最后加入 Extra Trees。"
            "Δ 为外层 OOF 上的增量；负值表示该层没有增益，未做截断。占比按三层正增量归一。")
    return df, note


# ---------------------------------------------------------------- T12
def t12():
    rob = _json("robustness/summary.json")
    pt = _read("composite/per_target.csv")
    pt = pt[pt.variant == "avg"]
    med = {}
    for scheme in ["random", "group"]:
        g = pt[pt.scheme == scheme]
        w = g.pivot_table(index="target", columns="seed", values="r2")
        sd = w.std(axis=1, ddof=1)
        med[scheme] = float(sd.median())
    ks = pd.DataFrame(rob["k_sensitivity"])
    rng = {s: (ks[ks.scheme == s]["mean_r2"].min(), ks[ks.scheme == s]["mean_r2"].max())
           for s in ["random", "group"]}
    ms = pd.DataFrame(rob["missingness_stress"]).sort_values("missing_rate")
    m0 = ms.iloc[0]
    m20 = ms.iloc[-1]
    cp = pd.DataFrame(rob["constraint_pareto"])
    rows = [
        ("种子波动", "逐维 R² 种子 SD", "随机 CV 中位数", f4(med["random"])),
        ("种子波动", "逐维 R² 种子 SD", "分组 CV 中位数", f4(med["group"])),
        ("种子波动", "主模型平均 R² 种子 SD", "最大值", f4(rob["max_seed_std"])),
        ("k 敏感性", "k = 6–22", "随机 CV 平均 R² 范围",
         f"{rng['random'][0]:.3f}–{rng['random'][1]:.3f}"),
        ("k 敏感性", "k = 6–22", "分组 CV 平均 R² 范围",
         f"{rng['group'][0]:.3f}–{rng['group'][1]:.3f}"),
        ("缺失压力", "随机缺失 0 → 20%", "平均 R² 变化",
         f"{m20['mean_r2'] - m0['mean_r2']:+.4f}"),
        ("缺失压力", "20% 缺失", "平均 R²", f4(m20["mean_r2"])),
        ("缺失压力", "20% 缺失", "中位 R²", f4(m20["median_r2"])),
    ]
    for mode, zh in [("none", "无约束"), ("loose", "松约束"), ("strict", "严约束")]:
        g = cp[cp["mode"] == mode]
        rows.append(("约束 λ", "λ = 0.02–0.50", f"{zh}：风格位移中位数范围",
                     f"{g['style_shift_median'].min():.3f}–{g['style_shift_median'].max():.3f}"))
    for mode, zh in [("none", "无约束"), ("loose", "松约束"), ("strict", "严约束")]:
        g = cp[cp["mode"] == mode]
        rows.append(("约束 λ", "λ = 0.02–0.50", f"{zh}：QP 成本中位数范围",
                     f"{g['qp_cost_median'].min():.3f}–{g['qp_cost_median'].max():.3f}"))
    df = pd.DataFrame(rows, columns=["检查", "条件", "指标", "结果"])
    note = ("种子波动用主融合模型 avg 在 50 维上的逐维 SD 中位数；k 敏感性用化学潜状态 + Ridge；"
            "缺失压力为完全随机缺失，不代表真实检出限机制；λ 扫描只到 0.50，未外推。")
    return df, note


# ---------------------------------------------------------------- S01
def s01():
    m = _json("q1/metrics.json")["per_target"]
    rows = []
    for t, v in m.items():
        rows.append({
            "感官维度": target_label(t),
            "随机CV R²": f4(v["r2_random"]),
            "随机CV RMSE": f4(v["rmse_random"]),
            "随机CV MAE": f4(v["mae_random"]),
            "随机CV Spearman": f4(v["spearman_random"]),
            "分组CV R²": f4(v["r2_group"]),
            "分组CV RMSE": f4(v["rmse_group"]),
            "分组CV MAE": f4(v["mae_group"]),
            "分组CV Spearman": f4(v["spearman_group"]),
        })
    df = pd.DataFrame(rows)
    note = ("Q1 主模型（Extra Trees）的 50 维逐维指标；R² 为 3 个种子的均值。"
            "分组 CV 按风格留出，评估对未见风格的迁移。")
    return df, note


# ---------------------------------------------------------------- S02
def s02():
    tr = _json("data_gate/report.json")["transform_table"]
    rows = []
    for r in tr:
        rows.append({
            "列名": r["column"],
            "化合物": r["chemical_name"],
            "变换": r["transform"],
            "判定来源": r["source"],
            "有效样本": int(r["n_valid"]),
            "中位数": f4(r["median"]),
        })
    df = pd.DataFrame(rows)
    note = ("log10 表示在 log10 尺度建模；raw 为原始尺度。判定依据见 data_gate/report.json："
            "负值证明、校准表、峰面积推断三类证据。")
    return df, note


# ---------------------------------------------------------------- S03
def s03():
    c = _read("q2/communities.csv")
    df = c.rename(columns={"node": "化合物", "community": "模块",
                           "degree": "度", "module_size": "模块大小"})
    df["模块"] = "ME" + df["模块"].astype(str).replace("nan", "—")
    df = df.sort_values(["模块", "度", "化合物"], ascending=[True, False, True])
    note = ("223 个节点为通过候选池与排除规则的化合物；模块编号与正文、图 7 一致。")
    return df, note


# ---------------------------------------------------------------- S04
def s04():
    p = _read("q3_pairs/pairwise.csv")
    p = p[p["mode"] == "loose"].copy()
    df = p.rename(columns={
        "source": "源风格", "target": "目标风格", "n": "n",
        "feasible_rate": "可行率", "style_shift_median": "风格位移中位数",
        "qp_cost_median": "QP 成本中位数", "maha_median": "Mahalanobis 中位数",
        "sens_cos_median": "感官余弦中位数", "new_DMS": "新增 DMS",
        "new_diacetyl": "新增双乙酰", "new_4VG": "新增 4VG"})
    df["可行率"] = df["可行率"].map(lambda x: pct(x, 1))
    for c in ["风格位移中位数", "QP 成本中位数", "Mahalanobis 中位数", "感官余弦中位数"]:
        df[c] = df[c].map(f3)
    df = df[["源风格", "目标风格", "n", "可行率", "风格位移中位数", "QP 成本中位数",
             "Mahalanobis 中位数", "感官余弦中位数", "新增 DMS", "新增双乙酰", "新增 4VG"]]
    note = ("每个源风格 → 目标风格对取 12 个最近样本（不足 12 则取全部），"
            "表中只列松约束（loose），严格约束（strict）见 CSV 原始文件。")
    return df, note


# ---------------------------------------------------------------- S05
def s05():
    a = _read("supplementary/q3_aroma_oav_ratio.csv")
    df = a.rename(columns={"compound": "化合物", "Pils/Lager": "Pils/Lager OAV",
                           "Tripel": "Tripel OAV",
                           "ratio_tripel_vs_pils": "Tripel / Pils 倍数"})
    df["化合物"] = (df["化合物"].str.replace("_", " ", regex=False)
                    .str.replace(r"^.", lambda m: m.group(0).upper(), regex=True))
    for c in ["Pils/Lager OAV", "Tripel OAV", "Tripel / Pils 倍数"]:
        df[c] = df[c].map(f3)
    note = ("OAV = 浓度 / 阈值；倍数 > 1 表示该香气在 Tripel 中相对增强。"
            "仅列 7 个有明确阈值的香气化合物。")
    return df, note


# ---------------------------------------------------------------- S06
def s06():
    d = _read("supplementary/upgrade/q1_latent_delta_ci.csv")
    df = d.rename(columns={"latent": "潜状态方法", "scheme": "划分",
                           "delta_vs_pca": "Δ vs PCA",
                           "ci_lo": "95% CI 下", "ci_hi": "95% CI 上",
                           "n_dim_pos": "正增量维数"})
    df["划分"] = df["划分"].map({"random": "随机 CV", "group": "分组 CV"})
    for c in ["Δ vs PCA", "95% CI 下", "95% CI 上"]:
        df[c] = df[c].map(f4)
    note = ("以 14 维化学 PCA 为参照，比较因子分析（FA）与偏最小二乘（PLS）潜状态的增量；"
            "正增量维数指 50 维中 Δ > 0 的维数。")
    return df, note


# ---------------------------------------------------------------- S07
def s07():
    a = _read("q2_sensitivity/alpha_sensitivity.csv")
    o = _read("q2_sensitivity/top3_overlap.csv")
    df = a.merge(o, on="alpha", how="outer").sort_values("alpha")
    df = df.rename(columns={
        "alpha": "α", "n_boot": "Bootstrap 次数", "median_jaccard": "Jaccard 中位数",
        "mean_jaccard": "Jaccard 均值", "n_unstable": "低稳定维数",
        "mean_stable_rate": "平均稳定率", "top3_jaccard_vs_main": "Top-3 Jaccard vs 主表"})
    df["α"] = df["α"].map(lambda x: f"{x:.2f}")
    for c in ["Jaccard 中位数", "Jaccard 均值", "平均稳定率", "Top-3 Jaccard vs 主表"]:
        df[c] = df[c].map(f3)
    note = ("主分析取 α = 0.06；α = 0.04 更宽松、0.08 更严格。Top-3 Jaccard 为与主表 Top-3 的重合度。")
    return df, note


# ---------------------------------------------------------------- S08
def s08():
    e = _read("supplementary/external_abv_ibu.csv")
    df = e.rename(columns={"dataset": "数据", "style": "风格", "n": "n",
                           "ABV_mean": "ABV 均值", "IBU_mean": "IBU 均值"})
    df["数据"] = df["数据"].map({"competition": "赛道 A", "external": "外部核验"})
    df["风格"] = (df["风格"].str.replace("external_", "外部 ", regex=False)
                  .str.replace("_", " ", regex=False))
    df["ABV 均值"] = df["ABV 均值"].map(lambda x: f"{x:.2f}")
    df["IBU 均值"] = df["IBU 均值"].map(lambda x: f"{x:.1f}")
    note = ("competition 为赛道 A 数据；external 为独立公开风格数据。"
            "用于核验风格层面的酒精度/苦度方向，不参与建模。")
    return df, note


# ---------------------------------------------------------------- S09
def s09():
    rob = _json("robustness/summary.json")
    rows = []
    for r in rob["k_sensitivity"]:
        rows.append(("k 敏感性", f"k = {r['k']}",
                     "随机 CV" if r["scheme"] == "random" else "分组 CV",
                     f4(r["mean_r2"]), f4(r["median_r2"]), "", "", "", ""))
    for r in rob["missingness_stress"]:
        rows.append(("缺失压力", f"{100 * r['missing_rate']:.0f}% 缺失", "Extra Trees",
                     f4(r["mean_r2"]), f4(r["median_r2"]), "", "", "", ""))
    for r in rob["constraint_pareto"]:
        rows.append(("约束 Pareto", f"λ = {r['lam']:.2f}",
                     {"none": "无约束", "loose": "松", "strict": "严"}[r["mode"]],
                     "", "", f3(r["style_shift_median"]), f3(r["qp_cost_median"]),
                     int(r["new_DMS"]), int(r["new_diacetyl"])))
    df = pd.DataFrame(rows, columns=["检查", "条件", "方案", "平均 R²", "中位 R²",
                                     "风格位移中位数", "QP 成本中位数",
                                     "新增 DMS", "新增双乙酰"])
    note = ("k 与缺失压力用化学潜状态 + Ridge / Extra Trees；Pareto 的 λ 只扫到 0.50。"
            "表中数值与 results/robustness/summary.json 一一对应。")
    return df, note


# ---------------------------------------------------------------- S10
def s10():
    i = _read("q1/importance.csv").sort_values("rank").head(25)
    df = i[["rank", "display", "family", "module", "delta_mse_z", "rel_increase"]].copy()
    df.columns = ["排名", "化合物", "来源", "模块", "置换 ΔMSE (z)", "相对增幅"]
    df["来源"] = df["来源"].fillna("派生")
    df["模块"] = df["模块"].fillna("—").astype(str).str.replace(r"\.0$", "", regex=True)
    df["置换 ΔMSE (z)"] = df["置换 ΔMSE (z)"].map(f4)
    df["相对增幅"] = df["相对增幅"].map(f4)
    note = ("置换重要性：打乱该化合物后模型 MSE 的增量（按维标准化后求和）；"
            "只列前 25 名，完整 232 行见 results/q1/importance.csv。")
    return df, note


# ---------------------------------------------------------------- S11
def s11():
    d = _read("supplementary/q3_defect_chemical_summary.csv")
    df = d[["compound", "constraint", "unit", "threshold", "median_before",
            "median_after", "median_ratio", "n_increased", "n_decreased",
            "n_above_before", "n_above_after", "n_new_above",
            "wilcoxon_p", "bh_q"]].copy()
    df.columns = ["化合物", "约束", "单位", "阈值", "前中位数", "后中位数",
                  "倍数", "升高 n", "降低 n", "前超标 n", "后超标 n",
                  "新增超标 n", "Wilcoxon p", "BH q"]
    tr = _json("data_gate/report.json")["transform_table"]
    nm = {r["column"]: r["chemical_name"] for r in tr}
    df["化合物"] = df["化合物"].map(lambda x: nm.get(x, x))
    df["约束"] = df["约束"].map({"none": "无约束", "loose": "松", "strict": "严"})
    df["单位"] = df["单位"].str.replace("ug/L", "µg/L", regex=False)
    for c in ["阈值", "前中位数", "后中位数", "倍数"]:
        df[c] = df[c].map(f4)
    df["Wilcoxon p"] = df["Wilcoxon p"].map(fp)
    df["BH q"] = df["BH q"].map(fp)
    note = ("11 个 Pils/Lager 样本的配对检验；约束列 none/loose/strict 对应无约束/松/严。"
            "阈值来自 data_gate/report.json 的校准阈值；OAV 与超标分析见正文。")
    return df, note


SPECS = [
    dict(key="T01", slug="data_overview", title="数据与质量总览",
         source="data/raw/附件/1.csv, 4.csv; results/data_gate/report.json; results/qc/report.json",
         align="lll", build=t01),
    dict(key="T02", slug="q1_models", title="问题一：单模型交叉验证比较",
         source="results/q1/comparison.csv", align="lrrrrrrr", build=t02),
    dict(key="T03", slug="q1_fusion", title="问题一：融合与门控方案比较",
         source="results/composite/report.json", align="llrrrrrl", build=t03),
    dict(key="T04", slug="q1_ablation", title="问题一：特征组消融",
         source="results/q1/ablation.csv", align="lllrrrrl", build=t04),
    dict(key="T05", slug="q2_network", title="问题二：共表达网络与稳定性选择",
         source="results/q2/report.json; results/q2/stability.json",
         align="lll", build=t05),
    dict(key="T06", slug="q2_modules", title="问题二：共表达模块结构与富集",
         source="results/q2/communities.csv; results/q1/importance_modules.csv; results/supplementary/q2_module_annotation.csv",
         align="lrl lll l".replace(" ", ""), build=t06),
    dict(key="T07", slug="q2_top3", title="问题二：各感官维度的 Top-3 关键化合物",
         source="results/q2/top3.csv", align="llllrrl", build=t07, long=True),
    dict(key="T08", slug="q3_methods", title="问题三：风格迁移方法比较",
         source="results/supplementary/upgrade/q3_qp.json; q3_qp_long.csv; q3_qp_pairs_long.csv",
         align="llrrrrrrrrr", build=t08),
    dict(key="T09", slug="q3_defects", title="问题三：五个监测感官维度的劣化",
         source="results/supplementary/upgrade/q3_qp_pairs_long.csv",
         align="llrrr", build=t09),
    dict(key="T10", slug="consistency", title="三问互证检查（Q2↔Q3 双轨）",
         source="results/consistency/checks.json; results/supplementary/consistency_dual.json",
         align="llllll", build=t10),
    dict(key="T11", slug="variance", title="感官方差的序贯分解",
         source="results/supplementary/variance_decomposition_overall.csv; variance_decomposition_by_category.csv",
         align="lrrrrrrrrrr", build=t11),
    dict(key="T12", slug="robustness", title="稳健性检查汇总",
         source="results/robustness/summary.json; results/composite/per_target.csv",
         align="llll", build=t12),
    dict(key="S01", slug="q1_per_target", title="补充：问题一 50 维逐维性能",
         source="results/q1/metrics.json", align="lrrrrrrrr", build=s01, long=True),
    dict(key="S02", slug="transform", title="补充：231 个化学特征的变换与有效样本",
         source="results/data_gate/report.json", align="llllrr", build=s02, long=True),
    dict(key="S03", slug="q2_members", title="补充：问题二共表达模块成员",
         source="results/q2/communities.csv", align="lrrr", build=s03, long=True),
    dict(key="S04", slug="q3_pairs", title="补充：问题三风格对迁移（松约束）",
         source="results/q3_pairs/pairwise.csv", align="llrrrrrrrrr", build=s04, long=True),
    dict(key="S05", slug="q3_aroma", title="补充：Pils/Lager → Tripel 香气 OAV 倍数",
         source="results/supplementary/q3_aroma_oav_ratio.csv", align="lrrr", build=s05),
    dict(key="S06", slug="q1_latent", title="补充：潜状态替代方案比较",
         source="results/supplementary/upgrade/q1_latent_delta_ci.csv", align="llrrrr", build=s06),
    dict(key="S07", slug="q2_sensitivity", title="补充：问题二稳定性选择 α 敏感性",
         source="results/q2_sensitivity/alpha_sensitivity.csv; top3_overlap.csv",
         align="rrrrrrr", build=s07),
    dict(key="S08", slug="external", title="补充：外部风格数据核验",
         source="results/supplementary/external_abv_ibu.csv", align="llrrr", build=s08),
    dict(key="S09", slug="robustness_full", title="补充：稳健性扫描完整结果",
         source="results/robustness/summary.json", align="llrrrrrrr", build=s09, long=True),
    dict(key="S10", slug="importance_top25", title="补充：化合物置换重要性前 25",
         source="results/q1/importance.csv", align="rl lrrr".replace(" ", ""), build=s10),
    dict(key="S11", slug="q3_defect_chem", title="补充：缺陷化合物 OAV 与配对检验",
         source="results/supplementary/q3_defect_chemical_summary.csv",
         align="llrrrrrrrrrrrr", build=s11, long=True, size="scriptsize"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", help="只生成指定表，如 T01 S09")
    args = ap.parse_args()
    wanted = {k.upper() for k in args.keys}
    index = []
    for spec in SPECS:
        if wanted and spec["key"].upper() not in wanted:
            continue
        print(f"[{spec['key']}] {spec['title']}")
        df, note = spec["build"]()
        index.append(write_table(spec, df, note))

    idx = OUT / "INDEX.md"
    lines = ["# 论文表格索引", "",
             "所有表格由 `scripts/make_tables.py` 从 `results/` 与 `data/raw/` 生成；"
             "CSV 为核对版，`.tex` 为 booktabs 骨架，正文与表注由作者撰写。", "",
             "| 表号 | 标题 | 行 × 列 | 源数据 | CSV | LaTeX |",
             "|---|---|---|---|---|---|"]
    for it in index:
        lines.append(
            f"| {it['key']} | {it['title']} | {it['rows']} × {it['cols']} | "
            f"`{it['source']}` | [{Path(it['csv']).name}]({Path(it['csv']).name}) | "
            f"[{Path(it['tex']).name}]({Path(it['tex']).name}) |")
    lines += ["", "## 表注与使用提醒", ""]
    for it in index:
        lines.append(f"**{it['key']}** — {it['note']}")
        lines.append("")
    idx.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n索引写入 {idx.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
