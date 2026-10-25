# 本文件由 AI 辅助编写；建模与算法设计决策由参赛队员做出。详见《AI工具使用详情说明》。
"""数据门 D0–D6：在建模之前把数据的结构与口径钉死。

D0 结构    行数、主键、命名对照、矩阵形状
D1 缺失    每表缺失分层；哪些是结构性空列
D2 变换    231 列的 log10/原始量纲，对照契约文件，防止口径漂移
D3 离群    IQR 稳健标记，只报告不删除
D4 阈值    缺陷物的单位与阈值能否与数据量级对上
D5 自洽    用 1/4 表复算 2/5/6 三个相关矩阵，必须逐元素一致
D6 契约    D2/D4 的结论与 docs/data_contract/ 下已提交版本是否一致

设计原则：
- 只读 data/raw/，不写任何中间数据；报告落在 results/data_gate/。
- 硬门失败就停（退出码 1）；D1/D3 只报告，不拦。
- 口径变了必须显式 --bless 刷新契约，避免"悄悄改分类"。
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "附件"
CONTRACT = ROOT / "docs" / "data_contract"
REPORT_JSON = ROOT / "results" / "data_gate" / "report.json"
TRANSFORM_V1 = CONTRACT / "transform_classification_v1.csv"
UNITS_V1 = CONTRACT / "units_and_thresholds_v1.json"

META = ["beer", "beer_id", "tasting_category_fine"]
REQUIRED_DEFECTS = ["Dimethyl sulfide", "Diacetyl", "4-vinylguaiacol"]
UNVERIFIED_DEFECTS = ["Trans-2-nonenal"]

TRANSFORM_TOL = 1e-6
CORR_TOL = 1e-6
UNIT_RATIO_MIN = 1e-3
UNIT_RATIO_MAX = 1e3


# ---------------------------------------------------------------- 工具

def fix_mojibake(s) -> str:
    """8.xlsx 里希腊字母是 UTF-8 被按 Latin-1 解码后的乱码，这里还原。"""
    try:
        return str(s).encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return str(s)


def key(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _num(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return x


@dataclass
class Gate:
    code: str
    title: str
    passed: bool
    summary: str
    hard: bool = True
    details: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "title": self.title,
            "passed": self.passed,
            "hard": self.hard,
            "summary": self.summary,
            "details": self.details,
        }


# ---------------------------------------------------------------- 读取

@dataclass
class Tables:
    chem: pd.DataFrame
    sens: pd.DataFrame
    comments: pd.DataFrame
    corr_chem: pd.DataFrame
    corr_sens: pd.DataFrame
    corr_cross: pd.DataFrame
    thresholds: pd.DataFrame
    sensory_names: pd.DataFrame
    calibrated: pd.DataFrame


def load_tables() -> Tables:
    xl = RAW / "8.xlsx"
    return Tables(
        chem=pd.read_csv(RAW / "1.csv"),
        sens=pd.read_csv(RAW / "4.csv"),
        comments=pd.read_csv(RAW / "7.csv"),
        corr_chem=pd.read_csv(RAW / "2.csv", index_col=0),
        corr_sens=pd.read_csv(RAW / "5.csv", index_col=0),
        corr_cross=pd.read_csv(RAW / "6.csv", index_col=0),
        thresholds=pd.read_excel(xl, sheet_name="Sensory thresholds"),
        sensory_names=pd.read_excel(xl, sheet_name="Sensory names"),
        calibrated=pd.read_excel(xl, sheet_name="Calibrated chemical dataset"),
    )


def chem_columns(t: Tables) -> list[str]:
    return list(t.chem.columns[len(META):])


def sens_columns(t: Tables) -> list[str]:
    return list(t.sens.columns[len(META):])


def build_maps(t: Tables) -> tuple[dict[str, str], dict[str, str]]:
    """1.csv 列 → 2.csv 化学名；4.csv 列 → 5.csv 感官名。"""
    th = t.thresholds.copy()
    th["chemical_name"] = th.chemical_name.map(fix_mojibake)
    th = th.dropna(subset=["field_name_R"]).drop_duplicates("field_name_R")
    by_r = {key(r.field_name_R): r.chemical_name for r in th.itertuples()}
    chem_map = {c: by_r.get(key(c)) for c in chem_columns(t)}
    missing = sorted(c for c, v in chem_map.items() if not v)
    if missing:
        raise SystemExit(f"🔴 化学名对照缺失 {len(missing)} 列：{missing[:10]}")

    sn = t.sensory_names.dropna(subset=["R_name"]).drop_duplicates("R_name").set_index("R_name")
    sens_map = {
        c: str(sn.loc[c, "CleanName_PanelFeedback"]).replace("_x000D_", "\r")
        for c in sens_columns(t)
    }
    return chem_map, sens_map


# ---------------------------------------------------------------- D0

def gate_structure(t: Tables, chem_map: dict, sens_map: dict) -> tuple[Gate, list[str]]:
    details: list[str] = []
    problems: list[str] = []

    shapes = {
        "1.csv": t.chem.shape,
        "2.csv": t.corr_chem.shape,
        "4.csv": t.sens.shape,
        "5.csv": t.corr_sens.shape,
        "6.csv": t.corr_cross.shape,
        "7.csv": t.comments.shape,
    }
    details.append("形状：" + "，".join(f"{k} {v[0]}×{v[1]}" for k, v in shapes.items()))

    if t.chem.shape[0] != 250 or t.sens.shape[0] != 250 or t.comments.shape[0] != 250:
        problems.append("三张样本表不是各 250 行")
    if len(chem_columns(t)) != 231:
        problems.append(f"1.csv 化合物列应为 231，实际 {len(chem_columns(t))}")
    if len(sens_columns(t)) != 50:
        problems.append(f"4.csv 感官列应为 50，实际 {len(sens_columns(t))}")

    beers = t.chem["beer"]
    if beers.nunique() != 250 or sorted(beers) != list(range(1, 251)):
        problems.append("1.csv 的 beer 不是 1–250 连续唯一主键")
    same_order = (
        (t.sens["beer"].to_numpy() == beers.to_numpy()).all()
        and (t.comments["beer"].to_numpy() == beers.to_numpy()).all()
    )
    details.append(f"主键 beer：1–250 连续唯一；1/4/7 行序{'一致' if same_order else '不一致'}")
    if not same_order:
        problems.append("1/4/7 的 beer 行序不一致，不能按行号直接对齐")

    trunc4 = float((t.sens["beer_id"] % 1000 == 0).mean())
    trunc7 = float((t.comments["beer_id"] % 1000 == 0).mean())
    if trunc4 > 0.5 or trunc7 > 0.5:
        details.append(
            f"⚠️ 4/7 的 beer_id 末三位已截断（末三位为 0 的比例 {trunc4:.0%}/{trunc7:.0%}），"
            "不能用来跨表连接；一律用 beer 列"
        )

    for name, m in [("2.csv", t.corr_chem), ("5.csv", t.corr_sens)]:
        sq = m.shape[0] == m.shape[1]
        sym = float(np.nanmax(np.abs(m.to_numpy() - m.to_numpy().T)))
        diag = float(np.nanmax(np.abs(np.diag(m.to_numpy()) - 1.0)))
        details.append(f"{name}：{'方阵' if sq else '非方阵'}，对称误差 {sym:.1e}，对角偏差 {diag:.1e}")
        if not sq or sym > 1e-12 or diag > 1e-12:
            problems.append(f"{name} 不是单位对角的对称矩阵")
    if not set(t.corr_cross.index) <= set(chem_map.values()):
        problems.append("6.csv 的行名不是 2.csv 化学名的子集")
    else:
        details.append(f"6.csv：{len(t.corr_cross.index)} 化学物 × {len(t.corr_cross.columns)} 感官")

    if sorted(chem_map.values()) != sorted(t.corr_chem.index):
        problems.append("1.csv 映射后的化学名与 2.csv 标签不相等")
    else:
        details.append("命名对照：231/231 化学物完全匹配")

    if sorted(sens_map.values()) != sorted(t.corr_sens.index):
        problems.append("4.csv 映射后的感官名与 5.csv 标签不相等")
    else:
        details.append("命名对照：50/50 感官维度完全匹配")

    agg = [
        (c, v) for c, v in chem_map.items()
        if "total" in v.lower() or c.lower().endswith("sum") or c == "kcalperc"
    ]
    if agg:
        details.append("聚合列（不是单一化合物，Top-3 结论需排除）：" + "，".join(f"{c}→{v}" for c, v in agg))

    y = t.sens[sens_columns(t)]
    details.append(
        f"感官目标范围 {y.min().min():.2f}–{y.max().max():.2f}，"
        f"列均值 {y.mean().min():.3f}–{y.mean().max():.3f}，列标准差 {y.std().min():.3f}–{y.std().max():.3f}"
        "（不是原始 0–6 分，预测时不要裁剪到 0–6）"
    )

    return Gate(
        "D0", "结构", not problems, "结构核对通过" if not problems else "；".join(problems),
        details=details,
    ), problems


# ---------------------------------------------------------------- D1

def gate_missing(t: Tables) -> Gate:
    details: list[str] = []
    feat = chem_columns(t)
    rate = t.chem[feat].isna().mean()
    tiers = {
        "无缺失": int((rate == 0).sum()),
        "(0,5%]": int(((rate > 0) & (rate <= 0.05)).sum()),
        "(5%,30%]": int(((rate > 0.05) & (rate <= 0.30)).sum()),
        "(30%,100%]": int(((rate > 0.30) & (rate < 1)).sum()),
        "全空": int((rate == 1).sum()),
    }
    details.append("1.csv 缺失分层：" + "，".join(f"{k} {v} 列" for k, v in tiers.items()))
    details.append(f"1.csv 缺失总数 {int(t.chem[feat].isna().sum().sum())} 格，涉及 {int((rate > 0).sum())} 列")
    cols = rate[rate > 0].sort_values(ascending=False)
    if len(cols):
        details.append("缺失列：" + "，".join(f"{c} {rate[c]:.1%}" for c in cols.index))

    sens_na = int(t.sens[sens_columns(t)].isna().sum().sum())
    details.append(f"4.csv 缺失 {sens_na} 格" + ("（完整）" if sens_na == 0 else ""))
    empty = [c for c in t.comments.columns[len(META):] if t.comments[c].isna().all()]
    details.append(f"7.csv 全空列 {len(empty)} 个：{empty}")
    others = int(t.comments.drop(columns=empty).iloc[:, len(META):].isna().sum().sum())
    details.append(f"7.csv 除全空列外缺失 {others} 格")

    ok = sens_na == 0 and others == 0
    return Gate("D1", "缺失", ok, "缺失结构已登记（不拦门）", hard=False, details=details)


# ---------------------------------------------------------------- D2

def classify_transforms(t: Tables, chem_map: dict) -> pd.DataFrame:
    cal = t.calibrated
    rows = []
    for c in chem_columns(t):
        v = t.chem[c]
        n = int(v.notna().sum())
        if c in cal.columns:
            a = v.to_numpy(float)
            b = cal[c].to_numpy(float)
            ok = ~np.isnan(a) & ~np.isnan(b) & (b > 0)
            dlog = float(np.median(np.abs(a[ok] - np.log10(b[ok]))))
            draw = float(np.median(np.abs(a[ok] - b[ok])))
            if dlog < TRANSFORM_TOL:
                transform, source = "log10", "calibrated_log10"
            elif draw < TRANSFORM_TOL:
                transform, source = "raw", "calibrated_raw"
            else:
                transform, source = "unknown", "calibrated_ambiguous"
        elif (v < 0).any():
            transform, source = "log10", "negative_proves_log"
        else:
            transform, source = "log10", "peak_area_inferred"
        rows.append({
            "column": c,
            "chemical_name": chem_map[c],
            "transform": transform,
            "source": source,
            "n_valid": n,
            "min": _num(v.min()),
            "median": _num(v.median()),
            "max": _num(v.max()),
        })
    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


def gate_transform(trans: pd.DataFrame) -> Gate:
    counts = trans["transform"].value_counts().to_dict()
    src = trans["source"].value_counts().to_dict()
    details = [
        "变换分类：" + "，".join(f"{k} {v} 列" for k, v in counts.items()),
        "证据等级：" + "，".join(f"{k} {v} 列" for k, v in src.items()),
    ]
    unknown = trans.loc[trans["transform"] == "unknown", "column"].tolist()
    if unknown:
        details.append(f"无法判定：{unknown}")
    inferred = trans.loc[trans["source"] == "peak_area_inferred", "column"].tolist()
    if inferred:
        details.append(
            f"⚠️ {len(inferred)} 列只能按量级推断为 log10（全正、未校准的 GC-MS 峰面积），"
            "论文里要写成推断而不是实测"
        )
    ok = not unknown
    return Gate("D2", "变换", ok, "231 列变换分类完成" if ok else "存在无法判定的列", details=details)


# ---------------------------------------------------------------- D3

def gate_outliers(t: Tables) -> Gate:
    feat = chem_columns(t)
    flags = pd.DataFrame(False, index=t.chem.index, columns=feat)
    for c in feat:
        v = t.chem[c]
        q1, q3 = v.quantile(0.25), v.quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr <= 0:
            continue
        flags[c] = (v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)
    per_sample = flags.sum(axis=1)
    top = per_sample.sort_values(ascending=False).head(5)
    details = [
        f"3×IQR 标记 {int(flags.to_numpy().sum())} 格 / {flags.size} 格"
        f"（{flags.to_numpy().mean():.1%}），每样本中位 {int(per_sample.median())} 列",
    ]
    details.append("极端样本：" + "，".join(
        f"beer={t.chem.loc[i, 'beer']}({t.chem.loc[i, 'tasting_category_fine']}) {int(n)}列"
        for i, n in top.items()
    ))
    details.append("列标记最多：" + "，".join(
        f"{c} {int(n)}" for c, n in flags.sum().sort_values(ascending=False).head(5).items()
    ))
    details.append("只报告不删除：是否为真离群需结合品类判断，建模阶段再用树模型/稳健损失处理")
    return Gate("D3", "离群", True, "离群已登记（不拦门）", hard=False, details=details)


# ---------------------------------------------------------------- D4

def build_units(t: Tables, chem_map: dict, trans: pd.DataFrame) -> dict:
    col_of = {key(c): c for c in chem_columns(t)}
    kind = dict(zip(trans["column"], trans["transform"]))
    th = t.thresholds.copy()
    th["chemical_name_fixed"] = th.chemical_name.map(fix_mojibake)

    all_rows = []
    for r in th[th.threshold_mean.notna()].itertuples():
        col = col_of.get(key(r.field_name_R))
        if col is None:
            continue
        all_rows.append({
            "chemical_name": r.chemical_name_fixed,
            "column": col,
            "transform": kind[col],
            "unit": r.unit,
            "threshold_mean": _num(r.threshold_mean),
            "threshold_median": _num(r.threshold_median),
            "calibrated": None if pd.isna(r.calibrated) else r.calibrated,
            "off_flavors": None if pd.isna(r.off_flavors) else r.off_flavors,
            "origin_paper": _num(r.origin_paper),
        })
    all_rows.sort(key=lambda x: x["chemical_name"].lower())

    by_name = {r["chemical_name"]: r for r in all_rows}
    markers = {}
    for name in REQUIRED_DEFECTS:
        markers[name] = by_name.get(name, {"status": "missing"})
    not_used = {}
    for name in UNVERIFIED_DEFECTS:
        not_used[name] = {
            "column": "X.E..2.nonenal",
            "status": "missing_threshold",
            "reason": "8.xlsx 阈值表没有该化合物的阈值，不纳入缺陷门；论文里只描述浓度不升高",
        }
    return {
        "source": "data/raw/附件/8.xlsx / Sensory thresholds",
        "threshold_choice": "median",
        "defect_markers": markers,
        "not_used": not_used,
        "all_thresholds": all_rows,
    }


def gate_thresholds(t: Tables, units: dict, trans: pd.DataFrame) -> Gate:
    kind = dict(zip(trans["column"], trans["transform"]))
    details: list[str] = []
    problems: list[str] = []
    for name in REQUIRED_DEFECTS:
        rec = units["defect_markers"][name]
        if rec.get("status") == "missing":
            problems.append(f"{name} 找不到阈值")
            continue
        col = rec["column"]
        v = t.chem[col].dropna()
        med = float(10 ** v.median()) if kind[col] == "log10" else float(v.median())
        thr = rec["threshold_median"]
        ratio = med / thr if thr else float("nan")
        details.append(
            f"{name}（{col}，{rec['unit']}）：数据中位 {med:.3g} / 阈值 {thr:.3g} = {ratio:.2f}"
        )
        if not (UNIT_RATIO_MIN <= ratio <= UNIT_RATIO_MAX):
            problems.append(f"{name} 的量级与阈值差三个数量级以上，单位可能不一致")
    if "Trans-2-nonenal" in units["not_used"]:
        details.append("⚠️ Trans-2-nonenal：官方阈值表无阈值，缺陷门不覆盖，只做浓度不升高检查")
    return Gate(
        "D4", "阈值", not problems,
        "三个缺陷物的单位与阈值可对上" if not problems else "；".join(problems),
        details=details,
    )


# ---------------------------------------------------------------- D5

def gate_correlations(t: Tables, chem_map: dict, sens_map: dict) -> Gate:
    X = t.chem[chem_columns(t)].rename(columns=chem_map)
    Y = t.sens[sens_columns(t)].rename(columns=sens_map)

    got2 = X[list(t.corr_chem.index)].corr(method="spearman")
    d2 = float(np.nanmax(np.abs(got2.to_numpy() - t.corr_chem.to_numpy())))

    got5 = Y[list(t.corr_sens.index)].corr(method="spearman")
    d5 = float(np.nanmax(np.abs(got5.to_numpy() - t.corr_sens.to_numpy())))

    sub = list(t.corr_cross.index)
    sc = list(t.corr_cross.columns)
    r = spearmanr(X[sub].to_numpy(float), Y[sc].to_numpy(float), axis=0, nan_policy="omit").correlation
    got6 = r[: len(sub), len(sub):]
    d6 = float(np.nanmax(np.abs(got6 - t.corr_cross.to_numpy())))

    details = [
        f"2.csv 复算最大误差 {d2:.1e}（231×231）",
        f"5.csv 复算最大误差 {d5:.1e}（50×50）",
        f"6.csv 复算最大误差 {d6:.1e}（{len(sub)}×{len(sc)}）",
        "说明：1/4 表与三个相关矩阵逐元素一致，行对齐与命名映射均正确",
    ]
    ok = max(d2, d5, d6) <= CORR_TOL
    return Gate(
        "D5", "自洽", ok,
        "三个相关矩阵均可由原始表复算" if ok else "相关矩阵与原始表不一致",
        details=details,
    )


# ---------------------------------------------------------------- D6

def _write_contract(trans: pd.DataFrame, units: dict) -> None:
    CONTRACT.mkdir(parents=True, exist_ok=True)
    trans[["column", "chemical_name", "transform", "source", "n_valid"]].to_csv(
        TRANSFORM_V1, index=False, encoding="utf-8"
    )
    stable = {k: v for k, v in units.items()}
    UNITS_V1.write_text(
        json.dumps(stable, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _units_stable(units: dict) -> dict:
    """契约只比较源数据派生字段；运行时统计（_data_median 等）不算。"""
    def clean(d):
        return {k: v for k, v in d.items() if not str(k).startswith("_")}
    return {
        "threshold_choice": units["threshold_choice"],
        "defect_markers": {k: clean(v) for k, v in units["defect_markers"].items()},
        "not_used": units["not_used"],
        "all_thresholds": units["all_thresholds"],
    }


def gate_contract(trans: pd.DataFrame, units: dict) -> Gate:
    if not TRANSFORM_V1.exists() or not UNITS_V1.exists():
        return Gate(
            "D6", "契约", False,
            "缺少 docs/data_contract/ 下的契约文件",
            details=["先跑 python scripts/data_gate.py --bless，并提交 docs/data_contract/"],
        )
    problems: list[str] = []
    old = pd.read_csv(TRANSFORM_V1)
    new = trans[["column", "chemical_name", "transform", "source", "n_valid"]]
    cols = ["column", "chemical_name", "transform", "source"]
    merged = old[cols].merge(new[cols], on="column", how="outer", suffixes=("_old", "_new"), indicator=True)
    if (merged["_merge"] != "both").any():
        problems.append("列集合与契约不一致（数据文件被换过？）")
    diff = merged[
        (merged["_merge"] == "both")
        & (
            (merged["chemical_name_old"] != merged["chemical_name_new"])
            | (merged["transform_old"] != merged["transform_new"])
            | (merged["source_old"] != merged["source_new"])
        )
    ]
    if len(diff):
        problems.append(f"{len(diff)} 列的变换分类与契约不一致：" + "，".join(diff["column"].head(8)))

    old_units = json.loads(UNITS_V1.read_text(encoding="utf-8"))
    if _units_stable(old_units) != _units_stable(units):
        problems.append("阈值/单位契约发生变化")

    if problems:
        return Gate(
            "D6", "契约", False, "数据口径与已提交契约不一致",
            details=problems + ["若确认是官方更新：python scripts/data_gate.py --bless，并在 docs/决策记录.md 记一笔"],
        )
    return Gate("D6", "契约", True, "变换与阈值契约未漂移", details=[
        f"契约：{TRANSFORM_V1.relative_to(ROOT)}、{UNITS_V1.relative_to(ROOT)}",
    ])


# ---------------------------------------------------------------- 主流程

def _gates(t: Tables, chem_map: dict, sens_map: dict,
           trans: pd.DataFrame, units: dict) -> list[Gate]:
    return [
        gate_structure(t, chem_map, sens_map)[0],
        gate_missing(t),
        gate_transform(trans),
        gate_outliers(t),
        gate_thresholds(t, units, trans),
        gate_correlations(t, chem_map, sens_map),
        gate_contract(trans, units),
    ]


def evaluate() -> tuple[list[Gate], pd.DataFrame, dict]:
    """只算不打印，供 gate_check.py 调用。"""
    t = load_tables()
    chem_map, sens_map = build_maps(t)
    trans = classify_transforms(t, chem_map)
    units = build_units(t, chem_map, trans)
    return _gates(t, chem_map, sens_map, trans, units), trans, units


def run(bless: bool = False, report_path: Path | None = None) -> int:
    t = load_tables()
    chem_map, sens_map = build_maps(t)
    trans = classify_transforms(t, chem_map)
    units = build_units(t, chem_map, trans)

    if bless:
        _write_contract(trans, units)
        print(f"✅ 已刷新契约：{TRANSFORM_V1.relative_to(ROOT)} / {UNITS_V1.relative_to(ROOT)}")
        print("   请提交这两个文件，并在 docs/决策记录.md 记录口径变更。")

    gates = _gates(t, chem_map, sens_map, trans, units)

    print("=== 数据门 D0–D6 ===")
    for g in gates:
        mark = "✅" if g.passed else ("🔴" if g.hard else "🟡")
        print(f"{g.code} {g.title:<4s} {mark} {g.summary}")
        for d in g.details:
            print(f"     · {d}")

    hard_fail = [g for g in gates if g.hard and not g.passed]
    total = len(gates)

    out = REPORT_JSON if report_path is None else report_path
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gates": [g.as_dict() for g in gates],
        "transform_counts": trans["transform"].value_counts().to_dict(),
        "transform_sources": trans["source"].value_counts().to_dict(),
        "transform_table": trans.to_dict(orient="records"),
        "thresholds": units["all_thresholds"],
        "defect_markers": units["defect_markers"],
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"\n结果：{total} 门中 {sum(g.passed for g in gates)} 通过，{len(hard_fail)} 个硬门失败")
    print(f"报告：{out.relative_to(ROOT)}")
    return 0 if not hard_fail else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="数据门 D0–D6")
    ap.add_argument("--bless", action="store_true", help="刷新并覆盖 docs/data_contract/ 契约")
    ap.add_argument("--report", default="", help="报告输出路径（默认 results/data_gate/report.json）")
    args = ap.parse_args()
    path = Path(args.report) if args.report else None
    return run(bless=args.bless, report_path=path)


if __name__ == "__main__":
    raise SystemExit(main())
