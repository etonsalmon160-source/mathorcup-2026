"""模型融合与门控。

    python -m src.fuse --config src/configs/baseline.yaml

三种组合方式同台比较，**用同一套折**，所以数字可比：

1. **最好的单模型** —— 参照基线
2. **简单平均** —— 所有专家等权
3. **门控/堆叠** —— 用专家的**样本外预测（OOF）**训一个小门控器，学"该信谁、信多少"

G5 护栏：**门控必须赢过最好的单模型**，赢不了就退回单模型。
不是为了"看起来高级"而融合 —— 融合如果没赢，论文里写它就是自曝其短。

## 附带的"分群分析"（创新点的落地）

专家之间本来就不一样：有的在容易的样本上准，有的在难的样本上准。
我们用**专家间的预测分歧**当作"这条样本有多难"的尺子，分档看谁最强 ——
如果某一档里最优专家会换人，那就是**门控有存在必要**的直接证据，
也是论文里"样本异质性 / 亚群"那一节的实证来源（生物侧=亚型，经济侧=异质性处理效应）。

产出 `results/<名>/fusion.json`、`fusion_experts.csv`、`gating_analysis.csv`。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from scipy.special import softmax
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PATHS, SEED, get_logger, load_config, run_dir, set_global_seed
from src.cv import improvement, make_folds, primary_metric, score
from src.data import load_table
from src.models import detect_task, fit_predict

log = get_logger("fuse")

# 默认专家团：**必须异构**（同质专家堆一起没有意义）。
# 三个树模型 + 一个线性模型 —— 线性模型虽然弱，但它犯错的方式和别人不一样，
# 这正是融合能赚到钱的地方。
DEFAULT_EXPERTS = [
    {"name": "lgb", "model": "lightgbm",
     "params": {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 31}},
    {"name": "cat", "model": "catboost",
     "params": {"iterations": 600, "learning_rate": 0.05, "depth": 6}},
    {"name": "xgb", "model": "xgboost",
     "params": {"n_estimators": 600, "learning_rate": 0.05, "max_depth": 6}},
    {"name": "linear", "model": "linear"},
]


def build_linear(task: str, seed: int):
    """线性专家。树模型吃不了 NaN 和类别列，线性模型更吃不了 —— 得自己拼预处理。"""
    if task == "classification":
        return LogisticRegression(max_iter=2000, random_state=seed)
    return Ridge()


def make_preprocessor(X: pd.DataFrame):
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    return Pipeline([
        ("prep", _make_column_prep(num, cat)),
        ("scale", StandardScaler()),
    ])


def _make_column_prep(num: list, cat: list) -> ColumnTransformer:
    """树模型吃不了 NaN 和类别列，线性模型更吃不了 —— 这里自己拼一套预处理。

    注意 `sparse_output=False`：OneHot 默认出稀疏矩阵，而 StandardScaler 需要中心化，
    稀疏矩阵中心化会直接报错。`max_categories` 给列数封顶，防止高基数类别炸内存。
    """
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5,
                              max_categories=50, sparse_output=False), cat),
    ])


def fit_expert(expert: dict, task: str, seed: int, X_tr, y_tr, X_va):
    """训一个专家并预测验证集。返回 (预测值, 概率或 None)。"""
    name = expert["model"]
    if name == "linear":
        prep = make_preprocessor(X_tr)
        Xt = prep.fit_transform(X_tr)
        Xv = prep.transform(X_va)
        m = build_linear(task, seed)
        m.fit(Xt, y_tr)
        if task == "classification":
            return m.predict(Xv), m.predict_proba(Xv)
        return m.predict(Xv), None
    return fit_predict(name, expert.get("params") or {}, task, seed, X_tr, y_tr, X_va)


def _gate_features(X: pd.DataFrame, max_dim: int = 30) -> np.ndarray:
    """门控器看得懂的样本特征：取数值列，中位数填补，标准化，最多 max_dim 维。

    维度要压低——门控参数是 K×(d+1) 个，d 太大门控会过拟合，
    小数据上尤其明显。
    """
    num = X.select_dtypes(include=np.number)
    if num.shape[1] == 0:
        return np.zeros((len(X), 0))
    num = num.loc[:, num.notna().any()].copy()
    if num.shape[1] > max_dim:          # 先按方差筛，方差为 0 的本来也没用
        keep = num.var().sort_values(ascending=False).head(max_dim).index
        num = num[keep]
    vals = SimpleImputer(strategy="median").fit_transform(num.to_numpy(dtype=float))
    return StandardScaler().fit_transform(vals)


def fit_gate(z: np.ndarray, F: np.ndarray, y: np.ndarray, task: str,
             *, l2: float = 1e-3) -> np.ndarray:
    """拟合门控参数 theta。w_k(x) = softmax_k( A_k·φ(x) + b_k )。"""
    from scipy.optimize import minimize

    K, d = z.shape[1], F.shape[1]
    Fb = np.hstack([F, np.ones((len(F), 1))])       # 常数列当偏置
    y = np.asarray(y, dtype=float)

    def loss(theta):
        W = softmax(theta.reshape(K, d + 1) @ Fb.T, axis=0)     # (K, n)
        p = np.einsum("kn,kn->n", W, z.T)
        if task == "classification":
            p = np.clip(p, 1e-7, 1 - 1e-7)
            obj = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        else:
            obj = np.mean((y - p) ** 2)
        return obj + l2 * np.sum(theta ** 2)

    # 从"所有专家等权"起步：小数据上随机初始化方差很大
    res = minimize(loss, np.zeros(K * (d + 1)), method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x


def apply_gate(z_tr, F_tr, y_tr, task, z_va, F_va) -> tuple[np.ndarray, np.ndarray]:
    """在训练折上学门控，再打到验证折上。返回 (验证样本的门控权重 [n,K], 组合预测)。"""
    theta = fit_gate(z_tr, F_tr, y_tr, task)
    K, d = z_tr.shape[1], F_tr.shape[1]
    Fb = np.hstack([F_va, np.ones((len(F_va), 1))])
    W = softmax(theta.reshape(K, d + 1) @ Fb.T, axis=0)          # (K, n_va)
    pred = np.einsum("kn,kn->n", W, z_va.T)
    return W.T, pred


def expert_matrix(task: str, preds: list, probas: list) -> np.ndarray:
    """把各专家的输出拼成门控器的输入矩阵。

    二分类/回归 → 每个专家一列；多分类 → 每个专家一批概率列。
    """
    cols = []
    for p, pr in zip(preds, probas):
        if pr is not None and pr.ndim == 2 and pr.shape[1] > 2:
            cols.append(pr)
        elif pr is not None and pr.ndim == 2:
            cols.append(pr[:, 1:2])
        else:
            cols.append(np.asarray(p, dtype=float).reshape(-1, 1))
    return np.hstack(cols)


def main() -> None:
    ap = argparse.ArgumentParser(description="融合与门控")
    ap.add_argument("--config", required=True)
    ap.add_argument("--no-linear", action="store_true", help="去掉线性专家（数据很大时省时间）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(SEED)
    out_dir = run_dir(cfg)
    name = cfg["name"]

    tr_path = PATHS["processed"] / f"{name}_train.parquet"
    if not tr_path.exists():
        raise FileNotFoundError(f"缺 {tr_path}。先跑 make train。")
    X = pd.read_parquet(tr_path)

    y_raw = load_table(cfg["data"]["train"])[cfg["data"]["target"]].reset_index(drop=True)
    task = detect_task(y_raw)
    if task == "classification":
        classes = sorted(y_raw.dropna().unique().tolist(), key=str)
        y = y_raw.map({c: i for i, c in enumerate(classes)}).astype(int)
        n_class = len(classes)
    else:
        y = y_raw.astype(float)
        n_class = 0

    pm = primary_metric(task, cfg)
    seed = (cfg.get("seeds") or [SEED])[0]
    cv_cfg = cfg.get("cv") or {}

    experts = cfg.get("fusion", {}).get("experts") or DEFAULT_EXPERTS
    if args.no_linear:
        experts = [e for e in experts if e["model"] != "linear"]
    log.info("专家团：%s", [e["name"] for e in experts])

    folds, order = make_folds(X, y, {"cv": {**cv_cfg, "seed": seed}})
    Xs, ys = X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)

    # ---- 1. 每个专家跑一遍 OOF ----
    oof_preds, oof_probas, expert_scores = [], [], []
    for ex in experts:
        pred = np.zeros(len(ys), dtype=float)
        proba = np.zeros((len(ys), n_class)) if n_class else None
        for tr, va in folds:
            p, pr = fit_expert(ex, task, seed, Xs.iloc[tr], ys.iloc[tr], Xs.iloc[va])
            pred[va] = p
            if proba is not None and pr is not None:
                proba[va] = pr
        m = score(ys, pred, task, proba)
        expert_scores.append(m[pm])
        oof_preds.append(pred)
        oof_probas.append(proba)
        log.info("  专家 %-7s %s = %.5f", ex["name"], pm, m[pm])

    Z = expert_matrix(task, oof_preds, oof_probas)
    names = [e["name"] for e in experts]

    best_idx = int(np.argmax(expert_scores)) if pm not in ("rmse", "mae", "mse", "mape") \
        else int(np.argmin(expert_scores))
    best_single = expert_scores[best_idx]

    # ---- 2. 简单平均 ----
    if task == "classification":
        avg_proba = np.mean([p for p in oof_probas], axis=0)
        avg_pred = np.argmax(avg_proba, axis=1)
        avg_score = score(ys, avg_pred, task, avg_proba)[pm]
    else:
        avg_pred = np.mean(oof_preds, axis=0)
        avg_score = score(ys, avg_pred, task)[pm]

    # ---- 3a. 线性堆叠（只看专家预测）——作为对照，说明"不做门控会怎样" ----
    stack_pred = np.zeros(len(ys), dtype=float)
    stack_proba = np.zeros((len(ys), n_class)) if n_class else None
    for tr, va in folds:
        meta = (LogisticRegression(max_iter=2000, random_state=seed)
                if task == "classification" else Ridge())
        meta.fit(Z[tr], ys[tr])
        if task == "classification":
            stack_proba[va] = meta.predict_proba(Z[va])
            stack_pred[va] = meta.predict(Z[va])
        else:
            stack_pred[va] = meta.predict(Z[va])
    stack_score = score(ys, stack_pred, task, stack_proba)[pm]

    # ---- 3b. 软门控（看输入特征）——这才是 Mixture of Experts ----
    gate_score, gate_pred, gate_W = None, None, None
    if task == "classification" and n_class > 2:
        log.warning("多分类暂不做软门控（权重表会退化成无意义），看线性堆叠的结果")
    else:
        F = _gate_features(X, max_dim=30)
        z_cols = (np.array([pr[:, 1] for pr in oof_probas]) if task == "classification"
                  else np.array(oof_preds)).T                        # (n, K)
        yv = ys.to_numpy()
        n, K = z_cols.shape

        gate_pred = np.full(n, np.nan)
        gate_W = np.full((n, K), np.nan)
        for tr, va in folds:   # 逐折：在训练折上学门控 → 打到验证折，不泄漏
            W_va, p_va = apply_gate(z_cols[tr], F[tr], yv[tr], task,
                                    z_cols[va], F[va])
            gate_pred[va] = p_va
            gate_W[va] = W_va

        if task == "classification":
            gate_score = score(ys, (gate_pred >= 0.5).astype(int), task,
                               np.column_stack([1 - gate_pred, gate_pred]))[pm]
        else:
            gate_score = score(ys, gate_pred, task)[pm]

    rows = [{"方案": f"最好单模型（{names[best_idx]}）", pm: round(best_single, 6)},
            {"方案": "简单平均", pm: round(avg_score, 6)},
            {"方案": "线性堆叠（只看预测）", pm: round(stack_score, 6)}]
    if gate_score is not None:
        rows.append({"方案": "软门控（看特征）", pm: round(gate_score, 6)})
    log.info("最好单模型 %s=%.5f | 简单平均=%.5f | 线性堆叠=%.5f | 软门控=%s",
             names[best_idx], best_single, avg_score, stack_score,
             f"{gate_score:.5f}" if gate_score is not None else "未启用")

    # ---- 4. 分群分析一：按专家间分歧分档，看谁在哪档最强 ----
    lower_better = pm in ("rmse", "mae", "mse", "mape")
    hard = [(pr[:, 1] if pr.shape[1] == 2 else np.argmax(pr, axis=1))
            for pr in oof_probas] if task == "classification" else oof_preds
    diff = np.std(np.vstack(hard), axis=0)
    yv = ys.to_numpy()
    err_all = np.vstack([
        np.abs(yv - (pr[:, 1] >= 0.5).astype(float)) if task == "classification"
        else np.abs(yv - p)
        for p, pr in zip(oof_preds, oof_probas)])
    err = err_all[best_idx]          # 用最好单模型的误差做参照

    q = pd.Series(pd.qcut(diff, 4, labels=["最容易", "较容易", "较难", "最难"],
                          duplicates="drop"))
    gating_rows = []
    for lab in q.cat.categories:
        sel = (q == lab).to_numpy()
        if sel.sum() == 0:
            continue
        per = {nm: float(err_all[k][sel].mean()) for k, nm in enumerate(names)}
        row = {"难度分层": str(lab), "样本数": int(sel.sum())}
        row.update({f"{nm}_误差": round(v, 5) for nm, v in per.items()})
        row["该层最优专家"] = min(per, key=per.get)
        gating_rows.append(row)
    gating_df = pd.DataFrame(gating_rows)

    # ---- 4b. 分群分析二：门控权重本身 = 样本亚群（论文的"亚群识别"就靠这张表）----
    # 每条样本一个 K 维权重向量，按"谁占主导"归类，看每类的规模和权重均值。
    sub_df = None
    if gate_W is not None:
        dom = np.argmax(np.nan_to_num(gate_W, nan=-1), axis=1)
        rows_sub = []
        for k, nm in enumerate(names):
            sel = dom == k
            if sel.sum() == 0:
                continue
            row = {"主导专家": nm, "样本数": int(sel.sum()),
                   "占比": round(float(sel.mean()), 4)}
            for j, n2 in enumerate(names):
                row[f"权重均值_{n2}"] = round(float(np.nanmean(gate_W[sel, j])), 4)
            rows_sub.append(row)
        sub_df = pd.DataFrame(rows_sub)
        sub_df.to_csv(out_dir / "gating_subgroups.csv", index=False, encoding="utf-8-sig")

    # ---- 5. 该不该融合（G5 护栏）----
    # 在三种组合方式里挑最好的那个，再去跟最好单模型比 ——
    # 不是只拿门控去比（门控输了但堆叠赢了，那也该用堆叠）。
    candidates = {"简单平均": avg_score, "线性堆叠": stack_score}
    if gate_score is not None:
        candidates["软门控"] = gate_score
    pick = min(candidates, key=candidates.get) if lower_better else max(candidates, key=candidates.get)
    chosen = candidates[pick]
    beats = (chosen < best_single) if lower_better else (chosen > best_single)

    result = {
        "primary_metric": pm,
        "lower_is_better": lower_better,
        "experts": {nm: round(s, 6) for nm, s in zip(names, expert_scores)},
        "best_single": round(float(best_single), 6),
        "best_single_name": names[best_idx],
        "simple_average": round(float(avg_score), 6),
        "linear_stacking": round(float(stack_score), 6),
        "gated": None if gate_score is None else round(float(gate_score), 6),
        "chosen_combiner": pick,
        "chosen_score": round(float(chosen), 6),
        "gain_over_best_single": round(
            float(best_single - chosen) if lower_better else float(chosen - best_single), 6),
        "fused_beats_best_single": bool(beats),
        "expert_names": names,
    }
    (out_dir / "fusion.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "fusion_experts.csv", index=False, encoding="utf-8-sig")
    gating_df.to_csv(out_dir / "gating_analysis.csv", index=False, encoding="utf-8-sig")

    # ---- 人看的结论 ----
    print(f"\n=== 融合对比（主指标 {pm}，越小越好={pm in ('rmse','mae','mse')}）===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n=== 分群分析（按专家间分歧分档）===")
    print(gating_df.to_string(index=False))
    churn = gating_df["该层最优专家"].nunique() if len(gating_df) else 1
    if churn > 1:
        print(f"\n✓ 不同难度档里最优专家会换人（{churn} 个不同专家）——**门控有存在必要**。")
        print("  这可以直接写进论文的样本异质性一节。")
    else:
        print("\n· 各档最优专家是同一个 —— 门控的收益可能有限，实话实说。")

    print()
    if "软门控" in candidates and gate_score is not None and stack_score is not None:
        if gate_score > stack_score and not lower_better or gate_score < stack_score and lower_better:
            print("⚠ 本机上**软门控没有赢过线性堆叠**。这不奇怪：门控要学 K×(d+1) 个参数，")
            print("  小数据上比 stacking（只学 K 个系数）更容易过拟合。")
            print("  论文里要如实写这一点 —— 别硬说门控更好。")
    if beats:
        print(f"✓ 最优组合方式「{pick}」{chosen:.5f} 赢过最好单模型 {best_single:.5f}"
              f"（{names[best_idx]}）⇒ G5 通过。")
        print(f"  相对提升 {abs(best_single - chosen):.5f}。")
    else:
        print(f"✗ 三种组合方式里最好的「{pick}」{chosen:.5f} 仍**没有**赢过"
              f"最好单模型 {best_single:.5f}（{names[best_idx]}）。")
        print(f"  ⇒ 按 G5 护栏，**退回单模型 {names[best_idx]}**，别为了好看硬做融合。")
    print(f"\n产物 -> {out_dir}/fusion.json · fusion_experts.csv · gating_analysis.csv · gating_subgroups.csv")


if __name__ == "__main__":
    main()
