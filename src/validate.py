"""对抗验证：训练集和测试集到底是不是同一个分布？

    python -m src.validate --config src/configs/baseline.yaml      # 或 make validate

**为什么这个必须做**：本地交叉验证的分数能不能信，全看训练集和测试集像不像。
把两边的行混在一起，训练一个模型去猜"这行来自哪边"：

- 猜不出来（AUC ≈ 0.5）→ 两边同分布，**本地 CV 的分数可以信**。
- 一猜就中（AUC 明显 > 0.5）→ 两边不同分布，本地 CV **高估**了真实成绩，
  而且论文里必须处理这件事，不能装作没发生。

在生信里这是"批次效应"的标准检出手段，思路完全一样。

产出 `results/<名>/adversarial.json` 和 `adversarial_features.csv`
（后者告诉你**是哪些字段在漂移** —— 这直接决定下一步怎么补救）。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import PATHS, SEED, get_logger, load_config, run_dir, set_global_seed
from src.models import build_model

log = get_logger("validate")

AUC_THRESHOLD = 0.60   # 超过就算漂移明显，G3 会拦下


def _c2st_test(X: pd.DataFrame, y: np.ndarray, n_perm: int = 200) -> dict:
    """分类器两样本检验（C2ST）：给"两边是否同分布"一个 p 值。

    为什么不能只报 AUC：**AUC 多大才算"有漂移"是拍脑袋定的**（我们用 0.6，别人可能用 0.7）。
    正式检验给的是 p 值 —— "如果两边真的同分布，观察到这么高的准确率的概率有多大"。

    方法（照 C2ST 论文的做法）：
      留出法 —— 一半数据训分类器，另一半算准确率。
      零假设 H0：两边同分布 ⇒ 留出准确率 ≈ 0.5。

    给两个 p 值：
    - **二项检验**：C2ST 论文说零分布"很简单"，留出集上正确数 ~ Binomial(n, 0.5)。
    - **置换检验**：把标签打乱重训 n_perm 次，看真实准确率在经验零分布里的位置。
      **这是我们自己加的**（更稳健，不依赖二项近似）。
      ⚠️ 注意：**置换 p 值不是 C2ST 那篇论文提出的**，别挂在它头上。
    """
    from scipy import stats

    rng = np.random.default_rng(SEED)

    # 🔴 **必须先配平两类**，否则准确率这个统计量是废的。
    # 训练集 1200 条 / 测试集 300 条时，对半切后什么都不学、只猜多数类，
    # 准确率就有 0.8 —— 于是"零假设下准确率≈0.5"这个前提根本不成立，
    # 二项检验会给出荒谬的小 p 值，和 AUC 的结论直接打架。
    # （这个 bug 是实测跑出来的，不是推出来的。）
    n_min = int(min((y == 0).sum(), (y == 1).sum()))
    keep = np.r_[rng.choice(np.where(y == 0)[0], n_min, replace=False),
                 rng.choice(np.where(y == 1)[0], n_min, replace=False)]
    Xb, yb = X.iloc[keep].reset_index(drop=True), y[keep]

    idx = rng.permutation(len(yb))
    half = len(yb) // 2
    tr, te = idx[:half], idx[half:]

    def acc_on(tr_idx, te_idx, labels):
        m = build_model("lightgbm", {"n_estimators": 300, "learning_rate": 0.05,
                                     "num_leaves": 31}, "classification", SEED)
        m.fit(Xb.iloc[tr_idx], labels[tr_idx])
        pred = m.predict(Xb.iloc[te_idx])
        return float((pred == labels[te_idx]).mean())

    obs_acc = acc_on(tr, te, yb)
    n_te = len(te)
    k = int(round(obs_acc * n_te))

    # 二项零分布（C2ST 论文的"简单零分布"）
    p_binom = float(stats.binom.sf(k - 1, n_te, 0.5))

    # 置换零分布（打乱"来自哪边"的标签，重训）
    perm = []
    for _ in range(n_perm):
        y_shuf = rng.permutation(yb)
        perm.append(acc_on(tr, te, y_shuf))
    perm = np.array(perm)
    # ⚠ 置换 p 值有**分辨率下限** 1/(n_perm+1)：60 次置换时最小只能报到 0.016。
    #   想要更小的 p，就得加大置换次数（200 次的floor是 0.005）。
    #   二项检验没有这个限制，所以极显著时两者的数值会差很多 —— 这是正常的，
    #   不是矛盾：一个受次数限制，一个受分布假设限制。**看结论是否一致，别看数值是否相等。**
    p_perm = float((perm >= obs_acc).sum() + 1) / (n_perm + 1)

    return {
        "holdout_accuracy": round(obs_acc, 4),
        "n_holdout": n_te,
        "n_balanced": int(len(yb)),
        "balanced_note": f"已配平两类各 {n_min} 条（否则不平衡会放大准确率）",
        "p_binomial": p_binom,
        "p_permutation": round(p_perm, 4),
        "permutation_mean": round(float(perm.mean()), 4),
        "verdict": ("两边同分布（不能拒绝 H0）" if p_perm > 0.05
                    else "两边**不同分布**（p < 0.05，拒绝 H0）"),
        "note": "C2ST 见 Lopez-Paz & Oquab, ICLR 2017 (arXiv:1610.06545)；"
                "置换检验为本实现自行添加，非该论文内容。",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="对抗验证（训练集 vs 测试集）")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-permutations", type=int, default=200,
                    help="置换检验次数（默认 200；调大更稳但更慢）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(SEED)
    out_dir = run_dir(cfg)
    name = cfg["name"]

    tr_path = PATHS["processed"] / f"{name}_train.parquet"
    te_path = PATHS["processed"] / f"{name}_test.parquet"
    if not tr_path.exists() or not te_path.exists():
        raise FileNotFoundError(
            f"缺特征文件（{tr_path.name} / {te_path.name}）。先在配置里给 test 路径并跑 src.train。"
        )
    X_tr = pd.read_parquet(tr_path)
    X_te = pd.read_parquet(te_path)

    # 只保留两边都有的列，且顺序一致
    cols = [c for c in X_tr.columns if c in X_te.columns]
    X = pd.concat([X_tr[cols], X_te[cols]], axis=0, ignore_index=True)
    y = np.r_[np.zeros(len(X_tr), dtype=int), np.ones(len(X_te), dtype=int)]
    log.info("对抗验证：训练 %d 行 vs 测试 %d 行，%d 个特征",
             len(X_tr), len(X_te), len(cols))

    if y.mean() < 0.05 or y.mean() > 0.95:
        log.warning("两边样本量差得太多（测试占比 %.1f%%），AUC 可能不稳", y.mean() * 100)

    folds = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    imp = np.zeros(len(cols))

    for i, (tr, va) in enumerate(folds.split(X, y), 1):
        model = build_model("lightgbm", {"n_estimators": 300, "learning_rate": 0.05,
                                         "num_leaves": 31}, "classification", SEED)
        model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        imp += model.booster_.feature_importance(importance_type="gain")
        log.info("  第 %d/%d 折完成", i, args.n_splits)

    auc = float(roc_auc_score(y, oof))
    ok = auc < AUC_THRESHOLD

    # ---- 正式统计检验（不只是报一个 AUC）----
    # 这其实就是学界说的 **分类器两样本检验（Classifier Two-Sample Test, C2ST）**
    # —— Lopez-Paz & Oquab, "Revisiting Classifier Two-Sample Tests", ICLR 2017
    #    (arXiv:1610.06545)。做法正是"一边标正、一边标负，训分类器"。
    # 零假设 H0：两边同分布 ⇒ 留出集上的准确率应当**接近随机水平 0.5**。
    test = _c2st_test(X, y, args.n_permutations)
    log.info("C2ST 检验：留出准确率 %.4f，p = %.4g（置换），p = %.4g（二项）",
             test["holdout_accuracy"], test["p_permutation"], test["p_binomial"])

    imp_df = (pd.DataFrame({"特征": cols, "重要度": imp / args.n_splits})
              .sort_values("重要度", ascending=False).reset_index(drop=True))
    imp_df["重要度占比"] = imp_df["重要度"] / max(imp_df["重要度"].sum(), 1e-12)
    imp_df.to_csv(out_dir / "adversarial_features.csv", index=False, encoding="utf-8-sig")

    result = {
        "auc": round(auc, 4),
        "c2st": test,
        "threshold": AUC_THRESHOLD,
        "ok": ok,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "top_drifting_features": imp_df.head(10)["特征"].tolist(),
    }
    (out_dir / "adversarial.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 人看的结论 ----
    print(f"\n对抗验证 AUC = {auc:.4f}（阈值 {AUC_THRESHOLD}）")
    print(f"\n【正式统计检验 · C2ST】留出准确率 {test['holdout_accuracy']:.4f}"
          f"（随机水平 0.5，留出 {test['n_holdout']} 条）")
    print(f"  p = {test['p_permutation']:.4g}（置换检验，{args.n_permutations} 次）")
    print(f"  p = {test['p_binomial']:.4g}（二项检验，C2ST 论文的零分布）")
    print(f"  → {test['verdict']}")
    print(f"  注：置换检验是本实现自行添加，非 C2ST 原论文内容")
    if ok:
        print("→ 判读：训练集和测试集**分不开**，两边同分布。**本地 CV 的分数可以信。**")
    else:
        print("→ 判读：**两边分布不同**。这意味着：")
        print("   ① 本地 CV 很可能高估了测试集成绩；")
        print("   ② 论文里必须写明这一点，并说明怎么处理（重加权 / 去掉漂移特征 / 域适应）；")
        print("   ③ G3 门不过，先处理再往下走。")
    print("\n漂移最明显的 10 个特征（看这些字段是不是「事后信息」或时间相关）：")
    for _, r in imp_df.head(10).iterrows():
        print(f"   {r['特征']:<32s} {r['重要度占比']:.1%}")
    print(f"\n明细 -> {out_dir / 'adversarial_features.csv'}")


if __name__ == "__main__":
    main()
