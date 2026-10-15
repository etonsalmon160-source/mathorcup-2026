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


def main() -> None:
    ap = argparse.ArgumentParser(description="对抗验证（训练集 vs 测试集）")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-splits", type=int, default=5)
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

    imp_df = (pd.DataFrame({"特征": cols, "重要度": imp / args.n_splits})
              .sort_values("重要度", ascending=False).reset_index(drop=True))
    imp_df["重要度占比"] = imp_df["重要度"] / max(imp_df["重要度"].sum(), 1e-12)
    imp_df.to_csv(out_dir / "adversarial_features.csv", index=False, encoding="utf-8-sig")

    result = {
        "auc": round(auc, 4),
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
