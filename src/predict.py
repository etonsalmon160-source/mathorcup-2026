"""预测并生成提交文件。

    python -m src.predict --config src/configs/baseline.yaml

依赖 src.train 先跑过（模型和测试集特征都在那儿）。输出 results/<配置名>/submission.csv。
分类任务会额外输出概率列——阈值怎么定、误判代价不对称时怎么取舍，都要靠它（经济哥的场子）。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import PATHS, get_logger, load_config, run_dir, set_global_seed
from src.data import load_table
from src.models import MODEL_SUFFIX, load_model, predict_loaded

log = get_logger("predict")


def main() -> None:
    ap = argparse.ArgumentParser(description="生成提交文件")
    ap.add_argument("--config", required=True)
    ap.add_argument("--threshold", type=float, default=None,
                    help="二分类判定阈值，默认 0.5；按业务代价调整（经济哥）")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg.get("seed", 42))
    out_dir = run_dir(cfg)

    test_feat_path = PATHS["processed"] / f"{cfg['name']}_test.parquet"
    if not test_feat_path.exists():
        raise FileNotFoundError(
            f"找不到测试集特征 {test_feat_path}。先跑：python -m src.train --config {args.config}"
        )
    X_test = pd.read_parquet(test_feat_path)

    meta_path = out_dir / "metrics.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"找不到 {meta_path}。先跑 src.train。")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    task, classes = meta["task"], meta.get("classes")
    model_name = meta["model"]

    model_path = PATHS["models"] / f"{cfg['name']}{MODEL_SUFFIX.get(model_name, '.txt')}"
    model = load_model(model_name, task, model_path)

    pred, proba = predict_loaded(model, model_name, task, X_test)
    log.info("预测完成：%d 条", len(pred))

    # ---- 组装提交表 ----
    sub_cfg = cfg.get("submission", {})
    id_col = sub_cfg.get("id_col") or cfg["data"].get("id_col")
    pred_col = sub_cfg.get("pred_col", "prediction")

    out = pd.DataFrame()
    if id_col:
        raw = load_table(cfg["data"]["test"])
        if id_col not in raw.columns:
            raise KeyError(f"ID 列 {id_col!r} 不在测试集里：{list(raw.columns)[:20]}")
        out[id_col] = raw[id_col].to_numpy()
    else:
        log.warning("配置里没给 id 列，提交文件用行号占位——**多半不合规，开赛后改成真实 ID**")
        out["row_id"] = np.arange(len(pred))

    if task == "classification":
        if proba is not None and proba.ndim == 2 and proba.shape[1] == 2:
            out["proba"] = proba[:, 1]
            thr = args.threshold if args.threshold is not None else 0.5
            labels_idx = (proba[:, 1] >= thr).astype(int)
            if classes is not None:
                out[pred_col] = [classes[i] for i in labels_idx]
            else:
                out[pred_col] = labels_idx
            log.info("二分类阈值 %.3f：正类 %d 条（%.1f%%）",
                     thr, int(labels_idx.sum()), labels_idx.mean() * 100)
        elif proba is not None:
            for j in range(proba.shape[1]):
                out[f"proba_{classes[j] if classes else j}"] = proba[:, j]
            out[pred_col] = [classes[i] for i in pred] if classes is not None else pred
        else:
            out[pred_col] = [classes[i] for i in pred] if classes is not None else pred
    else:
        out[pred_col] = pred

    path = out_dir / "submission.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    log.info("提交文件 -> %s", path)
    print(f"\n提交文件：{path}\n形状：{out.shape}\n前 5 行：\n{out.head().to_string()}")


if __name__ == "__main__":
    main()
