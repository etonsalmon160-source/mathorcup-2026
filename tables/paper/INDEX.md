# 论文表格索引

所有表格由 `scripts/make_tables.py` 从 `results/` 与 `data/raw/` 生成；CSV 为核对版，`.tex` 为 booktabs 骨架，正文与表注由作者撰写。

| 表号 | 标题 | 行 × 列 | 源数据 | CSV | LaTeX |
|---|---|---|---|---|---|
| T01 | 数据与质量总览 | 11 × 3 | `data/raw/附件/1.csv, 4.csv; results/data_gate/report.json; results/qc/report.json` | [T01_data_overview.csv](T01_data_overview.csv) | [T01_data_overview.tex](T01_data_overview.tex) |
| T02 | 问题一：单模型交叉验证比较 | 14 × 8 | `results/q1/comparison.csv` | [T02_q1_models.csv](T02_q1_models.csv) | [T02_q1_models.tex](T02_q1_models.tex) |
| T03 | 问题一：融合与门控方案比较 | 16 × 8 | `results/composite/report.json` | [T03_q1_fusion.csv](T03_q1_fusion.csv) | [T03_q1_fusion.tex](T03_q1_fusion.tex) |
| T04 | 问题一：特征组消融 | 10 × 8 | `results/q1/ablation.csv` | [T04_q1_ablation.csv](T04_q1_ablation.csv) | [T04_q1_ablation.tex](T04_q1_ablation.tex) |
| T05 | 问题二：共表达网络与稳定性选择 | 14 × 3 | `results/q2/report.json; results/q2/stability.json` | [T05_q2_network.csv](T05_q2_network.csv) | [T05_q2_network.tex](T05_q2_network.tex) |
| T06 | 问题二：共表达模块结构与富集 | 8 × 7 | `results/q2/communities.csv; results/q1/importance_modules.csv; results/supplementary/q2_module_annotation.csv` | [T06_q2_modules.csv](T06_q2_modules.csv) | [T06_q2_modules.tex](T06_q2_modules.tex) |
| T07 | 问题二：各感官维度的 Top-3 关键化合物 | 50 × 7 | `results/q2/top3.csv` | [T07_q2_top3.csv](T07_q2_top3.csv) | [T07_q2_top3.tex](T07_q2_top3.tex) |
| T08 | 问题三：风格迁移方法比较 | 10 × 11 | `results/supplementary/upgrade/q3_qp.json; q3_qp_long.csv; q3_qp_pairs_long.csv` | [T08_q3_methods.csv](T08_q3_methods.csv) | [T08_q3_methods.tex](T08_q3_methods.tex) |
| T09 | 问题三：五个监测感官维度的劣化 | 18 × 5 | `results/supplementary/upgrade/q3_qp_pairs_long.csv` | [T09_q3_defects.csv](T09_q3_defects.csv) | [T09_q3_defects.tex](T09_q3_defects.tex) |
| T10 | 三问互证检查（Q2↔Q3 双轨） | 11 × 6 | `results/consistency/checks.json; results/supplementary/consistency_dual.json` | [T10_consistency.csv](T10_consistency.csv) | [T10_consistency.tex](T10_consistency.tex) |
| T11 | 感官方差的序贯分解 | 6 × 11 | `results/supplementary/variance_decomposition_overall.csv; variance_decomposition_by_category.csv` | [T11_variance.csv](T11_variance.csv) | [T11_variance.tex](T11_variance.tex) |
| T12 | 稳健性检查汇总 | 14 × 4 | `results/robustness/summary.json; results/composite/per_target.csv` | [T12_robustness.csv](T12_robustness.csv) | [T12_robustness.tex](T12_robustness.tex) |
| S01 | 补充：问题一 50 维逐维性能 | 50 × 9 | `results/q1/metrics.json` | [S01_q1_per_target.csv](S01_q1_per_target.csv) | [S01_q1_per_target.tex](S01_q1_per_target.tex) |
| S02 | 补充：231 个化学特征的变换与有效样本 | 231 × 6 | `results/data_gate/report.json` | [S02_transform.csv](S02_transform.csv) | [S02_transform.tex](S02_transform.tex) |
| S03 | 补充：问题二共表达模块成员 | 223 × 4 | `results/q2/communities.csv` | [S03_q2_members.csv](S03_q2_members.csv) | [S03_q2_members.tex](S03_q2_members.tex) |
| S04 | 补充：问题三风格对迁移（松约束） | 462 × 11 | `results/q3_pairs/pairwise.csv` | [S04_q3_pairs.csv](S04_q3_pairs.csv) | [S04_q3_pairs.tex](S04_q3_pairs.tex) |
| S05 | 补充：Pils/Lager → Tripel 香气 OAV 倍数 | 7 × 4 | `results/supplementary/q3_aroma_oav_ratio.csv` | [S05_q3_aroma.csv](S05_q3_aroma.csv) | [S05_q3_aroma.tex](S05_q3_aroma.tex) |
| S06 | 补充：潜状态替代方案比较 | 10 × 6 | `results/supplementary/upgrade/q1_latent_delta_ci.csv` | [S06_q1_latent.csv](S06_q1_latent.csv) | [S06_q1_latent.tex](S06_q1_latent.tex) |
| S07 | 补充：问题二稳定性选择 α 敏感性 | 4 × 7 | `results/q2_sensitivity/alpha_sensitivity.csv; top3_overlap.csv` | [S07_q2_sensitivity.csv](S07_q2_sensitivity.csv) | [S07_q2_sensitivity.tex](S07_q2_sensitivity.tex) |
| S08 | 补充：外部风格数据核验 | 4 × 5 | `results/supplementary/external_abv_ibu.csv` | [S08_external.csv](S08_external.csv) | [S08_external.tex](S08_external.tex) |
| S09 | 补充：稳健性扫描完整结果 | 29 × 9 | `results/robustness/summary.json` | [S09_robustness_full.csv](S09_robustness_full.csv) | [S09_robustness_full.tex](S09_robustness_full.tex) |
| S10 | 补充：化合物置换重要性前 25 | 25 × 6 | `results/q1/importance.csv` | [S10_importance_top25.csv](S10_importance_top25.csv) | [S10_importance_top25.tex](S10_importance_top25.tex) |
| S11 | 补充：缺陷化合物 OAV 与配对检验 | 42 × 14 | `results/supplementary/q3_defect_chemical_summary.csv` | [S11_q3_defect_chem.csv](S11_q3_defect_chem.csv) | [S11_q3_defect_chem.tex](S11_q3_defect_chem.tex) |

## 表注与使用提醒

**T01** — “缺失”按原始 1.csv/4.csv 直接计数；下限哨兵值比例按 250 × 231 化学单元格计。风格检验为 Kruskal–Wallis 后 BH-FDR；质量标记的定义见 results/qc/report.json。

**T02** — 每格为 3 个随机种子（42/43/44）的均值 ± 标准差；随机CV 与分组CV 的划分不同，两列不能直接做配对检验。R² 为 50 个感官维度的平均 OOF R²。

**T03** — R² 为 3 个种子预测取平均后的 OOF R²；因此单模型数值与表 T02 的逐种子均值略有差异。Δ、CI、p 均以同划分下的 Extra Trees 为参照；CI 为按啤酒聚类的 Bootstrap 95% 区间，p 为逐维 Wilcoxon 符号秩检验。Oracle 使用真实标签选方案，只作上界，不可采纳。

**T04** — Δ = 去掉该特征组后的平均 OOF R² − 完整模型；CI 为 3 个种子 × 50 维的 Bootstrap 区间。“有贡献或中性”表示 CI 未显示明确损失，不等同于该组必要。

**T05** — 主表只收正向激动剂（Lasso positive、coef > 0、Spearman > 0）；负向掩蔽物另表。稳定性为 5 次随机分半、两半各自 Top-3 的 Jaccard；γ 事先固定为 0.20，未按测试结果调节。

**T06** — 富集检验只针对化合物来源标签（Hops / Malt / Yeast / Wild flora / Others），q 为 BH-FDR；ME3 的酵母富集是唯一通过 FDR 的模块。代表性化合物按 Q1 置换重要性排序。

**T07** — Top-3 为正向激动剂；稳定率为 300 次分层 Bootstrap 中进入 Top-3 的频率，分半 Jaccard 为 5 次随机分半的中位数。“低稳定”= 稳定率 < 0.60，共 24 维，论文中只能作探索性结论。

**T08** — “新增”指从阈值以下升到阈值以上；“任一维劣化”是 stale_hops、DMS、diacetyl、barnyard、metallic 五个感官维度中至少一个变差。风格位移与 QP 成本均为中位数，越小越接近源风格 / 约束越弱。

**T09** — Δ = 约束后 − 约束前的风格内标准化感官分；Δ > 0 表示该缺陷维度变差。五个维度是预注册的监测维度，未使用测试集或生成结果反向挑选。

**T10** — Q2↔Q3 按两种口径并列：有符号 Lasso 把正、负效应都计入入选频率，平均 ρ = 0.1621（28/50 维 ≥ 0.15，通过）；正向 Lasso 即 Q2 主结果，平均 ρ = 0.1225（15/50 维，未通过）。两口径的 Spearman 中位数同号，Top-20 重合均显著富集于随机期望（超几何检验 p = 0.021）。主结果 results/q2/top3.csv 仍是纯正向激动剂，未因本表改动。Q1↔Q3 取自 results/consistency/checks.json；Q2↔Q3 取自 results/supplementary/consistency_dual.json（同 Bootstrap 分层、权重、候选池与图平滑）。checks.json 中旧的 Q2↔Q3 数值（平均 0.1723）早于 Q2 重跑，已不可复现，不再引用。此表不修改 results/q1–q3 或 results/composite。

**T11** — 序贯残差分解：先只用风格均值，再加入 8 维感官潜状态，最后加入 Extra Trees。Δ 为外层 OOF 上的增量；负值表示该层没有增益，未做截断。占比按三层正增量归一。

**T12** — 种子波动用主融合模型 avg 在 50 维上的逐维 SD 中位数；k 敏感性用化学潜状态 + Ridge；缺失压力为完全随机缺失，不代表真实检出限机制；λ 扫描只到 0.50，未外推。

**S01** — Q1 主模型（Extra Trees）的 50 维逐维指标；R² 为 3 个种子的均值。分组 CV 按风格留出，评估对未见风格的迁移。

**S02** — log10 表示在 log10 尺度建模；raw 为原始尺度。判定依据见 data_gate/report.json：负值证明、校准表、峰面积推断三类证据。

**S03** — 223 个节点为通过候选池与排除规则的化合物；模块编号与正文、图 7 一致。

**S04** — 每个源风格 → 目标风格对取 12 个最近样本（不足 12 则取全部），表中只列松约束（loose），严格约束（strict）见 CSV 原始文件。

**S05** — OAV = 浓度 / 阈值；倍数 > 1 表示该香气在 Tripel 中相对增强。仅列 7 个有明确阈值的香气化合物。

**S06** — 以 14 维化学 PCA 为参照，比较因子分析（FA）与偏最小二乘（PLS）潜状态的增量；正增量维数指 50 维中 Δ > 0 的维数。

**S07** — 主分析取 α = 0.06；α = 0.04 更宽松、0.08 更严格。Top-3 Jaccard 为与主表 Top-3 的重合度。

**S08** — competition 为赛道 A 数据；external 为独立公开风格数据。用于核验风格层面的酒精度/苦度方向，不参与建模。

**S09** — k 与缺失压力用化学潜状态 + Ridge / Extra Trees；Pareto 的 λ 只扫到 0.50。表中数值与 results/robustness/summary.json 一一对应。

**S10** — 置换重要性：打乱该化合物后模型 MSE 的增量（按维标准化后求和）；只列前 25 名，完整 232 行见 results/q1/importance.csv。

**S11** — 11 个 Pils/Lager 样本的配对检验；约束列 none/loose/strict 对应无约束/松/严。阈值来自 data_gate/report.json 的校准阈值；OAV 与超标分析见正文。
