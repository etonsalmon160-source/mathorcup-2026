# MathorCup 2026 大数据竞赛 — 参赛工作区

> 第七届 MathorCup 数学应用挑战赛 · 大数据竞赛
> 主办：中国优选法统筹法与经济数学研究会

## 关键时间

| 事项 | 时间 | 状态 |
|---|---|---|
| 报名截止 | **2026-10-23 12:00** | ⏰ 还剩 8 天 |
| **初赛** | **2026-10-23 18:00 → 10-30 20:00**（7 天） | 主战场 |
| 复赛 | 2026-12-04 18:00 → 12-11 20:00 | 前 10% 晋级 |
| 获奖公布 | 2027-01 | |

## 硬约束（从《章程》摘出来的，逐条都别踩）

- **每队 ≤ 3 人** + ≤ 1 名指导教师（我们是 3 人满编）。
- 赛题共 **2 道，任选 1 道**，开赛时在官网和报名网站同时公布。
- **每天最多提交 1 次**。→ 所以有 `submissions/REGISTRY.md` 台账，每次提交前必须登记。
- 提交内容三件套：**电子版论文 + 结果数据集 + 计算程序源代码**。
  支撑材料（程序/数据/结果）压缩打包上传；源程序**即使放进了论文附录，也要单独再交一份**。
- 评分四轴：①论文撰写质量 ②**代码完整性/正确性/效率** ③测试集指标（准确率、召回率等）
  ④**创新性 + 跨学科交叉应用能力**。
- 参赛费 200 元/队；组别按队内最高在读学历定（研究生组/本科组/专科组）。
- 允许引用公开文献，须注明来源；禁止抄袭。

## 团队分工（3 人）

| 角色 | 负责 | 主要产出 |
|---|---|---|
| **系统生物学 / 组学**（你） | 领域理解、特征工程、生物学可解释性、论文主线 | 特征设计、结果解读、论文摘要与结论 |
| **CS** | 数据管线、建模与调参、代码工程化、复现保障 | `src/`、`scripts/`、可运行代码包 |
| **经济学** | 业务/经济视角、指标体系设计、方案价值论证、创新点包装 | 问题建模、评价指标、论文的创新性论证 |

详见 [docs/角色分工.md](docs/角色分工.md)。

## 目录约定

```
data/raw/          原始数据（只读，永不修改）        ← 不入库
data/interim/      中间产物                          ← 不入库
data/processed/    建模用特征表                      ← 不入库
src/               可复用的 Python 代码
src/configs/       实验配置文件（yaml）
notebooks/         探索性分析（结论要回填到 src/）
results/           跑出来的指标/预测                   ← 不入库
figures/           出图                                ← 入库
submissions/       提交台账 REGISTRY.md + 归档
paper/             LaTeX 论文
logs/              运行日志                             ← 不入库
docs/              赛制、分工、复现规范、备战计划
```

**大文件一律不入库**（见 `.gitignore`）；数据与模型只留在本地，备份走网盘。

## 常用命令

```bash
conda activate mathorcup          # 激活竞赛环境（Python 3.12）

make check                        # 环境自检：包版本 + 数据目录 + 复现性
make smoke                        # 用造假数据把整条流水线跑通（不碰真实数据）
make run-all                      # 从 data/raw 一路跑到提交文件
make paper                        # 编译论文 PDF
make clean                        # 清理中间产物
```

想分步跑（换个配置就加 `CONFIG=src/configs/xxx.yaml`）：

```bash
make data                         # 数据体检：缺失/分布/类别不平衡，先看清楚再动手
make train                        # 交叉验证 + 全量重训，报"均值 ± 标准差"
make predict                      # 出 results/<配置名>/submission.csv
make plots                        # 出图到 figures/<配置名>/
```

## 流水线怎么走（拿到数据后的顺序）

| 步骤 | 命令 | 产出 |
|---|---|---|
| 1. 填配置 | 复制 `src/configs/baseline.yaml` 改路径和目标列 | 一份实验配置 |
| 2. 看数据 | `make data CONFIG=...` | 数据体检报告（终端） |
| 3. 训模型 | `make train CONFIG=...` | `results/<名>/metrics.json`、特征重要度、模型 |
| 4. 出提交 | `make predict CONFIG=...` | `results/<名>/submission.csv` |
| 5. 出图 | `make plots CONFIG=...` | `figures/<名>/*.png`（直接进论文） |

每次训练都会往 `results/ledger.csv` 追加一行台账，对比不同实验看这个文件。
**换模型只改配置里 `model` 一行**，但注意三个库的参数名不通用——用 `src/configs/` 下对应的那份。

**生信队友从这里开始**：`src/features.py` 里的 `domain_features()` 是领域特征的入口，
机制驱动的特征写在那里，见 [docs/角色分工.md](docs/角色分工.md)。

## 每天的最低动作

1. 开新分支或直接改 `src/`，跑实验，**固定随机种子**（见 `docs/复现规范.md`）。
2. 结果登记进 `results/`，图表进 `figures/`。
3. 决定是否提交 → 若要，先写 `submissions/REGISTRY.md`，再上传，再归档到 `submissions/archive/`。
4. 晚上同步一次：谁改了哪块、明天交什么。

## 参考

- 官网：http://www.mathorcup.org
- 报名：https://new.saikr.com/vse/bigdata2026
