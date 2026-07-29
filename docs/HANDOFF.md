# HANDOFF — quantcontest2026 进展交接（2026-07-28）

> 新对话接手用。完整结论另见记忆 `quantcontest2026-next-steps.md`（已同步）。本文是精简现状 + 瓶颈 + 待决。

## 比赛与约束

- **任务**：预测匿名多标的时序的 `target`（风险调整未来表现）。指标 = 加权零均值 R² = `1 - Σw(y-ŷ)²/Σw·y²`，全零预测=0。
- **数据**：train 13.2M 行 / 9 parquet 分区 / 15 asset / 323 feature / 47 responder / float32。test 3.2M 行只有 row_id,time_id,asset_id,feature_*。train time_id 0→888479，test 从 888480 起（严格连续）。`data/manifest.json`。
- **强分布漂移**：train vs test adversarial AUC=1.0。但**单 feature 边缘分布几乎不漂移**（KS 中位 0.060，无一>0.5），漂移是**多变量联合结构**的。drop-drift 失败正因为没有单 feature 真正漂移。
- **推理环境**：私榜提交代码，4核/12GB/无GPU/无网络，Time-Series API 逐 time_id 释放 `predict(test)` 收 ~15 行，超时该 step 置 0。**公榜阶段提交 CSV，无推理限制（可用 GPU 任意算）**。
- **★ 最终成绩 = 私榜实盘 + 答辩，公榜只是气氛**。所以重心是"私榜 4核无GPU 能跑"的模型。公榜可重算但私榜跑不动的东西对最终成绩无用。
- 提交：公榜每天 5 次，私榜共 10 次策略提交。私榜截止 8/31。**公榜截止 8/23，之后发扩展训练数据（带公榜测试期 responder，多 ~3.2M 行）**——确定性增量，私榜可用。

## 分支与最佳成绩

| 分支 | 内容 | 公榜 |
|---|---|---|
| **asset-capacity** | GBDT K=5 per-asset 增量列三集成（**私榜可用最佳**） | **0.00310366** |
| feature-physics（当前） | 上面 + EDA m5 + 大 NN（train_nn.py / strategy_nn） | NN 单独 0.00281055；GBDT+NN 等权 0.00314700（公榜刷分，私榜无用） |
| breakthrough | asset_id categorical 三集成 | 0.00295844 |
| seed | 早期三集成 | 0.00288933 |

- **asset-capacity 已 push**。feature-physics 未 push（含 NN 代码，本地 commit 到 `f804c8b`）。
- **公榜刷分最佳 0.00314700**（GBDT K=5 + 大 NN 等权），但⚠️ NN 私榜跑不动，集成只公榜用，**对最终成绩无用**。
- **私榜可用最佳 = GBDT K=5 per-asset 增量列三集成 0.00310366**（asset-capacity 分支）。

## 核心规律（AUC=1.0 极端漂移下，必读）

**只有"固定 per-row 属性 / within-row 逐行变换"有效，任何 train 或 test 分布衍生的跨样本/跨时间结构都失效。**

- 活的：asset_id categorical（+2.4%）、per-asset topk5 增量列（within-row，+4.9%）。
- 死的（公榜实测反降/负）：cs 截面、rolling 时序、drop-drift、AV-reweight（-14%）、target rank（-0.0138）、最近数据训练、多種子 bagging（+0.000028≈0）、OOF stacking（+0.000027≈0）、per-asset 全独立模型（+0.7%但耗时25×超时）、DAE（-0.000067）、responder 两阶段堆叠/样本权重、transductive（机制判死未实现）。

## EDA 关键结论（out/eda/ + out/eda_full/）

- **合法信号天花板 ≈ 0.02**（单 feature 最强 |corr with target|=0.022，feature_014）。滞后相关证实：有队伍公榜刷 0.4 用了**时序偏移 feature（未来 feature 泄漏）**被撤回警告——feature_008 用下一 time_id 的 feature 相关 +0.21（同时刻仅 -0.021）。**0.4 全是未来泄漏，私榜逐 time_id 释放拿不到，违规**。合法侧（过去 feature）几乎无信号（±0.007）。我们 0.003 的根因是合法信号本身就弱。
- **responder 协方差稳定**（procrustes 0.9999 早/中/晚一致），responder_03 与 target 相关 0.817。但 responder 推理不可用（API 剥除）。作辅助信号/样本权重已公榜实测判死（abstop5 全量变负）。剩余价值仅在 8/23 回补数据（多数据）。
- **asset 间差异在 feature→target 斜率，不在 mean**：wyy_share≈w_sum_share（高 share asset 占分母多纯粹因 weight 大，比赛 weight 已正确倾斜，容量倾斜=扭曲指标=死）。per-asset mean 偏置天花板 0.014%。
- **feature 物理类型可反推**（out/eda/feature_physics.csv）：price_level 4 个（feature_301-304，per-asset 近常数 9.49/10.49/11.49/12.49）、return_like 34、volatility_like 7、volume_event 17、ratio_probability 60+。主办方确认 feature 含收益类、target 是收益+风险。但物理交互预筛判死（无交互对 target 相关超单 feature，GBDT 自己能学）。
- **最强信号 feature 恰恰最不漂移**（top-20 信号 mean KS=0.031 < 中位 0.060，corr=-0.477）。

## GBDT 当前最佳（asset-capacity 分支，0.00310366）

- **per-asset topk5 增量列**：每 asset 的 top-5 feature（来自 `out/eda_full/m3_per_asset_importance.csv`）加一列 `pa_{a}_{f}`（asset a 行取 feature f 值，其余 0），within-row drift-safe。323 raw + 75 pa + asset_id = 399 列。仍是单一共享模型（非 15 独立，避免 25× 耗时）。
- K 调参到顶：K=3=0.00306641 / **K=5=0.00310366** / K=8=0.00309791（山形 peak K=5）。去冗余（去普适 feature）有害 0.00308404（普适 feature masked 形式有独立价值，保留）。
- 训练：`train.py --backend {lgb,xgb,cat} --asset-as-categorical --per-asset-topk 5 --per-asset-importance-csv out/eda_full/m3_per_asset_importance.csv`（其余参数见记忆，注意 `--cs-topk 0 --rolling-windows 0 --interaction-topk 0` 必须显式，cs-topk 默认 25）。
- 三 backend 等权集成。推理 0 超时。parity 测试过（train polars vs 推理 numpy 逐位一致）。

## 大 NN（feature-physics 分支）— 活了但独立增量极小

- **train_nn.py**：Periodic feature embedding(PLR) + asset embedding + 4层残差MLP(945k参数) + LayerNorm + 监督加权MSE（无 reconstruction）。推理 strategy_nn/main.py 自包含 torch。
- **推翻了"NN 无信号"**：DAE 公榜 -0.000067 是架构问题（小网络+reconstruction），大 NN 单独公榜 **0.00281055**（正分）。用户判断"DAE 不能说明大 NN 无信号"正确。
- **NN 漂移鲁棒性更好**：公榜/CV=1.81（GBDT 1.55）。NN 在 time CV 后段 valid 保守输出 0（CV 0.0015 严重低估），但 test 上微弱信号有效（公榜 0.00281）。**判断 NN 只能看公榜，CV 不可信**。
- **★ 私榜 NN 跑不动**（4核无GPU，torch CPU 推理 945k×5折×21万切片超时）。当前公榜 NN 结果都是手动改 strategy_nn/main.py 硬编码 cpu→cuda 用 GPU 跑的。**NN 只能公榜 CSV 用，私榜必须 GBDT。**
- **NN 瓶颈诊断**：train_loss 卡 1.207≈target 方差（信号 0.002 量级，loss 下降幅度本就 ~0.002，视觉像卡住）。信噪比 1:600，NN 靠梯度学弱信号天生难。堆大（4.3M 参数 hidden512/periodic32）优化失败崩盘（CV 0.0001）。架构修复（feat/asset 分别 LayerNorm + mix init 0.2）无效，valid_r2 反略降。
- **★ NN 独立增量极小**：GBDT 0.003104，NN 0.002811，等权集成 0.003147。集成 > GBDT 仅 +0.000043 → **NN 的 GBDT-没有的独立信号 ~0.00004，绝大部分重叠**。蒸馏/学残差天花板就在 0.00004。
- **NN+pa 列有害**：加 per-asset topk5 后 NN 单独 0.00270 < 0.00281（asset embedding 已捕获 asset 身份，pa 稀疏列冗余）。NN 不加 pa 列。
- 未试的 NN 本质方向：attention（FT-Transformer 风格）、NN 学 GBDT 残差。但独立增量 0.00004 暗示天花板低。

## 瓶颈总结

1. **合法信号天花板 0.02**（0.4 是未来泄漏）。所有合法方法在这 0.02 里榨取，GBDT 已到 0.0031。
2. **NN 独立增量 ~0.00004**，蒸馏天花板低，提升 NN 单独超 GBDT 受 1:600 低信噪比固有限制。
3. **CV 对 NN 不可信**（公榜/CV 1.81），调 NN 只能靠公榜配额（每天 5 次），迭代慢。
4. asset×feature 已到顶（K 调参、去冗余、mean 偏置都试过）。

## 待决 / 下一步建议

用户上一个决定是"先注重提升 NN"，但诊断显示 NN 独立增量极小（0.00004），性价比存疑。需用户重新定方向：

- **(a) NN 再试 1-2 个本质方向**（attention / 学 GBDT 残差）确认天花板。GPU 充裕（4-6 次预算），但独立增量 0.00004 暗示收益小。
- **(b) 收尾 NN，转 8/23 备战**（确定性大增量）+ 答辩准备。8/23 后扩展数据到，用已验证 GBDT K=5 + 更多数据重训，增量可能远大于 0.00004。
- **(c) 其他**：GBDT 多 seed bagging / 三集成调权（+0.000027~0.000028，小但免费）。

**我的倾向：(b)**。NN 给了两个有价值认知（DAE 不代表大 NN、NN 漂移鲁棒），但独立增量太小，继续投入性价比低。8/23 扩展数据是确定性大增量，且 GBDT K=5 已验证，提前备好一键重训脚本。

## 工程备忘

- **polars 1.42 坑**：323 列同时聚合在 13.2M 全量扫描报假 OOM，必须分块（chunk=64）。quantile 表达式尤甚，moments 在采样 frame 上算。`np.corrcoef(X, rowvar=False)` 必须传 rowvar=False。
- **EDA 脚本**：`scripts/eda_common.py` + `eda_m1/2/3/4/5.py`，支持 `--sample-rows 0`（云端全量）/ `>0`（本地采样）。绘图 try/except 降级（无 matplotlib 不阻塞）。云端缺 tabulate 已改手写 md_table。
- **推理自包含**：strategy*/main.py 不能 import src（提交包只把 strategy_dir 加 sys.path）。新变换必须内联（_per_asset_block 等）。
- **依赖**：requirements 有 polars/lightgbm/xgboost/catboost/sklearn1.7.2/torch。scipy/matplotlib/seaborn 环境有但未列 requirements。**sklearn 必须降 1.7.2**（1.9 与 xgboost 不兼容）。
- **关键坑**：test.loc[...].to_numpy 只读视图，np.nan_to_num(copy=False) 报错需 copy=True。XGB/Cat 早停用内置 rmse（自定义加权R² feval 退化）。LightGBM random_state 要透传。

## 记忆索引

- `quantcontest2026-inference.md` — 推理协议、训练 backend、集成闭环、坑
- `quantcontest2026-next-steps.md` — **完整结论、死路、方向、所有实验数字**（最详细，新对话必读）
