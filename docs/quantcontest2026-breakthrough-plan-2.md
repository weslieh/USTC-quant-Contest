# Deep Research: 量化竞赛瓶颈突破——Responder 利用与范式级方法
> Generated 2026-07-17 | Depth: deep | Sources: 40+

## TL;DR

当前 GBDT+raw 的天花板（0.00295844）很可能可以通过两个方向突破：(1) 利用 47 个 responder 的内部结构（非直接预测单个 responder）作为输入特征——responder 协方差矩阵的 PCA 分量、GMM 软聚类标签、CCA 变体等；(2) 引入类似 Jane Street 2021 冠军的监督自编码器架构，在保持 GBDT 骨架的同时加入 NN 表征学习。k-NN 聚合特征（Optiver 的核心武器）和 within-time_id rank 变换是性价比最高的增量手段。纯 NN 方案（TabPFN、FT-Transformer）在当前 4 核无 GPU 约束下不实用，但作为集成组件可行。

## Executive Summary

本次深度调研覆盖了 40+ 个来源，包括 Kaggle Jane Street（2 届）、Optiver（2 届）、Numerai、G-Research 等类似竞赛的顶级方案，以及辅助标签利用、极端分布漂移、表格深度学习等学术文献。验证代理对 7 个关键论断进行了引文交叉检验（3 个完全证实，3 个部分证实需限定条件，1 个无法验证已降级）。

核心发现：responder 的价值不在两阶段堆叠（已验证失败，-18.8%），而在其**结构信息**——47 个 responder 之间的相关性、聚类、协方差矩阵包含了市场微观结构的压缩表示。所有类似竞赛的冠军方案都利用了某种形式的"辅助标签结构"（Jane Street 2021 的多标签联合训练、Numerai 的特征中性化、Optiver 的 k-NN 聚合本质上也是利用标签结构）。

从已试方向的历史规律（cs/rolling/drop/reweight/rank/per-asset/OOF/DAE 全部失败）可以推断：突破需要的不是"更好的 train 分布衍生特征"，而是"引入 train 分布之外的信息源"。三个最具潜力的方向：(1) responder 结构特征——responder 的协方差和聚类是固定属性，不受逐行漂移影响；(2) asset 间关系图——15 个 asset 的特征相似性矩阵是固定结构；(3) 监督自编码器——NN 的非线性表征可能捕捉 GBDT 轴分裂无法表达的交互。

## 1. Responder 的未开发潜力 [Confidence: Medium]

### 1.1 为什么两阶段堆叠失败，但 responder 仍可能有用

记忆文件记录：responder_03 与 target 相关 0.82，但 feature 预测 responder 极弱（pseudo corr 0.06），1 partition 曾误显 +26% 而 2 partition 稳健验证 -18.8%。这个失败的根因是**逐行预测**——用 323 个 feature 预测单个 responder 的值，信号太弱。

但换个角度：我们不需要准确预测每个 responder 的值。我们只需要从 47 个 responder 的**联合分布结构**中提取对 target 有用的信息。

一个关键类比：Jane Street 2021 竞赛的冠军方案使用多个 resp 列（resp、resp_1 到 resp_4）作为**多标签分类目标**，同时将 resp 绝对值的均值作为**样本权重**——而非试图预测其中任何一个 [1][2][9]。这与我们的直觉一致：responder 不是要被准确预测的，而是要被"结构性地利用"的。

### 1.2 具体技术方案

**方案 A：Responder 协方差 PCA 分量作为静态特征**

在训练集上计算 47×47 responder 协方差矩阵，取前 K 个主成分（通常 3-8 个解释 80%+ 方差）。对于每一行，将该行的 47 个 responder 值投影到这些主成分上，得到 K 个"responder 因子得分"。这些因子得分捕捉了当前行所处的 responder 联合状态（例如"所有 responder 同涨"vs"短期 responder 涨但长期 responder 跌"）。

关键优势：PCA 分量是**在全部训练数据上一次性计算的固定投影矩阵**，不依赖逐行分布，不受 AUC=1.0 漂移影响。推理时不可用 responder，但**训练时这些因子可以作为额外特征喂给模型**——模型学到的是"在某种 responder 结构状态下，feature→target 的映射规律不同"。[80][81][82]

**方案 B：GMM 软聚类标签**

在 47 维 responder 空间上训练高斯混合模型（GMM，K=5-10）。每个训练样本得到一个 K 维软分配概率向量——该样本属于每种"市场状态"的概率。这些概率作为特征加入训练。同样，特征本身不受漂移影响（GMM 在训练集上一次拟合）。

**方案 C：CCA 变体——特征-Responder 典型相关方向**

对特征矩阵 X 和 responder 矩阵 R 做典型相关分析（CCA），提取前 M 对典型变量。X 在典型方向上的投影捕捉了"特征空间中与 responder 联动模式最相关的方向"——这种信息 GBDT 无法直接从原始特征中发现，因为它是跨特征的线性组合。

**方案 D：Responder 自编码器瓶颈特征**

在训练集上训练一个自编码器，输入/输出为 47 维 responder 向量，瓶颈层 8-16 维。瓶颈表示是 responder 联合分布的压缩编码。训练时将瓶颈表示作为额外特征拼接到原始 323 特征上。推理时无法计算（无 responder），因此**仅在训练时生效**，相当于一个"训练时辅助信息正则化"——模型学到了利用 responder 结构来辅助特征→target 的映射。

**方案 E：多任务学习 + 梯度冲突管理**

同时预测 target 和 47 个 responder（48 个输出头），共享特征提取层。Wave 1 检索发现的关键技术是 **Impartial Auxiliary Learning (IAL)** [33]：传统多任务学习中 auxiliary task 往往被分配过低权重导致训练不充分，IAL 通过同方差不确定性自动平衡各任务，确保 noisy auxiliary 获得足够训练。**Gradient Cosine Similarity (GCS)** [34] 进一步通过检测梯度冲突来重加权 auxiliary loss——当某个 responder 的梯度方向与 target 梯度冲突时自动降权。这解决了之前"pseudo corr 0.06 太弱"的问题——即使单个 responder 信号弱，联合训练的共享表示可能有增量。

### 1.3 为什么这些方法与已试死路不同

| 已试死路 | 失败原因 | 新方案的不同之处 |
|---------|---------|---------------|
| 两阶段堆叠 (predict responder → target) | 逐行预测太弱，误差累积 | 不用预测，用结构特征（PCA/GMM/CCA） |
| cs/rolling 特征 | 跨样本/时间依赖，漂移下过拟合 | responder 结构是全局固定属性，不依赖逐行分布 |
| target rank 变换 | 逆映射依赖 train 分布假设 | 不改变 target 空间，只增加输入特征 |
| AV reweight | 向 test-like 重加权=向弱信号尾加权 | 不改样本权重，只加列 |

**核心差异**：所有已试死路都是"操作 train 分布"（改样本权重、改 target 尺度、加跨样本依赖）。而 responder 结构方案是"引入 train 分布之外的信息源"——responder 之间的相关性矩阵是一个全局常数，与样本的 time_id 无关。主办方提供了 47 个 responder 列，最合理的用途不是预测它们，而是利用它们的联合结构。

## 2. 类似竞赛的突破性技巧 [Confidence: High]

### 2.1 Jane Street 2021：监督自编码器 + MLP 融合

这是与本次竞赛结构最相似的方案。Jane Street 2021 竞赛同样有匿名特征、多个 responder 目标（resp 到 resp_4）、以及一个自定义效用函数 [1][2][9][10]。

**架构**：
- 输入特征先经 BatchNorm → GaussianNoise → Dense → BatchNorm → Swish（编码器）
- 解码器重建原始特征（MSE loss）
- 两个分类头：一个从瓶颈预测 action（ae_action），一个从 raw+瓶颈拼接预测 action（main action）
- 瓶颈同时接收 target 信号的梯度（监督自编码器），迫使表示学习包含预测信息
- 三个模型（不同种子）平均

**关键训练技巧**：
- **Swish 激活**（避免 dead neurons，比 ReLU 稳定）
- **只保留最后 2 个 CV fold 的模型**（见到的数据最多）
- **PurgedGroupTimeSeriesSplit**（时间序列专用 CV，含 gap/embargo）
- **标签平滑**（BCE + label_smoothing）
- **Hyperopt** 超参搜索
- **多 resp 列作为多标签**，resp 均值作为样本权重

**可迁移性评估**：我们的 Time-Series API 是逐 time_id 预测，与 Jane Street 的逐行预测略有不同，但自编码器架构本身是 per-row 的，不受时序 API 影响。4 核 12GB 无 GPU 下，一个轻量自编码器（323→128→32→128→323 + MLP head）推理开销约 0.1ms/切片，完全可行 [67]。

### 2.2 Optiver Realized Volatility：k-NN 聚合特征

Optiver 竞赛的核心武器是**最近邻聚合特征** [5][8][9]。对于每个 (time_id, stock_id) 样本，在特征空间中找 K 个最近邻（K=2,3,5,10,20,40），然后聚合这些邻居的 target 相关值（如已实现波动率）。

**可迁移性**：在我们的场景中，类似思路是：对每个样本，在 323 维特征空间中找最近邻，聚合邻居的 target 值。关键是**邻居必须在训练集内**（推理时没有 target），这实际上是一种"隐式最近邻回归"——用 KNN 作为 GBDT 的补充信号。

评估：这在我们的 AUC=1.0 漂移下可能失效——特征的最近邻在 train 和 test 之间意义不同（因为漂移）。需要类似 Optiver 的 within-time_id rank 变换来预处理。

### 2.3 Optiver：Within-Time_id Rank 变换

Optiver 数据中 trade.order_count 和 book.total_volume 存在严重的时间漂移（adversarial validation 证实）[5][8]。解决方案：将这些特征替换为它们在每个 time_id 内的**排名**（rank），从而消除跨时间的分布漂移。

**可迁移性**：这是与当前项目最直接相关的技巧。记忆已证"特征自身的跨样本/跨时间变换都过拟合"，但 rank 变换的本质不同——它不是"跨时间聚合"而是"同一时间内的相对排序"。在我们的场景中：对每个 time_id 切片，对每个特征列计算 rank（1 到 N），这个 rank 不依赖于跨时间的分布。预期效果：缩小 train/test 的 AUC=1.0 差距，提高 CV 可信度。

### 2.4 Numerai：特征中性化与 Era-Wise 验证

Numerai 的核心理念 [30][31]：(1) 数据按 ~120 个 era 划分，每个 era 内独立评估相关度——一致性比总得分重要；(2) 特征中性化——用伪逆消除预测对每个特征的线性依赖，迫使模型学习非线性交互。

**可迁移性**：中性化在我们的加权 R² 指标下可能不直接适用（Numerai 用 Pearson 相关度，中性化直接优化它）。但 era-wise 验证的思想可迁移：将训练数据按 time_id 段划分为多个"era"，每个 era 独立算 R²，一致性作为额外判据。

### 2.5 Jane Street Real-Time (2024-2025)：Lags 机制与 NN 集成

浏览器调研发现 [Kaggle 论坛]：2024-2025 版 Jane Street 竞赛的 `lags_` 变量提供前一天的 responder_6 值。利用 rolling 聚合 lags_ 的团队获得了显著提升。顶级方案仍以 GBDT+NN 集成为主，纯 NN（TabM, ICLR 2025）开始在某些条件下匹敌 GBDT。

## 3. 超越 GBDT 的可行路径 [Confidence: Medium]

### 3.1 表格深度学习现状

学术界的最新元分析 [66][67][68] 表明：经过充分超参优化的前馈神经网络（TabM、RealMLP）已经开始在某些基准上超越 GBDT。但前提是：(1) 超参搜索预算充足（~100 次试验）；(2) 使用数值嵌入层（PLR/PL）；(3) 在 train+val 上 refit。

TabPFN v2.5 [65][67] 是最受关注的"表格基础模型"——7M 参数，在 100M+ 合成数据集上预训练，无需训练即可推理，在某些 benchmark 上击败 tuned LightGBM。但有三个致命限制：(1) 最大 50K 行 / 2K 特征（我们的训练集是 13.2M）；(2) **不考虑行序**——对时序数据完全不适合；(3) CPU 上比 LightGBM 慢 10 倍。

**在 Numerai 上的实测**（与我们的低信噪比场景类似）：TabICL v2 的 NC~0.0039，而 GBDT~0.0170（GBDT 强 4 倍）[Numerai 论坛]。结论：**表格基础模型在低 SNR 场景下远不如 GBDT，与我们的 DAE 评估结论一致。**

### 3.2 实用 NN 方案：监督自编码器 + GBDT 集成

基于上述分析，最实用且风险最低的 NN 方案是：

1. **轻量监督自编码器**（323→128→32→128→323，MSE 重建 + target 预测头），~15 万参数
2. 在训练集上训练，提取瓶颈 32 维表示
3. 将瓶颈表示作为 32 个额外特征拼接到 raw 特征上
4. 用 LGB/XGB/Cat 在 raw+瓶颈特征上训练
5. 推理侧：自编码器推理 0.1ms/切片（瓶颈表示计算），加 GBDT 0.14ms

这个方案的核心理由：(1) 监督自编码器已在实际竞赛中验证有效（Jane Street 2021 冠军）；(2) 瓶颈表示可能捕捉 GBDT 无法表达的交互；(3) 不改变现有 GBDT 流水线，作为增量列加入；(4) 4 核 CPU 推理完全可行。

### 3.3 在线学习 / 测试时适应

Wave 2 检索发现 FSNet [83] 是一个针对流式时序预测的框架：(1) 慢速骨干网络在历史数据上训练；(2) 每层轻量 adapter 使用最近梯度更新；(3) 关联记忆存储历史 adapt 系数，通过余弦相似度检索相似历史模式。

**可迁移性**：我们的 Time-Series API 天然支持这种模式——每次 predict(time_id) 后可以用误差梯度更新 adapter（约 1-2ms 开销）。但风险较高（线上学习不稳定，可能导致得分波动），建议作为最后尝试。

### 3.4 图神经网络用于 Asset 间关系

15 个 asset 之间可能存在可学习的关联结构 [87][88][89]。具体做法：计算 15×15 asset 相似度矩阵（基于特征相关性或 responder 相关性），用轻量 GAT 生成每个 asset 的"上下文嵌入"，拼接到原始特征上。

评估：在 4 核无 GPU 条件下，轻量 GAT（15 节点，32 维隐藏）的推理开销可忽略。但 15 个 asset 的图太小，信息增益可能有限。优先级低于自编码器方案。

## 4. 优先级行动建议

基于 (1) 预期增益、(2) 实现成本、(3) 与已证伪方向的差异性、(4) AV-CV 可验证性 四个维度排序：

### P0：立即执行（高增益，低成本，可与现有流水线无缝集成）

- [ ] **Responder PCA/GMM 结构特征**（方案 A/B）：在训练集上计算 47 responder 的协方差矩阵 PCA（取 K=5-8）和 GMM 聚类（K=5-10），生成新特征列。用 AV-CV 验证（期望 +0.0002-0.0005）。实现量级：~100 行 Python。
- [ ] **Within-Time_id Rank 特征**：对 raw 323 特征做 per-time_id rank 变换（替换或拼接）。Optiver 验证这种变换对漂移 A/B 测试有效，我们的回忆中"cs 失败"是因为跨样本依赖而非 rank 本身。实现量级：~50 行代码。

### P1：优先执行（高增益，中等成本，需要新模型训练）

- [ ] **监督自编码器瓶颈特征**（方案 E 变体）：构建 323→128→32→128→323 + target 头的轻量自编码器，训练后提取瓶颈 32 维，拼接到 raw 特征上再训 GBDT。预期推理开销 +0.1ms/切片。需引入 torch（requirements.txt 加一行）或 pure numpy 手写 MLP（避免依赖）。
- [ ] **多任务学习（48 头：1 target + 47 responder）**：基于现有 GBDT 代码扩展——LGB 天然支持多输出吗？不（单输出）。考虑用 NN 多任务头或在 GBDT 上训练 48 个单输出模型共享特征（不现实）。实际方案：用 NN（MLP）做 48 头多任务学习，提取共享表示后再训练 GBDT。实现量级：~300 行。

### P2：值得试（中等增益，中等成本，需要公榜验证）

- [ ] **CCA 典型相关特征**（方案 C）：计算 X 与 R 的典型方向，提取前 M=5-10 对典型变量。与 PCA/GMM 互补（CCA 利用的是 feature-responder 的关联结构，而非 responder-responder）。实现量级：~100 行。
- [ ] **k-NN 聚合特征**（Optiver 核心武器）：在特征空间中找最近邻，聚合邻居的 target 值作为特征。注意：需要 within-time_id rank 预处理来缓解漂移。实现量级：~150 行。

### P3：备选（可能需要公榜配额，或等待扩展数据）

- [ ] **特征交互/比值扩展**（已有代码但 AV-CV 显示单 partition 反降）：考虑更智能的交互选择（基于 SHAP 重要性排名而非 top-K 暴力组合）。
- [ ] **Online Adapter（FSNet 风格）**：模型推理后基于误差更新轻量 adapter。风险高（线上行为不确定），但理论上有漂移适应能力。
- [ ] **Asset 间 GAT 嵌入**：15×15 asset 图 + GAT，生成 per-asset 上下文嵌入。图太小，增益存疑。

### 不推荐（与已验证死路同机制，或成本收益不成比例）

- ❌ TabPFN / TabICL / 时序 foundation model：Numerai 实测远弱于 GBDT（低 SNR），且不支持时序结构。
- ❌ DAE（去噪自编码器特征）：Plan agent 实测评估≈0，与 cs/rolling 失败机制相同（BN eval mode 在极端漂移下编码失真）。
- ❌ OOF Stacking：全量结果 +0.000027≈0，已放弃。
- ❌ Per-Asset 独立模型：+0.7% 但耗时 25×+超时，已放弃。

## 5. 关于 Responder 的补充论证

用户的直觉"主办方提供 47 个 responder 不可能完全没用"从信息论角度是正确的。一个更严谨的论证：

**信息论视角**：47 个 responder 构成一个 47 维向量空间。即使单个维度的逐行预测 signal-to-noise 比极低（pseudo corr 0.06），这个 47 维空间的**几何结构**——协方差矩阵、聚类中心、局部密度——是 47×N_train 样本的统计量，其估计精度远高于单个 responder 的逐行预测。关键在于**维度聚合**——类似于 CAPM 中个股 alpha 噪声大但因子组合的信噪比显著提升。

**竞争博弈视角**：前 19 队 0.003+ 的成绩比我们高 ~33%。在 AUC=1.0 漂移下，任何基于 train 分布的操作都失败（我们连续验证了 8 个方向），头部队伍大概率找到了"train 分布之外"的信息源。Responder 结构是最可能的候选——因为它是主办方明确提供且推理时不可用的数据，恰好构成一个"训练时独有"的信息优势。

**实施验证策略**：所有 responder 结构方案（PCA/GMM/CCA）都可以通过 AV-CV 快速验证是否为虚假信号。关键判据：(1) AV-CV 是否正向？(2) 单 partition AV-CV 的 std（~0.001）是否能分辨？若 AV-CV 显著正向且方向与公榜一致，再烧公榜配额。

## 6. 开放问题与风险

1. **Responder 结构特征的漂移风险**：虽然 PCA/GMM 的投影矩阵是固定的，但 test 集上特征分布不同，可能导致投影后的"responder 因子得分"在 test 上的分布与 train 不同。风险等级：低（因投影矩阵是全局常数，不依赖逐行分布）。

2. **多任务学习的负迁移风险**：47 个 responder 头中很多噪声极大，可能拖累共享表示。IAL [33] 和 GCS [34] 提供了缓解机制但未在竞赛场景验证。风险等级：中。

3. **NN 推理稳定性**：自编码器在 4 核 CPU 上推理，需要 ONNX 或纯 numpy 部署以确保确定性。PyTorch 推理可能有浮点精度差异导致每次提交得分不同。风险等级：低（ONNX 可解决）。

4. **公榜配额有限**：每天 5 次提交，私榜共 10 次。需要严格 AV-CV 预筛选后再上公榜。记忆已证实 AV-CV 虽不能分辨 10-15% 效应但能抓 gross 错误。

5. **8 月 23 日公榜截止**：只剩 ~5 周，需要快速迭代。优先 P0 方向（实现量级小），P1 次之。

6. **扩展数据（8/23 后发布）**：确定性增量，但私榜截止 8/31 只有 8 天窗口。建议提前准备好 responder 结构特征提取代码，扩展数据一到即可快速重训。

## Methodology

- **深度模式**：4+2 子代理，覆盖 40+ 来源
- **Wave 1**：4 个检索子代理，覆盖 Jane Street/Optiver/Two Sigma、Numerai/Responder、漂移处理/超越 GBDT、浏览器 Kaggle 论坛直访
- **Wave 2**：2 个 Gap-Fill 子代理，针对性补充 responder 结构化利用、在线学习、图神经网络
- **Phase 3.1 引文验证**：1 个验证子代理，对 7 个关键论断进行交叉检验，3 个 SUPPORTED，3 个 PARTIAL（已标注限定条件），1 个 UNSUPPORTED（已从报告降级）
- **来源筛选**：Tier 1（学术论文/官方文档）≥10 篇，Tier 2（竞赛 writeup/公司博客）≥12 篇，Tier 3（社区讨论/博客）≥18 篇
- **限制**：多数 Kaggle 竞赛 writeup 为 Tier 2-3（非学术但实战验证）；Jane Street 2024-2025 讨论为浏览器直接提取（较高可信度）；学术方法（IAL、GCS、FSNet）为 Tier 1 但未在此竞赛场景验证

## Bibliography

[1] CSDN/xmj — "Kaggle量化竞赛第一名的网络模型" — https://blog.csdn.net/csdn_xmj/article/details/141524614 — 2025 — Tier: 3
[2] CSDN/weixin_51484067 — "jane street market prediction 冠军方案 (3/3)" — https://blog.csdn.net/weixin_51484067/article/details/114635812 — 2025 — Tier: 3
[3] CSDN/sre5engineer — "Jane Street XGBoost与数据压缩" — https://blog.csdn.net/sre5engineer/article/details/152310109 — 2025 — Tier: 3
[4] CSDN/sunmoonstar432 — "Jane Street Real-Time 反面教材" — https://blog.csdn.net/sunmoonstar432/article/details/144629032 — 2025 — Tier: 3
[5] Juejin — "Optiver Realized Volatility 高分方案解析" — https://juejin.cn/post/7485729208108777482 — 2025 — Tier: 3
[6] CSDN/SuiZuoZhuLiu — "Trading at the Close 第一名方案" — https://blog.csdn.net/SuiZuoZhuLiu/article/details/139684005 — 2025 — Tier: 3
[7] CSDN/SuiZuoZhuLiu — "Optiver 第七名解决方案" — https://blog.csdn.net/SuiZuoZhuLiu/article/details/140137621 — 2025 — Tier: 3
[8] CSDN/qq_39970492 — "Optiver股票大赛Top2开源" — https://blog.csdn.net/qq_39970492/article/details/142676145 — 2025 — Tier: 3
[9] Mdnice — "Kaggle量化竞赛Top方案汇总" — https://mdnice.com/writing/ad11b39aed744785944fcb3638de80f1 — 2025 — Tier: 3
[10] Tencent Cloud — "基于AutoEncoder与MLP的预测模型" — https://cloud.tencent.com/developer/article/2242044 — 2025 — Tier: 3
[30] Numerai Forum — "What exactly is neutralization?" — https://forum.numer.ai/t/what-exactly-is-neutralization/2016 — 2025 — Tier: 2
[31] ApacheCN/TowardsDataScience — "Numerai Tournament Technical Overview" — https://www.cnblogs.com/apachecn/p/18463778 — 2025 — Tier: 3
[32] Sohu — "机器学习应用量化投资" — https://m.sohu.com/a/404767704_505915/ — ~2020 — Tier: 3
[33] Anonymous — "Unprejudiced Training Auxiliary Tasks Makes Primary Better (IAL)" — https://arxiv.org/html/2412.19547v1 — Dec 2024 — Tier: 1
[34] MAC Benchmark authors — "Multi-Attribution Learning with GCS/PCGrad" — https://arxiv.org/html/2603.02184v1 — Mar 2025 — Tier: 1
[35] Survey authors — "Deep Learning within Tabular Data: Foundations" — https://arxiv.org/html/2501.03540v1 — Jan 2025 — Tier: 1
[36] Google Research — "Efficient Method of Training Small Models for Regression" — https://arxiv.org/abs/2002.12597 — Feb 2020 — Tier: 1
[60] Anonymous — "Fully Test-time Adaptation for Tabular Data (FTAT)" — AAAI 2024 — Tier: 1
[61] J. Chen et al. — "TabLog: Test-Time Adaptation for Tabular Data Using Logic Rules" — ICML 2024 — Tier: 1
[62] Anonymous — "AdapTable: Shift-Aware Uncertainty Calibrator" — 2024 — Tier: 1
[63] Y. He et al. — "Sample Weight Averaging for Stable Prediction (SAWA)" — arXiv 2502.07414, 2025 — Tier: 1
[64] Z. Liu et al. — "Data Heterogeneity Modeling for Trustworthy ML" — arXiv 2506.00969, 2025 — Tier: 1
[65] PriorLabs — "Complete Guide to TabPFN v2.5" — https://zenn.dev/takatophy/articles/tabpfn-complete-guide — 2025 — Tier: 2
[66] Anonymous — "LATTLE: LLM Attention Transplant for Tabular Data" — arXiv 2511.06161, 2025 — Tier: 1
[67] Coggle Data Science — "TabPFN 2.5 Analysis" — CSDN, 2025 — Tier: 3
[68] Grid Dynamics — "AI Models for Demand Forecasting: TSFMs Compared" — 2025 — Tier: 2
[69] Toutiao — "CatBoost Still King? 2 Billion Rows Tested" — 2025 — Tier: 3
[80] Multi-Label Learning survey — label correlation exploitation — Tier: 2
[81] Manifold regularized feature selection — graph-based label correlation — Tier: 1
[82] Multi-task vs single task learning comparison — ScienceDirect — Tier: 1
[83] Pham et al. — "FSNet: Fast and Slow Learning Network" — ICLR 2023 — Tier: 2
[84] CRAD journal — "Elastic Gradient Ensemble for Concept Drift Adaptation" — Tier: 1
[85] CSDN survey — "Online Learning Algorithm Selection for Time Series" — Tier: 3
[86] MATLAB Incremental Learning Regression — official documentation — Tier: 1
[87] MDPI Analytics — "Temporal Fusion Transformer + GNN for Multi-Asset" — 2025 — Tier: 1
[88] Springer FITEE — "QuantBench: Benchmarking AI Methods for Quantitative Investment" — 2025 — Tier: 1
[89] Information Sciences — "Graph-based stock prediction with multisource information" — Tier: 1
[90] Nature Scientific Reports — "Deep learning for stock prediction with GNN" — 2025 — Tier: 1

## Source Extracts

### Key Sources (Tier 1-2)

**[1] Jane Street 2021 1st Place Architecture**
- **Summary:** Supervised denoising autoencoder + MLP. Input → BatchNorm → GaussianNoise → Dense(128) → BatchNorm → Swish (encoder). Decoder reconstructs input (MSE). Two action heads: one from bottleneck, one from raw+bottleneck. Joint training. 3 seeds averaged. Only last 2 CV folds kept.
- **Source type:** Competition solution analysis (CSDN)
- **Credibility tier:** 3 (blog aggregator, but directly describes verified 1st place solution)

**[5] Optiver k-NN Aggregation Features**
- **Summary:** ~600 features built; 360 are nearest-neighbor features. 7 different distance metrics over price-pivot matrix. N=2,3,5,10,20,40 neighbors. Within-time_id rank transform for drifting features. Log1p for skewed features. Ensemble: LightGBM + MLP + 1D-CNN. 4-fold time-series CV.
- **Source type:** Competition solution analysis (Juejin)
- **Credibility tier:** 3 (blog, but detailed technical content)

**[30] Numerai Neutralization**
- **Summary:** Neutralization = pseudo-inverse least squares to remove linear component each feature contributes to predictions. Proportion parameter controls subtraction amount. Forces meta-model to rely on non-linear interactions.
- **Key quotes:** "removes the component that the risky feature contributes alone, leaving only the interactions"
- **Source type:** Official competition forum
- **Credibility tier:** 2 (official forum, practitioner discussion)

**[33] IAL — Impartial Auxiliary Learning**
- **Summary:** Two-stage multi-task: decoder stage optimizes each task independently with homoscedastic uncertainty weights; encoder stage normalizes auxiliary gradient norms to match primary magnitude. Well-trained auxiliaries get higher weight, noisy ones suppressed.
- **Key quotes:** "better-optimized auxiliary tasks lead to improved performance on the primary task"
- **Source type:** Academic paper (arXiv)
- **Credibility tier:** 1

**[65] TabPFN v2.5 Complete Guide**
- **Summary:** 7M params, pre-trained on 100M+ synthetic datasets. No training needed. Beats default LightGBM on small data (<3K samples). NOT designed for time-series (row order ignored). CPU is slow. Max 50K samples, 2K features.
- **Source type:** Technical guide (Zenn)
- **Credibility tier:** 2

### Critical Browser Findings (Kaggle Forums)

**Jane Street Real-Time (2024-2025) Forum Analysis:**
- lags_ mechanism: API provides previous day's responder_6 at time_id=0; rolling aggregations of lags_ values create powerful features
- NN vs GBDT consensus: Ensembles of both dominate; GBDT for reliability, NN for ceiling; TabM (ICLR 2025) competitive
- Adversarial validation handling: Remove leakage features (IDs), then use adversarial validation set (top 20% most test-like) for hyperparameter tuning; within-time_id rank transform for temporal drift
- Multi-label approach from 2021 competition: multiple resp columns as binary action targets, average of resp values as sample weight

**Numerai Forum — Foundation Models on Low-SNR Data:**
- TabICL v2 on Numerai: NC~0.0039 vs GBDT NC~0.0170 (GBDT ~4x stronger)
- Conclusion: "With training window of ~100 eras subsampled to 20k-40k rows, the teacher sees extremely weak signal relative to noise. These foundation models were validated on datasets where features actually predict the target. Numerai is not that."
- Relevance: Our competition has similar low-SNR characteristics; foundation models unlikely to help
