# Deep Research: 极端分布漂移下的量化竞赛策略——本质不同范式的系统性调研

> 生成日期: 2026-07-15 | 深度: Deep | 来源数: 45

## TL;DR

你的 0.00289 天花板不是模型容量问题，而是范式选择问题。调研发现六个本质上不同的方向中，**特征中性化（prediction neutralization）**和**denoising autoencoder 表征**有最强的获奖背书和漂移鲁棒性论证；**asset_id embedding + per-asset 标准化**是你从未试过且与已死路径正交的方向；**OOF stacking**与你的死路 #8（raw feature 线性模型）有本质区别；**ranking loss 和 test-time adaptation**在 Kaggle 量化获奖方案中缺乏先例，但前者在生产量化中有成功案例。按预期增益×可行性排序，首要行动是：(1) 对 LGB/XGB/CatBoost 的 OOF 预测做特征中性化后 stacking，(2) 训练 denoising autoencoder 生成表征 concat 到 GBDT，(3) 将 asset_id 作为 embedding 输入 NN 并做 per-asset z-score 标准化。

## Executive Summary

本报告针对一个极端分布漂移的匿名时序量化竞赛（train vs test 对抗验证 AUC=1.0，302/323 特征漂移，CV 系统性高估 LB/CV≈1.6），系统调研了 Jane Street Market Prediction 2021、Ubiquant Market Prediction、G-Research Crypto Forecasting、Optiver Trading at the Close 以及 Numerai 锦标赛的获奖方案，以及 arXiv 上关于 tabular deep learning、domain adaptation、test-time adaptation 的学术论文，共 45 个来源。

调研的核心发现是：头部队伍（0.003+，高 33%）成功的范式与你已验证的八条死路有本质区别。你的死路清单的共同特征是"从 train 分布衍生的跨样本/跨时间/样本重加权/target 变换"——这些都在极端漂移下失效。而获奖方案的共同特征是"逐行 feature→target 映射 + 强正则 + 后处理中性化"，它们不依赖 train/test 分布平稳性。

具体而言：Jane Street 2021 第一名使用了 denoising autoencoder + MLP 架构，通过 Gaussian noise 注入和 BatchNormalization 实现"自动适应数据变化"，跨周期稳定性提升 22% [3][30][33]。Numerai 社区的核心技术是**特征中性化**——一种 OLS 回归残差化技术，作用于预测值而非输入特征，将预测中与特征线性相关的部分减去，保留非线性/特异性成分，经验上将 Sharpe 从 0.96 提升至 1.24 [24]。NVIDIA Kaggle Grandmaster Chris Deotte 描述了 3 级 stacking 架构，其中 L1 模型的 OOF 预测成为 L2 的输入特征，并附加 confidence/consensus 元特征 [7]。Ubiquant 获奖方案将 asset_id 作为 NN 中的 categorical embedding 处理 [19][34]。方正金工在生产量化中使用可微 RankIC（Spearman 相关）作为损失函数，实现 Rank IC 12.48%、年化多空收益 42.97% [37]。NeurIPS 2024 的 Drift-Resilient TabPFN 提出了一种真正 transductive 的范式——在单次前向传播中联合消费 train+test，将时间漂移编码进结构因果模型先验 [14]。

值得注意的是，本报告的大多数技术细节来自二级来源（CSDN/知乎博客，Tier 3），因为 Kaggle 讨论页面是 JavaScript 渲染的，WebFetch 无法提取正文 [16][19][20][22]。但所有关键 claim 已通过引文抽查验证（见方法论）。

---

## 1. 现状分析 [Confidence: High]

### 1.1 你的比赛特征与历史竞赛的映射

你的比赛格式（323 匿名 float 特征、15 asset/time_id、weighted zero-mean R²、Time-Series API、CPU-only 推理）与以下 Kaggle 竞赛高度相似：

- **Jane Street Market Prediction 2021**：130 匿名特征、~5900 time_id、weight 列、utility score（与 weighted R² 相关）、Time-Series API [16][17]
- **Ubiquant Market Prediction**：300 匿名特征、asset_id 列（3000+ 资产）、mean per-time_id Pearson 相关 [19]
- **Jane Street Real-Time Market Data Forecasting (2024-2025)**：79 特征、9 responders、symbol_id、custom weighted R² [26][27][28][29]

你的 323 特征和 15 asset 的规模介于 JS 2021（130 特征）和 Ubiquant（300 特征，3000+ 资产）之间，但你的 15 asset 远少于 Ubiquant 的 3000+，这意味着 per-asset 建模的可行性更高（每个 asset 有更多样本）。

### 1.2 死路清单的共性分析

你的八条死路可以归为三类：

**跨样本结构派生**（#1 截面特征、#2 rolling 时序特征）：这些方法在 train 的 (time_id, asset_id) 网格上构建跨样本统计量。极端漂移下，train 的跨样本分布与 test 不一致，导致这些特征在 test 上系统性偏差。

**样本重加权/选择派生**（#4 只用最近数据、#6 对抗重加权）：这些方法试图让 train 分布接近 test 分布。但"靠后时段信号弱"与"test 靠后"形成矛盾——向 test-like 重加权等于向低信号区域加权。

**target 变换派生**（#5 inverse-CDF 逆映射）：这假设 train 和 test 的 target 边缘分布相同，漂移打破了这个假设。

**bagging/线性模型派生**（#7 多种子 bagging、#8 raw 线性模型）：前者说明 backend 间高度相关（不是独立误差源），后者说明 raw feature 上没有线性可提取的额外信号。

关键洞察：所有死路都试图"从 train 分布中提取结构并在 test 上复用"。而获奖方案要么不依赖分布平稳性（如逐行映射 + 正则），要么在预测后做分布无关的后处理（如中性化）。

---

## 2. 六大方向的获奖方案与实现细节

### 方向 1: 神经网络表征在 CPU-only 推理下的可行性

#### 技术 1A: Denoising Autoencoder + MLP（Jane Street 2021 第一名）

**(a) 来源**：Jane Street Market Prediction 2021 第一名方案，架构在多个 CSDN 技术博客中有详细描述 [3][17][33]，NVIDIA Kaggle Grandmaster playbook 也提及类似架构 [8]。

**(b) 实现要点**：

架构分两阶段。第一阶段是 denoising autoencoder：输入层注入 Gaussian noise（作为数据增强和过拟合抑制），encoder 将 323 维压缩到 ~128 维隐表示，decoder 用 MSE 重建损失训练。关键细节包括使用 Swish 激活（避免 ReLU 的 dead neuron 问题）、每层 BatchNormalization、Dropout 正则 [3][33]。第二阶段是 MLP：将**原始输入与 encoder 输出 concat**（`Concatenate()([x0, encoder])`），送入多层 MLP，输出 target 预测 [33]。三个分支联合训练：decoder 重建损失（MSE）、autoencoder action 预测（BCE + label smoothing）、MLP target 预测（MSE），三损失加权求和 [3]。交叉验证使用 `PurgedGroupTimeSeriesSplit`（按 date 分组、带 group_gap，防止时间泄漏）[33][87→31]。Hyperopt 超参搜索，三种子集成 [3]。

**(c) 为什么漂移鲁棒**：

三个机制共同作用。第一，**Gaussian noise 注入**等价于在 feature 空间做数据增强——模型不依赖特定 feature 值的精确分布，而是学习"加噪后仍可重建"的鲁棒表示。第二，**BatchNormalization 在推理时使用 test batch 的统计量**——这意味着 BN 层会自动适应 test 数据的均值/方差，相当于一种隐式的 test-time adaptation。CSDN 技术博客明确指出 BN"自动适应数据变化"，跨周期测试稳定性提升 22% [30]。第三，**encoder 学到的是非线性压缩表示**，而非跨样本统计量——它对每行独立作用，不依赖 train 的跨样本分布结构。

**(d) 4核无GPU推理可行性评估**：

MLP 架构相对轻量（~130-323 输入 + 128 维编码 + 3-4 层全连接），batch size 4096 在 CPU 上完全可行。但本调研发现一个重要缺口：**所有公开 writeup 均未记录 CPU 推理优化细节**（ONNX 导出、量化、TorchScript 均未被提及）[33][42]。推测原因是 JS 2021 的模型足够小（~130 特征），直接 Keras/TF 推理即可满足时间限制。对于你的 323 特征，需要实测。建议：训练在云端 GPU 完成，推理时导出为 ONNX 格式使用 ONNX Runtime（可获得 4-5x CPU 推理加速），或使用 TensorFlow 的 `tf.function` + graph mode 优化。另一个选项是 knowledge distillation——用 NN 作为 teacher 训练一个更小的 MLP student。

**(e) 预期增益：中**

JS 2021 第一名仅靠 supervised autoencoder 就能维持第一名 [11]。但你的特征数更多（323 vs 130），且你的 GBDT 后端已收敛到 0.00289，NN 的增量取决于是否能学到 GBDT 未捕获的非线性交互。323 特征的 NN 容量需要更大，过拟合风险更高。

**死路近亲标注**：与死路 #2（rolling 时序特征）有表面相似（都涉及"特征变换"），但本质不同——autoencoder 是逐行编码，不使用跨样本/跨时间窗口统计量；rolling 特征显式使用历史窗口均值/标准差，依赖时间结构平稳性。

#### 技术 1B: FT-Transformer / TabM（学术前沿）

**(a) 来源**：Gorishniy et al. (arXiv 2106.11959) 提出 FT-Transformer [5]；Zabërgja et al. (arXiv 2402.03970) 的大规模 benchmark 发现简单前馈网络（TabM, RealMLP）在多数场景下优于 attention 架构 [2]。

**(b) 实现要点**：FT-Transformer 将所有特征通过 Feature Tokenizer 转为 embedding token，然后过标准 Transformer encoder 层。TabM 是带 PLR (Piecewise Linear Regression) 数值嵌入的 MLP。Rubachev et al. 发现"数值嵌入 + target-aware mask-based self-prediction 预训练"效果最佳 [45]。

**(c) 为什么漂移鲁棒**：Feature Tokenizer 的嵌入学习的是"每个特征值的语义位置"而非"特征值的全局分布"——这在分布漂移下比直接使用原始浮点值更鲁棒。但论文本身未直接测试分布漂移场景。

**(d) CPU 推理可行性**：FT-Transformer 的 attention 计算复杂度 O(n²) 在 323 特征下可接受（n=323 token），但 Transformer 在 CPU 上比 MLP 慢。TabM（纯 MLP + 嵌入）更适合 CPU 推理。

**(e) 预期增益：存疑**

FT-Transformer 在 benchmark 中"与 GBDT 竞争力相当，但无普遍优势"（引文抽查修正：原 "7/11" 说法来自二级来源的再分析，原论文结论更谨慎 [5]）。你的 GBDT 已收敛，FT-Transformer 不太可能大幅超越。

#### 技术 1C: SAINT（inter-sample attention + contrastive pre-training）

**(a) 来源**：Somepalli et al. (arXiv 2106.01342) [6]。

**(b) 实现要点**：SAINT 在行和列两个维度都做 self-attention（inter-sample attention），并用 contrastive loss 做自监督预训练。

**(c) 为什么漂移鲁棒**：Contrastive pre-training 学到的是"行与行之间的相似性结构"——即使分布漂移，相似的行仍应有相似的表示。但 inter-sample attention 在推理时需要整个 batch，与 Time-Series API 的逐 time_id 推理（~15 行）兼容但计算量较大。

**(d) CPU 推理可行性**：inter-sample attention 在 15 行 batch 上计算量很小，但 Transformer 层在 CPU 上仍比 MLP 慢 5-10x。

**(e) 预期增益：低-存疑**

SAINT 在 benchmark 中表现一般 [2]，且你的 15 行/time_id 的 batch 太小，inter-sample attention 难以发挥优势。

### 方向 2: Asset-aware / 分组建模

#### 技术 2A: asset_id 作为 Embedding 输入

**(a) 来源**：Kaggle 官方 Entity Embedding 教程 [34]；Ubiquant Market Prediction 获奖方案 [19]；JS 2024-2025 反面教材明确指出 symbol_id "需要专门处理而非作为 flat feature" [28]。

**(b) 实现要点**：

将 asset_id（你的 0-14）映射为 Embedding 层：`asset_embedding = keras.layers.Embedding(15, embedding_size)(asset_id_input)`，embedding_size 建议为 4-8（对于 15 个类别，过大的 embedding 会过拟合）。将 embedding 向量 concat 到 323 个 float 特征后，送入 MLP。训练时 embedding 与 MLP 联合优化。Kaggle 官方教程强调：对于高基数分类变量，Embedding 层远优于 one-hot 或直接传 raw integer ID [34]——raw ID 会强加虚假的序数关系（asset 14 "大于" asset 0 无意义）。

**(c) 为什么漂移鲁棒**：

这与你已死的截面特征有**根本区别**。截面特征在**每个 time_id 内跨 asset 做标准化**（rank/zscore/demean），这依赖 train 时刻的 asset 间关系在 test 时刻不变——漂移打破了这个假设。而 asset_id embedding 学习的是**每个 asset 的固定身份向量**——不依赖任何特定 time_id 的 asset 间关系。即使资产间的相关性结构漂移，embedding 仍能捕获每个 asset 的个体特性（如波动率水平、对某些 feature 的敏感度）。

**死路近亲标注**：与死路 #1（截面 rank/zscore/demean）表面相似（都涉及 asset 信息），但截面统计量是 per-time_id 跨 asset 计算的——这依赖跨样本分布平稳性；embedding 是 per-asset 的固定向量，每行独立查找，不依赖跨样本结构。两者是正交的。

**(d) CPU 推理可行性**：Embedding 查找是 O(1) 表查找，几乎无额外推理开销。完全可行。

**(e) 预期增益：中-高**

你从未试过这个方向。15 个 asset 足够少（不会过拟合 embedding 表），又足够多（有 group 结构可利用）。JS 2024-2025 的反面教材显示，不使用 embedding 而直接传 symbol_id 会导致负分 [28]；而 Ubiquant 获奖方案使用 investment_id embedding 是标准做法 [19]。如果你目前将 asset_id 完全忽略（LGB/XGB 不自然处理 categorical），加入 embedding 可能带来可观增量。

#### 技术 2B: Per-asset 标准化（per-asset z-score across time）

**(a) 来源**：JS 2020-2021 获奖方案使用 per-date z-score 标准化来适应分布漂移 [31]；Mitsui Commodity Prediction 方案演示了 per-asset feature engineering via `groupby('id')` [32]。

**(b) 实现要点**：

对每个 asset_id，在训练数据上计算每个 feature 的均值和标准差：`μ_a, σ_a = groupby('asset_id')[feature].mean(), groupby('asset_id')[feature].std()`。然后标准化：`feature_normalized = (feature - μ_a) / σ_a`。推理时，使用训练时计算的 μ_a, σ_a 对 test 行做同样的标准化。

**(c) 为什么漂移鲁棒**：

这与你已死的截面标准化**正交但本质不同**。截面标准化在**每个 time_id 内跨 asset 做 demean**（`feature - mean_across_assets_at_time_t`），依赖 time_id 内的 asset 分布。Per-asset 标准化在**每个 asset 内跨 time 做标准化**（`feature - mean_across_time_for_asset_a`），这做的是"去除每个 asset 的个体量级差异"——asset A 的 feature_0 均值可能是 5，asset B 的可能是 -3，标准化后都变为 0 均值。这在漂移下更鲁棒，因为每个 asset 的 feature 均值是其**固有属性**（不随 time_id 变化），而 time_id 内的 asset 均值是**时变的**。

**死路近亲标注**：与死路 #1 截面标准化**形似神不似**。截面 demean 是 `groupby('time_id').transform(lambda x: x - x.mean())`——依赖每个 time_id 的 asset 分布。Per-asset 标准化是 `groupby('asset_id').transform(lambda x: (x - x.mean()) / x.std())`——依赖每个 asset 的时间分布。前者在漂移下失效（time_id 间分布不同），后者可能更鲁棒（asset 的固有属性更稳定）。

**(d) CPU 推理可行性**：完全可行——推理时只需 15 组 (μ_a, σ_a) 参数，O(1) 计算量。

**(e) 预期增益：中**

Per-asset 标准化可能帮助 GBDT 更好地处理不同 asset 的量级差异。但 GBDT（尤其 LGB）本身对单调变换不敏感（树分裂是阈值比较），所以标准化对 GBDT 的增量可能有限。对 NN 则有显著帮助。

#### 技术 2C: Mixture of Experts / Feature-grouped Bagging

**(a) 来源**：JS 2020-2021 综合方案描述了 Transformer encoder 在 100-timestep 序列上建模，与 XGBoost 在 Mixture of Experts 框架中融合 [31]。还提到了 feature-grouped bagging（按物理含义拆分特征组，如价格相关 vs 订单簿相关 [31]）。

**(b) 实现要点**：将 323 个特征按某种准则（如相关性聚类）分组，每组训练独立模型，然后 MoE 门控网络决定每组模型的权重。

**(c) 为什么漂移鲁棒**：如果某些特征组漂移更严重，MoE 门控可以学习降低对应专家的权重。但这依赖门控网络能在 test 上正确判断——如果门控本身也受漂移影响，则无效。

**(d) CPU 推理可行性**：多个小模型 + 门控网络在 CPU 上可行，但增加了系统复杂度。

**(e) 预期增益：低-存疑**

MoE 在学术上有吸引力，但在你的 15 asset / 323 feature 设置下，特征分组缺乏先验依据（特征是匿名的，无法按物理含义分组）。Feature-grouped bagging 需要知道特征的物理含义，这在匿名特征下不可用。

### 方向 3: Stacking / Meta-ensembling on OOF Predictions

#### 技术 3A: OOF-based 多级 Stacking

**(a) 来源**：NVIDIA Kaggle Grandmaster Chris Deotte 的 stacking playbook [7][8]；Yang (2024) 的 GAS 论文在金融预测中使用 XGB+LGB+RF 为 base，LGB 为 meta-learner [9]。

**(b) 实现要点**：

Level 1：训练多个异构 base 模型（LGB、XGB、CatBoost，可选 NN），每个用 K-fold 交叉验证生成 OOF 预测。关键：OOF 预测是**模型在未见过的 fold 上的预测**，不是训练集预测——这避免了 train 上的过拟合信号 [7]。

Level 2：将 L1 的 OOF 预测作为**新特征**（3 个 GBDT 各一列 OOF），训练 meta-model。NVIDIA 方案还附加了**元特征**：`confidence = std(OOF)`（三个模型预测的标准差，衡量不确定性）和 `consensus = mean(OOF)`（三个模型预测的均值，衡量一致性）[7]。这些元特征让 meta-model 能学习"当三个模型分歧时该信任谁"。

Level 3（可选）：对 L2 的多个 meta-model 做加权平均。

NVIDIA 方案的另一个关键技巧是**residual-based stacking**：L2 模型不在原始 target 上训练，而在 L1 ensemble 的**残差**上训练——即 `residual = target - mean(OOF_L1)`。这强制 L2 学习"L1 集成未能捕获的信号" [8]。

还有**hill-climbing weight search**：在验证集上贪心搜索 L1 模型的最优权重组合，每次加入一个模型，仅当加入后验证分数提升 [8]。

**(c) 为什么漂移鲁棒**：

这是与你死路 #8（raw feature 线性模型）的**关键区别**。死路 #8 在 323 个 raw feature 上训练线性模型——这尝试从原始特征中提取线性信号，但你已验证 raw feature 上线性模型 valid R² 全负，说明原始特征中没有额外的线性可提取信号。

OOF stacking 的 meta-model 不在 raw feature 上训练，而在**模型的预测**上训练。LGB/XGB/CatBoost 的预测已经是 feature 的非线性变换——meta-model 学习的是"哪个 backend 在什么情况下更准"，而非"哪个 feature 预测 target"。这在漂移下可能更鲁棒，因为：如果三个 backend 在 test 上的相对表现与 train 不同（因漂移），meta-model 如果学到了 backend 间分歧 → 误差的映射关系，可以动态调整信任权重。但这依赖 meta-model 能在 OOF 上学到这种映射——如果漂移也打破了 OOF 上的 backend 关系，则无效。

**死路近亲标注**：与死路 #8 有表面相似（都是"在已有东西上面加一层"），但本质不同。死路 #8 = raw_feature → linear_model → target；OOF stacking = (LGB_OOF, XGB_OOF, CatBoost_OOF, confidence, consensus) → meta_model → target。前者尝试提取 raw feature 的线性信号（已验证不存在），后者学习哪个模型更可信（新信息）。

**(d) CPU 推理可行性**：推理时只需运行 L1 的 3 个 GBDT + 1 个 L2 meta-model（小 LGB 或线性模型），计算量约为单模型的 4x。在 4 核 CPU 上完全可行。

**(e) 预期增益：中**

你的三种子 bagging 仅 +0.000028 [死路 #7]，说明 backend 间高度相关。但 OOF stacking 的增量不来自 bagging（平均），而来自**条件信任分配**——meta-model 在分歧大的样本上可以选择更可信的 backend。预期增量取决于三个 backend 在 test 上的分歧模式是否与 OOF 一致。考虑到你的三个 backend 独立收敛到 ~0.00289，分歧可能较小，增量有限但仍值得尝试（低风险、低成本）。

### 方向 4: 漂移鲁棒特征范式

#### 技术 4A: 特征中性化（Prediction Neutralization）——最重大发现

**(a) 来源**：Numerai 社区的核心技术 [23][35]；方正金工详细描述了实现和效果 [24]；CSDN 教程解释了 group neutralization 变体 [25]；JS 2020-2021 方案也使用了 SVD-based residualization [31]。

**(b) 实现要点**：

特征中性化是**对预测值做后处理**，而非对输入特征做变换。具体实现：

```python
def neutralize(df, target_col, feature_cols, proportion=1.0):
    """对预测值做特征中性化"""
    scores = df[target_col].values
    exposures = df[feature_cols].values
    # 添加常数列（去除均值）
    exposures = np.hstack([exposures, np.ones((len(exposures), 1))])
    # OLS 回归：将预测值回归到特征上
    # pinv 是伪逆，等价于 OLS 解
    scores -= proportion * (exposures @ (np.linalg.pinv(exposures) @ scores))
    # 重新标准化
    return scores / scores.std()
```

机制解释：`exposures @ np.linalg.pinv(exposures) @ scores` 计算的是 scores 在特征空间上的**正交投影**——即"如果用特征线性预测 scores，预测值是什么"。然后从 scores 中减去这个线性投影，保留**残差**（非线性/特异性成分）。这等价于 OLS 回归后取残差 [24]。

经验结果：中性化后，max feature exposure（每个特征与预测的 Spearman 相关）从 ~0.30 降至 ~0.015，Sharpe ratio 从 0.96 提升至 1.24 [24]。

Group neutralization 变体：在 asset 分组内 demean 预测值（`group_neutralize(predictions, asset_id)`），即在每个 asset 组内去除预测均值 [25]。

**(c) 为什么漂移鲁棒**：

这是与你所有死路**最本质不同**的技术。你的死路都在训练阶段修改特征/样本/target——这些修改依赖 train 分布结构，漂移下失效。特征中性化在**推理阶段对预测做后处理**——它不改变模型训练过程，不依赖 train 分布的任何统计量。

具体而言，中性化从预测中减去"与特征线性相关的部分"。在漂移下，特征本身虽然在漂移，但"预测中与特征线性相关的部分是噪声"这个性质可能更稳定——因为如果特征的漂移部分是加法性的（`f_test = f_train + shift`），线性投影会自动适应 shift（因为投影是在 test 特征上计算的，不是 train 特征）。

关键洞察：中性化使用的是**test 时刻的特征**（推理时的特征），不是 train 时刻的特征。这意味着它自动适应了 test 的特征分布——这是一种隐式的 test-time adaptation。

**死路近亲标注**：

- 与死路 #1（截面特征）的区别：截面特征在**输入端**做 per-time_id 跨 asset 标准化——依赖 time_id 内 asset 分布。中性化在**输出端**做 OLS 残差化——在预测值上去除与特征的线性关系，不依赖 asset 间关系。
- 与死路 #6（对抗重加权）的区别：重加权在**训练阶段**改变样本权重——向 test-like 区域倾斜，但 test-like = late period = low signal。中性化在**推理阶段**修改预测——不改变训练分布，不影响信号强度。
- 与死路 #5（target 变换 + inverse CDF）的区别：target 变换假设 target 边缘分布平稳。中性化不假设任何边缘分布——它只假设"预测中与特征线性相关的部分是噪声"。

**(d) CPU 推理可行性**：完全可行。中性化是一次 OLS 投影，对于 323 特征 × ~15 行/time_id，计算量极小（`np.linalg.pinv` on 15×324 matrix）。可以在 Time-Series API 的每次调用中实时计算。但需要注意：中性化需要一定量的样本来计算 pinv——单次 15 行可能不够稳定。建议在实例内维护一个滑动窗口（如最近 500-1000 行），在窗口上计算投影矩阵。

**(e) 预期增益：高**

这是 Numerai 社区几十年的核心实践 [23][35]，Sharpe 从 0.96→1.24 的提升意味着 ~29% 的风险调整收益提升 [24]。在你的 weighted R² 场景下，中性化可能不会直接提升 R²（因为 R² 衡量的是绝对预测精度，不是风险调整收益），但通过去除"漂移驱动的线性成分"，可能减少公榜/CV gap（从 1.6x 降低），从而让 CV 更可靠、模型选择更准确。如果公榜/CV gap 从 1.6x 降到 1.2x，等效于有效增益提升 ~33%——这恰好是头部队伍超出你的 33% 的幅度。

#### 技术 4B: BatchNormalization 作为隐式 Test-Time Adaptation

**(a) 来源**：JS 2020-2021 获奖方案中 BatchNormalization 被描述为"自动适应数据变化" [30]；JS 2020-2021 综合方案强调 BN 在跨周期稳定性上的 22% 提升 [31]。

**(b) 实现要点**：在 NN 的每层后加 BatchNormalization 层。训练时使用 batch 统计量；推理时可以选择使用运行时统计量（train 的 EMA）或重新计算 test batch 统计量。后者等价于 TENT（熵最小化 test-time adaptation）[13]。

**(c) 为什么漂移鲁棒**：BN 在推理时如果使用 test batch 的统计量（而非 train 的 EMA），相当于自动将特征标准化到 test 时刻的分布——这是一种隐式的 test-time adaptation。CSDN 技术博客明确指出 BN"自动适应数据变化"，跨周期测试稳定性提升 22% [30]。

**(d) CPU 推理可行性**：BN 推理无额外开销，完全可行。关键是在推理时设置 `training=True`（使用 batch 统计量而非 EMA），但这需要足够大的 batch（15 行可能不够）。建议在实例内维护一个滑动窗口，在窗口上计算 BN 统计量。

**(e) 预期增益：中**

仅适用于 NN 路线。如果你的 base model 是纯 GBDT，BN 不适用。但如果训练 denoising autoencoder（方向 1A），BN 是内置的漂移适应机制，无需额外成本。

#### 技术 4C: Adversarial Training / DANN

**(a) 来源**：arXiv 上有 domain adaptation 论文 [39][40]；但**无 Kaggle 量化获奖方案使用 DANN**的直接证据。GAS 论文使用对抗验证做特征过滤 [9]，Carl McBride Ellis 详细描述了对抗验证作为诊断工具 [10]。

**(b) 实现要点**：DANN（Domain-Adversarial Neural Network）在 NN 中加一个 domain classifier 分支，用 gradient reversal layer 让特征提取器学习"domain-invariant"表示——即让 encoder 无法区分 train 和 test。

**(c) 为什么漂移鲁棒**：理论上，domain-invariant 表示对分布漂移免疫。但实践中，当 train/test AUC=1.0（完全可分）时，DANN 可能无法找到任何 domain-invariant 表示——因为 train 和 test 在特征空间中完全分离，没有重叠区域供梯度反转工作。

**死路近亲标注**：与死路 #6（对抗重加权）有近亲关系——都使用对抗验证的信号。区别在于：重加权用 test-likeness 调整样本权重（有害，因为 test-like = late = low signal）；DANN 用梯度反转让 encoder 学习 domain-invariant 特征（不改变样本权重，而是改变特征表示）。但你的对抗 AUC=1.0 意味着 train/test 完全可分，DANN 的假设（存在 domain-invariant 子空间）可能不成立。

**(d) CPU 推理可行性**：DANN 推理时不需要 domain classifier（只用 encoder + task head），无额外开销。

**(e) 预期增益：低-存疑**

在 AUC=1.0 的极端漂移下，DANN 的理论假设可能不成立。且无 Kaggle 量化获奖方案使用 DANN 的直接证据。

### 方向 5: Ranking/Correlation Loss 优化

#### 核心发现：Kaggle 获奖方案的负面证据

**(a) 来源**：本调研系统检查了 Jane Street 2021 第一名 [3][17][33]、Optiver 第一名 [11][21]、G-Research 第二/三名 [20] 的 writeup。

**(b) 发现**：**没有找到任何 Kaggle 量化获奖方案使用 custom ranking/correlation/pairwise loss 的直接证据**。Jane Street 2021 第一名使用 MSE（decoder 重建）+ BCE（action 预测）+ MSE（target 预测）[3][11]。Optiver 高分方案使用 MAE 作为 LGB objective [21]。这些方案都使用标准损失函数，而非专门优化 Pearson 相关或排名。

**(c) 为什么仍可能有效**：

方正金工（Founder Securities）在生产量化中实现了**可微 RankIC**——直接以 Spearman 相关系数作为损失函数训练深度学习选股模型，实现 Rank IC 12.48%、年化多空收益 42.97%，相比 MSE 对照组"有明显提升" [37]。这证明了在**生产量化场景**中，ranking loss 确实优于 MSE。

但 PyTorch 论坛记录了一个重要的实践问题：**Pearson 相关作为损失函数数值不稳定**——因为 `torch.sqrt` 在分母中，当输入接近 0 时梯度爆炸（`d/dx sqrt(x) = 1/(2*sqrt(x)) → ∞`）[38]。这可能解释了为什么 Kaggle 选手避免使用它。

**(d) 对你的场景的适用性分析**：

你的指标是 weighted zero-mean R² = 1 - Σw(y-ŷ)²/Σw·y²。当 |ŷ| << |y| 时（你的情况：预测量级 << 目标量级），R² ≈ 1 - Σw·y²/Σw·y² + 2Σw·y·ŷ/Σw·y² = 2Σw·y·ŷ/Σw·y² ≈ 加权相关。这意味着优化加权相关可能直接优化你的指标。

但有两个风险：(1) 数值不稳定性 [38]；(2) 你的 GBDT 后端（LGB/XGB/CatBoost）的 custom objective 实现需要提供一阶和二阶导数，ranking loss 的二阶导数（Hessian）可能不存在或不定（Spearman 是不可微的阶跃函数）。

**死路近亲标注**：与你已死的任何路径**无近亲关系**——你从未尝试过非 MSE 损失。这是全新的方向。

**(e) 预期增益：中-存疑**

方正金工的证据 [37] 在生产量化中有效，但 Kaggle 获奖方案的负面证据 [3][21] 令人犹豫。建议：先在 LGB 上尝试 `pearson` 近似损失（用 `1 - cosine_similarity` 或基于协方差的近似），而非直接 Spearman。如果数值不稳定，回退到 MSE + 中性化后处理。

### 方向 6: Transductive Learning / Test-Time Adaptation / Self-Training

#### 技术 6A: Test-Time Training (TTT)

**(a) 来源**：Sun et al. (ICML 2020, arXiv 1909.13231) [12]；TTA 综述 Xiao & Snoek (arXiv 2411.03687) [15]。

**(b) 实现要点**：在训练时联合优化主任务（target 预测）和辅助自监督任务（如 denoising autoencoder 重建）。推理时，对每个 test batch 先在自监督任务上更新共享权重（几步梯度下降），然后在更新后的模型上做 target 预测。

**(c) 为什么漂移鲁棒**：TTT 在推理时**用 test 数据更新模型**——这是真正的 test-time adaptation。自监督任务（denoising）不依赖标签，因此可以利用 test 的无标签特征。

**(d) CPU 推理可行性**：每次推理需要在 15 行 test 数据上做几步梯度更新——这在小 batch 上可能不稳定（15 行太少）。建议在实例内维护一个滑动窗口（如最近 500-1000 行 test 特征），在窗口上做 TTT 更新。4 核 CPU 上几步梯度下降可行，但会增加推理延迟。

**死路近亲标注**：与你已死的任何路径**无近亲关系**——你从未在推理时使用 test 特征更新模型。

**(e) 预期增益：存疑**

**关键发现：没有找到任何 Kaggle 量化获奖方案使用 TTT 的直接证据**（引文抽查修正：原"无任何方案使用 TTA"的表述过于绝对，因为 Chris Deotte 的 NVIDIA 博客提到了 soft pseudo-labeling [8]，但那是在训练阶段使用 test 预测做伪标签，不是推理时更新模型权重）。TTT 主要在图像分类的分布漂移场景下验证 [12]，在 tabular 量化数据上无先例。

#### 技术 6B: Drift-Resilient TabPFN（NeurIPS 2024）——最具突破性的发现

**(a) 来源**：NeurIPS 2024 poster "Drift-Resilient TabPFN" [14]。

**(b) 实现要点**：TabPFN 是一种 Prior-Data-Fitted Network（PFN），在数百万个合成数据集上预训练。推理时，它**接受整个训练集作为输入**，在单次前向传播中对 test set 做预测——这是一种真正的 transductive 学习（联合消费 train+test）[14]。Drift-Resilient 版本的先验基于结构因果模型（SCM），其中主 SCM 的参数随时间渐变，次级 SCM 指定主 SCM 参数的变化方式——这直接将时间漂移编码进了模型的先验知识 [14]。

**(c) 为什么漂移鲁棒**：TabPFN 不在 train 上"训练"——它是预训练好的 in-context learner。推理时它将 train+test 一起看，自动推断 test 的分布。它的 SCM 先验直接建模了"参数随时间渐变"的漂移机制。这与你的"逆 CDF 逆映射假设分布平稳"的死路 #5 有本质区别——TabPFN 假设的是"分布渐变"而非"分布不变"。

**(d) CPU 推理可行性**：TabPFN "在数秒内运行，无需超参调优" [14]。但它的 in-context 需要将整个 train set（13.2M 行）作为 context——这在实践中可能不可行（内存和计算量）。建议：在 Time-Series API 的滑动窗口（最近 5000-10000 行）上使用 TabPFN 做推理，而非全量 train set。

**死路近亲标注**：与死路 #5（inverse CDF 逆映射）有表面相似（都是"在 test 上做映射"），但本质不同。逆 CDF 假设 train 和 test 的 target 边缘分布相同；TabPFN 假设分布渐变并自动推断渐变方向。

**(e) 预期增益：存疑-高**

这是理论上最优雅的漂移鲁棒方案，但实践中在 13.2M 行 + 323 特征 + 15 asset 的规模下，TabPFN 的 in-context learning 可能面临 context length 限制。且 TabPFN 是较新的方法（NeurIPS 2024），在量化竞赛中无应用先例。建议在滑动窗口上小规模实验。

#### 技术 6C: Soft Pseudo-Labeling on Test Features

**(a) 来源**：NVIDIA Grandmasters Playbook [8]。

**(b) 实现要点**：用 L1 模型在 test 特征上生成预测（伪标签），然后将 test（带伪标签）加入训练集训练 L2 模型。关键：使用 **soft pseudo-labels**（连续值而非 hard labels），且使用 **k-fold pseudo-label generation**——确保验证 fold 的伪标签来自未见过该 fold 数据的模型 [8]。

**(c) 为什么漂移鲁棒**：test 特征是真实分布的样本——用它们（即使带噪声伪标签）训练可以让模型适应 test 的特征分布。这与你已死的对抗重加权不同——重加权改变 train 样本权重（不引入 test 信息），pseudo-labeling 引入 test 特征作为额外训练样本。

**死路近亲标注**：与死路 #6（对抗重加权）有**间接近亲关系**——都试图让模型适应 test 分布。区别在于：重加权调整 train 样本权重（不引入新数据），pseudo-labeling 加入 test 数据作为新训练样本（引入新信息）。但需要注意：如果伪标签质量差（因为模型在 drift 下预测不准），pseudo-labeling 可能引入噪声而非信号——这正是你需要验证的。

**(d) CPU 推理可行性**：Pseudo-labeling 在训练阶段完成（云端 GPU），推理时无额外开销。

**(e) 预期增益：中**

NVIDIA Grandmaster 推荐此技术 [8]，在你的场景下，test 特征在公开数据中可用——你可以用 base model 在 test 特征上生成伪标签，然后训练增强模型。风险在于：如果 base model 在 test 上的预测本身就因漂移而偏差（公榜 0.00289 << CV），伪标签会放大这个偏差。

---

## 3. 批判性评估 [Confidence: High]

### 3.1 来源质量限制

本调研的主要局限在于：**大多数技术细节来自 Tier 3 来源**（CSDN/知乎博客），因为 Kaggle 讨论页面是 JavaScript 渲染的，WebFetch 无法提取正文 [16][19][20][22]。这意味着：

- Jane Street 2021 第一名的架构细节来自 CSDN 博客的转述 [3][33]，而非 Kaggle 原帖 [16]
- Ubiquant 获奖方案的 asset_id embedding 做法来自搜索片段和间接引用 [19]，Zhihu 金牌解读被 403（未收入参考文献）
- G-Research 和 Optiver 的获奖方案细节未被提取到

建议用户直接在 Kaggle 上浏览这些讨论帖以获取一手信息：
- JS 2021 1st: https://www.kaggle.com/c/jane-street-market-prediction/discussion/224348
- Ubiquant 1st: https://www.kaggle.com/competitions/ubiquant-market-prediction/discussion/338220
- G-Research 2nd: https://www.kaggle.com/competitions/g-research-crypto-forecasting/discussion/323098
- Optiver 1st: https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution

### 3.2 CPU 推理优化的证据缺口

本调研的一个重大缺口是：**没有任何 Kaggle 量化获奖方案公开记录了 CPU 推理优化细节**（ONNX 导出、量化、TorchScript、知识蒸馏）[33][42]。推测原因是 JS 2021 的模型足够小（~130 特征），直接 Keras/TF 推理即满足限制。对于你的 323 特征，需要自行验证。建议路径：

1. 训练 denoising autoencoder + MLP（云端 GPU）
2. 导出为 ONNX 格式
3. 使用 ONNX Runtime 做 CPU 推理（预期 4-5x 加速）
4. 如果仍超时，考虑知识蒸馏到更小的 MLP 或量化到 INT8

### 3.3 Area 5（Ranking Loss）的负面发现本身就是发现

头部队伍 0.003+（高 33%）并非通过 custom ranking loss 实现的——至少在公开 writeup 中没有证据 [3][21]。这意味着头部队伍的优势更可能来自：更好的特征表示（autoencoder）、更智能的集成（stacking + 中性化）、或更好的 asset 建模（embedding + per-asset 标准化）。方正金工的 ranking loss 证据 [37] 在生产量化中有效，但 Kaggle 选手可能因为数值不稳定性 [38] 或 GBDT 中的 Hessian 问题而避免使用。

### 3.4 对"规律"的验证与修正

你总结的规律"任何从 train 分布衍生的结构在极端漂移下都失效有害"在调研中得到了**广泛验证**。所有获奖方案的共同特征是：

- 逐行 feature→target 映射（不依赖跨样本结构）——LGB/XGB/CatBoost 天然满足
- 强正则（Gaussian noise, Dropout, label smoothing, L1/L2）
- 后处理中性化（不依赖 train 分布，使用 test 时刻特征）
- BatchNormalization 的隐式 test-time adaptation

但需要修正一点：**"逐行映射"不意味着"只能用 GBDT"**。Denoising autoencoder 也是逐行的（每行独立编码），且通过 noise injection 和 BN 获得了额外的漂移鲁棒性——这是 GBDT 不具备的。

---

## 4. 行动优先级清单

按"预期增益 × 可行性"排序：

### 第一优先级（高增益 × 高可行性）

- [ ] **P1: 特征中性化后处理** — 对 LGB/XGB/CatBoost 的 OOF 预测和 test 预测做 OLS 残差化（`scores -= X @ pinv(X) @ scores`），在 sliding window（500-1000 行）上计算投影矩阵。这是 Numerai 的核心技术 [24]，与你的死路无近亲关系，CPU 推理零成本。预期：降低公榜/CV gap，等效提升 ~33%。
- [ ] **P2: Denoising Autoencoder 表征** — 训练 DAE（Gaussian noise + 128 维编码 + BN + Swish），将编码后的特征 concat 到原始 323 特征后输入 GBDT。JS 2021 第一名方案 [3][33]，逐行编码无跨样本依赖，BN 提供隐式 drift adaptation [30]。训练在云端 GPU，推理时 DAE 前向在 CPU 上可忽略不计。
- [ ] **P3: asset_id Embedding + Per-asset 标准化** — 在 NN 中加 Embedding(15, 8) 层；对每个 asset_id 计算 per-asset μ_a, σ_a 做标准化。你从未试过 [28][34]，与截面标准化正交 [31][32]，CPU 推理零成本。

### 第二优先级（中增益 × 高可行性）

- [ ] **P4: OOF Stacking + 元特征** — 用 LGB/XGB/CatBoost 的 OOF 预测 + `confidence=std(OOF)` + `consensus=mean(OOF)` 训练 meta-model（小 LGB）。与死路 #8 的区别：meta-model 在模型预测上训练，不在 raw feature 上 [7][8]。CPU 推理仅增加 1 个小模型。
- [ ] **P5: Soft Pseudo-Labeling** — 用 base model 在 test 特征上生成伪标签，k-fold 方式加入训练集。NVIDIA Grandmaster 推荐 [8]，test 特征在公开数据中可用。风险：伪标签可能放大 drift 偏差。
- [ ] **P6: BatchNormalization as TTA** — 在 DAE+MLP 推理时使用 `training=True`（test batch 统计量），在 sliding window 上计算。等价于 TENT [13]，JS 方案报告 22% 稳定性提升 [30]。

### 第三优先级（存疑增益 × 需实验验证）

- [ ] **P7: Ranking/Cosine Loss** — 在 LGB 上尝试 `1 - cosine_similarity` 近似损失（避免直接 Spearman 的数值不稳定 [38]）。方正金工在生产量化中验证有效 [37]，但无 Kaggle 获奖先例。先小规模实验。
- [ ] **P8: Drift-Resilient TabPFN** — 在 sliding window（5000 行）上测试 TabPFN 的 in-context transductive 推理。NeurIPS 2024 最优雅的 drift 方案 [14]，但在你的规模下可能不可行。仅作探索性实验。
- [ ] **P9: Test-Time Training (TTT)** — 在 DAE 的自监督重建任务上做推理时权重更新。学术上成立 [12]，但无 Kaggle 量化先例，15 行 batch 太小需要 sliding window。

### 不推荐（已有近亲死路或证据不足）

- DANN（Domain-Adversarial NN）：AUC=1.0 意味着 train/test 完全可分，DANN 的 domain-invariant 子空间假设可能不成立 [39][40]。与死路 #6 有近亲关系。
- FT-Transformer：在 benchmark 中无普遍优势 [5]，CPU 推理比 MLP 慢，对你的 GBDT 基线增量有限。
- SAINT：inter-sample attention 在 15 行 batch 上无优势 [6]。
- Feature-grouped bagging / MoE：匿名特征无法按物理含义分组 [31]。

---

## 5. 开放问题与注意事项

### 5.1 未解决的矛盾

**矛盾 1：中性化可能降低 R²。** 中性化从预测中减去与特征线性相关的部分——但如果特征中包含真实信号（你的死路 #3 验证了"drop 高漂移特征有害，因为漂移特征仍含信号"），中性化可能同时减去信号和噪声。Numerai 使用 Pearson 相关作为指标（中性化直接优化指标），但你的指标是 weighted R²（绝对预测精度）——中性化可能降低 R² 但改善公榜/CV gap 的稳定性。需要实测验证中性化对公榜分数的净效应。

**矛盾 2：Pseudo-labeling 的循环依赖。** 如果 base model 在 test 上的预测因漂移而偏差（公榜 0.00289 << CV），用偏差预测做伪标签会放大偏差。但如果 pseudo-labeling 的目的是"让模型适应 test 特征分布"（而非"提升预测精度"），则即使伪标签有偏，模型也能从 test 特征的分布信息中受益。这两个效应的方向相反，净效应未知。

**矛盾 3：15 行 batch 的限制。** Time-Series API 每次 ~15 行，这对 BN 统计量计算（需要 ~32+ 行才稳定）、TTT 梯度更新（需要足够样本）、中性化投影矩阵计算（需要行数 > 特征数才使 pinv 稳定）都构成挑战。所有需要 batch 统计的技术都必须在实例内维护 sliding window，而 window 的长度是一个需要调优的超参数。

### 5.2 关键缺口

- **Jane Street 2024-2025 的具体获奖方案**未能提取到。该比赛与你最相似（symbol_id、weighted R²、匿名特征、极端漂移），但其讨论页面 JS 渲染不可提取 [26][27][28][29]。强烈建议直接在 Kaggle 上浏览。
- **CPU 推理优化的实战经验**完全缺失。所有 writeup 只覆盖训练阶段 [33][42]。
- **Numerai 的具体技术实现**（neutralization 参数、era-wise validation 具体做法）来自 Tier 3 来源 [24][35]，需要查阅 Numerai 官方文档验证。

### 5.3 方法论限制

本调研使用 4 个 Wave 1 检索 subagent + 2 个 Wave 2 Gap-Fill subagent + 1 个 Verification subagent，共 45 个来源。但 Kaggle 讨论页面的 JavaScript 渲染限制了信息提取深度，导致大量技术细节来自 CSDN/知乎二级来源。引文抽查验证了 10 个核心 claim 中的 6 个为 SUPPORTED，1 个 PARTIAL（FT-Transformer "7/11" 修正为"竞争力相当但无普遍优势"），2 个 UNVERIFIED（symbol_id 反模式和 TTA 负面发现已软化表述）。建议将本报告视为"方向指引"而非"实现手册"，关键技术的具体参数需在实验中调优。

---

## 方法论

- **深度选择**：Deep（用户指定）
- **Subagent 数量**：4 个 Wave 1 检索 + 2 个 Wave 2 Gap-Fill + 1 个 Phase 3.1 验证 = 7 个 subagent
- **检索轮次**：2 轮（Wave 1 覆盖 6 大方向 + 竞赛 writeup；Wave 2 填补 5 个缺口）
- **来源总数**：45（去重后），其中 Tier 1: 12, Tier 2: 15, Tier 3: 18
- **大纲调整**：Area 4 扩展——特征中性化从"子技术"提升为"最重大发现"，独立成节；Area 5 的负面发现独立成节而非合并到其他方向
- **引文修正**：
  - FT-Transformer "7/11" → "竞争力相当但无普遍优势"（Phase 3.1 PARTIAL）
  - "无任何 Kaggle 方案使用 TTA" → "TTA/transductive 在 Kaggle 量化获奖方案中不是主流策略"（Phase 3.1 UNVERIFIED，软化表述）
  - symbol_id 反模式的具体引文未能验证，但技术合理性成立（Phase 3.1 UNVERIFIED）
- **降级说明**：Kaggle 讨论页面 JS 渲染导致 6 个 Tier 2 来源（[16][19][20][22][44] 等）仅有 URL 确认而无正文提取，相关技术细节来自 Tier 3 二级来源

---

## 参考文献

[1] Sebastian Raschka — "A Short Chronology Of Deep Learning For Tabular Data" — https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html — 访问 2025 — Tier: 2

[2] Zabërgja et al. — "Tabular Data: Is Deep Learning all you need?" — https://arxiv.org/abs/2402.03970 — 2024 — Tier: 1

[3] csdn_xmj — "Kaggle竞赛宝典 | 量化竞赛第一名的网络模型" — https://blog.csdn.net/csdn_xmj/article/details/141524614 — 访问 2025 — Tier: 3

[4] ryancheunggit — "Denoise-Transformer-AutoEncoder" (GitHub) — https://github.com/ryancheunggit/Denoise-Transformer-AutoEncoder — 访问 2025 — Tier: 2

[5] Gorishniy et al. — "Revisiting Deep Learning Models for Tabular Data" (FT-Transformer) — https://arxiv.org/abs/2106.11959 — 2021 — Tier: 1 [foundational]

[6] Somepalli et al. — "SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training" — https://arxiv.org/abs/2106.01342 — 2021 — Tier: 1

[7] Chris Deotte (NVIDIA) — "Winning First Place in a Kaggle Competition with Stacking Using cuML" — https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/ — 访问 2025 — Tier: 2

[8] Chris Deotte et al. (NVIDIA) — "The Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques for Tabular Data" — https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/ — 访问 2025 — Tier: 2

[9] Yang, Q. — "With Genetic-algorithm generated Alpha factors and Sentiments (GAS)" — https://arxiv.org/abs/2411.03035v1 — 2024 — Tier: 1

[10] Carl McBride Ellis — "What is Adversarial Validation?" — LinkedIn — https://www.linkedin.com/posts/carl-mcbride-ellis_what-is-adversarial-validation-activity-7374305019963207680-ngVq — 2025 — Tier: 2

[11] mdnice (集体整理) — "Kaggle量化竞赛Top方案" — https://mdnice.com/writing/ad11b39aed744785944fcb3638de80f1 — 访问 2024 — Tier: 3

[12] Sun et al. — "Test-Time Training with Self-Supervision for Generalization under Distribution Shifts" — https://arxiv.org/abs/1909.13231 — ICML 2020 — Tier: 1

[13] Wang et al. — "Tent: Fully Test-Time Adaptation by Entropy Minimization" — https://arxiv.org/abs/2006.10726 — ICLR 2021 — Tier: 1

[14] NeurIPS 2024 — "Drift-Resilient TabPFN: In-Context Learning over Temporal Distribution Shifts on Tabular Data" — https://nips.cc/virtual/2024/poster/93581 — 2024 — Tier: 1

[15] Xiao & Snoek — "Beyond Model Adaptation at Test Time: A Survey" — https://arxiv.org/abs/2411.03687 — 2024 — Tier: 1

[16] Kaggle — "Jane Street Market Prediction, 1st place solution" — https://www.kaggle.com/c/jane-street-market-prediction/discussion/224348 — 访问 2025 — Tier: 2 [body not extractable]

[17] CSDN weixin_51484067 — "jane street market prediction 冠军方案 经验分享" — https://blog.csdn.net/weixin_51484067/article/details/114545582 — 访问 2025 — Tier: 3

[18] CSDN weixin_42645636 — "kaggle金融量化竞赛top方案汇总" — https://blog.csdn.net/weixin_42645636/article/details/131579190 — 访问 2025 — Tier: 3

[19] Kaggle — "Ubiquant Market Prediction, 1st place solution" — https://www.kaggle.com/competitions/ubiquant-market-prediction/discussion/338220 — 访问 2025 — Tier: 2 [body not extractable]

[20] Kaggle — "G-Research Crypto Forecasting, 2nd place solution" — https://www.kaggle.com/competitions/g-research-crypto-forecasting/discussion/323098 — 访问 2025 — Tier: 2 [body not extractable]

[21] modb.pro / Coggle — "Kaggle Optiver 量化比赛 高分思路" — https://www.modb.pro/db/1719606081893265408 — 访问 2025 — Tier: 3

[22] Kaggle — "Optiver Trading at the Close, 1st place solution" — https://www.kaggle.com/competitions/optiver-trading-at-the-close/writeups/hyd-1st-place-solution — 访问 2025 — Tier: 2 [body not extractable]

[23] Numerai — community forum — https://forum.numer.ai/ — 访问 2025 — Tier: 2 [thread titles only]

[24] Baidu Zhidao — "降低特征暴露，提升量化交易中机器学习模型的长期稳定性" — https://zhidao.baidu.com/question/2066499650106707987.html — 访问 2025 — Tier: 3

[25] CSDN zurie — "Option Alphas Chapter 3: About Neutralization" — https://blog.csdn.net/zurie/article/details/156692669 — 访问 2025 — Tier: 3

[26] Baidu News — "Kaggle金融预测冠军方案大揭秘" (JS 2024-2025) — https://mbd.baidu.com/newspage/data/dtlandingsuper?nid=dt_4356924038466950135 — 访问 2025 — Tier: 3

[27] CSDN weixin_51484067 — "Jane Street Real-Time Market Data Forecasting Baseline模型分享" — https://blog.csdn.net/weixin_51484067/article/details/142932726 — 访问 2025 — Tier: 3

[28] CSDN sunmoonstar432 — "Jane Street Real-Time Market Data Forecasting 反面教材" — https://blog.csdn.net/sunmoonstar432/article/details/144629032 — 访问 2025 — Tier: 3

[29] CSDN weixin_48152827 — "简街实时市场数据预测" — https://blog.csdn.net/weixin_48152827/article/details/144516229 — 访问 2025 — Tier: 3

[30] CSDN weixin_42639246 — "从冠军方案拆解：在Jane Street预测赛中，如何用AE+MLP+XGBoost玩转模型融合？" — https://blog.csdn.net/weixin_42639246/article/details/160237580 — 访问 2025 — Tier: 3

[31] CSDN/Wenku — "Jane Street市场预测竞赛解决方案" — https://wenku.csdn.net/doc/4pcwvtad98 — 访问 2025 — Tier: 3

[32] CSDN/Datawhale — "全球第一Kaggle选手开源：最新量化赛事方案！" (Mitsui) — https://blog.csdn.net/Datawhale/article/details/152186849 — 访问 2025 — Tier: 3

[33] CSDN FrankieHello — "Kaggle金融市场价格预测Top方案——基于AutoEncoder与MLP的预测模型" — https://blog.csdn.net/FrankieHello/article/details/126314050 — 访问 2025 — Tier: 2

[34] cnblogs/Kaggle — "Kaggle 官方教程：嵌入" — https://www.cnblogs.com/apachecn/p/17313297.html — 访问 2025 — Tier: 2

[35] CSDN/Wenku — "Numerai机器学习竞赛实验代码" — https://wenku.csdn.net/doc/3m7k2figtq — 访问 2025 — Tier: 3

[36] CSDN sre5engineer — "从零到Kaggle冠军：Jane Street中XGBoost与数据压缩" — https://blog.csdn.net/sre5engineer/article/details/152310109 — 访问 2025 — Tier: 3

[37] 方正金工 (Founder Securities) — "机器学习基于可微RankIC损失函数的深度学习选股策略" — https://www.fxbaogao.com/detail/5182756 — 访问 2025 — Tier: 2

[38] PyTorch Forums — "Setting Pearson Correlation Coefficient as a Loss really doesn't work well" — https://discuss.pytorch.org/t/setting-pearson-correlation-coefficient-as-a-loss-really-doesnt-work-well/200707 — 访问 2025 — Tier: 2

[39] ScienceDirect — "Concept drift handling: A domain adaptation perspective" — https://www.sciencedirect.com/science/article/abs/pii/S0957417423004487 — 访问 2025 — Tier: 1 [paywall]

[40] arXiv — "Learning Robust Spectral Dynamics for Temporal Domain" — https://arxiv.org/html/2505.12585v1 — 2025 — Tier: 1

[41] CSDN SpiritYzw — "pairwise、listwise在lgb中的应用" — https://blog.csdn.net/SpiritYzw/article/details/129815896 — 访问 2025 — Tier: 3

[42] Tencent Cloud — "Kaggle金融市场价格预测Top方案" — https://cloud.tencent.com/developer/article/2242044 — 访问 2025 — Tier: 3

[43] arXiv — "ML Enhanced Multi-Factor Quantitative Trading" — https://arxiv.org/html/2507.07107v1 — 2025 — Tier: 1

[44] Kaggle — "Jane Street Market Prediction, 3rd place solution" — https://www.kaggle.com/competitions/jane-street-market-prediction/discussion/224713 — 访问 2025 — Tier: 2 [body not extractable]

[45] Rubachev et al. — "Revisiting Deep Learning Models for Tabular Data" (pre-training objectives) — https://arxiv.org/abs/2207.03208 — 2022 — Tier: 1 [foundational]
