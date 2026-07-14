# 量化比赛 2026 — 突破策略详案

> **当前状态**：公榜 0.00288933（第22名）| 头部 19 队 0.003+ | 最高 0.0038378 | 差距约 33%
> **核心瓶颈**：强分布漂移（train vs test adversarial AUC=1.0），CV 不可信（公榜/CV≈1.6），raw feature + 树模型汇聚至 ~0.00289 天花板

---

## 一、问题根因分析

### 1.1 为什么到了天花板

记忆文件已充分证明：LGB（0.0028205）、XGB（0.00284690）、Cat（0.00285836）三个异构 backend 各自独立达到相近水平，三模型集成（0.00288933）也只比最强单模型高约 1%。这说明**不是模型容量或集成多样性不够，而是特征侧的信息量本身在漂移下衰减到了极限**。

### 1.2 为什么之前的尝试都失败了

| 尝试方向 | 操作类型 | 失败原因 |
|---------|---------|---------|
| cs 截面特征 | 跨样本/同feature变换 | 全局截面统计量在漂移下不稳定 |
| rolling 时序特征 | 跨时间/同feature变换 | 时序模式在 train/test 之间完全不同 |
| drop-drift feature | 删除漂移特征 | 漂移特征仍含预测信号，删除损失 > 漂移危害 |
| responder 堆叠 | 两阶段预测 | pseudo responder 预测极弱（corr 0.06），无增量 |
| w^α 加权 | 调整样本权重 | 零和游戏，高weight样本R²低是噪声特性非权重不够 |
| 最近数据训练 | 缩小时间距离 | train 末段与 test 漂移同样大，任何 train 段都不行 |

**共性规律**：所有跨样本/跨时间的特征变换都过拟合，因为 train 的截面结构和时序模式在 test 中不成立。

### 1.3 与头部队伍的差距在哪

33% 的差距不是参数调优能弥补的。头部队伍大概率在以下至少一个维度上做了我们没有的事：

- 有一个**可靠的验证机制**，能在本地判断策略好坏，从而快速迭代
- 对**目标变量做了变换**，使其在漂移下更稳定
- 构造了**质上不同的特征**（非跨样本/跨时间，而是跨 feature 或基于局部近邻）

---

## 二、五个探索方向

### 方向一：对抗验证选验证集（基础设施，最优先）

**核心思路**：对抗验证不是为了删特征（已验证有害），而是为了选出训练集中与测试集最相似的样本作为验证集，让 CV 逼近公榜。

**为什么这是最重要的**：当前每天 5 次公榜提交是唯一可靠判据，所有实验必须烧公榜才能判断好坏。如果 CV 可信，迭代速度提升 10 倍以上，后续所有方向都能快速验证。

**实现步骤**：

1. 合并 train 和 test 数据，构造二分类标签（train=1, test=0）
2. 用 LightGBM 训练对抗分类器，对所有 train 样本输出属于 train 的概率
3. 概率越低的 train 样本越像 test（分类器难以将其与 test 区分）
4. 在 time CV 的每个 fold 中，选择**概率最低的连续 time_id 段**作为验证集
5. 用这个验证集重新评估现有模型，验证 CV 与公榜的相关性是否改善

**关键细节**：

```python
# 伪代码：对抗验证选验证集

# 1. 训练对抗分类器
train['is_train'] = 1
test['is_train'] = 0
combined = pd.concat([train[features], test[features]])
labels = np.concatenate([np.ones(len(train)), np.zeros(len(test))])

adv_model = lgb.LGBMClassifier(...)
adv_model.fit(combined, labels)

# 2. 获取 train 样本的"像 train 程度"
train_scores = adv_model.predict_proba(train[features])[:, 1]
# score 越低的样本越像 test

# 3. 按 time_id 聚合，选出最像 test 的时间段
time_adv_score = train.groupby('time_id')['adv_score'].mean()
# 选 adv_score 最低的 time_id 段作为验证集

# 4. 在 time CV 中使用这些 time_id 作为 valid
# 确保 valid 的 time_id 在 train 的 time_id 之后（时序约束）
```

**备选方案**：如果按 time_id 聚合后选出的验证段与原有 CV 划分差异太大（违背时序约束），可以退一步——在保持时间顺序的前提下，从每个 fold 的训练部分中按 adv_score 加权采样，让训练集也向 test 分布靠拢。

**预期收益**：CV 与公榜相关性从 ~1.6 倍偏差缩小至接近 1:1，后续所有实验不再依赖公榜验证。

---

### 方向二：Target 工程（标签侧改造）

**核心思路**：不改变特征，而是改变预测目标，使其在分布漂移下更稳定。这是质上完全不同于之前所有尝试的方向。

#### 2.1 按 time_id 截面标准化

每个 time_id 内对 target 做 z-score 标准化，模型预测标准化后的 target。推理时每个 time_id 独立传入，无法做截面标准化，但模型输出的相对排序信号可直接使用。

```python
# 训练时
train['target_z'] = train.groupby('time_id')['target'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# 推理时直接预测 target_z，按原始 target_std 裁剪
```

**为什么有效**：不同 time_id 的 target 分布（均值、方差）天然不同，且 train 和 test 的时间段完全不同。标准化消除了这种跨时间段的分布差异，让模型专注于学习"同一个截面上谁好谁坏"的排序能力，而非跨截面的绝对量级。

#### 2.2 Rank/分位数变换

在每个 time_id 内将 target 映射到 [0,1] 分位数。

```python
train['target_rank'] = train.groupby('time_id')['target'].rank(pct=True)
```

**为什么有效**：分位数天然对异常值鲁棒，且不受 target 分布形状变化的影响。train 和 test 的 target 量级可能完全不同，但"前 10% 的样本"这个概念是稳定的。

#### 2.3 方向+强度分离

将 target 拆成两个子任务：
- 二分类：预测 target > 0（方向）
- 回归：预测 |target|（强度）

两个子任务各自建模型，最终预测 = sign_prob × magnitude。

**为什么有效**：方向预测本质上是对分布漂移更鲁棒的排序问题。在强漂移下，模型判断"哪个资产更好"的方向信号可能比判断"好多少"的量级信号更稳定。分离后各自优化，不互相干扰。

---

### 方向三：特征交互与比值（特征侧新维度）

**核心思路**：构造跨 feature 的交互项（乘积、比值），与之前试过的跨样本/跨时间变换质上完全不同。

**为什么之前的 cs/rolling 失败而这个可能成功**：

- cs/rolling → 操作在"样本轴"和"时间轴"上 → 依赖 train 的截面结构和时序模式在 test 中成立 → 不成立
- 特征交互 → 操作在"特征轴"上 → 同一个样本内 feature_A 和 feature_B 的关系 → 这种物理关系在 train/test 之间更稳定

**实现方案**：

```python
# 选出 importance 最高的 top-k 特征（建议 50-100）
important_features = get_top_k_features(model, k=80)

# 构造 pairwise 交互
for i in range(len(important_features)):
    for j in range(i+1, len(important_features)):
        f_i, f_j = important_features[i], important_features[j]
        # 乘积
        data[f'{f_i}_mul_{f_j}'] = data[f_i] * data[f_j]
        # 比值（需要处理分母为0）
        data[f'{f_i}_div_{f_j}'] = data[f_i] / (data[f_j] + 1e-8)
```

**建议分批尝试**：
- 第一批：top-30 特征的 pairwise 交互 → 435 个新特征
- 第二批：top-50 特征 → 1225 个新特征
- 第三批：top-100 特征 → 4950 个新特征

每批训练后用 feature importance 筛选出真正有用的交互项，下一批只保留有效的。

**参考依据**：特征交互与组合策略在量化金融中有成熟的物理意义（价量比、波动率比等），匿名特征虽然无法解释，但数学上等价。

---

### 方向四：Nearest-Neighbor 聚合特征（局部结构）

**核心思路**：对每个样本找 k 近邻，用邻居的特征统计量作为新特征。这与 cs 特征的核心区别在于——cs 是全局统计（所有样本），NN 是局部统计（只找最相似的样本）。

**为什么 cs 失败而 NN 可能成功**：

cs 特征的问题是：一个样本的"截面排名"是用全 time_id 的均值/std 算的，test 的全局分布和 train 不同，这个排名就偏了。NN 只看局部：在 test 中找一个样本的最近邻，这些邻居都在**同一个 test time_id 内**，分布是自洽的。

```python
from sklearn.neighbors import NearestNeighbors

# 对每个 time_id 独立计算近邻
for time_id in data['time_id'].unique():
    mask = data['time_id'] == time_id
    X = data.loc[mask, features].values
    
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    
    # 聚合邻居特征
    for agg_func in [np.mean, np.std, np.min, np.max]:
        agg_name = agg_func.__name__
        neighbor_agg = np.array([
            agg_func(X[indices[i][1:]], axis=0)  # 排除自己
            for i in range(len(X))
        ])
        for j, feat in enumerate(features):
            data.loc[mask, f'nn_{agg_name}_{feat}'] = neighbor_agg[:, j]
```

**关键参数**：k（近邻数量）和 distance metric。建议 k=10~30，metric 可以尝试 euclidean、manhattan、cosine。

**参考依据**：Optiver 收盘交易比赛冠军方案使用了 7 类 nearest-neighbor 聚合特征，被其描述为"获胜的关键"。虽然比赛不同，但局部结构捕捉的思路对匿名特征的量化场景高度通用。

---

### 方向五：Denoising Autoencoder 特征表示

**核心思路**：训练一个有监督的自编码器，在重构特征的同时学习对预测有用的隐含表示。

**架构设计**（参考 Jane Street 获胜方案）：

```
输入(323 features) → Gaussian Noise → Encoder → Bottleneck(64-128维) → Decoder → 重构(323 features)
                                                       ↓
                                                  MLP Head → target 预测
```

训练时三路 loss：重构 MSE + 预测 loss（加权 R² 或 RMSE）+ 可选正则项。

**为什么有用**：自编码器学到的隐层表示是特征的压缩和去噪版本，天然过滤掉了对重构无用的噪声维度。加入有监督的预测 head 后，bottleneck 会保留对预测最有用的信息，同时抑制漂移引起的虚假模式（因为这些模式对重构无帮助）。

**面临的约束**：
- 评测环境 4 核 / 无 GPU → 不能跑大网络
- 但可以在云端训练，导出 weights，推理时只做前向传播（CPU 上一个 323→128→64→128→323 的小网络前向耗时可控）

**性价比评估**：考虑到 4 核无 GPU 的推理约束和训练调试成本，此方向优先级低于前四个。但如果前四个方向都尝试后仍有差距，这是最有力的差异化方向。

---

## 三、推荐行动路线

### 第一阶段：建立可靠验证（预计 1-2 天）

1. 实现对抗验证选验证集（方向一）
2. 在单 partition 上验证 CV 与公榜的相关性是否改善
3. 如改善明显，在多 partition 上确认

**判定标准**：新 CV 的排名/趋势与公榜一致，不需要完全等值，只要方向对即可。

### 第二阶段：并行探索（预计 3-5 天）

有了可靠 CV 后，可以并行尝试：

- 方向二（Target 工程）：在实验成本极低（只改标签不增加特征维度），可以三个子方向一起试
- 方向三（特征交互）：先试 top-30 的交互项，如果 CV 正向再扩大到 top-50/100

方向四（NN 特征）可以同步准备代码，在第二阶段后期加入。

### 第三阶段：组合与调优（预计 2-3 天）

将第二阶段验证有效的方向组合：
- 最优 target 变换 + 最优特征交互 + 原始特征 → 训练三 backend → 集成
- 确认组合后是否有叠加效应（1+1>2）还是替代效应（最优单项就够）

### 第四阶段（如有时间）：方向五

Denoising Autoencoder 作为终极差异化手段。

---

## 四、风险与注意事项

1. **对抗验证选验证集可能遇到的问题**：如果 train 中"最像 test"的样本全部集中在某个特定时期（如 train 末段），但这段时间的预测信号恰好很弱（记忆文件提到末段 valid 越接近 test 越差），需要灵活调整——不是简单选"最像"，而是选"最像且信号不弱"的。

2. **特征交互的维度爆炸**：top-100 的 pairwise 产生 4950 个新特征，加上原始 323 共 5273 个。需要确保训练时内存和时间可控。建议每批后用 feature importance 裁剪。

3. **Target 变换的还原问题**：如果用标准化或分位数，推理时预测的是变换后的值。对于分位数方法，需要确定如何映射回原始 target 量级（可能需要基于训练集的 target 分布做逆变换）。

4. **每日 5 次公榜提交的分配**：第一阶段验证对抗验证时可能需要 2-3 次公榜提交。为第二阶段留至少 2-3 次/天。

---

## 五、参考来源

本次调研主要参考了以下公开的量化比赛方案和经验：

- Kaggle Optiver 收盘交易比赛 — 冠军方案的 7 类 nearest-neighbor 聚合特征
- Kaggle Jane Street 市场预测 — 有监督 Denoising Autoencoder + XGBoost 集成方案
- Kaggle Ubiquant 市场预测 — 多模型集成与对抗验证实践
- 对抗验证（Adversarial Validation）在 Kaggle 中的通用实践 — 特征筛选与验证集选择
- 量化特征工程中的特征交叉与组合策略
