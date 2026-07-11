# 2026量化比赛改进计划

> 当前状态: dev 分支, LightGBM 基线, CV mean=0.001815, 仅用323个原始特征
> 目标: 系统性提升 CV 分数, 在 8/31 前完成策略提交
> 时间: 7/10 - 8/31 (约7周)

---

## 核心诊断：当前模型的三个关键瓶颈

1. **特征未启用** — `features.py` 已实现截面特征(rank/zscore/demean)和滚动特征(lag1/rolling_mean/rolling_std), 但部署的模型只用了323个原始 feature_*。`model_meta.json` 显示 `n_features: 323, rolling_windows: []`。

2. **推理管线断裂** — `strategy/main.py` 的 `predict()` 只做 `test.loc[:, self.feature_columns]`, 无法计算截面特征和滚动特征。即使训练时启用了这些特征, 推理时也无法对齐。

3. **单模型无调参** — 使用一组手写默认参数的 LightGBM, 无超参搜索, 无集成, 无自定义损失函数。

---

## 阶段一：补齐基线 (7/10 - 7/16, 约1周)

目标: 把已实现但未启用的能力全部打通, 建立可验证的完整管线。

### 1.1 启用截面特征训练

截面特征在 `features.py` 中已实现: 每个 time_id 内对15个 asset 计算 rank、zscore、demean。严格因果, 推理时可实时重算。

- 执行: `python train.py --cross-sectional --save-model --fresh`
- 预期: 特征数 323 → 1292 (323×4), CV 分数应有提升
- 验证: 检查 `model_meta.json` 中 `n_features` 和 `cross_sectional` 字段

### 1.2 修复推理管线支持截面特征

`strategy/main.py` 的 `predict()` 需要在每个 time_id 步骤内实时计算截面特征:

```
# 伪代码
for each feature_col:
    rank = fractional_rank(test[feature_col])  # 当前 time_id 内15个 asset
    zscore = (test[feature_col] - mean) / std
    demean = test[feature_col] - mean
拼接 [raw, rank, zscore, demean] → 送入 booster
```

关键: 每个 time_id 只有15行(15个 asset), 截面统计就在这15行上计算。需确保列顺序与训练完全一致。

### 1.3 启用滚动特征训练

选择初始窗口 `[5, 10, 20]`:

- 执行: `python train.py --cross-sectional --rolling-windows 5 10 20 --save-model --fresh`
- 特征数估算: 323(raw) + 323×3(cs) + 323×1(lag1) + 323×3×2(rm+rs) = 323 + 969 + 323 + 1938 = 3553
- 注意: 内存控制, 可能需要 `--partitions` 限制分区数

### 1.4 修复推理管线支持滚动特征

推理时需维护 per-asset 的历史 deque:

```
# __init__ 中初始化
self.asset_history = {}  # asset_id -> deque(maxlen=max_window)

# predict 中
for asset_id in test["asset_id"]:
    history = self.asset_history.get(asset_id, deque())
    # 用 history 计算 lag1, rolling_mean, rolling_std
    # 更新 history: append 当前 feature 值
    self.asset_history[asset_id] = history
```

关键: 必须在 `predict()` 返回后才更新历史, 不能用当前步的值计算当前步的滚动特征(否则泄露)。lag1 = 上一步的值, rolling_mean/std = 过去 w 步的统计。

### 1.5 增加 CV 折数并加入 purge gap

当前 3 折, valid_frac=0.1。改为 5 折, 并在训练集和验证集之间加入 gap:

- 修改 `src/cv.py`: 在 `time_cv_split` 中增加 `gap` 参数, valid_start 之前留出 gap 个 time_id 不用
- gap = 5~10 个 time_id, 减少时序自相关导致的验证分数虚高
- 执行: `python train.py --n-folds 5 --valid-frac 0.08 --save-model --fresh`

### 阶段一交付物

- [x] 截面+滚动特征训练通过
- [x] 推理管线支持全部特征
- [x] `timeseries_api/runner.py` 本地验证通过
- [x] CV 分数 ≥ 0.0025 (预期)
- [x] 提交公榜验证

---

## 阶段二：特征工程深化 (7/17 - 7/30, 约2周)

目标: 在完整管线基础上, 通过更丰富的特征提升信号捕捉能力。

### 2.1 扩展滚动窗口和统计量

在 `features.py` 中扩展:

- 滚动窗口: `[3, 5, 10, 20, 40]` (覆盖短中长期)
- 统计量: 增加 `rolling_min`, `rolling_max`, `rolling_skew`, `rolling_kurt`
- lag: 增加 `lag2, lag3, lag5`
- 动量特征: `(current - rolling_mean) / rolling_std` (标准化离差)
- 加速度: `lag1 - lag2` (二阶差分)

注意特征数爆炸问题: 323 × (3cs + 5lag + 5window×6stat) ≈ 323 × 38 ≈ 12274。需配合特征选择。

### 2.2 跨 asset 特征

15个 asset 在同一 time_id 的关系:

- 截面均值/排名已覆盖基础
- 增加: asset 间特征差分 (当前 asset vs 截面 top asset)
- 增加: asset 的历史排名变化 (rank 动量)
- 增加: asset pair 相关性 (选取 top-K 高相关 pair)

### 2.3 特征交互

- 对 LightGBM feature importance top-20 的特征, 构造 pairwise 交互 (乘法、除法)
- 避免暴力全组合 (323×322/2 ≈ 5万), 只选重要特征

### 2.4 特征选择

训练后做:
- LightGBM `feature_importance(split/gain)` 排序, 保留 top-K
- Null importance: 打乱 target 后重新训练, 对比 importance, 过滤噪声特征
- SHAP 值聚合, 识别真正贡献特征

目标: 从上万个特征中筛出 1000-2000 个有效特征, 控制模型大小和推理时间。

### 2.5 时间结构特征

- `time_id` 的周期性分解 (mod 操作, 假设存在日内/日间周期)
- `time_id` 的趋势分量
- asset 在 time_id 上的活跃度 (是否出现在该 time_id)

### 阶段二交付物

- [x] 特征工程模块 v2 (扩展后的 `features.py`)
- [x] 特征选择脚本
- [x] CV 分数 ≥ 0.004 (预期)
- [x] 公榜分数验证

---

## 阶段三：模型增强 (7/24 - 8/6, 约2周, 与阶段二后半重叠)

目标: 从单模型升级为多模型集成, 优化超参和目标函数。

### 3.1 超参搜索 (Optuna)

用 Optuna 对 LightGBM 做贝叶斯优化, 目标函数 = CV mean score:

- 搜索空间: `num_leaves(31-255)`, `learning_rate(0.01-0.1)`, `n_estimators(500-5000)`, `min_child_samples`, `subsample`, `colsample`, `reg_alpha`, `reg_lambda`
- 约束: 理论最多 50-100 次 trial (每次需训练5折)
- 早停: 用 Hyperband 或 MedianPruner 加速

注意: 必须在本地完整跑, 不能依赖云端。每次 trial 约 30-60 分钟, 50 次约 25-50 小时。

### 3.2 自定义目标函数

比赛指标是加权零均值 R², LightGBM 默认用 MSE。虽然不能用 R² 直接做目标(不可微的0处理), 但可以用加权 MSE:

```python
# 加权 MSE objective
def weighted_mse(y_true, y_pred, weight):
    grad = -2 * weight * (y_true - y_pred)
    hess = 2 * weight
    return grad, hess
```

这会让模型更关注高权重样本。同时调整 eval_metric 为加权 R² 而非 L2。

### 3.3 多模型集成

- LightGBM (已有): 树模型, 擅长非线性交互
- XGBoost: 不同的树生长策略, 增加多样性
- CatBoost: 对类别特征友好, 对称树结构提供不同视角
- 每个模型用不同随机种子训练 3-5 个, 取平均

集成方式:
- 简单平均 (baseline)
- 加权平均 (按 CV 分数加权)
- Stacking (用简单线性模型做 meta-learner)

注意: 最终推理环境 4核/12GB/无GPU, 需控制模型总大小和推理时间。建议 3-5 个 booster 的简单平均。

### 3.4 Per-asset 模型

15 个 asset 可能有不同的行为模式:
- 对每个 asset_id 单独训练一个轻量 LightGBM
- 或在全局模型基础上, 为每个 asset 加一个修正项
- 验证: 对比全局模型 vs per-asset 模型的 CV 分数

### 3.5 多种子 Bagging

- 固定超参, 变化 `random_state` (42, 123, 456, 789, 2024)
- 5 个模型的预测取平均, 降低方差

### 阶段三交付物

- [x] Optuna 调参结果
- [x] 2-3 种模型的集成管线
- [x] 自定义目标函数实现
- [x] CV 分数 ≥ 0.006 (预期)
- [x] 推理时间 < 限制 (需实测)

---

## 阶段四：Responder 辅助策略 (7/31 - 8/13, 约2周, 与阶段三重叠)

目标: 利用训练集中47个 responder 增强模型, 但推理时不依赖 responder。

### 4.1 Residual Stacking (残差堆叠)

两阶段训练:

1. 第一阶段: 用 feature_* 训练 responder 预测模型 (47个或 top-15 个 responder)
2. 第二阶段: 将第一阶段的 responder 预测作为额外特征, 训练 target 模型

推理时: 只有 feature_* → 先预测 responder → 再预测 target。

关键: responder 模型在 CV 中也必须用 train 折训练, 不能在 valid 折上用全量训练的 responder 模型, 否则泄露。

### 4.2 多任务学习

在 LightGBM 中不太直接, 但可以:
- 训练一个 target 模型 + N 个 responder 模型
- 用 responder 的 feature importance 做交叉验证: 如果某个 feature 对多个 responder 都重要, 它可能包含更通用的信号
- 用 responder 预测值做 feature selection: 只保留与 target 高相关且与 responder 也相关的 feature

### 4.3 Responder 标签回补利用

8/23 公榜截止后会发布扩展训练数据 (responder 标签回补到测试集):
- 用回补后的 responder 做半监督学习: 如果 responder 在 test 上的值与 train 一致, 可以用 responder 预测做 pseudo-label
- 扩大训练集: 把 test 数据的 feature + 预测 responder 纳入训练

注意时间节点: 8/23 才回补, 只有8天到截止。需要提前准备好代码框架。

### 阶段四交付物

- [x] Residual stacking 管线
- [x] Responder 辅助特征选择
- [x] 8/23 标签回补的半监督学习框架(代码就绪)
- [x] CV 分数 ≥ 0.008 (预期)

---

## 阶段五：最终打磨 (8/7 - 8/31, 约3.5周, 与前面阶段重叠)

### 5.1 预测后处理

- 预测值裁剪: clip 到 [target 的 1%-99% 分位数], 防止极端值拖低分数
- 截面中性化: 对每个 time_id 的预测做去均值(如果 target 是零均值的)
- 分位数对齐: 将预测的分位数分布对齐到训练集 target 的分布

### 5.2 时间衰减权重

近期数据可能更代表未来分布:
- 在 `sample_weight` 上乘以时间衰减因子: `w * exp(-alpha * (t_max - t))`
- alpha 通过 CV 搜索最优值

### 5.3 推理性能优化

在 4核/12GB 环境下:
- 实测推理时间: 用 `timeseries_api/runner.py` 跑完整 test 集
- 内存峰值: 确保模型 + 特征矩阵 < 10GB
- 如果超出: 减少模型数量, 减少特征数, 使用 float16 推理
- 滚动特征的 deque 内存: 15 asset × 323 feature × 40 window × 4 bytes ≈ 770KB, 无问题

### 5.4 提交策略

- 公榜阶段 (8/23前): 每天最多5次提交, 用于验证管线和快速迭代
- 策略提交 (8/31前最多10次): 谨慎使用, 每次提交前必须通过本地 runner 验证
- 保留2-3次策略提交用于最后调整

### 5.5 报告准备

10月答辩需要提交报告:
- 记录每次实验的 CV 分数、公榜分数、改动内容
- 保留特征重要性分析、EDA 结论
- 准备方法论说明: 特征工程、模型架构、验证策略

### 阶段五交付物

- [x] 后处理管线
- [x] 推理性能达标
- [x] 策略 zip 包提交
- [x] 实验记录文档

---

## 技术风险和缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| 滚动特征推理时 deque 状态管理复杂 | 推理错误或超时 | 用 `timeseries_api/runner.py` 完整验证, 加单元测试 |
| 特征数爆炸导致内存不足 | 训练 OOM | 分区加载, 特征选择, 控制特征在 2000 以内 |
| Optuna 搜索耗时过长 | 延误进度 | 用 Hyperband 早停, 限制 trial 数, 可用较少分区做快速搜索 |
| 8/23 标签回补后只有8天 | 半监督策略来不及 | 提前写好代码框架, 回补后只需改数据路径 |
| 推理环境无GPU/4核/12GB | 模型大小受限 | 模型文件控制在 10MB 以内, 推理时间 < 现有限制 |
| LazyFrame collect 死锁 | 全量重训失败 | 已知问题, 用分区加载或最后一折部署策略规避 |

---

## 优先级排序 (如果时间不够)

1. 阶段一全部 (必须, 最高性价比)
2. 阶段三的 3.1 超参搜索 + 3.5 多种子 (简单有效)
3. 阶段二的 2.4 特征选择 (控制规模)
4. 阶段三的 3.3 多模型集成 (显著提升)
5. 阶段四的 4.1 residual stacking (如果 responder 有信号)
6. 阶段五的 5.1 后处理 + 5.3 推理优化 (必须)
7. 其他 (时间允许时做)

---

## 立即行动项 (今天)

1. 在 dev 分支上创建 feature 分支: `git checkout -b feature/cross-sectional-rolling`
2. 运行 `python train.py --cross-sectional --rolling-windows 5 10 20 --n-folds 5 --save-model --fresh` 观察 CV 变化
3. 同步修改 `strategy/main.py` 支持截面+滚动特征的推理
4. 用 `timeseries_api/run_timeseries_api.py` 跑本地验证
5. 提交公榜看分数变化
