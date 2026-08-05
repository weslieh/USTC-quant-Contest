# 官方 LightGBM Baseline 对比分析 — 我们能学什么

## 核心事实

官方"无特殊技巧"单 LightGBM 模型公榜 **~0.0031** = 我们 3-backend K=5 集成 0.00310366。
说明 0.0031 **不是 GBDT 上限**，我们可能在某个基础环节上和官方走了不同的路。

---

## Baseline 做了什么（逐环节）

| 环节 | Baseline 做法 | 我们对应做法 | 差异 |
|---|---|---|---|
| **特征** | raw 323 + `lag1`/`diff1`/`rmean5`(top-48 相关 feature 的 per-asset 因果历史) + `asset_id` | raw 323 + per-asset topk5 增量列 + `asset_id` | **baseline 用 per-asset 因果历史特征(lag/diff/rmean)，我们用 per-asset masked 增量列** |
| **特征筛选** | top-48 by \|加权 corr with target\|（采样 20 万行算） | per-asset top-5 by LGB gain | 不同筛选标准 |
| **特征冻结** | 只用**最早一折 train** 做健康检查+冻结 schema，后面时段不反向决定特征 | 未显式冻结（用全量 importance） | baseline 防止未来泄漏到特征选择 |
| **CV** | **等块 purged K-fold**（K=5，每折两侧 purge 30 time_id）+ 尾部 15% holdout | expanding-window time CV + embargo 5000 | **不同 CV 方案** |
| **超参** | 4 组预注册候选(浅/深 × 中/强正则)，CV 选优，不做网格搜索 | 单组手调超参(lgb/xgb/cat 各一套) | baseline 有 CV 选参流程 |
| **早停** | 加权 zero-mean R² feval + early_stopping(80) | 同(LGB)；XGB/Cat 用 rmse | 一致(LGB) |
| **最终训练** | 全量(含 holdout) × 3 seed 重训，取均值 | 全量 × 5 折 booster 取均值 | baseline 是 seed-bag，我们是 fold-ensemble |
| **正则强度** | `lambda_l2=10~20`, `min_data_in_leaf=2000~5000` | `min_child_samples=2000`, 无 L2 | **baseline 用了强 L2(10-20)，我们没用 L2** |
| **轮数** | 700 上限，early stop，折均 best_iteration | 2000 上限，early stop 100 | baseline 轮数更少+更早停 |
| **scale** | `fitted_oof_scale` 只诊断不应用 | 不做 scale | 一致 |

---

## 关键差异分析（按可能影响排序）

### 1. ★ Per-asset 因果历史特征 (lag1/diff1/rmean5) — 最可能的差异

**这是最大的发现。** Baseline 的核心特征是每个 asset 的 top-48 feature 的：
- `lag1_{f}` = 上一 time_id 该 asset 的 feature 值
- `diff1_{f}` = 当前值 − lag1（即一阶变化）
- `rmean5_{f}` = 最近 5 个 time_id 的滚动均值（含当前）

我们的记忆明确写着 **"cs 截面 + rolling 时序特征：公榜实测 5折raw=0.0028205, 5折cs=0.0025475, 5折cs+rolling=0.0025952。cs/rolling 均低于 raw"**，把 rolling 判死。

**但仔细看，我们的 rolling 和 baseline 的不完全一样：**
- 我们的 `build_rolling_features` 产出 `{c}_lag1` + `{c}_rm_{w}` + `{c}_rs_{w}`（**含 rolling std**）
- Baseline 产出 `lag1` + `diff1` + `rmean`（**含 diff，不含 std**）
- 我们测试时是 **cs + rolling 同时开**，不是纯 rolling
- 我们测试时用的 rolling_windows 值记忆没记录（默认是 10/20，baseline 用 5）
- **关键**：那个测试是在 `seed` 分支(0.00289 时代)，**在 asset-as-categorical 之前**，且**在 per-asset topk5 增量列之前**。现在我们的模型已经有 asset 身份(per-asset 增量列 + asset_id categorical)，加 per-asset 因果历史特征的相互作用可能完全不同。

**baseline 证明 per-asset 因果 lag/diff/rmean 是活的，且足以单模型到 0.0031。** 这直接推翻记忆"rolling 时序特征判死"——至少在"per-asset 因果 + diff + 不含 std + window=5 + 配合 asset_id"这个组合下是活的。

**为什么 baseline 的历史特征可能漂移安全**：`lag1`/`diff1`/`rmean5` 是**逐 asset 的因果历史**（只用该 asset 自己的过去），不依赖跨 asset 的截面结构，也不依赖 train 全局分布。每个 asset 自己的过去→未来是"固定 per-row 属性"的延伸（类似 asset_id），属于记忆里"活的"那一类。而我们判死的 cs 是跨 asset 截面（每 time_id 15 个 asset 的相对位置），那才是 train 分布衍生的。

### 2. 强 L2 正则 (lambda_l2=10~20)

我们 LGB 当前用 `reg_lambda=0.1`（默认，best 命令没显式设）。Baseline 用 `lambda_l2=10~20` —— **强 100-200 倍**。
弱信号 + 漂移下，强 L2 可能是 baseline 单模型就稳到 0.0031 的关键——抑制过拟合到 train 分布。我们没试过把 L2 调到这个量级。`--reg-lambda` CLI 已存在，实验只需改一个参数值。

### 3. 特征冻结（只用最早折选特征）

Baseline 只用最早一折 train 做特征健康检查和相关性排序，冻结后全流程用同一份。
我们用全量算 importance（per-asset topk5 来自全量 m3_per_asset_importance）。用后期数据选特征 = 让未来分布泄漏进特征选择，漂移下可能有害。这和记忆"最近数据训练负 R²"同向。

### 4. CV 方案（等块 purged K-fold vs expanding-window）

Baseline 用等块 K-fold（每折 valid 是一个连续块，train 是其他块去掉 purge），我们用 expanding-window（train 越来越大，valid 是尾部连续段）。两者衡量的是不同的东西；expanding-window 偏向"用更多数据训练后期 valid"，等块偏向"每段都能被独立验证"。对调参选候选来说，等块更稳。

### 5. diff1 特征（一阶变化）

我们 lag 有，但 **diff = current − lag 没有**。diff 捕捉"变化量"而非"水平值"，对收益类 feature 可能是更直接的信号载体。baseline 单独加了 diff 列。

---

## 被记忆否定但值得重试的方向（按优先级）

### ★★★ 方向 1：per-asset 因果历史特征 (lag1 + diff1 + rmean5) — 在我们当前最佳模型上加
- 记忆否定的是"cs+rolling(含std, window 10/20, 无 asset 身份, seed分支)"，**不是**"per-asset 因果 lag/diff/rmean(window=5, 配合 asset_id + per-asset 增量列, 在 asset-capacity 基础上)"。
- 我们的 `build_rolling_features` 已实现 lag1 + rm + rs，**只缺 diff1**。
- 建议实验：在 asset-capacity(K=5) 基础上，加 per-asset top-48 feature 的 lag1+diff1+rmean5，**不含 rs(std)**，window=5，看公榜。
- 机制判据：per-asset 因果历史 = 每个 asset 自己的过去，不依赖跨 asset/train 分布，应属"活的"。
- 风险：我们已有 per-asset 增量列，可能和 lag 列冗余。但 baseline 没有增量列也到 0.0031，说明 lag/diff/rmean 本身有信号。

### ★★ 方向 2：强 L2 正则 (lambda_l2=10~20)
- 我们 LGB 当前 `reg_lambda=0.1`，baseline 用 `10~20`（强 100-200 倍）。弱信号+漂移下强 L2 抑制过拟合。
- 实验简单：当前最佳 LGB 命令加 `--reg-lambda 10`（CLI 已存在），看 CV+公榜。

### ★★ 方向 3：特征冻结（只用最早折选特征）
- 当前 per-asset importance 用全量算。改成只用最早一段 train 算 importance + 冻结。
- 可能小幅改善漂移鲁棒性。实验成本中等（要重跑 EDA importance）。

### ★ 方向 4：diff1 特征单独加
- 即便不全套历史特征，单独加 top-K feature 的 diff1 可能有用（收益类 feature 的变化量）。
- 实验成本低。

### ★ 方向 5：等块 purged K-fold CV
- 换 CV 方案，看 CV 是否更接近公榜（我们 CV 公榜比 1.55，baseline 没 CV/公榜比数据）。
- 主要价值是让调参更可信，不直接提分。

---

## 不值得重试的（baseline 也没做或我们已验证）

- **seed bagging vs fold ensemble**：我们已试多种子 bagging 仅 +0.000028≈0，baseline 3 seed 也是 bagging，不是它到 0.0031 的原因。
- **3-backend 集成**：baseline 单 LGB 就 0.0031，说明集成不是关键，我们的多样性已到顶。
- **预注册超参选优**：我们手调超参已不错，baseline 的 4 候选 CV 选优是流程规范，提分有限。

---

## 建议的下一步实验（按 ROI）

1. **(最高 ROI) 在 asset-capacity K=5 基础上加 per-asset 因果 lag1+diff1+rmean5**：我们的 `build_rolling_features` 加 diff1，window=5，top-48 相关筛选，**不含 std**，在当前最佳 LGB 上跑 CV+公榜。这是 baseline 的核心特征，我们没在当前模型上试过。
2. **(低成本) 强 L2=10~20**：当前最佳 LGB 命令加 L2，看 CV。
3. **(低成本) diff1 单独加**：最小改动。
4. **(中成本) 特征冻结**：只用最早折选 per-asset importance。

每个都可用 AV-CV 或公榜 1 次配额验证。方向 1 如果成立，可能直接解释 0.0031→更高的部分差距。
