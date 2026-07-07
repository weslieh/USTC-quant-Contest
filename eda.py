import polars as pl

# -----------------------
# 1. 读取数据（Lazy）
# -----------------------
train = pl.scan_parquet("data/train/train_partition_*.parquet")

print("=" * 60)
print("Dataset Overview")
print("=" * 60)

# -----------------------
# 2. 数据规模
# -----------------------
summary = train.select(
    pl.len().alias("rows"),
    pl.col("asset_id").n_unique().alias("n_assets"),
    pl.col("time_id").n_unique().alias("n_time_ids"),
)

print(summary.collect())

# -----------------------
# 3. target统计
# -----------------------
print("\nTarget Statistics")

target_stats = train.select(
    pl.col("target").mean().alias("mean"),
    pl.col("target").std().alias("std"),
    pl.col("target").min().alias("min"),
    pl.col("target").max().alias("max"),
)

print(target_stats.collect())

# -----------------------
# 4. weight统计
# -----------------------
print("\nWeight Statistics")

weight_stats = train.select(
    pl.col("weight").mean().alias("mean"),
    pl.col("weight").std().alias("std"),
    pl.col("weight").min().alias("min"),
    pl.col("weight").max().alias("max"),
)

print(weight_stats.collect())

# -----------------------
# 5. 缺失值
# -----------------------
print("\nMissing Value Report")

n_rows = train.select(pl.len()).collect().item()

missing = (
    train
    .select(pl.all().null_count())
    .collect()
    .transpose(include_header=True)
)

missing.columns = ["column", "missing_count"]

missing = missing.with_columns(
    (
        pl.col("missing_count") / n_rows
    ).alias("missing_ratio")
)

missing = missing.sort(
    "missing_ratio",
    descending=True
)

print(missing.head(20))

missing.write_csv("missing_report.csv")

print("\nSaved -> missing_report.csv")