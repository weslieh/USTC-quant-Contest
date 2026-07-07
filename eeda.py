import polars as pl

# -----------------------
# 1. 读取数据（Lazy）
# -----------------------
train = pl.scan_parquet("data/train/train_partition_*.parquet")

sample = (
    train
    .head(1000000)
    .collect()
)
# -----------------------
# 5. 缺失值
# -----------------------
print("\nMissing Value Report")

n_rows = sample.select(pl.len()).item()

missing = (
    sample
    .select(pl.all().null_count())
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

#print(missing.head(20))
missing.write_csv("missing_report.csv")
