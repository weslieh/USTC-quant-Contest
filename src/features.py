import polars as pl


def get_feature_columns(df):
    """Return all feature_* column names."""

    return [
        c
        for c in df.collect_schema().names()
        if c.startswith("feature_")
    ]


def get_responder_columns(df):
    """Return all responder_* column names."""

    return [
        c
        for c in df.collect_schema().names()
        if c.startswith("responder_")
    ]


def build_cross_sectional_features(df, feature_cols):
    """Per-time_id cross-sectional transforms over the 15 assets.

    For each feature, within each `time_id` slice compute:
      - rank:    fractional rank in [0, 1] (stable, ignores nulls)
      - zscore:  (x - mean) / std
      - demean:  x - mean

    Uses only the current time_id's values, so it is strictly causal and
    safe under the Time-Series API (no future information, no per-asset
    history needed — can be recomputed at inference per step).
    """
    exprs = []
    for c in feature_cols:
        col = pl.col(c)
        # group stats over the cross-section at each time_id
        mean = col.mean().over("time_id")
        std = col.std().over("time_id")
        n = col.count().over("time_id")
        # fractional rank in [0, 1]; guard divide-by-zero when a slice has 1 row
        rank_frac = pl.when(n > 1).then((col.rank("average") - 1) / (n - 1)).otherwise(0.5)
        exprs.append(rank_frac.alias(f"{c}_csrank"))
        exprs.append(((col - mean) / std).alias(f"{c}_csz"))
        exprs.append((col - mean).alias(f"{c}_csdm"))
    return df.with_columns(exprs)


def build_rolling_features(
    df,
    feature_cols,
    windows=(10, 20),
):
    """
    Compute per-asset rolling statistics and lag-1 features.
    Requires df to be sorted by (time_id, asset_id) or (asset_id, time_id).
    Sorts internally by (asset_id, time_id).
    """
    df = df.sort(["asset_id", "time_id"])

    exprs = []

    # lag-1: previous value for the same asset
    for c in feature_cols:
        exprs.append(
            pl.col(c).shift(1).over("asset_id").alias(f"{c}_lag1")
        )

    # rolling mean and std per asset for each window
    for c in feature_cols:
        for w in windows:
            exprs.append(
                pl.col(c)
                .rolling_mean(window_size=w, min_periods=1)
                .over("asset_id")
                .alias(f"{c}_rm_{w}")
            )
            exprs.append(
                pl.col(c)
                .rolling_std(window_size=w, min_periods=1)
                .over("asset_id")
                .alias(f"{c}_rs_{w}")
            )

    return df.with_columns(exprs)


def fill_infinite(df, feature_cols):
    """Replace inf/-inf with NaN, then fill NaN with 0."""

    exprs = []
    for c in feature_cols:
        exprs.append(
            pl.when(pl.col(c).is_infinite())
            .then(None)
            .otherwise(pl.col(c))
            .fill_null(0.0)
            .alias(c)
        )

    return df.with_columns(exprs)
