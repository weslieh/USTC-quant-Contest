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
    """Per-time_id cross-sectional rank/zscore/demean for selected features.

    For each column in ``feature_cols`` add three columns computed within each
    ``time_id`` slice (across the ~15 assets):
      * ``{c}_cs_rank``  : rank fraction in [0,1] = (rank-1)/(n-1)
      * ``{c}_cs_z``     : (x - mean) / (std + eps)
      * ``{c}_cs_dm``    : x - mean

    Degenerate slices (n<2, or std==0) yield rank=0.5, z=0, dm=0 so the model
    gets a neutral signal instead of NaN. This is stateless per time_id, so
    the inference side (which receives one time_id at a time) can reproduce
    the exact same transform from a numpy/pandas slice — see strategy/main.py.

    Returns (df_with_new_columns, list_of_new_column_names).
    """

    eps = 1e-8
    exprs = []
    new_cols = []
    for c in feature_cols:
        # Real features have no NaN, but guard so rank/mean/std are well-defined:
        # treat NaN as 0 within the cross-sectional computation.
        col = pl.when(pl.col(c).is_nan()).then(0.0).otherwise(pl.col(c))
        n = col.count().over("time_id")
        mean = col.mean().over("time_id")
        std = col.std().over("time_id")

        # rank fraction: average rank (ties) mapped to [0,1]
        rank_avg = col.rank("average").over("time_id")
        rank_frac = pl.when(n > 1).then((rank_avg - 1.0) / (n - 1.0)).otherwise(0.5)

        z = (col - mean) / (std + eps)
        z = pl.when(std > eps).then(z).otherwise(0.0)
        z = pl.when(col.is_null()).then(0.0).otherwise(z)

        dm = col - mean
        dm = pl.when(col.is_null()).then(0.0).otherwise(dm)

        rcol = f"{c}_cs_rank"
        zcol = f"{c}_cs_z"
        dcol = f"{c}_cs_dm"
        exprs.append(rank_frac.alias(rcol))
        exprs.append(z.alias(zcol))
        exprs.append(dm.alias(dcol))
        new_cols.extend([rcol, zcol, dcol])

    return df.with_columns(exprs), new_cols


def build_rolling_features(
    df,
    feature_cols,
    windows=(10, 20),
):
    """
    Compute per-asset rolling statistics and lag-1 features.
    Requires df to be sorted by (time_id, asset_id) or (asset_id, time_id).
    Sorts internally by (asset_id, time_id).

    lag-1 / rolling use shift(1) so the current row never enters its own
    window — the inference side mirrors this with a per-asset history buffer
    that is updated *after* the current time_id is predicted.
    """
    df = df.sort(["asset_id", "time_id"])

    exprs = []

    # lag-1: previous value for the same asset (current row excluded)
    for c in feature_cols:
        exprs.append(
            pl.col(c).shift(1).over("asset_id").alias(f"{c}_lag1")
        )

    # rolling mean and std per asset for each window (current row excluded)
    for c in feature_cols:
        for w in windows:
            exprs.append(
                pl.col(c)
                .rolling_mean(window_size=w, min_periods=1)
                .shift(1)
                .over("asset_id")
                .alias(f"{c}_rm_{w}")
            )
            exprs.append(
                pl.col(c)
                .rolling_std(window_size=w, min_periods=1)
                .shift(1)
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
