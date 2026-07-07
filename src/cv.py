import polars as pl


def time_split(df, split_time):
    """Single time-based split. Kept for backward compatibility."""

    train = df.filter(
        pl.col("time_id") < split_time
    )

    valid = df.filter(
        pl.col("time_id") >= split_time
    )

    return train, valid


def time_cv_split(
    df,
    n_folds=5,
    valid_frac=0.1,
):
    """
    Expanding-window time-series cross-validation.
    Each fold uses more training data than the previous one.

    Example with n_folds=5, 100 time_ids:
      fold 0: train=[0:50], valid=[50:60]
      fold 1: train=[0:60], valid=[60:70]
      ...
      fold 4: train=[0:90], valid=[90:100]
    """
    import numpy as np

    times = (
        df.select("time_id")
        .unique()
        .sort("time_id")
        .collect()["time_id"]
        .to_numpy()
    )

    n_total = len(times)
    valid_size = max(1, int(n_total * valid_frac))

    folds = []
    for fold in range(n_folds):
        valid_end = n_total - (n_folds - fold - 1) * valid_size
        valid_start = valid_end - valid_size

        if valid_start <= 0:
            continue

        train_ids = times[:valid_start].tolist()
        valid_ids = times[valid_start:valid_end].tolist()

        train = df.filter(pl.col("time_id").is_in(train_ids))
        valid = df.filter(pl.col("time_id").is_in(valid_ids))
        folds.append((train, valid))

    return folds
