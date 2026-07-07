import polars as pl


def time_split(df, split_time):

    train = df.filter(
        pl.col("time_id") < split_time
    )

    valid = df.filter(
        pl.col("time_id") >= split_time
    )

    return train, valid