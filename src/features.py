import polars as pl


def get_feature_columns(df: pl.LazyFrame):

    return [
        c
        for c in df.collect_schema().names()
        if c.startswith("feature_")
    ]


def build_features(df: pl.DataFrame, feature_cols):

    return df.select(feature_cols)