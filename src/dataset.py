from pathlib import Path
import polars as pl


def load_train(path):

    return pl.scan_parquet(
        str(Path(path) / "*.parquet")
    )