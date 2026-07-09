from pathlib import Path
import json

import polars as pl


def _partition_paths(path, split):
    """Resolve parquet partition paths for a split via manifest.json, falling
    back to a glob. `path` may point at the data root (containing manifest.json)
    or directly at the split directory (e.g. data/train)."""
    path = Path(path)
    # Direct split directory given (e.g. "data/train").
    if (path / f"{split}_partition_000.parquet").exists() or list(path.glob(f"{split}_partition_*.parquet")):
        return sorted(path.glob(f"{split}_partition_*.parquet"))

    # Data root with manifest.json.
    manifest = path / "manifest.json"
    if manifest.exists():
        files = json.loads(manifest.read_text(encoding="utf-8")).get("files", {}).get(split, [])
        if files:
            return [path / rel for rel in files]

    # Fallback: <path>/<split>/*.parquet.
    return sorted((path / split).glob("*.parquet"))


def load_train(path, partitions=None):
    """Lazily scan train parquet partitions.

    `path` is either the data root (containing manifest.json) or the train
    directory directly. `partitions` optionally limits to the first N
    partition files (useful for memory-constrained machines)."""
    files = _partition_paths(path, "train")
    if not files:
        raise FileNotFoundError(f"no train parquet partitions found under {path}")
    if partitions is not None:
        files = files[:partitions]
    return pl.scan_parquet([str(f) for f in files])


def load_test(path, partitions=None):
    """Lazily scan test parquet partitions (feature-only)."""
    files = _partition_paths(path, "test")
    if not files:
        raise FileNotFoundError(f"no test parquet partitions found under {path}")
    if partitions is not None:
        files = files[:partitions]
    return pl.scan_parquet([str(f) for f in files])
