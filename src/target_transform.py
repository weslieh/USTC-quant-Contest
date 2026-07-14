"""Target rank transformation with a stored inverse-CDF lookup table.

The model is trained to predict the per-time_id rank percentile of the target
(in [0,1]) instead of the raw target. Rank is scale-invariant, so it is more
robust to the per-time_id distribution shifts between train and test. At
inference we cannot compute the test time_id's target distribution (those are
the predictions themselves), so we map the predicted rank back to the original
target scale via a global inverse-CDF lookup table (LUT) computed from the
training target distribution and stored in model_meta.json.

The LUT stores, for a grid of quantiles ``q in [0,1]``, the corresponding
training-target value ``np.quantile(y_train, q)``. Inference interpolates:
``target_pred = interp(rank_pred, q_grid, value_grid)``. The clip bound
(``3*target_std``) stays in the original scale and is applied after the
inverse transform.

NOTE: per-time_id z-scoring is NOT viable here because inference sees only one
time_id slice and would need that slice's target mean/std — which is exactly
what we are predicting. Rank + global inverse-CDF avoids that circularity.
"""

from __future__ import annotations

import numpy as np


def target_rank_per_time(time_ids: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-time_id rank percentile of ``target`` in [0,1].

    Uses average rank (ties share the mean of their positions) divided by the
    group size, matching ``pandas.groupby.rank(pct=True, method='average')``.
    Returns a float32 array aligned with the input rows.
    """
    import pandas as pd

    s = pd.Series(target)
    rank_pct = s.groupby(pd.Series(time_ids)).rank(method="average", pct=True)
    return rank_pct.to_numpy().astype(np.float32)


def build_inverse_cdf_lut(target: np.ndarray, n_points: int = 1001) -> dict:
    """Build the global inverse-CDF lookup table from training targets.

    Returns ``{"q": [...n_points], "v": [...n_points]}`` where ``v[i]`` is the
    target value at quantile ``q[i]``. ``q`` spans [0,1] inclusive. Stored in
    model_meta.json so inference can map predicted rank -> target value.
    """
    y = np.asarray(target, dtype=np.float64)
    y = y[np.isfinite(y)]
    q = np.linspace(0.0, 1.0, n_points)
    v = np.quantile(y, q)
    return {"q": q.tolist(), "v": [float(x) for x in v]}


def inverse_cdf_map(rank_pred: np.ndarray, lut: dict) -> np.ndarray:
    """Map predicted ranks in [0,1] back to target scale via the LUT.

    Linear interpolation between LUT points; clamped to the LUT's value range.
    """
    q = np.asarray(lut["q"], dtype=np.float64)
    v = np.asarray(lut["v"], dtype=np.float64)
    r = np.clip(np.asarray(rank_pred, dtype=np.float64), 0.0, 1.0)
    return np.interp(r, q, v).astype(np.float32)
