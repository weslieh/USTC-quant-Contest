"""Covariate-shift sample reweighting via adversarial validation.

Given a per-time_id "test-likeness" probability ``p_like_test`` from an
adversarial classifier (train vs real test, see src/adversarial_cv.py), we
reweight training samples by the odds ratio ``p/(1-p)`` so the reweighted
training distribution better matches the test distribution. This corrects the
strong train/test drift (adversarial AUC=1.0) **without dropping features** —
the key difference from the discredited drop-drift approach, which deleted
drifting features and lost signal.

The weights are applied to training rows only (``sample_weight``); the
validation metric keeps the original competition ``weight`` so CV still reports
the public-leaderboard quantity. Inference needs no changes — reweighting is
train-only.

Per-row ``p_like_test`` is approximated by the per-time_id mean (computed once
in ``adversarial_cv_split`` and reused here), which is both cheaper and
consistent with the validation-set selection.
"""

from __future__ import annotations

import numpy as np


def odds_ratio_weights(
    p: np.ndarray,
    clip_quantile: float = 0.99,
    eps: float = 1e-3,
) -> np.ndarray:
    """Convert test-likeness probabilities to normalized odds-ratio weights.

    ``w_i = (p_i / (1 - p_i))`` shifted and scaled so the mean weight is 1
    (preserves the effective sample size / loss scale). Probabilities are
    clipped to ``[eps, 1-eps]`` and further to the ``clip_quantile`` tail to
    keep the odds ratio from exploding when AUC≈1.0 pushes p→1.

    Args:
        p: array of ``p_like_test`` in [0, 1].
        clip_quantile: winsorize p at this upper quantile before the odds
            ratio (default 0.99). Set to 1.0 to disable.
        eps: hard floor/ceiling on p.

    Returns:
        weights with mean ≈ 1.0.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    if clip_quantile < 1.0:
        hi = np.quantile(p, clip_quantile)
        lo = np.quantile(p, 1.0 - clip_quantile)
        p = np.clip(p, max(lo, eps), min(hi, 1.0 - eps))
    odds = p / (1.0 - p)
    w = odds / odds.mean()  # mean → 1.0
    return w.astype(np.float32)


def time_id_to_weights(
    time_ids: np.ndarray,
    score_by_time: dict,
    clip_quantile: float = 0.99,
    eps: float = 1e-3,
) -> np.ndarray:
    """Map each row's time_id to its odds-ratio weight.

    ``score_by_time`` maps ``time_id -> p_like_test`` (from
    ``adversarial_cv.compute_time_adv_scores``). Rows whose time_id is missing
    get weight 1.0.
    """
    ps = np.array(
        [float(score_by_time.get(int(t), 0.5)) for t in time_ids],
        dtype=np.float64,
    )
    return odds_ratio_weights(ps, clip_quantile=clip_quantile, eps=eps)
