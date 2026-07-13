import numpy as np

def neutralize_predictions(
    preds: np.ndarray,
    features: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Orthogonalize predictions against a set of features.
    This removes the linear component of the predictions that can be explained
    by the given features, preventing the model from relying on drifting distributions.

    Args:
        preds: (N,) array of raw predictions.
        features: (N, K) array of features to neutralize against.
        alpha: Proportion of exposure to subtract (0 = no neutralization, 1 = full).

    Returns:
        (N,) array of neutralized predictions.
    """
    n = preds.shape[0]
    if n < 2 or alpha <= 0.0:
        return preds

    # We need to treat NaNs/Infs carefully. Since we are doing linear algebra,
    # any NaNs will ruin the calculation. Replace with column means or zeros.
    # In our inference pipeline, features might have NaNs (e.g. raw features).
    features_clean = np.where(np.isfinite(features), features, 0.0)

    # To avoid relying solely on the zero-mean assumption, we add an intercept term
    X = np.concatenate([features_clean, np.ones((n, 1))], axis=1)

    # Find the linear exposure of predictions to the features: X * (X^T * X)^-1 * X^T * preds
    # We can solve this efficiently using np.linalg.lstsq
    try:
        # lstsq returns: x, residuals, rank, s
        # where x minimizes ||X * x - preds||_2
        exposure_weights, _, _, _ = np.linalg.lstsq(X, preds, rcond=None)
        exposure = X @ exposure_weights

        # Subtract a proportion of the exposure
        preds_neutral = preds - alpha * exposure
        return preds_neutral
    except np.linalg.LinAlgError:
        # If the matrix is completely singular or numerical issues occur,
        # fallback to returning the original predictions
        return preds
