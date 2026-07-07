import numpy as np

def weighted_zero_mean_r2(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    denominator = np.sum(weight * y_true * y_true)
    if denominator <= 0:
        return 0.0
    numerator = np.sum(weight * (y_true - y_pred) ** 2)
    return float(1.0 - numerator / denominator)