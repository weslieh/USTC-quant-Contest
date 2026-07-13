import numpy as np
import polars as pl
import lightgbm as lgb


def adversarial_cv_split(
    df: pl.LazyFrame,
    raw_feature_cols: list[str],
    adv_val_ratio: float = 0.1,
    sample_rows: int = 1_000_000,
    seed: int = 42,
):
    """
    Splits data using an adversarial validation approach.
    It builds a lightweight classifier to distinguish train vs. test distributions.
    Since we don't have test targets, we simulate this by training on early vs late
    portions, or ideally we would use actual test features if available.
    In this competition setup, the training data is highly non-stationary.
    We split the data chronologically to get a pseudo-test set (the last X%),
    train an adversarial classifier, and score the entire training set.
    The validation set becomes the top `adv_val_ratio` portion of the training set
    that is most similar to the pseudo-test set.

    Args:
        df: LazyFrame of the training data.
        raw_feature_cols: Features to use for adversarial classification.
        adv_val_ratio: Fraction of the data to use as validation.
        sample_rows: Maximum rows to use for the adversarial classifier.
        seed: Random seed.

    Returns:
        List of one tuple: [(train_df, valid_df)] as LazyFrames.
    """
    print(f"  Building adversarial validation set (ratio: {adv_val_ratio}) ...")

    # Materialize a sample to build the classifier
    sample = df.head(sample_rows).collect()
    times = np.sort(sample["time_id"].unique().to_numpy())

    # Define pseudo-test as the last 20% of the sample chronologically
    split_idx = int(len(times) * 0.8)
    pseudo_test_time = times[split_idx]

    # Create binary target: 0 for early (train), 1 for late (pseudo-test)
    is_test = (sample["time_id"] >= pseudo_test_time).to_numpy().astype(int)

    X = sample.select(raw_feature_cols).to_numpy().astype(np.float32)

    print("  Training adversarial classifier...")
    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=seed,
        n_jobs=-1
    )
    clf.fit(X, is_test)

    # Now score the ENTIRE dataset
    # We do this in chunks if needed, but Polars select/with_columns is preferred
    # For simplicity, since the dataset is large, we'll collect the time_ids and feature matrix
    print("  Scoring full dataset for adversarial similarity...")
    full_df = df.collect()
    X_full = full_df.select(raw_feature_cols).to_numpy().astype(np.float32)

    # Get probability of being in the pseudo-test set
    adv_scores = clf.predict_proba(X_full)[:, 1]

    # Find the threshold for the top adv_val_ratio
    threshold = np.percentile(adv_scores, 100 * (1 - adv_val_ratio))
    print(f"  Adversarial score threshold for top {adv_val_ratio*100}%: {threshold:.4f}")

    # Create mask for validation set
    is_valid = adv_scores >= threshold

    # Add mask to dataframe
    full_df = full_df.with_columns(
        pl.Series("is_valid", is_valid)
    )

    # Split
    train_df = full_df.filter(~pl.col("is_valid")).drop("is_valid").lazy()
    valid_df = full_df.filter(pl.col("is_valid")).drop("is_valid").lazy()

    # Return as a single fold list to match time_cv_split signature
    return [(train_df, valid_df)]
