"""
Fit a RandomForestRegressor to predict frame-level RMS (log1p) and return
top-K important features per JSON feature file.

All constants are imported from constants.py for centralized configuration.
"""

from typing import List, Tuple
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Import Constants class
from src.python.app.constants.constants import Constants


def fit_rf_and_get_top_features_from_json(
    json_path: str,
    top_k: int = Constants.RF_TOP_K,
    random_state: int = Constants.RANDOM_STATE,
) -> List[Tuple[str, float]]:
    """
    Train a RandomForestRegressor on features from a JSON file
    and return the top-K most important features.

    Args:
        json_path (str): Path to JSON feature file.
        top_k (int): Number of top features to return.
        random_state (int): Random seed for reproducibility.

    Returns:
        List[Tuple[str, float]]: List of (feature_name, importance_score) pairs.
    """
    try:
        with open(json_path, "r", encoding=Constants.FILE_ENCODING) as fh:
            arr = json.load(fh)
    except Exception:
        return []

    try:
        df = pd.DataFrame(arr)
    except Exception:
        return []

    # Drop rows containing any NaNs to keep RF training stable
    df = df.dropna(axis=Constants.DROP_NA_AXIS, how=Constants.DROP_NA_HOW)
    if df.shape[0] < Constants.MIN_ROWS_REQUIRED:
        # Not enough data to train reliably
        return []

    # Select numeric feature columns only (excluding metadata)
    feature_cols = [
        c for c in df.columns
        if c not in Constants.NON_FEATURE_COLS and np.issubdtype(df[c].dtype, np.number)
    ]
    if not feature_cols:
        return []

    if Constants.TARGET_COLUMN not in df.columns:
        return []

    y = df[Constants.TARGET_COLUMN].values

    X = df[feature_cols].values

    # Transform target with log1p to stabilize variance
    y_clipped = np.clip(y, a_min=0.0, a_max=None)
    y_log = np.log1p(y_clipped)

    try:
        rf = RandomForestRegressor(
            n_estimators=Constants.N_ESTIMATORS,
            random_state=random_state,
            n_jobs=-1,
        )

        # Best-effort quick CV to detect gross issues
        try:
            cv_folds = min(
                Constants.MAX_CV_FOLDS,
                max(Constants.MIN_CV_FOLDS, int(len(y_log) / Constants.CV_DIVISOR)),
            )
            _ = cross_val_score(
                rf, X, y_log, cv=cv_folds, scoring=Constants.CV_SCORING, n_jobs=-1
            )
        except Exception:
            pass

        rf.fit(X, y_log)
        importances = rf.feature_importances_
        feat_importance_pairs = sorted(
            zip(feature_cols, importances), key=lambda x: x[1], reverse=True
        )
        return [
            (f, float(im))
            for f, im in feat_importance_pairs[: min(top_k, len(feat_importance_pairs))]
        ]
    except Exception:
        return []
