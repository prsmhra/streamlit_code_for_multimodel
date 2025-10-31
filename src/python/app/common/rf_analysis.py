"""
Fit a RandomForestRegressor to predict frame-level RMS (log1p) and return
top-K important features per JSON feature file.

Constants like TOP_K, RANDOM_STATE, and NON_FEATURE_COLS
are imported from constant.py for centralized configuration.
"""

from typing import List, Tuple
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Import constants
from src.python.app.constants.constants import Constants 


def fit_rf_and_get_top_features_from_json(json_path: str, top_k: int = Constants.TOP_K, random_state: int = Constants.RANDOM_STATE) -> List[Tuple[str, float]]:
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            arr = json.load(fh)
    except Exception:
        return []

    try:
        df = pd.DataFrame(arr)
    except Exception:
        return []

    # Drop rows containing any NaNs to keep RF training stable
    df = df.dropna(axis=0, how="any")
    if df.shape[0] < 5:
        # not enough data to train reliably
        return []

    # Select numeric feature columns only (excluding metadata)
    feature_cols = [c for c in df.columns if c not in Constants.NON_FEATURE_COLS and np.issubdtype(df[c].dtype, np.number)]
    if not feature_cols:
        return []

    X = df[feature_cols].values
    y = df.get("rms").values if "rms" in df.columns else None
    if y is None:
        return []

    # Transform target with log1p to stabilize variance
    y_clipped = np.clip(y, a_min=0.0, a_max=None)
    y_log = np.log1p(y_clipped)

    try:
        rf = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
        # best-effort quick CV to detect gross issues
        try:
            cv_folds = min(5, max(2, int(len(y_log) / 10)))
            _ = cross_val_score(rf, X, y_log, cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1)
        except Exception:
            pass

        rf.fit(X, y_log)
        importances = rf.feature_importances_
        feat_importance_pairs = sorted(list(zip(feature_cols, importances)), key=lambda x: x[1], reverse=True)
        return [(f, float(im)) for f, im in feat_importance_pairs[:min(top_k, len(feat_importance_pairs))]]
    except Exception:
        return []
