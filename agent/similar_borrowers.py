import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "credit_risk_dataset.csv")

NUMERICAL_FEATURES = [
    "person_age", "person_income", "person_emp_length",
    "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL_FEATURES = [
    "person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file",
]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

_dataset_cache: Optional[pd.DataFrame] = None


def _load_dataset() -> pd.DataFrame:
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df[df["person_age"] <= 100]
    df = df[df["person_emp_length"] <= 60]
    df = df.dropna(subset=["loan_int_rate"])
    df = df.reset_index(drop=True)
    _dataset_cache = df
    logger.info("Loaded historical dataset: %d rows", len(df))
    return _dataset_cache


def _gower_distance_row(query: np.ndarray, dataset: np.ndarray,
                        num_indices: List[int], cat_indices: List[int],
                        ranges: np.ndarray) -> np.ndarray:
    """
    Gower distance for mixed data types.
    Numerical: |x - y| / range. Categorical: 0 if same, 1 if different.
    """
    n_features = len(num_indices) + len(cat_indices)
    distances = np.zeros(dataset.shape[0])
    for idx in num_indices:
        r = ranges[idx]
        if r > 0:
            distances += np.abs(dataset[:, idx].astype(float) - float(query[idx])) / r
    for idx in cat_indices:
        distances += (dataset[:, idx] != query[idx]).astype(float)
    return distances / n_features


def find_similar_borrowers(borrower_profile: Dict, top_k: int = 5) -> List[Dict]:
    df = _load_dataset()

    for col in ALL_FEATURES:
        if col not in borrower_profile:
            raise ValueError(f"Missing feature in borrower profile: {col}")

    feature_df = df[ALL_FEATURES].copy()
    query_series = pd.Series({col: borrower_profile[col] for col in ALL_FEATURES})

    num_indices = [ALL_FEATURES.index(col) for col in NUMERICAL_FEATURES]
    cat_indices = [ALL_FEATURES.index(col) for col in CATEGORICAL_FEATURES]

    dataset_array = feature_df.values
    query_array = query_series[ALL_FEATURES].values

    ranges = np.zeros(len(ALL_FEATURES))
    for idx in num_indices:
        col_values = dataset_array[:, idx].astype(float)
        ranges[idx] = col_values.max() - col_values.min()

    distances = _gower_distance_row(query_array, dataset_array, num_indices, cat_indices, ranges)
    top_indices = np.argsort(distances)[:top_k]

    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        match = {col: row[col] for col in ALL_FEATURES}
        loan_status = int(row.get("loan_status", 0))
        match["loan_status"] = loan_status
        match["outcome_label"] = "Defaulted" if loan_status == 1 else "No Default"
        match["similarity_score"] = round(float(distances[idx]), 4)
        results.append(match)

    logger.info(
        "Found %d similar borrowers (closest: %.4f, default rate: %.0f%%)",
        len(results),
        results[0]["similarity_score"] if results else 0.0,
        sum(1 for r in results if r["loan_status"] == 1) / max(len(results), 1) * 100,
    )
    return results
