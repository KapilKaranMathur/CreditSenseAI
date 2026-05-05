import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "credit_risk_dataset.csv")

AGE_BINS = [(18, 25, "18-25"), (26, 35, "26-35"), (36, 50, "36-50"), (51, 100, "51+")]
FAIRNESS_THRESHOLD = 0.80


def _load_and_prepare_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df[df["person_age"] <= 100]
    df = df[df["person_emp_length"] <= 60]
    df = df.dropna(subset=["loan_int_rate"])
    return df.reset_index(drop=True)


def _bin_age(age: int) -> str:
    for low, high, label in AGE_BINS:
        if low <= age <= high:
            return label
    return "Unknown"


def _bin_income(income: float, quartiles: List[float]) -> str:
    if income <= quartiles[0]:
        return "Q1 (Lowest)"
    elif income <= quartiles[1]:
        return "Q2"
    elif income <= quartiles[2]:
        return "Q3"
    return "Q4 (Highest)"


def _compute_group_metrics(df: pd.DataFrame, group_col: str,
                           pred_col: str, prob_col: str,
                           actual_col: str) -> pd.DataFrame:
    groups = df.groupby(group_col).agg(
        count=(pred_col, "count"),
        approval_rate=(pred_col, lambda x: (x == 0).mean()),
        avg_default_prob=(prob_col, "mean"),
        actual_default_rate=(actual_col, "mean"),
    ).reset_index()
    groups.columns = ["Group", "Count", "Approval Rate", "Avg Default Probability", "Actual Default Rate"]
    return groups.sort_values("Group").reset_index(drop=True)


def _compute_disparate_impact(group_metrics: pd.DataFrame) -> Dict:
    rates = group_metrics["Approval Rate"]
    if rates.max() == 0:
        return {"ratio": 0.0, "is_fair": False, "assessment": "Cannot compute — all groups have 0% approval"}
    ratio = round(float(rates.min() / rates.max()), 4)
    is_fair = ratio >= FAIRNESS_THRESHOLD
    assessment = (
        f"Fair (DIR = {ratio:.2f} ≥ {FAIRNESS_THRESHOLD})"
        if is_fair
        else f"Potential Bias Detected (DIR = {ratio:.2f} < {FAIRNESS_THRESHOLD})"
    )
    return {"ratio": ratio, "is_fair": is_fair, "assessment": assessment}


def run_bias_analysis(model_name: str = "logistic") -> Dict:
    from agent.model_loader import load_model
    from agent.schema import FEATURE_COLUMNS

    df = _load_and_prepare_data()
    pipeline = load_model(model_name)

    X = df[FEATURE_COLUMNS]
    df = df.copy()
    df["prediction"] = pipeline.predict(X)
    df["probability"] = pipeline.predict_proba(X)[:, 1]

    df["age_group"] = df["person_age"].apply(_bin_age)
    age_metrics = _compute_group_metrics(df, "age_group", "prediction", "probability", "loan_status")
    age_dir = _compute_disparate_impact(age_metrics)

    income_quartiles = df["person_income"].quantile([0.25, 0.5, 0.75]).tolist()
    df["income_group"] = df["person_income"].apply(lambda x: _bin_income(x, income_quartiles))
    income_metrics = _compute_group_metrics(df, "income_group", "prediction", "probability", "loan_status")
    income_dir = _compute_disparate_impact(income_metrics)

    ownership_metrics = _compute_group_metrics(df, "person_home_ownership", "prediction", "probability", "loan_status")
    ownership_dir = _compute_disparate_impact(ownership_metrics)

    logger.info(
        "Bias analysis complete — Age DIR: %.2f, Income DIR: %.2f, Ownership DIR: %.2f",
        age_dir["ratio"], income_dir["ratio"], ownership_dir["ratio"],
    )

    return {
        "age_analysis": age_metrics,
        "age_dir": age_dir,
        "income_analysis": income_metrics,
        "income_dir": income_dir,
        "ownership_analysis": ownership_metrics,
        "ownership_dir": ownership_dir,
        "model_name": model_name,
        "total_samples": len(df),
    }
