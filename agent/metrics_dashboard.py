import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_full_metrics(model_name: str = "logistic") -> Dict:
    from agent.model_loader import load_model
    from agent.schema import FEATURE_COLUMNS

    try:
        pipeline = load_model(model_name)
    except FileNotFoundError:
        return {"error": f"Model {model_name} not found."}

    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "credit_risk_dataset.csv")
    if not os.path.exists(data_path):
        return {"error": f"Dataset not found at {data_path}"}

    df = pd.read_csv(data_path)
    df = df.dropna()
    
    X = df[FEATURE_COLUMNS]
    y_true = df["loan_status"]

    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "log_loss_val": log_loss(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    metrics["roc_curve_data"] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }

    if model_name == "logistic":
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        coefs = pipeline.named_steps["classifier"].coef_[0]
        importance = [(name, abs(val)) for name, val in zip(feature_names, coefs)]
    else:
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = pipeline.named_steps["classifier"].feature_importances_
        importance = [(name, float(val)) for name, val in zip(feature_names, importances)]

    importance.sort(key=lambda x: x[1], reverse=True)
    metrics["feature_importance"] = importance

    logger.info("Computed full metrics for %s (Accuracy: %.4f, AUC: %.4f)",
                model_name, metrics["accuracy"], metrics["roc_auc"])
    return metrics


def compute_comparison_metrics() -> Dict[str, Dict]:
    return {
        "logistic": compute_full_metrics("logistic"),
        "decision_tree": compute_full_metrics("decision_tree"),
    }


def plot_roc_comparison(comparison: Dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    colors = {"logistic": "#e5e5e5", "decision_tree": "#22c55e"}
    labels = {"logistic": "Logistic Regression", "decision_tree": "Decision Tree"}

    for model_key, metrics in comparison.items():
        roc_data = metrics["roc_curve_data"]
        auc = metrics["roc_auc"]
        ax.plot(
            roc_data["fpr"], roc_data["tpr"],
            color=colors.get(model_key, "#888"),
            lw=2.5,
            label=f"{labels.get(model_key, model_key)} (AUC = {auc:.4f})",
        )

    ax.plot([0, 1], [0, 1], "--", color="#555", lw=1, label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12, color="#ccc")
    ax.set_ylabel("True Positive Rate", fontsize=12, color="#ccc")
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold", color="#f0f0f0")
    ax.legend(loc="lower right", fontsize=11, facecolor="#141414", edgecolor="#333", labelcolor="#ccc")
    ax.tick_params(colors="#888")
    ax.grid(alpha=0.15, color="#555")
    for spine in ax.spines.values():
        spine.set_color("#333")
    fig.tight_layout()

    return fig


def plot_feature_importance(metrics: Dict) -> plt.Figure:
    importance_pairs = metrics["feature_importance"][:10]
    features = [p[0].replace("_", " ").title() for p in importance_pairs]
    values = [p[1] for p in importance_pairs]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")
    y_pos = range(len(features))

    bars = ax.barh(y_pos, values, color="#e5e5e5", edgecolor="#0a0a0a", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=10, color="#ccc")
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12, color="#ccc")

    model_label = "Logistic Regression" if metrics["model_name"] == "logistic" else "Decision Tree"
    ax.set_title(f"Feature Importance — {model_label}", fontsize=14, fontweight="bold", color="#f0f0f0")
    ax.tick_params(colors="#888")
    ax.grid(axis="x", alpha=0.15, color="#555")
    for spine in ax.spines.values():
        spine.set_color("#333")
    fig.tight_layout()

    return fig
