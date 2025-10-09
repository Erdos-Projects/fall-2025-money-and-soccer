"""
outlier_test.py
----------------
Stress test: Remove top outlier samples (e.g. transfer_fee) and re-evaluate model.

Usage:
------
from src.eval.outlier_test import run_outlier_test
run_outlier_test(pipe, X_encoded, y, splitter, df, feature="transfer_fee")
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import os, datetime, json

def evaluate(pipe, X, y, splitter, groups=None):
    """Helper: return metric dict for given data."""
    pred_proba = cross_val_predict(
        pipe, X, y, cv=splitter.split(X, y, groups=groups),
        method="predict_proba"
    )[:, 1]
    pred = (pred_proba >= 0.5).astype(int)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "ROC_AUC": roc_auc_score(y, pred_proba),
        "PR_AUC": average_precision_score(y, pred_proba)
    }

def run_outlier_test(pipe, X, y, splitter, df, feature="transfer_fee", group_col=None):
    os.makedirs("../results/stress_tests", exist_ok=True)

    # Drop missing and detect threshold
    vals = df[feature].dropna()
    threshold = np.percentile(vals, 99)  # top 1%
    mask = df[feature] <= threshold

    X_reduced, y_reduced = X[mask], y[mask]
    # Clean group column before using
    if group_col:
        groups_full = df[group_col].fillna("Unknown").astype(str)
        groups_reduced = df.loc[mask, group_col].fillna("Unknown").astype(str)
    else:
        groups_full = groups_reduced = None

    metrics_full = evaluate(pipe, X, y, splitter, groups=groups_full)
    metrics_reduced = evaluate(pipe, X_reduced, y_reduced, splitter, groups=groups_reduced)

    result = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "feature": feature,
        "threshold": float(threshold),
        "samples_before": len(df),
        "samples_after": len(X_reduced),
        "metrics_full": metrics_full,
        "metrics_reduced": metrics_reduced
    }

    path = f"../results/stress_tests/outlier_test_{feature}_{result['timestamp']}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"[Outlier Test] Results saved to {path}")
    return result
