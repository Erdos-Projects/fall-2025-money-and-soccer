"""
train.py
--------
Reusable training and evaluation utilities for model experimentation.

Handles:
- Model fitting (baseline or tuned)
- Cross-validation evaluation
- Metric computation (accuracy, precision, recall, F1, ROC-AUC)
- Artifact saving (models + results)

Used by:
- notebooks/modeling_baselines.ipynb
- notebooks/modeling_experiments.ipynb
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)


def train_and_evaluate(model, X, y, groups=None, cv=None, scoring=None, n_jobs=-1, model_name=None):
    """
    Fit a model with cross-validation and return evaluation metrics.

    Parameters
    ----------
    model : estimator or pipeline
        Model or pipeline to train.

    X, y : array-like
        Features and target.

    groups : array-like, optional
        Grouping labels (e.g., player_id) for GroupKFold.

    cv : cross-validator
        Cross-validation strategy.

    scoring : dict or str, optional
        Scoring metrics for evaluation.

    n_jobs : int, default=-1
        Number of parallel jobs.

    model_name : str, optional
        Name used for reporting/logging.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame summarizing mean CV metrics for the given model.
    """

    if model_name is None:
        model_name = model.__class__.__name__

    print(f"\n🏋️ Training model: {model_name}")

    scores = cross_validate(
        model, X, y, groups=groups, cv=cv,
        scoring=scoring, n_jobs=n_jobs, return_train_score=False
    )

    results = {
        "model": model_name,
        "accuracy": np.mean(scores.get("test_accuracy", [np.nan])),
        "precision": np.mean(scores.get("test_precision", [np.nan])),
        "recall": np.mean(scores.get("test_recall", [np.nan])),
        "f1": np.mean(scores.get("test_f1", [np.nan])),
        "roc_auc": np.mean(scores.get("test_roc_auc", [np.nan]))
    }

    results_df = pd.DataFrame([results])
    print(f"✅ Completed {model_name} — F1: {results['f1']:.4f}, AUC: {results['roc_auc']:.4f}")

    return results_df


def fit_full_model(model, X, y, save_path=None, model_name=None):
    """
    Train the best model on full data and optionally save it.

    Parameters
    ----------
    model : estimator or pipeline
        Best model or pipeline to train.

    X, y : array-like
        Features and target.

    save_path : str, optional
        File path to save model artifact (e.g., '../artifacts/model.joblib').

    model_name : str, optional
        Custom name for logging.

    Returns
    -------
    model : estimator
        Trained model.
    """
    if model_name is None:
        model_name = model.__class__.__name__

    print(f"\n🚀 Fitting final model on full data: {model_name}")
    model.fit(X, y)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)
        print(f"💾 Model saved to: {save_path}")

    return model


def save_results(results_df, save_path="../results/model_comparison.csv"):
    """
    Save model comparison results to disk.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results_df.to_csv(save_path, index=False)
    print(f"📊 Results saved to {save_path}")
