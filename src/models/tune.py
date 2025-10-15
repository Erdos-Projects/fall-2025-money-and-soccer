"""
tune.py
-------
Reusable hyperparameter tuning utilities for the football performance modeling project.

This module provides flexible tuning functions that respect group-aware
cross-validation strategies (e.g., GroupKFold). It supports both grid and random
search, integrates smoothly with sklearn pipelines, and outputs consistent,
well-formatted results for downstream analysis and reporting.

Usage Example (inside modeling_experiments.ipynb):
--------------------------------------------------
from src.models.tune import tune_model

best_model, cv_results = tune_model(
    model=pipeline_rf,
    param_grid={
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [5, 10, None],
        "model__min_samples_leaf": [1, 3, 5]
    },
    X=X,
    y=y,
    groups=groups,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    },
    search_type="grid"  # or "random"
)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.validation import check_is_fitted
from joblib import dump
import os


def tune_model(
    model,
    param_grid,
    X,
    y,
    groups=None,
    cv=None,
    scoring=None,
    search_type="grid",
    n_iter=30,
    n_jobs=-1,
    random_state=42,
    refit_metric="f1",
    save_results_path=None,
    save_model_path=None,
):
    """
    Perform CV-aware hyperparameter tuning for a given model or pipeline.

    Parameters
    ----------
    model : estimator or sklearn.pipeline.Pipeline
        The model or pipeline to tune. It should implement `fit()` and `predict()`.
        For pipelines, use 'model__' prefixes in param_grid.

    param_grid : dict
        Dictionary of hyperparameters to search over.
        Example: {"model__n_estimators": [100, 200], "model__max_depth": [5, 10, None]}

    X, y : array-like
        Features and target.

    groups : array-like, optional
        Group labels for the samples used while splitting the dataset.

    cv : cross-validator
        GroupKFold or other CV splitter.

    scoring : str or dict, optional
        Metric(s) for evaluation. If dict, the refit metric must exist in keys.

    search_type : {"grid", "random"}, default="grid"
        Which type of hyperparameter search to perform.

    n_iter : int, default=30
        Number of iterations for random search.

    n_jobs : int, default=-1
        Number of parallel jobs for CV.

    random_state : int, default=42
        Random seed for reproducibility.

    refit_metric : str, default="f1"
        Metric to use for selecting the best model.

    save_results_path : str, optional
        File path to save the CV results as a CSV.

    save_model_path : str, optional
        File path to save the best trained model as a joblib artifact.

    Returns
    -------
    best_estimator : estimator
        The fitted model with the best hyperparameters.

    results_df : pd.DataFrame
        DataFrame of cross-validation results sorted by the refit metric.
    """

    print(f"\n🔍 Starting {search_type.capitalize()}SearchCV for {model.__class__.__name__}")
    print(f"→ Optimizing for metric: '{refit_metric}'")

    # --- Select Search Strategy ---
    if search_type == "grid":
        search = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=refit_metric,
            verbose=2,
        )
    elif search_type == "random":
        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            refit=refit_metric,
            random_state=random_state,
            verbose=2,
        )
    else:
        raise ValueError("search_type must be either 'grid' or 'random'")

    # --- Run Search ---
    search.fit(X, y, groups=groups)

    # --- Collect Results ---
    results_df = pd.DataFrame(search.cv_results_)
    score_col = f"mean_test_{refit_metric}"
    if score_col in results_df.columns:
        results_df = results_df.sort_values(score_col, ascending=False)
    else:
        results_df = results_df.sort_values("rank_test_" + refit_metric)

    best_params = search.best_params_
    print("\n✅ Best Parameters Found:")
    for k, v in best_params.items():
        print(f"   {k}: {v}")

    print(f"\nBest {refit_metric.upper()}: {search.best_score_:.4f}")

    # --- Save Results ---
    if save_results_path:
        os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
        results_df.to_csv(save_results_path, index=False)
        print(f"\n📄 Saved CV results to: {save_results_path}")

    # --- Save Best Model ---
    if save_model_path:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        dump(search.best_estimator_, save_model_path)
        print(f"💾 Saved best model to: {save_model_path}")

    return search.best_estimator_, results_df


def evaluate_best_model(model, X, y, metric_func, metric_name="metric"):
    """
    Evaluate a fitted model on data using a provided metric function.

    Example:
        from sklearn.metrics import f1_score
        f1 = evaluate_best_model(model, X_val, y_val, f1_score)
    """
    check_is_fitted(model)
    y_pred = model.predict(X)
    score = metric_func(y, y_pred)
    print(f"{metric_name}: {score:.4f}")
    return score
