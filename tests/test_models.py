"""
test_models.py
---------------
End-to-end tests for the modeling pipeline components.

These tests ensure that:
1. The processed dataset loads correctly.
2. Training and evaluation run without errors.
3. The hyperparameter tuning logic executes properly.
4. Cross-validation returns consistent shapes and metrics.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.models.train import train_and_evaluate
from src.models.tune import tune_model


# ---------- CONFIG ----------

DATA_PATH = "data/processed/merged_clean_refined.csv"
TARGET_COL = "DeclineFlag"
GROUP_COL = "player_id"


# ---------- HELPER FUNCTIONS ----------

def build_test_pipeline():
    """Builds a lightweight preprocessing + model pipeline."""
    df = pd.read_csv(DATA_PATH)

    # Handle edge cases
    if GROUP_COL not in df.columns:
        df[GROUP_COL] = df.index

    X = df.drop(columns=[TARGET_COL, GROUP_COL])
    y = df[TARGET_COL]
    groups = df[GROUP_COL]

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )

    # Use a small RandomForest for quick test runs
    model = RandomForestClassifier(n_estimators=5, random_state=42)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    return df, X, y, groups, pipeline


# ---------- TESTS ----------

def test_data_integrity():
    """Check dataset loads correctly and target column exists."""
    assert os.path.exists(DATA_PATH), f"{DATA_PATH} not found."
    df = pd.read_csv(DATA_PATH)
    assert TARGET_COL in df.columns, f"Missing target column: {TARGET_COL}"
    assert not df.empty, "Dataset is empty."
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")


def test_train_and_evaluate_runs():
    """Ensure train_and_evaluate runs end-to-end without errors."""
    df, X, y, groups, pipeline = build_test_pipeline()

    cv = GroupKFold(n_splits=3)
    scoring = {"accuracy": "accuracy", "f1": "f1"}

    results_df = train_and_evaluate(
        model=pipeline,
        X=X,
        y=y,
        groups=groups,
        cv=cv,
        scoring=scoring,
        model_name="RandomForest_Test"
    )

    assert not results_df.empty, "No results returned from train_and_evaluate()."
    assert all(metric in results_df.columns for metric in ["accuracy", "f1"]), "Missing metrics in results."
    print(f"train_and_evaluate passed. F1: {results_df['f1'].iloc[0]:.3f}")


def test_tune_model_runs():
    """Ensure tune_model executes a small randomized search successfully."""
    df, X, y, groups, pipeline = build_test_pipeline()

    param_grid = {
        "model__n_estimators": [5, 10],
        "model__max_depth": [None, 3]
    }

    cv = GroupKFold(n_splits=3)
    scoring = {"f1": "f1", "accuracy": "accuracy"}

    best_model, results_df = tune_model(
        model=pipeline,
        param_grid=param_grid,
        X=X,
        y=y,
        groups=groups,
        cv=cv,
        scoring=scoring,
        search_type="random",
        n_iter=2,
        refit_metric="f1"
    )

    assert best_model is not None, "tune_model did not return a best model."
    assert not results_df.empty, "No CV results returned from tune_model()."
    assert "mean_test_f1" in results_df.columns, "F1 metric missing in tune results."
    print(f"tune_model passed. Best F1: {results_df['mean_test_f1'].max():.3f}")
