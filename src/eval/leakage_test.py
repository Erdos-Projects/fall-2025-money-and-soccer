"""
leakage_test.py
----------------
Stress test: Shuffle targets to ensure the model cannot fit noise.

Usage:
------
from src.eval.leakage_test import run_leakage_test
run_leakage_test(pipe, X_encoded, y, splitter, df, group_col="from_league")
"""

import numpy as np
import json, datetime, os
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

def run_leakage_test(pipe, X, y, splitter, df=None, group_col=None):
    os.makedirs("../results/stress_tests", exist_ok=True)
    rng = np.random.default_rng(42)

    # Shuffle targets
    y_shuffled = y.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Clean group column if available
    if df is not None and group_col:
        groups = df[group_col].fillna("Unknown").astype(str)
    else:
        groups = None

    # Run cross-validation on shuffled targets
    scores = cross_val_score(
        pipe, X, y_shuffled,
        cv=splitter.split(X, y_shuffled, groups=groups),
        scoring=make_scorer(f1_score)
    )

    result = {
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        "mean_f1_shuffled": float(scores.mean()),
        "std_f1_shuffled": float(scores.std())
    }

    path = f"../results/stress_tests/leakage_test_{result['timestamp']}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"[Leakage Test] Results saved to {path}")
    return result
