"""
shift_test.py
--------------
Stress test: Train/test performance under different cohorts (e.g., leagues).

Usage:
------
from src.eval.shift_test import run_shift_test
run_shift_test(pipe, X_encoded, y, df, cohort_col="from_league")
"""

import pandas as pd
import json, datetime, os
from sklearn.metrics import f1_score, roc_auc_score

def run_shift_test(pipe, X, y, df, cohort_col="from_league"):
    os.makedirs("../results/stress_tests", exist_ok=True)
    cohorts = df[cohort_col].dropna().unique()
    results = {}

    for cohort in cohorts:
        mask = df[cohort_col] == cohort
        if mask.sum() < 30:  # skip tiny cohorts
            continue
        Xc, yc = X[mask], y[mask]
        pipe.fit(Xc, yc)
        y_pred = pipe.predict(Xc)
        y_proba = pipe.predict_proba(Xc)[:, 1]
        results[cohort] = {
            "F1": float(f1_score(yc, y_pred)),
            "ROC_AUC": float(roc_auc_score(yc, y_proba))
        }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"../results/stress_tests/shift_test_{cohort_col}_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[Shift Test] Results saved to {path}")
    return results
