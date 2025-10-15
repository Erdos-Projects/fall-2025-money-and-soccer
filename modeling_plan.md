Modeling Plan
=============

Project Goal
-------------
Predict player performance decline (DeclineFlag) using transfer and season statistics
from the merged dataset (merged_clean_refined.csv). Models are designed to
identify early signs of decline while avoiding data leakage through player-based grouping.

Model Families Tried
---------------------
1. DummyClassifier
   - Baseline to establish minimum expected performance.

2. LogisticRegression
   - Linear baseline to check feature separability and general signal strength.

3. DecisionTreeClassifier
   - Simple non-linear model; interpretable but prone to overfitting.

4. RandomForestClassifier
   - Robust ensemble baseline capturing non-linear relationships;
     handles mixed feature types effectively.

5. XGBoostClassifier / LightGBMClassifier
   - Gradient-boosted ensembles for advanced non-linear modeling and fine-grained
     hyperparameter tuning; used to test improvements beyond RandomForest.

Cross-Validation Strategy
--------------------------
- Used GroupKFold (n_splits=5) with player_id as the grouping variable.
- Ensures no data leakage across seasons for the same player.
- All models share identical splits for fair comparison.

Metrics
--------
Primary Metric: F1-score
Secondary Metrics: ROC-AUC, Precision, Recall, Accuracy

Hyperparameter Tuning
----------------------
- Implemented via tune_model() in src/models/tune.py
- RandomizedSearchCV used with GroupKFold
- Refit based on F1 metric
- Results saved to results/*_tuning_results.csv
- Tuned models serialized under artifacts/

Model Comparison Summary
-------------------------
| Model         | F1    | ROC-AUC |
|----------------|-------|---------|
| RandomForest   | 0.73  | 0.90    |
| XGBoost        | 0.71  | 0.88    |
| LightGBM       | 0.72  | 0.89    |

Interpretation
---------------
- RandomForest achieved the highest F1 and ROC-AUC balance.
- Ensemble models outperformed LogisticRegression, confirming
  presence of non-linear interactions.
- SHAP/Feature Importance plots show top predictors:
  PreMinutes, Age, transfer_fee, market_value_in_eur.

Final Model Selection
----------------------
Chosen Model: RandomForestClassifier
Reason: Best F1-score with good interpretability and stable performance across folds.

Interpretability
-----------------
- SHAP values and feature importance computed.
- Fallback to feature_importances_ handled automatically for sparse encodings.
- Top 10 influential features visualized and stored under results/interpretability/.

Iteration and Next Steps
-------------------------
1. Validate generalization over multiple seasons or leagues.
2. Revisit feature design (positional stats, temporal windows).
3. Consider temporal models (e.g., LSTM, temporal boosting) for longer-term trends.
4. Evaluate model calibration and fairness across positions.

Reproducibility
----------------
- All experiments encapsulated in reproducible pipelines.
- Training utilities in src/models/train.py
- Hyperparameter search logic in src/models/tune.py
- Baseline and experiment notebooks:
  - notebooks/modeling_baselines.ipynb
  - notebooks/modeling_experiments.ipynb
- Results automatically saved to results/ and artifacts/
- Verified via automated tests in tests/test_models.py
