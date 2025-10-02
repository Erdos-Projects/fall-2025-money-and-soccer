# Key Performance Indicators (KPIs)

This document defines the primary and secondary KPIs for evaluating our project:
**"Competition vs. Cash: Do Big-Money Transfers Affect Soccer Performance?"**

---

## 🎯 Primary KPI
**DeclineFlag_1y Accuracy (classification metric)**  
- Definition: Whether the model correctly predicts if a player experiences a performance decline (≥20% drop in GA/90 within 1 year post-transfer).  
- Metric: **F1-score (weighted)**  
- Justification: Balances precision and recall, ensuring we do not bias towards predicting only “no decline” cases.  

---

## 📊 Secondary KPIs

1. **ROC AUC (classification)**  
   - Captures ability to discriminate between decline vs. no-decline across thresholds.  
   - Important for model robustness.  

2. **PR AUC (classification)**  
   - Focuses on the positive class (decline), which is less frequent.  
   - Reflects usefulness in real scouting contexts.  

3. **MAE & RMSE (regression)**  
   - For models predicting *magnitude of performance change* (PerfChange_1y as continuous).  
   - Lower values indicate better accuracy in estimating true performance shifts.  

4. **Baseline Comparisons**  
   - DummyClassifier (majority / stratified) and DummyRegressor (mean) define minimum expectations.  
   - All advanced models must outperform these baselines on primary metrics.  

---

## ⚖️ Fairness & Interpretability KPIs

- **GroupKFold by player** ensures no leakage (a player’s pre/post-transfer records stay in same fold).  
- **Feature importance / coefficients** will be logged to check if decisions are dominated by single confounding features (e.g., transfer fee).  
- Models should generalize across positions and leagues.  

---

## 🚀 Improvement Directions

- Track gains in F1-score over baseline.  
- Reduce MAE in regression tasks below baseline mean-prediction levels.  
- Ensure PR AUC improvement indicates better handling of minority “decline” cases.  

---

**In summary:**  
- **Primary success metric:** F1-score on DeclineFlag_1y.  
- **Secondary metrics:** ROC AUC, PR AUC, MAE, RMSE.  
- A project is “portfolio worthy” if advanced models show significant gains over trivial baselines, while being interpretable and robust.
