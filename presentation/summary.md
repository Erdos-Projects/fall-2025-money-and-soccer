# Executive Summary — Football Transfer Decline Prediction

## Problem
Football clubs often invest heavily in player transfers, but many signings underperform after joining a new team. Predicting whether a player's performance will decline post-transfer can help clubs make more data-driven, cost-effective decisions.

## Data
- **Sources:** FBref (player performance stats) and Transfermarkt (transfer and market value data).
- **Total records:** ~10,000 player-season entries.
- **Key variables:** from_league, to_league, position, transfer_fee, market_value_in_eur, height_in_cm, and DeclineFlag (target label).

## Target
- **DeclineFlag:** Binary target (1 = player performance declined after transfer, 0 = performance maintained/improved).

## Methodology
1. **Data Preparation:** Cleaned and merged multi-source datasets, handled missing values, encoded categorical variables, and standardized numeric ones.
2. **Feature Engineering:** Derived indicators such as pre-transfer minutes, goal averages, and league movement (upward/downward transfer).
3. **Model Selection:** Compared Logistic Regression, Decision Trees, and Random Forests; the Random Forest pipeline performed best.
4. **Final Model:** RandomForestClassifier (n_estimators=300, max_depth=10, class_weight='balanced').

## Results
| Metric | Score |
|---------|-------|
| Accuracy | 0.72 |
| ROC-AUC | 0.76 |
| F1 (Decline class) | 0.59 |

**Interpretation:**
- The model correctly predicts outcomes ~72% of the time.
- It distinguishes decline vs. non-decline with an AUC of 0.76.
- Recall for decline cases (0.51) suggests moderate sensitivity to risk-prone transfers.

## Insights
- Players moving to higher-tier leagues tend to show higher decline risk.
- Age, playing minutes, and transfer fee are among the top predictors.
- Defensive players (CB, LB, GK) exhibit less variance in post-transfer decline.

## Limitations
- Missing injury, morale, and tactical-fit data.
- Class imbalance (more non-decline than decline cases).
- Categorical simplifications (league codes may hide within-league differences).

## Next Steps
- Integrate FIFA stats and injury history.
- Apply ensemble boosting (XGBoost, LightGBM) for improved recall.
- Develop a deployment dashboard for scouting analytics.

## Business Relevance
This model provides quantitative support to scouting and transfer committees by identifying high-risk transfers early, helping optimize recruitment budgets and long-term roster value.
