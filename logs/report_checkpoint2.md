# Checkpoint 2 Report: EDA, Feature Selection, and Feature Engineering

This report documents our progress and lessons learned in the second checkpoint of the project: Exploratory Data Analysis (EDA), Feature Selection, and Feature Engineering.

---

## 1. Data Loading and Merging

- Loaded raw FBref (performance metrics) and Transfermarkt (transfer outcomes, fees, market values) datasets.  
- Implemented data merging logic in `src/data/data_preprocess.py`, ensuring alignment by player ID, names, and transfer events.  
- Produced merged dataset: `data/processed/merged/players_transfer_outcomes.csv`.  
- Refined cleaned version saved as `data/processed/merged_clean_refined.csv` with selected features.

Key steps:
- Duplicates removed.  
- Standardized column names and formats (e.g., dates, categorical league codes).  
- Added derived target variable: `DeclineFlag` (binary: performance decline post-transfer).  

---

## 2. Exploratory Data Analysis (EDA)

### 2.1 Distributions
- Univariate histograms and KDEs plotted for key numeric variables (`GA90_pre`, `PreMinutes`, `market_value_in_eur`, `transfer_fee`).  
- Engineered features (e.g., `age_at_transfer`, `contract_years_left`, `value_ratio`, `market_premium`, `league_strength_change`) inspected using histograms and countplots.

### 2.2 Missing Data
- Missingness quantified with `pandas.isna`.  
- Visualized using `missingno.matrix` and `missingno.heatmap`.  
- Found expected missing values in market-related fields for some players, handled with median/most-frequent imputers.

### 2.3 Relationships Between Variables
- Correlation heatmaps for numeric variables (Pearson).  
- Scatterplots of `market_value_in_eur` vs `transfer_fee`.  
- Grouped boxplots comparing `DeclineFlag` across positions and leagues.

### 2.4 Outlier Detection
- Applied z-score thresholding (`scipy.stats.zscore`) to detect extreme values in `transfer_fee`.  
- Tested `IsolationForest` for multivariate outlier detection.  
- Boxplots via seaborn confirmed extreme but plausible high-value transfers (e.g., top stars).  

### 2.5 Time-Series and Lag Structure
- Investigated lagged GA90 values using `pandas.DataFrame.shift`.  
- Considered autocorrelation via `statsmodels.plot_acf` and `plot_pacf`.  
- Decided lags will be tested further in optional temporal models but not in baseline.

---

## 3. Feature Selection

### 3.1 Domain Knowledge Filter
- Retained variables with direct theoretical relevance:  
  - Contextual: leagues, position, footedness, nationality.  
  - Performance: GA90_pre, GA_pre, PreMinutes.  
  - Market/transfer: fees, values, contract expiration.  
- Dropped identifiers (`player_id`, club names, URLs).  
- Dropped temporal identifiers (`transfer_season`, `last_season`) pending feature engineering.

### 3.2 Leakage Checks
- Removed `GA_post`, `PostMinutes`, and `PerfChange` (not available pre-transfer).  
- Ensured DeclineFlag target is strictly post-transfer.

### 3.3 Low-Information Features
- Dropped near-constant variables and embeddings (`Attribute Vector`, `Percentiles`).  
- Verified high-missingness columns excluded.

### 3.4 Redundancy / Correlation
- Identified collinear pairs in performance stats.  
- Variance Inflation Factor (VIF) tested to confirm redundancy.  
- Kept representative features only.

### 3.5 Model-Based Diagnostics
- Ran RandomForest feature importance ranking.  
- Confirmed importance of GA90_pre, market value, and league strength change.  

All decisions logged to `results/feature_selection.csv`.

---

## 4. Feature Engineering

### 4.1 Domain-Driven Features
Implemented in `src/features/transformers.py` via `DomainFeatureEngineer`:  
- `age_at_transfer` (transfer_date – date_of_birth).  
- `contract_years_left` (contract_expiration_date – transfer_date).  
- `value_ratio` (transfer_fee ÷ market_value_in_eur).  
- `market_premium` (current ÷ highest market value).  
- `league_strength_change` (to_league – from_league tier mapping).  

### 4.2 Dimensionality Reduction
- Added PCA (`PCAReducer`) and UMAP (`UMAPReducer`) transformers for exploratory visualization and potential downstream use.  

### 4.3 Time Features
- Built `TimeFeatureEngineer` for optional lagged/rolling GA90 features.  
- Not yet deployed in main pipeline, but framework ready.  

---

## 5. Iterative EDA + Feature Work

- EDA and feature engineering informed each other. Example:  
  - Initial lack of distribution for `age_at_transfer` → realized `transfer_date` was excluded → recovered from raw and engineered column.  
  - Redundant features flagged in correlation matrix guided pruning.  
- Iteration cycle repeated 2–3 times to refine kept features.

---

## 6. Deliverables

- **notebooks/eda.ipynb**: Visualizations, missingness, correlations, outliers, lag inspection.  
- **notebooks/feature_selection.ipynb**: Justification logs, `feature_selection.csv`.  
- **src/features/transformers.py**: Custom transformers (Domain, PCA, UMAP, Time).  
- **src/features/preprocessing.py**: Unified pipeline (imputers, scalers, encoders + Domain features).  
- **notebooks/pipeline_demo.ipynb**: Demonstrates preprocessing fit/transform, recovering feature names, feature importance plots.  
- **results/eda/**: Plots exported.  
- **results/feature_selection.csv**: Feature log.  
- **logs/report_checkpoint2.md** (this file): Written summary of experiences, issues, and solutions.

---

## 7. Key Lessons Learned

- Importance of recovering engineered features explicitly after dropping raw temporal columns.  
- `__pycache__` mistakenly committed → resolved by updating `.gitignore` and removing from Git cache.  
- Shape mismatches when reconstructing `X_df` → fixed by carefully aligning pipeline output dimensions with feature name recovery.  
- Iterative workflow: EDA revealed missing/erroneous assumptions multiple times (e.g., transfer_date availability).  
- Model-based feature diagnostics (RandomForest importances) validated domain intuitions.

---

## 8. Next Steps

- Expand feature engineering with interaction terms and time-aware structures.  
- Implement model experimentation (Logistic, RF, XGBoost).  
- Perform error analysis and robustness checks.  
- Prepare stakeholder-oriented executive summary and final presentation.

---

**End of Report**  
