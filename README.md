# Competition vs. Cash: Do Big-Money Transfers Affect Soccer Performance?

This project investigates whether soccer players who leave Europe's top-5 leagues for other (often financially lucrative but less competitive) leagues experience a measurable decline in performance.  
We use **FBref** (performance metrics) and **Transfermarkt** (transfer history) data to study this phenomenon.

---

## Problem Definition

**Research Question**  
Do players who leave Europe’s top-5 leagues (England, Spain, Italy, France, Germany) for other domestic leagues experience a measurable decline in on-field performance (goals + assists per 90 minutes) compared to peers who remain in the top leagues?

**Decision or Action Informed**  
This analysis informs scouts, analysts, and clubs about the trade-off between short-term financial gain and long-term competitive performance. It helps identify whether moving to a less competitive league increases the risk of player decline.

**Stakeholders**  
- **Football clubs & scouts:** care about predicting future performance when signing or selling players.  
- **Players & agents:** care about understanding the career risks of moving away from top-5 leagues.  
- **Fans & analysts:** interested in whether money-driven transfers affect competitiveness.  

**Unit of Analysis**  
The unit of analysis is a **player transfer event** — each row corresponds to one player’s move from a top-5 league to another domestic league, with performance tracked before and after the transfer.

**Scope and Boundaries**  
- **Time horizon:** one season (≈365 days) before and after the transfer.  
- **Geographic region:** Europe’s top-5 leagues as the source; all other European domestic leagues as destinations (“emerging Europe”).  
- **Population:** outfield players with sufficient minutes before and after the transfer.  
- **Features included:** FBref performance attributes, GA90 (goals + assists per 90), Transfermarkt market value, position, and transfer metadata.  
- **Features excluded:** subjective factors like media narratives, injuries without recorded minutes, or contract details not in the datasets.  

**Anti-Goals**  
- We do **not** predict a player’s financial success, wage outcomes, or sponsorship opportunities.  
- We do **not** claim causality between transfer destination and decline (only correlation based on observed performance).  
- We do **not** analyze purely non-European transfers (e.g., to MLS, Saudi Pro League, China).  


## 📂 Project Structure

```
DSBootcampP1/
│
├── artifacts/                     <- Saved artifacts (pipelines, baseline summaries)
│   └── baseline_summary.csv
│
├── data/
│   ├── raw/                       <- Raw datasets (FBref + Transfermarkt)
│   └── processed/                 <- Processed datasets (merged, cleaned, refined)
│
├── logs/                          <- Logs and reports
│   ├── baseline_kpis.json
│   └── report_checkpoint2.md
│
├── notebooks/                     <- Analysis notebooks
│   ├── baseline.ipynb             <- Baseline models (Dummy, Logistic, RF)
│   ├── eda.ipynb                  <- Exploratory Data Analysis
│   ├── feature_selection.ipynb    <- Feature elimination & rationale
│   └── pipeline_demo.ipynb        <- Preprocessing & engineered pipeline demo
│
├── results/                       <- Outputs and visualizations
│   ├── eda/                       <- Saved plots, correlation maps, missingness heatmaps
│   └── feature_selection.csv      <- Feature log (kept vs dropped + reasons)
│
├── src/                           <- Source code
│   ├── data/
│   │   ├── data_loader.py         <- Load & unify raw FBref data
│   │   ├── dummy_checker.py       <- Simple sanity checks
│   │
│   └── features/
│       ├── preprocessing.py       <- Full preprocessing pipeline
│       └── transformers.py        <- Custom transformers (domain-driven, lag, ratios, etc.)
│
├── README.md                      <- Project documentation
├── requirements.txt               <- Python dependencies
├── kpis.md                        <- Key Performance Indicators & model metrics
└── .gitignore                     <- Git ignore rules

```

---

## 📥 Data Access

Since raw data files are large and/or licensed, we do **not** commit them to this repository.  
Please download them manually from Kaggle and place them in `data/raw/`:

1. **FBref player performance data**  
   https://www.kaggle.com/datasets/soulcelestia/fbref-data  

2. **Transfermarkt player scores and transfer records**  
   https://www.kaggle.com/datasets/davidcariboo/player-scores  

After downloading, your structure should look like:

```
data/raw/fbref/...
data/raw/transfermarkt/...
```

---

## ⚙️ Data Loading

- `src/data/data_loader.py` combines position-wise FBref CSVs (Midfielders, Forwards, etc.) into one unified file:  
  → `data/processed/fbref/fbref.csv`

- `src/data/data_preprocess.py` merges FBref with Transfermarkt transfers, builds **pre/post performance features** (GA90_pre, GA90_post), and labels each transfer with a **DeclineFlag_1y** target.  
  → Outputs `data/processed/merged/players_transfer_outcomes.csv` and cleaned versions.

---

### **Data Assessment**

### Volume and Coverage
Our merged dataset contains ~886 labeled player transfers with both FBref performance metrics and Transfermarkt transfer histories. The data spans multiple European leagues and captures both pre- and post-transfer performance within a one-year window. This provides enough coverage to train baseline models and evaluate trends, though the sample size is modest compared to the global transfer market.

### Granularity
The unit of analysis is the individual player transfer. Each record links a player’s performance (from FBref) to their corresponding transfer event (from Transfermarkt), with features aggregated at the per-player level (e.g., mean percentile scores, attribute vectors) and outcomes defined as performance decline flags. The granularity is appropriate and consistent with the research question.

### Bias and Representativeness
The dataset focuses on transfers out of Europe’s top-5 leagues into other European domestic leagues, leaving out moves to non-European destinations (e.g., MLS, Saudi Pro League, Chinese Super League). As such, it may underrepresent the financial incentives driving high-profile transfers outside Europe. Additionally, performance measures rely on publicly available FBref statistics, which may not fully capture qualitative aspects of player contributions. Nonetheless, the dataset is reasonably balanced (586 no-decline vs. 300 decline cases) and provides a fair basis for modeling without extreme class imbalance.

### **Assessing Learnability**

- Signal vs. noise: The FBref features (performance metrics) plausibly explain performance change, while Transfermarkt captures transfer context — both relevant to the target.

- Data sufficiency: ~886 examples with a balanced decline vs. no-decline split (≈2:1) — enough for baseline modeling.

- Time series cycles: We focus on pre/post one-year windows, not long seasonal trends; cycles beyond one year are out of scope.

- Feature-target alignment: All features are available before transfer, avoiding leakage from post-transfer performance.

- Explanatory variables: Position, league tier, and performance metrics are plausible predictors of decline.

## 🧹 Data Preprocessing

- Removes duplicates and obvious errors.  
- Handles missing values with imputers.  
- Detects outliers via IQR + robust z-score, with optional winsorization.  
- Encodes categorical features (OneHotEncoder).  
- Scales numeric features (RobustScaler).  
- Drops highly correlated features to prevent multicollinearity.  
- Outputs:  
  - Clean semantic dataset (`*_clean.csv`)  
  - Model-ready dataset (`*_model_ready.csv`)  
  - Preprocessing pipeline (`artifacts/preprocessor.pkl`)  
  - Cleaning report (`logs/cleaning_report.md`)

---

## 🤖 Baselines

Implemented in `notebooks/baseline.ipynb`:

**Classification (target = DeclineFlag_1y):**
- DummyClassifier (most frequent, stratified)
- Logistic Regression (L2, balanced)
- Random Forest Classifier (balanced subsample)

**Regression (target = PerfChange_1y as continuous):**
- DummyRegressor (mean baseline)
- Linear Regression
- Random Forest Regressor

Evaluation uses **GroupKFold** by player to avoid leakage.

Metrics:  
- Classification → Accuracy, F1, ROC AUC, PR AUC  
- Regression → MAE, RMSE, R²  

---

## 📦 Requirements

Key Python packages:

- pandas, numpy, scipy  
- scikit-learn  
- matplotlib, seaborn (for plots)  

Install with:

```bash
pip install -r requirements.txt
```

---

## 🏆 Deliverables

- **Annotated GitHub repo** (this repo)  
- **Executive Summary (PDF)** summarizing results and implications  
- **5-min video presentation**  
- **Baselines & KPIs** documented (`kpis.md`)  

This project is designed to be **portfolio-worthy**, highlighting the ability to clean data, engineer features, avoid leakage, and evaluate models rigorously.

---

cat << 'EOF' >> README.md

---

## 🔍 Checkpoint 2 Deliverables

In **Checkpoint 2**, we focused on **EDA, feature selection, and feature engineering**.  
Here’s how our repo addresses the guide’s checklist:

- **EDA (`notebooks/eda.ipynb` + `results/eda/`)**
  - Univariate distributions → histograms, KDEs, boxplots.  
  - Missing data → quantified + visualized (using `missingno`).  
  - Relationships → scatterplots, correlation heatmaps.  
  - Outlier detection → `zscore`, `IsolationForest`, and boxplots.  
  - Exploratory time-series structure → lag shifts, ACF/PACF plots.  
  - Plots are exported to `results/eda/`.  

- **Feature Selection (`notebooks/feature_selection.ipynb` + `results/feature_selection.csv`)**
  - **Domain knowledge filter**: dropped identifiers (IDs, names), URLs, agent info.  
  - **Leakage check**: removed post-outcome features (e.g., `GA_post`, `PerfChange`).  
  - **Low-information filter**: constant/high-missingness columns flagged.  
  - **Redundancy**: correlation pruning + VIF.  
  - Final feature log stored in `results/feature_selection.csv`.  

- **Feature Engineering (`src/features/transformers.py` + `src/features/preprocessing.py`)**
  - **Domain-driven**: age at transfer, contract years left, value ratio, market premium, league strength change.  
  - **Dimensionality reduction**: PCA & UMAP transformers.  
  - **Time features**: lag + rolling statistics via `TimeFeatureEngineer`.  
  - Integrated into preprocessing pipeline for reuse in modeling.  

- **Pipeline Demo (`notebooks/pipeline_demo.ipynb`)**
  - Demonstrates fitting/transforming refined dataset with preprocessing pipeline.  
  - Recovers transformed feature names for interpretability.  
  - Includes RandomForest feature importances (model-based diagnostics).  

- **Deliverables**
  - Data audit summary: distributions, missingness, correlations.  
  - Dropped features list: reasons logged in CSV.  
  - Engineered features list: rationale documented in code + README.  
  - Reusable preprocessing pipeline (`src/features/preprocessing.py`). 