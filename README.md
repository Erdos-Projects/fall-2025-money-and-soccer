# Competition vs. Cash: Do Big-Money Transfers Affect Soccer Performance?

This project investigates whether soccer players who leave Europe's top-5 leagues for other (often financially lucrative but less competitive) leagues experience a measurable decline in performance.  
We use **FBref** (performance metrics) and **Transfermarkt** (transfer history) data to study this phenomenon.

---

## 📂 Project Structure

```
DSBootcampP1/
│
├── data/
│   ├── raw/              <- Raw data (ignored via .gitignore; must be downloaded separately)
│   └── processed/        <- Processed datasets (combined CSVs)
│
├── src/
│   └── data/
│       ├── data_loader.py       <- Loads raw FBref CSVs into one combined dataset
│       ├── data_preprocess.py   <- Cleans and merges FBref + Transfermarkt datasets
│       └── data_cleaner.py      <- Preprocessing, outlier handling, feature scaling (under construction)
│
├── notebooks/
│   └── baseline.ipynb    <- Trivial + linear + tree baselines
│
├── artifacts/            <- Saved sklearn preprocessing pipeline (pickle)
├── logs/                 <- Cleaning & build reports
└── requirements.txt      <- Python dependencies
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
