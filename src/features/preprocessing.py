import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from .transformers import DomainFeatureEngineer


# ==================================
# Columns
# ==================================
# Categorical predictors
CATEGORICAL_COLS = [
    "from_league", "to_league", "Position", "sub_position", "position", 
    "foot", "country_of_birth", "country_of_citizenship"
]

# Numeric predictors (pre-transfer & market features)
NUMERIC_COLS = [
    "GA90_pre", "GA_pre", "PreMinutes",
    "transfer_fee", "market_value_in_eur", "market_value_in_eur_player", 
    "highest_market_value_in_eur", "height_in_cm"
]

# Temporal columns used for engineered features
TEMPORAL_COLS = ["transfer_date", "date_of_birth", "contract_expiration_date"]

# Target
TARGET_COL = "DeclineFlag"


# ==================================
# Build Preprocessing Pipeline
# ==================================
def build_preprocessing_pipeline():
    """
    Create preprocessing pipeline:
    - Domain-driven feature engineering (age, contract length, ratios, league movement)
    - Impute missing values
    - Scale numeric features
    - One-hot encode categoricals
    """
    
    # Numeric pipeline
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS)
        ],
        remainder="passthrough"   # keep extra engineered features
    )
    
    # Full pipeline with domain-driven features first
    pipeline = Pipeline(steps=[
        ("domain", DomainFeatureEngineer()),   # adds engineered features
        ("preprocessor", preprocessor)
    ])
    
    return pipeline


# ==================================
# Utility
# ==================================
def preprocess_fit_transform(df):
    """
    Fit and transform preprocessing pipeline on a DataFrame.
    Returns numpy array ready for modeling.
    """
    pipeline = build_preprocessing_pipeline()
    X = df.drop(columns=[TARGET_COL], errors="ignore")
    y = df[TARGET_COL] if TARGET_COL in df else None
    X_processed = pipeline.fit_transform(X, y)
    return X_processed, y, pipeline
