import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import umap


# ==============================
# Domain-driven transformations
# ==============================
class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Create engineered features based on football transfer domain knowledge.
    - Age at transfer
    - Contract length left
    - Value ratio (fee/value)
    - Market premium (current vs highest value)
    - League strength movement
    """
    def __init__(self, league_strength=None):
        # Default mapping for league strength if not provided
        self.league_strength = league_strength or {
            "GB1": 5, "ES1": 5, "IT1": 5, "DE1": 5, "FR1": 5,   # Top-5 leagues
            "NL1": 3, "PT1": 3, "BE1": 3, "TR1": 3,             # Mid-tier
            # Default fallback will be 1
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        # --- Age at transfer
        if "transfer_date" in X and "date_of_birth" in X:
            X["age_at_transfer"] = (
                pd.to_datetime(X["transfer_date"], errors="coerce")
                - pd.to_datetime(X["date_of_birth"], errors="coerce")
            ).dt.days / 365.25

        # --- Contract length left
        if "contract_expiration_date" in X and "transfer_date" in X:
            X["contract_years_left"] = (
                pd.to_datetime(X["contract_expiration_date"], errors="coerce")
                - pd.to_datetime(X["transfer_date"], errors="coerce")
            ).dt.days / 365.25

        # --- Value ratio
        if "transfer_fee" in X and "market_value_in_eur" in X:
            X["value_ratio"] = X["transfer_fee"] / (X["market_value_in_eur"] + 1e-6)
            X["value_ratio"] = X["value_ratio"].replace([np.inf, -np.inf], np.nan)

        # --- Market premium
        if "market_value_in_eur_player" in X and "highest_market_value_in_eur" in X:
            X["market_premium"] = (
                X["market_value_in_eur_player"] / (X["highest_market_value_in_eur"] + 1e-6)
            )
            X["market_premium"] = X["market_premium"].replace([np.inf, -np.inf], np.nan)



        # --- League strength movement
        if "from_league" in X and "to_league" in X:
            X["from_league_strength"] = X["from_league"].map(self.league_strength).fillna(1)
            X["to_league_strength"] = X["to_league"].map(self.league_strength).fillna(1)
            X["league_strength_change"] = X["to_league_strength"] - X["from_league_strength"]

        return X


# ==============================
# Dimensionality Reduction
# ==============================
class PCAReducer(BaseEstimator, TransformerMixin):
    """Apply PCA for dimensionality reduction on numeric features."""
    def __init__(self, n_components=5, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)

    def fit(self, X, y=None):
        self.pca.fit(X.select_dtypes(include=[np.number]))
        return self

    def transform(self, X, y=None):
        X = X.copy()
        numeric = X.select_dtypes(include=[np.number])
        pca_features = self.pca.transform(numeric)
        for i in range(self.n_components):
            X[f"PCA_{i+1}"] = pca_features[:, i]
        return X


class UMAPReducer(BaseEstimator, TransformerMixin):
    """Apply UMAP for dimensionality reduction on numeric features."""
    def __init__(self, n_components=2, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.reducer = umap.UMAP(n_components=n_components, random_state=random_state)

    def fit(self, X, y=None):
        self.reducer.fit(X.select_dtypes(include=[np.number]))
        return self

    def transform(self, X, y=None):
        X = X.copy()
        numeric = X.select_dtypes(include=[np.number])
        umap_features = self.reducer.transform(numeric)
        for i in range(self.n_components):
            X[f"UMAP_{i+1}"] = umap_features[:, i]
        return X


# ==============================
# Time Features
# ==============================
class TimeFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Add lag and rolling features for selected numeric columns.
    Useful for time-series like PreMinutes, GA90_pre, etc.
    """
    def __init__(self, lag_cols=None, lags=[1, 2, 3], rolling_windows=[3, 5]):
        self.lag_cols = lag_cols or []
        self.lags = lags
        self.rolling_windows = rolling_windows

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.lag_cols:
            if col not in X:
                continue
            # Lag features
            for lag in self.lags:
                X[f"{col}_lag{lag}"] = X[col].shift(lag)
            # Rolling mean features
            for window in self.rolling_windows:
                X[f"{col}_rollmean{window}"] = X[col].rolling(window).mean()
        return X
