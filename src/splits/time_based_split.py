"""
time_based_split.py
-------------------

Custom time-based splitter to prevent *temporal leakage* in
transfer prediction tasks.

Each fold trains on all data up to a specific cutoff date
and tests on the immediately following period. Ideal when
predicting future seasons or transfer windows.

Usage:
------
from src.splits.time_based_split import TimeBasedSplit
splitter = TimeBasedSplit(n_splits=3, date_column="transfer_date")
for train_idx, test_idx in splitter.split(df):
    ...
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional


class TimeBasedSplit:
    """
    Custom rolling time-based splitter.

    Parameters
    ----------
    n_splits : int, default=3
        Number of rolling splits (train on earlier → test on later).
    date_column : str, default="transfer_date"
        Column in DataFrame containing temporal information.
    min_train_size : int, optional
        Minimum number of samples in training before first split.
    """

    def __init__(
        self,
        n_splits: int = 3,
        date_column: str = "transfer_date",
        min_train_size: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.date_column = date_column
        self.min_train_size = min_train_size

    def split(
        self, df: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield train/test indices ordered by date.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a datetime column specified by `date_column`.
        y : pd.Series, optional
            Target vector (ignored, for sklearn compatibility).

        Yields
        ------
        train_idx, test_idx : tuple of np.ndarray
            Indices for training and testing subsets.
        """
        if self.date_column not in df.columns:
            raise ValueError(f"'{self.date_column}' not found in dataframe columns.")

        df_sorted = df.sort_values(self.date_column).reset_index(drop=False)
        unique_dates = np.sort(df_sorted[self.date_column].unique())

        # Select cut points based on chronological order
        n_dates = len(unique_dates)
        split_points = np.linspace(0, n_dates, self.n_splits + 1, dtype=int)[1:-1]

        for cutoff in split_points:
            cutoff_date = unique_dates[cutoff]
            train_idx = df_sorted[df_sorted[self.date_column] < cutoff_date].index.values
            test_idx = df_sorted[df_sorted[self.date_column] >= cutoff_date].index.values

            if self.min_train_size and len(train_idx) < self.min_train_size:
                continue  # skip if not enough training data

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of time-based splits."""
        return self.n_splits
