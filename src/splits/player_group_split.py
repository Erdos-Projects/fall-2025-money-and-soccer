"""
player_group_split.py
---------------------

Custom splitter to avoid *group-level leakage* by ensuring that
the same player never appears in both training and test folds.

Intended generalization scenario:
---------------------------------
We want to measure how well the model generalizes to *unseen players*
(i.e., new transfers by players not present during training).

Implements a thin wrapper around sklearn.model_selection.GroupKFold
but adds extra validation and reproducibility checks.

Usage:
------
from src.splits.player_group_split import PlayerGroupSplit
splitter = PlayerGroupSplit(n_splits=5, random_state=42)
for train_idx, test_idx in splitter.split(X, y, groups=df["player_id"]):
    ...
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from typing import Iterator, Tuple, Optional


class PlayerGroupSplit:
    """
    Custom player-level GroupKFold splitter.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    random_state : Optional[int]
        Random seed for reproducibility. Groups are shuffled before folding.
    """

    def __init__(self, n_splits: int = 5, random_state: Optional[int] = None):
        self.n_splits = n_splits
        self.random_state = random_state
        self._base_splitter = GroupKFold(n_splits=n_splits)

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series, optional
            Target vector.
        groups : pd.Series, required
            Player identifiers used to avoid leakage.

        Yields
        ------
        train_idx, test_idx : tuple of np.ndarray
            Indices for training and testing subsets.
        """
        if groups is None:
            raise ValueError("`groups` (player_id) must be provided.")

        unique_groups = np.unique(groups)
        rng = np.random.default_rng(self.random_state)
        shuffled_groups = rng.permutation(unique_groups)

        # Remap groups to shuffled order for reproducibility
        group_to_fold = {
            g: i % self.n_splits for i, g in enumerate(shuffled_groups)
        }
        fold_ids = groups.map(group_to_fold)

        for fold in range(self.n_splits):
            test_mask = fold_ids == fold
            train_mask = ~test_mask
            yield np.where(train_mask)[0], np.where(test_mask)[0]

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations."""
        return self.n_splits
