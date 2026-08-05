"""
src/helper/split.py

Stratified train/test split for the holdout rule-extraction flow. Rare-class
removal is not this module's job — run it first (e.g. load_dataset's
min_samples_per_class), since every class needs >= 2 members for stratify.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.model_selection import train_test_split


def train_test_split_indices(
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Stratified train/test split, returning row indices only (not the data
    itself) so callers can slice X/y/sample_ids consistently.

    Returns
    -------
    dict with keys ``train_idx``, ``test_idx`` (np.ndarray of row indices,
    each sorted ascending for reproducible, readable ordering).
    """
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y,
    )
    return {"train_idx": np.sort(train_idx), "test_idx": np.sort(test_idx)}
