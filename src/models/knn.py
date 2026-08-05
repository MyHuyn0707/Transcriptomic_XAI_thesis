"""
src/models/knn.py

k-Nearest Neighbors classifier.
Author: Thesis Framework
"""

from sklearn.neighbors import KNeighborsClassifier
from typing import Any


def build_knn(**kwargs: Any) -> KNeighborsClassifier:
    """
    Build a k-Nearest Neighbors model.

    KNN is deterministic (no random_state) and distance-based, so it relies on
    the StandardScaler the trainer already fits per fold. ``n_jobs=-1`` uses all
    cores for neighbor search unless the caller overrides it.
    """
    kwargs.setdefault("n_jobs", -1)
    return KNeighborsClassifier(**kwargs)
