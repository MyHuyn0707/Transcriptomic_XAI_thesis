"""
src/models/rf.py

Random Forest classifier.
Author: Thesis Framework
"""

from sklearn.ensemble import RandomForestClassifier
from typing import Any


def build_rf(random_state: int = 42, **kwargs: Any) -> RandomForestClassifier:
    """
    Build a Random Forest model.
    """
    kwargs.setdefault("random_state", random_state)
    kwargs.setdefault("class_weight", "balanced")
    kwargs.setdefault("n_jobs", -1)
    return RandomForestClassifier(**kwargs)
