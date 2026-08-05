"""
src/models/dt.py

Decision Tree classifier.
Framework: Classification Transcriptomic with XAI
"""

from sklearn.tree import DecisionTreeClassifier
from typing import Any


def build_dt(random_state: int = 42, **kwargs: Any) -> DecisionTreeClassifier:
    """
    Build a Decision Tree model.
    """
    kwargs.setdefault("random_state", random_state)
    kwargs.setdefault("class_weight", "balanced")
    return DecisionTreeClassifier(**kwargs)
