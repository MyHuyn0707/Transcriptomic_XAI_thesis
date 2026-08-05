"""
src/models/svm.py

Support Vector Machine classifier.
Author: Thesis Framework
"""

from sklearn.svm import SVC
from typing import Any


def build_svm(random_state: int = 42, **kwargs: Any) -> SVC:
    """
    Build an SVM model with probability estimation enabled.
    """
    kwargs.setdefault("probability", True)
    kwargs.setdefault("random_state", random_state)
    kwargs.setdefault("class_weight", "balanced")
    return SVC(**kwargs)
