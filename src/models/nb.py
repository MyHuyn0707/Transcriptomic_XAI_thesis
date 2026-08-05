"""
src/models/nb.py

Gaussian Naive Bayes classifier.
Author: Thesis Framework
"""

from sklearn.naive_bayes import GaussianNB
from typing import Any


def build_nb(**kwargs: Any) -> GaussianNB:
    """
    Build a Gaussian Naive Bayes model.
    """
    return GaussianNB(**kwargs)
