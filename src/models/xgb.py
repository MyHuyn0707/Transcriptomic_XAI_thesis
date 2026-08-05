"""
src/models/xgb.py

XGBoost classifier.
Framework: Classification Transcriptomic with XAI
"""

from typing import Any
from xgboost import XGBClassifier


def build_xgb(random_state: int = 42, **kwargs: Any) -> XGBClassifier:
    """Build an XGBoost classifier."""
    kwargs.setdefault("random_state", random_state)
    kwargs.setdefault("eval_metric", "mlogloss")
    kwargs.setdefault("n_jobs", -1)
    return XGBClassifier(**kwargs)
