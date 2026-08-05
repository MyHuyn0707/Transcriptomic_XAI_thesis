"""
src/helper/scaling.py

Which models need their input standardized before fit/predict. Distance- and
gradient-based models (knn, svm, ann) need StandardScaler; tree/boosting
models (rf, xgboost, dt) split on raw feature values and don't need it —
scaling them was previously unconditional (harmless numerically, but wastes
the fit/persist step and obscures native feature-space thresholds elsewhere,
e.g. rule extraction).

configs/models.yaml -> scaling is the source of truth; DEFAULT_NEEDS_SCALING
below is only a fallback for callers that don't have a ConfigLoader in scope
(keeps this module usable standalone / in tests).

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

DEFAULT_NEEDS_SCALING: Dict[str, bool] = {
    "nb": True, "knn": True, "svm": True, "ann": True,
    "rf": False, "dt": False, "xgboost": False,
}


def needs_scaling(model_name: str, needs_scaling_map: Optional[Dict[str, bool]] = None) -> bool:
    """Whether ``model_name`` should be scaled. Unknown models default to True (safe)."""
    m = needs_scaling_map if needs_scaling_map is not None else DEFAULT_NEEDS_SCALING
    return bool(m.get(model_name.lower().strip(), True))


def fit_scaler(
    model_name: str,
    X_train: np.ndarray,
    needs_scaling_map: Optional[Dict[str, bool]] = None,
) -> Optional[StandardScaler]:
    """Fit a StandardScaler on X_train, or return None if the model doesn't need one."""
    if not needs_scaling(model_name, needs_scaling_map):
        return None
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(scaler: Optional[StandardScaler], X: np.ndarray) -> np.ndarray:
    """Transform X through scaler, or pass through unchanged when scaler is None."""
    return X if scaler is None else scaler.transform(X)
