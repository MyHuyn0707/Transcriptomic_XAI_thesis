"""
src/models/factory.py

Model factory — instantiates a model by name.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

from typing import Any

from .nb import build_nb
from .knn import build_knn
from .svm import build_svm
from .rf import build_rf
from .dt import build_dt
from .xgb import build_xgb
from .ann import SklearnANNWrapper

SUPPORTED_MODELS = ("nb", "knn", "svm", "rf", "dt", "xgboost", "ann")


def get_model(
    name: str,
    input_dim: int = 0,
    num_classes: int = 2,
    random_state: int = 42,
    **kwargs: Any,
) -> Any:
    """
    Instantiate a model by name.

    Parameters
    ----------
    name : str
        One of: nb, knn, svm, rf, dt, xgboost, ann
    input_dim : int
        Number of input features. Required only for ANN.
    num_classes : int
        Number of output classes. Required only for ANN.
    random_state : int
        Reproducibility seed.
    **kwargs :
        Extra hyperparameters passed to the model constructor.

    Returns
    -------
    Unfitted model instance compatible with sklearn API.
    """

    name = name.lower().strip()

    if name == "nb":
        return build_nb(**kwargs)

    elif name == "knn":
        return build_knn(**kwargs)

    elif name == "svm":
        return build_svm(random_state=random_state, **kwargs)

    elif name == "rf":
        return build_rf(random_state=random_state, **kwargs)

    elif name == "dt":
        return build_dt(random_state=random_state, **kwargs)

    elif name == "xgboost":
        return build_xgb(random_state=random_state, **kwargs)

    elif name == "ann":
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0 for ANN.")
        return SklearnANNWrapper(
            input_dim=input_dim,
            num_classes=num_classes,
            random_state=random_state,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown model '{name}'. Supported models: {SUPPORTED_MODELS}"
        )
