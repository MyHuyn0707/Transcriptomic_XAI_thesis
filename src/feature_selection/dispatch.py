"""
src/feature_selection/dispatch.py

Dispatch to the configured feature-selection method (raw, mrmr*, boruta) —
the one place that maps an ``fs_method`` string to its implementation, shared
by every caller that runs feature selection (full-dataset or train-only).

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .raw import run_raw_selection
from .mrmr_fs import run_mrmr_selection
from .boruta_fs import run_boruta_selection


def run_feature_selection(
    fs_method: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    sample_ids: List[str],
    dataset_name: str,
    output_root: str | Path,
    params: Dict[str, Any],
    allow_raw: bool = True,
    source_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run one feature-selection method and export its artifacts.

    Parameters
    ----------
    fs_method : str
        'raw', 'mrmr_mid'/'mrmr_k50'/'mrmr_k75' (anything starting with 'mrmr'), or 'boruta'.
    allow_raw : bool
        Whether 'raw' (zero-copy passthrough) is a valid choice here.
    source_path : str, optional
        Original dataset path, forwarded to ``run_raw_selection`` as its
        zero-copy source pointer (required when ``fs_method == 'raw'``).
    """
    if fs_method == "raw":
        if not allow_raw:
            raise ValueError("fs_method 'raw' is not supported here.")
        return run_raw_selection(
            X=X, y=y, feature_names=feature_names,
            dataset_name=dataset_name, output_root=output_root,
            sample_ids=sample_ids,
            source_path=source_path,
        )

    elif fs_method.startswith("mrmr"):
        return run_mrmr_selection(
            X=X, y=y, feature_names=feature_names,
            dataset_name=dataset_name, output_root=output_root,
            sample_ids=sample_ids,
            fs_label_override=fs_method,
            **params,
        )

    elif fs_method == "boruta":
        return run_boruta_selection(
            X=X, y=y, feature_names=feature_names,
            dataset_name=dataset_name, output_root=output_root,
            sample_ids=sample_ids,
            **params,
        )

    else:
        supported = "raw, mrmr, boruta" if allow_raw else "mrmr, boruta"
        raise ValueError(
            f"Unknown feature selection method: '{fs_method}'. Supported: {supported}."
        )
