"""
src/feature_selection/raw.py

Raw (no-op) feature selection.

Provides a unified interface identical to mrmr_fs and boruta_fs so that
all three methods are interchangeable in the pipeline.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import (
    build_feature_table,
    build_fs_summary,
    create_fs_dirs,
    print_fs_summary,
    save_metadata,
    save_params,
    save_ranking,
)


def run_raw_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    dataset_name: str,
    output_root: str | Path,
    sample_ids: Optional[List[str]] = None,
    source_path: Optional[str | Path] = None,
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Identity feature selection — returns all features unchanged.

    Zero-copy baseline: unlike mRMR/Boruta, this does NOT write a reduced
    dataset to disk (it would be a full clone of a 20k+ probe matrix). Instead
    it records a pointer (``source_path``) back to the original on-disk CuMiDa
    dataset, which interpretation resolves and re-loads on demand.

    Parameters
    ----------
    X : np.ndarray          Shape (n_samples, n_features)
    y : np.ndarray          Shape (n_samples,)
    feature_names : list
    dataset_name : str
    output_root : str | Path
    sample_ids : list, optional
    source_path : str | Path, optional
        Path to the original on-disk dataset. Stored as the baseline's source
        pointer (no data is copied).
    params : dict, optional
    verbose : bool

    Returns
    -------
    dict with keys:
        X_selected, y_selected, selected_features, selected_indices,
        ranking_df, summary, output_dirs, runtime_seconds
    """

    start_time = time.time()

    dirs = create_fs_dirs(
        output_root=output_root,
        dataset_name=dataset_name,
        fs_method="raw",
    )

    X_selected = X.copy()
    y_selected = y.copy()
    selected_features = list(feature_names)
    selected_indices = np.arange(len(feature_names), dtype=np.int64)

    # Ranking (identity — all features, rank by position)
    ranking_df = build_feature_table(
        feature_names=selected_features,
        ranks=np.arange(1, len(selected_features) + 1),
    )

    save_ranking(
        ranking_df=ranking_df,
        save_dir=dirs["selected_features"],
        filename="ranking.csv",
    )

    # Zero-copy baseline: do NOT clone the full matrix to disk.
    # Record a pointer back to the original on-disk dataset instead.
    source_pointer = str(source_path) if source_path is not None else None
    dirs["source_path"] = source_pointer

    runtime = time.time() - start_time

    summary = build_fs_summary(
        dataset_name=dataset_name,
        fs_method="raw",
        n_samples=X.shape[0],
        n_original_features=X.shape[1],
        n_selected_features=X.shape[1],
        runtime_seconds=runtime,
        extra={
            "description": "All features retained (no selection).",
            "materialized": False,
            "source_path": source_pointer,
        },
    )

    save_metadata(metadata=summary, save_dir=dirs["params"])

    fs_params = {
        "dataset_name": dataset_name,
        "feature_selection": "raw",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "materialized": False,
        "source_path": source_pointer,
    }
    if params:
        fs_params.update(params)
    save_params(params=fs_params, save_dir=dirs["params"])

    if verbose:
        print_fs_summary(summary)

    return {
        "X_selected": X_selected,
        "y_selected": y_selected,
        "selected_features": selected_features,
        "selected_indices": selected_indices,
        "ranking_df": ranking_df,
        "source_path": source_pointer,
        "summary": summary,
        "output_dirs": dirs,
        "runtime_seconds": runtime,
    }
