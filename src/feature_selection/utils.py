"""
src/feature_selection/utils.py

Shared utilities for feature selection artifact export.

Output structure
----------------
outputs/
└── feature_selection/
    └── {dataset_name}/
        └── {fs_method}/
            ├── selected_features/   ranking CSVs
            ├── processed_datasets/  reduced dataset CSVs
            ├── params/              JSON metadata
            └── logs/

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Directories
# =============================================================================


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_fs_dirs(
    output_root: str | Path,
    dataset_name: str,
    fs_method: str,
) -> Dict[str, Path]:
    """
    Create and return the standard feature-selection directory tree.

    Returns
    -------
    dict with keys: base, selected_features, processed_datasets, params, logs
    """

    base = Path(output_root) / "feature_selection" / dataset_name / fs_method

    return {
        "base": ensure_dir(base),
        "selected_features": ensure_dir(base / "selected_features"),
        "processed_datasets": ensure_dir(base / "processed_datasets"),
        "params": ensure_dir(base / "params"),
        "logs": ensure_dir(base / "logs"),
    }


# =============================================================================
# CSV export
# =============================================================================


def save_dataframe(
    df: pd.DataFrame,
    save_path: str | Path,
    index: bool = False,
) -> Path:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    df.to_csv(save_path, index=index)
    return save_path


def save_ranking(
    ranking_df: pd.DataFrame,
    save_dir: str | Path,
    filename: str,
) -> Path:
    return save_dataframe(ranking_df, Path(save_dir) / filename)


# =============================================================================
# JSON export
# =============================================================================


def convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialisation."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    return obj


def save_json(data: Dict[str, Any], save_path: str | Path) -> Path:
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(convert_numpy(data), f, indent=4, ensure_ascii=False)
    return save_path


def save_params(params: Dict[str, Any], save_dir: str | Path) -> Path:
    return save_json(params, Path(save_dir) / "params_des.json")


def save_metadata(metadata: Dict[str, Any], save_dir: str | Path) -> Path:
    return save_json(metadata, Path(save_dir) / "metadata.json")


# =============================================================================
# Processed dataset export
# =============================================================================


def build_processed_dataset(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    sample_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Assemble a DataFrame ready to export.

    Format
    ------
    [sample_id,] feature_1, ..., feature_n, label
    """
    df = pd.DataFrame(X, columns=feature_names)
    if sample_ids is not None:
        df.insert(0, "sample_id", sample_ids)
    df["label"] = y
    return df


def save_processed_dataset(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    save_path: str | Path,
    sample_ids: Optional[List[str]] = None,
) -> Path:
    df = build_processed_dataset(X, y, feature_names, sample_ids)
    return save_dataframe(df, save_path)


# =============================================================================
# Feature table helpers
# =============================================================================


def build_feature_table(
    feature_names: List[str],
    scores: Optional[np.ndarray] = None,
    ranks: Optional[np.ndarray] = None,
    decisions: Optional[List[str]] = None,
) -> pd.DataFrame:
    data: Dict[str, Any] = {"Feature": feature_names}
    if ranks is not None:
        data["Rank"] = ranks
    if scores is not None:
        data["Score"] = scores
    if decisions is not None:
        data["Decision"] = decisions
    return pd.DataFrame(data)


# =============================================================================
# Summary
# =============================================================================


def build_fs_summary(
    dataset_name: str,
    fs_method: str,
    n_samples: int,
    n_original_features: int,
    n_selected_features: int,
    runtime_seconds: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = {
        "framework": "Classification Transcriptomic with XAI",
        "dataset_name": dataset_name,
        "feature_selection": fs_method,
        "n_samples": int(n_samples),
        "n_original_features": int(n_original_features),
        "n_selected_features": int(n_selected_features),
        "runtime_seconds": float(runtime_seconds),
    }
    if extra:
        summary.update(convert_numpy(extra))
    return summary


def print_fs_summary(summary: Dict[str, Any]) -> None:
    import pandas as pd

    from src.helper import report

    rows = [{"field": k, "value": str(v)} for k, v in summary.items()]
    report.dataframe_table(
        pd.DataFrame(rows),
        title="Feature selection summary",
        index=False,
    )
