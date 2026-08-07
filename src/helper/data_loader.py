"""
src/data_loader.py

Gene expression dataset loader — CuMiDa format.

Format
------
samples, type, probe_1, probe_2, ...
  col 0  : sample ID  (string, e.g. GSM519812)
  col 1  : class label (string, e.g. "LumA")
  col 2+ : feature values (float32)

The GEO probe/gene CSVs produced by src/dataset_builder.py share this layout.

Outputs
-------
X               : np.ndarray (float32)  shape (n_samples, n_features)
y               : np.ndarray (int64)    0-based encoded class labels
feature_names   : List[str]             probe / gene names
sample_ids      : List[str]             GSM IDs or equivalent
metadata        : dict
label_encoder   : LabelEncoder          always present; use .classes_ for names
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# =============================================================================
# Label encoding
# =============================================================================


def encode_labels(
    y_raw: pd.Series,
) -> Tuple[np.ndarray, LabelEncoder, Dict[str, int]]:
    """
    Encode class labels to contiguous integers starting at 0.

    Always uses LabelEncoder so callers can rely on .classes_ for
    inverse-mapping, regardless of whether labels are strings or numbers.
    """
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw.astype(str)).astype(np.int64)
    mapping = {str(cls): int(idx) for idx, cls in enumerate(encoder.classes_)}
    return y, encoder, mapping


# =============================================================================
# CuMiDa loader
# =============================================================================


def load_cumida_dataset(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Dict, LabelEncoder]:
    """
    Load a CuMiDa-format DataFrame.

    Expected columns: [samples, type, probe_1, probe_2, ...]
    """
    if df.shape[1] < 3:
        raise ValueError(
            "Dataset must have at least 3 columns: sample_id, class, and one feature."
        )

    sample_ids = df.iloc[:, 0].astype(str).tolist()
    y_raw = df.iloc[:, 1]
    y, encoder, label_mapping = encode_labels(y_raw)

    feature_names = list(df.columns[2:])
    X = df.iloc[:, 2:].to_numpy(dtype=np.float32, copy=True)

    metadata = {
        "dataset_type": "cumida",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(np.unique(y).size),
        "label_mapping": label_mapping,
    }

    return X, y, feature_names, sample_ids, metadata, encoder


# =============================================================================
# Missing value handling
# =============================================================================


def handle_missing_values(
    X: np.ndarray,
    strategy: str = "mean",
    feature_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Impute missing values (NaN) in the feature matrix.

    Parameters
    ----------
    strategy : 'mean' | 'median' | 'zero' | 'drop'
    feature_names : column names for X, kept in sync when strategy='drop'
        removes columns. Required (and only meaningful) for 'drop'.

    Returns
    -------
    (X, feature_names) — feature_names is unchanged unless strategy='drop'.
    """
    feature_names = list(feature_names) if feature_names is not None else []

    if not np.isnan(X).any():
        return X, feature_names

    n_missing = int(np.isnan(X).sum())
    print(f"  Warning: {n_missing} missing values detected. Strategy: '{strategy}'.")

    if strategy == "mean":
        X = X.copy()
        col_means = np.nanmean(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_means, idx[1])

    elif strategy == "median":
        X = X.copy()
        col_medians = np.nanmedian(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(col_medians, idx[1])

    elif strategy == "zero":
        X = np.nan_to_num(X, nan=0.0)

    elif strategy == "drop":
        keep = ~np.isnan(X).any(axis=0)
        X = X[:, keep]
        if feature_names:
            feature_names = [f for f, k in zip(feature_names, keep) if k]
        print(
            f"  Dropped {int((~keep).sum())} features with missing values. "
            f"Remaining: {int(keep.sum())}."
        )

    else:
        raise ValueError(f"Unknown missing_strategy '{strategy}'. "
                         "Choose: 'mean', 'median', 'zero', 'drop'.")

    return X, feature_names


# =============================================================================
# Rare-class removal
# =============================================================================


def remove_rare_classes(
    X: np.ndarray,
    y: np.ndarray,
    min_samples: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove samples belonging to classes with fewer than min_samples members.

    Necessary before StratifiedKFold (which requires >= n_splits samples per class).
    Re-encodes y to 0-based contiguous after removal.
    """
    counts = np.bincount(y)
    rare = np.where(counts < min_samples)[0]

    if rare.size == 0:
        return X, y

    print(
        f"  Warning: removing {len(rare)} class(es) with < {min_samples} samples: "
        f"{rare.tolist()}"
    )
    valid_classes = np.where(counts >= min_samples)[0]
    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y).astype(np.int64)
    return X, y


# =============================================================================
# Main entry point
# =============================================================================


def load_dataset(
    dataset_path: str | Path,
    dataset_type: str = "cumida",
    missing_strategy: str = "mean",
    remove_rare: bool = True,
    min_samples_per_class: int = 2,
    **read_csv_kwargs,
) -> Dict[str, Any]:
    """
    Load a CuMiDa-format gene expression CSV.

    Parameters
    ----------
    dataset_path : str | Path
        Path to the CSV (plain or gzip-compressed).
    dataset_type : str
        Accepted for backward compatibility; must be 'cumida' or 'auto'.
    missing_strategy : str
        How to handle NaN values: 'mean', 'median', 'zero', 'drop'.
    remove_rare : bool
        Drop classes with fewer than min_samples_per_class samples.
    min_samples_per_class : int
        Minimum samples required per class.
    **read_csv_kwargs
        Forwarded to pandas.read_csv.

    Returns
    -------
    dict
        X             np.ndarray float32 (n_samples, n_features)
        y             np.ndarray int64   0-based class labels
        feature_names list[str]
        sample_ids    list[str]
        metadata      dict  (n_samples, n_features, n_classes, label_mapping, …)
        label_encoder LabelEncoder
        dataframe     pd.DataFrame  (original, before any filtering)
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_type.lower() not in {"cumida", "auto"}:
        raise ValueError(
            f"dataset_type='{dataset_type}' is not supported. "
            "All datasets must be in CuMiDa format (samples, type, probe_1, ...)."
        )

    read_kwargs: dict = {"low_memory": False}
    read_kwargs.update(read_csv_kwargs)
    df = pd.read_csv(dataset_path, **read_kwargs)

    X, y, feature_names, sample_ids, metadata, encoder = load_cumida_dataset(df)

    X, feature_names = handle_missing_values(X, strategy=missing_strategy, feature_names=feature_names)
    metadata["n_features"] = int(X.shape[1])

    if remove_rare:
        # Record which old integer labels survive BEFORE removal, so we can
        # rebuild label_encoder and metadata to stay in sync with the new y.
        counts = np.bincount(y)
        surviving_old_labels = np.where(counts >= min_samples_per_class)[0]

        # Same per-sample mask remove_rare_classes applies internally. We apply
        # it to sample_ids too, otherwise sample_ids keeps its original length
        # while X/y shrink — any downstream DataFrame that pairs sample_ids with
        # X rows (e.g. build_processed_dataset) then raises a length mismatch.
        keep_mask = np.isin(y, surviving_old_labels)

        X, y = remove_rare_classes(X, y, min_samples=min_samples_per_class)
        sample_ids = [s for s, keep in zip(sample_ids, keep_mask) if keep]
        metadata["n_samples"] = int(X.shape[0])

        if len(surviving_old_labels) < len(encoder.classes_):
            # Rebuild encoder so .classes_ only contains surviving class names,
            # in the same order as the re-encoded y (sorted by old label index).
            surviving_names = encoder.classes_[surviving_old_labels]
            encoder = LabelEncoder()
            encoder.fit(surviving_names)

            metadata["n_classes"] = int(np.unique(y).size)
            metadata["label_mapping"] = {
                str(cls): int(idx) for idx, cls in enumerate(encoder.classes_)
            }

    metadata["dataset_path"] = str(dataset_path)
    metadata["min_samples_per_class"] = min_samples_per_class

    return {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "sample_ids": sample_ids,
        "metadata": metadata,
        "label_encoder": encoder,
        "dataframe": df,
    }


# =============================================================================
# FS-reduced matrix loader
# =============================================================================


def load_fs_reduced_matrix(csv_dir: str | Path) -> Optional[Tuple[np.ndarray, List[str]]]:
    """
    Load the feature-selection-reduced CuMiDa CSV written under
    ``<output_root>/feature_selection/<dataset>/<fs_method>/processed_datasets/``.

    Returns None if csv_dir has no CSV yet (caller decides how to report that).
    If more than one CSV is present, takes the alphabetically-first one.
    """
    csv_dir = Path(csv_dir)
    csvs = sorted(csv_dir.glob("*.csv"))
    if not csvs:
        return None
    df_fs = pd.read_csv(csvs[0], low_memory=False)
    if "sample_id" in df_fs.columns:
        df_fs = df_fs.drop(columns=["sample_id"])
    feature_names = list(df_fs.columns[:-1])
    X = df_fs[feature_names].to_numpy(dtype=np.float32)
    return X, feature_names
