"""
src/helper/split.py

Stratified train/test split for the holdout rule-extraction flow. Rare-class
removal is not this module's job — run it first (e.g. load_dataset's
min_samples_per_class), since every class needs >= 2 members for stratify.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def train_test_split_indices(
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Stratified train/test split, returning row indices only (not the data
    itself) so callers can slice X/y/sample_ids consistently.

    Splits each class independently (instead of calling
    ``sklearn.model_selection.train_test_split(stratify=y)`` directly) to
    GUARANTEE every class contributes >= 1 sample to both train and test.
    sklearn's own stratify rounds each class's share of test_size to the
    nearest integer, which can round a rare class (e.g. 2-3 members) down to
    0 in one split without raising — confirmed empirically: a class with 2
    members at test_size=0.15 landed entirely in train, 0 in test. That
    silently makes the class unscoreable on the held-out set and invisible
    to the UI's "pick a test sample" feature. Rare-class removal upstream
    (e.g. load_dataset's min_samples_per_class) guarantees every class here
    has >= 2 members, so ``len(idx_cls) - 1 >= 1`` always holds — no class
    can end up with 0 samples in either split.

    Returns
    -------
    dict with keys ``train_idx``, ``test_idx`` (np.ndarray of row indices,
    each sorted ascending for reproducible, readable ordering).
    """
    rng = np.random.RandomState(random_state)
    y = np.asarray(y)
    train_idx: list = []
    test_idx: list = []
    for cls in np.unique(y):
        idx_cls = np.where(y == cls)[0]
        rng.shuffle(idx_cls)
        n_test_cls = max(1, min(round(len(idx_cls) * test_size), len(idx_cls) - 1))
        test_idx.extend(idx_cls[:n_test_cls])
        train_idx.extend(idx_cls[n_test_cls:])
    return {
        "train_idx": np.sort(np.array(train_idx, dtype=np.int64)),
        "test_idx": np.sort(np.array(test_idx, dtype=np.int64)),
    }


def build_dataset_split(
    ds_cfg: Dict[str, Any],
    min_samples_per_class: int,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Load a dataset, drop rare classes at ``min_samples_per_class``, and
    produce a stratified train/test split — the single implementation shared
    by the batch train/test-split baseline
    (``src/pipeline/baseline_split.py::run_baseline_split``) and the live
    UI's on-demand split (``src/api/registry.compute_holdout_split``), so the
    two never drift out of sync even though they read their
    test_size/random_state from different config blocks
    (``holdout.split_baseline`` vs. ``holdout.live_ui_defaults``).

    Returns
    -------
    dict with: ``data`` (the raw load_dataset() result), ``X``, ``y``,
    ``feature_names``, ``sample_ids``, ``train_idx``, ``test_idx``,
    ``class_labels``, ``raw_class_counts``, ``dropped_classes``, plus the
    resolved ``min_samples_per_class``/``test_size``/``split_random_state``.
    """
    from src.helper.data_loader import load_dataset

    data = load_dataset(
        dataset_path=Path(ds_cfg["path"]),
        dataset_type=ds_cfg.get("type", "auto"),
        min_samples_per_class=min_samples_per_class,
    )
    X, y = data["X"], data["y"]

    # Real class names aligned with y — same preference order as
    # GeneExpressionPipeline._resolve_class_labels (encoder.classes_ first,
    # then the dataset's configured class_labels, then empty).
    encoder = data.get("label_encoder")
    classes = getattr(encoder, "classes_", None)
    if classes is not None and len(classes):
        class_labels = [str(c) for c in classes]
    else:
        class_labels = list(ds_cfg.get("class_labels", []) or [])

    split = train_test_split_indices(y, test_size=test_size, random_state=random_state)
    train_idx, test_idx = split["train_idx"], split["test_idx"]

    # Class distribution BEFORE rare-class removal, straight from the
    # already-loaded raw dataframe (data["dataframe"] — see load_dataset()'s
    # docstring), same source plot_class_distribution() uses.
    raw_counts_series = data["dataframe"].iloc[:, 1].astype(str).value_counts()
    raw_class_counts = {str(c): int(n) for c, n in raw_counts_series.items()}
    dropped_classes = [c for c in raw_class_counts if c not in class_labels]

    return {
        "data": data,
        "X": X,
        "y": y,
        "feature_names": data["feature_names"],
        "sample_ids": data["sample_ids"],
        "train_idx": train_idx,
        "test_idx": test_idx,
        "class_labels": class_labels,
        "raw_class_counts": raw_class_counts,
        "dropped_classes": dropped_classes,
        "min_samples_per_class": min_samples_per_class,
        "test_size": test_size,
        "split_random_state": random_state,
    }


def build_test_set_artifacts(
    feature_names: List[str],
    sample_ids: List[str],
    X: np.ndarray,
    y: np.ndarray,
    test_idx: np.ndarray,
    class_labels: List[str],
) -> Dict[str, Any]:
    """
    Build the held-out test set in the raw (pre-feature-selection) feature
    space — the single implementation shared by
    ``run_baseline_split()`` (writes it to
    ``outputs_baseline_split/k{k}/{dataset}/test_set/``) and the live UI's
    "download test set" endpoint (zips it in-memory, no disk write, for a
    dataset/split that has no cache yet — a fresh upload or a custom
    "Thực hiện lại" split).

    Returns
    -------
    dict with:
      ``test_set_df``  : pd.DataFrame — CuMiDa-format (samples, type, <features...>)
      ``manifest_df``  : pd.DataFrame — sample_id, true_label, row_index
      ``samples``      : {sample_id: {"sample_id", "true_label", "features"}} —
                          same shape as a single test_set/samples/{id}.json
                          export, so each entry round-trips through the
                          existing "upload 1 sample to predict" flow
                          (src/api/inference.py::parse_uploaded_sample).
    """
    import pandas as pd

    test_sample_ids = [str(sample_ids[i]) for i in test_idx]
    test_labels = [
        class_labels[c] if c < len(class_labels) else str(c) for c in y[test_idx]
    ]
    X_test = X[test_idx]

    test_set_df = pd.DataFrame(X_test, columns=feature_names)
    test_set_df.insert(0, "type", test_labels)
    test_set_df.insert(0, "samples", test_sample_ids)

    manifest_df = pd.DataFrame({
        "sample_id": test_sample_ids,
        "true_label": test_labels,
        "row_index": np.arange(len(test_sample_ids)),
    })

    samples: Dict[str, Any] = {}
    for row_i, (sid, label) in enumerate(zip(test_sample_ids, test_labels)):
        samples[sid] = {
            "sample_id": sid,
            "true_label": label,
            "features": {fn: float(v) for fn, v in zip(feature_names, X_test[row_i])},
        }

    return {"test_set_df": test_set_df, "manifest_df": manifest_df, "samples": samples}
