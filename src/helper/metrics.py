"""
src/metrics.py

Classification evaluation metrics.
Supports binary and multi-class problems.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Keys compute_metrics() always produces (besides confusion_matrix) — the
# single source of truth for callers building a metric-column list for a
# summary table, instead of hand-maintaining a copy that can drift.
METRIC_COLUMNS = [
    "accuracy", "balanced_accuracy", "f1", "f1_macro",
    "precision", "recall", "roc_auc", "log_loss",
]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "macro",
    labels: Optional[list] = None,
    class_labels: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    y_prob : np.ndarray, optional
        Shape (n_samples, n_classes). Required for roc_auc and log_loss.
    average : str
        Averaging strategy for multi-class metrics: macro, micro, weighted.
    labels : list, optional
        The full set of encoded class labels (e.g. ``range(len(class_labels))``).
        Pass this whenever a class can be entirely absent from ``y_true`` for a
        given split (small/imbalanced datasets) — without it, sklearn's
        ``confusion_matrix`` infers labels from the values actually observed,
        silently shrinking the matrix below ``len(class_labels)`` and crashing
        ``plot_confusion_matrix`` downstream (tick-label count mismatch).
    class_labels : list of str, optional
        Human-readable name per entry of ``labels`` (same order), used only to
        key the ``per_class`` breakdown below. Falls back to ``str(label)``
        when omitted.

    Returns
    -------
    dict with keys:
        accuracy, balanced_accuracy, f1_macro, precision, recall,
        roc_auc, log_loss, confusion_matrix, per_class
    """

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        # Uses the `average` passed in, same as precision/recall above.
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # Per-class (non-aggregated) precision/recall/f1 — same `labels` set used
    # for confusion_matrix below, so a class absent from this split still gets
    # a (zero) row instead of silently disappearing.
    per_class_labels = labels if labels is not None else sorted(np.unique(y_true).tolist())
    prec_arr = precision_score(y_true, y_pred, labels=per_class_labels, average=None, zero_division=0)
    rec_arr = recall_score(y_true, y_pred, labels=per_class_labels, average=None, zero_division=0)
    f1_arr = f1_score(y_true, y_pred, labels=per_class_labels, average=None, zero_division=0)
    metrics["per_class"] = {
        (str(class_labels[i]) if class_labels and i < len(class_labels) else str(lbl)): {
            "precision": float(p), "recall": float(r), "f1": float(f),
        }
        for i, (lbl, p, r, f) in enumerate(zip(per_class_labels, prec_arr, rec_arr, f1_arr))
    }

    n_classes = len(np.unique(y_true))

    if y_prob is not None:
        # ROC-AUC. For binary problems sklearn expects the positive-class score
        # (1-D), not the full (n, 2) probability matrix — passing the 2-column
        # array raises, so slice out the positive class. Multi-class uses OvR.
        try:
            y_prob_arr = np.asarray(y_prob)
            if n_classes <= 2:
                score = (
                    y_prob_arr[:, 1]
                    if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2
                    else y_prob_arr
                )
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, score, average=average)
                )
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_true, y_prob_arr, multi_class="ovr", average=average
                    )
                )
        except ValueError:
            metrics["roc_auc"] = None

        # Log loss
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob))
        except ValueError:
            metrics["log_loss"] = None
    else:
        metrics["roc_auc"] = None
        metrics["log_loss"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    """Pretty-print metrics dict."""

    print("\n" + "=" * 48)
    print("EVALUATION METRICS")
    print("=" * 48)
    print(f"  Accuracy           : {metrics.get('accuracy', 0):.4f}")
    print(f"  Balanced Accuracy  : {metrics.get('balanced_accuracy', 0):.4f}")
    print(f"  F1 (macro)         : {metrics.get('f1_macro', 0):.4f}")
    print(f"  Precision (macro)  : {metrics.get('precision', 0):.4f}")
    print(f"  Recall (macro)     : {metrics.get('recall', 0):.4f}")

    roc_auc = metrics.get("roc_auc")
    if roc_auc is not None:
        print(f"  ROC-AUC            : {roc_auc:.4f}")

    ll = metrics.get("log_loss")
    if ll is not None:
        print(f"  Log Loss           : {ll:.4f}")

    print("=" * 48 + "\n")
