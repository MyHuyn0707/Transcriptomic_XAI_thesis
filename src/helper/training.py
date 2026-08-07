"""
src/helper/training.py

Shared "train N repeats, keep the best" bookkeeping used by both
BaselineSplitMixin (baseline_split.py) and HoldoutMixin (holdout.py). The
actual model-fitting loop stays in each caller (they use different model
factories and track different per-repeat data) — only the "pick the best row
by a metric column" and "mean/std per metric across repeats" steps, which
were duplicated verbatim, live here.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def pick_best_run(results_df: pd.DataFrame, select_col: str, fallback_col: str) -> Tuple[int, int]:
    """Return ``(best_pos, best_seed)`` — the row with the highest
    ``select_col``, falling back to ``fallback_col`` when ``select_col`` is
    missing from ``results_df`` or entirely null. ``results_df`` must have a
    ``seed`` column."""
    col = select_col
    if col not in results_df.columns or results_df[col].isnull().all():
        col = fallback_col
    best_pos = int(results_df[col].idxmax())
    best_seed = int(results_df.loc[best_pos, "seed"])
    return best_pos, best_seed


def summarize_repeats(results_df: pd.DataFrame, metric_names: List[str], prefix: str = "") -> pd.DataFrame:
    """Mean/std per metric across repeated-training runs.

    ``metric_names`` are the bare metric names (e.g. ``"f1_macro"``);
    ``prefix`` is prepended to look up each column in ``results_df`` (e.g.
    ``"test_"``) but is stripped back off in the returned ``metric`` column.
    """
    rows = []
    for name in metric_names:
        col = f"{prefix}{name}"
        if (
            col in results_df.columns
            and pd.api.types.is_numeric_dtype(results_df[col])
            and results_df[col].notnull().any()
        ):
            rows.append({
                "metric": name,
                "mean": float(results_df[col].mean()),
                "std": float(results_df[col].std()),
            })
    return pd.DataFrame(rows)
