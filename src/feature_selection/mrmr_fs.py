"""
src/feature_selection/mrmr_fs.py

mRMR (Minimum Redundancy Maximum Relevance) feature selection.

Implementation
--------------
- Uses mrmr-selection (mrmr_classif) — already in pyproject.toml.
  Relevance : F-statistic (ANOVA F-test) — suitable for classification.
  Redundancy: Pearson correlation.
- No discretisation required (unlike pymrmr which needed integer bins).
- MI scores via sklearn.mutual_info_classif computed AFTER selection
  for ranking_explained.csv — does not affect which features are selected.

Config parameters
-----------------
criterion : str  Legacy — informational only. mrmr-selection does not
                 distinguish MID/MIQ; the F-statistic criterion is used
                 for both. Kept for API / config compatibility.
K         : int  Number of features to select.
n_bins    : int  Legacy no-op — no discretisation needed.

Note on dataset scope
---------------------
Feature selection is performed on the FULL dataset (all samples).
Model evaluation leakage is controlled separately by fitting scalers
and models inside CV folds in trainer.py.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from sklearn.feature_selection import f_classif

from .utils import (
    build_fs_summary,
    create_fs_dirs,
    print_fs_summary,
    save_json,
    save_metadata,
    save_params,
    save_processed_dataset,
    save_ranking,
)


# =============================================================================
# True mRMR incremental ranking (MID / MIQ)
# =============================================================================

# Floor on the redundancy denominator for MIQ, mirroring mrmr-selection's
# clipping so the score stays finite when correlation is ~0.
_REDUNDANCY_FLOOR = 1e-3


def compute_mrmr_ranking(
    X: np.ndarray,
    y: np.ndarray,
    selected_features: List[str],
    feature_names: List[str],
    variant: str = "MID",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Reconstruct the true incremental mRMR score for each selected feature,
    evaluated strictly against the subset of features selected BEFORE it.

    This does NOT change selection — `mrmr_classif` still drives which features
    are picked. It only scores/ranks the already-selected set in the variant's
    own metric.

    Score components (matching the mrmr-selection F-statistic engine)
    ---------------------------------------------------------------
    relevance  : ANOVA F-statistic of the feature vs. the target (univariate;
                 `sklearn.feature_selection.f_classif`).
    redundancy : mean |Pearson correlation| of the feature with the features
                 selected BEFORE it in greedy order (0 for Rank 1).

    Incremental rule
    ----------------
    Rank 1  → Redundancy = 0
    Rank 2  → Redundancy = |corr(f2, f1)|
    Rank k  → Redundancy = mean( |corr(fk, f1)|, ..., |corr(fk, f_{k-1})| )

    variant
    -------
    "MID" : score = relevance - redundancy        (difference)
    "MIQ" : score = relevance / redundancy         (quotient; floored)

    Returns
    -------
    pd.DataFrame with columns:
        Rank, Feature, Relevance_F, Redundancy, mRMR_{MID|MIQ}_Score
    in mRMR selection order (Rank 1 = first selected).
    """
    variant = variant.upper()
    if variant not in ("MID", "MIQ"):
        variant = "MID"

    feature_to_idx = {f: i for i, f in enumerate(feature_names)}

    # Extract each selected feature's column from X (preserving selection order).
    sel_cols: List[np.ndarray] = [X[:, feature_to_idx[f]] for f in selected_features]

    # Relevance — univariate ANOVA F-statistic (independent of subset).
    X_sel = np.column_stack(sel_cols) if len(sel_cols) > 1 else sel_cols[0].reshape(-1, 1)
    f_scores, _ = f_classif(X_sel, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)

    score_col = f"mRMR_{variant}_Score"
    rows: List[Dict[str, Any]] = []

    # Incrementally build the "already selected" pool.
    # For Rank k, redundancy is computed ONLY against the k-1 features already
    # in `prior_cols` — never against features selected later.
    prior_cols: List[np.ndarray] = []

    for i, feat in enumerate(selected_features):
        rel = float(f_scores[i])
        feat_col = sel_cols[i]

        if not prior_cols:
            # Rank 1: no prior selection → redundancy = 0
            red = 0.0
        else:
            # mean |Pearson r| between this feature and each prior feature
            corrs = [
                abs(float(np.corrcoef(feat_col, prev)[0, 1]))
                for prev in prior_cols
            ]
            red = float(np.mean(corrs))
            red = 0.0 if np.isnan(red) else red

        if variant == "MIQ":
            score = rel / max(red, _REDUNDANCY_FLOOR)
        else:  # MID
            score = rel - red

        rows.append(
            {
                "Rank": i + 1,
                "Feature": feat,
                "Relevance_F": round(rel, 6),
                "Redundancy": round(red, 6),
                score_col: round(float(score), 6),
            }
        )

        # Only AFTER scoring do we add this feature to the prior pool.
        prior_cols.append(feat_col)

    return pd.DataFrame(rows)


def compute_mi_ranking(
    X: np.ndarray,
    y: np.ndarray,
    selected_features: List[str],
    feature_names: List[str],
    random_state: int = 42,
) -> pd.DataFrame:
    """Deprecated alias — kept for backward compatibility.

    Routes to :func:`compute_mrmr_ranking` (MID variant). The old post-hoc
    Mutual-Information ranking is no longer produced.
    """
    return compute_mrmr_ranking(
        X, y, selected_features, feature_names,
        variant="MID", random_state=random_state,
    )


# =============================================================================
# Public entry point
# =============================================================================


def run_mrmr_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    dataset_name: str,
    output_root: str | Path,
    sample_ids: Optional[List[str]] = None,
    criterion: str = "MID",
    K: int = 50,
    n_bins: int = 3,
    random_state: int = 42,
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    fs_label_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run mRMR feature selection on the full dataset and export artifacts.

    Parameters
    ----------
    X : np.ndarray          (n_samples, n_features)
    y : np.ndarray          (n_samples,) integer labels
    feature_names : list
    dataset_name : str
    output_root : str | Path
    sample_ids : list, optional
    criterion : str         Legacy — informational only (F-statistic is used).
    K : int                 Number of features to select.
    n_bins : int            Legacy no-op.
    random_state : int      Used for post-selection MI score computation.
    params : dict, optional Extra params merged into params_des.json.
    verbose : bool
    fs_label_override : str, optional
        On-disk fs_method folder/artifact-name to use instead of the derived
        ``mrmr_{criterion}``. Required whenever more than one config entry
        shares the same criterion (e.g. ``mrmr_k50``/``mrmr_k75``, both
        criterion="MIQ") — otherwise they'd all write to the same
        ``mrmr_miq/`` folder and overwrite each other's artifacts. Dispatch
        (src/feature_selection/dispatch.py) passes the caller's original
        fs_method string here.

    Returns
    -------
    dict with keys:
        X_selected, y_selected, selected_features, selected_indices,
        ranking_df, ranking_explained_df, summary, output_dirs, runtime_seconds
    """

    start_time = time.time()
    criterion = criterion.upper()

    if K > X.shape[1]:
        warnings.warn(
            f"K={K} exceeds n_features={X.shape[1]}. Using K={X.shape[1]}.",
            stacklevel=2,
        )
        K = X.shape[1]

    fs_label = fs_label_override or f"mrmr_{criterion.lower()}"

    dirs = create_fs_dirs(
        output_root=output_root,
        dataset_name=dataset_name,
        fs_method=fs_label,
    )

    # ---- mRMR selection via mrmr-selection ----
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    selected_features: List[str] = mrmr_classif(X=X_df, y=y_series, K=K)

    # ---- Indices ----
    feature_to_idx = {f: i for i, f in enumerate(feature_names)}
    selected_indices = np.array(
        [feature_to_idx[f] for f in selected_features], dtype=np.int64
    )
    X_selected = X[:, selected_indices]

    # ---- Original mRMR order ranking (ranking.csv) ----
    ranking_df = pd.DataFrame(
        {
            "Rank": np.arange(1, len(selected_features) + 1),
            "Feature": selected_features,
        }
    )
    save_ranking(ranking_df=ranking_df, save_dir=dirs["selected_features"], filename="ranking.csv")

    # ---- True mRMR incremental ranking (MID / MIQ) ----
    variant = criterion if criterion in ("MID", "MIQ") else "MID"
    ranking_scored_df = compute_mrmr_ranking(
        X=X,
        y=y,
        selected_features=selected_features,
        feature_names=feature_names,
        variant=variant,
        random_state=random_state,
    )
    # e.g. mrmr_mid_feature_rankings.csv / mrmr_miq_feature_rankings.csv
    rankings_filename = f"{fs_label}_feature_rankings.csv"
    save_ranking(
        ranking_df=ranking_scored_df,
        save_dir=dirs["selected_features"],
        filename=rankings_filename,
    )

    # ---- Selected features JSON ----
    save_json(
        {"selected_features": selected_features},
        dirs["selected_features"] / "selected_features.json",
    )

    # ---- Processed dataset ----
    dataset_filename = f"{dataset_name}_{fs_label}_K{K}.csv"
    save_processed_dataset(
        X=X_selected,
        y=y,
        feature_names=selected_features,
        save_path=dirs["processed_datasets"] / dataset_filename,
        sample_ids=sample_ids,
    )

    runtime = time.time() - start_time

    # ---- Summary & params ----
    summary = build_fs_summary(
        dataset_name=dataset_name,
        fs_method=fs_label,
        n_samples=X.shape[0],
        n_original_features=X.shape[1],
        n_selected_features=len(selected_features),
        runtime_seconds=runtime,
        extra={
            "criterion": criterion,
            "K": K,
            "implementation": "mrmr-selection (F-statistic relevance)",
        },
    )
    save_metadata(metadata=summary, save_dir=dirs["params"])

    fs_params = {
        "dataset_name": dataset_name,
        "feature_selection": fs_label,
        "criterion": criterion,
        "K": K,
        "random_state": random_state,
    }
    if params:
        fs_params.update(params)
    save_params(params=fs_params, save_dir=dirs["params"])

    if verbose:
        from src.helper import report

        print_fs_summary(summary)
        report.dataframe_table(
            ranking_scored_df.head(15),
            title=f"mRMR-{variant} feature ranking",
            index=False,
        )

    return {
        "X_selected": X_selected,
        "y_selected": y.copy(),
        "selected_features": selected_features,
        "selected_indices": selected_indices,
        "ranking_df": ranking_df,
        "ranking_scored_df": ranking_scored_df,
        "ranking_explained_df": ranking_scored_df,   # backward-compat alias key
        "variant": variant,
        "summary": summary,
        "output_dirs": dirs,
        "runtime_seconds": runtime,
    }
