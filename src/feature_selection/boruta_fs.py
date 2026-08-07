"""
src/feature_selection/boruta_fs.py

Boruta feature selection using BorutaPy + RandomForestClassifier.

Exports
-------
- Confirmed features  → data_{dataset}_boruta_confirmed.csv
- Confirmed+Tentative → data_{dataset}_boruta_confirmed_tentative.csv (only when non-empty)
- ranking.csv         → Feature, Decision, Boruta_Rank, MI_Score
- selected_features.json
- support.csv         → raw BorutaPy output
- params_des.json / metadata.json

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

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
# Estimator
# =============================================================================


def build_boruta_estimator(
    n_estimators: int = 500,
    max_depth: Optional[int] = None,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """
    Build the RandomForest used as Boruta's internal estimator.

    Parameters
    ----------
    n_estimators : int        Number of trees. 500 is the recommended default.
    max_depth : int | None    Tree depth. None = unlimited (original Boruta default).
    class_weight : str | None 'balanced' recommended for imbalanced gene datasets.
    """

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )


# =============================================================================
# Core Boruta run
# =============================================================================


def run_boruta(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int | str = "auto",
    max_iter: int = 100,
    perc: int = 100,
    alpha: float = 0.05,
    rf_n_estimators: int = 500,
    max_depth: Optional[int] = None,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 0,
) -> BorutaPy:
    """
    Fit BorutaPy and return the fitted object.

    Parameters
    ----------
    n_estimators : int | 'auto'
        Trees per Boruta shadow-feature iteration ('auto' = sqrt(n_features)).
    max_iter : int      Maximum Boruta iterations.
    perc : int          Percentile of max shadow feature importance (100 = max).
    alpha : float       FWER threshold for Bonferroni correction.
    rf_n_estimators : int  Trees in the internal RF estimator.
    max_depth : int | None  Depth of RF trees (None = unlimited).
    """

    estimator = build_boruta_estimator(
        n_estimators=rf_n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    boruta = BorutaPy(
        estimator=estimator,
        n_estimators=n_estimators,
        perc=perc,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
    )

    boruta.fit(X, y)
    return boruta


# =============================================================================
# Post-selection helpers
# =============================================================================


def build_support_dataframe(
    feature_names: List[str], boruta: BorutaPy
) -> pd.DataFrame:
    """Raw BorutaPy outputs as a DataFrame."""
    return pd.DataFrame(
        {
            "Feature": feature_names,
            "Support": boruta.support_,
            "Support_Weak": boruta.support_weak_,
            "Boruta_Rank": boruta.ranking_,
        }
    )


def extract_feature_groups(
    support_df: pd.DataFrame,
) -> Dict[str, List[str]]:
    """Split features into confirmed / tentative / rejected lists."""
    confirmed = support_df.loc[support_df["Support"], "Feature"].tolist()
    tentative = support_df.loc[support_df["Support_Weak"], "Feature"].tolist()
    rejected = support_df.loc[
        ~support_df["Support"] & ~support_df["Support_Weak"], "Feature"
    ].tolist()
    return {
        "confirmed": confirmed,
        "tentative": tentative,
        "rejected": rejected,
        "confirmed_tentative": confirmed + tentative,
    }


def build_ranking_dataframe(
    support_df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build ranking.csv.

    Columns: Feature, Decision, Boruta_Rank, MI_Score
    Sorted by: Decision (Confirmed first) then MI_Score descending.
    """

    df = support_df.copy()

    df["Decision"] = df.apply(
        lambda r: "Confirmed" if r["Support"] else (
            "Tentative" if r["Support_Weak"] else "Rejected"
        ),
        axis=1,
    )

    # MI scores on all features for reference
    mi_scores = mutual_info_classif(X, y, random_state=random_state)
    feature_to_idx = {f: i for i, f in enumerate(feature_names)}
    df["MI_Score"] = df["Feature"].map(
        lambda f: float(mi_scores[feature_to_idx[f]])
        if f in feature_to_idx else 0.0
    )

    decision_order = {"Confirmed": 0, "Tentative": 1, "Rejected": 2}
    df["_order"] = df["Decision"].map(decision_order)
    df = (
        df.sort_values(["_order", "MI_Score"], ascending=[True, False], kind="stable")
        .reset_index(drop=True)
        .drop(columns=["_order"])
    )

    df.insert(0, "Rank", np.arange(1, len(df) + 1))

    return df[["Rank", "Feature", "Decision", "Boruta_Rank", "MI_Score"]]


def features_to_indices(
    selected: List[str], all_features: List[str]
) -> np.ndarray:
    feat_map = {f: i for i, f in enumerate(all_features)}
    return np.array([feat_map[f] for f in selected], dtype=np.int64)


# =============================================================================
# Public entry point
# =============================================================================


def run_boruta_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    dataset_name: str,
    output_root: str | Path,
    sample_ids: Optional[List[str]] = None,
    n_estimators: int | str = "auto",
    max_iter: int = 100,
    perc: int = 100,
    alpha: float = 0.05,
    rf_n_estimators: int = 500,
    max_depth: Optional[int] = None,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    selection_mode: Literal["confirmed", "confirmed_tentative", "top_k", "auto"] = "confirmed",
    k: Optional[int] = None,
    dynamic_threshold: int = 50,
) -> Dict[str, Any]:
    """
    Run Boruta feature selection on the full dataset and export artifacts.

    Parameters
    ----------
    X : np.ndarray          (n_samples, n_features)
    y : np.ndarray          (n_samples,) integer labels
    feature_names : list
    dataset_name : str
    output_root : str | Path
    sample_ids : list, optional
    n_estimators : int | 'auto'   Boruta shadow-feature trees per iteration.
    max_iter : int                Maximum Boruta iterations.
    perc : int                    Shadow-feature percentile (100 = strict max).
    alpha : float                 FWER threshold.
    rf_n_estimators : int         Trees in the internal RF estimator (default 500).
    max_depth : int | None        RF tree depth (None = unlimited).
    class_weight : str | None     'balanced' recommended.
    random_state : int
    n_jobs : int
    params : dict, optional       Extra params to merge into params_des.json.
    verbose : bool
    selection_mode : "confirmed" | "confirmed_tentative" | "top_k" | "auto"
        Which feature set the UNIFIED interface (``X_selected``/
        ``selected_features``) reports:
        - "confirmed": only Boruta-confirmed features (default, matches the
          original behaviour of this function).
        - "confirmed_tentative": confirmed + tentative.
        - "top_k": the ``k`` lowest-``Boruta_Rank`` features overall (not the
          Decision-then-MI_Score order ``ranking_df`` uses) — ``k`` must cover
          at least every confirmed+tentative feature and at most all features.
        - "auto": resolved at runtime to "confirmed" if
          ``len(confirmed_features) >= dynamic_threshold``, else
          "confirmed_tentative" (too few confirmed features to train on
          reliably). The resolved mode is recorded alongside the configured
          policy in every output artifact (selected_features.json,
          metadata.json, params_des.json) as ``resolved_selection_mode``.
    k : int, optional
        Required when ``selection_mode == "top_k"``.
    dynamic_threshold : int
        Confirmed-feature-count cutoff used by ``selection_mode == "auto"``.

    Returns
    -------
    dict with keys:
        X_selected (per selection_mode), y_selected, selected_features (per
        selection_mode), selected_indices,
        X_confirmed, confirmed_features, confirmed_indices,
        X_confirmed_tentative, confirmed_tentative_features,
        X_top_k, top_k_features (only when selection_mode == "top_k"),
        ranking_df, support_df, summary, output_dirs, runtime_seconds
    """

    if selection_mode == "top_k":
        if k is None:
            raise ValueError("selection_mode='top_k' requires 'k' to be set.")
        if not (1 <= k <= len(feature_names)):
            raise ValueError(
                f"'k' must be in range [1, {len(feature_names)}] (total features), got {k}."
            )

    start_time = time.time()

    dirs = create_fs_dirs(
        output_root=output_root,
        dataset_name=dataset_name,
        fs_method="boruta",
    )

    # ---- Run Boruta ----
    boruta_obj = run_boruta(
        X=X,
        y=y,
        n_estimators=n_estimators,
        max_iter=max_iter,
        perc=perc,
        alpha=alpha,
        rf_n_estimators=rf_n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=2 if verbose else 0,
    )

    # ---- Support dataframe ----
    support_df = build_support_dataframe(feature_names, boruta_obj)
    support_df.to_csv(dirs["selected_features"] / "support.csv", index=False)

    # ---- Feature groups ----
    groups = extract_feature_groups(support_df)
    confirmed_features = groups["confirmed"]
    tentative_features = groups["tentative"]
    confirmed_tentative_features = groups["confirmed_tentative"]

    # ---- Resolve "auto" mode now that confirmed-count is known ----
    effective_mode = selection_mode
    if selection_mode == "auto":
        effective_mode = (
            "confirmed" if len(confirmed_features) >= dynamic_threshold
            else "confirmed_tentative"
        )
        if verbose:
            from src.helper import report as _report

            _report.info(
                f"selection_mode='auto' -> '{effective_mode}' "
                f"({len(confirmed_features)} confirmed vs threshold {dynamic_threshold})"
            )

    # ---- Ranking ----
    ranking_df = build_ranking_dataframe(
        support_df=support_df,
        X=X,
        y=y,
        feature_names=feature_names,
        random_state=random_state,
    )
    save_ranking(
        ranking_df=ranking_df,
        save_dir=dirs["selected_features"],
        filename="ranking.csv",
    )

    # ---- Individual gene-group CSVs ----
    for group_name, group_list in [
        ("confirmed", confirmed_features),
        ("tentative", tentative_features),
        ("rejected", groups["rejected"]),
    ]:
        pd.DataFrame({"Feature": group_list}).to_csv(
            dirs["selected_features"] / f"{group_name}.csv", index=False
        )

    # ---- Indices ----
    confirmed_indices = features_to_indices(confirmed_features, feature_names)
    confirmed_tentative_indices = features_to_indices(
        confirmed_tentative_features, feature_names
    )

    X_confirmed = X[:, confirmed_indices]
    X_confirmed_tentative = X[:, confirmed_tentative_indices]

    # ---- Export confirmed dataset ----
    save_processed_dataset(
        X=X_confirmed,
        y=y,
        feature_names=confirmed_features,
        save_path=dirs["processed_datasets"] / f"{dataset_name}_boruta_confirmed.csv",
        sample_ids=sample_ids,
    )

    # ---- Export confirmed+tentative dataset ----
    if confirmed_tentative_features:
        save_processed_dataset(
            X=X_confirmed_tentative,
            y=y,
            feature_names=confirmed_tentative_features,
            save_path=dirs["processed_datasets"]
            / f"{dataset_name}_boruta_confirmed_tentative.csv",
            sample_ids=sample_ids,
        )

    # ---- Top-K (by raw Boruta_Rank, NOT ranking_df's Decision-then-MI_Score order) ----
    top_k_features: List[str] = []
    if effective_mode == "top_k":
        if k < len(confirmed_tentative_features):
            raise ValueError(
                f"'k'={k} is smaller than confirmed+tentative ({len(confirmed_tentative_features)}); "
                f"'k' must be in range [{len(confirmed_tentative_features)}, {len(feature_names)}]."
            )
        by_rank = support_df.sort_values("Boruta_Rank", ascending=True, kind="stable")
        top_k_features = by_rank["Feature"].head(k).tolist()

    if effective_mode == "confirmed_tentative":
        X_selected, selected_features = X_confirmed_tentative, confirmed_tentative_features
    elif effective_mode == "top_k":
        top_k_indices = features_to_indices(top_k_features, feature_names)
        X_selected, selected_features = X[:, top_k_indices], top_k_features
    else:
        X_selected, selected_features = X_confirmed, confirmed_features

    # ---- Selected features JSON — "selected_features" is the mode-aware set
    # actually used downstream (training/rule extraction); "confirmed"/
    # "tentative" stay for backward-compat callers that read those directly.
    save_json(
        {
            "confirmed": confirmed_features,
            "tentative": tentative_features,
            "selected_features": selected_features,
            "selection_mode": selection_mode,
            "resolved_selection_mode": effective_mode,
        },
        dirs["selected_features"] / "selected_features.json",
    )

    runtime = time.time() - start_time

    # ---- Summary & params ----
    summary = build_fs_summary(
        dataset_name=dataset_name,
        fs_method="boruta",
        n_samples=X.shape[0],
        n_original_features=X.shape[1],
        n_selected_features=len(selected_features),
        runtime_seconds=runtime,
        extra={
            "confirmed": len(confirmed_features),
            "tentative": len(tentative_features),
            "rejected": len(groups["rejected"]),
            "confirmed_tentative": len(confirmed_tentative_features),
            "max_iter": max_iter,
            "alpha": alpha,
            "selection_mode": selection_mode,
            "resolved_selection_mode": effective_mode,
            "dynamic_threshold": dynamic_threshold,
            "k": k,
        },
    )
    save_metadata(metadata=summary, save_dir=dirs["params"])

    fs_params = {
        "dataset_name": dataset_name,
        "feature_selection": "boruta",
        "n_estimators": n_estimators,
        "rf_n_estimators": rf_n_estimators,
        "max_depth": max_depth,
        "max_iter": max_iter,
        "perc": perc,
        "alpha": alpha,
        "class_weight": class_weight,
        "random_state": random_state,
        "selection_mode": selection_mode,
        "resolved_selection_mode": effective_mode,
        "dynamic_threshold": dynamic_threshold,
        "k": k,
    }
    if params:
        fs_params.update(params)
    save_params(params=fs_params, save_dir=dirs["params"])

    if verbose:
        from src.helper import report

        print_fs_summary(summary)
        report.info(
            f"confirmed: {len(confirmed_features)} · "
            f"tentative: {len(tentative_features)} · "
            f"rejected: {len(groups['rejected'])}"
        )
        if confirmed_features:
            report.info("top confirmed: " + ", ".join(confirmed_features[:10]))

    return {
        # Unified interface — pipeline uses X_selected / selected_features,
        # shaped by selection_mode (see docstring)
        "X_selected": X_selected,
        "y_selected": y.copy(),
        "selected_features": selected_features,
        "selected_indices": features_to_indices(selected_features, feature_names),
        "resolved_selection_mode": effective_mode,
        # Confirmed
        "X_confirmed": X_confirmed,
        "confirmed_features": confirmed_features,
        "confirmed_indices": confirmed_indices,
        # Confirmed + Tentative
        "X_confirmed_tentative": X_confirmed_tentative,
        "confirmed_tentative_features": confirmed_tentative_features,
        # Top-K (only populated when selection_mode == "top_k")
        "top_k_features": top_k_features,
        # Artifacts
        "ranking_df": ranking_df,
        "support_df": support_df,
        "summary": summary,
        "output_dirs": dirs,
        "runtime_seconds": runtime,
    }
