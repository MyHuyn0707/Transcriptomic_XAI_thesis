"""
src/interpretation/shap_utils.py

SHAP explainability for all model types.

Supported
---------
- Tree models (RF, XGBoost, DecisionTree)  → shap.TreeExplainer (exact, fast)
- Everything else (NB, KNN, SVM, ANN)      → shap.KernelExplainer with
  background subsampling (model-agnostic, approximate, slow on wide inputs)

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# =============================================================================
# Model-type detection
# =============================================================================

_TREE_TYPES = (
    "RandomForestClassifier",
    "XGBClassifier",
    "GradientBoostingClassifier",
    "ExtraTreesClassifier",
    "DecisionTreeClassifier",
)

_LINEAR_TYPES = (
    "LinearSVC",
    "LogisticRegression",
    "SGDClassifier",
    "RidgeClassifier",
)


def _model_type(model: Any) -> str:
    """Return 'tree', 'linear', or 'kernel'."""
    class_name = type(model).__name__

    if class_name in _TREE_TYPES:
        return "tree"

    if class_name in _LINEAR_TYPES:
        return "linear"

    return "kernel"


# =============================================================================
# SHAP computation
# =============================================================================


def _shap_tree(
    model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
) -> np.ndarray:
    explainer = shap.TreeExplainer(
        model, X_background, feature_perturbation="interventional"
    )
    sv = explainer.shap_values(X_explain)
    # Multi-class: list of arrays → stack into (n_samples, n_features, n_classes)
    if isinstance(sv, list):
        return np.stack(sv, axis=-1)
    return sv


def _shap_kernel(
    model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    max_background: int = 100,
) -> np.ndarray:
    """KernelExplainer with background subsampling for speed."""

    if X_background.shape[0] > max_background:
        idx = np.random.default_rng(42).choice(
            X_background.shape[0], size=max_background, replace=False
        )
        bg = shap.kmeans(X_background[idx], min(10, max_background))
    else:
        bg = shap.kmeans(X_background, min(10, X_background.shape[0]))

    predict_fn = (
        model.predict_proba
        if hasattr(model, "predict_proba")
        else model.predict
    )

    explainer = shap.KernelExplainer(predict_fn, bg)
    sv = explainer.shap_values(X_explain, silent=True)

    if isinstance(sv, list):
        return np.stack(sv, axis=-1)
    return sv


# =============================================================================
# Global importance helper
# =============================================================================


def _mean_abs_shap(shap_values: np.ndarray) -> np.ndarray:
    """
    Compute per-feature mean absolute SHAP value.

    Handles both 2-D (binary/single output) and
    3-D (n_samples, n_features, n_classes) arrays.
    """
    if shap_values.ndim == 3:
        # Sum absolute values across classes, then mean across samples
        return np.mean(np.sum(np.abs(shap_values), axis=2), axis=0)
    return np.mean(np.abs(shap_values), axis=0)


# =============================================================================
# Post-SHAP probe → gene collapse (interpretation only — model stays probe-level)
# =============================================================================


def _collapse_to_gene(
    shap_values: np.ndarray,
    X: np.ndarray,
    probe_names: List[str],
    mapping: dict,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Collapse probe-level SHAP results to gene level for interpretation display.

    SHAP is computed on the probe-level features the model was trained on; this
    runs AFTER, purely to relabel/aggregate for plots, the importance matrix, and
    gene text lists. Probes mapping to the same gene symbol are merged:
      - SHAP values  → summed across co-mapped probes (SHAP is additive).
      - feature values → averaged across co-mapped probes.
    Probes with no gene symbol keep their probe ID (status 'Unmapped').

    Returns
    -------
    gene_shap     : np.ndarray, same ndim as shap_values, gene-width.
    gene_X        : np.ndarray (n_samples, n_genes).
    gene_names    : List[str], gene symbols in first-appearance order.
    provenance_df : pd.DataFrame [Probe_ID, Gene_Symbol, Status]
                    Status ∈ {Kept, Collapsed_Mean, Unmapped}.
    """
    is_mapped = {p: bool(str(mapping.get(p, "")).strip()) for p in probe_names}
    probe_to_gene = {p: (str(mapping.get(p, "")).strip() or p) for p in probe_names}

    gene_order: List[str] = []
    gene_to_cols: dict = {}
    for j, p in enumerate(probe_names):
        g = probe_to_gene[p]
        if g not in gene_to_cols:
            gene_to_cols[g] = []
            gene_order.append(g)
        gene_to_cols[g].append(j)

    n_samples = X.shape[0]
    n_genes = len(gene_order)
    gene_X = np.empty((n_samples, n_genes), dtype=float)

    if shap_values.ndim == 3:
        gene_shap = np.empty((n_samples, n_genes, shap_values.shape[2]), dtype=float)
    else:
        gene_shap = np.empty((n_samples, n_genes), dtype=float)

    prov_rows: List[dict] = []
    for gi, gene in enumerate(gene_order):
        cols = gene_to_cols[gene]
        gene_X[:, gi] = X[:, cols].mean(axis=1)
        if shap_values.ndim == 3:
            gene_shap[:, gi, :] = shap_values[:, cols, :].sum(axis=1)
        else:
            gene_shap[:, gi] = shap_values[:, cols].sum(axis=1)

        if len(cols) == 1:
            probe = probe_names[cols[0]]
            prov_rows.append({
                "Probe_ID": probe,
                "Gene_Symbol": gene,
                "Status": "Kept" if is_mapped[probe] else "Unmapped",
            })
        else:
            for j in cols:
                prov_rows.append({
                    "Probe_ID": probe_names[j],
                    "Gene_Symbol": gene,
                    "Status": "Collapsed_Mean",
                })

    provenance_df = pd.DataFrame(
        prov_rows, columns=["Probe_ID", "Gene_Symbol", "Status"]
    )
    return gene_shap, gene_X, gene_order, provenance_df


# =============================================================================
# Visualisation helpers
# =============================================================================


def _plot_shap_for_class(
    shap_vals: np.ndarray,
    X_explain: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    class_idx: Optional[int] = None,
    top_k: int = 20,
    class_label: Optional[str] = None,
) -> None:
    """Save bar and beeswarm plots for one class (or binary case).

    ``class_idx`` keeps the filename stable (``_class{idx}``, filesystem-safe);
    ``class_label`` (the real class name, when known) is used only in the plot
    title — falling back to ``Class {idx}`` when no name is available.
    """

    suffix = f"_class{class_idx}" if class_idx is not None else ""
    # Human-readable tag for the title only (names may contain spaces/commas,
    # so they never touch the filename).
    if class_idx is None:
        tag = ""
    else:
        tag = f" ({class_label})" if class_label else f" (Class {class_idx})"

    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_vals, X_explain,
        feature_names=feature_names,
        plot_type="bar",
        max_display=top_k,
        show=False,
    )
    plt.title(f"SHAP Global Feature Importance{tag}")
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_bar{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_vals, X_explain,
        feature_names=feature_names,
        max_display=top_k,
        show=False,
    )
    plt.title(f"SHAP Beeswarm{tag}")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"shap_beeswarm{suffix}.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


# =============================================================================
# Public entry point
# =============================================================================


def explain_model(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    output_dir: str | Path,
    top_k: int = 20,
    kernel_max_background: int = 100,
    class_names: Optional[List[str]] = None,
    probe_gene_map: Optional[dict] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute SHAP values, save explanation plots, and export numeric summaries.

    Automatically selects the appropriate SHAP explainer based on model type:
    - Tree models  → TreeExplainer (exact)
    - Others       → KernelExplainer (approximate, slower)

    Parameters
    ----------
    model : Any
        Trained model (RF, XGBoost, SVM, NB, ANN wrapper, etc.)
    X_train : np.ndarray
        Background / training data for SHAP.
    X_test : np.ndarray
        Samples to explain.
    feature_names : List[str]
    output_dir : str | Path
    top_k : int
        Number of top features to extract.
    kernel_max_background : int
        Maximum background samples for KernelExplainer (controls speed vs accuracy).
    class_names : List[str], optional
        Human-readable class labels used to name columns in the exported matrix
        and section headers in gene-list files.
    probe_gene_map : dict, optional
        {probe_id -> gene_symbol}. When provided, SHAP is still computed on the
        probe-level inputs, then results are collapsed to gene level (post-SHAP)
        so plot axes, the importance matrix, and the gene text lists use Gene
        Symbols. A probe_to_gene_metadata.csv provenance file is written too.

    Returns
    -------
    shap_values : np.ndarray
        Probe-level SHAP values. Shape (n_samples, n_features) for binary,
        or (n_samples, n_features, n_classes) for multi-class.
    top_features : List[str]
        Top-K names sorted by mean absolute SHAP — Gene Symbols when
        ``probe_gene_map`` is supplied, otherwise probe IDs.
    """

    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    text_dir = output_dir / "text"
    plots_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    mtype = _model_type(model)

    print(f"  SHAP explainer: {mtype} (model={type(model).__name__})")

    if mtype == "tree":
        shap_values = _shap_tree(model, X_train, X_test)
    else:
        warnings.warn(
            f"Using KernelExplainer for {type(model).__name__}. "
            "This is approximate and may be slow for large feature sets.",
            stacklevel=2,
        )
        shap_values = _shap_kernel(
            model, X_train, X_test, max_background=kernel_max_background
        )

    # ------------------------------------------------------------------ #
    # Post-SHAP probe → gene collapse (display only; model stays probe).  #
    # ------------------------------------------------------------------ #
    if probe_gene_map:
        disp_shap, disp_X, disp_names, provenance_df = _collapse_to_gene(
            shap_values, X_test, list(feature_names), probe_gene_map
        )
        prov_path = text_dir / "probe_to_gene_metadata.csv"
        provenance_df.to_csv(prov_path, index=False)
        n_unmapped = int((provenance_df["Status"] == "Unmapped").sum())
        print(
            f"  Probe->gene collapse: {len(feature_names)} probes -> "
            f"{len(disp_names)} genes ({n_unmapped} unmapped)"
        )
        print(f"  Provenance -> {prov_path}")
    else:
        disp_shap, disp_X, disp_names = shap_values, X_test, list(feature_names)

    # Global importance (gene-level when mapped)
    mean_abs = _mean_abs_shap(disp_shap)
    top_indices = np.argsort(mean_abs)[::-1][:top_k]
    top_features = [disp_names[i] for i in top_indices]

    # Real class name for SHAP class-slice ``c`` — used in plot titles, CSV
    # column headers, and gene-list headers. Only applied when the number of
    # class-slices matches the provided names, so the binary single-output case
    # and the unnamed case fall back to the original ``Class_{c}`` scheme.
    n_slices = disp_shap.shape[2] if disp_shap.ndim == 3 else 1

    def _cls_name(c: int) -> Optional[str]:
        if class_names and len(class_names) == n_slices and 0 <= c < len(class_names):
            return str(class_names[c])
        return None

    def _cls_col(c: int) -> str:
        name = _cls_name(c)
        return f"{name}_Mean_Abs" if name else f"Class_{c}_Mean_Abs"

    # ------------------------------------------------------------------ #
    # Plots → plots/                                                       #
    # ------------------------------------------------------------------ #
    if disp_shap.ndim == 3:
        n_classes = disp_shap.shape[2]
        for cls in range(n_classes):
            _plot_shap_for_class(
                shap_vals=disp_shap[:, :, cls],
                X_explain=disp_X,
                feature_names=disp_names,
                output_dir=plots_dir,
                class_idx=cls,
                top_k=top_k,
                class_label=_cls_name(cls),
            )
        agg_abs = np.sum(np.abs(disp_shap), axis=2)
        _plot_shap_for_class(
            shap_vals=agg_abs,
            X_explain=disp_X,
            feature_names=disp_names,
            output_dir=plots_dir,
            class_idx=None,
            top_k=top_k,
        )
    else:
        _plot_shap_for_class(
            shap_vals=disp_shap,
            X_explain=disp_X,
            feature_names=disp_names,
            output_dir=plots_dir,
            class_idx=None,
            top_k=top_k,
        )

    print(f"  SHAP plots saved -> {plots_dir}")

    # ------------------------------------------------------------------ #
    # Export numeric feature-importance matrix → text/                     #
    # ------------------------------------------------------------------ #
    if disp_shap.ndim == 3:
        n_cls = disp_shap.shape[2]
        class_abs = [
            np.mean(np.abs(disp_shap[:, :, c]), axis=0) for c in range(n_cls)
        ]
    else:
        n_cls = 1
        class_abs = [np.mean(np.abs(disp_shap), axis=0)]

    records = []
    for fi, fname in enumerate(disp_names):
        row: dict = {
            "Feature_Gene": fname,
            "Overall_Mean_Abs": round(float(mean_abs[fi]), 6),
        }
        for c in range(n_cls):
            row[_cls_col(c)] = round(float(class_abs[c][fi]), 6)
        records.append(row)

    importance_df = (
        pd.DataFrame(records)
        .sort_values("Overall_Mean_Abs", ascending=False)
        .reset_index(drop=True)
    )
    matrix_path = text_dir / "shap_feature_importance_matrix.csv"
    importance_df.to_csv(matrix_path, index=False)
    print(f"  SHAP matrix -> {matrix_path}")

    # ------------------------------------------------------------------ #
    # Export per-class gene lists (clean, copy-paste ready) → text/        #
    # ------------------------------------------------------------------ #
    combined_lines: List[str] = []
    for c in range(n_cls):
        lbl = _cls_name(c) or str(c)
        col = _cls_col(c)
        if col not in importance_df.columns:
            continue

        top_genes = (
            importance_df.nlargest(top_k, col)["Feature_Gene"]
            .drop_duplicates()
            .tolist()
        )

        per_class_path = text_dir / f"top_shap_genes_class{c}.txt"
        per_class_path.write_text("\n".join(top_genes) + "\n", encoding="utf-8")

        combined_lines.append(f"# Class {c}: {lbl}")
        combined_lines.extend(top_genes)
        combined_lines.append("")  # blank line between classes

    combined_path = text_dir / "top_shap_genes_by_class.txt"
    combined_path.write_text("\n".join(combined_lines), encoding="utf-8")
    print(f"  Gene lists -> {combined_path}")

    return shap_values, top_features


# =============================================================================
# Backward-compat alias
# =============================================================================

def explain_tree_model(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    output_dir: str | Path,
    top_k: int = 20,
) -> Tuple[np.ndarray, List[str]]:
    """Backward-compatible alias for explain_model()."""
    return explain_model(
        model=model,
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
        output_dir=output_dir,
        top_k=top_k,
    )
