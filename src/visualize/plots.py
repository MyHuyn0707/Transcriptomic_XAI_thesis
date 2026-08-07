"""
src/visualize/plots.py

Dataset visualization utilities for gene expression data (CuMiDa / GEO format).

All public functions accept the dict returned by src.data_loader.load_dataset()
and return matplotlib Figure objects, making them composable in notebooks.

Plot catalogue
--------------
plot_class_distribution  -- sample counts per class (bar chart)
plot_pca                 -- PCA 2-D scatter coloured by class
plot_tsne                -- t-SNE 2-D scatter coloured by class
plot_expression_heatmap  -- top-N variable probes × samples heatmap (z-score)
plot_dataset_overview    -- 2×2 grid: distribution + PCA + t-SNE + heatmap
plot_all_datasets        -- iterate over a list of dataset paths, save PNGs
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.helper.data_loader import load_dataset


# ---------------------------------------------------------------------------
# Palette helpers
# ---------------------------------------------------------------------------

# Up to 20 distinct colours; falls back to a continuous map for more classes.
_PALETTE_20 = plt.cm.tab20.colors  # type: ignore[attr-defined]


def _class_colors(n_classes: int) -> list:
    if n_classes <= 10:
        return list(plt.cm.tab10.colors[:n_classes])  # type: ignore[attr-defined]
    if n_classes <= 20:
        return list(_PALETTE_20[:n_classes])
    cmap = plt.cm.get_cmap("hsv", n_classes)
    return [cmap(i) for i in range(n_classes)]


def _legend_patches(class_names: list[str], colors: list) -> list[mpatches.Patch]:
    return [
        mpatches.Patch(color=c, label=lbl)
        for c, lbl in zip(colors, class_names)
    ]


def _class_names(data: dict) -> list[str]:
    """Return ordered list of original class name strings."""
    enc = data.get("label_encoder")
    if enc is not None:
        return [str(c) for c in enc.classes_]
    # No encoder: labels were already integers; use label_mapping
    mapping = data["metadata"].get("label_mapping", {})
    n = data["metadata"]["n_classes"]
    return [str(mapping.get(i, i)) for i in range(n)]


def class_names(data: dict) -> list[str]:
    """Public accessor for a dataset's ordered class-name strings."""
    return _class_names(data)


def _class_names_with_ids(data: dict) -> list[str]:
    """Return class names formatted as 'name (id)' — e.g. 'basal (0)'."""
    enc = data.get("label_encoder")
    if enc is not None:
        return [f"{str(c)} ({i})" for i, c in enumerate(enc.classes_)]
    mapping = data["metadata"].get("label_mapping", {})
    n = data["metadata"]["n_classes"]
    inv = {int(v): str(k) for k, v in mapping.items()}
    return [f"{inv.get(i, str(i))} ({i})" for i in range(n)]


def _name_to_id_map(data: dict) -> dict:
    """Return {class_name_str: integer_id} for building ID-annotated labels."""
    enc = data.get("label_encoder")
    if enc is not None:
        return {str(cls): i for i, cls in enumerate(enc.classes_)}
    mapping = data["metadata"].get("label_mapping", {})
    return {str(k): int(v) for k, v in mapping.items()}


# ---------------------------------------------------------------------------
# 1. Class distribution
# ---------------------------------------------------------------------------


def plot_class_distribution(
    data: dict,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (6, 4),
    show_class_ids: bool = False,
) -> plt.Figure:
    """
    Horizontal bar chart of sample counts per class.

    Reads raw class counts from data["dataframe"] so that classes removed
    by the rare-sample filter are still shown — rendered in light grey with
    a '(Deleted)' suffix so the audit trail is always visible.

    Parameters
    ----------
    data    : dict returned by load_dataset()
    title   : plot title (defaults to "Class distribution")
    ax      : existing Axes to draw on; creates a new figure if None
    figsize : figure size when ax is None
    """
    # --- Raw counts from the original (unfiltered) DataFrame ---
    raw_df = data.get("dataframe")
    if raw_df is not None and len(raw_df.columns) >= 2:
        raw_counts: pd.Series = raw_df.iloc[:, 1].astype(str).value_counts()
    else:
        # Graceful fallback: reconstruct from filtered y
        names_fb = _class_names(data)
        y_fb = data["y"]
        raw_counts = pd.Series(
            {names_fb[i]: int(np.sum(y_fb == i)) for i in range(len(names_fb))}
        )

    # Surviving classes (those NOT removed by rare-sample filter)
    encoder = data.get("label_encoder")
    surviving_classes: list = (
        list(encoder.classes_) if encoder is not None else _class_names(data)
    )
    n_surviving = len(surviving_classes)
    palette = _class_colors(max(n_surviving, 1))
    class_color_map: dict = {cls: palette[i] for i, cls in enumerate(surviving_classes)}

    name_to_id = _name_to_id_map(data) if show_class_ids else {}

    # Build bar data (count, display label, colour)
    bar_data: list = []
    for cls_name in raw_counts.index:
        count = int(raw_counts[cls_name])
        is_deleted = cls_name not in class_color_map
        if is_deleted:
            display_name = f"{cls_name} (Deleted)"
        elif show_class_ids:
            display_name = f"{cls_name} ({name_to_id.get(cls_name, '?')})"
        else:
            display_name = cls_name
        color = "#D3D3D3" if is_deleted else class_color_map[cls_name]
        bar_data.append((count, display_name, color))

    bar_data.sort(key=lambda x: x[0], reverse=True)
    counts_s, names_s, colors_s = zip(*bar_data)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    bars = ax.barh(
        range(len(names_s)), counts_s, color=colors_s,
        edgecolor="white", linewidth=0.5,
    )
    ax.set_yticks(range(len(names_s)))
    ax.set_yticklabels(names_s, fontsize=9)
    ax.set_xlabel("Sample count")
    ax.set_title(title or "Class distribution", fontsize=11, fontweight="bold")
    ax.invert_yaxis()

    for bar, cnt in zip(bars, counts_s):
        ax.text(
            bar.get_width() + max(counts_s) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(cnt), va="center", ha="left", fontsize=8,
        )

    ax.spines[["top", "right"]].set_visible(False)
    if standalone:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 1.b Train / test split distribution (holdout flow)
# ---------------------------------------------------------------------------


def plot_train_test_split(
    data: dict,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (7, 4),
    show_class_ids: bool = False,
) -> plt.Figure:
    """
    Grouped bar chart: train vs. test sample counts per class.

    Used by the holdout rule-extraction flow (pipeline.run_rule_extraction_holdout)
    to visualise the stratified split. Operates on `data` AFTER rare-class
    removal (surviving classes only) — dropped classes have no train/test
    samples to show and are already covered by plot_class_distribution().

    Parameters
    ----------
    data      : dict from load_dataset() (already rare-class filtered)
    train_idx, test_idx : row indices into data["y"] for the split
    """
    y = data["y"]
    n_classes = data["metadata"]["n_classes"]
    names = _class_names_with_ids(data) if show_class_ids else _class_names(data)
    colors = _class_colors(n_classes)

    train_counts = np.bincount(y[train_idx], minlength=n_classes)
    test_counts = np.bincount(y[test_idx], minlength=n_classes)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    x = np.arange(n_classes)
    width = 0.35
    bars_train = ax.bar(
        x - width / 2, train_counts, width, label="Train",
        color=colors, edgecolor="white", linewidth=0.5,
    )
    bars_test = ax.bar(
        x + width / 2, test_counts, width, label="Test",
        color=colors, alpha=0.5, hatch="//", edgecolor="white", linewidth=0.5,
    )

    for bar, cnt in zip(list(bars_train) + list(bars_test),
                        list(train_counts) + list(test_counts)):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            str(cnt), ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Sample count")
    ax.set_title(title or "Train / test split per class", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    if standalone:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 1.c Confusion matrix (model evaluation)
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: np.ndarray | list,
    class_labels: Optional[list[str]] = None,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (5, 4.5),
    normalize: bool = False,
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Heatmap of a confusion matrix (rows = true class, columns = predicted class).

    Parameters
    ----------
    cm : array-like (n_classes, n_classes) — e.g. compute_metrics()'s "confusion_matrix".
    class_labels : display names indexed by class integer; falls back to "0", "1", ...
    normalize : row-normalise to fractions of each true class (else raw counts).
    """
    cm = np.asarray(cm, dtype=float)
    n = cm.shape[0]
    labels = list(class_labels) if class_labels else [str(i) for i in range(n)]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = ".0f"

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    vmax = cm_display.max() if cm_display.max() > 0 else 1.0
    im = ax.imshow(cm_display, cmap=cmap, vmin=0, vmax=vmax)

    for i in range(n):
        for j in range(n):
            val = cm_display[i, j]
            ax.text(
                j, i, format(val, fmt), ha="center", va="center", fontsize=9,
                color="white" if val > vmax / 2 else "black",
            )

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(title or "Confusion matrix", fontsize=11, fontweight="bold")

    if standalone:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. PCA scatter
# ---------------------------------------------------------------------------


def plot_pca(
    data: dict,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (6, 5),
    explained_variance: bool = True,
    show_class_ids: bool = False,
) -> plt.Figure:
    """
    2-D PCA scatter plot coloured by class.

    Parameters
    ----------
    explained_variance : annotate axes with PC explained-variance ratios
    show_class_ids     : append integer class ID to each legend label
    """
    X, y = data["X"], data["y"]
    n_classes = data["metadata"]["n_classes"]
    names = _class_names_with_ids(data) if show_class_ids else _class_names(data)
    colors = _class_colors(n_classes)

    pca = PCA(n_components=2, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)
    coords = pca.fit_transform(X_scaled)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for i, (name, color) in enumerate(zip(names, colors)):
        mask = y == i
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color], label=name, s=25, alpha=0.75, edgecolors="none",
        )

    if explained_variance:
        ev = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=9)
    else:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    ax.set_title(title or "PCA", fontsize=11, fontweight="bold")
    ax.legend(
        handles=_legend_patches(names, colors),
        fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left",
        frameon=False,
    )
    ax.spines[["top", "right"]].set_visible(False)

    if standalone:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. t-SNE scatter
# ---------------------------------------------------------------------------


def plot_tsne(
    data: dict,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (6, 5),
    perplexity: Optional[float] = None,
    n_pca_components: int = 50,
    random_state: int = 42,
    show_class_ids: bool = False,
) -> plt.Figure:
    """
    2-D t-SNE scatter plot coloured by class.

    For high-dimensional data, PCA is applied first (n_pca_components)
    to speed up t-SNE without information loss.
    Perplexity defaults to min(30, n_samples/5).
    show_class_ids : append integer class ID to each legend label.
    """
    X, y = data["X"], data["y"]
    n_samples = X.shape[0]
    n_classes = data["metadata"]["n_classes"]
    names = _class_names_with_ids(data) if show_class_ids else _class_names(data)
    colors = _class_colors(n_classes)

    X_scaled = StandardScaler().fit_transform(X)

    # PCA pre-reduction for speed
    n_comp = min(n_pca_components, n_samples - 1, X.shape[1])
    if X.shape[1] > n_comp:
        X_scaled = PCA(n_components=n_comp, random_state=random_state).fit_transform(X_scaled)

    perp = perplexity or max(5.0, min(30.0, n_samples / 5.0))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state, max_iter=1000)
    coords = tsne.fit_transform(X_scaled)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for i, (name, color) in enumerate(zip(names, colors)):
        mask = y == i
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color], label=name, s=25, alpha=0.75, edgecolors="none",
        )

    ax.set_xlabel("t-SNE 1", fontsize=9)
    ax.set_ylabel("t-SNE 2", fontsize=9)
    ax.set_title(title or "t-SNE", fontsize=11, fontweight="bold")
    ax.legend(
        handles=_legend_patches(names, colors),
        fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left",
        frameon=False,
    )
    ax.spines[["top", "right"]].set_visible(False)

    if standalone:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Expression heatmap
# ---------------------------------------------------------------------------


def plot_expression_heatmap(
    data: dict,
    title: str = "",
    ax: Optional[plt.Axes] = None,
    figsize: tuple = (10, 5),
    top_n: int = 50,
    show_class_ids: bool = False,
) -> plt.Figure:
    """
    Heatmap of the top-N most variable probes × samples (z-score normalised).

    Samples are sorted by class label so clusters are visually grouped.
    A colour bar on top encodes the class of each sample.
    show_class_ids : append integer class ID to each legend label.
    """
    X, y = data["X"], data["y"]
    n_classes = data["metadata"]["n_classes"]
    names = _class_names_with_ids(data) if show_class_ids else _class_names(data)
    colors = _class_colors(n_classes)
    feature_names = data["feature_names"]

    # Select top-N by variance
    variances = X.var(axis=0)
    top_idx = np.argsort(variances)[::-1][:top_n]
    X_top = X[:, top_idx]
    feat_labels = [feature_names[i] for i in top_idx]

    # Sort samples by class
    order = np.argsort(y)
    X_sorted = X_top[order]
    y_sorted = y[order]

    # Z-score per feature (probe)
    X_z = (X_sorted - X_sorted.mean(axis=0)) / (X_sorted.std(axis=0) + 1e-8)
    X_z = np.clip(X_z, -3, 3)

    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=figsize)
        # Two rows: thin class-colour bar on top, heatmap below
        gs = fig.add_gridspec(2, 2, height_ratios=[0.04, 1], width_ratios=[1, 0.02],
                               hspace=0.02, wspace=0.03)
        ax_bar  = fig.add_subplot(gs[0, 0])
        ax_heat = fig.add_subplot(gs[1, 0])
        ax_cbar = fig.add_subplot(gs[1, 1])
    else:
        fig = ax.get_figure()
        ax_heat = ax
        ax_bar = None
        ax_cbar = None

    # Class colour bar
    if ax_bar is not None:
        bar_data = np.array([[colors[c] for c in y_sorted]])  # (1, n_samples, 4)
        # Convert list-of-tuples to RGBA array
        rgba = np.zeros((1, len(y_sorted), 4))
        for j, c in enumerate(y_sorted):
            rgba[0, j, :] = colors[c][:4] if len(colors[c]) == 4 else (*colors[c], 1.0)
        ax_bar.imshow(rgba, aspect="auto")
        ax_bar.set_axis_off()

        # Legend patches for class colours
        patches = _legend_patches(names, colors)
        ax_bar.legend(
            handles=patches, ncol=min(n_classes, 6),
            loc="lower left", bbox_to_anchor=(0, 1.1),
            fontsize=7, frameon=False, handlelength=1,
        )

    # Heatmap
    im = ax_heat.imshow(
        X_z.T, aspect="auto", cmap="RdBu_r",
        vmin=-3, vmax=3, interpolation="nearest",
    )
    ax_heat.set_xlabel("Samples (sorted by class)", fontsize=9)
    ax_heat.set_ylabel(f"Top {top_n} variable probes", fontsize=9)
    ax_heat.set_yticks([])
    ax_heat.set_xticks([])
    ax_heat.set_title(title or f"Expression heatmap (top {top_n} variable probes, z-score)",
                      fontsize=11, fontweight="bold")

    # Colour bar
    if ax_cbar is not None:
        plt.colorbar(im, cax=ax_cbar, label="z-score")

    if standalone:
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Dataset overview (2 × 2 grid)
# ---------------------------------------------------------------------------


def plot_dataset_overview(
    data: dict,
    dataset_name: str = "",
    figsize: tuple = (16, 12),
    top_n_heatmap: int = 50,
    show_class_ids: bool = False,
) -> plt.Figure:
    """
    Single figure with four panels:
      [0,0] Class distribution   [0,1] PCA
      [1,0] t-SNE                [1,1] Expression heatmap

    Parameters
    ----------
    data           : dict from load_dataset()
    dataset_name   : used as the figure suptitle
    top_n_heatmap  : number of most-variable probes shown in heatmap
    show_class_ids : when True, append the integer class ID to every label
                     (e.g. "basal (0)") across all four panels.
    """
    n_samples  = data["metadata"]["n_samples"]
    n_features = data["metadata"]["n_features"]
    n_classes  = data["metadata"]["n_classes"]
    info = f"{n_samples} samples · {n_features} features · {n_classes} classes"

    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"{dataset_name}\n{info}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.55,
                          left=0.08, right=0.92, top=0.90, bottom=0.08)
    ax_dist  = fig.add_subplot(gs[0, 0])
    ax_pca   = fig.add_subplot(gs[0, 1])
    ax_tsne  = fig.add_subplot(gs[1, 0])
    ax_heat  = fig.add_subplot(gs[1, 1])

    plot_class_distribution(data, title="Class distribution", ax=ax_dist,
                            show_class_ids=show_class_ids)
    plot_pca(data, title="PCA", ax=ax_pca, show_class_ids=show_class_ids)
    plot_tsne(data, title="t-SNE", ax=ax_tsne, show_class_ids=show_class_ids)

    # Heatmap in the bottom-right panel (simplified: no inner gridspec)
    X, y = data["X"], data["y"]
    n_cls  = n_classes
    names  = _class_names_with_ids(data) if show_class_ids else _class_names(data)
    colors = _class_colors(n_cls)

    variances = X.var(axis=0)
    top_idx   = np.argsort(variances)[::-1][:top_n_heatmap]
    order     = np.argsort(y)
    X_z       = X[order][:, top_idx]
    X_z       = (X_z - X_z.mean(axis=0)) / (X_z.std(axis=0) + 1e-8)
    X_z       = np.clip(X_z, -3, 3)

    im = ax_heat.imshow(X_z.T, aspect="auto", cmap="RdBu_r",
                        vmin=-3, vmax=3, interpolation="nearest")
    ax_heat.set_xlabel("Samples (sorted by class)", fontsize=9)
    ax_heat.set_ylabel(f"Top {top_n_heatmap} variable probes", fontsize=9)
    ax_heat.set_yticks([])
    ax_heat.set_xticks([])
    ax_heat.set_title(f"Expression heatmap (z-score)", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.04, label="z-score")

    # Class boundary lines on heatmap
    y_sorted = y[order]
    boundaries = np.where(np.diff(y_sorted))[0] + 1
    for b in boundaries:
        ax_heat.axvline(b - 0.5, color="black", linewidth=0.8, alpha=0.6)

    return fig


# ---------------------------------------------------------------------------
# 6. Batch visualisation
# ---------------------------------------------------------------------------


def plot_all_datasets(
    dataset_paths: list[str | Path],
    output_dir: Optional[str | Path] = None,
    figsize: tuple = (16, 12),
    top_n_heatmap: int = 50,
    dpi: int = 150,
    show: bool = True,
) -> list[plt.Figure]:
    """
    Load and visualise every dataset in dataset_paths.

    Parameters
    ----------
    dataset_paths : list of CSV paths (CuMiDa format)
    output_dir    : if given, save each figure as <dataset_name>.png
    figsize       : passed to plot_dataset_overview
    top_n_heatmap : top variable probes shown in heatmap
    dpi           : PNG resolution when saving
    show          : call plt.show() after each figure

    Returns
    -------
    List of matplotlib Figure objects (one per dataset).
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    figures: list[plt.Figure] = []

    for path in dataset_paths:
        path = Path(path)
        print(f"Loading {path.name} ...")
        data = load_dataset(path, dataset_type="cumida")

        dataset_name = path.stem   # e.g. 'Breast_GEOD20711'
        fig = plot_dataset_overview(
            data,
            dataset_name=dataset_name,
            figsize=figsize,
            top_n_heatmap=top_n_heatmap,
        )

        if output_dir is not None:
            out_path = output_dir / f"{dataset_name}.png"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            print(f"  Saved -> {out_path}")

        if show:
            plt.show()

        figures.append(fig)
        plt.close(fig)

    return figures


def plot_all_class_distributions(
    datas: dict[str, dict],
    ncols: int = 4,
    cell_size: tuple = (5, 3.5),
    show_class_ids: bool = False,
    suptitle: str = "Class distribution — all datasets",
) -> plt.Figure:
    """
    Grid of :func:`plot_class_distribution` panels, one per dataset — the
    batch overview used to eyeball rare-class drops across many datasets at once.

    Parameters
    ----------
    datas : dict of ``{title: data}`` — ``data`` is a dict from load_dataset(),
            ``title`` labels that dataset's panel (e.g. its dataset name).
    ncols : columns in the grid (rows are added as needed).
    cell_size : (width, height) per panel, in inches.
    show_class_ids : append the integer class ID to every label.
    suptitle : figure-level title.
    """
    n = len(datas)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * cell_size[0], nrows * cell_size[1]))
    axes_flat = np.atleast_1d(axes).flatten()

    i = -1
    for i, (title, data) in enumerate(datas.items()):
        plot_class_distribution(data, title=title, ax=axes_flat[i], show_class_ids=show_class_ids)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_all_datasets_metric_comparison(
    results_by_dataset: dict,
    metric: str = "f1_macro",
    ncols: int = 3,
    cell_size: tuple = (6, 4),
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """
    Grid of grouped-bar panels (one per dataset) comparing every
    fs_method × model combination on ``metric`` — the collective
    cross-dataset comparison view for a single rare-class-k sweep value
    (see src/pipeline/baseline_split.py::run_baseline_split).

    Parameters
    ----------
    results_by_dataset : ``{dataset_name: {fs_method: {model_name: metrics_dict}}}``
        ``metrics_dict`` must contain ``metric`` as a key (e.g. a test-set
        metrics dict from compute_metrics()).
    metric : which scalar metric to plot (e.g. "f1_macro", "balanced_accuracy").
    ncols : columns in the grid (rows are added as needed).
    cell_size : (width, height) per panel, in inches.
    suptitle : figure-level title.
    """
    datasets = list(results_by_dataset.keys())
    n = len(datasets)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * cell_size[0], nrows * cell_size[1]), squeeze=False
    )
    axes_flat = axes.flatten()

    i = -1
    for i, ds_name in enumerate(datasets):
        ax = axes_flat[i]
        fs_results = results_by_dataset[ds_name]
        fs_methods = sorted(fs_results.keys())
        models = sorted({m for fs in fs_results.values() for m in fs.keys()})
        if not models:
            ax.set_visible(False)
            continue

        x = np.arange(len(models))
        width = 0.8 / max(len(fs_methods), 1)
        for j, fs in enumerate(fs_methods):
            vals = [fs_results[fs].get(m, {}).get(metric, np.nan) for m in models]
            ax.bar(x + j * width - 0.4 + width / 2, vals, width, label=fs)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_title(ds_name, fontsize=10)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        suptitle or f"{metric} — fs_method × model, all datasets",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Metadata provenance logger
# ---------------------------------------------------------------------------


def update_dataset_description(
    data: dict,
    description_path: str | Path,
    min_samples: Optional[int] = None,
    train_idx: Optional[np.ndarray] = None,
    test_idx: Optional[np.ndarray] = None,
) -> None:
    """
    Append a structured Markdown provenance block to a platform description file.

    The block records the exact class distribution — including classes that were
    removed by the rare-sample filter — so every pipeline run is fully auditable.
    Idempotent: skips silently if the sentinel line already exists in the file.

    Parameters
    ----------
    data             : dict returned by load_dataset()
    description_path : Path to the target file (created if absent).
                       Typically a README or a per-dataset description file
                       that already contains platform metadata
                       (e.g. 'A-AFFY-33 / HG-U133A').
    min_samples      : Override the rare-sample threshold shown in the log.
                       Falls back to data["metadata"]["min_samples_per_class"]
                       if available, then defaults to 2.
    train_idx, test_idx : row indices into data["y"] for a train/test split.
                       When both are given, adds a train/test sample-count
                       table (overall + per surviving class) to the log.
    """
    SENTINEL = "--- DATASET CLASS DISTRIBUTION LOG ---"
    description_path = Path(description_path)

    existing = (
        description_path.read_text(encoding="utf-8", errors="replace")
        if description_path.exists()
        else ""
    )
    if SENTINEL in existing:
        print(
            f"  Provenance log already present in '{description_path.name}'. Skipping."
        )
        return

    # ---------- gather raw class counts from unfiltered DataFrame ----------
    metadata = data.get("metadata", {})
    raw_df = data.get("dataframe")
    if raw_df is not None and len(raw_df.columns) >= 2:
        raw_counts: pd.Series = raw_df.iloc[:, 1].astype(str).value_counts()
    else:
        y_filt = data["y"]
        enc_fb = data.get("label_encoder")
        names_fb: list = (
            list(enc_fb.classes_) if enc_fb is not None
            else [str(i) for i in range(metadata.get("n_classes", 1))]
        )
        raw_counts = pd.Series(
            {names_fb[i]: int((y_filt == i).sum()) for i in range(len(names_fb))}
        )

    encoder = data.get("label_encoder")
    surviving_classes: set = (
        set(encoder.classes_.tolist()) if encoder is not None else set()
    )
    label_mapping: dict = metadata.get("label_mapping", {})

    threshold = (
        min_samples
        if min_samples is not None
        else metadata.get("min_samples_per_class", 2)
    )
    n_features_kept = metadata.get("n_features", "N/A")
    dataset_src = Path(metadata.get("dataset_path", "N/A")).name

    # ---------- build Markdown table ----------
    header = (
        "| Class Name | Original Sample Count "
        "| Encoded Integer ID | Operational Status |"
    )
    sep = (
        "|:-----------|:---------------------:"
        "|:------------------:|:-------------------|"
    )
    rows = [header, sep]
    for cls_name in sorted(raw_counts.index):
        count = int(raw_counts[cls_name])
        is_kept = cls_name in surviving_classes
        enc_id = label_mapping.get(str(cls_name), "—")
        status = (
            "KEPT"
            if is_kept
            else f"[DELETED] due to low sample count (< {threshold})"
        )
        rows.append(f"| {cls_name} | {count} | {enc_id} | {status} |")

    # ---------- rare-class removal summary (prominent, at-a-glance note) ----------
    removed = [
        (cls_name, int(raw_counts[cls_name]))
        for cls_name in raw_counts.index
        if cls_name not in surviving_classes
    ]
    total_samples = int(raw_counts.sum())
    n_removed = sum(n for _, n in removed)
    n_kept = total_samples - n_removed

    if removed:
        removed_desc = ", ".join(
            f"`{c}` (n={n})" for c, n in sorted(removed, key=lambda t: t[0])
        )
        removal_note = (
            f"> **NOTE — rare-class removal was applied before training.**  \n"
            f"> {n_removed} sample(s) across {len(removed)} class(es) were dropped "
            f"(threshold: < {threshold} samples/class): {removed_desc}.  \n"
            f"> Training set: {total_samples} → **{n_kept}** samples. "
            f"All models were trained on this reduced set.\n\n"
        )
    else:
        removal_note = (
            f"> **NOTE — no rare-class removal:** all {total_samples} samples retained "
            f"(every class has ≥ {threshold} samples).\n\n"
        )

    # ---------- train/test split breakdown (holdout flow only) ----------
    split_block = ""
    if train_idx is not None and test_idx is not None:
        y = data["y"]
        names = list(encoder.classes_) if encoder is not None else sorted(surviving_classes)
        n_train, n_test = int(len(train_idx)), int(len(test_idx))
        train_counts = np.bincount(y[train_idx], minlength=len(names))
        test_counts = np.bincount(y[test_idx], minlength=len(names))

        split_rows = [
            "| Class Name | Train Count | Test Count |",
            "|:-----------|:-----------:|:-----------:|",
        ]
        for i, cls_name in enumerate(names):
            split_rows.append(f"| {cls_name} | {int(train_counts[i])} | {int(test_counts[i])} |")

        test_frac = n_test / (n_train + n_test) if (n_train + n_test) else 0.0
        split_block = (
            f"> **Train / test split:** {n_train} train / {n_test} test "
            f"({test_frac:.0%} test)\n\n"
            + "\n".join(split_rows) + "\n\n"
        )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_block = (
        f"\n\n---\n\n"
        f"## {SENTINEL}\n\n"
        f"> Auto-generated by `update_dataset_description()` on {timestamp}  \n"
        f"> Source dataset: `{dataset_src}`\n\n"
        + removal_note
        + split_block
        + f"**Total features retained:** {n_features_kept}  \n"
        f"**Rare-sample filter threshold:** < {threshold} samples per class  \n\n"
        + "\n".join(rows)
        + "\n"
    )

    description_path.parent.mkdir(parents=True, exist_ok=True)
    with open(description_path, "a", encoding="utf-8") as fh:
        fh.write(log_block)

    print(f"  Provenance log appended -> {description_path}")
