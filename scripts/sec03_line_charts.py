"""Generate dataset-wise model line charts and matching summary tables for Section 3.

The caller supplies the already validated K-Fold/train-test data frames. CSV files
are written first, then read back as the only input to charts so that every
displayed mean can be traced to an exported value. Thesis LaTeX is intentionally
kept outside this output pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "precision", "recall"]
CHART_METRICS = ["accuracy", "f1_macro"]
METRIC_DISPLAY = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "f1_macro": "F1-macro",
    "precision": "Precision-macro",
    "recall": "Recall-macro",
}


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _ordered_subset(values: Sequence[str], available: set[str]) -> list[str]:
    return [value for value in values if value in available]


def _prepare_csv_tables(
    data: pd.DataFrame,
    table_root: Path,
    dataset_order: Sequence[str],
    fs_order: Sequence[str],
    model_order: Sequence[str],
) -> tuple[Path, Path, list[Path]]:
    """Export detail points, horizontal-line means, and one audit CSV per k/FS."""
    table_root.mkdir(parents=True, exist_ok=True)
    keys = ["k", "fs_method", "model"]

    means = data.groupby(keys, as_index=False, sort=False)[METRICS].mean()
    means = means.rename(columns={metric: f"mean_{metric}_across_datasets" for metric in METRICS})
    counts = (
        data.groupby(keys, as_index=False, sort=False)["dataset"]
        .nunique()
        .rename(columns={"dataset": "n_datasets_in_mean"})
    )
    means = means.merge(counts, on=keys, validate="one_to_one")

    detail = data[["k", "fs_method", "dataset", "model", *METRICS]].copy()
    detail = detail.merge(means, on=keys, how="left", validate="many_to_one")

    dataset_rank = {name: index for index, name in enumerate(dataset_order)}
    fs_rank = {name: index for index, name in enumerate(fs_order)}
    model_rank = {name: index for index, name in enumerate(model_order)}
    detail["_dataset_rank"] = detail["dataset"].map(dataset_rank)
    detail["_fs_rank"] = detail["fs_method"].map(fs_rank)
    detail["_model_rank"] = detail["model"].map(model_rank)
    detail = detail.sort_values(["k", "_fs_rank", "_dataset_rank", "_model_rank"])
    detail = detail.drop(columns=["_dataset_rank", "_fs_rank", "_model_rank"])

    means["_fs_rank"] = means["fs_method"].map(fs_rank)
    means["_model_rank"] = means["model"].map(model_rank)
    means = means.sort_values(["k", "_fs_rank", "_model_rank"])
    means = means.drop(columns=["_fs_rank", "_model_rank"])

    detail_path = table_root / "dataset_model_metrics_with_means.csv"
    means_path = table_root / "model_mean_metrics_by_k_fs.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    means.to_csv(means_path, index=False, encoding="utf-8-sig")

    audit_paths: list[Path] = []
    for (k, fs_method), group in detail.groupby(["k", "fs_method"], sort=False):
        group_dir = table_root / f"k{int(k)}"
        group_dir.mkdir(parents=True, exist_ok=True)
        point_rows = group[["dataset", "model", *METRICS]].copy()
        point_rows.insert(0, "row_type", "dataset_result")
        point_rows["n_datasets_in_mean"] = np.nan

        group_means = means[(means["k"] == k) & (means["fs_method"] == fs_method)].copy()
        mean_rows = pd.DataFrame({
            "row_type": "model_mean_across_datasets",
            "dataset": "MEAN_ACROSS_DATASETS",
            "model": group_means["model"].to_numpy(),
            **{
                metric: group_means[f"mean_{metric}_across_datasets"].to_numpy()
                for metric in METRICS
            },
            "n_datasets_in_mean": group_means["n_datasets_in_mean"].to_numpy(),
        })
        audit = pd.concat([point_rows, mean_rows], ignore_index=True)
        audit_path = group_dir / f"{fs_method}_dataset_model_metrics_and_means.csv"
        audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
        audit_paths.append(audit_path)

    # Read-back validation: exported means must equal an independent recomputation.
    exported = pd.read_csv(means_path)
    recomputed = data.groupby(keys, sort=False)[METRICS].mean().reset_index()
    check = exported.merge(recomputed, on=keys, validate="one_to_one")
    for metric in METRICS:
        if not np.allclose(
            check[f"mean_{metric}_across_datasets"], check[metric], rtol=0, atol=1e-12
        ):
            raise AssertionError(f"Sai lệch mean sau khi xuất CSV: {metric}")

    return detail_path, means_path, audit_paths


def _draw_lines(
    ax: plt.Axes,
    group: pd.DataFrame,
    means: pd.DataFrame,
    metric: str,
    dataset_order: Sequence[str],
    dataset_display: Mapping[str, str],
    model_order: Sequence[str],
    model_display: Mapping[str, str],
    colors: Mapping[str, object],
    include_mean_legend: bool,
) -> None:
    available = set(group["dataset"])
    datasets = _ordered_subset(dataset_order, available)
    x = np.arange(len(datasets))
    for model in _ordered_subset(model_order, set(group["model"])):
        model_rows = group[group["model"] == model].set_index("dataset").reindex(datasets)
        color = colors[model]
        ax.plot(
            x,
            model_rows[metric],
            marker="o",
            markersize=4.5,
            linewidth=1.6,
            color=color,
            label=model_display[model],
        )
        mean_row = means[means["model"] == model]
        if mean_row.empty:
            continue
        mean_value = float(mean_row.iloc[0][f"mean_{metric}_across_datasets"])
        mean_label = f"TB {model_display[model]} ({mean_value:.3f})" if include_mean_legend else None
        ax.axhline(mean_value, color=color, linestyle="--", linewidth=1.15, alpha=0.78, label=mean_label)

    ax.set_xticks(x, [dataset_display.get(name, name) for name in datasets], rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Bộ dữ liệu")
    ax.set_ylabel(METRIC_DISPLAY[metric])
    ax.grid(True, alpha=0.22)


def _generate_charts(
    detail_path: Path,
    means_path: Path,
    chart_root: Path,
    protocol_title: str,
    dataset_order: Sequence[str],
    dataset_display: Mapping[str, str],
    fs_order: Sequence[str],
    fs_display: Mapping[str, str],
    model_order: Sequence[str],
    model_display: Mapping[str, str],
) -> tuple[list[Path], list[Path]]:
    """Read exported CSVs and generate individual plus per-k overview charts."""
    detail = pd.read_csv(detail_path)
    means = pd.read_csv(means_path)
    colors = {
        model: plt.cm.tab10(index % 10) for index, model in enumerate(model_order)
    }
    individual_paths: list[Path] = []
    overview_paths: list[Path] = []

    for (k, fs_method), group in detail.groupby(["k", "fs_method"], sort=False):
        group_means = means[(means["k"] == k) & (means["fs_method"] == fs_method)]
        for metric in CHART_METRICS:
            fig, ax = plt.subplots(figsize=(15.5, 7.6))
            _draw_lines(
                ax, group, group_means, metric, dataset_order, dataset_display,
                model_order, model_display, colors, include_mean_legend=True,
            )
            ax.set_title(
                f"{protocol_title} | k={int(k)} | FS={fs_display[fs_method]} | "
                f"{METRIC_DISPLAY[metric]} trên {group['dataset'].nunique()} bộ dữ liệu"
            )
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.2, ncol=1)
            fig.tight_layout()
            path = chart_root / "by_k_feature_selection" / f"k{int(k)}" / f"{fs_method}_{metric}.png"
            _save_figure(fig, path)
            individual_paths.append(path)

    for k in sorted(detail["k"].unique()):
        k_data = detail[detail["k"] == k]
        methods = _ordered_subset(fs_order, set(k_data["fs_method"]))
        fig, axes = plt.subplots(
            len(methods), len(CHART_METRICS),
            figsize=(15.5, 3.45 * len(methods) + 1.2),
            sharex=False, sharey=True, squeeze=False,
        )
        for row_index, fs_method in enumerate(methods):
            group = k_data[k_data["fs_method"] == fs_method]
            group_means = means[(means["k"] == k) & (means["fs_method"] == fs_method)]
            for column_index, metric in enumerate(CHART_METRICS):
                ax = axes[row_index, column_index]
                _draw_lines(
                    ax, group, group_means, metric, dataset_order, dataset_display,
                    model_order, model_display, colors, include_mean_legend=False,
                )
                ax.set_title(f"FS={fs_display[fs_method]} | {METRIC_DISPLAY[metric]}")
                ax.tick_params(axis="x", labelsize=7.5)

        handles = [
            plt.Line2D([0], [0], color=colors[model], marker="o", linewidth=1.6,
                       label=model_display[model])
            for model in _ordered_subset(model_order, set(k_data["model"]))
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.948),
            ncol=len(handles),
            frameon=False,
        )
        fig.suptitle(
            f"{protocol_title}: Accuracy và F1-macro theo bộ dữ liệu tại k={int(k)}\n"
            "Đường nét đứt biểu diễn trung bình của mô hình trên các bộ dữ liệu",
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.905))
        path = chart_root / "overview_by_k" / f"k{int(k)}_accuracy_f1_macro.png"
        _save_figure(fig, path)
        overview_paths.append(path)

    return individual_paths, overview_paths


def generate_protocol_line_artifacts(
    data: pd.DataFrame,
    protocol_key: str,
    protocol_title: str,
    table_root: Path,
    chart_root: Path,
    dataset_order: Sequence[str],
    dataset_display: Mapping[str, str],
    fs_order: Sequence[str],
    fs_display: Mapping[str, str],
    model_order: Sequence[str],
    model_display: Mapping[str, str],
) -> dict[str, object]:
    """Generate all dataset-line artifacts for one evaluation protocol."""
    line_table_root = table_root / "dataset_model_line_charts"
    detail_path, means_path, audit_paths = _prepare_csv_tables(
        data, line_table_root, dataset_order, fs_order, model_order
    )
    individual_paths, overview_paths = _generate_charts(
        detail_path, means_path, chart_root, protocol_title,
        dataset_order, dataset_display, fs_order, fs_display,
        model_order, model_display,
    )
    return {
        "detail_csv": detail_path,
        "mean_csv": means_path,
        "audit_csvs": audit_paths,
        "individual_charts": individual_paths,
        "overview_charts": overview_paths,
    }
