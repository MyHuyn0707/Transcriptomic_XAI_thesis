"""
src/pipeline/baseline_split.py

Train/test-split baseline (ALL models, boruta/mrmr only) — sweeps rare-class
thresholds (k = min_samples_per_class). Deliberately kept separate from
HoldoutMixin (src/pipeline/holdout.py), which also backs the live UI training
job (src/api/jobs.py calls its per-model method directly) — this flow is
reporting-only and must not risk that path.

Writes outputs_baseline_split/k{k}/{dataset}/... which
run_rule_extraction_holdout() (holdout.py) reads split_info.json +
feature_selection/<fs_method>/ artifacts from for the same k, instead of
re-deriving its own split/FS — see configs/holdout.yaml -> baseline_split_root.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.helper import report
from src.helper.data_loader import load_dataset
from src.helper.metrics import compute_metrics
from src.helper.scaling import apply_scaler, fit_scaler
from src.helper.split import train_test_split_indices
from src.feature_selection import run_feature_selection
from src.models.factory import get_model
from src.visualize import (
    plot_class_distribution,
    plot_train_test_split,
    update_dataset_description,
)

# fs_methods this flow (and rule extraction, reading its output) ever touches —
# no 'raw' here, confirmed scope: baseline_split + rule extraction = boruta/mrmr only.
_DEFAULT_FS_METHODS = ["boruta", "mrmr_k50", "mrmr_k75"]


def _serialize(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class BaselineSplitMixin:
    """Train/test-split baseline: all models, boruta/mrmr FS, rare-class k sweep."""

    def run_baseline_split(
        self,
        dataset_names: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        For each k in ``k_values`` (default feature_selection.yaml ->
        rare_class_k_values), for each dataset: drop rare classes at that k,
        stratified train/test split, run FS (``boruta`` [auto mode],
        ``mrmr_k50``, ``mrmr_k75``) on train only, then fit every enabled
        model (configs/models.yaml) on the FS-reduced train set and evaluate
        on both train and test.

        Writes ``outputs_baseline_split/k{k}/{dataset}/`` — split_info.json,
        visualizations/, feature_selection/<fs_method>/ (from
        run_feature_selection), and <fs_method>/<model>/ (model+scaler
        artifact, train+test metrics incl. per-class breakdown).

        Returns
        -------
        dict ``{k: {dataset_name: {fs_method: {model_name: test_metrics}}}}``
        """
        import matplotlib.pyplot as plt

        report.section(
            "Baseline (Train/Test Split) — All Models",
            "Rare-class k sweep · train-only FS (boruta/mrmr) · full model roster",
        )

        fs_cfg = self.config_loader.load_feature_selection_config()
        k_values = k_values or list(fs_cfg.get("rare_class_k_values", [4]))

        holdout_cfg = dict(self.config_loader.load_yaml("holdout.yaml").get("holdout", {}))
        split_baseline_cfg = dict(holdout_cfg.get("split_baseline", {}))
        batch_cfg = dict(holdout_cfg.get("batch", {}))

        split_root = Path(split_baseline_cfg.get("baseline_split_root", "outputs_baseline_split"))
        if not split_root.is_absolute():
            split_root = self.config_loader.project_root / split_root

        fs_methods = list(batch_cfg.get("fs_methods", _DEFAULT_FS_METHODS))
        test_size = float(split_baseline_cfg.get("test_size", 0.15))
        split_seed = int(split_baseline_cfg.get("split_random_state", 42))

        enabled_models = self.config_loader.get_enabled_models()
        needs_scaling_map = self.config_loader.load_models_config().get("scaling")

        dataset_names = dataset_names or self.config_loader.get_enabled_datasets()
        if not dataset_names:
            report.warn("No datasets enabled. Edit configs/datasets.yaml.")
            return {}

        all_results: Dict[int, Any] = {}

        for k in k_values:
            report.subsection(f"Rare-class k = {k}")
            k_root = split_root / f"k{k}"
            k_root.mkdir(parents=True, exist_ok=True)
            k_results: Dict[str, Any] = {}
            data_by_dataset: Dict[str, Any] = {}  # for the combined class-distribution grid below

            for ds_name in dataset_names:
                report.subsection(f"Dataset: {ds_name} (k={k})")
                try:
                    ds_cfg = self.config_loader.get_dataset_config(ds_name)
                    ds_root = k_root / ds_name
                    ds_root.mkdir(parents=True, exist_ok=True)

                    data = load_dataset(
                        dataset_path=Path(ds_cfg["path"]),
                        dataset_type=ds_cfg.get("type", "auto"),
                        min_samples_per_class=k,
                    )
                    X, y = data["X"], data["y"]
                    feature_names = data["feature_names"]
                    sample_ids = data["sample_ids"]
                    class_labels = self._resolve_class_labels(data, ds_cfg)
                    n_classes = len(np.unique(y))
                    data_by_dataset[ds_name] = data

                    split = train_test_split_indices(y, test_size=test_size, random_state=split_seed)
                    train_idx, test_idx = split["train_idx"], split["test_idx"]
                    report.info(
                        f"Split: {len(train_idx)} train / {len(test_idx)} test "
                        f"· {n_classes} classes survive (< {k} dropped)"
                    )

                    viz_dir = ds_root / "visualizations"
                    viz_dir.mkdir(parents=True, exist_ok=True)
                    fig1 = plot_class_distribution(
                        data, title=f"{ds_name} — class distribution (k={k})",
                        show_class_ids=True,
                    )
                    fig1.savefig(viz_dir / "class_distribution_after_dropna.png",
                                 dpi=150, bbox_inches="tight")
                    plt.close(fig1)

                    fig2 = plot_train_test_split(
                        data, train_idx, test_idx, title=f"{ds_name} — train/test split (k={k})",
                        show_class_ids=True,
                    )
                    fig2.savefig(viz_dir / "train_test_split.png", dpi=150, bbox_inches="tight")
                    plt.close(fig2)

                    update_dataset_description(
                        data, description_path=ds_root / "DATASET_PROVENANCE_SPLIT.md",
                        min_samples=k, train_idx=train_idx, test_idx=test_idx,
                    )

                    # Class distribution BEFORE rare-class removal, straight from the
                    # already-loaded raw dataframe (data["dataframe"], see
                    # load_dataset()'s docstring) — same source plot_class_distribution()
                    # uses. Folding this into split_info.json means the API
                    # (src/api/registry.load_raw_class_counts) no longer needs a
                    # separate cache file or a manual export script to have this data.
                    raw_counts_series = data["dataframe"].iloc[:, 1].astype(str).value_counts()
                    raw_class_counts = {str(c): int(n) for c, n in raw_counts_series.items()}
                    dropped_classes = [c for c in raw_class_counts if c not in class_labels]

                    split_info = {
                        "dataset": ds_name,
                        "min_samples_per_class": k,
                        "test_size": test_size,
                        "split_random_state": split_seed,
                        "n_samples_total": int(len(y)),
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "class_labels": class_labels,
                        "train_class_counts": {
                            class_labels[c] if c < len(class_labels) else str(c): int(n)
                            for c, n in enumerate(np.bincount(y[train_idx], minlength=len(class_labels)))
                        },
                        "test_class_counts": {
                            class_labels[c] if c < len(class_labels) else str(c): int(n)
                            for c, n in enumerate(np.bincount(y[test_idx], minlength=len(class_labels)))
                        },
                        "raw_class_counts": raw_class_counts,
                        "dropped_classes": dropped_classes,
                        "train_idx": train_idx.tolist(),
                        "test_idx": test_idx.tolist(),
                    }
                    (ds_root / "split_info.json").write_text(
                        json.dumps(_serialize(split_info), indent=2), encoding="utf-8"
                    )

                    X_train, y_train = X[train_idx], y[train_idx]
                    X_test, y_test = X[test_idx], y[test_idx]
                    sample_ids_train = [sample_ids[i] for i in train_idx]

                    ds_results: Dict[str, Any] = {}

                    for fs_method in fs_methods:
                        report.step(f"Feature selection ({fs_method}) on TRAIN only")
                        try:
                            params = self.config_loader.get_fs_method_params(fs_method)
                            fs_result = run_feature_selection(
                                fs_method, X_train, y_train, feature_names, sample_ids_train,
                                ds_name, k_root, params, allow_raw=False,
                            )
                        except Exception as e:
                            report.err(f"{fs_method} (train-only) failed: {e}")
                            continue

                        feat_sel = fs_result["selected_features"]
                        idx_map = {f: i for i, f in enumerate(feature_names)}
                        sel_idx = np.array([idx_map[f] for f in feat_sel], dtype=np.int64)
                        X_train_sel = X_train[:, sel_idx]
                        X_test_sel = X_test[:, sel_idx]
                        report.ok(f"{fs_method}: {len(feat_sel)} features selected (train-only)")

                        fs_root = ds_root / fs_method
                        model_summaries: Dict[str, Any] = {}

                        for model_name in enabled_models:
                            report.step(f"Training {model_name.upper()} on {fs_method} (k={k})")
                            try:
                                model_kwargs = self.config_loader.get_model_hyperparams(model_name)
                                scaler = fit_scaler(model_name, X_train_sel, needs_scaling_map)
                                X_train_s = apply_scaler(scaler, X_train_sel)
                                X_test_s = apply_scaler(scaler, X_test_sel)

                                model = get_model(
                                    model_name,
                                    input_dim=X_train_s.shape[1],
                                    num_classes=n_classes,
                                    random_state=42,
                                    **model_kwargs,
                                )
                                if model_name == "ann":
                                    model.fit(X_train_s, y_train, X_val=X_test_s, y_val=y_test)
                                else:
                                    model.fit(X_train_s, y_train)

                                def _predict(Xs):
                                    pred = model.predict(Xs)
                                    prob = model.predict_proba(Xs) if hasattr(model, "predict_proba") else None
                                    return pred, prob

                                y_train_pred, y_train_prob = _predict(X_train_s)
                                y_test_pred, y_test_prob = _predict(X_test_s)

                                train_metrics = compute_metrics(
                                    y_train, y_train_pred, y_train_prob,
                                    labels=list(range(n_classes)), class_labels=class_labels,
                                )
                                test_metrics = compute_metrics(
                                    y_test, y_test_pred, y_test_prob,
                                    labels=list(range(n_classes)), class_labels=class_labels,
                                )
                                per_class = {
                                    "train": train_metrics.pop("per_class"),
                                    "test": test_metrics.pop("per_class"),
                                }

                                model_dir = fs_root / model_name / "models"
                                model_dir.mkdir(parents=True, exist_ok=True)
                                joblib.dump(
                                    {"model": model, "scaler": scaler},
                                    model_dir / f"{model_name}.joblib",
                                )
                                (fs_root / model_name / "metrics.json").write_text(
                                    json.dumps(_serialize({
                                        "train": train_metrics, "test": test_metrics,
                                    }), indent=2),
                                    encoding="utf-8",
                                )
                                (fs_root / model_name / "per_class_metrics.json").write_text(
                                    json.dumps(_serialize(per_class), indent=2), encoding="utf-8"
                                )

                                model_summaries[model_name] = {"train": train_metrics, "test": test_metrics}
                                self._wandb_log_baseline_split(
                                    ds_name, k, fs_method, model_name, train_metrics, test_metrics,
                                )
                                report.ok(
                                    f"{model_name}: train f1_macro={train_metrics['f1_macro']:.4f} · "
                                    f"test f1_macro={test_metrics['f1_macro']:.4f}"
                                )
                            except Exception as e:
                                report.err(f"{fs_method} x {model_name} failed: {e}")

                        ds_results[fs_method] = model_summaries

                    k_results[ds_name] = ds_results
                    self._write_baseline_split_summary(ds_results, ds_root)

                except Exception as e:
                    report.err(f"Baseline split failed for {ds_name} (k={k}): {e}")

            all_results[k] = k_results
            self._wandb_log_baseline_split_k_comparison(k, k_results)
            self._write_combined_visualizations(k, k_root, data_by_dataset, k_results)

        return all_results

    @staticmethod
    def _write_combined_visualizations(
        k: int,
        k_root: Path,
        data_by_dataset: Dict[str, Any],
        k_results: Dict[str, Any],
    ) -> None:
        """Collective ("all datasets at once") visualizations for this k —
        the batch counterpart to notebooks/01_Datasets_Visualization.ipynb's
        hand-rolled grids, generated automatically per k sweep value."""
        if not data_by_dataset:
            return
        import matplotlib.pyplot as plt
        from src.visualize import plot_all_class_distributions, plot_all_datasets_metric_comparison

        viz_dir = k_root / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        try:
            fig = plot_all_class_distributions(
                data_by_dataset, show_class_ids=True,
                suptitle=f"Class distribution — all datasets (k={k})",
            )
            fig.savefig(viz_dir / "all_datasets_class_distribution.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            report.warn(f"Combined class-distribution plot failed (k={k}): {e}")

        # {dataset: {fs_method: {model: test_metrics_dict}}}
        results_by_dataset = {
            ds_name: {
                fs_method: {m: summary["test"] for m, summary in models.items()}
                for fs_method, models in ds_results.items()
            }
            for ds_name, ds_results in k_results.items()
        }
        try:
            fig = plot_all_datasets_metric_comparison(
                results_by_dataset, metric="f1_macro",
                suptitle=f"f1_macro — fs_method × model, all datasets (k={k})",
            )
            fig.savefig(viz_dir / "all_datasets_metric_comparison.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            report.warn(f"Combined metric-comparison plot failed (k={k}): {e}")

        report.ok(f"Combined visualizations (k={k}) → {viz_dir}")

    @staticmethod
    def _write_baseline_split_summary(
        ds_results: Dict[str, Any],
        ds_root: Path,
    ) -> None:
        """Write train_test_summary.csv (one row per fs_method x model, test-set
        metrics) and append a provenance block to DATASET_PROVENANCE_SPLIT.md."""
        metric_cols = ["accuracy", "balanced_accuracy", "f1_macro", "precision", "recall"]
        rows = []
        detail: List[tuple] = []
        pivot: Dict[str, Dict[str, float]] = {}
        for fs_method, models in ds_results.items():
            for model_name, summary in models.items():
                test_metrics = summary.get("test", {})
                row = {"fs_method": fs_method, "model": model_name}
                row.update({f"test_{c}": test_metrics.get(c) for c in metric_cols})
                row.update({f"train_{c}": summary.get("train", {}).get(c) for c in metric_cols})
                rows.append(row)
                vals = {c: float(test_metrics.get(c, float("nan"))) for c in metric_cols}
                detail.append((fs_method, model_name, vals))
                pivot.setdefault(fs_method, {})[model_name] = vals.get("f1_macro", float("nan"))

        pd.DataFrame(rows).to_csv(ds_root / "train_test_summary.csv", index=False)

        prov_path = ds_root / "DATASET_PROVENANCE_SPLIT.md"
        START = "<!-- BASELINE_SPLIT_COMPARISON_START -->"
        END = "<!-- BASELINE_SPLIT_COMPARISON_END -->"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines: List[str] = [
            START, "",
            "## --- BASELINE (TRAIN/TEST SPLIT) COMPARISON ---", "",
            f"> Auto-generated by run_baseline_split() on {ts}",
            f"> Selection metric shown in pivot: `f1_macro` (test-set)",
            "",
        ]
        lines.extend(report.markdown_metric_table(
            detail, pivot, metric_cols, "f1_macro",
            detail_title="Each fs_method × model (test-set metrics)",
        ))
        lines.append(END)
        report.upsert_markdown_block(prov_path, "\n".join(lines), START, END)
        report.ok(f"Baseline-split comparison → {prov_path.name}")

    def _wandb_log_baseline_split(
        self,
        dataset_name: str,
        k: int,
        fs_method: str,
        model_name: str,
        train_metrics: Dict[str, Any],
        test_metrics: Dict[str, Any],
    ) -> None:
        """One W&B run per (dataset, k, fs_method, model) — train+test scalars."""
        if not self.wandb_project:
            return
        try:
            import wandb
        except ImportError:
            return
        run = wandb.init(
            project=self.wandb_project,
            name=f"{dataset_name}_k{k}_{fs_method}_{model_name}_split",
            group=f"{dataset_name}_k{k}_{fs_method}_{model_name}",
            job_type="baseline_split",
            reinit=True,
        )
        try:
            log_data = {}
            for split_name, m in (("train", train_metrics), ("test", test_metrics)):
                for key, val in m.items():
                    if key not in ("confusion_matrix", "per_class") and val is not None:
                        log_data[f"{split_name}_{key}"] = val
            log_data["k"] = k
            run.log(log_data)
        finally:
            run.finish()

    def _wandb_log_baseline_split_k_comparison(
        self,
        k: int,
        k_results: Dict[str, Any],
    ) -> None:
        """One summary W&B run per k — a table of every (dataset, fs_method,
        model) test-set row, for cross-model/fs comparison at this k."""
        if not self.wandb_project or not k_results:
            return
        try:
            import wandb
        except ImportError:
            return
        rows = []
        for ds_name, ds_results in k_results.items():
            for fs_method, models in ds_results.items():
                for model_name, summary in models.items():
                    test_metrics = summary.get("test", {})
                    rows.append([
                        ds_name, fs_method, model_name,
                        test_metrics.get("accuracy"), test_metrics.get("balanced_accuracy"),
                        test_metrics.get("f1_macro"),
                        test_metrics.get("precision"), test_metrics.get("recall"),
                    ])
        if not rows:
            return
        run = wandb.init(
            project=self.wandb_project,
            name=f"baseline_split_k{k}_comparison",
            job_type="k_comparison",
            reinit=True,
        )
        try:
            table = wandb.Table(
                columns=["dataset", "fs_method", "model", "accuracy", "balanced_accuracy",
                         "f1_macro", "precision", "recall"],
                data=rows,
            )
            run.log({f"k{k}_comparison": table})
        finally:
            run.finish()
