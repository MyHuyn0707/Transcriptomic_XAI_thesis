"""
src/pipeline/benchmark.py

Benchmark — Feature Selection & Model Comparison.

For each dataset: for each enabled FS method (raw, mrmr, boruta), run feature
selection on the full dataset, then for each enabled model run StratifiedKFold
CV on the reduced dataset. SHAP explains the winning (fs × model) combo
automatically (interpretation.yaml -> shap.on_best_benchmark) and any other
combo on demand via run_shap().

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.helper import report
from src.helper.data_loader import load_dataset, load_fs_reduced_matrix
from src.helper.scaling import apply_scaler
from src.feature_selection import run_feature_selection
from src.benchmark.trainer import BenchmarkTrainer
from src.interpretation.shap_utils import explain_model
from src.visualize import update_dataset_description


class BenchmarkMixin:
    """Feature selection + model benchmarking, and on-demand SHAP."""

    def run_benchmark(
        self,
        dataset_names: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Run the benchmark: Feature Selection + Model Benchmarking.

        k_values : list of int, optional
            Rare-class thresholds (min_samples_per_class) to sweep — each k
            writes to ``self.output_root / f"k{k}" / dataset / ...`` instead
            of ``self.output_root / dataset / ...``. Omit for today's
            single-run behavior (uses load_dataset's own default k, writes
            straight under self.output_root, no k subfolder) — used e.g. by
            the live UI training job.

        Returns
        -------
        Without k_values: nested dict dataset → fs_method → model → summary_df.
        With k_values: nested dict k → dataset → fs_method → model → summary_df.
        """

        report.section(
            "Benchmark — Feature Selection & Model Benchmarking",
            "Classification Transcriptomic with XAI",
        )

        dataset_names = dataset_names or self.config_loader.get_enabled_datasets()

        if not dataset_names:
            report.warn("No datasets enabled. Edit configs/datasets.yaml.")
            return {}

        mode = "single dataset" if len(dataset_names) == 1 else f"{len(dataset_names)} datasets"
        report.info(f"Run mode: {mode} → {', '.join(dataset_names)}")

        if not k_values:
            all_results: Dict[str, Any] = {}
            for ds_name in dataset_names:
                report.subsection(f"Dataset: {ds_name}")
                try:
                    all_results[ds_name] = self._process_dataset(ds_name)
                except Exception as e:
                    report.err(f"Error processing {ds_name}: {e}")
            return all_results

        all_k_results: Dict[int, Any] = {}
        base_root = self.output_root
        for k in k_values:
            report.subsection(f"Rare-class k = {k}")
            self.output_root = base_root / f"k{k}"
            self.output_root.mkdir(parents=True, exist_ok=True)
            k_results: Dict[str, Any] = {}
            try:
                for ds_name in dataset_names:
                    report.subsection(f"Dataset: {ds_name} (k={k})")
                    try:
                        k_results[ds_name] = self._process_dataset(ds_name, min_samples_per_class=k)
                    except Exception as e:
                        report.err(f"Error processing {ds_name} (k={k}): {e}")
            finally:
                self.output_root = base_root
            all_k_results[k] = k_results
        return all_k_results

    def _process_dataset(
        self, dataset_name: str, min_samples_per_class: Optional[int] = None,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Full pipeline for one dataset: FS × models."""

        ds_cfg = self.config_loader.get_dataset_config(dataset_name)
        dataset_path = Path(ds_cfg["path"])
        dataset_type = ds_cfg.get("type", "auto")

        report.step(f"Loading {dataset_path.name}")
        load_kwargs = {} if min_samples_per_class is None else {"min_samples_per_class": min_samples_per_class}
        data = load_dataset(dataset_path=dataset_path, dataset_type=dataset_type, **load_kwargs)

        provenance_path = self.output_root / dataset_name / "DATASET_PROVENANCE.md"
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        update_dataset_description(data, description_path=provenance_path)

        X = data["X"]
        y = data["y"]
        feature_names = data["feature_names"]
        sample_ids = data["sample_ids"]

        report.info(
            f"{X.shape[0]} samples × {X.shape[1]} features · "
            f"{len(np.unique(y))} classes"
        )

        enabled_fs_methods = self.config_loader.get_enabled_fs_methods()
        enabled_models = self.config_loader.get_enabled_models()

        # Class names come from the dataset's LabelEncoder (aligned with y) so
        # SHAP plots show real names, not integers, for every dataset.
        class_labels = self._resolve_class_labels(data, ds_cfg)
        shap_cfg = self.config_loader.get_shap_config()
        # SHAP on the benchmark runs only when enabled AND on_best_benchmark;
        # it always targets the single winning (fs × model) by select_metric.
        shap_enabled = shap_cfg.get("enabled", True)
        shap_on_best = shap_cfg.get("on_best_benchmark", True)
        shap_top_k = shap_cfg.get("top_k", 20)
        shap_metric = shap_cfg.get("select_metric", "f1_macro")
        run_bench_shap = shap_enabled and shap_on_best

        # Build the probe→gene map once per dataset and reuse it for every model's SHAP.
        probe_gene_map: Dict[str, str] = {}
        if run_bench_shap:
            probe_gene_map = self._build_probe_gene_map(ds_cfg, feature_names)

        dataset_results: Dict[str, Dict[str, pd.DataFrame]] = {}
        # Keep each FS method's reduced matrix so SHAP can run after all CV.
        fs_outputs: Dict[str, Dict[str, Any]] = {}

        for fs_method in enabled_fs_methods:
            report.step(f"Feature selection: {fs_method}")

            fs_result = self._run_feature_selection(
                X=X,
                y=y,
                feature_names=feature_names,
                sample_ids=sample_ids,
                dataset_name=dataset_name,
                fs_method=fs_method,
            )

            X_sel = fs_result["X_selected"]
            feat_sel = fs_result["selected_features"]
            fs_outputs[fs_method] = {"X_sel": X_sel, "feat_sel": feat_sel}
            report.ok(f"{fs_method}: {len(feat_sel)} features selected")

            model_results: Dict[str, pd.DataFrame] = {}

            for model_name in enabled_models:
                report.step(f"Training {model_name.upper()} on {fs_method}")

                trainer = BenchmarkTrainer(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    fs_method=fs_method,
                    output_root=self.output_root,
                    wandb_project=self.wandb_project,
                    needs_scaling_map=self.config_loader.load_models_config().get("scaling"),
                )

                model_kwargs = self.config_loader.get_model_hyperparams(model_name)
                cv_cfg = self.config_loader.get_cv_config()
                max_splits = cv_cfg.get("n_splits", 5)

                try:
                    summary_df = trainer.run_cv(
                        X=X_sel,
                        y=y,
                        feature_names=feat_sel,
                        model_kwargs=model_kwargs,
                        max_splits=max_splits,
                        class_labels=class_labels,
                    )
                    model_results[model_name] = summary_df
                    report.ok(f"{model_name} complete")

                except Exception as e:
                    report.err(f"{model_name} failed: {e}")

            dataset_results[fs_method] = model_results

        self._report_dataset_results(dataset_name, dataset_results)

        # SHAP on the best benchmark combo — run AFTER benchmarking so we can
        # target the winning (fs × model). Reused verbatim by interpretation.
        shap_info: Optional[Dict[str, Any]] = None
        if run_bench_shap:
            shap_info = self._run_best_benchmark_shap(
                dataset_name=dataset_name,
                dataset_results=dataset_results,
                fs_outputs=fs_outputs,
                y=y,
                class_labels=class_labels,
                probe_gene_map=probe_gene_map,
                top_k=shap_top_k,
                metric=shap_metric,
            )

        if self.config_loader.get_write_benchmark_provenance():
            self._append_benchmark_provenance(
                dataset_name=dataset_name,
                dataset_results=dataset_results,
                shap_info=shap_info,
                metric=shap_metric,
            )

        return dataset_results

    def _append_benchmark_provenance(
        self,
        dataset_name: str,
        dataset_results: Dict[str, Dict[str, pd.DataFrame]],
        shap_info: Optional[Dict[str, Any]],
        metric: str,
    ) -> None:
        """
        Append (or refresh) a benchmark-comparison + SHAP section in
        ``outputs/{dataset}/DATASET_PROVENANCE.md``.

        Two tables: a detailed (fs × model) metric table and an (fs × model)
        pivot on ``metric``; followed by a SHAP summary (target, best combo,
        top genes). Idempotent — the block between sentinels is replaced each run.
        """
        prov_path = self.output_root / dataset_name / "DATASET_PROVENANCE.md"
        prov_path.parent.mkdir(parents=True, exist_ok=True)

        START = "<!-- BENCHMARK_COMPARISON_START -->"
        END = "<!-- BENCHMARK_COMPARISON_END -->"
        metric_cols = ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc"]

        detail: List[tuple] = []          # (fs, model, {metric: value})
        pivot: Dict[str, Dict[str, float]] = {}
        for fs_method, models in dataset_results.items():
            for model_name, summary_df in models.items():
                if summary_df is None or summary_df.empty:
                    continue
                means = summary_df.set_index("metric")["mean"]
                vals = {c: float(means.get(c, float("nan"))) for c in metric_cols}
                detail.append((fs_method, model_name, vals))
                pivot.setdefault(fs_method, {})[model_name] = vals.get(metric, float("nan"))

        lines: List[str] = [START, "", "## --- BENCHMARK COMPARISON ---", ""]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"> Auto-generated after the benchmark on {ts}")
        lines.append(f"> Selection metric: `{metric}`")
        lines.append("")
        lines.extend(report.markdown_metric_table(detail, pivot, metric_cols, metric))

        lines.append("### SHAP")
        if not shap_info:
            lines.append("_SHAP disabled (interpretation.shap.enabled/on_best_benchmark = false)._")
        else:
            lines.append(f"- target: best benchmark combo · metric: `{shap_info.get('metric')}`")
            best = shap_info.get("best")
            if best:
                lines.append(
                    f"- best combo: **{best['fs_method']} × {best['model']}** "
                    f"({shap_info.get('metric')}={report.fmt_metric(best['metric_value'])})"
                )
                top = best.get("top_features", [])
                if top:
                    lines.append(f"- top {len(top)} SHAP features: " + ", ".join(map(str, top)))
            explained = shap_info.get("explained", [])
            if len(explained) > 1:
                combos = ", ".join(f"{e['fs_method']}×{e['model']}" for e in explained)
                lines.append(f"- explained combos: {combos}")
        lines.append("")
        lines.append(END)
        block = "\n".join(lines)

        report.upsert_markdown_block(prov_path, block, START, END)
        report.ok(f"Benchmark comparison + SHAP summary → {prov_path.name}")

    def _run_best_benchmark_shap(
        self,
        dataset_name: str,
        dataset_results: Dict[str, Dict[str, pd.DataFrame]],
        fs_outputs: Dict[str, Dict[str, Any]],
        y: np.ndarray,
        class_labels: List[str],
        probe_gene_map: Dict[str, str],
        top_k: int,
        metric: str,
    ) -> Dict[str, Any]:
        """
        Run SHAP on the single best (fs_method, model) benchmark combo.

        The winner is the highest ``metric`` (default 'balanced_accuracy') among
        all successful combinations. SHAP is expensive on wide probe matrices
        (e.g. raw), so only the winner is explained automatically — interpretation
        reuses that combo's SHAP log directly, and :meth:`run_shap` explains any
        other combo on demand.

        Returns a summary dict (metric, best, explained combos + top features)
        for the benchmark provenance log.
        """
        info: Dict[str, Any] = {"metric": metric, "best": None, "explained": []}

        combos: List[tuple] = []
        for fs_method, models in dataset_results.items():
            for model_name, summary_df in models.items():
                if summary_df is not None and not summary_df.empty:
                    combos.append((fs_method, model_name))

        if not combos:
            report.warn("SHAP skipped — no successful (fs × model) combinations.")
            return info

        def _score(fs_model: tuple) -> float:
            fs_method, model_name = fs_model
            means = dataset_results[fs_method][model_name].set_index("metric")["mean"]
            return float(means.get(metric, float("nan")))

        best = max(combos, key=lambda c: (_score(c) if _score(c) == _score(c) else -1.0))
        report.info(
            f"SHAP → best benchmark combo: {best[0]} × {best[1]} "
            f"({metric}={_score(best):.4f})"
        )

        fs_method, model_name = best
        top_features = self._run_shap(
            dataset_name=dataset_name,
            fs_method=fs_method,
            model_name=model_name,
            X_sel=fs_outputs[fs_method]["X_sel"],
            feature_names_sel=fs_outputs[fs_method]["feat_sel"],
            class_labels=class_labels,
            probe_gene_map=probe_gene_map,
            top_k=top_k,
        )
        info["explained"].append({
            "fs_method": fs_method,
            "model": model_name,
            "metric_value": _score(best),
            "top_features": top_features or [],
        })
        info["best"] = info["explained"][0]
        return info

    @staticmethod
    def _report_dataset_results(
        dataset_name: str,
        dataset_results: Dict[str, Dict[str, pd.DataFrame]],
    ) -> None:
        """Render a consolidated fs × model overview table for one dataset."""
        rows = []
        for fs_method, models in dataset_results.items():
            for model_name, summary_df in models.items():
                if summary_df is None or summary_df.empty:
                    continue
                means = summary_df.set_index("metric")["mean"]
                rows.append(
                    {
                        "fs_method": fs_method,
                        "model": model_name,
                        "accuracy": float(means.get("accuracy", float("nan"))),
                        "balanced_accuracy": float(
                            means.get("balanced_accuracy", float("nan"))
                        ),
                        "f1_macro": float(means.get("f1_macro", float("nan"))),
                        "roc_auc": float(means.get("roc_auc", float("nan"))),
                    }
                )
        if rows:
            overview = pd.DataFrame(rows)
            report.dataframe_table(
                overview, title=f"{dataset_name} — CV results (mean)", index=False
            )

    def _run_feature_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        sample_ids: List[str],
        dataset_name: str,
        fs_method: str,
    ) -> Dict[str, Any]:
        """Run feature selection on the full dataset (raw/mrmr/boruta).

        NOTE — selection leakage: features are ranked using the whole
        dataset (including labels) before run_cv splits into CV folds, so
        each fold's "validation" samples already influenced which features
        were kept. This makes the CV metrics from run_benchmark() optimistic
        relative to a true held-out evaluation. src/pipeline/holdout.py runs
        feature selection on the train split only, so treat holdout.py's
        numbers as the leakage-free reference and this benchmark path's
        numbers as an upper-bound/model-comparison signal, not an unbiased
        estimate of generalization performance.
        """
        params = self.config_loader.get_fs_method_params(fs_method)
        source_path = (
            self.config_loader.get_dataset_config(dataset_name).get("path")
            if fs_method == "raw" else None
        )
        return run_feature_selection(
            fs_method, X, y, feature_names, sample_ids, dataset_name,
            self.output_root, params, allow_raw=True, source_path=source_path,
        )

    def _run_shap(
        self,
        dataset_name: str,
        fs_method: str,
        model_name: str,
        X_sel: np.ndarray,
        feature_names_sel: List[str],
        class_labels: List[str],
        probe_gene_map: Dict[str, str],
        top_k: int = 20,
        fold: int = 1,
    ) -> Optional[List[str]]:
        """
        Compute SHAP for a trained benchmark model and export to
        ``outputs/{dataset}/{fs}/{model}/shap/``.

        Loads fold-``fold`` model + scaler saved during CV. Probe-level SHAP is
        collapsed to gene symbols for display when ``probe_gene_map`` resolves.
        Non-fatal: returns None and warns on any failure.
        """
        model_path = (
            self.output_root / dataset_name / fs_method / model_name
            / "models" / f"{model_name}_fold_{fold}.joblib"
        )
        if not model_path.exists():
            report.warn(f"SHAP skipped — model not found: {model_path}")
            return None

        try:
            import joblib
            artifact = joblib.load(model_path)
            model = artifact["model"]
            scaler = artifact["scaler"]
            X_scaled = apply_scaler(scaler, X_sel)

            shap_dir = self.output_root / dataset_name / fs_method / model_name / "shap"
            report.step(f"SHAP ({model_name} · {fs_method} · top-{top_k})")
            _, top_features = explain_model(
                model=model,
                X_train=X_scaled,
                X_test=X_scaled,
                feature_names=feature_names_sel,
                output_dir=shap_dir,
                top_k=top_k,
                class_names=class_labels if class_labels else None,
                probe_gene_map=probe_gene_map or None,
            )
            report.ok(f"SHAP complete → {shap_dir}")
            return top_features
        except Exception as e:
            report.warn(f"SHAP failed for {model_name}/{fs_method}: {e}")
            return None

    def run_shap(
        self,
        dataset_name: str,
        fs_method: str,
        model_name: str,
        fold: int = 1,
        top_k: Optional[int] = None,
    ) -> Optional[List[str]]:
        """
        Run SHAP on ANY already-trained benchmark model, on demand.

        Explains the model saved by :meth:`run_benchmark` at
        ``outputs/{dataset}/{fs_method}/{model_name}/models/{model}_fold_{fold}.joblib``
        and writes to ``outputs/{dataset}/{fs_method}/{model_name}/shap/``. Use
        this to explain a combo that wasn't the benchmark winner (the automatic
        SHAP only covers the best combo) — no re-training needed.

        Parameters
        ----------
        dataset_name, fs_method, model_name : str
            The trained combination to explain.
        fold : int
            CV fold to load (1-based, default 1).
        top_k : int, optional
            Top features to extract (default: interpretation.yaml shap.top_k).

        Returns
        -------
        list[str] | None
            Top-K feature/gene names, or None if the model or SHAP failed.
        """
        report.section(
            "SHAP — On-Demand Explanation",
            f"{dataset_name} · {fs_method} · {model_name} · fold {fold}",
        )

        shap_cfg = self.config_loader.get_shap_config()
        top_k = top_k if top_k is not None else shap_cfg.get("top_k", 20)

        ds_cfg = self.config_loader.get_dataset_config(dataset_name)

        # Resolve the FS-reduced (raw-scale) matrix this model was trained on.
        data = load_dataset(
            dataset_path=Path(ds_cfg["path"]),
            dataset_type=ds_cfg.get("type", "auto"),
        )
        feature_names = data["feature_names"]
        class_labels = self._resolve_class_labels(data, ds_cfg)

        if fs_method == "raw":
            X_sel, feat_sel = data["X"], list(feature_names)
        else:
            csv_dir = (
                self.output_root / "feature_selection"
                / dataset_name / fs_method / "processed_datasets"
            )
            loaded = load_fs_reduced_matrix(csv_dir)
            if loaded is None:
                report.err(f"Processed dataset missing: {csv_dir}")
                report.info("Run run_benchmark() for this fs_method first.")
                return None
            X_sel, feat_sel = loaded

        probe_gene_map = self._build_probe_gene_map(ds_cfg, feature_names)

        return self._run_shap(
            dataset_name=dataset_name,
            fs_method=fs_method,
            model_name=model_name,
            X_sel=X_sel,
            feature_names_sel=feat_sel,
            class_labels=class_labels,
            probe_gene_map=probe_gene_map,
            top_k=top_k,
            fold=fold,
        )
