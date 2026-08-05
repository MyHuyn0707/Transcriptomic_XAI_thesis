"""
src/pipeline/interpretation.py

Biological interpretation for one (dataset, fs_method, model, fold) combination.

Reuses the benchmark's SHAP top-K genes (never recomputes SHAP) and loads the
tree-based rules produced by run_rule_extraction.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.helper import report
from src.helper.data_loader import load_dataset, load_fs_reduced_matrix
from src.helper.scaling import apply_scaler
from src.visualize import update_dataset_description


class InterpretationMixin:
    """Biological interpretation: reused SHAP + tree rules."""

    def run_interpretation(
        self,
        dataset_name: str,
        fs_method: str = "mrmr_mid",
        model_name: str = "rf",
        fold: int = 1,
        platform: Optional[str] = None,
        annotation_path: Optional[str] = None,
        annotation_source: str = "auto",
        organism: str = "hsapiens",
    ) -> Dict[str, Any]:
        """
        Biological interpretation — reuse benchmark SHAP and load tree-based
        rules for one specific trained model.

        SHAP is not recomputed here (it is produced by the benchmark; see
        ``run_shap``) and rules are not mined here (they are produced by
        ``run_rule_extraction`` and merely loaded for display).

        Parameters
        ----------
        dataset_name : str
        fs_method : str
            Which feature selection output to use (e.g. 'mrmr_mid', 'boruta').
        model_name : str
        fold : int
            CV fold to load (1-based).
        platform : str, optional
            GPL platform for probe→gene annotation (used only when
            annotation_source='gpl'); falls back to the dataset's configured platform.
        annotation_path : str, optional
            Local annotation file (always highest priority, any source).
        annotation_source : str
            'auto' (default — prefer the dataset's local annotation_file, same
            as benchmark SHAP) | 'mygene' | 'gpl' | 'none' — probe→gene strategy.
        organism : str
            'hsapiens' | 'mmusculus' (or 'human' / 'mouse') — resolves the
            species used for probe→gene annotation.

        Returns
        -------
        dict with keys: top_features, mapped_genes, shap_values, rules
        """

        report.section(
            "Biological Interpretation",
            f"{dataset_name} · {fs_method} · {model_name} · fold {fold}",
        )

        ds_cfg = self.config_loader.get_dataset_config(dataset_name)

        data = load_dataset(
            dataset_path=Path(ds_cfg["path"]),
            dataset_type=ds_cfg.get("type", "auto"),
        )

        # Skips silently if the benchmark already wrote this provenance log.
        provenance_path = self.output_root / dataset_name / "DATASET_PROVENANCE.md"
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        update_dataset_description(data, description_path=provenance_path)

        X_full = data["X"]
        y = data["y"]
        feature_names = data["feature_names"]

        model_path = (
            self.output_root
            / dataset_name
            / fs_method
            / model_name
            / "models"
            / f"{model_name}_fold_{fold}.joblib"
        )

        if not model_path.exists():
            report.err(f"Model not found: {model_path}")
            report.info("Run run_benchmark() first, then call this method.")
            return {}

        import joblib
        artifact = joblib.load(model_path)
        model = artifact["model"]
        scaler = artifact["scaler"]

        # Resolve the PROBE-level matrix the model was trained on:
        #   raw  → original on-disk dataset (zero-copy pointer).
        #   else → the FS-reduced probe CSV written by the benchmark.
        interp_dir = self.output_root / dataset_name / fs_method / "interpretation"
        interp_dir.mkdir(parents=True, exist_ok=True)

        if fs_method == "raw":
            report.step("Loading raw baseline from source dataset (pointer)")
            feature_names_reduced = list(feature_names)
            X_reduced = X_full
        else:
            fs_csv_dir = (
                self.output_root
                / "feature_selection"
                / dataset_name
                / fs_method
                / "processed_datasets"
            )
            report.step(f"Loading reduced dataset from: {fs_csv_dir}")
            loaded = load_fs_reduced_matrix(fs_csv_dir)
            if loaded is None:
                report.err(f"No processed dataset found at: {fs_csv_dir}")
                return {}
            X_reduced, feature_names_reduced = loaded

        # Probe-level scaled matrix the model expects — no collapse before SHAP.
        X_scaled = apply_scaler(scaler, X_reduced)

        top_k = self.config_loader.get_shap_config().get("top_k", 20)

        # Shared resolver: prefers the dataset's local annotation_file so the
        # gene names match benchmark SHAP; annotation_source/annotation_path/
        # platform still override. An empty map leaves interpretation at probe IDs.
        species = "mouse" if organism in ("mmusculus", "mouse") else "human"
        report.step("Resolving probe -> gene mapping")
        mapping = self._build_probe_gene_map(
            ds_cfg, feature_names_reduced,
            annotation_source=annotation_source or "auto",
            annotation_path=annotation_path,
            platform=platform,
            species=species,
        )

        # SHAP is produced by the benchmark and saved per (fs × model);
        # interpretation reads that log directly rather than recomputing it.
        # With shap.on_best_benchmark, only the winning combo has a SHAP log
        # — interpret that combo, or run pipe.run_shap(dataset, fs, model)
        # first to generate the log for any other combo.
        shap_matrix = (
            self.output_root / dataset_name / fs_method / model_name
            / "shap" / "text" / "shap_feature_importance_matrix.csv"
        )
        if not shap_matrix.exists():
            report.err(f"No benchmark SHAP log for {fs_method} × {model_name}: {shap_matrix}")
            report.info(
                "SHAP is computed during the benchmark. Run run_benchmark() first. "
                "If this combo wasn't the benchmark winner, interpret the best combo "
                "(see select_best_model) or run run_shap(dataset, fs_method, model) first."
            )
            return {}

        report.step("Loading benchmark SHAP top features")
        imp_df = pd.read_csv(shap_matrix)
        top_features = imp_df["Feature_Gene"].head(top_k).tolist()
        shap_values = None
        report.ok(f"Reused top-{len(top_features)} features from benchmark SHAP")
        # top_features are Gene Symbols when mapping resolved, else probe IDs.

        # Rules come from the dedicated tree-based extractor (run_rule_extraction),
        # which writes dataset-level rules to outputs/{dataset}/rules/{rf,dt,compare}/;
        # loaded here for display only — this function does not mine rules.
        report.step("Loading tree-based rules")
        rules_df = self._load_tree_rules(dataset_name)
        if rules_df.empty:
            report.info(
                f"No tree-based rules found for {dataset_name} — run "
                "run_rule_extraction() to generate them "
                f"(outputs/{dataset_name}/rules/). SHAP still complete."
            )
        else:
            report.ok(
                f"Loaded {len(rules_df)} tree-based rule(s) "
                f"from outputs/{dataset_name}/rules/"
            )

        report.ok(f"Interpretation complete → {interp_dir}")

        return {
            "top_features": top_features,
            "mapped_genes": list(top_features),
            "shap_values": shap_values,
            "rules": rules_df,
        }

    def _load_tree_rules(self, dataset_name: str) -> pd.DataFrame:
        """Load tree-based rules written by :meth:`run_rule_extraction`.

        Reads ``outputs/{dataset}/rules/{model}/rules.csv`` for each rule model
        (``rf`` preferred, then ``dt``), tagging each row with its ``source_model``.
        Returns an empty DataFrame if rule extraction has not been run.
        """
        rules_root = self.output_root / dataset_name / "rules"
        frames: List[pd.DataFrame] = []
        for folder in ("rf", "dt"):
            csv_path = rules_root / folder / "rules.csv"
            if not csv_path.exists():
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                report.warn(f"Could not read {csv_path}: {e}")
                continue
            if df.empty:
                continue
            df.insert(0, "source_model", folder)
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def select_best_model(
        self,
        dataset_name: str,
        metric: str = "balanced_accuracy",
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best (fs_method, model, fold) for a dataset from benchmark outputs.

        Scans ``outputs/{dataset}/{fs}/{model}/cv_summary.csv`` for every trained
        combination and picks the one with the highest mean ``metric``. The best
        fold is read from that combination's ``cv_results.csv`` (default fold 1).

        Returns
        -------
        dict with keys ``fs_method, model_name, fold, score`` — or ``None`` if no
        benchmark results exist for this dataset yet.
        """
        ds_root = self.output_root / dataset_name
        best: Optional[Dict[str, Any]] = None

        for summary_path in sorted(ds_root.glob("*/*/cv_summary.csv")):
            model_name = summary_path.parent.name
            fs_method = summary_path.parent.parent.name
            try:
                summary_df = pd.read_csv(summary_path)
                row = summary_df.loc[summary_df["metric"] == metric]
                if row.empty:
                    continue
                score = float(row["mean"].iloc[0])
            except Exception:
                continue

            if best is None or score > best["score"]:
                fold = self._best_fold(summary_path.parent, metric)
                best = {
                    "fs_method": fs_method,
                    "model_name": model_name,
                    "fold": fold,
                    "score": score,
                }

        return best

    @staticmethod
    def _best_fold(model_dir: Path, metric: str) -> int:
        """Return the 1-based fold with the highest ``metric`` (default 1)."""
        cv_results = model_dir / "cv_results.csv"
        if cv_results.exists():
            try:
                df = pd.read_csv(cv_results)
                if metric in df.columns and "fold" in df.columns:
                    return int(df.loc[df[metric].idxmax(), "fold"])
            except Exception:
                pass
        return 1

    def run_interpretation_batch(
        self,
        dataset_names: Optional[List[str]] = None,
        fs_method: Optional[str] = None,
        model_name: Optional[str] = None,
        fold: Optional[int] = None,
        select: str = "best",
        platform: Optional[str] = None,
        annotation_path: Optional[str] = None,
        annotation_source: str = "mygene",
        organism: str = "hsapiens",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run interpretation for several datasets (the "multiple" run mode).

        Parameters
        ----------
        dataset_names : list, optional
            Datasets to interpret (default: all enabled in datasets.yaml).
        select : {"best", "fixed"}
            ``"best"`` auto-selects the top (fs_method, model, fold) per dataset
            from benchmark results via :meth:`select_best_model`. ``"fixed"`` uses
            the explicit ``fs_method`` / ``model_name`` / ``fold`` for every dataset.
        fs_method, model_name, fold :
            Explicit target (used when ``select="fixed"``, or as a fallback).
        platform, annotation_path, organism :
            Forwarded to :meth:`run_interpretation`.

        Returns
        -------
        dict ``{dataset_name: interpretation_result}``.
        """
        dataset_names = dataset_names or self.config_loader.get_enabled_datasets()

        report.section(
            "Biological Interpretation (batch)",
            f"{len(dataset_names)} dataset(s) · select={select}",
        )

        results: Dict[str, Dict[str, Any]] = {}
        summary_rows = []

        for ds_name in dataset_names:
            if select == "best":
                best = self.select_best_model(ds_name)
                if best is None:
                    report.warn(f"{ds_name}: no benchmark results found — skipping.")
                    continue
                fs, model, fld = best["fs_method"], best["model_name"], best["fold"]
                report.info(
                    f"{ds_name}: best = {fs}/{model} fold {fld} "
                    f"(balanced_accuracy={best['score']:.4f})"
                )
            else:
                fs = fs_method or "mrmr_mid"
                model = model_name or "rf"
                fld = fold or 1

            interp = self.run_interpretation(
                dataset_name=ds_name,
                fs_method=fs,
                model_name=model,
                fold=fld,
                platform=platform,
                annotation_path=annotation_path,
                annotation_source=annotation_source,
                organism=organism,
            )
            results[ds_name] = interp

            summary_rows.append(
                {
                    "dataset": ds_name,
                    "fs_method": fs,
                    "model": model,
                    "fold": fld,
                    "top_features": len(interp.get("top_features", [])),
                    "rules": len(interp.get("rules", [])),
                }
            )

        if summary_rows:
            report.subsection("Batch interpretation summary")
            report.dataframe_table(
                pd.DataFrame(summary_rows),
                title="Interpretation (batch) results",
                index=False,
            )

        return results
