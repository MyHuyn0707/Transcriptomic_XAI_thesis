"""
src/pipeline/rule_extraction.py

Tree-based classification rule extraction (stage 4-5).

For each dataset, picks the feature-selection set with the best benchmark
score, trains dedicated shallow RF/DecisionTree models on it (refit on all
samples), walks every root->leaf path into scored IF-THEN rules, simplifies
and cross-checks them. Config: feature_selection.yaml -> interpretation.rules.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.helper import report
from src.helper.data_loader import load_dataset, load_fs_reduced_matrix
from src.helper.scaling import apply_scaler, fit_scaler
from src.interpretation.rule_mining import (
    RULE_MODELS,
    fit_rule_model,
    mine_rules,
    save_rule_outputs,
    explain_rule_model,
)


class RuleExtractionMixin:
    """Dedicated rule-model training, IF-THEN rule extraction, cross-checking."""

    _RULE_MODELS = RULE_MODELS

    @staticmethod
    def _select_best_fs(
        dataset_results: Dict[str, Dict[str, pd.DataFrame]],
        metric: str,
        prefer_model: str = "rf",
    ) -> Optional[str]:
        """Pick the FS method with the highest benchmark score.

        Uses ``prefer_model`` (the tree benchmark, RF) to rank FS methods so the
        rule models are built on a validated feature set; falls back to the best
        score across any model if the preferred one was not benchmarked.
        """
        def _score(fs: str, model: str) -> float:
            df = dataset_results.get(fs, {}).get(model)
            if df is None or df.empty:
                return float("nan")
            return float(df.set_index("metric")["mean"].get(metric, float("nan")))

        best_fs, best_score = None, float("-inf")
        for fs, models in dataset_results.items():
            score = _score(fs, prefer_model)
            if score != score:  # NaN → preferred model absent, use best of any
                scores = [_score(fs, m) for m in models]
                scores = [s for s in scores if s == s]
                score = max(scores) if scores else float("-inf")
            if score > best_score:
                best_fs, best_score = fs, score
        return best_fs

    def _extract_rules_for_model(
        self,
        model_name: str,
        X_sel: np.ndarray,
        feature_names: List[str],
        y: np.ndarray,
        class_labels: List[str],
        probe_gene_map: Dict[str, str],
        rules_cfg: Dict[str, Any],
        output_dir: Path,
        shap_cfg: Optional[Dict[str, Any]] = None,
    ) -> Optional[pd.DataFrame]:
        """Fit one rule model, mine → save its rules, then explain it.

        Returns the simplified rules DataFrame (or None on failure) so the caller
        can cross-check models against each other.
        """
        params = dict(rules_cfg.get("model_params", {}).get(model_name, {}))

        report.step(f"Rule extraction ({model_name})")
        try:
            scaler = fit_scaler(model_name, X_sel)
            X_scaled = apply_scaler(scaler, X_sel)
            model = fit_rule_model(model_name, X_scaled, y, params)

            simplified, summary = mine_rules(
                model, X_sel, y, feature_names, scaler,
                class_labels, probe_gene_map, rules_cfg,
            )
            save_rule_outputs(
                simplified, summary, output_dir, model_name, rules_cfg, class_labels
            )
            explain_rule_model(
                model, X_scaled, feature_names, output_dir,
                class_labels, probe_gene_map, shap_cfg, model_name=model_name,
            )
            return simplified
        except Exception as e:
            report.warn(f"Rule extraction failed for {model_name}: {e}")
            return None

    def run_rule_extraction(
        self,
        dataset_names: Optional[List[str]] = None,
        metric: str = "balanced_accuracy",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract tree-based classification rules for each dataset (stage 4-5).

        Must be called **after** :meth:`run_benchmark` — it reads the
        benchmark CV results to choose the best feature-selection set. For each
        dataset it:

        1. Picks the FS method with the best benchmark score (by ``metric``,
           ranked on the RF benchmark).
        2. Trains the dedicated, shallow rule models configured in
           ``interpretation.rules.model_params`` (RF and/or DecisionTree) on
           that FS feature set, refit on **all** samples.
        3. Walks every root->leaf path, converting scaled thresholds back to raw
           expression, and scores each rule (support, confidence, lift,
           class_specificity, fidelity).
        4. Simplifies (merge same-gene conditions, filter, deduplicate, rank)
           and writes per model to ``outputs/{ds}/rules/{rf,dt}/``.
        5. When >=2 models are configured, cross-checks them and writes the
           confirmed-by-both rules to ``outputs/{ds}/rules/compare/``.

        Config: ``feature_selection.yaml -> interpretation.rules``.

        Parameters
        ----------
        dataset_names : list, optional
            Datasets to process (default: all enabled in datasets.yaml).
        metric : str
            Metric used to pick the best FS feature set (default
            ``balanced_accuracy``); overridden by ``rules.select_metric``.

        Returns
        -------
        dict ``{dataset_name: {"fs_method", "models": {name: summary}, "compare"}}``.
        """
        report.section(
            "Rule Extraction — Tree-Based Classification Rules",
            "Stage 4-5: dedicated RF/DecisionTree rule models per dataset",
        )

        dataset_names = dataset_names or self.config_loader.get_enabled_datasets()
        if not dataset_names:
            report.warn("No datasets enabled. Edit configs/datasets.yaml.")
            return {}

        rules_cfg = self.config_loader.get_rules_config()
        if not rules_cfg.get("enabled", True):
            report.warn("Rule extraction disabled (interpretation.rules.enabled = false).")
            return {}
        shap_cfg = self.config_loader.get_shap_config()

        metric = rules_cfg.get("select_metric", metric)
        rule_models = [
            m for m in rules_cfg.get("models", ["rf", "decisiontree"])
            if m in self._RULE_MODELS
        ]
        if not rule_models:
            report.warn(
                f"No valid rule models configured (supported: {', '.join(self._RULE_MODELS)})."
            )
            return {}

        all_results: Dict[str, Dict[str, Any]] = {}
        dt_aliases = {"decisiontree": "dt"}  # folder names: rf/, dt/

        for ds_name in dataset_names:
            report.subsection(f"Dataset: {ds_name}")
            try:
                ds_cfg = self.config_loader.get_dataset_config(ds_name)
                ds_root = self.output_root / ds_name

                # Load benchmark CV results to rank feature-selection methods.
                dataset_results: Dict[str, Dict[str, pd.DataFrame]] = {}
                for summ in sorted(ds_root.glob("*/*/cv_summary.csv")):
                    model_name = summ.parent.name
                    fs_method = summ.parent.parent.name
                    dataset_results.setdefault(fs_method, {})[model_name] = (
                        pd.read_csv(summ)
                    )
                if not dataset_results:
                    report.warn(
                        f"{ds_name}: no benchmark CV results found — "
                        "run run_benchmark() first."
                    )
                    continue

                best_fs = self._select_best_fs(dataset_results, metric, prefer_model="rf")
                if best_fs is None:
                    report.warn(f"{ds_name}: could not determine a best FS method.")
                    continue
                report.info(f"Best FS feature set: {best_fs} (by {metric})")

                # Load the dataset and the best FS feature matrix (raw scale).
                data = load_dataset(
                    dataset_path=Path(ds_cfg["path"]),
                    dataset_type=ds_cfg.get("type", "auto"),
                )
                y = data["y"]
                feature_names = data["feature_names"]

                # Real class names (aligned with encoded y) so rules read e.g.
                # "Class=Tumor" instead of "Class=1" for every dataset.
                class_labels = self._resolve_class_labels(data, ds_cfg)

                if best_fs == "raw":
                    X_sel, feat_sel = data["X"], list(feature_names)
                else:
                    csv_dir = (
                        self.output_root / "feature_selection"
                        / ds_name / best_fs / "processed_datasets"
                    )
                    loaded = load_fs_reduced_matrix(csv_dir)
                    if loaded is None:
                        report.warn(f"{ds_name}/{best_fs}: processed dataset missing.")
                        continue
                    X_sel, feat_sel = loaded

                probe_gene_map = self._build_probe_gene_map(ds_cfg, feature_names)

                # Extract rules per model into outputs/{ds}/rules/{rf,dt}/.
                rules_root = ds_root / "rules"
                model_rules: Dict[str, pd.DataFrame] = {}
                model_summaries: Dict[str, Any] = {}
                for model_name in rule_models:
                    folder = dt_aliases.get(model_name, model_name)
                    simplified = self._extract_rules_for_model(
                        model_name=model_name,
                        X_sel=X_sel,
                        feature_names=feat_sel,
                        y=y,
                        class_labels=class_labels,
                        probe_gene_map=probe_gene_map,
                        rules_cfg=rules_cfg,
                        output_dir=rules_root / folder,
                        shap_cfg=shap_cfg,
                    )
                    if simplified is not None:
                        model_rules[model_name] = simplified
                        model_summaries[model_name] = {
                            "n_rules": int(len(simplified)),
                            "n_rules_raw": int(simplified.attrs.get("n_raw", len(simplified))),
                            "n_rules_passed_filter": int(
                                simplified.attrs.get("n_passed_filter", len(simplified))
                            ),
                            "n_rules_kept": int(len(simplified)),
                            "per_class": {
                                str(k): int(v) for k, v in
                                simplified.groupby("consequent_label").size().items()
                            } if not simplified.empty else {},
                        }

                # Cross-check when at least two models produced rules.
                compare_info: Optional[Dict[str, Any]] = None
                if len(model_rules) >= 2:
                    compare_info = self._crosscheck_models(
                        model_rules, rules_root / "compare", best_fs
                    )

                all_results[ds_name] = {
                    "fs_method": best_fs,
                    "models": model_summaries,
                    "compare": compare_info,
                }
                self._report_rule_extraction_summary(ds_name, all_results[ds_name])

            except Exception as e:
                report.err(f"Rule extraction failed for {ds_name}: {e}")

        return all_results

    def describe_genes_in_rules(
        self,
        dataset_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Enrich the genes appearing in rules with platform biological annotation.

        A **separate** step (does not train or extract anything): it reads the
        ``genes_in_rules.csv`` files that :meth:`run_rule_extraction` wrote under
        ``outputs/{dataset}/rules/{rf,dt}/`` and writes ``gene_description.csv`` +
        ``gene_description.md`` next to each, joining every gene to its platform
        annotation (Gene title, Entrez ID, GenBank, chromosome, GO BP/CC/MF) via
        the gene's backing probe IDs. Runs on already-produced rule outputs — no
        re-run of extraction needed. The platform file is the dataset's
        ``annotation_file`` in datasets.yaml (NBCI ``.annot`` or CuMiDa ``_limpo``).

        Returns ``{dataset: {model_folder: {"n_genes", "n_annotated", "paths"}}}``.
        """
        from src.interpretation.gene_annotation import load_platform_annotation, describe_gene_files

        report.section(
            "Gene Descriptions — Biological Annotation of Rule Genes",
            "Enrich genes_in_rules with platform annotation (Gene title, GO, …)",
        )
        dataset_names = dataset_names or self.config_loader.get_enabled_datasets()
        results: Dict[str, Dict[str, Any]] = {}

        for ds_name in dataset_names:
            report.subsection(f"Dataset: {ds_name}")
            try:
                ds_cfg = self.config_loader.get_dataset_config(ds_name)
            except Exception as e:
                report.err(f"{ds_name}: config error: {e}")
                continue

            gene_files = sorted(
                (self.output_root / ds_name / "rules").glob("*/genes_in_rules.csv")
            )
            if not gene_files:
                report.warn(
                    f"{ds_name}: no genes_in_rules.csv under "
                    f"outputs/{ds_name}/rules/ — run run_rule_extraction() first."
                )
                continue

            annotation_file = ds_cfg.get("annotation_file")
            if not annotation_file or not Path(annotation_file).exists():
                report.warn(
                    f"{ds_name}: annotation_file missing "
                    f"({annotation_file}); cannot describe genes."
                )
                continue

            annotation_df = load_platform_annotation(annotation_file)
            if annotation_df.empty:
                report.warn(f"{ds_name}: empty platform annotation; skipping.")
                continue

            platform = str(ds_cfg.get("platform", ""))
            results[ds_name] = describe_gene_files(gene_files, annotation_df, ds_name, platform)

        return results

    def _crosscheck_models(
        self,
        model_rules: Dict[str, pd.DataFrame],
        output_dir: Path,
        fs_method: str,
    ) -> Dict[str, Any]:
        """Compare two models' rule sets and write the confirmed-by-both rules.

        Rules sharing the same antecedent gene set and consequent class across
        models are the highest-trust set, saved to ``compare/crosscheck.json``.
        """
        from src.interpretation.rules import compare_rule_sets

        names = list(model_rules.keys())
        a, b = names[0], names[1]
        result = compare_rule_sets(model_rules[a], model_rules[b], name_a=a, name_b=b)

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "crosscheck.json").write_text(
            json.dumps({"fs_method": fs_method, "model_a": a, "model_b": b, **result},
                       indent=2),
            encoding="utf-8",
        )
        report.ok(
            f"Cross-check {a} vs {b}: {result['n_overlap']} identical rule(s), "
            f"{result['literal_overlap']} shared gene-direction literal(s) "
            f"(literal Jaccard={result['literal_jaccard']:.2f}) -> {output_dir}"
        )
        return result

    @staticmethod
    def _report_rule_extraction_summary(
        dataset_name: str, info: Dict[str, Any]
    ) -> None:
        """Print a per-model rule count table for one dataset."""
        rows = []
        for model_name, summ in info.get("models", {}).items():
            kept = summ.get("n_rules_kept", summ.get("n_rules", 0))
            row = {
                "model": model_name,
                "raw": summ.get("n_rules_raw", kept),               # (A) before any filter
                "passed_filter": summ.get("n_rules_passed_filter", kept),  # (B) before cap
                "kept": kept,                                       # (C) final
            }
            row.update({f"cls:{k}": v for k, v in summ.get("per_class", {}).items()})
            rows.append(row)
        if rows:
            report.dataframe_table(
                pd.DataFrame(rows),
                title=f"{dataset_name} — extracted rules (FS: {info.get('fs_method')})",
                index=False,
            )
        else:
            report.warn(f"{dataset_name}: no rules extracted.")
        cmp = info.get("compare")
        if cmp:
            report.info(
                f"Cross-check: {cmp['n_overlap']} identical rule(s), "
                f"{cmp['literal_overlap']} shared gene-direction literal(s) "
                f"(literal Jaccard={cmp['literal_jaccard']:.2f})"
            )
