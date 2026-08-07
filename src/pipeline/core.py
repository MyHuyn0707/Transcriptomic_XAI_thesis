"""
src/pipeline/core.py

GeneExpressionPipeline — end-to-end pipeline for transcriptomic classification
with XAI. Combines the benchmark, rule-extraction, holdout and interpretation
mixins into one class, and holds the state and helpers shared by all of them:
config loading, output root, W&B project, class-label resolution, and
probe→gene mapping.

Usage from notebook
--------------------
    import sys; sys.path.insert(0, '..')   # repo root
    from src.pipeline import GeneExpressionPipeline

    pipe = GeneExpressionPipeline()
    pipe.run_benchmark()                          # all datasets
    pipe.run_benchmark(['GEO-20711'])             # one dataset
    pipe.run_baseline_split(['GEO-20711'], k_values=[2,3,4,5,6])  # train/test-split baseline, all models
    pipe.run_shap('GEO-20711', 'boruta', 'svm')          # explain any combo
    pipe.run_rule_extraction(['GEO-20711'])              # rules + SHAP
    pipe.run_rule_extraction_holdout(['GEO-20711'])      # rules on a train/test split
    pipe.run_interpretation('GEO-20711', 'mrmr_mid', 'rf')

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force UTF-8 output on Windows so Unicode in print() doesn't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - very old interpreters
        pass

import pandas as pd

from src.helper import report
from src.helper.config_loader import ConfigLoader
from src.dataset_builder.annotation import annotate_features

from .benchmark import BenchmarkMixin
from .baseline_split import BaselineSplitMixin
from .rule_extraction import RuleExtractionMixin
from .holdout import HoldoutMixin
from .interpretation import InterpretationMixin

warnings.filterwarnings("ignore")


class GeneExpressionPipeline(
    BenchmarkMixin, BaselineSplitMixin, RuleExtractionMixin, HoldoutMixin, InterpretationMixin,
):
    """
    End-to-end pipeline for transcriptomic classification with XAI.

    Parameters
    ----------
    config_root : str | Path
        Directory containing YAML configs.
    output_root : str | Path
        Root directory for all outputs.
    use_wandb : bool
        Whether to log to Weights & Biases.
    """

    def __init__(
        self,
        config_root: str | Path = "configs",
        output_root: str | Path = "outputs",
        use_wandb: bool = True,
    ) -> None:

        self.config_loader = ConfigLoader(config_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        # Resolve W&B project(s) from config (config can disable it). Three
        # separate projects — one per pipeline stage — so each stage's runs
        # don't mix together in the W&B UI: run_benchmark uses
        # self.wandb_project, run_baseline_split uses
        # self.wandb_project_baseline_split, run_rule_extraction_holdout uses
        # self.wandb_project_holdout. Each falls back to the main
        # wandb_project if its own key isn't set in configs/models.yaml.
        self.wandb_project: Optional[str] = (
            self.config_loader.get_wandb_project() if use_wandb else None
        )
        self.wandb_project_baseline_split: Optional[str] = (
            self.config_loader.get_wandb_project_baseline_split() if use_wandb else None
        )
        self.wandb_project_holdout: Optional[str] = (
            self.config_loader.get_wandb_project_holdout() if use_wandb else None
        )

        if self.wandb_project or self.wandb_project_baseline_split or self.wandb_project_holdout:
            try:
                from src.helper.wandb_utils import import_wandb
                wandb = import_wandb()
                wandb.login()
            except Exception as e:
                report.warn(f"W&B login failed: {e}. Continuing without W&B.")
                self.wandb_project = None
                self.wandb_project_baseline_split = None
                self.wandb_project_holdout = None

    @staticmethod
    def _resolve_class_labels(
        data: Dict[str, Any], ds_cfg: Dict[str, Any]
    ) -> List[str]:
        """Real class names for display (SHAP plots, rules), aligned with y.

        Prefers the dataset's ``LabelEncoder.classes_`` (always aligned with the
        encoded ``y``) so labels read e.g. "Tumor" instead of "1" for every
        dataset — no class_labels needed in datasets.yaml, and no risk of a
        config order disagreeing with the encoder. Falls back to the configured
        ``class_labels``, then to an empty list (callers then show integers).
        """
        enc = data.get("label_encoder") if data else None
        classes = getattr(enc, "classes_", None)
        if classes is not None and len(classes):
            return [str(c) for c in classes]
        return list(ds_cfg.get("class_labels", []) or [])

    def _build_probe_gene_map(
        self,
        ds_cfg: Dict[str, Any],
        feature_names: List[str],
        *,
        annotation_source: str = "auto",
        annotation_path: Optional[str] = None,
        platform: Optional[str] = None,
        species: str = "human",
    ) -> Dict[str, str]:
        """
        Resolve probe → gene symbols, preferring the dataset's local
        ``annotation_file`` (offline, complete) so benchmark SHAP and interpretation
        rules share identical gene names.

        Resolution order
        ----------------
        1. Explicit ``annotation_path`` (any source) — always wins.
        2. ``annotation_source='none'`` → empty map (keep probe IDs).
        3. Dataset's prebuilt ``mapping_file`` (dataset_builder output) — the
           canonical probe→gene map used to build the gene dataset. Preferred:
           no re-parsing, and SHAP names match the gene CSV exactly.
        4. Dataset's configured ``annotation_file`` (.annot/.annot.gz) — offline.
        5. Fallback: GPL SOFT download (source='gpl') or MyGene.info.
        """
        annotation_source = (annotation_source or "auto").lower()

        if annotation_path:
            try:
                return annotate_features(
                    feature_names=feature_names,
                    annotation_path=annotation_path,
                    species=species,
                ) or {}
            except Exception as e:
                report.warn(f"Annotation file failed ({e}); keeping probe IDs.")
                return {}

        if annotation_source == "none":
            return {}

        # Prefer the prebuilt mapping CSV (probID | gene_symbol | …) from
        # dataset_builder — reuse, don't recompute.
        mapping_file = ds_cfg.get("mapping_file")
        if (
            mapping_file
            and Path(mapping_file).exists()
            and annotation_source in ("auto", "mapping", "file")
        ):
            try:
                map_df = pd.read_csv(mapping_file, dtype=str)
                valid = map_df[map_df["gene_symbol"].fillna("").str.strip() != ""]
                mapping = dict(zip(valid["probID"], valid["gene_symbol"]))
                report.ok(
                    f"Reused prebuilt mapping: {len(mapping):,} probe→gene "
                    f"entries from {Path(mapping_file).name}"
                )
                return mapping
            except Exception as e:
                report.warn(f"mapping_file {mapping_file} failed ({e}); falling back.")

        annot_file = ds_cfg.get("annotation_file")
        if (
            annot_file
            and Path(annot_file).exists()
            and annotation_source in ("auto", "gpl", "annot", "file")
        ):
            try:
                from src.dataset_builder import build_mapping_table
                _, mapping = build_mapping_table(
                    list(feature_names), annot_file,
                    use_mygene_fallback=False, species=species,
                )
                return mapping
            except Exception as e:
                report.warn(f"annotation_file {annot_file} failed ({e}); falling back.")

        # Fallback: GPL SOFT (explicit gpl) or MyGene.info (platform=None).
        map_platform = (platform or ds_cfg.get("platform")) if annotation_source == "gpl" else None
        try:
            return annotate_features(
                feature_names=feature_names,
                platform=map_platform,
                species=species,
            ) or {}
        except Exception as e:
            report.warn(f"Annotation unavailable ({e}); keeping probe IDs.")
            return {}

    def print_summary(self) -> None:
        """Print pipeline configuration summary."""
        self.config_loader.print_summary()
