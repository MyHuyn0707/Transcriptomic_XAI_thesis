"""
src/config_loader.py

YAML configuration manager.

Paths in datasets.yaml are resolved relative to the project root
(the directory that contains the configs/ folder), so the pipeline
works regardless of the current working directory.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.helper.paths import resolve_path


class ConfigLoader:
    """
    Load and validate YAML configuration files.

    Parameters
    ----------
    config_root : str | Path
        Directory containing datasets.yaml, models.yaml, feature_selection.yaml.
        May be an absolute path or relative to the caller's working directory.
    """

    def __init__(self, config_root: str | Path = "configs") -> None:

        self.config_root = Path(config_root).resolve()

        if not self.config_root.exists():
            raise FileNotFoundError(
                f"Config directory not found: {self.config_root}"
            )

        # Project root = parent of configs/
        self.project_root = self.config_root.parent

    # ------------------------------------------------------------------
    # Raw YAML loading
    # ------------------------------------------------------------------

    def load_yaml(self, filename: str) -> Dict[str, Any]:
        filepath = self.config_root / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def load_datasets_config(self) -> Dict[str, Any]:
        return self.load_yaml("datasets.yaml")

    def load_dataset_origin_config(self) -> Dict[str, Any]:
        return self.load_yaml("dataset_origin.yaml")

    def load_models_config(self) -> Dict[str, Any]:
        return self.load_yaml("models.yaml")

    def load_feature_selection_config(self) -> Dict[str, Any]:
        return self.load_yaml("feature_selection.yaml")

    def load_interpretation_config(self) -> Dict[str, Any]:
        """Load configs/interpretation.yaml (SHAP, rules).

        Backward compatible: if the standalone file is absent, fall back to the
        legacy ``feature_selection.interpretation`` block so older configs keep
        working.
        """
        try:
            cfg = self.load_yaml("interpretation.yaml")
            if cfg.get("interpretation"):
                return cfg
        except FileNotFoundError:
            pass
        legacy = (
            self.load_feature_selection_config()
            .get("feature_selection", {})
            .get("interpretation", {})
        )
        return {"interpretation": legacy}

    # ------------------------------------------------------------------
    # Interpretation sub-configs (shap / rules)
    # ------------------------------------------------------------------

    def get_shap_config(self) -> Dict[str, Any]:
        return dict(
            self.load_interpretation_config()
            .get("interpretation", {})
            .get("shap", {})
        )

    def get_rules_config(self) -> Dict[str, Any]:
        return dict(
            self.load_interpretation_config()
            .get("interpretation", {})
            .get("rules", {})
        )

    # ------------------------------------------------------------------
    # Enabled-item helpers
    # ------------------------------------------------------------------

    def get_enabled_datasets(self) -> List[str]:
        """
        Return one run-key per (enabled dataset × level).

        A dataset with ``levels: ["probe", "gene"]`` expands to two run-keys:
        ``"<name>"`` (probe) and ``"<name>-gene"`` (gene) — preserving the
        external naming convention used by outputs/ folders.
        """
        config = self.load_datasets_config()
        run_keys: List[str] = []
        for name, cfg in config.get("datasets", {}).items():
            if not cfg.get("enabled", False):
                continue
            for level in cfg.get("levels", ["probe"]):
                run_keys.append(name if level == "probe" else f"{name}-{level}")
        return run_keys

    def get_enabled_models(self) -> List[str]:
        config = self.load_models_config()
        return [
            name
            for name, cfg in config.get("models", {}).items()
            if cfg.get("enabled", False)
        ]

    def get_enabled_fs_methods(self) -> List[str]:
        config = self.load_feature_selection_config()
        methods = (
            config.get("feature_selection", {})
            .get("benchmark", {})
            .get("methods", {})
        )
        return [
            name for name, cfg in methods.items() if cfg.get("enabled", False)
        ]

    # ------------------------------------------------------------------
    # Per-item config access
    # ------------------------------------------------------------------

    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """
        Resolve a run-key (e.g. ``"GEO-Breast-20711"`` or
        ``"GEO-Breast-20711-gene"``) to its dataset config.

        A ``-gene`` suffix selects the base entry's ``gene_path`` instead of
        ``probe_path``; any other run-key uses ``probe_path``.
        """
        config = self.load_datasets_config()
        datasets = config.get("datasets", {})

        if dataset_name in datasets:
            base_name, level = dataset_name, "probe"
        elif dataset_name.endswith("-gene") and dataset_name[: -len("-gene")] in datasets:
            base_name, level = dataset_name[: -len("-gene")], "gene"
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in datasets.yaml."
            )

        cfg = dict(datasets[base_name])

        # Resolve dataset path relative to project root
        raw_path = cfg.get(f"{level}_path")
        if not raw_path:
            raise ValueError(
                f"Dataset '{dataset_name}' is missing '{level}_path' in datasets.yaml."
            )
        resolved = self.project_root / raw_path
        cfg["path"] = str(resolved)
        cfg["path_resolved"] = resolved

        # Resolve other path-like keys (relative → project root) so they work
        # regardless of the caller's working directory (e.g. notebooks/).
        for key in ("annotation_file", "series_matrix", "mapping_file"):
            val = cfg.get(key)
            if val:
                cfg[key] = resolve_path(self.project_root, val)

        return cfg

    def get_origin_config(self, gse_id: str) -> Dict[str, Any]:
        """
        Look up one raw-GEO-series build spec from dataset_origin.yaml,
        with path-like keys resolved relative to the project root.
        """
        config = self.load_dataset_origin_config()
        origins = config.get("origins", {})
        if gse_id not in origins:
            raise ValueError(f"Origin '{gse_id}' not found in dataset_origin.yaml.")
        cfg = dict(origins[gse_id])

        for key in ("series_matrix", "annotation_file", "output_dir"):
            val = cfg.get(key)
            if val:
                cfg[key] = resolve_path(self.project_root, val)

        return cfg

    def get_cumida_origin_config(self, gse_id: str) -> Dict[str, Any]:
        """
        Look up one CuMiDa-format build spec from ``dataset_origin.yaml ->
        cumida_origins``, with path-like keys resolved relative to the
        project root.
        """
        config = self.load_dataset_origin_config()
        origins = config.get("cumida_origins", {})
        if gse_id not in origins:
            raise ValueError(f"CuMiDa origin '{gse_id}' not found in dataset_origin.yaml.")
        cfg = dict(origins[gse_id])

        for key in ("probe_csv", "annotation_file", "output_dir"):
            val = cfg.get(key)
            if val:
                cfg[key] = resolve_path(self.project_root, val)

        return cfg

    def get_model_hyperparams(self, model_name: str) -> Dict[str, Any]:
        config = self.load_models_config()
        if model_name not in config.get("models", {}):
            raise ValueError(
                f"Model '{model_name}' not found in models.yaml."
            )
        return dict(config["models"][model_name].get("hyperparams", {}))

    def get_fs_method_params(self, fs_method: str) -> Dict[str, Any]:
        config = self.load_feature_selection_config()
        methods = (
            config.get("feature_selection", {})
            .get("benchmark", {})
            .get("methods", {})
        )
        if fs_method not in methods:
            raise ValueError(
                f"FS method '{fs_method}' not found in feature_selection.yaml."
            )
        return dict(methods[fs_method].get("params", {}))

    def get_cv_config(self) -> Dict[str, Any]:
        config = self.load_models_config()
        return dict(config.get("cross_validation", {}))

    def get_write_benchmark_provenance(self) -> bool:
        config = self.load_models_config()
        return bool(config.get("training", {}).get("write_benchmark_provenance", True))

    def get_wandb_project(self) -> Optional[str]:
        config = self.load_models_config()
        training = config.get("training", {})
        if not training.get("use_wandb", False):
            return None
        return training.get("wandb_project", None)

    def _get_wandb_project_for(self, flow_key: str) -> Optional[str]:
        """W&B project for a specific flow (e.g. 'baseline_split', 'holdout')
        — falls back to the main wandb_project when wandb_project_<flow_key>
        isn't set."""
        config = self.load_models_config()
        training = config.get("training", {})
        if not training.get("use_wandb", False):
            return None
        return training.get(f"wandb_project_{flow_key}") or training.get("wandb_project")

    def get_wandb_project_baseline_split(self) -> Optional[str]:
        return self._get_wandb_project_for("baseline_split")

    def get_wandb_project_holdout(self) -> Optional[str]:
        return self._get_wandb_project_for("holdout")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_configs(self) -> bool:
        try:
            self.load_datasets_config()
            self.load_models_config()
            self.load_feature_selection_config()

            enabled_datasets = self.get_enabled_datasets()
            enabled_models = self.get_enabled_models()
            enabled_fs = self.get_enabled_fs_methods()

            if not enabled_datasets:
                warnings.warn("No datasets enabled in datasets.yaml.", stacklevel=2)
            if not enabled_models:
                warnings.warn("No models enabled in models.yaml.", stacklevel=2)
            if not enabled_fs:
                warnings.warn(
                    "No feature selection methods enabled in feature_selection.yaml.",
                    stacklevel=2,
                )

            # Check dataset paths exist
            for ds_name in enabled_datasets:
                cfg = self.get_dataset_config(ds_name)
                path = Path(cfg["path"])
                if not path.exists():
                    warnings.warn(
                        f"Dataset path not found for '{ds_name}': {path}",
                        stacklevel=2,
                    )

            return True

        except Exception as e:
            warnings.warn(f"Config validation failed: {e}", stacklevel=2)
            return False

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_summary(self) -> None:
        import pandas as pd

        from . import report

        report.section(
            "Configuration Summary",
            "Classification Transcriptomic with XAI",
        )

        enabled_datasets = self.get_enabled_datasets()
        enabled_models = self.get_enabled_models()
        enabled_fs = self.get_enabled_fs_methods()

        report.info(f"Project root : {self.project_root}")
        report.info(f"Config root  : {self.config_root}")

        # Datasets table (with file-exists status)
        rows = []
        for ds in enabled_datasets:
            cfg = self.get_dataset_config(ds)
            rows.append(
                {
                    "status": "✓" if Path(cfg["path"]).exists() else "✗ missing",
                    "dataset": ds,
                    "description": cfg.get("description", ""),
                }
            )
        if rows:
            report.dataframe_table(
                pd.DataFrame(rows),
                title=f"Datasets ({len(enabled_datasets)} enabled)",
                index=False,
            )
        else:
            report.warn("No datasets enabled. Edit configs/datasets.yaml.")

        cv = self.get_cv_config()
        report.info(f"Models ({len(enabled_models)}): " + ", ".join(enabled_models))
        report.info(
            f"Feature selection ({len(enabled_fs)}): " + ", ".join(enabled_fs)
        )
        report.info(
            f"Cross-validation : max {cv.get('n_splits', 5)}-fold stratified"
        )
        report.info(f"W&B project      : {self.get_wandb_project() or 'disabled'}")
