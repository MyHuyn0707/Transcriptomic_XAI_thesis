"""
src/benchmark/trainer.py

Stratified K-Fold cross-validation trainer for all models.

Key design decisions
--------------------
- Adaptive n_splits = min(5, smallest_class_count) to handle imbalanced datasets.
- StandardScaler fitted on training fold only — no leakage.
- W&B is fully optional; pass wandb_project=None to disable.
- Works uniformly for classical ML and ANN via the factory interface.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.helper import report
from src.helper.metrics import METRIC_COLUMNS, compute_metrics
from src.helper.scaling import apply_scaler, fit_scaler
from src.models.factory import get_model

# Optional W&B — imported lazily inside run_cv
_WANDB_AVAILABLE: Optional[bool] = None


def _try_import_wandb() -> bool:
    global _WANDB_AVAILABLE
    if _WANDB_AVAILABLE is None:
        try:
            import wandb  # noqa: F401
            _WANDB_AVAILABLE = True
        except ImportError:
            _WANDB_AVAILABLE = False
    return _WANDB_AVAILABLE


def _compute_n_splits(y: np.ndarray, max_splits: int = 5) -> int:
    """
    Adaptive n_splits: min(max_splits, smallest class count).

    Ensures every fold contains at least one sample of each class.
    Raises if any class has fewer than 2 samples.
    """
    counts = np.bincount(y)
    min_count = int(counts.min())
    if min_count < 2:
        raise ValueError(
            f"Class with only {min_count} sample(s) found. "
            "Remove classes with < 2 samples before training."
        )
    return min(max_splits, min_count)


class BenchmarkTrainer:
    """
    Stratified K-Fold cross-validation trainer.

    Supports all models from models.factory: nb, knn, svm, rf, xgboost, ann.

    Parameters
    ----------
    dataset_name : str
    model_name : str
    fs_method : str
        Feature selection method used (raw, mrmr, boruta).
        Used to organise the output directory.
    output_root : str | Path
    wandb_project : str | None
        W&B project name. Pass None to disable W&B logging entirely.
    seed : int
    needs_scaling_map : dict, optional
        Per-model scaling flag (see src/helper/scaling.py), typically
        configs/models.yaml -> scaling. Falls back to a hardcoded default
        when omitted.
    """

    def __init__(
        self,
        dataset_name: str,
        model_name: str,
        fs_method: str,
        output_root: str | Path,
        wandb_project: Optional[str] = None,
        seed: int = 42,
        needs_scaling_map: Optional[Dict[str, bool]] = None,
    ) -> None:

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.fs_method = fs_method
        self.seed = seed
        self.wandb_project = wandb_project
        self.needs_scaling_map = needs_scaling_map

        # Output: outputs/{dataset}/{fs_method}/{model}/
        self.output_dir = (
            Path(output_root) / dataset_name / fs_method / model_name
        )
        self.model_dir = self.output_dir / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._use_wandb = (wandb_project is not None) and _try_import_wandb()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_kwargs: Optional[Dict[str, Any]] = None,
        max_splits: int = 5,
        class_labels: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Run Stratified K-Fold cross-validation.

        Parameters
        ----------
        X : np.ndarray  (n_samples, n_features)
        y : np.ndarray  (n_samples,)
        feature_names : List[str]
        model_kwargs : dict, optional
            Hyperparameters forwarded to get_model().
        max_splits : int
            Upper bound on folds; actual value = min(max_splits, min_class_count).

        Returns
        -------
        pd.DataFrame
            Summary statistics (mean ± std) across folds.
        """

        model_kwargs = model_kwargs or {}
        n_splits = _compute_n_splits(y, max_splits)
        n_classes = len(np.unique(y))
        if not class_labels or len(class_labels) != n_classes:
            class_labels = [f"Class_{i}" for i in range(n_classes)]

        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.seed
        )

        fold_results: List[Dict[str, Any]] = []
        per_class_by_fold: Dict[int, Dict[str, Any]] = {}

        report.info(
            f"[{self.model_name.upper()}] {n_splits}-fold CV · "
            f"{X.shape[0]} samples × {X.shape[1]} features"
        )

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            fold = fold_idx + 1

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Scaler fitted only on training data (None when the model
            # doesn't need scaling, e.g. tree/boosting models — see
            # src/helper/scaling.py).
            scaler = fit_scaler(self.model_name, X_train, self.needs_scaling_map)
            X_train_s = apply_scaler(scaler, X_train)
            X_val_s = apply_scaler(scaler, X_val)

            # Model
            model = get_model(
                self.model_name,
                input_dim=X_train_s.shape[1],
                num_classes=n_classes,
                random_state=self.seed,
                **model_kwargs,
            )

            if self.model_name == "ann":
                model.fit(X_train_s, y_train, X_val=X_val_s, y_val=y_val)
            else:
                model.fit(X_train_s, y_train)

            y_pred = model.predict(X_val_s)
            y_prob = (
                model.predict_proba(X_val_s)
                if hasattr(model, "predict_proba")
                else None
            )

            fold_metrics = compute_metrics(
                y_val, y_pred, y_prob,
                labels=list(range(n_classes)), class_labels=class_labels,
            )
            fold_metrics["fold"] = fold
            # Kept out of fold_results (-> cv_results.csv) so that CSV stays
            # flat/scalar per column; per-class breakdown goes to its own
            # per_class_metrics.json instead (see _save_and_summarise).
            per_class_by_fold[fold] = fold_metrics.pop("per_class")
            fold_results.append(fold_metrics)

            # Log to W&B if enabled
            if self._use_wandb:
                self._wandb_log_fold(
                    fold, fold_metrics, y_val, y_pred, y_prob,
                    class_labels, feature_names, model,
                )

            # Save model artifact
            model_path = self.model_dir / f"{self.model_name}_fold_{fold}.joblib"
            joblib.dump({"model": model, "scaler": scaler}, model_path)

            report.console.print(
                f"      fold {fold}/{n_splits} · "
                f"acc={fold_metrics['accuracy']:.4f} · "
                f"f1_macro={fold_metrics['f1_macro']:.4f} · "
                f"bal_acc={fold_metrics['balanced_accuracy']:.4f}",
                style="dim",
            )

        return self._save_and_summarise(fold_results, n_splits, model_kwargs, per_class_by_fold)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_and_summarise(
        self,
        fold_results: List[Dict[str, Any]],
        n_splits: int,
        model_kwargs: Dict[str, Any],
        per_class_by_fold: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """Aggregate fold results, save CSVs and params JSON."""

        results_df = pd.DataFrame(fold_results)

        if per_class_by_fold:
            with open(self.output_dir / "per_class_metrics.json", "w") as f:
                json.dump(_serialize(per_class_by_fold), f, indent=2)

        metric_cols = METRIC_COLUMNS
        summary_rows = []
        for col in metric_cols:
            if col in results_df.columns and results_df[col].notnull().any():
                summary_rows.append(
                    {
                        "metric": col,
                        "mean": float(results_df[col].mean()),
                        "std": float(results_df[col].std()),
                    }
                )

        summary_df = pd.DataFrame(summary_rows)

        # cv_results.csv — one row per fold
        results_df.to_csv(self.output_dir / "cv_results.csv", index=False)

        # cv_summary.csv — mean ± std
        summary_df.to_csv(self.output_dir / "cv_summary.csv", index=False)

        # params_des.json
        params = {
            "framework": "Classification Transcriptomic with XAI",
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "fs_method": self.fs_method,
            "n_splits": n_splits,
            "random_state": self.seed,
            "model_hyperparams": _serialize(model_kwargs),
        }
        with open(self.output_dir / "params_des.json", "w") as f:
            json.dump(params, f, indent=4)

        # W&B summary run
        if self._use_wandb:
            self._wandb_log_summary(results_df, summary_df)

        report.info(f"saved → {self.output_dir}")
        return summary_df

    def _wandb_log_fold(
        self,
        fold: int,
        metrics: Dict[str, Any],
        y_val: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        class_labels: List[str],
        feature_names: List[str],
        model: Any,
    ) -> None:
        """Log a single fold to W&B."""
        import wandb

        run = wandb.init(
            project=self.wandb_project,
            name=f"{self.dataset_name}_{self.fs_method}_{self.model_name}_fold{fold}",
            group=f"{self.dataset_name}_{self.fs_method}_{self.model_name}",
            job_type="train_fold",
            reinit=True,
        )
        try:
            log_data = {k: v for k, v in metrics.items()
                        if k not in ("confusion_matrix", "fold", "per_class") and v is not None}
            log_data["fold"] = fold
            run.log(log_data)

            run.log(
                {
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_val.tolist(),
                        preds=y_pred.tolist(),
                        class_names=class_labels,
                    )
                }
            )

            if y_prob is not None:
                run.log(
                    {"roc_curve": wandb.plot.roc_curve(y_val, y_prob, labels=class_labels)}
                )
                run.log(
                    {"pr_curve": wandb.plot.pr_curve(y_val, y_prob, labels=class_labels)}
                )

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                top_idx = np.argsort(importances)[::-1][:20]
                data = [
                    [feature_names[i], float(importances[i])]
                    for i in top_idx
                ]
                table = wandb.Table(data=data, columns=["Feature", "Importance"])
                run.log(
                    {
                        "top20_feature_importance": wandb.plot.bar(
                            table, "Feature", "Importance", title="Top 20 Features"
                        )
                    }
                )
        finally:
            run.finish()

    def _wandb_log_summary(
        self,
        results_df: pd.DataFrame,
        summary_df: pd.DataFrame,
    ) -> None:
        """Upload summary table to W&B."""
        import wandb

        run = wandb.init(
            project=self.wandb_project,
            name=f"{self.dataset_name}_{self.fs_method}_{self.model_name}_SUMMARY",
            job_type="summary",
            reinit=True,
        )
        try:
            run.log(
                {
                    "cv_results": wandb.Table(dataframe=results_df),
                    "cv_summary": wandb.Table(dataframe=summary_df),
                }
            )
        finally:
            run.finish()


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _serialize(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj
