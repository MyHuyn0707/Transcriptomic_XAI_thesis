"""
src/pipeline/holdout.py

Rule extraction on a stratified train/test split (configs/holdout.yaml).

Reuses the split + feature-selection artifacts already produced by
run_baseline_split() (src/pipeline/baseline_split.py) for the same rare-class
k, then trains each rule model several times with a different seed per run
and keeps the run with the best test-set metric before mining rules. Also
used directly (per-model method only) by the live UI training job, which
supplies its own on-demand split.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.helper import report
from src.helper.data_loader import load_dataset
from src.helper.metrics import METRIC_COLUMNS, compute_metrics
from src.helper.paths import resolve_path
from src.helper.scaling import apply_scaler, fit_scaler
from src.helper.training import pick_best_run, summarize_repeats
from src.helper.wandb_utils import wandb_log_per_class
from src.benchmark.trainer import _serialize as _to_native
from src.interpretation.rules import evaluate_rules
from src.interpretation.rule_mining import (
    fit_rule_model,
    mine_rules,
    save_rule_outputs,
    explain_rule_model,
)
from src.visualize import plot_confusion_matrix


class HoldoutMixin:
    """Stratified train/test split, repeated rule-model training, best-by-metric selection."""

    def run_rule_extraction_holdout(
        self,
        dataset_names: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Rule extraction on a stratified train/test split.

        For each k in ``k_values`` (default ``holdout.rare_class_k_values``),
        for each dataset:

        1. Reads the split (train/test row indices) and feature-selection
           artifacts (``holdout.fs_methods`` — boruta/mrmr only) already
           written by ``run_baseline_split()`` (src/pipeline/baseline_split.py)
           under ``holdout.baseline_split_root/k{k}/{dataset}/`` for this same
           k — does NOT re-derive its own split or re-run FS (that flow is
           reporting-only and shared here to avoid duplicated compute; this
           method itself is unchanged and still used directly by the live UI
           training job, which supplies its own on-demand split).
        2. For each rule model (``holdout.models`` — rf, decisiontree): trains
           ``holdout.n_repeats`` times with a different model seed each run,
           evaluates on the held-out test set, and keeps the run with the
           best ``holdout.select_metric`` (default f1_macro). Both train- and
           test-set metrics of the winning run are recorded.
        3. Extracts rules from the winning model — mined and evaluated on the
           train split, plus a generalisation check on the test split
           (``rules_test_eval.csv``). Runs SHAP on the winning model
           (``interpretation.yaml -> shap.on_rule_model``).
        4. Cross-checks rf vs decisiontree rules; writes a provenance
           comparison keyed on ``select_metric``.
        5. Exports the held-out test set (full raw feature space, one CSV +
           one JSON per sample) for later single-sample UI testing.

        Rule-model hyperparameters, simplification filters and SHAP config are
        read from ``configs/interpretation.yaml``.

        Returns
        -------
        dict ``{k: {dataset_name: {fs_method: {"models": {...}, "compare": {...}}}}}``.
        """
        report.section(
            "Rule Extraction (Holdout) — Train/Test Split",
            "Reuses run_baseline_split()'s split + FS · repeated training · best-by-metric",
        )

        holdout_cfg = dict(self.config_loader.load_yaml("holdout.yaml").get("holdout", {}))
        if not holdout_cfg.get("enabled", True):
            report.warn("Holdout rule extraction disabled (holdout.enabled = false).")
            return {}

        # Same convention as ConfigLoader.get_dataset_config: relative paths
        # resolve against the project root (parent of configs/), not the
        # caller's working directory (e.g. notebooks/).
        holdout_root = Path(resolve_path(
            self.config_loader.project_root, holdout_cfg.get("output_root", "outputs_holdout")
        ))
        holdout_root.mkdir(parents=True, exist_ok=True)

        split_baseline_cfg = dict(holdout_cfg.get("split_baseline", {}))
        batch_cfg = dict(holdout_cfg.get("batch", {}))

        split_root = Path(resolve_path(
            self.config_loader.project_root,
            split_baseline_cfg.get("baseline_split_root", "outputs_baseline_split"),
        ))

        k_values = k_values or list(batch_cfg.get("rare_class_k_values", [4]))
        fs_methods = list(batch_cfg.get("fs_methods", ["boruta", "mrmr_k50", "mrmr_k75"]))
        n_repeats = int(batch_cfg.get("n_repeats", 10))
        base_seed = int(batch_cfg.get("repeat_base_seed", 0))
        select_metric = batch_cfg.get("select_metric", "f1_macro")
        rule_models = [
            m for m in batch_cfg.get("models", ["rf", "decisiontree"])
            if m in self._RULE_MODELS
        ]
        if not rule_models:
            report.warn(
                f"No valid holdout.models configured (supported: {', '.join(self._RULE_MODELS)})."
            )
            return {}

        rules_cfg = self.config_loader.get_rules_config()
        shap_cfg = self.config_loader.get_shap_config()

        dataset_names = dataset_names or self.config_loader.get_enabled_datasets()
        if not dataset_names:
            report.warn("No datasets enabled. Edit configs/datasets.yaml.")
            return {}

        dt_aliases = {"decisiontree": "dt"}
        all_k_results: Dict[int, Any] = {}

        for k in k_values:
            report.subsection(f"Rare-class k = {k}")
            k_split_root = split_root / f"k{k}"
            k_holdout_root = holdout_root / f"k{k}"
            k_holdout_root.mkdir(parents=True, exist_ok=True)
            all_results: Dict[str, Any] = {}

            for ds_name in dataset_names:
                report.subsection(f"Dataset: {ds_name} (k={k})")
                try:
                    ds_cfg = self.config_loader.get_dataset_config(ds_name)
                    split_ds_root = k_split_root / ds_name
                    split_info_path = split_ds_root / "split_info.json"
                    if not split_info_path.exists():
                        report.err(
                            f"{ds_name}: no split found at {split_info_path} — "
                            "run pipe.run_baseline_split(k_values=[...]) for this k first."
                        )
                        continue
                    split_info = json.loads(split_info_path.read_text(encoding="utf-8"))
                    train_idx = np.array(split_info["train_idx"], dtype=np.int64)
                    test_idx = np.array(split_info["test_idx"], dtype=np.int64)
                    class_labels = list(split_info.get("class_labels", []))

                    ds_root = k_holdout_root / ds_name
                    ds_root.mkdir(parents=True, exist_ok=True)

                    # Copy split_info.json (now including raw_class_counts/
                    # dropped_classes — see baseline_split.py) into this flow's
                    # own output root, since the API (src/api/registry.py) reads
                    # dataset overview/split data from holdout_root_for(), not
                    # from outputs_baseline_split. Keeps GET /overview and the
                    # UI's class-distribution / train-test-split charts working
                    # without duplicating the split+FS artifacts themselves.
                    (ds_root / "split_info.json").write_text(
                        split_info_path.read_text(encoding="utf-8"), encoding="utf-8"
                    )

                    # Reload with the same k — load_dataset's rare-class
                    # removal is deterministic given the same source file and
                    # k, so these rows line up with the indices saved above.
                    data = load_dataset(
                        dataset_path=Path(ds_cfg["path"]),
                        dataset_type=ds_cfg.get("type", "auto"),
                        min_samples_per_class=k,
                    )
                    X, y = data["X"], data["y"]
                    feature_names = data["feature_names"]
                    sample_ids = data["sample_ids"]
                    if not class_labels:
                        class_labels = self._resolve_class_labels(data, ds_cfg)

                    # test_set/ is generated once by run_baseline_split() (same
                    # train/test split, same raw feature space) — copy it here
                    # instead of re-deriving it, so both flows always agree.
                    split_test_set_dir = split_ds_root / "test_set"
                    if split_test_set_dir.exists():
                        shutil.copytree(split_test_set_dir, ds_root / "test_set", dirs_exist_ok=True)
                    else:
                        report.warn(f"{ds_name}: no test_set/ at {split_test_set_dir} (run_baseline_split outdated?)")

                    X_train, y_train = X[train_idx], y[train_idx]
                    X_test, y_test = X[test_idx], y[test_idx]

                    probe_gene_map = self._build_probe_gene_map(ds_cfg, feature_names)

                    ds_results: Dict[str, Any] = {}

                    for fs_method in fs_methods:
                        report.step(f"Reusing FS artifacts ({fs_method}, k={k})")
                        split_fs_dir = k_split_root / "feature_selection" / ds_name / fs_method
                        sel_json_path = split_fs_dir / "selected_features" / "selected_features.json"
                        if not sel_json_path.exists():
                            report.err(
                                f"{ds_name}/{fs_method}: no selected_features.json at "
                                f"{sel_json_path} — run run_baseline_split() for k={k} first."
                            )
                            continue
                        sel_json = json.loads(sel_json_path.read_text(encoding="utf-8"))
                        feat_sel = sel_json.get("selected_features") or []
                        if not feat_sel:
                            report.warn(f"{ds_name}/{fs_method}: empty selected_features; skipping.")
                            continue

                        idx_map = {f: i for i, f in enumerate(feature_names)}
                        sel_idx = np.array([idx_map[f] for f in feat_sel], dtype=np.int64)
                        X_train_sel = X_train[:, sel_idx]
                        X_test_sel = X_test[:, sel_idx]
                        report.ok(f"{fs_method}: {len(feat_sel)} features reused from split-baseline (k={k})")

                        # Copy the FS metadata/selected_features (not the
                        # train-only processed_datasets/ CSV, which nothing
                        # here reads) into this flow's own output root — the
                        # API's GET .../feature-selection/{fs_method} and the
                        # live predict path (src/api/inference.py) both read
                        # feature_selection/{dataset}/{fs_method}/ under
                        # holdout_root_for(), not under outputs_baseline_split.
                        holdout_fs_dir = k_holdout_root / "feature_selection" / ds_name / fs_method
                        for sub in ("params", "selected_features"):
                            src_sub = split_fs_dir / sub
                            if src_sub.exists():
                                shutil.copytree(src_sub, holdout_fs_dir / sub, dirs_exist_ok=True)

                        fs_root = ds_root / fs_method
                        model_summaries: Dict[str, Any] = {}
                        model_rules: Dict[str, pd.DataFrame] = {}

                        for model_name in rule_models:
                            folder = dt_aliases.get(model_name, model_name)
                            report.step(f"Rule model (holdout): {fs_method} × {model_name} (k={k})")
                            result = self._extract_rules_for_model_holdout(
                                model_name=model_name,
                                X_train=X_train_sel, y_train=y_train,
                                X_test=X_test_sel, y_test=y_test,
                                feature_names=feat_sel,
                                class_labels=class_labels,
                                probe_gene_map=probe_gene_map,
                                rules_cfg=rules_cfg,
                                shap_cfg=shap_cfg,
                                output_dir=fs_root / folder,
                                n_repeats=n_repeats,
                                base_seed=base_seed,
                                select_metric=select_metric,
                            )
                            if result is not None:
                                model_summaries[model_name] = result["summary"]
                                model_rules[model_name] = result["rules_df"]
                                self._wandb_log_holdout_model(
                                    ds_name, k, fs_method, model_name, result["summary"], n_repeats, class_labels,
                                )

                        compare_info: Optional[Dict[str, Any]] = None
                        if len(model_rules) >= 2:
                            compare_info = self._crosscheck_models(
                                model_rules, fs_root / "compare", fs_method
                            )

                        ds_results[fs_method] = {"models": model_summaries, "compare": compare_info}

                    all_results[ds_name] = ds_results
                    self._append_holdout_provenance(ds_results, ds_root, select_metric)

                except Exception as e:
                    report.err(f"Holdout rule extraction failed for {ds_name} (k={k}): {e}")

            all_k_results[k] = all_results
            self._wandb_log_holdout_k_comparison(k, all_results)

        return all_k_results

    def _wandb_log_holdout_model(
        self,
        dataset_name: str,
        k: int,
        fs_method: str,
        model_name: str,
        summary: Dict[str, Any],
        n_repeats: int,
        class_labels: List[str],
    ) -> None:
        """One W&B run per (dataset, k, fs_method, model) — train+test scalars,
        an interactive confusion matrix, and per-class precision/recall/f1,
        for the winning repeated-training run.

        Logged to its own project (training.wandb_project_holdout), separate
        from run_benchmark's and run_baseline_split's. ``config=`` carries
        dataset/k/fs_method/model/seed as real, filterable/groupable columns
        in the W&B UI — the run ``name``/``group`` strings are for human
        reading, not for W&B's own filter/group-by controls.
        """
        if not self.wandb_project_holdout:
            return
        try:
            import wandb
        except ImportError:
            return
        best_seed = summary.get("best_seed")
        run = wandb.init(
            project=self.wandb_project_holdout,
            name=f"{dataset_name}_k{k}_{fs_method}_{model_name}_seed{best_seed}_holdout",
            group=f"{dataset_name}_k{k}_{fs_method}_{model_name}",
            job_type="holdout_repeat",
            config={
                "flow": "holdout",
                "dataset": dataset_name,
                "k": k,
                "fs_method": fs_method,
                "model": model_name,
                "best_seed": best_seed,
                "n_repeats": n_repeats,
            },
            reinit=True,
        )
        try:
            log_data: Dict[str, Any] = {"k": k}
            for split_name, key in (("train", "best_run_train_metrics"), ("test", "best_run_test_metrics")):
                for mk, mv in summary.get(key, {}).items():
                    if mv is not None:
                        log_data[f"{split_name}_{mk}"] = mv
            run.log(log_data)

            y_test = summary.get("_y_test")
            y_pred_test = summary.get("_y_pred_test")
            if y_test is not None and y_pred_test is not None:
                run.log({
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None, y_true=list(y_test), preds=list(y_pred_test),
                        class_names=class_labels,
                    )
                })

            per_class = summary.get("_per_class")
            if per_class:
                wandb_log_per_class(run, per_class, class_labels)
        finally:
            run.finish()

    def _wandb_log_holdout_k_comparison(
        self,
        k: int,
        k_results: Dict[str, Any],
    ) -> None:
        """One summary W&B run per k — a table of every (dataset, fs_method,
        model) test-set row, for cross-model/fs comparison at this k."""
        if not self.wandb_project_holdout or not k_results:
            return
        try:
            import wandb
        except ImportError:
            return
        rows = []
        for ds_name, ds_results in k_results.items():
            for fs_method, info in ds_results.items():
                for model_name, summary in info.get("models", {}).items():
                    tm = summary.get("best_run_test_metrics", {})
                    rows.append([
                        ds_name, fs_method, model_name,
                        tm.get("accuracy"), tm.get("balanced_accuracy"),
                        tm.get("f1_macro"),
                        tm.get("precision"), tm.get("recall"),
                    ])
        if not rows:
            return
        run = wandb.init(
            project=self.wandb_project_holdout,
            name=f"holdout_k{k}_comparison",
            job_type="k_comparison",
            config={"flow": "holdout", "k": k},
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

    def _extract_rules_for_model_holdout(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        class_labels: List[str],
        probe_gene_map: Dict[str, str],
        rules_cfg: Dict[str, Any],
        shap_cfg: Dict[str, Any],
        output_dir: Path,
        n_repeats: int,
        base_seed: int,
        select_metric: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Train ``model_name`` ``n_repeats`` times (different ``random_state``
        each run, otherwise the hyperparameters configured in
        ``interpretation.yaml``) on train, evaluate on test, keep the run with
        the best ``select_metric``. Extracts rules from the winning model —
        mined and evaluated on train, plus a generalisation re-evaluation on
        test.

        Exports (under ``output_dir``)::

            models/  model_best.joblib, repeats.csv, repeats_summary.csv,
                     params_des.json, best_run.json,
                     confusion_matrix_test.png/.csv (winning run, test set)
            rules/   rules.json/.csv/.../rules_test_eval.csv, shap/

        ``repeats.csv``/``repeats_summary.csv`` are named for what this
        actually is — ``n_repeats`` runs of the SAME train/test split with a
        different model ``random_state`` each time, NOT k-fold
        cross-validation (there is no k-fold anywhere in this flow; compare
        ``src/benchmark/trainer.py``'s genuinely CV-based
        ``cv_results.csv``/``cv_summary.csv``, which this deliberately does
        not share a name with).

        Returns ``{"summary": rules_summary_dict, "rules_df": simplified_df}``
        or ``None`` if every run failed to fit.
        """
        import joblib
        import matplotlib.pyplot as plt

        params = dict(rules_cfg.get("model_params", {}).get(model_name, {}))

        models_dir = output_dir / "models"
        rules_dir = output_dir / "rules"
        models_dir.mkdir(parents=True, exist_ok=True)
        rules_dir.mkdir(parents=True, exist_ok=True)

        metric_cols = METRIC_COLUMNS

        # rf/decisiontree are both tree-based (NEEDS_SCALING=False) — see
        # src/helper/scaling.py; scaler will be None here.
        scaler = fit_scaler(model_name, X_train)
        X_train_s = apply_scaler(scaler, X_train)
        X_test_s = apply_scaler(scaler, X_test)

        run_rows: List[Dict[str, Any]] = []
        fitted_models: List[Any] = []

        for i in range(n_repeats):
            seed_i = base_seed + i
            run_params = dict(params)
            run_params["random_state"] = seed_i
            try:
                model = fit_rule_model(model_name, X_train_s, y_train, run_params)
            except Exception as e:
                report.warn(f"{model_name} run {i} (seed {seed_i}) failed to fit: {e}")
                continue

            y_pred_test = model.predict(X_test_s)
            y_prob_test = (
                model.predict_proba(X_test_s) if hasattr(model, "predict_proba") else None
            )
            m = compute_metrics(
                y_test, y_pred_test, y_prob_test,
                labels=list(range(len(class_labels))),
            )
            m.pop("per_class", None)  # per-repeat breakdown not persisted; only the winner's is
            m["run"] = i
            m["seed"] = seed_i
            run_rows.append(m)
            fitted_models.append(model)

            report.console.print(
                f"      run {i + 1}/{n_repeats} (seed {seed_i}) · "
                f"f1_macro={m['f1_macro']:.4f} · acc={m['accuracy']:.4f}",
                style="dim",
            )

        if not run_rows:
            report.warn(f"Rule extraction (holdout) aborted for {model_name}: every run failed to fit.")
            return None

        results_df = pd.DataFrame(run_rows)
        summary_df = summarize_repeats(results_df, metric_cols)

        csv_cols = ["run", "seed"] + [c for c in metric_cols if c in results_df.columns]
        results_df[csv_cols].to_csv(models_dir / "repeats.csv", index=False)
        summary_df.to_csv(models_dir / "repeats_summary.csv", index=False)

        select_metric_eff = select_metric
        if select_metric_eff not in results_df.columns or results_df[select_metric_eff].isnull().all():
            report.warn(f"select_metric '{select_metric}' unavailable; falling back to 'f1_macro'.")
            select_metric_eff = "f1_macro"

        best_pos, best_seed = pick_best_run(results_df, select_metric_eff, "f1_macro")
        best_row = results_df.loc[best_pos].to_dict()
        best_model = fitted_models[best_pos]

        joblib.dump({"model": best_model, "scaler": scaler}, models_dir / "model_best.joblib")

        # Train-set metrics for the winning model — computed once (not per
        # repeat) since only the winner's train performance is reported
        # alongside its test performance ("log train, test" for rule
        # extraction, no separate validation split). Test metrics are
        # recomputed here (with per_class, unlike the per-repeat `m` above)
        # purely to capture the winner's per-class breakdown.
        y_pred_train_best = best_model.predict(X_train_s)
        y_prob_train_best = (
            best_model.predict_proba(X_train_s) if hasattr(best_model, "predict_proba") else None
        )
        train_metrics_best = compute_metrics(
            y_train, y_pred_train_best, y_prob_train_best,
            labels=list(range(len(class_labels))), class_labels=class_labels,
        )
        y_pred_test_best = best_model.predict(X_test_s)
        y_prob_test_best = (
            best_model.predict_proba(X_test_s) if hasattr(best_model, "predict_proba") else None
        )
        test_metrics_best_full = compute_metrics(
            y_test, y_pred_test_best, y_prob_test_best,
            labels=list(range(len(class_labels))), class_labels=class_labels,
        )
        (models_dir / "best_run_train.json").write_text(
            json.dumps(_to_native({k: v for k, v in train_metrics_best.items() if k != "per_class"}), indent=2),
            encoding="utf-8",
        )
        (models_dir / "best_run_per_class.json").write_text(
            json.dumps(_to_native({
                "train": train_metrics_best["per_class"],
                "test": test_metrics_best_full["per_class"],
            }), indent=2),
            encoding="utf-8",
        )

        # random_state is overridden every run (base_seed + i), so the
        # configured base value would be misleading next to best_seed —
        # n_repeats/repeat_base_seed/best_seed cover it instead.
        hyperparams_snapshot = {k: v for k, v in params.items() if k != "random_state"}
        params_des = {
            "framework": "Classification Transcriptomic with XAI",
            "mode": "holdout_repeated_training",
            "model_name": model_name,
            "n_repeats": n_repeats,
            "repeat_base_seed": base_seed,
            "best_seed": best_seed,
            "select_metric": select_metric_eff,
            "model_hyperparams": _to_native(hyperparams_snapshot),
        }
        (models_dir / "params_des.json").write_text(
            json.dumps(_to_native(params_des), indent=2), encoding="utf-8"
        )
        (models_dir / "best_run.json").write_text(
            json.dumps(_to_native(best_row), indent=2), encoding="utf-8"
        )

        # Confusion matrix of the winning run, on the held-out test set.
        cm = np.asarray(best_row["confusion_matrix"])
        cm_labels = list(class_labels) if class_labels else [str(i) for i in range(cm.shape[0])]
        cm_fig = plot_confusion_matrix(
            cm, class_labels=cm_labels,
            title=f"{model_name} — confusion matrix (test, seed={best_seed})",
        )
        cm_fig.savefig(models_dir / "confusion_matrix_test.png", dpi=150, bbox_inches="tight")
        plt.close(cm_fig)
        pd.DataFrame(
            cm,
            index=[f"true_{c}" for c in cm_labels],
            columns=[f"pred_{c}" for c in cm_labels],
        ).to_csv(models_dir / "confusion_matrix_test.csv")

        report.ok(
            f"{model_name}: best run seed={best_seed} "
            f"{select_metric_eff}={best_row[select_metric_eff]:.4f} (of {len(run_rows)} run(s))"
        )

        try:
            simplified, summary = mine_rules(
                best_model, X_train, y_train, feature_names, scaler,
                class_labels, probe_gene_map, rules_cfg,
            )
            summary["best_seed"] = best_seed
            summary["best_run_test_metrics"] = {
                k: best_row[k] for k in metric_cols
                if k in best_row and best_row[k] == best_row[k]  # NaN-safe
            }
            summary["best_run_train_metrics"] = {
                k: train_metrics_best[k] for k in metric_cols
                if k in train_metrics_best and train_metrics_best[k] == train_metrics_best[k]
            }
            save_rule_outputs(
                simplified, summary, rules_dir, model_name, rules_cfg, class_labels
            )
            # Added AFTER save_rule_outputs (which JSON-serializes `summary`)
            # so these raw arrays never get written to rules_summary.json —
            # only consumed in-memory by _wandb_log_holdout_model below.
            summary["_y_test"] = y_test
            summary["_y_pred_test"] = y_pred_test_best
            summary["_per_class"] = {
                "train": train_metrics_best["per_class"],
                "test": test_metrics_best_full["per_class"],
            }

            if not simplified.empty:
                temp_df = pd.DataFrame({
                    "conditions": simplified["conditions_raw"].tolist(),
                    "class_idx": simplified["class_idx"].tolist(),
                })
                y_pred_test_best = best_model.predict(X_test_s)
                evaluated_test = evaluate_rules(
                    temp_df, X_test, y_test,
                    y_pred=y_pred_test_best, class_labels=class_labels or None,
                )
                test_eval_df = pd.DataFrame({
                    "antecedents": simplified["antecedents"].values,
                    "consequents": simplified["consequents"].values,
                    "n_conditions": simplified["n_conditions"].values,
                    "support_test": evaluated_test["support"].values,
                    "confidence_test": evaluated_test["confidence"].values,
                    "error_test": evaluated_test["error"].values,
                    "lift_test": evaluated_test["lift"].values,
                    "class_specificity_test": evaluated_test["class_specificity"].values,
                    "fidelity_test": evaluated_test["fidelity"].values,
                    "strength_score_test": evaluated_test["strength_score"].values,
                })
            else:
                test_eval_df = pd.DataFrame(columns=[
                    "antecedents", "consequents", "n_conditions", "support_test",
                    "confidence_test", "error_test", "lift_test",
                    "class_specificity_test", "fidelity_test", "strength_score_test",
                ])
            test_eval_df.to_csv(rules_dir / "rules_test_eval.csv", index=False)

            explain_rule_model(
                best_model, X_train_s, feature_names, rules_dir,
                class_labels, probe_gene_map, shap_cfg, model_name=model_name,
            )

            return {"summary": summary, "rules_df": simplified}

        except Exception as e:
            report.warn(f"Rule extraction (holdout) failed for {model_name}: {e}")
            return None

    @staticmethod
    def _append_holdout_provenance(
        ds_results: Dict[str, Any],
        ds_root: Path,
        metric: str,
    ) -> None:
        """
        Write ``{ds_root}/DATASET_PROVENANCE_HOLDOUT.md`` — a comparison table
        across fs_method × model (rf, dt), sorted/pivoted on ``metric``
        (f1_macro by default). Idempotent — the block between sentinels is
        replaced each run; content before/after it (e.g. the rare-class/split
        section appended earlier in the same run by
        ``update_dataset_description``) is preserved.
        """
        prov_path = ds_root / "DATASET_PROVENANCE_HOLDOUT.md"
        START = "<!-- HOLDOUT_COMPARISON_START -->"
        END = "<!-- HOLDOUT_COMPARISON_END -->"
        metric_cols = ["accuracy", "balanced_accuracy", "f1_macro", "precision", "recall"]

        detail: List[tuple] = []
        pivot: Dict[str, Dict[str, float]] = {}
        for fs_method, info in ds_results.items():
            for model_name, summary in info.get("models", {}).items():
                test_metrics = summary.get("best_run_test_metrics", {})
                vals = {c: float(test_metrics.get(c, float("nan"))) for c in metric_cols}
                detail.append((fs_method, model_name, vals))
                pivot.setdefault(fs_method, {})[model_name] = vals.get(metric, float("nan"))

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines: List[str] = [
            START, "",
            "## --- HOLDOUT RULE-MODEL COMPARISON (train/test split) ---", "",
            f"> Auto-generated by run_rule_extraction_holdout() on {ts}",
            f"> Selection metric: `{metric}` (test-set, best of N repeated trainings)",
            "",
        ]
        lines.extend(report.markdown_metric_table(
            detail, pivot, metric_cols, metric,
            detail_title="Each fs_method × model (test-set metrics of the winning run)",
        ))
        lines.append(END)

        block = "\n".join(lines)
        report.upsert_markdown_block(prov_path, block, START, END)
        report.ok(f"Holdout comparison → {prov_path.name}")

    def describe_genes_in_rules_holdout(
        self,
        dataset_names: Optional[List[str]] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Enrich the genes appearing in holdout rules with platform biological annotation.

        A separate step (does not train or extract anything): it reads the
        ``genes_in_rules.csv`` files that :meth:`run_rule_extraction_holdout` wrote
        under ``outputs_holdout/k{k}/{dataset}/{fs_method}/{rf,dt}/rules/`` and
        writes ``gene_description.csv`` + ``gene_description.md`` next to each —
        same biological columns as :meth:`describe_genes_in_rules` (Gene title,
        Entrez ID, GenBank, chromosome, GO BP/CC/MF), via :func:`describe_gene_files`.

        k : int, optional
            Which rare-class k's outputs to describe. Defaults to
            ``holdout.active_min_samples_per_class`` (the k the API/UI serves).

        Returns ``{dataset: {"{fs_method}/{rf,dt}": {"n_genes", "n_annotated", "paths"}}}``.
        """
        from src.interpretation.gene_annotation import load_platform_annotation, describe_gene_files

        report.section(
            "Gene Descriptions (Holdout) — Biological Annotation of Rule Genes",
            "Enrich genes_in_rules with platform annotation (Gene title, GO, …)",
        )

        holdout_cfg = dict(self.config_loader.load_yaml("holdout.yaml").get("holdout", {}))
        holdout_root = Path(resolve_path(
            self.config_loader.project_root, holdout_cfg.get("output_root", "outputs_holdout")
        ))
        k = k if k is not None else int(holdout_cfg.get("active_min_samples_per_class", 4))
        holdout_root = holdout_root / f"k{k}"

        dataset_names = dataset_names or self.config_loader.get_enabled_datasets()
        results: Dict[str, Dict[str, Any]] = {}

        for ds_name in dataset_names:
            report.subsection(f"Dataset: {ds_name}")
            try:
                ds_cfg = self.config_loader.get_dataset_config(ds_name)
            except Exception as e:
                report.err(f"{ds_name}: config error: {e}")
                continue

            ds_root = holdout_root / ds_name
            gene_files = sorted(ds_root.glob("*/*/rules/genes_in_rules.csv"))
            if not gene_files:
                report.warn(
                    f"{ds_name}: no genes_in_rules.csv under {ds_root} "
                    f"— run run_rule_extraction_holdout(k_values=[{k}]) first."
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
            # gf = .../{fs_method}/{rf,dt}/rules/genes_in_rules.csv
            label_fn = lambda gf: f"{gf.parent.parent.parent.name}/{gf.parent.parent.name}"
            results[ds_name] = describe_gene_files(
                gene_files, annotation_df, ds_name, platform, label_fn=label_fn
            )

        return results
