"""
src/interpretation/rule_mining.py

Fit dedicated rule-extraction tree models and mine/export/explain their rules —
the orchestration layer above the rule mechanics in ``rules.py``
(extract_rules_from_forest / evaluate_rules / simplify_rules / ruleset_summary /
save_rules) and the SHAP explainer in ``shap_utils.py``. Shared by every caller
that needs "fit a shallow rule model, mine its rules, save them, optionally
explain it" — full-dataset rule extraction and the train/test holdout variant
both build on these functions rather than repeating the sequence.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.helper import report
from src.helper.scaling import apply_scaler
from .rules import (
    extract_rules_from_forest,
    evaluate_rules,
    simplify_rules,
    ruleset_summary,
    save_rules,
)
from .shap_utils import explain_model

# Tree models this module can build dedicated rule-extraction estimators for.
RULE_MODELS = ("rf", "decisiontree")


def fit_rule_model(
    model_name: str,
    X_scaled: np.ndarray,
    y: np.ndarray,
    params: Dict[str, Any],
) -> Any:
    """Build and fit a dedicated, shallow rule-extraction tree model.

    These estimators are intentionally separate from the deep benchmark
    models: they are depth-capped (configured in ``rules.model_params``) so
    the extracted rules stay short and interpretable.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    if model_name == "rf":
        model = RandomForestClassifier(n_jobs=-1, **params)
    elif model_name == "decisiontree":
        model = DecisionTreeClassifier(**params)
    else:
        raise ValueError(
            f"Unsupported rule model '{model_name}'. Supported: {', '.join(RULE_MODELS)}."
        )
    model.fit(X_scaled, y)
    return model


def mine_rules(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    scaler: Any,
    class_labels: Optional[Sequence[str]],
    probe_gene_map: Optional[Dict[str, str]],
    rules_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Walk a fitted tree model's root->leaf paths into rules, evaluate them
    against ``(X, y)``, and simplify/filter/rank them.

    Does not write anything to disk — see :func:`save_rule_outputs`.

    Returns
    -------
    (simplified_rules_df, summary_dict) — summary includes the rule-count
    provenance (n_rules_raw/passed_filter/kept/dropped_by_filter/dropped_by_cap)
    and set-level metrics (coverage, mean_fidelity, per_class) from
    :func:`ruleset_summary`.
    """
    filter_cfg = dict(rules_cfg.get("filter", {}))

    X_scaled = apply_scaler(scaler, X)
    y_pred = model.predict(X_scaled)  # source-model predictions for fidelity

    raw = extract_rules_from_forest(
        model, feature_names, scaler=scaler, gene_map=probe_gene_map or None
    )
    evaluated = evaluate_rules(
        raw, X, y, y_pred=y_pred, class_labels=class_labels or None
    )

    cfg = dict(filter_cfg)
    cfg["discretize"] = rules_cfg.get("discretize", False)
    cfg["_gene_medians"] = {i: float(np.median(X[:, i])) for i in range(X.shape[1])}
    simplified = simplify_rules(evaluated, cfg)

    # Rule-count provenance: (A) raw paths → (B) passed filters → (C) kept.
    n_raw = int(simplified.attrs.get("n_raw", len(evaluated)))
    n_passed = int(simplified.attrs.get("n_passed_filter", len(simplified)))
    n_kept = int(len(simplified))

    summary = ruleset_summary(
        simplified, X, y, y_pred=y_pred, class_labels=class_labels or None
    )
    summary["n_rules_raw"] = n_raw
    summary["n_rules_passed_filter"] = n_passed
    summary["n_rules_kept"] = n_kept
    summary["n_dropped_by_filter"] = n_raw - n_passed
    summary["n_dropped_by_cap"] = n_passed - n_kept

    # The actual filter config used for this run — rules_summary.json
    # otherwise only records counts (n_dropped_by_filter etc.), leaving no way
    # for a later reader (e.g. the results UI) to tell what config actually
    # produced them. Every key simplify_rules itself reads from cfg (see its
    # docstring) — kept in sync with that list.
    summary["filter_config"] = {
        "min_confidence": filter_cfg.get("min_confidence"),
        "min_fidelity": filter_cfg.get("min_fidelity"),
        "min_support": filter_cfg.get("min_support"),
        "min_abs_support": filter_cfg.get("min_abs_support"),
        "max_conditions": filter_cfg.get("max_conditions"),
        "merge_same_gene": filter_cfg.get("merge_same_gene"),
        "dedup": filter_cfg.get("dedup"),
        "dedup_sig_figs": filter_cfg.get("dedup_sig_figs"),
        "merge_generalization": filter_cfg.get("merge_generalization"),
        "max_rules_per_class": filter_cfg.get("max_rules_per_class"),
        "max_rules_total": filter_cfg.get("max_rules_total"),
    }

    return simplified, summary


def save_rule_outputs(
    simplified: pd.DataFrame,
    summary: Dict[str, Any],
    output_dir: Path,
    model_name: str,
    rules_cfg: Dict[str, Any],
    class_labels: Optional[Sequence[str]],
) -> Dict[str, Path]:
    """Write rules.json/.csv/.../genes_in_rules.csv + rules_summary.json to ``output_dir``."""
    exports = rules_cfg.get(
        "export", ["json", "csv", "csv_probes", "human", "summary", "genes"]
    )
    return save_rules(
        simplified, output_dir, class_labels=class_labels or None,
        model_name=model_name, summary=summary, exports=exports,
    )


def explain_rule_model(
    model: Any,
    X_scaled: np.ndarray,
    feature_names: Sequence[str],
    output_dir: Path,
    class_labels: Optional[Sequence[str]],
    probe_gene_map: Optional[Dict[str, str]],
    shap_cfg: Optional[Dict[str, Any]],
    model_name: str = "",
) -> None:
    """Run SHAP on a fitted rule model when ``shap.enabled & shap.on_rule_model`` — non-fatal."""
    if not (shap_cfg and shap_cfg.get("enabled", True) and shap_cfg.get("on_rule_model", True)):
        return
    try:
        shap_dir = output_dir / "shap"
        report.step(f"SHAP (rule model · {model_name})" if model_name else "SHAP (rule model)")
        explain_model(
            model=model,
            X_train=X_scaled,
            X_test=X_scaled,
            feature_names=feature_names,
            output_dir=shap_dir,
            top_k=shap_cfg.get("top_k", 20),
            class_names=class_labels if class_labels else None,
            probe_gene_map=probe_gene_map or None,
        )
        report.ok(f"SHAP complete → {shap_dir}")
    except Exception as e:
        report.warn(f"SHAP failed for rule model {model_name}: {e}")
