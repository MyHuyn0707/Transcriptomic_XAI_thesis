"""
src/interpretation/rules.py

Tree-based classification rule extractor.

Extracts human-readable IF-THEN rules from fitted tree models (Decision Tree,
Random Forest) by walking every root->leaf path. Split thresholds live in scaled
space (models train on StandardScaler output) and are converted back to raw
expression values so rules read as real gene levels (e.g. ``ERBB2 > 8.12``).

Two families of metrics are recomputed empirically on the full dataset:
  - data-based      : support, confidence, error, lift, class_specificity
                      (how well the rule matches the ground-truth labels)
  - prediction-based: fidelity
                      (how well the rule matches the source model's predictions)
This data-vs-prediction split follows the inTrees framework; fidelity is what a
downstream rule-guided model (e.g. an ANN with a rule-consistency loss) needs.

Public API
----------
    extract_rules_from_forest   Walk trees -> raw root->leaf path rules.
    evaluate_rules              Add empirical data- and prediction-based metrics.
    simplify_rules              Merge, filter, deduplicate and rank rules.
    ruleset_summary             Set-level metrics (coverage, mean fidelity).
    compare_rule_sets           Cross-check overlap between two models' rules.
    genes_in_rules              Distinct genes actually used by a rule set.
    save_rules                  Write rules.json / rules.csv / rules_with_probes.csv /
                                rules_human.txt / genes_in_rules.csv.

Called by
---------
    pipeline.GeneExpressionPipeline.run_rule_extraction

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.tree._tree import TREE_LEAF
except Exception:  # pragma: no cover
    TREE_LEAF = -1

from src.helper import report


# =============================================================================
# Tree walking — raw rule extraction
# =============================================================================


def _iter_trees(forest: Any) -> List[Any]:
    """Return a flat list of fitted decision trees from a forest or single tree."""
    if hasattr(forest, "estimators_"):
        ests = list(forest.estimators_)
        flat: List[Any] = []
        for e in ests:
            if isinstance(e, np.ndarray):
                flat.extend(e.tolist())
            else:
                flat.append(e)
        return [t for t in flat if hasattr(t, "tree_")]
    if hasattr(forest, "tree_"):
        return [forest]
    raise TypeError(
        "Expected a fitted forest with `estimators_` or a single tree with `tree_`."
    )


def _threshold_to_raw(thr_scaled: float, fidx: int, scaler: Any) -> float:
    """Convert a StandardScaler-space split threshold to raw expression.

    ``raw = thr_scaled * scale_[f] + mean_[f]`` — scale_ is positive so the
    inequality direction is preserved. Returns ``thr_scaled`` unchanged when no
    scaler is supplied.
    """
    if scaler is None:
        return float(thr_scaled)
    scale = getattr(scaler, "scale_", None)
    mean = getattr(scaler, "mean_", None)
    if scale is None or mean is None:
        return float(thr_scaled)
    return float(thr_scaled) * float(scale[fidx]) + float(mean[fidx])


def extract_rules_from_forest(
    forest: Any,
    feature_names: Sequence[str],
    scaler: Any = None,
    gene_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Enumerate every root->leaf path of every tree as one raw rule.

    Parameters
    ----------
    forest : fitted RandomForest / DecisionTree
        Walked node-by-node via the underlying ``tree_`` arrays.
    feature_names : sequence of str
        Column names (probe IDs) the model was trained on, in order.
    scaler : StandardScaler, optional
        Scaler the model was trained with; converts thresholds to raw expression.
    gene_map : dict, optional
        ``{probe_id: gene_symbol}`` for human-readable condition display.

    Returns
    -------
    pd.DataFrame
        One row per leaf path: ``tree_id``, ``class_idx``, ``leaf_n_samples``,
        ``leaf_value``, ``leaf_purity`` and ``conditions`` (list of dicts
        ``{idx, probe, gene, op, thr}``).
    """
    gene_map = gene_map or {}
    trees = _iter_trees(forest)
    records: List[Dict[str, Any]] = []

    for tree_id, tree in enumerate(trees):
        t = tree.tree_
        stack: List[tuple] = [(0, [])]
        while stack:
            node, conds = stack.pop()
            left = t.children_left[node]
            right = t.children_right[node]

            if left == TREE_LEAF:
                value = np.asarray(t.value[node]).ravel()  # class counts at leaf
                total = float(value.sum())
                records.append({
                    "tree_id": tree_id,
                    "class_idx": int(np.argmax(value)),
                    "leaf_n_samples": int(t.n_node_samples[node]),
                    "leaf_value": value.tolist(),
                    "leaf_purity": float(value.max() / total) if total else 0.0,
                    "conditions": conds,
                })
                continue

            fidx = int(t.feature[node])
            thr_raw = _threshold_to_raw(float(t.threshold[node]), fidx, scaler)
            probe = str(feature_names[fidx])
            gene = gene_map.get(probe, probe)

            base = {"idx": fidx, "probe": probe, "gene": gene, "thr": thr_raw}
            # Left child -> feature <= threshold; right child -> feature > threshold.
            stack.append((right, conds + [{**base, "op": ">"}]))
            stack.append((left,  conds + [{**base, "op": "<="}]))

    df = pd.DataFrame(records)
    report.ok(f"Extracted {len(df):,} raw rules from {len(trees):,} tree(s)")
    return df


# =============================================================================
# Empirical metric evaluation
# =============================================================================


def _antecedent_mask(
    conditions: Sequence[Dict[str, Any]], X: np.ndarray
) -> np.ndarray:
    """Boolean mask of samples satisfying every condition (raw-scale X)."""
    mask = np.ones(X.shape[0], dtype=bool)
    for c in conditions:
        col = X[:, c["idx"]]
        if c["op"] == "<=":
            mask &= col <= c["thr"]
        else:
            mask &= col > c["thr"]
    return mask


def evaluate_rules(
    rules: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    class_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Add empirical data-based and prediction-based metrics to raw rules.

    Parameters
    ----------
    rules : DataFrame from :func:`extract_rules_from_forest`.
    X : np.ndarray (n_samples, n_features)
        Raw (unscaled) matrix, columns aligned to the extraction feature order.
    y : np.ndarray
        Integer-encoded ground-truth class labels.
    y_pred : np.ndarray, optional
        The source model's predictions on the same samples. When given, enables
        ``fidelity`` (agreement of the rule with the model it was extracted from).
    class_labels : sequence of str, optional
        Display names indexed by class integer.

    Returns
    -------
    DataFrame with added columns: ``abs_support``, ``support``, ``confidence``,
    ``error``, ``lift``, ``class_specificity``, ``fidelity``, ``strength_score``,
    ``consequent_label``.
    """
    if rules.empty:
        return rules.copy()

    y = np.asarray(y)
    y_pred = None if y_pred is None else np.asarray(y_pred)
    n = len(y)
    classes = np.unique(y)
    priors = {int(k): float((y == k).mean()) for k in classes}

    out_rows: List[Dict[str, Any]] = []
    for _, rule in rules.iterrows():
        conds = rule["conditions"]
        cls = int(rule["class_idx"])
        mask = _antecedent_mask(conds, X)
        abs_support = int(mask.sum())
        support = abs_support / n if n else 0.0

        if abs_support == 0:
            confidence = lift = specificity = 0.0
            fidelity = float("nan") if y_pred is None else 0.0
        else:
            in_ant = y[mask]
            confidence = float((in_ant == cls).mean())             # vs ground truth
            base = priors.get(cls, 0.0)
            lift = confidence / base if base > 0 else 0.0
            others = [float((in_ant == int(k)).mean()) for k in classes if int(k) != cls]
            specificity = confidence - (max(others) if others else 0.0)
            # Fidelity: fraction of covered samples the SOURCE MODEL also routes
            # to this rule's class (how faithfully the rule mirrors the model).
            fidelity = (
                float((y_pred[mask] == cls).mean()) if y_pred is not None else float("nan")
            )

        label = (
            class_labels[cls]
            if class_labels is not None and cls < len(class_labels)
            else str(cls)
        )
        row = rule.to_dict()
        row.update({
            "abs_support": abs_support,
            "support": round(support, 4),
            "confidence": round(confidence, 4),
            "error": round(1.0 - confidence, 4),
            "lift": round(lift, 4),
            "class_specificity": round(specificity, 4),
            "fidelity": round(fidelity, 4) if fidelity == fidelity else float("nan"),
            "strength_score": round(support * confidence * lift, 6),
            "consequent_label": label,
        })
        out_rows.append(row)

    return pd.DataFrame(out_rows)


# =============================================================================
# Simplification & filtering
# =============================================================================


def _merge_conditions(
    conditions: Sequence[Dict[str, Any]], by_gene: bool = True
) -> List[Dict[str, Any]]:
    """Collapse splits on the same feature into the tightest single interval.

    ``by_gene=True`` keys by gene symbol, so multiple probes of one gene (common
    in probe-level data) read as a single condition. ``by_gene=False`` keys by
    probe. The lower bound is the max of all ``>`` thresholds; the upper bound is
    the min of all ``<=`` thresholds. Display-only: metrics were computed
    per-probe in :func:`evaluate_rules` and are not affected.
    """
    grouped: Dict[Any, Dict[str, Any]] = {}
    order: List[Any] = []
    for c in conditions:
        key = c["gene"] if by_gene else c["idx"]
        m = grouped.get(key)
        if m is None:
            m = {"idx": c["idx"], "gene": c["gene"], "probes": [],
                 "low": None, "high": None}
            grouped[key] = m
            order.append(key)
        if c["probe"] not in m["probes"]:
            m["probes"].append(c["probe"])
        if c["op"] == ">":
            m["low"] = c["thr"] if m["low"] is None else max(m["low"], c["thr"])
        else:
            m["high"] = c["thr"] if m["high"] is None else min(m["high"], c["thr"])
    merged = [grouped[k] for k in order]
    for m in merged:
        m["probe"] = ",".join(m["probes"])
    return merged


def _fmt_num(x: float) -> str:
    return f"{x:.4g}"


def _dedup_key_str(m: Dict[str, Any], sig_figs: int) -> str:
    """Coarser-precision variant of :func:`_expr_str`, used ONLY as the
    dedup key (never displayed) — collapses near-duplicate splits such as
    ``PIR<=5.189`` / ``PIR<=5.182`` (different trees of the same random
    forest independently splitting on the same gene at a slightly different
    bootstrap-sample-dependent threshold) into one signature, so the SECOND
    one (lower strength_score, since rules are processed strength-sorted) is
    dropped by the existing ``seen_keys`` check instead of surviving as a
    "distinct" rule.
    """
    # Numeric rule identity is probe-level: several probes may share one gene
    # symbol but have unrelated expression scales.
    g, low, high = m["probe"], m["low"], m["high"]
    def r(v: float) -> str:
        return f"{v:.{sig_figs}g}"
    if low is not None and high is not None:
        return f"{r(low)}<{g}<={r(high)}"
    if low is not None:
        return f"{g}>{r(low)}"
    return f"{g}<={r(high)}"


def _condition_signature(merged: List[Dict[str, Any]]) -> tuple:
    """Sorted gene names — identical for two rules iff they test the exact
    same set of genes, regardless of threshold values OR bound shape (a
    gene bounded only below, e.g. ``GENE>7``, is a valid dominance candidate
    against the same gene bounded both ways, e.g. ``7<GENE<=8`` — treating
    "no bound" as an infinitely loose bound in :func:`_dominates` handles
    that; the grouping itself only needs to fix the gene set).
    """
    return tuple(sorted(m["probe"] for m in merged))


def _dominates(mi: Dict[str, Any], mj: Dict[str, Any]) -> bool:
    """True if condition-set ``mi`` is componentwise at-least-as-general as
    ``mj`` on every shared gene (lower-or-equal ``low``, higher-or-equal
    ``high``, treating ``None`` as an unbounded/infinitely loose bound on
    either side) — i.e. every sample satisfying ``mj`` also satisfies ``mi``.
    ``GENE>7`` (no high bound at all) dominates ``7<GENE<=8`` this way, since
    "no upper bound" is more general than any finite one.
    """
    for gene, cond_i in mi.items():
        cond_j = mj[gene]
        # low: mi has no lower restriction -> always dominates on this side.
        # Otherwise mi only dominates if mj ALSO has a low bound and it's
        # >= mi's (mj with low=None accepts values below mi's floor, which
        # mi rejects — mi can't dominate that).
        if cond_i["low"] is not None:
            if cond_j["low"] is None or cond_i["low"] > cond_j["low"]:
                return False
        if cond_i["high"] is not None:
            if cond_j["high"] is None or cond_i["high"] < cond_j["high"]:
                return False
    return True


def _drop_generalized_duplicates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Within each (same gene set + consequent class) group, drop
    every rule that's componentwise DOMINATED by ANOTHER SINGLE rule in the
    same group (pairwise — NOT "one rule must beat the whole group at once",
    which misses cases like {A, B, C} where B dominates C but neither
    dominates A: A being Pareto-incomparable with both must not block the
    B-dominates-C removal). The survivor of a dominated pair is always one of
    the existing, already-evaluated rows — no fabricated merged rule or stats.

    A pair with neither side dominating the other (two genes disagree on
    which rule is "more general") is left as-is — collapsing it would risk
    silently discarding a rule that isn't actually redundant.
    """
    groups: Dict[Any, List[int]] = {}
    for i, r in enumerate(rows):
        key = (_condition_signature(r["conditions_merged"]), r["consequent_label"])
        groups.setdefault(key, []).append(i)

    drop: set = set()
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        conds = {i: {m["probe"]: m for m in rows[i]["conditions_merged"]} for i in idxs}
        # rows is already strength_score-sorted (see simplify_rules' main
        # loop), so idxs is too — on a tie (mutual dominance, i.e. identical
        # ranges) the earlier/better-ranked rule survives.
        for a in range(len(idxs)):
            i = idxs[a]
            for b in range(a + 1, len(idxs)):
                j = idxs[b]
                if _dominates(conds[i], conds[j]):
                    drop.add(j)
                elif _dominates(conds[j], conds[i]):
                    drop.add(i)
    return [r for k, r in enumerate(rows) if k not in drop]


def _expr_str(m: Dict[str, Any]) -> str:
    """Threshold-form string for one merged condition, e.g. ``ERBB2>8.12``."""
    g, low, high = m["gene"], m["low"], m["high"]
    if low is not None and high is not None:
        return f"{_fmt_num(low)}<{g}<={_fmt_num(high)}"
    if low is not None:
        return f"{g}>{_fmt_num(low)}"
    return f"{g}<={_fmt_num(high)}"


def _expr_str_probe(m: Dict[str, Any]) -> str:
    """Same as :func:`_expr_str` but keyed by probe ID(s) instead of gene symbol.

    ``m["probe"]`` is a comma-joined list when several probes of the same gene
    were merged into one condition (see :func:`_merge_conditions`).
    """
    p, low, high = m["probe"], m["low"], m["high"]
    if low is not None and high is not None:
        return f"{_fmt_num(low)}<{p}<={_fmt_num(high)}"
    if low is not None:
        return f"{p}>{_fmt_num(low)}"
    return f"{p}<={_fmt_num(high)}"


def _disc_str(m: Dict[str, Any], medians: Optional[Dict[int, float]]) -> str:
    """Discretised ``GENE=High/Low`` label relative to the gene's raw median."""
    g, low, high = m["gene"], m["low"], m["high"]
    if low is not None and high is None:
        level = "High"
    elif high is not None and low is None:
        level = "Low"
    else:
        med = (medians or {}).get(m["idx"])
        center = (low + high) / 2.0
        level = "High" if (med is None or center >= med) else "Low"
    return f"{g}={level}"


def simplify_rules(rules: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Merge, filter, deduplicate and rank evaluated rules.

    Config keys read from ``cfg`` (the ``rules.filter`` block, plus a few extras)
    ---------------------------------------------------------------------------
    min_confidence, min_fidelity, min_support, min_abs_support : float
    max_conditions : int            cap on distinct genes in antecedent
    merge_same_gene : bool          collapse multiple probes of one gene
    dedup : bool                    drop duplicate antecedent -> class rules
    dedup_sig_figs : int            precision (significant figures) used ONLY for the
                                    dedup key, coarser than the 4 sig figs shown in the
                                    rule text — collapses near-duplicate splits from
                                    different trees (e.g. PIR<=5.189 vs PIR<=5.182)
    merge_generalization : bool     drop a rule when ANOTHER rule in the same class
                                    tests the exact same genes (same bound shape) with
                                    a strictly more general threshold on every one of
                                    them (e.g. keeps ``A>1`` over ``A>2``, ``B<=2`` over
                                    ``B<=1``) — unlike ``dedup`` above, this catches
                                    near-duplicates whose thresholds AREN'T close enough
                                    to round to the same value. Never merges across
                                    genes that disagree on which rule is more general.
    max_rules_per_class : int
    max_rules_total : int or None   None/"" -> no cap on the final rule count
    discretize : bool               ``GENE=High/Low`` vs numeric thresholds
    _gene_medians : dict            ``{feature_idx: median}`` for discretize

    Returns
    -------
    DataFrame ranked by ``strength_score`` with string ``antecedents`` /
    ``consequents`` columns, all metrics, and structured ``conditions_merged`` /
    ``conditions_raw`` (used by :func:`save_rules` to build machine-readable cards).
    """
    n_raw = int(len(rules))   # (A) every root->leaf path, before any filtering
    if rules.empty:
        report.warn("simplify_rules: no rules to simplify.")
        out = rules.copy()
        out.attrs["n_raw"] = 0
        out.attrs["n_passed_filter"] = 0
        return out

    min_conf = float(cfg.get("min_confidence", 0.80))
    min_fid = float(cfg.get("min_fidelity", 0.0))
    min_sup = float(cfg.get("min_support", 0.05))
    min_abs = int(cfg.get("min_abs_support", 3))
    max_cond = cfg.get("max_conditions", 5)
    # ``merge_same_gene`` is a display preference.  Never combine numeric
    # thresholds from different probes merely because they share a gene symbol:
    # they must remain separate terms in the exported rule.
    do_dedup = bool(cfg.get("dedup", True))
    dedup_sig_figs = int(cfg.get("dedup_sig_figs", 2))
    merge_generalization = bool(cfg.get("merge_generalization", True))
    max_per_class = int(cfg.get("max_rules_per_class", 20))
    # None/"" -> no cap (keep every rule that passed the filter stage), same
    # convention max_conditions already uses below — NOT the same as leaving
    # the key out of cfg entirely (that still falls back to the 100 default).
    max_total = cfg.get("max_rules_total", 100)
    max_total = None if max_total in (None, "") else int(max_total)
    discretize = bool(cfg.get("discretize", False))
    medians = cfg.get("_gene_medians")

    rows: List[Dict[str, Any]] = []
    seen_keys: set = set()

    for _, rule in rules.sort_values("strength_score", ascending=False).iterrows():
        # Repeated splits of the SAME probe become a valid interval; probes
        # sharing a gene remain separate conditions (e.g. RGCC>10.69 AND
        # RGCC<=8.006, rather than 10.69<RGCC<=8.006).
        merged = _merge_conditions(rule["conditions"], by_gene=False)

        if max_cond is not None and len(merged) > int(max_cond):
            continue
        if rule["confidence"] < min_conf:
            continue
        fid = rule.get("fidelity", float("nan"))
        if min_fid > 0 and fid == fid and fid < min_fid:
            continue
        if rule["support"] < min_sup and rule["abs_support"] < min_abs:
            continue

        expr_parts = [_expr_str(m) for m in merged]
        probe_expr_parts = [_expr_str_probe(m) for m in merged]
        disc_parts = [_disc_str(m, medians) for m in merged]
        antecedent_str = ", ".join(disc_parts) if discretize else ", ".join(expr_parts)
        antecedent_probe_str = ", ".join(probe_expr_parts)
        consequent_str = f"Class={rule['consequent_label']}"

        if do_dedup:
            # Coarser-precision, order-independent key so near-duplicate
            # splits on the same gene (different trees, slightly different
            # bootstrap-sample threshold) collapse together instead of each
            # surviving as a "distinct" rule — see _dedup_key_str.
            dedup_parts = tuple(sorted(_dedup_key_str(m, dedup_sig_figs) for m in merged))
            key = (dedup_parts, consequent_str)
            if key in seen_keys:
                continue
            seen_keys.add(key)

        rows.append({
            "antecedents": antecedent_str,
            "antecedents_probe": antecedent_probe_str,
            "consequents": consequent_str,
            "support": rule["support"],
            "confidence": rule["confidence"],
            "error": rule.get("error", round(1.0 - rule["confidence"], 4)),
            "lift": rule["lift"],
            "class_specificity": rule["class_specificity"],
            "fidelity": rule.get("fidelity", float("nan")),
            "strength_score": rule["strength_score"],
            "class_idx": int(rule["class_idx"]),
            "consequent_label": rule["consequent_label"],
            "n_conditions": len(merged),
            "antecedent_expr": " AND ".join(expr_parts),
            "antecedent_probe_expr": " AND ".join(probe_expr_parts),
            "conditions_merged": merged,
            "conditions_raw": list(rule["conditions"]),
            "tree_id": int(rule.get("tree_id", -1)),
            "leaf_n_samples": int(rule.get("leaf_n_samples", 0)),
            "leaf_purity": float(rule.get("leaf_purity", float("nan"))),
        })

    if merge_generalization and rows:
        rows = _drop_generalized_duplicates(rows)

    n_passed = int(len(rows))   # (B) passed all filters + dedup, before the caps
    if not rows:
        report.warn(
            f"simplify_rules: all rules filtered out "
            f"(min_confidence={min_conf}, min_support={min_sup}, "
            f"min_abs_support={min_abs}, max_conditions={max_cond})."
        )
        out = pd.DataFrame(rows)
        out.attrs["n_raw"] = n_raw
        out.attrs["n_passed_filter"] = 0
        return out

    df = (
        pd.DataFrame(rows)
        .sort_values("strength_score", ascending=False)
        .groupby("class_idx", group_keys=False)
        .head(max_per_class)
        .sort_values("strength_score", ascending=False)
        .head(max_total)
        .reset_index(drop=True)
    )

    # Expose the three counts (A raw -> B passed_filter -> C kept) so callers
    # can log/save them; pandas ops above don't preserve attrs, so set on `df`.
    df.attrs["n_raw"] = n_raw
    df.attrs["n_passed_filter"] = n_passed

    report.ok(
        f"Rules: raw={n_raw:,} → passed_filter={n_passed:,} → kept={len(df):,}  "
        f"(filter cut {n_raw - n_passed:,}, cap cut {n_passed - len(df):,}) "
        f"across {df['class_idx'].nunique()} class(es)"
    )
    return df


# =============================================================================
# Set-level metrics & cross-model comparison
# =============================================================================


def ruleset_summary(
    rules: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    class_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute set-level metrics for a simplified rule set.

    coverage     : fraction of samples that fire at least one rule's antecedent.
    mean_fidelity: support-weighted mean of per-rule fidelity (model agreement).
    per_class    : rule count per consequent class.
    """
    if rules is None or rules.empty:
        return {"n_rules": 0, "coverage": 0.0, "mean_fidelity": float("nan"),
                "per_class": {}}

    n = len(y)
    covered = np.zeros(n, dtype=bool)
    for conds in rules["conditions_raw"]:
        covered |= _antecedent_mask(conds, X)

    fid = rules["fidelity"].to_numpy(dtype=float)
    sup = rules["support"].to_numpy(dtype=float)
    valid = ~np.isnan(fid)
    mean_fid = (
        float(np.average(fid[valid], weights=sup[valid]))
        if valid.any() and sup[valid].sum() > 0 else float("nan")
    )

    per_class = rules.groupby("consequent_label").size().to_dict()
    return {
        "n_rules": int(len(rules)),
        "coverage": round(float(covered.mean()), 4),
        "mean_fidelity": round(mean_fid, 4) if mean_fid == mean_fid else float("nan"),
        "per_class": {str(k): int(v) for k, v in per_class.items()},
    }


def _condition_direction(m: Dict[str, Any]) -> str:
    """Direction label for a merged condition: High / Low / Range."""
    if m["low"] is not None and m["high"] is None:
        return "High"
    if m["high"] is not None and m["low"] is None:
        return "Low"
    return "Range"


def _rule_literals(row: pd.Series) -> frozenset:
    """The (gene, direction) literal set of a rule's antecedent."""
    return frozenset((m["gene"], _condition_direction(m)) for m in row["conditions_merged"])


def _rule_signature(row: pd.Series) -> tuple:
    """Strict rule identity: (gene+direction literal set, consequent class)."""
    return (_rule_literals(row), row["consequent_label"])


def compare_rule_sets(
    rules_a: pd.DataFrame, rules_b: pd.DataFrame,
    name_a: str = "rf", name_b: str = "dt",
) -> Dict[str, Any]:
    """Cross-check two models' rule sets at two levels.

    Rule level (strict): two rules match when they share the same antecedent
    (gene + direction) literal set AND consequent class — these are the
    highest-trust rules, confirmed by both models.

    Literal level (lenient): each ``(gene, direction) -> class`` literal that
    appears in both models' rule sets, even if the full antecedents differ
    (e.g. both models agree ``ERBB2=High -> HER2``). More informative when a
    single tree and an ensemble pick different exact splits.

    Both report a Jaccard similarity.
    """
    empty = rules_a is None or rules_a.empty or rules_b is None or rules_b.empty
    if empty:
        return {"n_a": 0 if rules_a is None else len(rules_a),
                "n_b": 0 if rules_b is None else len(rules_b),
                "n_overlap": 0, "jaccard": 0.0, "overlap": [],
                "literal_overlap": 0, "literal_jaccard": 0.0, "shared_literals": []}

    # --- Strict rule-level match -------------------------------------------
    sig_a = {_rule_signature(r): r for _, r in rules_a.iterrows()}
    sig_b = {_rule_signature(r): r for _, r in rules_b.iterrows()}
    common, union = set(sig_a) & set(sig_b), set(sig_a) | set(sig_b)

    overlap = []
    for sig in common:
        literals, cls = sig
        ra, rb = sig_a[sig], sig_b[sig]
        overlap.append({
            "literals": [f"{g}={d}" for g, d in sorted(literals)],
            "consequent": f"Class={cls}",
            f"{name_a}_confidence": float(ra["confidence"]),
            f"{name_b}_confidence": float(rb["confidence"]),
        })
    overlap.sort(key=lambda d: d[f"{name_a}_confidence"], reverse=True)

    # --- Lenient literal-level match ---------------------------------------
    def _literal_class_set(rules: pd.DataFrame) -> set:
        s = set()
        for _, r in rules.iterrows():
            for g, d in _rule_literals(r):
                s.add((g, d, r["consequent_label"]))
        return s

    lit_a, lit_b = _literal_class_set(rules_a), _literal_class_set(rules_b)
    lit_common, lit_union = lit_a & lit_b, lit_a | lit_b
    shared_literals = [
        {"gene": g, "direction": d, "consequent": f"Class={c}"}
        for (g, d, c) in sorted(lit_common)
    ]

    return {
        "n_a": int(len(rules_a)),
        "n_b": int(len(rules_b)),
        "n_overlap": len(common),
        "jaccard": round(len(common) / len(union), 4) if union else 0.0,
        "overlap": overlap,
        "literal_overlap": len(lit_common),
        "literal_jaccard": round(len(lit_common) / len(lit_union), 4) if lit_union else 0.0,
        "shared_literals": shared_literals,
    }


def genes_in_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Distinct genes that actually appear in a simplified rule set.

    One row per gene symbol, so downstream steps (GO enrichment, manual
    review) know exactly which genes the ruleset is built on — as opposed to
    the full SHAP top-K, which may include genes no surviving rule uses.

    Returns
    -------
    pd.DataFrame  columns: ``gene``, ``probes`` (``;``-joined probe IDs backing
    the gene), ``n_rules`` (how many rules reference it), ``classes``
    (``;``-joined consequent classes it appears in).
    """
    if rules is None or rules.empty:
        return pd.DataFrame(columns=["gene", "probes", "n_rules", "classes"])

    per_gene: Dict[str, Dict[str, Any]] = {}
    for _, r in rules.iterrows():
        cls = r["consequent_label"]
        seen_in_rule: set[str] = set()
        for m in r["conditions_merged"]:
            gene = m["gene"]
            entry = per_gene.setdefault(
                gene, {"probes": set(), "n_rules": 0, "classes": set()}
            )
            entry["probes"].update(m["probe"].split(","))
            # The same gene may now have several probe terms in one rule;
            # this is still one rule mentioning that gene.
            if gene not in seen_in_rule:
                entry["n_rules"] += 1
                seen_in_rule.add(gene)
            entry["classes"].add(str(cls))

    rows = [
        {
            "gene": gene,
            "probes": ";".join(sorted(e["probes"])),
            "n_rules": e["n_rules"],
            "classes": ";".join(sorted(e["classes"])),
        }
        for gene, e in per_gene.items()
    ]
    return (
        pd.DataFrame(rows)
        .sort_values(["n_rules", "gene"], ascending=[False, True])
        .reset_index(drop=True)
    )


# =============================================================================
# Output
# =============================================================================


def _build_cards(rules: pd.DataFrame, model_name: Optional[str]) -> List[Dict[str, Any]]:
    """Build structured rule cards (machine-readable, for downstream stages)."""
    cards: List[Dict[str, Any]] = []
    for rid, r in enumerate(
        rules.sort_values("strength_score", ascending=False).to_dict("records")
    ):
        cards.append({
            "rule_id": rid,
            "model": model_name,
            "consequent": {
                "class_idx": int(r["class_idx"]),
                "class_label": r["consequent_label"],
            },
            # Gene-level antecedent for display / LLM (stage 8-9).
            "antecedent": [
                {"gene": m["gene"], "probe": m["probe"],
                 "low": m["low"], "high": m["high"]}
                for m in r["conditions_merged"]
            ],
            # Exact per-probe conditions for reproduction and rule-guided ANN
            # (stage 10): one (feature, op, threshold) triple per split.
            "antecedent_raw": [
                {"probe": c["probe"], "gene": c["gene"],
                 "op": c["op"], "threshold": c["thr"]}
                for c in r["conditions_raw"]
            ],
            "metrics": {
                "support": r["support"],
                "confidence": r["confidence"],
                "error": r["error"],
                "lift": r["lift"],
                "class_specificity": r["class_specificity"],
                "fidelity": None if pd.isna(r["fidelity"]) else r["fidelity"],
                "strength_score": r["strength_score"],
                "n_conditions": r["n_conditions"],
            },
            "provenance": {
                "tree_id": r["tree_id"],
                "leaf_n_samples": r["leaf_n_samples"],
                "leaf_purity": None if pd.isna(r["leaf_purity"]) else r["leaf_purity"],
            },
            "text": f"IF {r['antecedent_expr']} THEN {r['consequents']}",
        })
    return cards


def save_rules(
    rules: pd.DataFrame,
    output_dir: str | Path,
    class_labels: Optional[Sequence[str]] = None,
    model_name: Optional[str] = None,
    summary: Optional[Dict[str, Any]] = None,
    exports: Sequence[str] = ("json", "csv", "human", "summary"),
) -> Dict[str, Path]:
    """Write rule artifacts to ``output_dir``.

    Artifacts (controlled by ``exports``)
    -------------------------------------
    rules.json         Structured cards (canonical, for stages 9-10): per rule a
                       gene-level antecedent, exact per-probe antecedent_raw,
                       metrics and provenance.
    rules.csv          Flat one-row-per-rule table, sortable by any metric.
    rules_with_probes.csv  Same as rules.csv but antecedents spelled out with
                       probe IDs (``antecedents_probe``) instead of gene symbols
                       — use when several probes collapse to one gene and you
                       need to know exactly which probe fired.
    rules_human.txt    ``IF ERBB2>8.12 AND ... THEN Class=HER2 (conf, support, fidelity)``.
    rules_summary.json Set-level metrics (coverage, mean_fidelity, per_class).
    genes_in_rules.csv Distinct genes actually used by the surviving rules
                       (gene, backing probe IDs, rule count, classes) — the
                       set to hand to GO enrichment, not the full SHAP top-K.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    csv_cols = [
        "antecedents", "consequents", "support", "confidence", "error",
        "lift", "class_specificity", "fidelity", "strength_score", "n_conditions",
    ]
    csv_probe_cols = [
        "antecedents_probe", "consequents", "support", "confidence", "error",
        "lift", "class_specificity", "fidelity", "strength_score", "n_conditions",
    ]

    if "csv" in exports:
        csv_path = output_dir / "rules.csv"
        cols = csv_cols if not rules.empty else csv_cols
        (rules[cols] if not rules.empty else pd.DataFrame(columns=cols)).to_csv(
            csv_path, index=False
        )
        paths["csv"] = csv_path

    if "csv_probes" in exports:
        probes_path = output_dir / "rules_with_probes.csv"
        (
            rules[csv_probe_cols] if not rules.empty else pd.DataFrame(columns=csv_probe_cols)
        ).to_csv(probes_path, index=False)
        paths["csv_probes"] = probes_path

    if "genes" in exports:
        genes_path = output_dir / "genes_in_rules.csv"
        genes_in_rules(rules).to_csv(genes_path, index=False)
        paths["genes"] = genes_path

    if "json" in exports:
        json_path = output_dir / "rules.json"
        cards = _build_cards(rules, model_name) if not rules.empty else []
        json_path.write_text(
            json.dumps(
                {"model": model_name,
                 "class_labels": list(class_labels) if class_labels else None,
                 "n_rules": len(cards), "rules": cards},
                indent=2,
            ),
            encoding="utf-8",
        )
        paths["json"] = json_path

    if "human" in exports:
        human_path = output_dir / "rules_human.txt"
        lines: List[str] = []
        if not rules.empty:
            for _, r in rules.sort_values("strength_score", ascending=False).iterrows():
                fid = "" if pd.isna(r["fidelity"]) else f", fidelity={r['fidelity']:.2f}"
                lines.append(
                    f"IF {r['antecedent_expr']} THEN {r['consequents']} "
                    f"(conf={r['confidence']:.2f}, support={r['support']:.2f}, "
                    f"lift={r['lift']:.2f}{fid})"
                )
        human_path.write_text(
            "\n".join(lines) if lines else "# No rules passed filtering.\n",
            encoding="utf-8",
        )
        paths["human"] = human_path

    if "summary" in exports and summary is not None:
        summary_path = output_dir / "rules_summary.json"
        summary_path.write_text(
            json.dumps({"model": model_name, **summary}, indent=2), encoding="utf-8"
        )
        paths["summary"] = summary_path

    report.ok(f"Saved {len(rules):,} rules -> {output_dir}")
    return paths
