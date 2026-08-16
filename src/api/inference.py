"""
src/api/inference.py

Predict a single held-out test sample: load the cached model + rules for a
(dataset, fs_method, model) combo, run the model's predict(), and find which
of its extracted rules the sample's raw feature values satisfy (reusing
``src.interpretation.rules._antecedent_mask`` — the same condition-matching
logic the pipeline uses to compute rule support/confidence, not a
reimplementation).

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

from src.helper.paths import safe_filename
from src.helper.scaling import apply_scaler
from src.interpretation.rules import _antecedent_mask
from src.api import llm
from src.api.registry import PROJECT_ROOT, rule_model_folder


def _fs_dir(root: Path, dataset: str, fs_method: str, model_folder: str) -> Path:
    return root / dataset / fs_method / model_folder


def load_test_sample(
    dataset: str, sample_id: str, split_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Always reads from outputs_holdout/ — the held-out split (same seed,
    same config) is identical whether the currently-selected model/rules came
    from the cache or a live run, and a live job never re-exports its own
    test_set/ (see run_rule_model_job), so this must not follow ``run_id``.

    ``split_params`` (non-default min_samples_per_class/test_size from Step 2's
    "Thực hiện lại") bypasses the cache entirely and recomputes the split live,
    since the cached test_set/ only ever reflects holdout.yaml's defaults.
    """
    if split_params:
        from src.api.registry import compute_holdout_split

        ctx = compute_holdout_split(dataset, split_params)
        return _sample_from_split_ctx(ctx, sample_id)

    from src.api.registry import holdout_root_for

    safe_sid = safe_filename(sample_id)
    path = holdout_root_for() / dataset / "test_set" / "samples" / f"{safe_sid}.json"
    if not path.exists():
        raise FileNotFoundError(f"Test sample not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_from_split_ctx(ctx: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
    """Same ``{"features": {...}, "true_label": ...}`` shape as a cached
    test_set/samples/*.json export, built from a live compute_holdout_split
    context instead — only test-split rows are eligible.
    """
    sample_ids = ctx["sample_ids"]
    id_to_row = {sample_ids[i]: i for i in ctx["test_idx"]}
    if sample_id not in id_to_row:
        raise FileNotFoundError(f"Test sample not found in current split: {sample_id!r}")
    row = id_to_row[sample_id]
    features = {f: float(ctx["X"][row, j]) for j, f in enumerate(ctx["feature_names"])}
    label_idx = int(ctx["y"][row])
    class_labels = ctx["class_labels"]
    true_label = class_labels[label_idx] if label_idx < len(class_labels) else str(label_idx)
    return {"features": features, "true_label": true_label}


def get_test_samples(dataset: str, split_params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Live equivalent of GET .../test-samples reading test_set/manifest.csv —
    recomputes the split with custom min_samples_per_class/test_size and lists
    only the test-split rows, same ``sample_id``/``true_label``/``row_index``
    shape the frontend's TestSample already expects.
    """
    from src.api.registry import compute_holdout_split

    ctx = compute_holdout_split(dataset, split_params)
    sample_ids, y, class_labels = ctx["sample_ids"], ctx["y"], ctx["class_labels"]
    out = []
    for row in ctx["test_idx"]:
        label_idx = int(y[row])
        out.append({
            "sample_id": sample_ids[row],
            "true_label": class_labels[label_idx] if label_idx < len(class_labels) else str(label_idx),
            "row_index": int(row),
        })
    return out


def build_test_set_zip(dataset: str, split_params: Optional[Dict[str, Any]] = None) -> bytes:
    """ZIP of the held-out test set — for downloading and re-uploading
    through the existing "test 1 sample" flow (predict-upload), and for
    auditing exactly which rows were held out ("đảm bảo tính tường minh").

    Two sources, covering both flows the UI supports:
      - No ``split_params`` AND a cached ``test_set/`` already exists on disk
        (outputs_holdout/k{active}/{dataset}/test_set/, written by
        run_baseline_split() + copied by run_rule_extraction_holdout()) ->
        zip those files directly, no recompute.
      - Otherwise (custom "Thực hiện lại" split, or a brand-new upload with
        no cache yet) -> recompute the split live via compute_holdout_split()
        and build the same artifacts in-memory (build_test_set_artifacts),
        so download works even before any model has been trained.

    Zip contents: test_set.csv, manifest.csv, samples/{sample_id}.json — same
    layout either way, so the downloaded file is identical in shape
    regardless of which source produced it.
    """
    import io
    import zipfile

    from src.api.registry import holdout_root_for

    cached_dir = holdout_root_for() / dataset / "test_set"
    buf = io.BytesIO()

    if not split_params and cached_dir.exists():
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in cached_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(cached_dir.parent)))
        return buf.getvalue()

    from src.api.registry import compute_holdout_split
    from src.helper.split import build_test_set_artifacts

    ctx = compute_holdout_split(dataset, split_params)
    artifacts = build_test_set_artifacts(
        ctx["feature_names"], ctx["sample_ids"], ctx["X"], ctx["y"],
        ctx["test_idx"], ctx["class_labels"],
    )
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("test_set/test_set.csv", artifacts["test_set_df"].to_csv(index=False))
        zf.writestr("test_set/manifest.csv", artifacts["manifest_df"].to_csv(index=False))
        for sid, payload in artifacts["samples"].items():
            safe_sid = safe_filename(sid)
            zf.writestr(f"test_set/samples/{safe_sid}.json", json.dumps(payload, indent=2))
    return buf.getvalue()


def _rule_matches(
    rule: Dict[str, Any], feature_values: Dict[str, float]
) -> Optional[bool]:
    """True/False if every antecedent_raw probe is present in the sample; None if not evaluable."""
    conds = rule.get("antecedent_raw", [])
    if not conds:
        return None
    for c in conds:
        probe = c.get("probe")
        if probe not in feature_values:
            return None
        val = feature_values[probe]
        thr = c["threshold"]
        ok = (val <= thr) if c["op"] == "<=" else (val > thr)
        if not ok:
            return False
    return True


def _rule_match_detail(rule: Dict[str, Any], feature_values: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Unlike ``_rule_matches``, evaluates EVERY antecedent condition (no
    short-circuit) so a rule that's close-but-not-quite-matching can still be
    surfaced as a "partial match" — how many conditions held, and which ones
    didn't. ``None`` if the rule isn't evaluable at all (missing probes).
    """
    conds = rule.get("antecedent_raw", [])
    if not conds:
        return None
    detail_conds: List[Dict[str, Any]] = []
    satisfied = 0
    # ``antecedent`` is only a display projection.  Match each canonical raw
    # probe condition independently; a gene can legitimately have many of
    # them.
    for c in conds:
        probe = c.get("probe")
        if probe not in feature_values:
            return None
        val = feature_values[probe]
        thr = c["threshold"]
        ok = (val <= thr) if c["op"] == "<=" else (val > thr)
        if ok:
            satisfied += 1
        detail_conds.append({
            "gene": c.get("gene") or probe,
            "probe": probe,
            "op": c["op"],
            "threshold": thr,
            "actual": val,
            "ok": ok,
        })
    return {"satisfied": satisfied, "total": len(conds), "conditions": detail_conds}


def _is_numeric(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def parse_uploaded_sample(text: str) -> tuple[Dict[str, float], Optional[str]]:
    """Parse an uploaded sample file into ``(feature_values, true_label)``.

    Accepts three loose formats — deliberately permissive about EXTRA fields
    (a file can be a trimmed-down copy of a test_set/samples/*.json export
    with unrelated metadata cut out; only the microarray feature values and
    an optional label matter here):

    1. JSON, either the same shape as ``test_set/samples/*.json``
       (``{"true_label": ..., "features": {probe: value, ...}}``) or a bare
       ``{probe: value, ...}`` object with no wrapper.
    2. CuMiDa-style wide row: header + one data row, e.g.
       ``samples,type,209469_at,...`` / ``GSM1,Astrocytoma,6.9,...`` — the
       first non-numeric header cells (sample id / label) are skipped and the
       'type' column (if present) is returned as ``true_label``.
    3. Long ``probe,value`` pairs, one per line (no header required, .csv or
       .txt) — no wrapper needed, just two columns.
    """
    import csv
    import io
    import json

    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
        except ValueError as e:
            raise ValueError(f"File JSON khong hop le: {e}")
        if "features" in data and isinstance(data["features"], dict):
            features = {k: float(v) for k, v in data["features"].items() if _is_numeric(v)}
            true_label = data.get("true_label") or None
        else:
            features = {k: float(v) for k, v in data.items() if _is_numeric(v)}
            true_label = None
        if not features:
            raise ValueError("Khong tim thay dac trung dang so nao trong file JSON.")
        return features, true_label

    rows = [r for r in csv.reader(io.StringIO(text)) if r]
    if not rows:
        raise ValueError("File rong hoac khong doc duoc.")

    header = rows[0]
    if len(rows) >= 2 and len(header) > 2:
        data_row = rows[1]
        true_label = None
        start = 0
        for i, val in enumerate(data_row):
            try:
                float(val)
                start = i
                break
            except ValueError:
                if header[i].strip().lower() in ("type", "label", "class"):
                    true_label = val.strip()
                continue
        features = {}
        for col, val in zip(header[start:], data_row[start:]):
            try:
                features[col.strip()] = float(val)
            except ValueError:
                continue
        if features:
            return features, true_label

    # Fallback: long "probe,value" format.
    features = {}
    for r in rows:
        if len(r) < 2:
            continue
        try:
            features[r[0].strip()] = float(r[1])
        except ValueError:
            continue
    if not features:
        raise ValueError("Khong tim thay cot dac trung dang so nao trong file.")
    return features, None


def _predict_from_features_sync(
    dataset: str,
    fs_method: str,
    model: str,
    feature_values: Dict[str, float],
    run_id: Optional[str],
) -> Dict[str, Any]:
    """The blocking half of predict_from_features (disk I/O, joblib.load,
    sklearn transform/predict) — run via asyncio.to_thread so a slow
    prediction doesn't stall the single-threaded event loop for every other
    concurrent request (the LLM call afterwards is already async/awaited).
    """
    from src.api.registry import holdout_root_for, resolve_root

    root = resolve_root(run_id)
    model_folder = rule_model_folder(model)
    base = _fs_dir(root, dataset, fs_method, model_folder)

    model_path = base / "models" / "model_best.joblib"
    rules_path = base / "rules" / "rules.json"
    if not model_path.exists() or not rules_path.exists():
        raise FileNotFoundError(f"No cached model/rules for {dataset}/{fs_method}/{model}")

    bundle = joblib.load(model_path)
    fitted_model, scaler = bundle["model"], bundle["scaler"]

    rules_payload = json.loads(rules_path.read_text(encoding="utf-8"))
    class_labels: List[str] = rules_payload.get("class_labels") or []
    rules: List[Dict[str, Any]] = rules_payload.get("rules", [])

    rules_llm_path = base / "rules" / "rules_llm.json"
    rules_llm: Dict[str, str] = {}
    if rules_llm_path.exists():
        rules_llm = json.loads(rules_llm_path.read_text(encoding="utf-8"))

    genes_llm_path = base / "rules" / "gene_description_llm.json"
    genes_llm: Dict[str, str] = {}
    if genes_llm_path.exists():
        genes_llm = json.loads(genes_llm_path.read_text(encoding="utf-8"))

    # model.predict needs the FULL selected-feature vector (in training order),
    # not just the probes referenced in surviving rules — read it from the fs
    # selected-feature list rather than guessing from rules.
    sel_path = root / "feature_selection" / dataset / fs_method / "selected_features" / "selected_features.json"
    if not sel_path.exists():
        sel_path = holdout_root_for() / "feature_selection" / dataset / fs_method / "selected_features" / "selected_features.json"
    if sel_path.exists():
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        full_feature_order = sel.get("selected_features") or sel.get("confirmed") or []
    else:
        full_feature_order = list(feature_values.keys())

    missing = [f for f in full_feature_order if f not in feature_values]
    if missing:
        raise ValueError(
            f"Mau thieu {len(missing)}/{len(full_feature_order)} dac trung can thiet "
            f"(vd: {', '.join(missing[:5])}) — file phai chua day du cac cot dac trung "
            f"da duoc chon boi {fs_method}."
        )

    x_row = np.array([[feature_values.get(f, 0.0) for f in full_feature_order]], dtype=np.float64)
    x_scaled = apply_scaler(scaler, x_row)
    pred_idx = int(fitted_model.predict(x_scaled)[0])
    classification = class_labels[pred_idx] if pred_idx < len(class_labels) else str(pred_idx)

    matched_rules: List[Dict[str, Any]] = []
    partial_candidates: List[Dict[str, Any]] = []
    for r in rules:
        matched = _rule_matches(r, feature_values)
        if not matched:
            detail = _rule_match_detail(r, feature_values)
            if detail and detail["total"] > 0:
                ratio = detail["satisfied"] / detail["total"]
                if ratio >= 0.5:
                    partial_candidates.append({
                        "rule_id": r["rule_id"],
                        "text": r["text"],
                        "consequent_label": r.get("consequent", {}).get("class_label", ""),
                        "satisfied": detail["satisfied"],
                        "total": detail["total"],
                        "ratio": ratio,
                        "conditions": detail["conditions"],
                    })
        if matched:
            # Keyed by gene symbol (not probe id) for display — multiple probes
            # of the same gene collapse to one entry, last one wins.
            sample_values = {
                c["gene"]: feature_values.get(c["probe"])
                for c in r.get("antecedent_raw", [])
                if c["probe"] in feature_values
            }
            matched_rules.append({
                "rule_id": r["rule_id"],
                "text": r["text"],
                "matched": True,
                "consequent_label": r.get("consequent", {}).get("class_label", ""),
                "metrics": r.get("metrics", {}),
                "sample_values": sample_values,
                "explanation": rules_llm.get(str(r["rule_id"])),
                "antecedent": r.get("antecedent", []),
                "consequent": r.get("consequent", {}),
            })

    gene_context: Dict[str, str] = {}
    for r in matched_rules:
        for a in r.get("antecedent", []):
            g = a.get("gene")
            if g and g in genes_llm and g not in gene_context:
                gene_context[g] = genes_llm[g]

    # How many matched rules point to each class, and what fraction of all
    # matched rules that is — a deterministic, model-independent signal
    # (do the RULES agree with the model's own classification?), sorted
    # descending so the frontend can render it directly without re-sorting.
    vote_counts: Dict[str, int] = {}
    for r in matched_rules:
        label = r["consequent_label"]
        vote_counts[label] = vote_counts.get(label, 0) + 1
    total_votes = sum(vote_counts.values()) or 1
    class_votes = {
        label: {"count": count, "percentage": round(count / total_votes * 100, 1)}
        for label, count in sorted(vote_counts.items(), key=lambda kv: kv[1], reverse=True)
    }

    # The rule set's own "opinion" — the class the majority of matched rules
    # point to — kept separate from the model's classification since the two
    # can disagree (a rule matching a sample's antecedent doesn't mean the
    # model used that rule to decide; see class_votes above).
    rule_prediction = next(iter(class_votes), None)

    # Per-class breakdown of the PARTIAL matches only (>=50% but <100% of
    # conditions satisfied — i.e. partial_candidates), deliberately excluding
    # full matches so this
    # doesn't just duplicate class_votes above with extra rules mixed in —
    # it's the per-class counterpart of the "Khớp một phần (>=50%)" row in
    # the match-tier comparison, not a combined "matched + partial" view.
    over50_counts: Dict[str, int] = {}
    for p in partial_candidates:
        label = p["consequent_label"]
        over50_counts[label] = over50_counts.get(label, 0) + 1
    total_over50 = sum(over50_counts.values()) or 1
    class_votes_over50 = {
        label: {"count": count, "percentage": round(count / total_over50 * 100, 1)}
        for label, count in sorted(over50_counts.items(), key=lambda kv: kv[1], reverse=True)
    }

    # Return every partial match so the visible list and the per-class/total
    # charts describe the same rule population.
    partial_matches = sorted(partial_candidates, key=lambda p: p["ratio"], reverse=True)

    from src.api.registry import get_class_description, get_class_display_name

    # Vietnamese display name for EVERY class label this response can surface
    # (classification, rule_prediction, true_label, class_votes keys) — sent
    # once as a lookup map so the frontend can name classes consistently with
    # the Dataset Overview page instead of falling back to raw labels anywhere.
    all_labels = (
        set(class_labels) | set(class_votes) | set(class_votes_over50)
        | ({classification} if classification else set())
    )
    class_display_names = {
        label: name for label in all_labels
        if (name := get_class_display_name(dataset, label))
    }

    return {
        "classification": classification,
        "class_description": get_class_description(dataset, classification),
        "class_display_name": get_class_display_name(dataset, classification),
        "class_display_names": class_display_names,
        "rule_prediction": rule_prediction,
        "rule_prediction_description": get_class_description(dataset, rule_prediction) if rule_prediction else None,
        "rule_prediction_display_name": get_class_display_name(dataset, rule_prediction) if rule_prediction else None,
        "matched_count": len(matched_rules),
        "rules": matched_rules,
        "partial_matches": partial_matches,
        # The aggregate remains explicit so clients can calculate summary
        # ratios without re-counting the detail list.
        "n_partial_matches_total": len(partial_candidates),
        "n_rules_total": len(rules),
        "class_votes": class_votes,
        "class_votes_over50": class_votes_over50,
        "gene_context": gene_context,
    }


async def predict_from_features(
    dataset: str,
    fs_method: str,
    model: str,
    feature_values: Dict[str, float],
    true_label: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    import asyncio

    partial = await asyncio.to_thread(
        _predict_from_features_sync, dataset, fs_method, model, feature_values, run_id,
    )
    gene_context = partial.pop("gene_context")

    llm_result = await llm.explain_prediction(
        partial["rules"], partial["classification"], gene_context, partial["class_votes"],
    )

    return {
        **partial,
        "biomedical_summary": llm_result["summary"],
        "biomedical_rationale": llm_result["rationale"],
        "biomedical_model_vs_rule": llm_result["model_vs_rule"],
        "biomedical_disclaimer": llm_result["disclaimer"],
        "llm_used": llm_result["llm_used"],
        "true_label": true_label or "",
    }


async def predict_sample(
    dataset: str,
    fs_method: str,
    model: str,
    sample_id: str,
    run_id: Optional[str] = None,
    split_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sample = load_test_sample(dataset, sample_id, split_params)
    return await predict_from_features(
        dataset, fs_method, model,
        feature_values=sample["features"],
        true_label=sample.get("true_label", ""),
        run_id=run_id,
    )
