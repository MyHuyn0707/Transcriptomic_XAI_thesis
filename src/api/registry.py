"""
src/api/registry.py

Scans configs/datasets.yaml + outputs_holdout/ to answer "what datasets, fs
methods and rule models actually have cached results on disk" — no hardcoded
dataset table. Also resolves the read path for a given (dataset, run_id):
run_id=None -> outputs_holdout/ ("Tai log cu"); run_id=<job_id> ->
outputs_live/<job_id>/ (a completed "Huan luyen" job).

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.helper.config_loader import ConfigLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_ROOT = PROJECT_ROOT / "outputs_holdout"
LIVE_ROOT = PROJECT_ROOT / "outputs_live"
DATASET_DESCRIPTION_ROOT = PROJECT_ROOT / "Dataset_Description"
RULE_MODEL_FOLDERS = {"rf": "rf", "decisiontree": "dt"}
# UI model key ('rf' | 'dt') -> internal pipeline model name.
UI_TO_INTERNAL_MODEL = {"rf": "rf", "dt": "decisiontree"}


@lru_cache(maxsize=1)
def get_config_loader() -> ConfigLoader:
    return ConfigLoader(PROJECT_ROOT / "configs")


def holdout_root_for(k: Optional[int] = None) -> Path:
    """``outputs_holdout/k{k}/`` — k defaults to holdout.yaml's
    ``active_min_samples_per_class``. Changing which k the API/UI serves is a
    config edit + process restart, not a runtime selector — this function is
    the single place every HOLDOUT_ROOT-derived path goes through, so a
    config change takes effect everywhere at once.
    """
    if k is None:
        cfg = get_config_loader().load_yaml("holdout.yaml").get("holdout", {})
        k = int(cfg.get("active_min_samples_per_class", 4))
    return HOLDOUT_ROOT / f"k{k}"


def sanitize_segment(value: str, name: str = "value") -> str:
    """Reject anything that isn't a single, literal path segment.

    All of dataset_id/fs_method/model/run_id ultimately get concatenated
    into filesystem paths (``root / dataset_id / fs_method / ...``) built
    from request path/query params. FastAPI's ``{param}`` matcher only
    forbids a literal ``/`` in the raw path, not ``..`` — so without this
    check a crafted ``dataset_id=".."`` (or similar) could walk outside
    outputs_holdout/outputs_live and read arbitrary files on disk.
    """
    if not value or value in (".", "..") or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"Invalid {name}: {value!r}")
    return value


def resolve_root(run_id: Optional[str] = None) -> Path:
    """outputs_holdout/k{active}/ when run_id is None, else outputs_live/<run_id>/."""
    if run_id:
        sanitize_segment(run_id, "run_id")
        root = LIVE_ROOT / run_id
        if not root.exists():
            raise FileNotFoundError(f"Unknown run_id '{run_id}' (not under {LIVE_ROOT})")
        return root
    return holdout_root_for()


def available_fs_models(ds_root: Path) -> Dict[str, List[str]]:
    """``{fs_method: [model_folder, ...]}`` for whichever combos have rules.json on disk."""
    out: Dict[str, List[str]] = {}
    if not ds_root.exists():
        return out
    for rules_json in sorted(ds_root.glob("*/*/rules/rules.json")):
        fs_method = rules_json.parent.parent.parent.name
        model_folder = rules_json.parent.parent.name
        out.setdefault(fs_method, [])
        if model_folder not in out[fs_method]:
            out[fs_method].append(model_folder)
    return out


def list_datasets() -> List[Dict[str, Any]]:
    """One entry per enabled dataset that has at least one cached holdout run."""
    loader = get_config_loader()
    datasets_cfg = loader.load_datasets_config().get("datasets", {})
    results: List[Dict[str, Any]] = []

    holdout_root = holdout_root_for()
    for run_key in loader.get_enabled_datasets():
        ds_root = holdout_root / run_key
        split_info_path = ds_root / "split_info.json"
        if not split_info_path.exists():
            continue

        split_info = json.loads(split_info_path.read_text(encoding="utf-8"))
        base_name = run_key[: -len("-gene")] if run_key.endswith("-gene") else run_key
        cfg = datasets_cfg.get(base_name, {})

        fs_models = available_fs_models(ds_root)
        n_original_features = None
        fs_root_for_dataset = holdout_root / "feature_selection" / run_key
        fs_candidates = list(fs_models) or (
            [p.name for p in fs_root_for_dataset.iterdir() if p.is_dir()]
            if fs_root_for_dataset.exists() else []
        )
        for fs_method in fs_candidates:
            meta_path = fs_root_for_dataset / fs_method / "params" / "metadata.json"
            if meta_path.exists():
                n_original_features = json.loads(meta_path.read_text(encoding="utf-8")).get(
                    "n_original_features"
                )
                break

        results.append({
            "id": run_key,
            "name": cfg.get("description", run_key),
            "platform": cfg.get("platform", ""),
            "n_samples": split_info.get("n_samples_total"),
            "n_features": n_original_features,
            "n_classes": len(split_info.get("class_labels", [])),
            "class_labels": split_info.get("class_labels", []),
            "description": cfg.get("description", ""),
            "fs_models": fs_models,
        })

    restore_temp_datasets_from_disk()
    results.extend(t["info"] for t in _TEMP_DATASETS.values())
    return results


def get_dataset_or_404(dataset_id: str) -> Dict[str, Any]:
    for ds in list_datasets():
        if ds["id"] == dataset_id:
            return ds
    raise FileNotFoundError(f"Dataset '{dataset_id}' has no cached holdout results.")


def rule_model_folder(model_name: str) -> str:
    """UI model key ('rf' | 'dt') or internal name ('rf' | 'decisiontree') -> on-disk folder name."""
    if model_name in ("rf", "dt"):
        return model_name
    return RULE_MODEL_FOLDERS.get(model_name, model_name)


# =============================================================================
# Session-only "uploaded dataset" registry (Step 1 "Tải lên dataset mới").
#
# Built from a user's raw GEO/CuMiDa file via build_geo_dataset/
# build_cumida_dataset. The registration (dataset_id -> CSV path/platform/
# description) lives ONLY in this in-memory dict — never written to
# datasets.yaml/Dataset_Description — but the BUILT CSV itself is a real file
# under outputs_live/uploads/<upload_id>/built/, which survives a restart.
# register_temp_dataset also drops a small registration.json next to it so
# restore_temp_datasets_from_disk() can re-populate this dict from whatever
# uploads are still on disk — without that, "Lịch sử tải lên" could list a
# past upload but never let you re-select it after the process restarts.
# =============================================================================

_TEMP_DATASETS: Dict[str, Dict[str, Any]] = {}
UPLOAD_ROOT = LIVE_ROOT / "uploads"


def register_temp_dataset(
    dataset_id: str,
    *,
    path: str,
    platform: str = "",
    description: str = "",
    annotation_file: Optional[str] = None,
    _persist: bool = True,
) -> Dict[str, Any]:
    """Register a session-only dataset and return its ``list_datasets()``-shaped
    info dict. Reads only the CSV header + the 'type' column (not the full
    feature matrix) to report n_samples/n_features/n_classes cheaply.

    ``_persist`` writes ``registration.json`` next to the CSV so this
    registration can be reconstructed later by
    :func:`restore_temp_datasets_from_disk` — set False only when THAT
    function is doing the restoring, to avoid rewriting the same file it just
    read.
    """
    import json as _json
    import pandas as pd

    type_col = pd.read_csv(path, usecols=[1]).iloc[:, 0].astype(str)
    header = pd.read_csv(path, nrows=0)
    class_labels = sorted(type_col.unique().tolist())
    info = {
        "id": dataset_id,
        "name": description or dataset_id,
        "platform": platform,
        "n_samples": int(len(type_col)),
        "n_features": len(header.columns) - 2,
        "n_classes": len(class_labels),
        "class_labels": class_labels,
        "description": description,
        "fs_models": {},
        "is_temp": True,
    }
    _TEMP_DATASETS[dataset_id] = {
        "cfg": {
            "path": path,
            "type": "auto",
            "platform": platform,
            "description": description,
            "annotation_file": annotation_file,
        },
        "info": info,
    }

    if _persist:
        try:
            reg_path = Path(path).parent / "registration.json"
            reg_path.write_text(
                json.dumps({
                    "dataset_id": dataset_id, "path": path, "platform": platform,
                    "description": description, "annotation_file": annotation_file,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass  # best-effort — a build that succeeds but can't persist the
            # registration.json still works for the rest of THIS session.

    return info


def restore_temp_datasets_from_disk() -> None:
    """Re-populate ``_TEMP_DATASETS`` from every ``registration.json`` still on
    disk under outputs_live/uploads/ — called before every ``list_datasets()``
    so a past upload reappears in the dropdown (and "Lịch sử tải lên" can
    re-select it) even after a server restart wiped the in-memory dict, as
    long as the built CSV itself is still there. Skips ids already registered
    this process (cheap early-exit — avoids re-reading pandas on every call).
    """
    if not UPLOAD_ROOT.exists():
        return
    for reg_path in UPLOAD_ROOT.glob("*/built/registration.json"):
        try:
            reg = json.loads(reg_path.read_text(encoding="utf-8"))
            dataset_id = reg["dataset_id"]
            if dataset_id in _TEMP_DATASETS:
                continue
            if not Path(reg["path"]).exists():
                continue
            register_temp_dataset(
                dataset_id,
                path=reg["path"], platform=reg.get("platform", ""),
                description=reg.get("description", ""),
                annotation_file=reg.get("annotation_file"),
                _persist=False,
            )
        except Exception:  # noqa: BLE001
            continue  # one corrupt/missing entry must not break the rest


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """Resolve a dataset id to its config — session-only uploads
    (:func:`register_temp_dataset`) first, then datasets.yaml. This is the ONE
    branch point every caller (holdout split, /api/datasets/*, ...) should go
    through instead of calling ``ConfigLoader.get_dataset_config`` directly,
    so a temp dataset behaves like a curated one everywhere without
    duplicating the "which registry" check at each call site.
    """
    if dataset_name not in _TEMP_DATASETS and dataset_name.startswith("live-upload-"):
        restore_temp_datasets_from_disk()
    if dataset_name in _TEMP_DATASETS:
        return dict(_TEMP_DATASETS[dataset_name]["cfg"])
    return get_config_loader().get_dataset_config(dataset_name)


def compute_holdout_split(dataset: str, split_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load + rare-class-drop + stratified split for one dataset — the same
    logic the cached holdout pipeline uses (configs/holdout.yaml), with
    ``min_samples_per_class``/``test_size`` overridable per call (Step 2's
    "Thực hiện lại") instead of only ever reading the yaml defaults. Always
    recomputed (no caching) — this is a filter + stratified split, not a
    training run, so it stays well under a second even on a large dataset.
    Shared by the live "Huấn luyện" jobs (src/api/jobs.py) and the
    ``/split/preview`` endpoint so both read train/test rows identically.
    """
    from src.helper.split import build_dataset_split

    loader = get_config_loader()
    holdout_cfg = dict(loader.load_yaml("holdout.yaml").get("holdout", {}))
    live_ui_defaults = dict(holdout_cfg.get("live_ui_defaults", {}))
    ds_cfg = get_dataset_config(dataset)

    sp = split_params or {}
    min_samples = int(sp.get("min_samples_per_class") or live_ui_defaults.get("min_samples_per_class", 5))
    test_size = float(sp.get("test_size") or live_ui_defaults.get("test_size", 0.2))
    split_seed = int(live_ui_defaults.get("split_random_state", 42))

    result = build_dataset_split(
        ds_cfg, min_samples_per_class=min_samples, test_size=test_size, random_state=split_seed,
    )
    result["ds_cfg"] = ds_cfg
    return result


def build_split_summary(dataset: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Same shape as the split-related fields of GET /api/datasets/{id}/overview,
    computed from a :func:`compute_holdout_split` context instead of read from
    ``outputs_holdout/{id}/split_info.json`` — lets ``/split/preview`` answer
    with a custom min_samples_per_class/test_size without any cache file.
    """
    class_labels = ctx["class_labels"]
    y, train_idx, test_idx = ctx["y"], ctx["train_idx"], ctx["test_idx"]
    train_class_counts = {lbl: int((y[train_idx] == i).sum()) for i, lbl in enumerate(class_labels)}
    test_class_counts = {lbl: int((y[test_idx] == i).sum()) for i, lbl in enumerate(class_labels)}
    return {
        "dataset": dataset,
        "class_labels": class_labels,
        "train_class_counts": train_class_counts,
        "test_class_counts": test_class_counts,
        "raw_class_counts": ctx["raw_class_counts"],
        "dropped_classes": ctx["dropped_classes"],
        "n_samples_total": int(len(train_idx) + len(test_idx)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_size": ctx["test_size"],
        "min_samples_per_class": ctx["min_samples_per_class"],
    }


# =============================================================================
# Live run history (outputs_live/runs_index.json)
# =============================================================================

RUNS_INDEX_PATH = LIVE_ROOT / "runs_index.json"


def list_runs(dataset_id: Optional[str] = None, kind: Optional[str] = None) -> List[Dict[str, Any]]:
    """Past live ('Huan luyen') jobs, newest first — read from the on-disk
    index so they survive a server restart even though JOBS (src/api/jobs.py)
    is in-memory only.
    """
    if not RUNS_INDEX_PATH.exists():
        return []
    try:
        runs = json.loads(RUNS_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    if dataset_id:
        runs = [r for r in runs if r.get("dataset") == dataset_id]
    if kind:
        runs = [r for r in runs if r.get("kind") == kind]
    return sorted(runs, key=lambda r: r.get("created_at", 0), reverse=True)


# =============================================================================
# Dataset overview extras: raw (pre-drop) class counts, provenance metadata,
# cached class descriptions.
# =============================================================================


def _compute_raw_class_counts_from_csv(dataset_id: str) -> Dict[str, int]:
    """Class distribution BEFORE rare-class removal — reads only the 'type'
    column (2nd column of the CuMiDa-format CSV) so this stays cheap even on
    a 50k-feature dataset, instead of loading the full matrix again. This is
    the raw source dataset, which never changes, so the result is only ever
    computed once — see :func:`load_raw_class_counts` for the cached path
    (outputs_holdout/{dataset}/raw_class_counts.json) that avoids re-reading
    this CSV on every request.
    """
    import pandas as pd

    try:
        ds_cfg = get_dataset_config(dataset_id)
    except ValueError:
        return {}
    path = Path(ds_cfg["path"])
    if not path.exists():
        return {}
    col = pd.read_csv(path, usecols=[1]).iloc[:, 0]
    return {str(k): int(v) for k, v in col.value_counts().items()}


def _raw_class_counts_cache_path(dataset_id: str) -> Path:
    return holdout_root_for() / dataset_id / "raw_class_counts.json"


@lru_cache(maxsize=32)
def load_raw_class_counts(dataset_id: str) -> Dict[str, Any]:
    """``{"raw_class_counts": {...}, "dropped_classes": [...]}``.

    Primary source: ``outputs_holdout/k{k}/{dataset}/split_info.json`` — since
    ``run_baseline_split()`` now computes and embeds ``raw_class_counts``/
    ``dropped_classes`` directly into ``split_info.json`` (and
    ``run_rule_extraction_holdout()`` copies that file into this flow's own
    output root), no separate cache file or manual export step is needed for
    any dataset produced by the current pipeline.

    Fallback (legacy ``outputs_holdout`` results written before this change,
    whose ``split_info.json`` doesn't have these two keys yet): the old
    separate cache file (``raw_class_counts.json``, written by
    ``scripts/export_raw_class_counts.py``), else a live CSV read. Cached
    in-memory per dataset_id for the process lifetime either way.
    """
    split_info_path = holdout_root_for() / dataset_id / "split_info.json"
    if split_info_path.exists():
        try:
            split_info = json.loads(split_info_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            split_info = {}
        if "raw_class_counts" in split_info and "dropped_classes" in split_info:
            return {
                "raw_class_counts": split_info["raw_class_counts"],
                "dropped_classes": split_info["dropped_classes"],
            }
    else:
        split_info = {}

    # --- Legacy fallback (pre-merge split_info.json, or none at all) ---
    cache_path = _raw_class_counts_cache_path(dataset_id)
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass

    raw_counts = _compute_raw_class_counts_from_csv(dataset_id)
    survived = set(split_info.get("class_labels", []))
    result = {
        "raw_class_counts": raw_counts,
        "dropped_classes": [c for c in raw_counts if c not in survived],
    }

    if raw_counts:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    return result


_GSE_RE = re.compile(r"GSE\d+")


@lru_cache(maxsize=1)
def _load_disease_context_library() -> Dict[str, Dict[str, Any]]:
    """``{GSE accession: parsed content json}`` — reads every file directly
    from Dataset_Description/ (the extracted, permanently-kept contents of
    biomedical_dataset_descriptions_vi_with_disease_context.zip), keyed by
    the zip's own "accession" field rather than filename, since naming
    conventions differ (Brain_GSE15824_probe.json vs our run-key
    CuMiDa-Brain-15824). Each file already has the GEO/CuMiDa build-time
    annotation report baked in under its "origin" key (see
    scripts/merge_annotation_into_description.py) — a single self-contained
    file per dataset, no second read from Dataset/ needed. Cached for the
    process lifetime — this content is static and only changes if
    Dataset_Description/ is edited and the process restarted.
    """
    library: Dict[str, Dict[str, Any]] = {}
    if not DATASET_DESCRIPTION_ROOT.exists():
        return library
    for path in DATASET_DESCRIPTION_ROOT.glob("*.json"):
        if path.name in ("index.json", "schema_notes_vi.json", "source_audit.json", "common_provenance_notes.json", "image_manifest.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        accession = data.get("accession")
        if accession:
            library[accession] = data
    return library


@lru_cache(maxsize=1)
def load_image_manifest() -> Dict[str, Dict[str, Any]]:
    """``{asset_id: asset metadata}`` from Dataset_Description/image_manifest.json
    (illustrative anatomy/microscopy images, NCI-sourced — see the file's own
    ``rights_notice_vi`` for reuse terms). A dataset's ``content.disease_context
    .media_asset_ids`` references keys into this map; the API resolves each ID
    to its metadata plus a servable ``url`` under ``/static/dataset-images/``.
    Empty dict if the manifest or images/ directory doesn't exist.
    """
    manifest_path = DATASET_DESCRIPTION_ROOT / "image_manifest.json"
    if not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = data.get("assets", {})
    for asset_id, meta in assets.items():
        filename = meta.get("filename")
        meta["url"] = f"/static/dataset-images/{filename}" if filename else None
    return assets


@lru_cache(maxsize=32)
def load_dataset_overview_static(dataset_id: str) -> Dict[str, Any]:
    """Static overview content for one dataset — a single read from
    Dataset_Description/{...}.json, matched by this dataset's GSE accession.
    Split into ``origin`` (build-time annotation report, baked into the file)
    and ``content`` (everything else: title_vi, description_vi, per-class
    display_name_vi/description_vi, disease_context, references).

    ``{}`` for either half (or both) if not found — frontend/predict just
    skip those panels/fields. Cached per dataset_id for the process
    lifetime (the source file is static, never written at runtime).
    """
    loader = get_config_loader()
    datasets_cfg = loader.load_datasets_config().get("datasets", {})
    base_name = dataset_id[: -len("-gene")] if dataset_id.endswith("-gene") else dataset_id
    description = datasets_cfg.get(base_name, {}).get("description", "")

    match = _GSE_RE.search(description) or _GSE_RE.search(dataset_id)
    if not match:
        return {}
    gse_id = match.group(0)

    data = _load_disease_context_library().get(gse_id)
    if not data:
        return {}
    content = {k: v for k, v in data.items() if k != "origin"}
    return {"origin": data.get("origin") or {}, "content": content}


def get_class_description(dataset_id: str, class_label: str) -> Optional[str]:
    """Vietnamese description for one class label, used to enrich the
    predict endpoint's "Du doan Phan loai" line — from
    ``content.classes[label].description_vi``.
    """
    content = load_dataset_overview_static(dataset_id).get("content") or {}
    classes = content.get("classes") or {}
    entry = classes.get(class_label)
    return entry.get("description_vi") if entry else None


def get_class_display_name(dataset_id: str, class_label: str) -> Optional[str]:
    """Vietnamese display name for one class label (``content.classes[label]
    .display_name_vi``) — the SAME name shown for this class in the Dataset
    Overview page, so a prediction result names classes consistently with it
    instead of only ever showing the raw internal label.
    """
    content = load_dataset_overview_static(dataset_id).get("content") or {}
    classes = content.get("classes") or {}
    entry = classes.get(class_label)
    return entry.get("display_name_vi") if entry else None
