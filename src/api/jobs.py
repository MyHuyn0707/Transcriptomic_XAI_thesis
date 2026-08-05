"""
src/api/jobs.py

In-memory job registry for the "Huan luyen" (retrain) buttons. Runs real
pipeline code (src.feature_selection.run_feature_selection,
HoldoutMixin._extract_rules_for_model_holdout) as FastAPI BackgroundTasks,
writing to outputs_live/<job_id>/ — outputs_holdout/ is never touched by a
live run. Once a job finishes, the frontend re-fetches the normal GET
endpoints with ?run_id=<job_id>, so live results render through the exact
same code path as cached ("Tai log cu") results.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.feature_selection import run_feature_selection
from src.api.registry import (
    PROJECT_ROOT,
    LIVE_ROOT,
    get_config_loader,
    rule_model_folder,
    UI_TO_INTERNAL_MODEL,
)

JOBS: Dict[str, Dict[str, Any]] = {}
# Hard cap on the in-memory job registry — without this, JOBS grows forever
# (nothing else ever removes an entry) and a long-running server's memory use
# climbs unbounded across many "Huan luyen" clicks. Finished results already
# persist to disk (outputs_live/<job_id>/, plus runs_index.json), so evicting
# the in-memory record of an old *completed* job loses nothing durable —
# it just stops being pollable via GET /api/jobs/{job_id} once evicted.
_MAX_JOBS = 200

# Model hyperparameters the frontend's single "Cau hinh Rule Extraction" panel
# is allowed to override, per internal model name. The rest of that panel's
# fields (min_confidence, min_support, max_conditions, ...) are rule-
# simplification FILTER settings (configs/interpretation.yaml -> rules.filter),
# not sklearn constructor kwargs — mixing them into model_params used to crash
# every training run (unexpected keyword argument), so both are split out
# explicitly in _split_hyperparams below.
_RF_HYPERPARAM_KEYS = {"n_estimators", "max_depth", "min_samples_leaf", "class_weight", "random_state"}
_DT_HYPERPARAM_KEYS = {"max_depth", "min_samples_leaf", "ccp_alpha", "class_weight", "random_state"}
_FILTER_KEYS = {
    "min_confidence", "min_fidelity", "min_support", "min_abs_support",
    "max_conditions", "merge_same_gene", "dedup", "dedup_sig_figs",
    "merge_generalization", "max_rules_per_class", "max_rules_total",
}

RUNS_INDEX_PATH = LIVE_ROOT / "runs_index.json"


def _normalize_class_weight(internal_model: str, value: Any) -> Any:
    """The frontend shares one class_weight <select> (balanced / balanced_subsample
    / none) across RF and DT, but sklearn's DecisionTreeClassifier only accepts
    'balanced' or None — not the literal string 'none' (must be real None) and
    not 'balanced_subsample' (RF/bagging-only). Normalize both here instead of
    letting every training repeat fail silently.
    """
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    if internal_model == "decisiontree" and value == "balanced_subsample":
        return "balanced"
    return value


def _split_hyperparams(internal_model: str, raw: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    valid_keys = _RF_HYPERPARAM_KEYS if internal_model == "rf" else _DT_HYPERPARAM_KEYS
    model_kwargs = {
        k: _normalize_class_weight(internal_model, v) for k, v in (raw or {}).items()
        if k in valid_keys and v is not None and v != ""
    }
    filter_overrides = {k: v for k, v in (raw or {}).items() if k in _FILTER_KEYS}
    return model_kwargs, filter_overrides


def _evict_old_jobs() -> None:
    if len(JOBS) < _MAX_JOBS:
        return
    finished = [jid for jid, j in JOBS.items() if j.get("status") != "running"]
    # Insertion order == creation order (dict preserves it), so the front of
    # this list is the oldest finished job.
    for jid in finished[: len(JOBS) - _MAX_JOBS + 1]:
        JOBS.pop(jid, None)


def create_job() -> str:
    _evict_old_jobs()
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "log": [], "result": None, "error": None}
    return job_id


def _log(job_id: str, message: str) -> None:
    JOBS[job_id]["log"].append(message)


def get_job(job_id: str) -> Dict[str, Any]:
    if job_id not in JOBS:
        raise KeyError(f"Unknown job_id '{job_id}'")
    return JOBS[job_id]


_RUNS_INDEX_LOCK = threading.Lock()


def record_run(entry: Dict[str, Any]) -> None:
    """Append one completed/errored job to outputs_live/runs_index.json so past
    live runs can be listed and reloaded even after the in-memory JOBS
    registry is gone (server restart) — the frontend's "Lich su chay that"
    panel reads this via GET /api/datasets/{id}/runs.

    Guarded by a lock since BackgroundTasks from concurrent requests can call
    this from different worker threads; without it, two simultaneous
    read-modify-write cycles can clobber each other's appended entry.
    """
    with _RUNS_INDEX_LOCK:
        LIVE_ROOT.mkdir(parents=True, exist_ok=True)
        runs: List[Dict[str, Any]] = []
        if RUNS_INDEX_PATH.exists():
            try:
                runs = json.loads(RUNS_INDEX_PATH.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                runs = []
        runs.append(entry)
        RUNS_INDEX_PATH.write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_holdout_split(dataset: str, split_params: Optional[Dict[str, Any]] = None):
    """Thin wrapper over registry.compute_holdout_split — kept as a local name
    since every job function below already calls it as ``_load_holdout_split``;
    the actual load + rare-class-drop + stratified split logic lives in
    registry.py so it's shared with the /split/preview endpoint.
    """
    from src.api.registry import compute_holdout_split

    return compute_holdout_split(dataset, split_params)


def _normalize_fs_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """The frontend's Boruta config panel sends class_weight='none' (string,
    from a <select>) and max_depth='' (empty text input) — both must become
    real Python None before reaching RandomForestClassifier/BorutaPy, else
    every run fails with a sklearn validation error.
    """
    out = dict(params or {})
    if str(out.get("class_weight", "")).strip().lower() == "none":
        out["class_weight"] = None
    if "max_depth" in out and out["max_depth"] == "":
        out["max_depth"] = None
    if "k" in out and out["k"] == "":
        out["k"] = None
    return out


def run_feature_selection_job(
    job_id: str, dataset: str, fs_method: str, params: Dict[str, Any],
    split_params: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        _log(job_id, f"[HỆ THỐNG] Bắt đầu chạy thuật toán {fs_method.upper()} (live)...")
        params = _normalize_fs_params(params)
        ctx = _load_holdout_split(dataset, split_params)
        X_train = ctx["X"][ctx["train_idx"]]
        y_train = ctx["y"][ctx["train_idx"]]
        sample_ids_train = [ctx["sample_ids"][i] for i in ctx["train_idx"]]

        _log(job_id, f"[INFO] Đang phân tích {X_train.shape[1]} đặc trưng trên {X_train.shape[0]} mẫu train...")
        t0 = time.time()
        output_root = LIVE_ROOT / job_id
        output_root.mkdir(parents=True, exist_ok=True)

        result = run_feature_selection(
            fs_method, X_train, y_train, ctx["feature_names"], sample_ids_train,
            dataset, output_root, params, allow_raw=False,
        )
        elapsed = time.time() - t0
        n_sel = len(result.get("selected_features", []))
        _log(job_id, f"[THÀNH CÔNG] Đã hoàn tất trích xuất đặc trưng: {n_sel} đặc trưng ({elapsed:.1f}s).")

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {
            "run_id": job_id,
            "dataset": dataset,
            "fs_method": fs_method,
            "n_selected_features": n_sel,
            "runtime_seconds": elapsed,
        }
        record_run({
            "run_id": job_id, "kind": "feature_selection", "dataset": dataset,
            "fs_method": fs_method, "status": "done", "created_at": time.time(),
            "summary": {"n_selected_features": n_sel, "runtime_seconds": round(elapsed, 1)},
        })
    except Exception as e:  # noqa: BLE001
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
        _log(job_id, f"[LỖI] {e}")
        _log(job_id, traceback.format_exc())
        record_run({
            "run_id": job_id, "kind": "feature_selection", "dataset": dataset,
            "fs_method": fs_method, "status": "error", "created_at": time.time(),
            "summary": {"error": str(e)},
        })


def _selected_features_for(dataset: str, fs_method: str, fs_run_id: Optional[str]) -> List[str]:
    from src.api.registry import holdout_root_for

    holdout_root = holdout_root_for()
    root = (LIVE_ROOT / fs_run_id) if fs_run_id else holdout_root
    sel_path = root / "feature_selection" / dataset / fs_method / "selected_features" / "selected_features.json"
    if not sel_path.exists():
        sel_path = holdout_root / "feature_selection" / dataset / fs_method / "selected_features" / "selected_features.json"
    if not sel_path.exists():
        raise FileNotFoundError(
            f"No feature-selection cache for {dataset}/{fs_method}; run feature selection first."
        )
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    return sel.get("selected_features") or sel.get("confirmed") or []


def _describe_genes_for_run(job_id: str, dataset: str, ds_cfg: Dict[str, Any], output_dir: Path) -> None:
    """Write gene_description.csv/md next to output_dir/rules/genes_in_rules.csv
    (same annotation join the cached pipeline's describe_genes_in_rules_holdout
    does) — non-fatal, a missing/unparseable annotation file just means
    GET /genes has nothing to enrich with for this run.
    """
    try:
        from src.interpretation.gene_annotation import describe_gene_files, load_platform_annotation

        gene_file = output_dir / "rules" / "genes_in_rules.csv"
        if not gene_file.exists():
            return
        annotation_file = ds_cfg.get("annotation_file")
        if not annotation_file or not Path(annotation_file).exists():
            _log(job_id, "[INFO] Không có annotation_file — bỏ qua mô tả gene.")
            return
        annotation_df = load_platform_annotation(annotation_file)
        if annotation_df.empty:
            return
        describe_gene_files(
            [gene_file], annotation_df, dataset, str(ds_cfg.get("platform", "")),
            label_fn=lambda p: dataset,
        )
        _log(job_id, "[INFO] Đã sinh gene_description.csv từ platform annotation.")
    except Exception as e:  # noqa: BLE001
        _log(job_id, f"[CẢNH BÁO] Không thể sinh mô tả gene: {e}")


def run_rule_model_job(
    job_id: str,
    dataset: str,
    fs_method: str,
    model: str,
    hyperparams: Dict[str, Any],
    fs_run_id: Optional[str] = None,
    split_params: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        internal_model = UI_TO_INTERNAL_MODEL.get(model, model)
        folder = rule_model_folder(model)
        _log(job_id, f"[HỆ THỐNG] Bắt đầu huấn luyện mô hình {model.upper()} (live) trên {fs_method}...")

        from src.pipeline import GeneExpressionPipeline
        pipe = GeneExpressionPipeline(
            config_root=PROJECT_ROOT / "configs",
            output_root=LIVE_ROOT / job_id / "_scratch",
            use_wandb=False,
        )

        ctx = _load_holdout_split(dataset, split_params)
        feat_sel = _selected_features_for(dataset, fs_method, fs_run_id)
        idx_map = {f: i for i, f in enumerate(ctx["feature_names"])}
        sel_idx = np.array([idx_map[f] for f in feat_sel if f in idx_map], dtype=np.int64)

        X_train_sel = ctx["X"][ctx["train_idx"]][:, sel_idx]
        X_test_sel = ctx["X"][ctx["test_idx"]][:, sel_idx]
        y_train = ctx["y"][ctx["train_idx"]]
        y_test = ctx["y"][ctx["test_idx"]]

        probe_gene_map = pipe._build_probe_gene_map(ctx["ds_cfg"], ctx["feature_names"])

        model_kwargs, filter_overrides = _split_hyperparams(internal_model, hyperparams)

        rules_cfg = dict(pipe.config_loader.get_rules_config())
        rules_cfg["model_params"] = dict(rules_cfg.get("model_params", {}))
        rules_cfg["model_params"][internal_model] = {
            **rules_cfg["model_params"].get(internal_model, {}),
            **model_kwargs,
        }
        if filter_overrides:
            rules_cfg["filter"] = {**rules_cfg.get("filter", {}), **filter_overrides}
            _log(job_id, f"[INFO] Ghi đè filter luật: {filter_overrides}")
        shap_cfg = pipe.config_loader.get_shap_config()

        holdout_cfg = dict(pipe.config_loader.load_yaml("holdout.yaml").get("holdout", {}))
        batch_cfg = dict(holdout_cfg.get("batch", {}))
        n_repeats = int(batch_cfg.get("n_repeats", 10))
        base_seed = int(batch_cfg.get("repeat_base_seed", 0))
        select_metric = batch_cfg.get("select_metric", "f1_macro")

        output_dir = LIVE_ROOT / job_id / dataset / fs_method / folder
        _log(job_id, f"[INFO] Huấn luyện {n_repeats} lần lặp với tham số tùy chỉnh...")
        t0 = time.time()

        result = pipe._extract_rules_for_model_holdout(
            model_name=internal_model,
            X_train=X_train_sel, y_train=y_train,
            X_test=X_test_sel, y_test=y_test,
            feature_names=list(feat_sel),
            class_labels=ctx["class_labels"],
            probe_gene_map=probe_gene_map,
            rules_cfg=rules_cfg,
            shap_cfg=shap_cfg,
            output_dir=output_dir,
            n_repeats=n_repeats,
            base_seed=base_seed,
            select_metric=select_metric,
        )
        elapsed = time.time() - t0

        if result is None:
            raise RuntimeError("Tất cả các lần huấn luyện đều thất bại (xem log server).")

        summary = result["summary"]
        _log(
            job_id,
            f"[THÀNH CÔNG] Huấn luyện xong ({elapsed:.1f}s) — "
            f"{select_metric}={summary.get('best_run_test_metrics', {}).get(select_metric, 'n/a')}, "
            f"{summary.get('n_rules_kept', 0)} luật.",
        )

        # _extract_rules_for_model_holdout only writes genes_in_rules.csv —
        # gene_description.csv/md (GET /genes reads this) is a separate
        # enrichment step the cached pipeline runs afterwards
        # (describe_genes_in_rules_holdout); reproduce it here so a live run
        # is browsable the same way a cached one is.
        _describe_genes_for_run(job_id, dataset, ctx["ds_cfg"], output_dir)

        # Same batched Gemini description generation the offline script runs
        # for cached (outputs_holdout) results — best-effort here too, so a
        # live "Huan luyen" run's rules/genes are browsable with the same
        # LLM-written descriptions a cached run has, not blank ones.
        try:
            from src.api import llm
            llm.generate_bio_descriptions_for_rules_dir(output_dir / "rules", log=lambda m: _log(job_id, m))
        except Exception as e:  # noqa: BLE001
            _log(job_id, f"[CẢNH BÁO] Bỏ qua sinh mô tả Gemini: {e}")

        best_metrics = summary.get("best_run_test_metrics", {})
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {
            "run_id": job_id,
            "dataset": dataset,
            "fs_method": fs_method,
            "model": model,
            "n_rules": summary.get("n_rules_kept", 0),
            "runtime_seconds": elapsed,
        }
        record_run({
            "run_id": job_id, "kind": "model", "dataset": dataset,
            "fs_method": fs_method, "model": model, "status": "done",
            "created_at": time.time(),
            "summary": {
                "n_rules": summary.get("n_rules_kept", 0),
                "accuracy": best_metrics.get("accuracy"),
                "f1_macro": best_metrics.get(select_metric) or best_metrics.get("f1_macro"),
                "runtime_seconds": round(elapsed, 1),
            },
        })
    except Exception as e:  # noqa: BLE001
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
        _log(job_id, f"[LỖI] {e}")
        _log(job_id, traceback.format_exc())
        record_run({
            "run_id": job_id, "kind": "model", "dataset": dataset,
            "fs_method": fs_method, "model": model, "status": "error",
            "created_at": time.time(), "summary": {"error": str(e)},
        })


# =============================================================================
# Dataset upload (Bước 1 "Tải lên dataset mới") — session-only, never written
# to datasets.yaml/Dataset_Description. Raw files land under
# outputs_live/uploads/<upload_id>/; the built CuMiDa CSV lands under
# outputs_live/uploads/<upload_id>/built/ and is registered as a temp dataset
# (registry.register_temp_dataset) with id "live-upload-<upload_id>".
# =============================================================================

UPLOAD_ROOT = LIVE_ROOT / "uploads"


def create_upload_id() -> str:
    return uuid.uuid4().hex[:12]


def save_upload(upload_id: str, filename: str, content: bytes) -> Path:
    dir_ = UPLOAD_ROOT / upload_id
    dir_.mkdir(parents=True, exist_ok=True)
    # filename comes from the client's Content-Disposition header — take only
    # the base name so a crafted "../../x" (or an absolute path) can't escape
    # dir_ via a path-traversal write.
    safe_name = Path(filename).name
    if not safe_name or safe_name in (".", ".."):
        raise ValueError(f"Invalid upload filename: {filename!r}")
    path = dir_ / safe_name
    path.write_bytes(content)
    return path


def inspect_geo_upload(series_matrix_path: Path) -> List[str]:
    """Ranked class-characteristic candidates for a raw GEO series matrix —
    same ranking discover_class_characteristics uses, exposed here just as
    the plain characteristic names (UI shows them as a dropdown)."""
    from src.dataset_builder.builder import parse_series_matrix, discover_class_characteristics

    _, sample_meta, _ = parse_series_matrix(series_matrix_path)
    ranking = discover_class_characteristics(sample_meta)
    return ranking["characteristic"].tolist() if not ranking.empty else []


def inspect_cumida_upload(probe_csv_path: Path) -> List[str]:
    """CuMiDa CSVs are pre-labelled via their 'type' column — nothing to rank,
    just validate the layout and hand back that single, fixed option so the
    UI's characteristic dropdown has the same shape for both upload paths."""
    import pandas as pd

    df = pd.read_csv(probe_csv_path, nrows=0)
    if list(df.columns[:2]) != ["samples", "type"]:
        raise ValueError(
            f"File CuMiDa không đúng layout: cần 2 cột đầu là 'samples','type', "
            f"nhận được {list(df.columns[:2])}."
        )
    return ["type"]


def run_dataset_upload_build_job(
    job_id: str,
    upload_id: str,
    source: str,
    class_characteristic: Optional[str],
    tissue: str,
) -> None:
    try:
        upload_dir = UPLOAD_ROOT / upload_id
        output_dir = upload_dir / "built"
        _log(job_id, f"[HỆ THỐNG] Bắt đầu xây dựng dataset từ {source.upper()}...")

        if source == "geo":
            from src.dataset_builder.builder import build_geo_dataset

            series_matrix_path = next(
                p for p in upload_dir.iterdir() if p.is_file() and "series_matrix" in p.name.lower()
            )
            annotation_candidates = [p for p in upload_dir.iterdir() if p.is_file() and p != series_matrix_path]
            annotation_path = annotation_candidates[0] if annotation_candidates else None
            result = build_geo_dataset(
                series_matrix_path,
                class_characteristic=class_characteristic,
                annotation_path=annotation_path,
                output_dir=output_dir,
                tissue=tissue,
            )
        elif source == "cumida":
            from src.dataset_builder.cumida import build_cumida_dataset

            probe_candidates = [p for p in upload_dir.iterdir() if p.is_file() and p.suffix == ".csv"]
            if not probe_candidates:
                raise ValueError("Không tìm thấy file CSV CuMiDa đã tải lên.")
            probe_csv_path = probe_candidates[0]
            annotation_candidates = [p for p in upload_dir.iterdir() if p.is_file() and p != probe_csv_path]
            annotation_path = annotation_candidates[0] if annotation_candidates else None
            result = build_cumida_dataset(
                probe_csv_path,
                annotation_path=annotation_path,
                output_dir=output_dir,
                tissue=tissue,
            )
        else:
            raise ValueError(f"Nguồn upload không xác định: {source!r}")

        geo_id = result["geo_id"]
        platform = result.get("platform") or ""
        probe_path = result["output_files"]["probe_dataset"]

        dataset_id = f"live-upload-{upload_id}"
        from src.api.registry import register_temp_dataset

        info = register_temp_dataset(
            dataset_id,
            path=probe_path,
            platform=platform,
            description=f"[Tải lên] {tissue} ({geo_id})",
            annotation_file=str(annotation_path) if annotation_path else None,
        )

        _log(
            job_id,
            f"[THÀNH CÔNG] Đã xây dựng dataset '{dataset_id}' — "
            f"{info['n_samples']} mẫu, {info['n_features']} đặc trưng, {info['n_classes']} lớp.",
        )
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = {"dataset_id": dataset_id, **info}
        # "upload" runs aren't scoped to a pre-existing dataset (the dataset is
        # the OUTPUT of this job) — dataset=dataset_id lets Bước 1's "Lịch sử
        # tải lên" list every past upload across the session (queried without
        # a dataset filter) while still using the same runs_index.json/
        # record_run plumbing the FS/model history panels use, for consistency.
        record_run({
            "run_id": job_id, "kind": "upload", "dataset": dataset_id,
            "status": "done", "created_at": time.time(),
            "summary": {
                "source": source, "tissue": tissue, "geo_id": geo_id,
                "n_samples": info["n_samples"], "n_features": info["n_features"],
                "n_classes": info["n_classes"],
            },
        })
    except Exception as e:  # noqa: BLE001
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
        _log(job_id, f"[LỖI] {e}")
        _log(job_id, traceback.format_exc())
        record_run({
            "run_id": job_id, "kind": "upload", "dataset": None,
            "status": "error", "created_at": time.time(),
            "summary": {"source": source, "tissue": tissue, "error": str(e)},
        })
