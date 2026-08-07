"""
src/api/main.py

FastAPI app serving the BioML defense UI: all endpoints read directly from
outputs_holdout/ (cached "Tai log cu") or outputs_live/<run_id>/ (a completed
live "Huan luyen" job) — no database, everything is a static file on disk
read per-request.

Run: uv run uvicorn src.api.main:app --reload

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

# Must run before anything (transitively) does `import matplotlib.pyplot` —
# BackgroundTasks (src/api/jobs.py -> the pipeline's plot_* helpers) render on
# a worker thread, and matplotlib's default interactive backend on this
# machine is TkAgg; Tkinter is not thread-safe, so a plt.savefig/plt.close
# from a background thread crashes the whole process (Tcl_AsyncDelete: async
# handler deleted by the wrong thread), not just that one job. Agg is the
# non-interactive, thread-safe raster backend — exactly what a server needs.
import matplotlib
matplotlib.use("Agg")

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api import jobs, registry
from src.api.registry import LIVE_ROOT, PROJECT_ROOT
from src.api.schemas import PredictRequest

load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(title="BioML Workspace API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {path.relative_to(PROJECT_ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    import pandas as pd

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Not found: {path.relative_to(PROJECT_ROOT)}")
    return pd.read_csv(path).fillna("").to_dict("records")


def _root(run_id: Optional[str]) -> Path:
    try:
        return registry.resolve_root(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _seg(value: str, name: str) -> str:
    """Validate a request-supplied value used as a filesystem path segment
    (dataset_id / fs_method / model) — see registry.sanitize_segment."""
    try:
        return registry.sanitize_segment(value, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Static files (confusion matrix / split PNGs)
# =============================================================================

_holdout_root = registry.holdout_root_for()
_holdout_root.mkdir(parents=True, exist_ok=True)
app.mount("/static/holdout", StaticFiles(directory=str(_holdout_root)), name="static-holdout")
LIVE_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/static/live", StaticFiles(directory=str(LIVE_ROOT)), name="static-live")

_dataset_images_dir = registry.DATASET_DESCRIPTION_ROOT / "images"
if _dataset_images_dir.exists():
    app.mount("/static/dataset-images", StaticFiles(directory=str(_dataset_images_dir)), name="static-dataset-images")


# =============================================================================
# Datasets
# =============================================================================


@app.get("/api/datasets")
def get_datasets():
    return registry.list_datasets()


@app.get("/api/datasets/{dataset_id}/overview")
def get_dataset_overview(dataset_id: str):
    dataset_id = _seg(dataset_id, "dataset_id")
    ds_root = registry.holdout_root_for() / dataset_id
    split_info = _read_json(ds_root / "split_info.json")
    viz_dir = ds_root / "visualizations"
    visualizations = {
        f.stem: f"/static/holdout/{dataset_id}/visualizations/{f.name}"
        for f in (viz_dir.glob("*.png") if viz_dir.exists() else [])
    }
    raw = registry.load_raw_class_counts(dataset_id)
    static = registry.load_dataset_overview_static(dataset_id)
    content = static.get("content") or {}

    # Resolve this dataset's illustrative image(s) (content.disease_context
    # .media_asset_ids -> image_manifest.json) to servable metadata — only
    # the assets this dataset actually references, not the whole manifest.
    asset_ids = (content.get("disease_context") or {}).get("media_asset_ids") or []
    manifest = registry.load_image_manifest()
    media_assets = {aid: manifest[aid] for aid in asset_ids if aid in manifest}

    return {
        **split_info,
        "visualizations": visualizations,
        "raw_class_counts": raw["raw_class_counts"],
        "dropped_classes": raw["dropped_classes"],
        "origin": static.get("origin") or {},
        "content": content,
        "media_assets": media_assets,
    }


@app.get("/api/datasets/{dataset_id}/runs")
def get_dataset_runs(dataset_id: str, kind: Optional[str] = Query(None)):
    dataset_id = _seg(dataset_id, "dataset_id")
    return registry.list_runs(dataset_id=dataset_id, kind=kind)


@app.post("/api/datasets/{dataset_id}/split/preview")
def split_preview(dataset_id: str, body: Optional[Dict[str, Any]] = None):
    """Recompute the holdout split (rare-class drop + stratified train/test)
    with custom min_samples_per_class/test_size, without writing any file —
    Step 2's "Thực hiện lại" (and the "Tải dữ liệu có sẵn" fallback for
    session-only uploaded datasets, which have no cached split to load).
    Same response shape as the split-related fields of GET .../overview so
    the frontend can reuse one renderer for both. Every successful call is
    logged as a "split" run (mirrors every FS/model "Huấn luyện lại" click
    logging a run) so Bước 2 can show a "Lịch sử xử lý và phân chia dữ liệu"
    panel the same way Bước 3/4 already do.
    """
    import time as _time
    import uuid as _uuid

    dataset_id = _seg(dataset_id, "dataset_id")
    try:
        ctx = registry.compute_holdout_split(dataset_id, body or {})
        summary = registry.build_split_summary(dataset_id, ctx)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Store the FULL split_summary (not just a reduced digest) so a past run
    # can be "loaded" straight from runs_index.json — same as every FS/model
    # history entry's "load" behavior — instead of needing to recompute (and
    # thus append yet another history entry) just to view it again.
    jobs.record_run({
        "run_id": _uuid.uuid4().hex[:12], "kind": "split", "dataset": dataset_id,
        "status": "done", "created_at": _time.time(),
        "summary": summary,
    })
    return summary


# =============================================================================
# Dataset upload (Bước 1 "Tải lên dataset mới")
# =============================================================================


@app.post("/api/datasets/upload/inspect")
async def upload_inspect(
    source: str = Form(...),
    tissue: str = Form("Dataset"),
    file1: UploadFile = File(...),
    file2: Optional[UploadFile] = File(None),
):
    if source not in ("geo", "cumida"):
        raise HTTPException(status_code=400, detail="source phải là 'geo' hoặc 'cumida'")
    upload_id = jobs.create_upload_id()
    try:
        path1 = jobs.save_upload(upload_id, file1.filename or "file1", await file1.read())
        if file2 is not None and file2.filename:
            jobs.save_upload(upload_id, file2.filename, await file2.read())
        characteristics = (
            jobs.inspect_geo_upload(path1) if source == "geo" else jobs.inspect_cumida_upload(path1)
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))
    return {"upload_id": upload_id, "source": source, "tissue": tissue, "characteristics": characteristics}


@app.post("/api/datasets/upload/build")
def upload_build(body: Dict[str, Any], background_tasks: BackgroundTasks):
    upload_id = body.get("upload_id")
    source = body.get("source")
    if not upload_id or not source:
        raise HTTPException(status_code=400, detail="Thiếu upload_id/source.")
    job_id = jobs.create_job()
    background_tasks.add_task(
        jobs.run_dataset_upload_build_job,
        job_id, upload_id, source, body.get("class_characteristic"), body.get("tissue") or "Dataset",
    )
    return {"job_id": job_id, "status": "running"}


@app.get("/api/datasets/upload/history")
def upload_history():
    """Every past "Tải lên dataset mới" build this session (or a previous one,
    read from the same runs_index.json FS/model history already persists to) —
    unlike GET /api/datasets/{id}/runs, this is NOT scoped to a currently
    selected dataset, since the dataset an upload produces is exactly what
    Bước 1's history list lets the user pick.
    """
    return registry.list_runs(kind="upload")


# =============================================================================
# Feature selection
# =============================================================================


@app.get("/api/datasets/{dataset_id}/feature-selection/{fs_method}")
def get_feature_selection(dataset_id: str, fs_method: str, run_id: Optional[str] = Query(None)):
    dataset_id, fs_method = _seg(dataset_id, "dataset_id"), _seg(fs_method, "fs_method")
    root = _root(run_id)
    base = root / "feature_selection" / dataset_id / fs_method / "params"
    metadata = _read_json(base / "metadata.json")
    params_des_path = base / "params_des.json"
    if params_des_path.exists():
        metadata.update(json.loads(params_des_path.read_text(encoding="utf-8")))
    return metadata


@app.post("/api/datasets/{dataset_id}/feature-selection/{fs_method}/train")
def train_feature_selection(dataset_id: str, fs_method: str, params: Dict[str, Any], background_tasks: BackgroundTasks):
    dataset_id, fs_method = _seg(dataset_id, "dataset_id"), _seg(fs_method, "fs_method")
    split_params = params.pop("split_params", None)
    job_id = jobs.create_job()
    background_tasks.add_task(jobs.run_feature_selection_job, job_id, dataset_id, fs_method, params, split_params)
    return {"job_id": job_id, "status": "running"}


# =============================================================================
# Rule model
# =============================================================================


@app.get("/api/datasets/{dataset_id}/model/{fs_method}/{model}")
def get_model_stats(dataset_id: str, fs_method: str, model: str, run_id: Optional[str] = Query(None)):
    dataset_id, fs_method, model = _seg(dataset_id, "dataset_id"), _seg(fs_method, "fs_method"), _seg(model, "model")
    root = _root(run_id)
    folder = registry.rule_model_folder(model)
    base = root / dataset_id / fs_method / folder

    best_run = _read_json(base / "models" / "best_run.json")
    params_des = _read_json(base / "models" / "params_des.json")
    rules_summary = _read_json(base / "rules" / "rules_summary.json")

    cm = best_run.get("confusion_matrix", [])
    rules_json_path = base / "rules" / "rules.json"
    class_labels = []
    if rules_json_path.exists():
        class_labels = json.loads(rules_json_path.read_text(encoding="utf-8")).get("class_labels") or []

    cm_root = "live" if run_id else "holdout"
    cm_rel = f"{run_id}/" if run_id else ""

    return {
        "model_name": model,
        "best_run_test_metrics": {
            k: v for k, v in best_run.items() if k not in ("confusion_matrix", "run", "seed")
        },
        "confusion_matrix": cm,
        "class_labels": class_labels,
        "n_rules": rules_summary.get("n_rules", 0),
        "hyperparams": params_des.get("model_hyperparams", {}),
        "confusion_matrix_png": f"/static/{cm_root}/{cm_rel}{dataset_id}/{fs_method}/{folder}/models/confusion_matrix_test.png",
        "rules_summary": rules_summary,
    }


@app.post("/api/datasets/{dataset_id}/model/{fs_method}/{model}/train")
def train_model(dataset_id: str, fs_method: str, model: str, body: Dict[str, Any], background_tasks: BackgroundTasks):
    dataset_id, fs_method, model = _seg(dataset_id, "dataset_id"), _seg(fs_method, "fs_method"), _seg(model, "model")
    hyperparams = body.get("hyperparams", body)
    fs_run_id = body.get("fs_run_id")
    split_params = body.get("split_params")
    job_id = jobs.create_job()
    background_tasks.add_task(
        jobs.run_rule_model_job, job_id, dataset_id, fs_method, model, hyperparams, fs_run_id, split_params,
    )
    return {"job_id": job_id, "status": "running"}


# =============================================================================
# Jobs
# =============================================================================


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return {"job_id": job_id, **jobs.get_job(job_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")


# =============================================================================
# Rules & genes
# =============================================================================


@app.get("/api/datasets/{dataset_id}/rules/{fs_method}/{model}")
def get_rules(dataset_id: str, fs_method: str, model: str, run_id: Optional[str] = Query(None)):
    dataset_id, fs_method, model = _seg(dataset_id, "dataset_id"), _seg(fs_method, "fs_method"), _seg(model, "model")
    root = _root(run_id)
    folder = registry.rule_model_folder(model)
    base = root / dataset_id / fs_method / folder / "rules"

    payload = _read_json(base / "rules.json")
    rules_llm_path = base / "rules_llm.json"
    rules_llm = json.loads(rules_llm_path.read_text(encoding="utf-8")) if rules_llm_path.exists() else {}

    for r in payload.get("rules", []):
        r["explanation"] = rules_llm.get(str(r["rule_id"]))
    return payload


@app.get("/api/datasets/{dataset_id}/genes/{fs_method}/{model}")
def get_genes(dataset_id: str, fs_method: str, model: str, run_id: Optional[str] = Query(None)):
    dataset_id, fs_method, model = _seg(dataset_id, "dataset_id"), _seg(fs_method, "fs_method"), _seg(model, "model")
    root = _root(run_id)
    folder = registry.rule_model_folder(model)
    base = root / dataset_id / fs_method / folder / "rules"

    genes = _read_csv_rows(base / "gene_description.csv")
    llm_path = base / "gene_description_llm.json"
    gene_llm = json.loads(llm_path.read_text(encoding="utf-8")) if llm_path.exists() else {}
    for g in genes:
        g["description_vn"] = gene_llm.get(g.get("gene"))
    return genes


# =============================================================================
# Test samples & prediction
# =============================================================================


@app.get("/api/datasets/{dataset_id}/test-samples/{fs_method}")
def get_test_samples(
    dataset_id: str,
    min_samples_per_class: Optional[int] = Query(None),
    test_size: Optional[float] = Query(None),
):
    dataset_id = _seg(dataset_id, "dataset_id")
    if min_samples_per_class is not None or test_size is not None:
        from src.api.inference import get_test_samples as get_test_samples_live

        split_params = {}
        if min_samples_per_class is not None:
            split_params["min_samples_per_class"] = min_samples_per_class
        if test_size is not None:
            split_params["test_size"] = test_size
        return get_test_samples_live(dataset_id, split_params)
    manifest = _read_csv_rows(registry.holdout_root_for() / dataset_id / "test_set" / "manifest.csv")
    return manifest


@app.get("/api/datasets/{dataset_id}/test-samples/{fs_method}/download")
def download_test_set(
    dataset_id: str,
    min_samples_per_class: Optional[int] = Query(None),
    test_size: Optional[float] = Query(None),
):
    """ZIP of the held-out test set (test_set.csv, manifest.csv,
    samples/*.json) — for transparency (audit exactly which rows were held
    out) and for re-testing via the existing "Tải lên file" predict flow.
    Works for both an existing cached dataset AND a brand-new upload/custom
    split with no cache yet (see inference.build_test_set_zip).
    """
    dataset_id = _seg(dataset_id, "dataset_id")
    from src.api.inference import build_test_set_zip

    split_params: Dict[str, Any] = {}
    if min_samples_per_class is not None:
        split_params["min_samples_per_class"] = min_samples_per_class
    if test_size is not None:
        split_params["test_size"] = test_size

    try:
        zip_bytes = build_test_set_zip(dataset_id, split_params or None)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{dataset_id}_test_set.zip"'},
    )


@app.post("/api/datasets/{dataset_id}/predict")
async def predict(dataset_id: str, req: PredictRequest):
    from src.api.inference import predict_sample

    dataset_id = _seg(dataset_id, "dataset_id")
    fs_method = _seg(req.fs_method, "fs_method")
    model = _seg(req.model, "model")
    run_id = _seg(req.run_id, "run_id") if req.run_id else req.run_id
    try:
        return await predict_sample(
            dataset_id, fs_method, model, req.sample_id, run_id, req.split_params,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/datasets/{dataset_id}/predict-upload")
async def predict_upload(
    dataset_id: str,
    fs_method: str = Form(...),
    model: str = Form(...),
    run_id: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    from src.api.inference import parse_uploaded_sample, predict_from_features

    dataset_id = _seg(dataset_id, "dataset_id")
    fs_method = _seg(fs_method, "fs_method")
    model = _seg(model, "model")
    run_id = _seg(run_id, "run_id") if run_id else run_id
    raw = (await file.read()).decode("utf-8", errors="replace")
    try:
        features, true_label = parse_uploaded_sample(raw)
        return await predict_from_features(
            dataset_id, fs_method, model, features, true_label=true_label, run_id=run_id,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# =============================================================================
# Compare (optional)
# =============================================================================


@app.get("/api/datasets/{dataset_id}/compare/{fs_method}")
def get_compare(dataset_id: str, fs_method: str, run_id: Optional[str] = Query(None)):
    dataset_id, fs_method = _seg(dataset_id, "dataset_id"), _seg(fs_method, "fs_method")
    root = _root(run_id)
    return _read_json(root / dataset_id / fs_method / "compare" / "crosscheck.json")
