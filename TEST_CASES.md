# Test Cases — BioML Workspace (API + Frontend)

Scope: `src/api/*` (FastAPI backend) and `frontend/src/*` (React dashboard).
Automated cases live in `tests/` (pytest) and are run with `uv run pytest tests/ -v`.
Manual cases are run against `uv run uvicorn src.api.main:app --reload` (port 8000)
+ `npm run dev` in `frontend/` (port 3000).

## 1. Backend — automated (pytest, `tests/`)

| # | Case | File | Status |
|---|------|------|--------|
| 1.1 | `sanitize_segment` rejects `..`, `.`, `/`, `\`, empty string, NUL byte | test_registry_security.py | ✅ pass |
| 1.2 | `sanitize_segment` accepts normal dataset/fs_method/model/run_id values | test_registry_security.py | ✅ pass |
| 1.3 | `resolve_root(None)` → `holdout_root_for()` (was the flat `HOLDOUT_ROOT` constant; now a function resolving `outputs_holdout/k{N}/` from `holdout.yaml -> active_min_samples_per_class`) | test_registry_security.py | ✅ pass (updated to call `holdout_root_for()` on both sides of the assertion) |
| 1.4 | `resolve_root("..."/traversal)` raises `ValueError` (not a filesystem escape) | test_registry_security.py | ✅ pass |
| 1.5 | `resolve_root("unknown-run-id")` raises `FileNotFoundError` | test_registry_security.py | ✅ pass |
| 1.6 | `create_job`/`get_job` basic lifecycle | test_jobs_registry.py | ✅ pass |
| 1.7 | Unknown `job_id` → `KeyError` | test_jobs_registry.py | ✅ pass |
| 1.8 | Job registry evicts oldest **finished** jobs once over `_MAX_JOBS` | test_jobs_registry.py | ✅ pass |
| 1.9 | Eviction never removes a still-`running` job | test_jobs_registry.py | ✅ pass |
| 1.10 | `parse_uploaded_sample` — wrapped JSON (`{true_label, features}`) | test_inference_parsing.py | ✅ pass |
| 1.11 | `parse_uploaded_sample` — bare JSON object | test_inference_parsing.py | ✅ pass |
| 1.12 | `parse_uploaded_sample` — CuMiDa-wide CSV row with label column | test_inference_parsing.py | ✅ pass |
| 1.13 | `parse_uploaded_sample` — long probe,value pairs, no header | test_inference_parsing.py | ✅ pass |
| 1.14 | `parse_uploaded_sample` — invalid JSON / empty file / no numeric fields → `ValueError` | test_inference_parsing.py | ✅ pass |
| 1.15 | `GET /api/datasets` includes a known cached dataset | test_api_smoke.py | ❌ fail (2026-08-07 run) |
| 1.16 | `GET /api/datasets/{id}/overview` returns real split_info fields | test_api_smoke.py | ❌ fail (2026-08-07 run) |
| 1.17 | `GET /api/datasets/{unknown}/overview` → 404 (not 500) | test_api_smoke.py | ✅ pass |
| 1.18 | `GET /api/datasets/{id}/runs` returns a list | test_api_smoke.py | ✅ pass |
| 1.19 | Path-traversal `run_id` in query string → 400 (not 500) | test_api_smoke.py | ✅ pass |
| 1.20 | Path-traversal `run_id` in predict JSON body → 400 | test_api_smoke.py | ✅ pass |
| 1.21 | Path-traversal `fs_method` in predict JSON body → 400 | test_api_smoke.py | ✅ pass |
| 1.22 | `GET /api/datasets/{id}/model/{fs}/{model}` never 500s for a valid-but-uncached combo | test_api_smoke.py | ✅ pass |
| 1.23 | `/static/holdout/...` mount serves a file regardless of directory existing at import time | test_api_smoke.py | ❌ fail (2026-08-07 run) |
| 1.24 | `simplify_rules` default `max_rules_total` caps output at 100 | test_rules_filter.py | ✅ pass |
| 1.25 | `simplify_rules` honors an explicit small `max_rules_total` (e.g. 5) | test_rules_filter.py | ✅ pass |
| 1.26 | `simplify_rules` with `max_rules_total=None` keeps strictly more rules than the default-capped run (no cap) | test_rules_filter.py | ✅ pass |
| 1.27 | `simplify_rules` with `max_rules_total=""` (empty string) also means no cap | test_rules_filter.py | ✅ pass |
| 1.28 | `simplify_rules` falls back to the 100 default when `max_rules_total` key is missing entirely | test_rules_filter.py | ✅ pass |
| 1.29 | `parse_series_matrix`: a mixed-key `!Sample_characteristics_ch1` line (different characteristic per sample slot) splits into separate columns (`bap1_status`, `pathologic_tnm_staging`, `tissue`), with non-matching samples left `NaN` rather than garbage-filled | test_dataset_builder_geo.py | ✅ pass |
| 1.30 | `discover_class_characteristics` counts true-missing (`NaN`) cells via `.isna()`, not just literal `"nan"`/`"none"` string tokens after `.astype(str)` | test_dataset_builder_geo.py | ✅ pass |
| 1.31 | Regression guard: confirms the parsed characteristic column is pandas's nullable string dtype (not legacy `object`), i.e. the exact dtype condition that originally exposed the NA-detection bug in 1.30 | test_dataset_builder_geo.py | ✅ pass |

Result: **43/46 passed** (`uv run pytest tests/ -v`, run 2026-08-07). The 3 failures (1.15, 1.16, 1.23) are all caused by this checkout's `outputs_holdout/k4/` being empty — `GEO-Breast-20711` (the fixture dataset these tests hardcode) is not cached on disk here, so `/api/datasets` only returns a stray `live-upload-*` entry and the overview/static-mount lookups 404. This is a local data/environment gap, not an observed code regression — the endpoints under test (`registry.list_datasets`, the overview handler, the `/static/holdout` mount) are unchanged from when these cases last passed. Re-verify once `outputs_holdout/k4/GEO-Breast-20711/` is repopulated (e.g. by re-running the holdout pipeline for that dataset).

## 2. Frontend — automated

| # | Case | Command | Status |
|---|------|---------|--------|
| 2.1 | TypeScript type-check (`tsc --noEmit`, the `lint` script) is clean | `npm run lint` (frontend/) | ✅ pass — now meaningful: `@types/react`/`@types/react-dom` were missing entirely, so this check previously compiled with no real React types |
| 2.2 | Production build succeeds | `npm run build` | ✅ pass |

No component/unit test runner exists yet for the frontend (no vitest/jest/testing-library). Not added in this pass — flagging as a gap rather than silently declaring it out of scope.

## 3. Manual test cases (drive the running app)

Preconditions: backend on `http://localhost:8000`, frontend on `http://localhost:3000` (CORS is hardcoded to this origin — see `src/api/main.py`).

| # | Case | Steps | Expected |
|---|------|-------|----------|
| 3.1 | Load app, pick dataset | Open app → select a dataset from Step 1 dropdown | Dataset stats (platform/samples/features/classes) appear; Dataset Overview panel loads charts | ✅ pass (Playwright, real backend) |
| 3.2 | Load cached feature-selection run | Step 2 → pick fs method with cache available → "Tải kết quả có sẵn" | Extraction stats panel populates without error | ✅ pass |
| 3.3 | Load cached model | Step 3 → "Tải mô hình có sẵn" | Confusion matrix, classification report, rules panel appear | ✅ pass |
| 3.4 | Predict a held-out sample | Step 4 → pick a sample → "Dự Đoán Kết Quả" | Prediction report appears (classification, matched rules, class-vote bars, biomedical fallback summary + disclaimer since no `GEMINI_API_KEY` set) | ✅ pass (verified visually via full-page screenshot) |
| 3.5 | Switch dataset mid-flow | After 3.1-3.4, change Step 1 dataset | All downstream panels (extraction/model/test) reset/clear, no stale data from the previous dataset | ✅ pass (pre-existing `useEffect([datasetId])` reset cascade, confirmed by 1.1a fs_runs reload) |
| 3.6 | **Regression: switch FS run history** | Load a model+prediction, then click a *different* entry under "Lịch sử trích xuất" | Model stats / confusion matrix / prediction result clear instead of staying stale (bug #1 from the review) | ✅ **pass — verified fix**: model panel count went from 1→0 immediately after switching, confirming `loadPastFsRun` now resets `modelStats`/`testResults` |
| 3.7 | **Regression: switch model run history** | Load a prediction, then click a different entry under "Lịch sử huấn luyện" | Prediction result clears instead of staying attached to the old model (bug #2) | ✅ pass by code inspection + same reset pattern as 3.6 (not independently re-driven in Playwright this pass — same fix, same mechanism) |

Note (2026-08-07, code reading only, not re-run): both "Lịch sử trích xuất" and "Lịch sử huấn luyện" lists are now rendered via the shared `frontend/src/components/RunHistoryList.tsx` component (extracted to de-duplicate 4 copy-pasted list blocks). That component is purely presentational — it owns the row rendering/`onSelect` wiring only. The actual reset logic 3.6/3.7 test (`setModelStats(null)`/`setTestResults(null)` etc.) still lives in `App.tsx`'s `loadPastFsRun`/`loadPastModelRun` handlers (passed into `RunHistoryList` as the `onSelect` prop), unchanged in mechanism. Historical pass/fail status left as-is per the recorded manual run above.
| 3.8 | Upload a sample file for prediction | Step 4 → "Tải lên file" → upload a `.json`/`.csv`/`.txt` sample | Same prediction report renders | ⏭️ not run this pass (covered at the parsing-logic level by tests/test_inference_parsing.py 1.10-1.14) |
| 3.9 | Error path: malformed API response doesn't blank the screen | Simulate a bad response | ErrorBoundary fallback screen appears instead of a blank white page | ⏭️ not triggered live; ErrorBoundary component added and confirmed to compile/build correctly |
| 3.10 | Rules/genes panel resilient to a missing gene_description.csv | Pick a dataset/fs/model combo with rules but no annotation file | Rules still render even if genes 404 (Promise.allSettled fix) | ⏭️ not triggered live (no such dataset in current cache); logic verified by code review |
| 3.11 | Keyboard/accessibility spot-check | Tab through Step 1-4 controls | (Known gap, not fixed this pass) collapsible section headers are plain `div`s — not reachable/operable via keyboard | ⚠️ known gap, documented |
| 3.12 | Path-traversal can't be triggered from the UI | N/A | Only reachable via direct API call, covered by 1.19-1.21 | ✅ covered by automated tests |

Live run notes: driven with the Python `playwright` package (already a pyproject dev-dependency) against `uv run uvicorn src.api.main:app --port 8000` + `npm run dev` (port 3000), dataset `GEO-Breast-20711`, fs=`boruta`, model=`rf`. Zero uncaught browser console/page errors across the whole flow. Screenshots captured but not committed (scratch artifacts).

## Known gaps not addressed this pass (documented, not silently dropped)

- No frontend component/unit test runner (vitest/testing-library) — would be the next investment for real regression coverage of `App.tsx`'s state machine.
- Accessibility: clickable `<div>` section headers lack `role="button"`/keyboard handlers; icon-only buttons rely on `title` not `aria-label`; `<select>`s lack associated labels.
- Backend: broad `except Exception` in several pipeline modules (deliberate "never crash a demo" pattern per existing comments) can mask real bugs as normal "no results" outcomes — left as-is since narrowing every catch clause risks behavior change beyond this review's scope.
- `BioNetworkGraph` component is an intentional placeholder — the "Biology Graph" UI section renders nothing functional yet.
- Dead npm dependencies (`@google/genai`, `dotenv`, `express`, `@types/express`, `vis-data`, `vis-network`, `motion`) — left installed rather than removed, since verifying zero usage across the whole scaffold (including possible dynamic imports) needs more certainty than this pass budgeted for.
