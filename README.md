# Classification Transcriptomic with XAI

A reproducible pipeline for gene expression classification and biological interpretation using Explainable AI (XAI).

---

## Overview

This framework processes microarray gene expression datasets through three stages:

| Stage | Description |
|-------|-------------|
| **Feature Selection** | Raw / mRMR (`mrmr_k50`, `mrmr_k75`) / Boruta |
| **Benchmark** | Model comparison — NB / KNN / SVM / RF / DT / XGBoost / ANN |
| **Interpretation** | SHAP + tree-based IF-THEN rule extraction |

SHAP is an explainability tool, not a stage: it is applied to the best benchmark
model, to the rule-extraction model, and on demand to any trained model (see
`run_shap`). It and the rule-extraction settings are configured in `configs/interpretation.yaml`.

There is no command-line entry point — `main.py` is an unused stub. Every flow
in this repo is driven from the notebooks under `notebooks/` (see "Quick Start"
below), which can also be run headlessly via `jupyter nbconvert --execute`.

---

## Project Structure

```
LV_code/
├── configs/
│   ├── datasets.yaml          Dataset paths, class labels, and platform info
│   ├── dataset_origin.yaml    Raw GEO / CuMiDa build specs (dataset_builder)
│   ├── models.yaml            Model hyperparameters, conditional-scaling map, CV settings
│   ├── feature_selection.yaml Raw / mRMR / Boruta params + rare_class_k_values sweep
│   ├── holdout.yaml           Split-baseline + holdout-rule-extraction sweep config
│   └── interpretation.yaml    SHAP + tree-based rule extraction
│
├── Dataset/
│   ├── GEO/                   Raw GEO downloads (series-matrix + platform files)
│   ├── GEO_Cumida/            GEO datasets converted to CuMiDa format
│   └── Cumida/                Original CuMiDa format datasets (CSV)
│
├── src/
│   ├── pipeline/               Orchestration layer (GeneExpressionPipeline)
│   │   ├── core.py             Mixin composition + shared state
│   │   ├── benchmark.py        run_benchmark — CV/full-data baseline (+ k-sweep)
│   │   ├── baseline_split.py   run_baseline_split — train/test-split baseline, all models (+ k-sweep)
│   │   ├── holdout.py          run_rule_extraction_holdout — reuses baseline_split's split+FS
│   │   ├── rule_extraction.py  run_rule_extraction — full-dataset rules (no split)
│   │   └── interpretation.py   run_interpretation — reuses benchmark SHAP + rules for display
│   ├── helper/
│   │   ├── config_loader.py   YAML configuration management
│   │   ├── data_loader.py     CuMiDa format loading + missing-value/rare-class handling
│   │   ├── metrics.py         Classification metrics (incl. per-class breakdown)
│   │   ├── scaling.py         Conditional StandardScaler (tree models skip scaling)
│   │   ├── split.py           Stratified train/test split
│   │   ├── report.py          Console/notebook output + markdown provenance helpers
│   │   ├── paths.py           resolve_path / safe_filename — shared path helpers
│   │   ├── training.py        pick_best_run / summarize_repeats — repeated-training bookkeeping
│   │   └── wandb_utils.py     wandb_log_per_class — shared per-class W&B logging
│   ├── feature_selection/
│   │   ├── raw.py             No-op baseline
│   │   ├── dispatch.py        run_feature_selection — routes fs_method to raw/mrmr*/boruta
│   │   ├── mrmr_fs.py         mRMR via mrmr-selection (mrmr_k50 / mrmr_k75)
│   │   └── boruta_fs.py       Boruta via BorutaPy (incl. "auto" selection_mode)
│   ├── benchmark/
│   │   └── trainer.py         Stratified K-Fold CV trainer (model comparison)
│   ├── interpretation/
│   │   ├── shap_utils.py      SHAP for tree, linear, and neural models
│   │   ├── rules.py           Tree-based IF-THEN rule extraction (mechanics)
│   │   ├── rule_mining.py     Orchestration layer above rules.py (fit → mine → save → explain)
│   │   └── gene_annotation.py Post-hoc biological enrichment of genes-in-rules
│   ├── dataset_builder/       GEO series-matrix → CuMiDa CSV + probe→gene mapping
│   ├── visualize/             Dataset EDA / provenance figures + cross-dataset comparisons
│   ├── api/                   FastAPI backend serving the frontend UI
│   └── models/
│       ├── factory.py         Model instantiation by name
│       ├── nb.py              Gaussian Naive Bayes
│       ├── knn.py              k-Nearest Neighbors
│       ├── svm.py             SVM (RBF kernel)
│       ├── rf.py              Random Forest
│       ├── dt.py              Decision Tree
│       ├── xgb.py             XGBoost
│       └── ann.py             MLP (PyTorch Lightning)
│
├── notebooks/
│   ├── 00.a_Format_GEO_Datasets.ipynb     Raw GEO series matrix → CuMiDa CSV
│   ├── 00.b_Format_Cumida_Datasets.ipynb  Re-annotate existing CuMiDa CSV
│   ├── 00.c_Explore_Dataset.ipynb         Rank candidate class-label columns for a GEO dataset
│   ├── 01_Datasets_Visualization.ipynb    Dataset EDA figures — configurable rare-class K (default 0 = no
│   │                                       filtering); K>0 writes to outputs/visualizations_k{K}/
│   ├── 02_Benchmark.ipynb                 CV/full baseline + train/test-split baseline, k-sweep, with an
│   │                                       artifact-completeness check that resumes any missing combos
│   ├── 03_Rules_Extraction.ipynb          Full-dataset tree-based rules (no split; needs 02's single run)
│   └── 04_Rule_Extraction_Holdout.ipynb   Rule extraction on a train/test split (needs 02's k-sweep)
│
├── scripts/                    One-shot offline utilities (cache warming, LLM description generation)
├── frontend/                   React + TypeScript SPA calling src/api
│
├── outputs_baseline_full/      run_benchmark's k-swept CV baseline: k{N}/{dataset}/...
├── outputs_baseline_split/     run_baseline_split's k-swept train/test baseline: k{N}/{dataset}/...
├── outputs_holdout/            run_rule_extraction_holdout: k{N}/{dataset}/...
├── outputs_live/               Live UI "Huấn luyện lại" training jobs: <job_id>/...
└── outputs/                    Flat, no k-nesting — run_rule_extraction / run_interpretation (older, non-split flows)
```

---

## Dataset Format

All datasets use **CuMiDa format** — both original CuMiDa datasets and GEO datasets converted via `src/dataset_builder/builder.py`:

```
samples,  type,    probe_1, probe_2, ...
GSM001,   tumor,   4.32,    2.11,    ...
GSM002,   normal,  3.80,    1.95,    ...
```

Column 0 = sample ID, column 1 = class label, columns 2+ = expression values (float32).

Each dataset's platform (chip/array) is recorded in the `platform:` field of `configs/datasets.yaml`, used by `src/dataset_builder/annotation.py` to resolve probe → gene symbol mappings.

---

## Installation

This project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Key dependencies (see `pyproject.toml`):
```
numpy pandas scikit-learn xgboost
mrmr-selection          # mRMR feature selection
boruta                  # Boruta feature selection
torch lightning         # ANN model
shap                    # SHAP explanations
rich                    # clean console / notebook output
matplotlib seaborn umap-learn
mygene                  # optional: MyGene.info probe mapping
wandb                   # optional: experiment tracking
fastapi uvicorn         # backend API (src/api)
```

---

## Quick Start — Notebooks

There is no CLI — every flow is driven from a notebook. Minimal example (any notebook under `notebooks/`):

```python
import sys
sys.path.insert(0, '..')          # repo root

from src.pipeline import GeneExpressionPipeline

pipe = GeneExpressionPipeline(
    config_root='../configs',
    output_root='../outputs_baseline_full',
    use_wandb=False,
)
pipe.print_summary()

# CV/full-data baseline (auto-SHAP on the best combo) — no k-sweep, single run
pipe.run_benchmark(['GEO-Breast-20711'])

# k-sweep both baselines (see notebooks/02_Benchmark.ipynb §2 for the full cell)
K_VALUES = pipe.config_loader.load_feature_selection_config().get('rare_class_k_values', [2, 3, 4, 5, 6])
pipe.run_benchmark(['GEO-Breast-20711'], k_values=K_VALUES)          # -> outputs_baseline_full/k{N}/
pipe.run_baseline_split(['GEO-Breast-20711'], k_values=K_VALUES)     # -> outputs_baseline_split/k{N}/, all models

# Explain any other trained combo on demand
pipe.run_shap('GEO-Breast-20711', 'mrmr_k50', 'svm')

# Full-dataset tree-based rules (needs the single-run pipe.run_benchmark call above)
pipe.run_rule_extraction(['GEO-Breast-20711'])

# Rule extraction on a train/test split — needs outputs_baseline_split/k{N}/ to exist first
pipe.run_rule_extraction_holdout(['GEO-Breast-20711'], k_values=[3, 4, 5])
```

To run a notebook headlessly from a terminal:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_Benchmark.ipynb
```

---

## Run Modes

`02_Benchmark.ipynb`: set `RUN_MODE = "single"` (one dataset, `DATASET`) or
`"multiple"` (a list, `DATASETS`, or `None` = all enabled in `datasets.yaml`).
This selects which datasets both the single-run cell and the k-sweep section
operate on.

`run_interpretation_batch(select="best")` picks the highest balanced-accuracy `(fs_method, model, fold)`
from the benchmark results for each dataset; `select="fixed"` applies one explicit target to all.

---

## Feature Selection

Methods run **independently**, each producing its own reduced CSV. In the CV/full-data
baseline (`run_benchmark`) all four run; in the train/test-split baseline and
holdout rule extraction (`run_baseline_split` / `run_rule_extraction_holdout`)
only the last three run (no `raw`):

| Method | Description | Output |
|--------|-------------|--------|
| `raw` | All features (baseline) — CV/full baseline only | source pointer to original CSV (no copy) |
| `mrmr_k50` | mRMR MIQ criterion, top 50 features | `{dataset}_mrmr_k50_K50.csv` |
| `mrmr_k75` | mRMR MIQ criterion, top 75 features | `{dataset}_mrmr_k75_K75.csv` |
| `boruta` | Boruta, `selection_mode: "auto"` — confirmed if ≥50 confirmed features, else confirmed+tentative | `{dataset}_boruta_confirmed.csv` or `_confirmed_tentative.csv` |

Configure the mRMR K values, Boruta's `dynamic_threshold`, and the rare-class
sweep range (`rare_class_k_values`) in `configs/feature_selection.yaml`.

---

## Evaluation Metrics

Each CV fold reports:

| Metric | Description |
|--------|-------------|
| `accuracy` | Overall accuracy |
| `balanced_accuracy` | Accuracy adjusted for class imbalance |
| `f1` | Averaged F1 score (averaging strategy from the `average` param) |
| `f1_macro` | Always macro-averaged F1, regardless of `average` |
| `precision` | Averaged precision |
| `recall` | Averaged recall |
| `roc_auc` | One-vs-Rest AUC |
| `log_loss` | Cross-entropy loss |
| `per_class` | Unaggregated precision/recall/f1 per class (`average=None`) — kept out of flat CSV summaries, written to a separate `per_class_metrics.json` |

Adaptive cross-validation: `n_splits = min(5, smallest_class_count)` — prevents errors when a class has very few samples.

---

## Probe → Gene Annotation

Datasets use probe IDs as feature names (e.g. `1007_s_at`). The `platform:` field in `configs/datasets.yaml` records each dataset's chip platform. Passed to `src/dataset_builder/annotation.py`, it enables automatic GPL SOFT download and probe → gene symbol mapping:

```python
# Automatic GPL download (requires internet) — platform from datasets.yaml
pipe.run_interpretation(
    dataset_name='GEO-Breast-20711',
    fs_method='mrmr_k50',
    model_name='rf',
    platform='HG-U133_Plus_2',
)

# Local annotation file (highest priority)
pipe.run_interpretation(
    ...,
    annotation_path='path/to/GPL570.csv',
)
```

For Agilent datasets, MyGene.info is the recommended fallback annotation source (used automatically when a local annotation file/mapping isn't available).

---

## Outputs per Experiment

Three separate, `k{N}`-nested output roots plus one flat legacy root:

```
outputs_baseline_full/k{N}/{dataset}/{fs_method}/{model}/     <- run_benchmark(k_values=[...])
  cv_results.csv, cv_summary.csv, per_class_metrics.json, params_des.json, models/{model}_fold_N.joblib

outputs_baseline_split/k{N}/{dataset}/                        <- run_baseline_split(k_values=[...])
  split_info.json, feature_selection/{fs_method}/...
  {fs_method}/{model}/{metrics.json, per_class_metrics.json,
    models/{model}.joblib, models/repeats.csv, models/repeats_summary.csv, models/confusion_matrix_test.png}
  visualizations/all_datasets_class_distribution.png, all_datasets_metric_comparison.png

outputs_holdout/k{N}/{dataset}/{fs_method}/{rf,dt}/            <- run_rule_extraction_holdout(k_values=[...])
  models/{cv_results.csv, cv_summary.csv, best_run.json, best_run_train.json, best_run_per_class.json}
  rules/{rules.json, rules.csv, rules_human.txt, rules_summary.json, rules_test_eval.csv}

outputs/{dataset}/{fs_method}/                                 <- run_rule_extraction (no split, no k)
  {rf,dt}/rules/..., compare/crosscheck.json

outputs_live/<job_id>/                                          <- live UI "Huấn luyện lại" jobs
```

`k{N}` = the rare-class threshold (`min_samples_per_class`) that run produced results for — see `configs/feature_selection.yaml -> rare_class_k_values` and `configs/holdout.yaml -> holdout.batch.rare_class_k_values`.

---

## Adding a New Dataset

1. Place the CuMiDa-format CSV in `Dataset/Cumida/` (or `Dataset/GEO_Cumida/` for converted GEO data).
2. Add an entry in `configs/datasets.yaml`:
   ```yaml
   MyDataset:
     enabled: true
     path: "Dataset/Cumida/my_dataset/data.csv"
     type: "cumida"
     platform: "HG-U133_Plus_2"    # for annotation
     description: "My dataset description"
   ```
3. Run `pipe.run_benchmark(['MyDataset'])` (single run) or include it in a k-sweep call.

---

## Key Design Decisions

- **Feature selection on full dataset (CV/full baseline only)** — produces one exportable reduced dataset per method. This makes `run_benchmark`'s CV metrics optimistic relative to a true held-out estimate; `run_baseline_split`/`run_rule_extraction_holdout`'s train/test-split numbers are the leakage-free reference.
- **Conditional scaling** (`src/helper/scaling.py`) — `StandardScaler` is fit only for distance/gradient-based models (nb, knn, svm, ann); tree/boosting models (rf, dt, xgboost) split on raw feature values and are left unscaled.
- **`run_baseline_split` repeats training like the holdout flow** — each model is trained `n_repeats` times per `(k, fs_method, model)` (varying only the model's random seed, split fixed), keeping the repeat with the best `select_metric` on the test set (`configs/holdout.yaml -> holdout.split_baseline`); the discarded repeats are still summarized in `models/repeats.csv` / `repeats_summary.csv`. This mirrors `run_rule_extraction_holdout`'s repeat pattern so both flows report a best-of-N estimate rather than a single noisy seed.
- **`run_baseline_split` kept separate from the holdout rule-extraction flow** — the per-model training method in `HoldoutMixin` also backs the live UI's "Huấn luyện lại" training job, so the reporting-only, all-models train/test baseline lives in its own mixin (`BaselineSplitMixin`) rather than risking that shared code path.
- **Holdout reuses the split baseline's artifacts** — `run_rule_extraction_holdout` no longer derives its own train/test split or re-runs feature selection; it reads `run_baseline_split`'s split + FS output for the same k, avoiding duplicated compute and guaranteeing both flows are directly comparable.
- **mRMR fixed-K variants** — `mrmr_k50`/`mrmr_k75` (both MIQ criterion) replace a single configurable-K method, so both run side by side in every sweep without extra config plumbing.
- **Boruta `"auto"` selection mode** — resolves to `confirmed` if the confirmed-feature count is ≥50, else `confirmed_tentative`, so a dataset that yields too few confirmed features doesn't silently train on an under-powered feature set.
- **Per-class metrics** — added without changing any existing aggregate metric (`f1_macro` unchanged); kept out of flat CSV summaries to avoid breaking existing column-based consumers.
- **Adaptive n_splits** — automatically reduced when a class has fewer than 5 samples, preventing `StratifiedKFold` errors.
- **mrmr-selection** — F-statistic relevance with Pearson-correlation redundancy; no discretisation required.
- **Boruta** — uses BorutaPy with 500-tree RandomForest; confirms features via FWER-corrected statistics.
- **SHAP** — TreeExplainer for RF/DT/XGBoost (exact); KernelExplainer for SVM/NB/ANN (approximate).
- **W&B** — fully optional; disable with `use_wandb=False` or `use_wandb: false` in `models.yaml`.
