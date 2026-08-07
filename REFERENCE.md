# Code Reference — Classification Transcriptomic with XAI

Complete reference for every file, function, parameter, input, and output in `src/`.

---

## Table of Contents

1. [Project layout](#1-project-layout)
2. [Data flow overview](#2-data-flow-overview)
3. [src/\_\_init\_\_.py](#3-src__init__py)
4. [src/helper/config\_loader.py](#4-srchelperconfigloaderpy)
5. [src/helper/data\_loader.py](#5-srchelperdataloaderpy)
6. [src/feature\_selection/](#6-srcfeature_selection)
   - [utils.py](#61-utilspy)
   - [raw.py](#62-rawpy)
   - [mrmr\_fs.py](#63-mrmr_fspy)
   - [boruta\_fs.py](#64-boruta_fspy)
7. [src/benchmark/trainer.py](#7-srcbenchmarktrainerpy)
8. [src/helper/metrics.py](#8-srchelpermetricspy)
9. [src/models/](#9-srcmodels)
   - [factory.py](#91-factorypy)
   - [nb.py](#92-nbpy)
   - [knn.py](#93-knnpy)
   - [svm.py](#94-svmpy)
   - [rf.py](#94-rfpy)
   - [dt.py](#95-dtpy)
   - [xgb.py](#96-xgbpy)
   - [ann.py](#97-annpy)
10. [src/dataset\_builder/annotation.py](#10-srcdataset_builderannotationpy)
11. [src/interpretation/shap\_utils.py](#11-srcinterpretationshap_utilspy)
12. [src/interpretation/rules.py](#12-srcinterpretationrulespy)
13. [src/interpretation/go\_enrichment.py](#13-srcinterpretationgo_enrichmentpy)
14. [src/pipeline/core.py](#14-srcpipelinecorepy)
15. [configs/](#15-configs)
16. [Output directory structure](#16-output-directory-structure)
17. [How to run](#17-how-to-run)
18. [Key design decisions](#18-key-design-decisions)

---

## 1. Project layout

```
LV_code/
├── src/                       All Python source code
│   ├── __init__.py
│   ├── pipeline/              Main orchestrator (GeneExpressionPipeline)
│   │   ├── core.py            GeneExpressionPipeline class; composes the mixins below
│   │   ├── benchmark.py       run_benchmark (CV sweep, optional k_values)
│   │   ├── baseline_split.py  run_baseline_split (train/test split sweep, all models)
│   │   ├── holdout.py         run_rule_extraction_holdout (reuses baseline_split artifacts)
│   │   ├── rule_extraction.py run_rule_extraction (full-dataset rules, no split)
│   │   └── interpretation.py  run_interpretation / run_interpretation_batch / select_best_model
│   ├── helper/
│   │   ├── config_loader.py   YAML configuration management
│   │   ├── data_loader.py     Dataset I/O (GEO-D, CuMiDa)
│   │   ├── metrics.py         Evaluation metrics
│   │   ├── scaling.py         Conditional StandardScaler (per-model needs_scaling)
│   │   ├── paths.py           resolve_path(), safe_filename() — shared path helpers
│   │   ├── training.py        pick_best_run(), summarize_repeats() — repeated-training bookkeeping
│   │   ├── wandb_utils.py     wandb_log_per_class() — shared W&B per-class logging
│   │   └── report.py          Console / notebook output
│   ├── feature_selection/
│   │   ├── raw.py
│   │   ├── mrmr_fs.py
│   │   └── boruta_fs.py
│   ├── benchmark/
│   │   └── trainer.py         Cross-validation trainer (model comparison)
│   ├── interpretation/
│   │   ├── shap_utils.py      SHAP explainability
│   │   ├── rules.py           Tree-based rule extraction
│   │   └── go_enrichment.py   GO enrichment
│   ├── dataset_builder/       GEO series-matrix → CuMiDa + probe→gene mapping
│   ├── visualize/             EDA / provenance figures
│   └── models/
│       ├── factory.py
│       ├── nb.py
│       ├── knn.py
│       ├── svm.py
│       ├── rf.py
│       ├── dt.py
│       ├── xgb.py
│       └── ann.py
├── configs/
│   ├── datasets.yaml
│   ├── dataset_origin.yaml
│   ├── models.yaml
│   ├── feature_selection.yaml
│   ├── holdout.yaml
│   └── interpretation.yaml
├── Dataset/
│   ├── GEO/
│   └── Cumida/
├── notebooks/
│   ├── 00.a_Format_GEO_Datasets.ipynb
│   ├── 00.b_Format_Cumida_Datasets.ipynb
│   ├── 00.c_Explore_Dataset.ipynb
│   ├── 01_Datasets_Visualization.ipynb
│   ├── 02_Benchmark.ipynb
│   ├── 03_Rules_Extraction.ipynb
│   └── 04_Pipeline.ipynb
├── outputs/                        Generated at runtime — flat, no k
│                                   (run_rule_extraction, run_interpretation,
│                                    non-swept run_benchmark calls)
├── outputs_baseline_full/k{N}/     run_benchmark's CV / full-data k sweep
├── outputs_baseline_split/k{N}/    run_baseline_split's train/test-split k sweep
├── outputs_holdout/k{N}/           run_rule_extraction_holdout
└── outputs_live/<job_id>/          live UI training jobs
```

`{N}` is the rare-class threshold `k` (minimum samples per class kept). See
[§16 Output directory structure](#16-output-directory-structure).

---

## 2. Data flow overview

```
Benchmark — Feature Selection & Model Comparison
─────────────────────────────────────────────────────────────────────
configs/*.yaml
    └─► ConfigLoader.get_dataset_config()
            └─► data_loader.load_dataset()
                    └─► X (n_samples × n_features),  y (int labels)
                            │
                    ┌───────┼───────────┐
                    ▼       ▼           ▼
             raw.py  mrmr_fs.py       boruta_fs.py
             (pass-  (mrmr-selection) (BorutaPy)
             through)
                    │       │           │
               X_sel  X_sel        X_sel (confirmed)
               + CSV  + CSV              + CSV
                    └───────┴───────────┘
                            │
                    ┌───────┼───────────┐
                    ▼       ▼           ▼
                nb/knn    svm    rf/dt/xgb/ann
                            │
                   BenchmarkTrainer.run_cv()
                       StratifiedKFold
                       StandardScaler — CONDITIONAL, fit on train fold only
                         scaled:     nb, knn, svm, ann
                         not scaled: rf, dt, xgboost
                         (src/helper/scaling.py DEFAULT_NEEDS_SCALING,
                          overridable via configs/models.yaml -> scaling:)
                       compute_metrics()
                            │
                   cv_results.csv, cv_summary.csv, per_class_metrics.json,
                   params_des.json, models/*.joblib

Explain  (best combo auto after benchmark; any combo via run_shap)
─────────────────────────────────────────────────────────────────────
Load model_fold_N.joblib (model + scaler) — PROBE level
    ├─► shap_utils.py   (SHAP on probes) ─► post-SHAP probe→gene collapse ─► gene-level plots/ + text/
    └─► go_enrichment.py (g:Profiler on gene symbols)   [run_interpretation reuses the SHAP log]

Find rule  (run_rule_extraction — separate step, config: interpretation.rules)
─────────────────────────────────────────────────────────────────────
Best FS feature set → dedicated shallow RF / DecisionTree → walk root→leaf paths
    └─► rules.py (extract → evaluate → simplify → cross-check) ─► outputs/{ds}/rules/{rf,dt,compare}/
```

---

## 3. src/\_\_init\_\_.py

Package marker. Defines:

| Symbol | Value |
|--------|-------|
| `__version__` | `"2.0.0"` |
| `__framework__` | `"Classification Transcriptomic with XAI"` |

---

## 4. src/helper/config\_loader.py

**Purpose:** Load and validate the three YAML config files. Resolve all relative dataset paths to absolute paths based on `project_root = configs/../`.

### Class `ConfigLoader`

**Constructor**

```python
ConfigLoader(config_root="configs")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `config_root` | `str \| Path` | Directory containing the YAML files. Can be relative. Resolved to absolute internally. |

`self.project_root` is set to `config_root.parent` — this is used to resolve dataset paths in `datasets.yaml`.

---

**Methods**

| Method | Returns | Description |
|--------|---------|-------------|
| `load_yaml(filename)` | `dict` | Load a YAML file from `config_root`. |
| `load_datasets_config()` | `dict` | Load `datasets.yaml`. |
| `load_models_config()` | `dict` | Load `models.yaml`. |
| `load_feature_selection_config()` | `dict` | Load `feature_selection.yaml`. |
| `get_enabled_datasets()` | `List[str]` | Names of all datasets with `enabled: true`. |
| `get_enabled_models()` | `List[str]` | Names of all models with `enabled: true`. |
| `get_enabled_fs_methods()` | `List[str]` | Names of all FS methods with `enabled: true`. |
| `get_dataset_config(name)` | `dict` | Config for one dataset, with path resolved to absolute. |
| `get_model_hyperparams(name)` | `dict` | `hyperparams` block for one model. |
| `get_fs_method_params(name)` | `dict` | `params` block for one FS method. |
| `get_cv_config()` | `dict` | `cross_validation` block from models.yaml. |
| `get_wandb_project()` | `str \| None` | W&B project name; `None` if `use_wandb: false`. |
| `validate_configs()` | `bool` | Check all files load, warn about missing paths. |
| `print_summary()` | `None` | Print enabled datasets (✓/✗ file exists), models, FS methods. |

---

## 5. src/helper/data\_loader.py

**Purpose:** Load gene expression CSV files in GEO-D (matrix) or CuMiDa format. Handles format detection, label encoding, missing value imputation, and rare-class removal.

### Supported dataset formats

**GEO-D (matrix)**
```
feature_1, feature_2, ..., feature_n, label
2.31,       5.42,     ..., 1.22,       0
```
- Last column = class label. No sample ID column.

**CuMiDa**
```
sample,  type,   probe_1, probe_2, ...
GSM001,  tumor,  4.32,    2.11,    ...
```
- Column 0 = sample ID (string), column 1 = class label, columns 2+ = expression features.

---

### Functions

#### `detect_dataset_type(df) → str`

Auto-detect format from column names. Returns `"cumida"` or `"matrix"`.

Heuristic:
- Column 0 name in `{sample, samples, sampleid, gsm, id}` **and** column 1 in `{type, label, class, phenotype, group, condition}` → `"cumida"`.
- Fallback: column 0 looks like a sample ID and column 1 has few unique values (< 20% of rows) → `"cumida"`.
- Otherwise → `"matrix"`.

---

#### `encode_labels(y_raw) → (y, encoder, label_mapping)`

Encode class labels to contiguous integers starting at 0.

| Input | Description |
|-------|-------------|
| `y_raw` | `pd.Series` of any type (string class names, integers, etc.) |

| Output | Description |
|--------|-------------|
| `y` | `np.ndarray (int64)` — 0-based contiguous labels |
| `encoder` | `LabelEncoder \| None` — `None` if y was already 0-based integers |
| `label_mapping` | `dict` — original label → integer index |

---

#### `handle_missing_values(X, strategy="mean") → np.ndarray`

Impute NaN values in feature matrix.

| `strategy` | Behaviour |
|------------|-----------|
| `"mean"` | Replace NaN with per-column mean |
| `"median"` | Replace NaN with per-column median |
| `"zero"` | Replace NaN with 0 |
| `"drop"` | Drop columns (features) that contain any NaN |

---

#### `remove_rare_classes(X, y, min_samples=2) → (X, y)`

Remove samples whose class has fewer than `min_samples` members. Re-encodes labels to 0-based after removal. Required before `StratifiedKFold`.

---

#### `load_dataset(dataset_path, dataset_type="auto", ...) → dict`

Main entry point.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_path` | `str \| Path` | — | Path to CSV or `.gz` file |
| `dataset_type` | `str` | `"auto"` | `"auto"`, `"matrix"`, or `"cumida"` |
| `missing_strategy` | `str` | `"mean"` | How to handle NaN values |
| `remove_rare` | `bool` | `True` | Remove classes with too few samples |
| `min_samples_per_class` | `int` | `2` | Minimum samples per class |
| `**read_csv_kwargs` | — | — | Forwarded to `pd.read_csv` |

**Returns** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `X` | `np.ndarray (float32)` | Feature matrix `(n_samples, n_features)` |
| `y` | `np.ndarray (int64)` | Integer labels `(n_samples,)` |
| `feature_names` | `List[str]` | Column names for features |
| `sample_ids` | `List[str]` | Sample identifiers |
| `metadata` | `dict` | n_samples, n_features, n_classes, label_mapping, dataset_type |
| `label_encoder` | `LabelEncoder \| None` | For inverse transforms |
| `dataframe` | `pd.DataFrame` | Original raw dataframe |

---

### Also in src/helper/: paths.py, training.py, wandb\_utils.py

Small shared helpers extracted out of duplicated code in `baseline_split.py` /
`holdout.py` (and, for `paths.py`, `api/inference.py`) — not full modules in their
own right, so no dedicated numbered section.

| File | Function | Description |
|------|----------|-------------|
| `paths.py` | `resolve_path(project_root, value)` | Resolve `value` against `project_root` unless it's already absolute. |
| `paths.py` | `safe_filename(sample_id)` | Replace path separators (`/`, `\`) in a sample id so it's safe as a filename. |
| `training.py` | `pick_best_run(results_df, select_col, fallback_col)` | Returns `(best_pos, best_seed)` — row with the highest `select_col`, falling back to `fallback_col` if `select_col` is missing/all-null. Requires a `seed` column. |
| `training.py` | `summarize_repeats(results_df, metric_names, prefix="")` | Mean/std per metric across repeated-training runs; returns a `DataFrame` with columns `metric, mean, std`. |
| `wandb_utils.py` | `wandb_log_per_class(run, per_class, class_labels)` | Logs per-class precision/recall/f1 (test split) as both a `wandb.Table` and flat `test_per_class_{metric}/{class}` scalars. |

---

## 6. src/feature\_selection/

All three methods share the same output contract so the pipeline can call them uniformly.
`dispatch.py`'s `run_feature_selection(fs_method, X, y, feature_names, sample_ids,
dataset_name, output_root, params, allow_raw=True, source_path=None)` is the single place
that maps an `fs_method` string to `run_raw_selection` / `run_mrmr_selection` /
`run_boruta_selection` — both `src/pipeline/benchmark.py`'s `_run_feature_selection()`
(§14) and `src/pipeline/baseline_split.py` call into it rather than dispatching themselves:

```python
result = run_*_selection(X, y, feature_names, dataset_name, output_root, ...)
X_sel   = result["X_selected"]       # np.ndarray (n_samples, k_features)
y_sel   = result["y_selected"]       # np.ndarray (n_samples,)
feats   = result["selected_features"] # List[str], length k
indices = result["selected_indices"]  # np.ndarray (k,) int64
```

---

### 6.1 utils.py

Shared helpers used by all three FS modules.

| Function | Description |
|----------|-------------|
| `create_fs_dirs(output_root, dataset_name, fs_method)` | Creates and returns the 4 standard sub-directories: `selected_features/`, `processed_datasets/`, `params/`, `logs/`. |
| `save_ranking(ranking_df, save_dir, filename)` | Save ranking DataFrame as CSV. |
| `save_processed_dataset(X, y, feature_names, save_path, sample_ids)` | Save reduced feature matrix as CSV with optional sample ID column. |
| `save_json(obj, path)` | Save dict as JSON. |
| `save_metadata(metadata, save_dir)` | Save as `metadata.json`. |
| `save_params(params, save_dir)` | Save as `params_des.json`. |
| `build_fs_summary(...)` | Build a summary dict with dataset info + runtime. Includes `"framework": "Classification Transcriptomic with XAI"`. |
| `print_fs_summary(summary)` | Pretty-print the summary. |

**Output directory layout created by `create_fs_dirs`:**
```
outputs/feature_selection/{dataset}/{fs_method}/
├── selected_features/     rankings, JSON
├── processed_datasets/    reduced CSV
├── params/                metadata.json, params_des.json
└── logs/                  (reserved)
```

---

### 6.2 raw.py — `run_raw_selection`

**Purpose:** Identity transform. Passes all features through unchanged. Used as a baseline.

```python
run_raw_selection(
    X, y, feature_names,
    dataset_name, output_root,
    sample_ids=None,
)
```

**Exports:**
- None — zero-copy baseline. Records a `source_path` pointer to the original on-disk dataset in `params/` (`"materialized": false`); interpretation re-loads from there.

**Returns:** Same dict contract as other FS methods with `X_selected = X` and `selected_features = feature_names`.

---

### 6.3 mrmr\_fs.py — `run_mrmr_selection`

**Purpose:** Select the top-K features using Minimum Redundancy Maximum Relevance (mRMR) via `mrmr-selection` (`mrmr_classif`).

**Algorithm:**
1. Call `mrmr_classif(X=DataFrame, y=Series, K=K)` → ordered list of K feature names.
   Relevance = F-statistic (ANOVA), redundancy = Pearson correlation. No discretisation needed.
2. Compute MI scores via `mutual_info_classif` **after** selection (for `ranking_explained.csv`
   only — does not affect which features are selected).

```python
run_mrmr_selection(
    X, y, feature_names,
    dataset_name, output_root,
    sample_ids=None,
    criterion="MID",     # legacy — informational only (F-statistic is used for both)
    K=50,                # number of features to select
    n_bins=3,            # legacy no-op (no discretisation)
    random_state=42,     # used only for the post-selection MI ranking
    params=None,         # extra dict merged into params_des.json
    verbose=True,
)
```

| Parameter | Description |
|-----------|-------------|
| `criterion` | Legacy/informational — `mrmr-selection` uses the F-statistic for both MID and MIQ. It only selects which scoring column appears in the explained ranking. |
| `K` | Number of features to select. Capped at `n_features` with a warning if K > n_features. |
| `n_bins` | Legacy no-op — `mrmr-selection` requires no discretisation. |

**Registered variants** (`configs/feature_selection.yaml`) — the FS-method *name* names
the output directory, so two K values coexist side by side:

| fs\_method name | criterion | K | Status |
|-----------------|-----------|---|--------|
| `mrmr_k50` | MIQ | 50 | enabled |
| `mrmr_k75` | MIQ | 75 | enabled |
| `mrmr_mid` | MID | 50 | legacy entry, **disabled** |

Any fs\_method whose name starts with `mrmr` dispatches to `run_mrmr_selection()`
(see `dispatch.py` above, and §14 `_run_feature_selection`), so adding another
fixed-K variant is config-only.

**Internal functions:**

| Function | Description |
|----------|-------------|
| `compute_mrmr_ranking(X, y, selected_features, feature_names, variant="MID", random_state)` | True incremental mRMR score per selected feature in subset context. relevance = ANOVA F-statistic; redundancy = mean \|Pearson corr\| with earlier-selected features; **MID** = rel − red, **MIQ** = rel / red. Returns `Rank, Feature, Relevance_F, Redundancy, mRMR_{MID\|MIQ}_Score` in selection order. |

**Exports:**
```
outputs/feature_selection/{dataset}/mrmr_k50/          (likewise mrmr_k75/)
├── selected_features/
│   ├── ranking.csv            (mRMR selection order)
│   ├── mrmr_k50_feature_rankings.csv  (Rank, Feature, Relevance_F, Redundancy, mRMR_MIQ_Score)
│   └── selected_features.json
└── processed_datasets/
    └── {dataset}_mrmr_k50_K50.csv
```

**Returns dict:**

| Key | Description |
|-----|-------------|
| `X_selected` | `(n_samples, K)` reduced matrix |
| `y_selected` | Labels (copy of input y) |
| `selected_features` | `List[str]` in mRMR selection order |
| `selected_indices` | `np.ndarray (K,)` original column indices |
| `ranking_df` | mRMR order ranking DataFrame |
| `ranking_scored_df` | mRMR-{MID/MIQ} scored ranking DataFrame (alias key `ranking_explained_df` retained) |
| `variant` | `"MID"` or `"MIQ"` |
| `summary` | Run summary dict |
| `output_dirs` | Dict of output paths |
| `runtime_seconds` | float |

---

### 6.4 boruta\_fs.py — `run_boruta_selection`

**Purpose:** Select features using the Boruta algorithm (all-relevant feature selection) via `BorutaPy + RandomForestClassifier`.

**Algorithm:**
1. Build a `RandomForestClassifier` with `rf_n_estimators=500` trees.
2. Run `BorutaPy.fit()` — iteratively compares feature importances to max shadow feature importance using Bonferroni-corrected binomial test.
3. Features are classified as **Confirmed** (always more important than shadow), **Tentative** (not yet decided), or **Rejected**.
4. Compute MI scores (`mutual_info_classif`) for ranking purposes.

```python
run_boruta_selection(
    X, y, feature_names,
    dataset_name, output_root,
    sample_ids=None,
    n_estimators="auto",   # Boruta shadow iterations ("auto" = sqrt)
    max_iter=100,          # max Boruta rounds
    perc=100,              # percentile of shadow max (100 = strict)
    alpha=0.05,            # FWER threshold
    rf_n_estimators=500,   # trees in internal RF
    max_depth=None,        # RF tree depth (None = unlimited)
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    params=None,
    verbose=True,
    selection_mode="confirmed",  # confirmed | confirmed_tentative | top_k | auto
    k=None,                      # required when selection_mode="top_k"
    dynamic_threshold=50,        # used by selection_mode="auto"
)
```

The pipeline config sets `selection_mode: "auto"` (the function default is still
`"confirmed"` for standalone callers).

| Parameter | Description |
|-----------|-------------|
| `n_estimators` | Boruta parameter: how many trees per shadow-feature iteration. `"auto"` = `sqrt(n_features)`. |
| `rf_n_estimators` | Number of trees in the **internal RandomForest estimator** used for feature importance scoring. Default 500. |
| `max_iter` | Maximum Boruta iterations before declaring remaining undecided features tentative. |
| `perc` | Percentile of shadow feature importances used as the acceptance threshold. 100 = strict (max shadow). |
| `alpha` | FWER (family-wise error rate) threshold for the binomial test. |
| `max_depth` | RandomForest tree depth. `None` = unlimited (original Boruta recommendation). |
| `selection_mode` | Which decision groups become `selected_features`. See table below. |
| `k` | Number of features to keep — required only when `selection_mode="top_k"`. |
| `dynamic_threshold` | Confirmed-feature count at/above which `"auto"` resolves to `"confirmed"`. Default `50`. |

**`selection_mode` values:**

| Mode | Selected feature set |
|------|----------------------|
| `"confirmed"` | Confirmed features only. |
| `"confirmed_tentative"` | Confirmed + tentative — useful when Boruta confirms very few features. |
| `"top_k"` | Top-K by Boruta rank / MI score, regardless of decision group. |
| `"auto"` | Resolves at runtime: `"confirmed"` if `len(confirmed) >= dynamic_threshold`, otherwise `"confirmed_tentative"`. Avoids near-empty feature sets on hard datasets without hand-tuning per dataset. |

The mode actually used is recorded as `resolved_selection_mode` in the returned dict and in
`params/metadata.json` / `params/params_des.json`, so downstream consumers can always tell
which decision group `"auto"` landed on.

**Internal functions:**

| Function | Description |
|----------|-------------|
| `build_boruta_estimator(n_estimators, max_depth, class_weight, random_state, n_jobs)` | Create the internal `RandomForestClassifier`. |
| `run_boruta(X, y, ...)` | Fit `BorutaPy` and return the fitted object. |
| `build_support_dataframe(feature_names, boruta)` | Raw BorutaPy outputs: `Feature, Support, Support_Weak, Boruta_Rank`. |
| `extract_feature_groups(support_df)` | Split into `confirmed`, `tentative`, `rejected` lists. |
| `build_ranking_dataframe(support_df, X, y, feature_names, random_state)` | Produce `ranking.csv` with `Rank, Feature, Decision, Boruta_Rank, MI_Score`. |
| `features_to_indices(selected, all_features)` | Map feature names → column indices. |

**Exports:**
```
outputs/feature_selection/{dataset}/boruta/
├── selected_features/
│   ├── ranking.csv                (all features: Confirmed/Tentative/Rejected)
│   ├── confirmed.csv
│   ├── tentative.csv
│   ├── rejected.csv
│   ├── support.csv                (raw BorutaPy output)
│   └── selected_features.json    {confirmed: [...], tentative: [...]}
└── processed_datasets/
    ├── {dataset}_boruta_confirmed.csv
    └── {dataset}_boruta_confirmed_tentative.csv
```

**Returns dict:**

| Key | Description |
|-----|-------------|
| `X_selected` | Confirmed features only `(n_samples, n_confirmed)` |
| `selected_features` | Confirmed feature names |
| `X_confirmed_tentative` | Confirmed + tentative |
| `confirmed_tentative_features` | List of confirmed + tentative names |
| `ranking_df` | Full ranking DataFrame |
| `support_df` | Raw BorutaPy support DataFrame |

---

## 7. src/benchmark/trainer.py

**Purpose:** Stratified K-Fold cross-validation for all model types. Controls scaler fitting (train-fold only), model instantiation, metric computation, artifact saving, and optional W&B logging.

### Module-level helpers

#### `_try_import_wandb() → bool`

Lazy W&B import — only attempts import once per process. Returns `True` if `wandb` is available, `False` otherwise. Never raises.

#### `_compute_n_splits(y, max_splits=5) → int`

Compute adaptive fold count.

```
n_splits = min(max_splits, min(class_counts))
```

Raises `ValueError` if any class has < 2 samples.

| Input | Description |
|-------|-------------|
| `y` | Integer label array |
| `max_splits` | Upper bound (usually 5) |

---

### Class `BenchmarkTrainer`

```python
BenchmarkTrainer(
    dataset_name,   # str — used to name output paths
    model_name,     # str — "nb", "knn", "svm", "rf", "dt", "xgboost", "ann"
    fs_method,      # str — "raw", "mrmr_k50", "mrmr_k75", "boruta", ...
    output_root,    # str | Path
    wandb_project=None,  # None disables W&B entirely
    seed=42,
    needs_scaling_map=None,  # dict, typically configs/models.yaml -> scaling;
                             # falls back to DEFAULT_NEEDS_SCALING when omitted
)
```

Output directory: `output_root/{dataset_name}/{fs_method}/{model_name}/`

---

#### `run_cv(X, y, feature_names, model_kwargs=None, max_splits=5) → pd.DataFrame`

Run Stratified K-Fold CV.

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `np.ndarray (n_samples, n_features)` | Feature matrix (NOT yet scaled) |
| `y` | `np.ndarray (n_samples,)` | Integer labels |
| `feature_names` | `List[str]` | Feature names (for W&B logging) |
| `model_kwargs` | `dict` | Hyperparameters passed to `get_model()` |
| `max_splits` | `int` | Upper bound on folds |

**Per-fold steps:**
1. Split into train/val with `StratifiedKFold`.
2. **Conditionally** fit a `StandardScaler` on the **training fold only** → transform both
   train and val, via `fit_scaler()` / `apply_scaler()` from `src/helper/scaling.py`.
   Distance/gradient-based models (`nb`, `knn`, `svm`, `ann`) are scaled; tree and boosting
   models (`rf`, `dt`, `xgboost`) are **not** — `fit_scaler()` returns `None` for them and
   `apply_scaler()` passes the matrix through unchanged. The per-model flags come from
   `DEFAULT_NEEDS_SCALING`, overridable via `configs/models.yaml -> scaling:`.
3. Instantiate model via `get_model()`.
4. For ANN: `model.fit(X_train_s, y_train, X_val=X_val_s, y_val=y_val)`.  
   For others: `model.fit(X_train_s, y_train)`.
5. Compute metrics via `compute_metrics()`.
6. Save model + scaler to `models/{model_name}_fold_{n}.joblib`.
7. (Optional) Log fold metrics, confusion matrix, ROC/PR curves to W&B.

**Returns:** `pd.DataFrame` with `metric, mean, std` rows (CV summary).

**Saved files:**

| File | Content |
|------|---------|
| `cv_results.csv` | One row per fold: `fold` + `METRIC_COLUMNS` (accuracy, balanced_accuracy, f1, f1_macro, precision, recall, roc_auc, log_loss) |
| `cv_summary.csv` | mean ± std across folds for each metric |
| `per_class_metrics.json` | Per-fold **per-class** precision / recall / F1 (the `per_class` key of `compute_metrics()`) |
| `params_des.json` | Experiment config: framework, dataset, model, fs_method, n_splits, hyperparams |
| `models/{model}_fold_{n}.joblib` | `{"model": fitted_model, "scaler": fitted_scaler_or_None}` |

> **Note:** per-class metrics are written to a separate `per_class_metrics.json` and are
> deliberately **not** merged into `cv_results.csv` / `cv_summary.csv` — those stay flat,
> fixed-width tables driven by `METRIC_COLUMNS` (see §8), so their schema does not vary
> with the number of classes in a dataset. The scaler entry in the joblib is `None` for
> models that do not need scaling (`rf`, `dt`, `xgboost`).

---

## 8. src/helper/metrics.py

### `compute_metrics(y_true, y_pred, y_prob=None, average="macro", labels=None, class_labels=None) → dict`

| Parameter | Description |
|-----------|-------------|
| `average` | Averaging strategy for the aggregate multi-class metrics: `macro` / `micro` / `weighted`. |
| `labels` | The full set of encoded class labels (e.g. `range(len(class_labels))`). Pass whenever a class can be entirely absent from `y_true` in a split — otherwise sklearn infers labels from observed values, silently shrinking `confusion_matrix` below the true class count. Also fixes the row set of `per_class`. |
| `class_labels` | Human-readable name per entry of `labels`, same order — used only to key the `per_class` dict. Falls back to `str(label)`. |

| Output key | Type | Notes |
|------------|------|-------|
| `accuracy` | `float` | Overall accuracy |
| `balanced_accuracy` | `float` | Adjusted for class imbalance |
| `f1` | `float` | F1 under the caller's `average` |
| `f1_macro` | `float` | **Always** macro-averaged regardless of `average` — a fixed, cross-run comparable metric |
| `precision` | `float` | Under the caller's `average` |
| `recall` | `float` | Under the caller's `average` |
| `roc_auc` | `float \| None` | OvR, macro. Requires `y_prob`. |
| `log_loss` | `float \| None` | Cross-entropy. Requires `y_prob`. |
| `confusion_matrix` | `List[List[int]]` | Rows = true, cols = predicted |
| `per_class` | `dict` | **Non-aggregated** per-class scores (computed with `average=None`): `{class_name: {"precision", "recall", "f1"}}`. Keyed by `class_labels` when given, else `str(label)`. Built over the **same `labels` set** as `confusion_matrix`, so a class absent from the split still gets a zero row rather than silently disappearing. |

For multi-class ROC-AUC, uses `multi_class="ovr"`. Returns `None` for roc_auc / log_loss if `y_prob` is not provided.

**`per_class` vs the aggregate keys.** `per_class` is purely additive — `f1`, `f1_macro`,
`precision`, `recall` and every other aggregate key are unchanged, and the module constant

```python
METRIC_COLUMNS = ["accuracy", "balanced_accuracy", "f1", "f1_macro",
                  "precision", "recall", "roc_auc", "log_loss"]
```

— the single source of truth for callers building a flat metric-column list — does **not**
include `per_class`. Because its width varies with the number of classes, it is kept out of
flat CSVs; `BenchmarkTrainer` persists it separately to `per_class_metrics.json` (§7), and
the holdout flow writes it to `best_run_per_class.json` (§14).

### `print_metrics(metrics) → None`

Pretty-print a metrics dict to stdout.

---

## 9. src/models/

### 9.1 factory.py — `get_model`

```python
get_model(
    name,            # "nb" | "knn" | "svm" | "rf" | "dt" | "xgboost" | "ann"
    input_dim=0,     # required for ANN
    num_classes=2,   # required for ANN
    random_state=42,
    **kwargs,        # model hyperparameters
) → model_instance
```

Returns an unfitted model instance with sklearn-compatible API (`.fit()`, `.predict()`, `.predict_proba()`).

Module constant `SUPPORTED_MODELS = ("nb", "knn", "svm", "rf", "dt", "xgboost", "ann")`.
An unknown `name` raises `ValueError` listing it.

Dispatch table:

| `name` | Builder called | Class |
|--------|----------------|-------|
| `"nb"` | `build_nb(**kwargs)` | `GaussianNB` |
| `"knn"` | `build_knn(**kwargs)` | `KNeighborsClassifier` |
| `"svm"` | `build_svm(random_state, **kwargs)` | `SVC(probability=True)` |
| `"rf"` | `build_rf(random_state, **kwargs)` | `RandomForestClassifier` |
| `"dt"` | `build_dt(random_state, **kwargs)` | `DecisionTreeClassifier` |
| `"xgboost"` | `build_xgb(random_state, **kwargs)` | `XGBClassifier` |
| `"ann"` | `SklearnANNWrapper(input_dim, num_classes, ...)` | PyTorch Lightning MLP |

---

### 9.2 nb.py — `build_nb`

```python
build_nb(**kwargs) → GaussianNB
```

Thin wrapper. All kwargs forwarded to `GaussianNB`.

---

### 9.3 svm.py — `build_svm`

```python
build_svm(random_state=42, **kwargs) → SVC
```

Always sets `probability=True` (needed for `predict_proba`) and `class_weight="balanced"`.

Common kwargs: `kernel="rbf"`, `C=1.0`, `gamma="scale"`.

---

### 9.4 rf.py — `build_rf`

```python
build_rf(random_state=42, **kwargs) → RandomForestClassifier
```

Always sets `class_weight="balanced"`, `n_jobs=-1`.

Common kwargs: `n_estimators=200`, `max_depth=20`, `min_samples_split=5`.

---

### 9.5 dt.py — `build_dt`

```python
build_dt(random_state=42, **kwargs) → DecisionTreeClassifier
```

Always sets `class_weight="balanced"`.

Structurally the single-tree counterpart of `build_rf` — same balanced-class handling,
no `n_jobs` (a single tree is not parallelised) and no `n_estimators`.

Common kwargs: `max_depth`, `min_samples_split=5`, `min_samples_leaf=2`,
`criterion="gini"`.

Not scaled by the trainer (`DEFAULT_NEEDS_SCALING["dt"] = False`, see §7).

**Two independent decision-tree configs** — the same duality `rf` already has:

| Consumer | Config block | Typical `max_depth` | Purpose |
|----------|--------------|--------------------|---------|
| Benchmark model `"dt"` | `configs/models.yaml -> models.dt` | 20 | Predictive comparison against rf/xgboost |
| Rule-extraction model `"decisiontree"` | `configs/interpretation.yaml -> interpretation.rules.model_params.decisiontree` | 5 | Shallow, human-readable root→leaf paths (§12) |

---

### 9.6 xgb.py — `build_xgb`

```python
build_xgb(random_state=42, **kwargs) → XGBClassifier
```

Always sets `eval_metric="mlogloss"`, `n_jobs=-1`.

Common kwargs: `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`.

---

### 9.7 ann.py — `SklearnANNWrapper`

MLP built with PyTorch Lightning, wrapped to expose the sklearn `.fit()/.predict()/.predict_proba()` interface.

```python
SklearnANNWrapper(
    input_dim,           # int — number of input features (required)
    num_classes=2,
    hidden_dims=[128, 64],
    dropout=0.3,
    learning_rate=0.001,
    batch_size=16,
    max_epochs=150,
    patience=10,         # early stopping patience
    random_state=42,
)
```

**Architecture:** `Linear → ReLU → Dropout → ... → Linear → Softmax`

**Training:** Uses `EarlyStopping` on validation loss with `patience` epochs.

**Special `.fit()` signature:**
```python
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
```
Pass `X_val` / `y_val` to enable early stopping. Without them, trains for `max_epochs`.

**Underlying classes:**
- `LitANN` — the `pl.LightningModule` with `forward()`, `training_step()`, `validation_step()`, `configure_optimizers()`.
- `SklearnANNWrapper` — wraps `LitANN` in a sklearn-compatible interface.

---

## 10. src/dataset_builder/annotation.py

**Purpose:** Map probe IDs (microarray feature names) to gene symbols.

Resolution order:
1. Local annotation file (if `annotation_path` provided).
2. GPL SOFT download from NCBI GEO FTP (if `platform` provided).
3. MyGene.info API fallback (if `mygene` is installed).

---

### Constants

**`PLATFORM_TO_GPL`** — dict mapping common chip names to GPL accessions:

| Common name | GPL ID |
|-------------|--------|
| `HG-U133_Plus_2` | GPL570 |
| `HG-U133A` | GPL96 |
| `Mouse430_2` | GPL1261 |
| `HumanHT-12_V4_0_R1` | GPL10558 |
| ... (see source) | ... |

---

### Functions

#### `resolve_gpl_id(platform) → str | None`

Resolve a platform string to a GPL accession (`"GPL570"`).

Accepts: direct GPL IDs (`"GPL570"`, `"gpl570"`), exact chip names, case-insensitive matches, and substring matches.

---

#### `download_gpl(platform, cache_dir, force_download=False) → Path`

Download GPL SOFT file from NCBI GEO FTP. Cached at `cache_dir/{GPL_ID}_family.soft.gz`. Does not re-download unless `force_download=True`.

URL pattern: `https://ftp.ncbi.nlm.nih.gov/geo/platforms/{prefix}/{GPL_ID}/soft/`

---

#### `parse_gpl_soft(soft_path) → pd.DataFrame`

Parse the data table between `!platform_table_begin` and `!platform_table_end` in a GPL SOFT file. Supports both `.soft` and `.soft.gz`.

---

#### `extract_probe_gene_map(platform_df, probe_col="ID", gene_col="Gene Symbol") → dict`

Build `{probe_id: gene_symbol}` dict from a GPL DataFrame. Handles:
- Multiple gene symbols per probe separated by ` /// ` (keeps first).
- Case variations of column names.

---

#### `deduplicate_probes(X, mapping, strategy="max_mean") → pd.DataFrame`

When multiple probes map to the same gene:

| `strategy` | Behaviour |
|------------|-----------|
| `"max_mean"` | Keep probe with highest mean expression across samples |
| `"mean"` | Average all probes per gene |
| `"first"` | Keep first probe per gene encountered |

---

#### `annotate_features(feature_names, platform=None, annotation_path=None, ...) → dict`

Main entry point. Returns `{probe_id: gene_symbol}` mapping.

| Parameter | Description |
|-----------|-------------|
| `feature_names` | List of probe IDs |
| `platform` | Platform name or GPL ID for auto-download |
| `annotation_path` | Local CSV/TSV annotation file |
| `cache_dir` | Where to cache downloaded GPL files (default `cache/gpl/`) |
| `probe_col` | Column name for probe IDs in annotation file |
| `gene_col` | Column name for gene symbols |
| `force_download` | Re-download even if cached |

---

#### `map_features(feature_names, mapping) → List[str]`

Apply a mapping dict to a feature name list. Unmapped probes retain their original name. Warns about unmapped count.

---

## 11. src/interpretation/shap\_utils.py

**Purpose:** Compute SHAP values and save explanation plots for any model type.

### Model-type detection

`_model_type(model) → "tree" | "linear" | "kernel"`

| Class name | Type |
|------------|------|
| RandomForestClassifier, XGBClassifier, GradientBoostingClassifier | `"tree"` |
| LinearSVC, LogisticRegression, SGDClassifier | `"linear"` |
| Everything else (SVC, GaussianNB, SklearnANNWrapper) | `"kernel"` |

---

### Internal SHAP functions

#### `_shap_tree(model, X_background, X_explain) → np.ndarray`

Uses `shap.TreeExplainer` with `feature_perturbation="interventional"`. Exact SHAP values.

For multi-class: returns `(n_samples, n_features, n_classes)` by stacking the per-class arrays.

#### `_shap_kernel(model, X_background, X_explain, max_background=100) → np.ndarray`

Uses `shap.KernelExplainer`. Subsamples background to at most `max_background` points (using k-means with min(10, n_bg) clusters). Approximate SHAP, slower.

#### `_mean_abs_shap(shap_values) → np.ndarray`

Aggregate SHAP values to one importance score per feature:
- 2D `(n_samples, n_features)`: `mean(|shap|)` across samples.
- 3D `(n_samples, n_features, n_classes)`: `mean(sum_classes(|shap|))`.

---

### Main entry point

#### `explain_model(model, X_train, X_test, feature_names, output_dir, top_k=20, kernel_max_background=100) → (shap_values, top_features)`

| Parameter | Description |
|-----------|-------------|
| `model` | Fitted model |
| `X_train` | Background data (used by TreeExplainer / KernelExplainer) |
| `X_test` | Samples to explain |
| `feature_names` | List of feature names |
| `output_dir` | Directory to save plots |
| `top_k` | Number of top features to return |
| `kernel_max_background` | Max background samples for KernelExplainer |

**Returns:**
- `shap_values`: `(n_samples, n_features)` for binary, `(n_samples, n_features, n_classes)` for multi-class.
- `top_features`: `List[str]` — top-K features by mean absolute SHAP.

**Output layout** (under `interpretation/shap/`):
- `plots/` (PNG, 150 dpi): `shap_bar.png`, `shap_beeswarm.png`, plus per-class `shap_bar_class{i}.png` / `shap_beeswarm_class{i}.png`.
- `text/`: `shap_feature_importance_matrix.csv`, `top_shap_genes_by_class.txt` (+ per-class `.txt`), and `probe_to_gene_metadata.csv` (`[Probe_ID, Gene_Symbol, Status]`, Status ∈ {Kept, Collapsed_Mean, Unmapped}).

When `probe_gene_map` is supplied, SHAP is computed on the probe-level inputs the model was trained on, then collapsed to gene level **post-SHAP** (values summed per gene, feature values averaged) so axes and lists use Gene Symbols.

#### `explain_tree_model(...)` — Backward-compatible alias for `explain_model()`.

---

## 12. src/interpretation/rules.py

Tree-based classification **rule extraction** — walks the fitted RF / DecisionTree
trained by `run_rule_extraction` into IF-THEN rules, scores them, simplifies, and
cross-checks the two models. (Replaces the former FP-Growth association-rule miner.)

### Rule form

```
IF ERBB2>8.12 AND GRB7>7.4 THEN Class=HER2   [conf=0.95, support=0.12, fidelity=0.98]
```

### `extract_rules_from_forest(forest, feature_names, scaler=None, gene_map=None) -> pd.DataFrame`

Enumerate every root->leaf path of every tree as one raw rule. `scaler` converts
scaled node thresholds back to raw expression; `gene_map` (`{probe_id: gene_symbol}`)
adds human-readable condition names. One row per leaf path.

### `evaluate_rules(rules, X, y, y_pred=None, class_labels=None) -> pd.DataFrame`

Add empirical metrics to raw rules: `abs_support`, `support`, `confidence`,
`error`, `lift`, `class_specificity`, `strength_score`, `consequent_label`, and —
when `y_pred` (the source model's predictions) is supplied — `fidelity` (rule<->model
agreement). `X` must be the raw (unscaled) matrix in extraction feature order.

### `simplify_rules(rules, cfg) -> pd.DataFrame`

Merge same-gene conditions, filter, deduplicate and rank. `cfg` is the
`interpretation.yaml -> rules.filter` block plus `discretize` / `_gene_medians`:

| Key | Meaning |
|-----|---------|
| `min_confidence`, `min_fidelity`, `min_support`, `min_abs_support` | metric floors |
| `max_conditions` | cap on distinct genes in an antecedent |
| `merge_same_gene` | collapse multiple probes of one gene into one interval |
| `dedup` | drop duplicate antecedent -> class rules |
| `max_rules_per_class`, `max_rules_total` | output caps |
| `discretize` | `GENE=High/Low` labels vs numeric thresholds |

Returns rules ranked by `strength_score`, with string `antecedents` / `consequents`
and structured `conditions_merged` / `conditions_raw` for machine-readable cards.

### `ruleset_summary(rules, X, y, y_pred=None, class_labels=None) -> dict`

Set-level metrics: `coverage` (fraction of samples firing >=1 rule),
`mean_fidelity` (support-weighted), `per_class` counts.

### `compare_rule_sets(rules_a, rules_b, name_a="rf", name_b="dt") -> dict`

Cross-check two models' rule sets at two levels: **strict** (same gene+direction
literal set AND consequent class - confirmed by both) and **lenient** (each
`(gene, direction) -> class` literal shared, even if full antecedents differ).
Both report a Jaccard similarity. Backs `outputs/{ds}/rules/compare/crosscheck.json`.

### `genes_in_rules(rules) -> pd.DataFrame`

The distinct genes actually used by surviving rules (columns: `gene`, `probes`,
`n_rules`, `classes`) - the gene set to hand to GO enrichment, not the full SHAP top-K.

### `save_rules(rules, output_dir, class_labels=None, model_name=None, summary=None, exports=(...)) -> dict`

Write rule artifacts under `output_dir` (per model: `outputs/{ds}/rules/{rf,dt}/`).
`exports` selects which to write:

| Artifact | Contents |
|----------|----------|
| `rules.json` | structured cards (gene antecedent + exact per-probe `antecedent_raw` + metrics) |
| `rules.csv` | flat one-row-per-rule table, sortable by any metric |
| `rules_with_probes.csv` | same, antecedents spelled out with probe IDs |
| `rules_human.txt` | `IF ... THEN Class=... (conf, support, fidelity)` |
| `rules_summary.json` | set-level metrics (coverage, mean_fidelity, per_class) |
| `genes_in_rules.csv` | distinct genes used by surviving rules -> GO enrichment |

## 13. src/interpretation/go\_enrichment.py

**Purpose:** Gene Ontology enrichment via the g:Profiler API (requires `pip install gprofiler-official` and internet).

### `run_go_enrichment(gene_list, organism="hsapiens", sources=None, ...) → pd.DataFrame`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gene_list` | — | List of gene symbols (SHAP top-K) |
| `organism` | `"hsapiens"` | g:Profiler organism code. Also accepts `"human"`, `"mouse"` as aliases. |
| `sources` | `["GO:BP", "GO:MF", "GO:CC"]` | Databases to query. Add `"KEGG"`, `"REAC"`, `"WP"` as needed. |
| `output_dir` | `None` | Directory to save CSV and params JSON |
| `significance_threshold` | `0.05` | FDR-corrected p-value cutoff |
| `correction_method` | `"fdr_bh"` | Multiple testing correction (Benjamini-Hochberg) |
| `ordered_query` | `False` | If True, treats gene_list as ranked |

**Returns** `pd.DataFrame` with columns: `source, native (GO term ID), name, p_value, significant, term_size, query_size, intersection_size, intersection, gene_ratio`.

Empty DataFrame if no significant terms or if `gprofiler-official` is not installed.

**Saved files:**
- `go_enrichment.csv`
- `go_enrichment_params.json` — query metadata

### `summarise_by_namespace(results) → dict`

Split result DataFrame into per-source dicts (`"GO:BP"`, `"GO:MF"`, `"GO:CC"`, etc.).

### `print_top_terms(results, source=None, n=10) → None`

Print top-N terms, optionally filtered to one source.

**Supported organism codes:**

| Alias | g:Profiler code |
|-------|----------------|
| `"human"` | `"hsapiens"` |
| `"mouse"` | `"mmusculus"` |
| `"rat"` | `"rnorvegicus"` |

---

## 14. src/pipeline/core.py

**Purpose:** End-to-end orchestrator. Ties together config loading, data loading, feature selection, model training, annotation, SHAP, rules, and GO enrichment. `core.py` itself defines only the `GeneExpressionPipeline` class shell (config/output-root/W&B setup, class-label resolution, probe→gene mapping) — the `run_*` methods live in the mixins it composes, in sibling modules.

### Class `GeneExpressionPipeline`

```python
GeneExpressionPipeline(
    config_root="configs",
    output_root="outputs",
    use_wandb=True,
)
```

Internally creates a `ConfigLoader`. W&B login is attempted at construction if `use_wandb=True` and the config enables it (resolves up to three separate W&B projects — `wandb_project`, `wandb_project_baseline_split`, `wandb_project_holdout` — one per pipeline stage, each falling back to the main `wandb_project` if unset).

`GeneExpressionPipeline` composes mixins that live in sibling modules:

| Module | Mixin | Public entry point |
|--------|-------|--------------------|
| `src/pipeline/benchmark.py` | `BenchmarkMixin` | `run_benchmark()`, `run_shap()` |
| `src/pipeline/baseline_split.py` | `BaselineSplitMixin` | `run_baseline_split()` |
| `src/pipeline/holdout.py` | `HoldoutMixin` | `run_rule_extraction_holdout()` |
| `src/pipeline/rule_extraction.py` | `RuleExtractionMixin` | `run_rule_extraction()` |
| `src/pipeline/interpretation.py` | `InterpretationMixin` | `run_interpretation()`, `run_interpretation_batch()`, `select_best_model()` |

Note `output_root` is honoured by `run_benchmark()` and the `outputs/`-based flows, but
**not** by `run_baseline_split()` / `run_rule_extraction_holdout()`, which read their roots
from `configs/holdout.yaml` (see §15).

---

#### `run_benchmark(dataset_names=None, k_values=None) → dict`

Run the cross-validation benchmark for one or more datasets. Implemented by
`src/pipeline/benchmark.py`.

| Parameter | Description |
|-----------|-------------|
| `dataset_names` | `List[str]` or `None` (run all enabled datasets) |
| `k_values` | `List[int]` or `None`. When given, sweeps the rare-class threshold `k`: for each `k`, rare classes with `< k` samples are dropped and the whole FS × model grid is re-run, writing to `self.output_root / f"k{k}" / {dataset}/...`. When `None` (default) the original single-run behaviour applies, writing to `self.output_root / {dataset}/...` — unchanged, and still what e.g. notebook 03's prerequisite run uses. |

**For each dataset (per `k`, if sweeping):**
1. Load dataset via `load_dataset()`.
2. For each enabled FS method → run `_run_feature_selection()`.
3. For each enabled model → run `BenchmarkTrainer.run_cv()`.

**Returns:** Nested dict `{dataset → {fs_method → {model → summary_df}}}` (keyed by
`k` first when `k_values` is given).

> Unlike `run_baseline_split()` below, this flow always bases its output path on the
> `output_root` passed to the `GeneExpressionPipeline` constructor — it is **not**
> config-driven. Point the constructor at `outputs_baseline_full` for the k sweep.

---

#### `run_interpretation(dataset_name, fs_method, model_name, fold, ...) → dict`

Run interpretation for a specific trained model.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_name` | — | |
| `fs_method` | `"mrmr_k50"` | Which FS method's reduced dataset to use |
| `model_name` | `"rf"` | |
| `fold` | `1` | Which CV fold model to load |
| `platform` | `None` | GPL platform (used only when `annotation_source='gpl'`) |
| `annotation_path` | `None` | Local annotation CSV (always highest priority) |
| `annotation_source` | `"mygene"` | Probe→gene mapping: `'mygene'` (default, lightweight) / `'gpl'` (NCBI SOFT, large) / `'none'` (keep probe IDs) |
| `run_go` | `True` | Run GO enrichment |
| `organism` | `"hsapiens"` | g:Profiler organism |

**Steps:**
1. Load dataset.
2. Load `models/{model}_fold_{fold}.joblib` → `{"model", "scaler"}`.
3. Load reduced dataset CSV from `outputs/feature_selection/{dataset}/{fs_method}/processed_datasets/`.
4. Scale with saved scaler.
5. (Optional) Annotate features with probe→gene mapping.
6. Reuse the benchmark SHAP top-K features (SHAP is **not** recomputed here — it is produced during the benchmark; see `run_shap`).
7. Run association rule mining on the SHAP top-K features.
8. Run GO enrichment on top genes.

**Returns** dict: `{top_features, mapped_genes, shap_values, rules, go_results}`.

---

#### `run_shap(dataset_name, fs_method, model_name, fold=1, top_k=None) → list[str] | None`

Run SHAP on **any** already-trained benchmark model, on demand — no re-training.
Loads `models/{model}_fold_{fold}.joblib`, resolves the FS-reduced matrix it was
trained on, and writes to `outputs/{dataset}/{fs_method}/{model_name}/shap/`. Use
it to explain a combo that wasn't the benchmark winner (automatic SHAP only
covers the best combo). Returns the top-K feature/gene names. Defaults `top_k` to
`interpretation.yaml → shap.top_k`.

---

#### `run_rule_extraction(dataset_names=None, metric="balanced_accuracy") → dict`

**Full-dataset rule extraction — no train/test split.** Train the dedicated shallow rule
models (`interpretation.rules.model_params`) on the best FS feature set and walk every
root→leaf path into IF-THEN rules, then simplify and (with ≥2 models) cross-check. When
`shap.on_rule_model` is set, also runs SHAP on each rule model. Must be called after
`run_benchmark`. Writes to the **flat** `outputs/{dataset}/rules/{rf,dt,compare}/` (no `k`
nesting).

This is a **different function and a different flow** from
`run_rule_extraction_holdout()` below, which is split-based, `k`-swept, and reuses
`run_baseline_split()`'s artifacts. Neither supersedes the other: this one answers "what
rules describe the whole dataset", the holdout one answers "how well do those rules
generalise to unseen samples".

---

#### `run_baseline_split(dataset_names=None, k_values=None) → dict`

**The canonical train/test-split baseline across the full model roster.** Implemented by
`src/pipeline/baseline_split.py` as `BaselineSplitMixin`.

| Parameter | Description |
|-----------|-------------|
| `dataset_names` | `List[str]` or `None` (all enabled datasets) |
| `k_values` | `List[int]` or `None`. Default read from `configs/feature_selection.yaml -> rare_class_k_values` (default `[2, 3, 4, 5, 6]`). |

**For each `k` × each dataset:**
1. Drop rare classes (those with `< k` samples).
2. Stratified train/test split, using `test_size` / `split_random_state` from
   `configs/holdout.yaml -> holdout.split_baseline`.
3. Run feature selection on the **train split only** (never on test) for the methods in
   `configs/holdout.yaml -> holdout.batch.fs_methods` — `boruta` (with
   `selection_mode: "auto"`), `mrmr_k50` and `mrmr_k75`. `raw` is deliberately **not**
   included; the shared list is exactly the set of FS methods
   `run_rule_extraction_holdout()` later reads back.
4. Fit **every enabled model** (the full roster — `nb`, `knn`, `svm`, `rf`, `dt`,
   `xgboost`, `ann`), unlike `holdout.py` which only trains `rf` / `decisiontree`.
5. **Repeated training, same pattern as the holdout rule models:** each model is trained
   `n_repeats` times (`configs/holdout.yaml -> holdout.split_baseline.n_repeats`, default
   `10`), varying only the model's `random_state` (`repeat_base_seed + i`) — the
   train/test split itself is fixed. The repeat with the best `select_metric`
   (`holdout.split_baseline.select_metric`, default `f1_macro`) on the test set is kept
   via `src/helper/training.py::pick_best_run()`; the rest are summarized but discarded.
6. Evaluate the winning repeat on **both** train and test sets.

**Output root:** this flow reads its own root from
`configs/holdout.yaml -> holdout.split_baseline.baseline_split_root`
(default `outputs_baseline_split`) and writes to
`outputs_baseline_split/k{k}/{dataset}/...`. It is **config-driven and deliberately
ignores** whatever `output_root` the `GeneExpressionPipeline` instance was constructed
with, because its artifacts are a shared, canonical input consumed by
`run_rule_extraction_holdout()` — they must land at a predictable path regardless of how
the pipeline object was built.

**Key artifacts per `k`/dataset** (consumed downstream):

| Artifact | Contents |
|----------|----------|
| `split_info.json` | Train / test **row indices** of the stratified split |
| `feature_selection/<fs_method>/selected_features/selected_features.json` | Per-FS-method selected feature list |
| `{fs_method}/{model}/models/{model}.joblib` | Winning repeat's `{"model", "scaler"}` |
| `{fs_method}/{model}/models/repeats.csv` | One row per repeat (`seed` + `test_*` metrics) |
| `{fs_method}/{model}/models/repeats_summary.csv` | Mean/std per metric across all `n_repeats` (`src/helper/training.py::summarize_repeats()`) |
| `{fs_method}/{model}/models/confusion_matrix_test.{png,csv}` | Winning repeat's test confusion matrix |
| `{fs_method}/{model}/metrics.json` | `{"train", "test", "n_repeats", "best_seed", "select_metric"}` for the winning repeat |
| `{fs_method}/{model}/per_class_metrics.json` | `{"train": ..., "test": ...}` per-class P/R/F1 (§8 `per_class`) |

**Cross-dataset visualizations** — auto-generated per `k` into
`outputs_baseline_split/k{k}/visualizations/` (each dataset also gets its own
per-dataset `{dataset}/visualizations/` folder):

| File | Produced by (`src/visualize/plots.py`) |
|------|----------------------------------------|
| `all_datasets_class_distribution.png` | `plot_all_class_distributions()` |
| `all_datasets_metric_comparison.png` | `plot_all_datasets_metric_comparison()` |

**Why a separate mixin?** `BaselineSplitMixin` is intentionally **not** part of
`HoldoutMixin`. `HoldoutMixin`'s lower-level per-model method
(`_extract_rules_for_model_holdout`) is also called directly by the live UI training job
in `src/api/jobs.py`; keeping this reporting-only batch sweep out of that class avoids
entangling a user-facing serving path with batch reporting logic.

**W&B:** run / group names fold `k` in — run `{dataset}_k{k}_{fs_method}_{model}_split`,
group `{dataset}_k{k}_{fs_method}_{model}` — plus one per-`k` summary run
(`baseline_split_k{k}_comparison`, `job_type="k_comparison"`) logging a `wandb.Table` of
every `(dataset, fs_method, model)` row. Previously only the CV benchmark logged to W&B at
all; both baseline flows and rule extraction now do.

---

#### `run_rule_extraction_holdout(dataset_names=None, k_values=None) → dict`

**Split-based rule extraction that *reuses* `run_baseline_split()`'s artifacts.**
Implemented by `src/pipeline/holdout.py` as `HoldoutMixin`.

| Parameter | Description |
|-----------|-------------|
| `dataset_names` | `List[str]` or `None` (all enabled datasets) |
| `k_values` | `List[int]` or `None`. Default read from `configs/holdout.yaml -> holdout.batch.rare_class_k_values` (default `[3, 4, 5]`) — a subset of the baseline sweep's `k` values. |

It **no longer derives its own train/test split and no longer re-runs feature selection.**
For each `k` × dataset it reads both directly out of
`outputs_baseline_split/k{k}/{dataset}/`:

| Read from | Gives |
|-----------|-------|
| `split_info.json` | The train / test row indices |
| `feature_selection/<fs_method>/selected_features/selected_features.json` | The selected feature list per FS method |

**Grid** (from `configs/holdout.yaml -> holdout.batch`):
- `fs_methods = ["boruta", "mrmr_k50", "mrmr_k75"]`
- `models = ["rf", "decisiontree"]`

Repeated-training params (`n_repeats`, `repeat_base_seed`, `select_metric`) also come from
that block. Each repeat varies only the *model* seed (`repeat_base_seed + i`) — the split is
fixed, inherited from the baseline — and the repeat with the best `select_metric` on the
test set is kept. Writes to
`outputs_holdout/k{k}/{dataset}/{fs_method}/{rf,dt}/` (root from `holdout.output_root`;
`"decisiontree"` maps to the `dt/` folder), plus `{fs_method}/compare/crosscheck.json`
when both models produced rules.

**Per-model method — `_extract_rules_for_model_holdout()`.** Unchanged in signature and
behaviour except that it now **also** computes and returns train-set metrics
(`best_run_train_metrics`) alongside the pre-existing test-set metrics
(`best_run_test_metrics`) for the winning repeated-training run. New files written into the
model's `models/` directory alongside the existing `best_run.json` (which holds the winning
run's row, i.e. its test metrics):

| File | Contents |
|------|----------|
| `best_run_train.json` | Train-set metrics of the winning run (aggregate keys only) |
| `best_run_per_class.json` | `{"train": ..., "test": ...}` per-class precision / recall / F1 (§8 `per_class`) |

This per-model method is **also still called directly by the live UI training job**
(`src/api/jobs.py`) and is unaffected by the rewrite of the batch loop around it.

**W&B:** same `k`-aware scheme as `run_baseline_split()` — run
`{dataset}_k{k}_{fs_method}_{model}_holdout`, group `{dataset}_k{k}_{fs_method}_{model}`,
plus a per-`k` `holdout_k{k}_comparison` run (`job_type="k_comparison"`) with a
`wandb.Table` of every `(dataset, fs_method, model)` row.

---

#### `select_best_model(dataset_name, metric="balanced_accuracy") → dict | None`

Scan `outputs/{dataset}/{fs}/{model}/cv_summary.csv` for every trained combination and return the
one with the highest mean `metric`, as `{fs_method, model_name, fold, score}` (best fold read from
`cv_results.csv`, default 1). Returns `None` if the dataset has no benchmark results yet.

---

#### `run_interpretation_batch(dataset_names=None, fs_method=None, model_name=None, fold=None, select="best", ...) → dict`

Run interpretation for several datasets — the **multiple-dataset** interpretation mode.

| Parameter | Description |
|-----------|-------------|
| `dataset_names` | Datasets to interpret (default: all enabled). |
| `select` | `"best"` auto-selects the top `(fs, model, fold)` per dataset via `select_best_model`; `"fixed"` uses the explicit `fs_method` / `model_name` / `fold` for every dataset. |
| `platform`, `annotation_path`, `run_go`, `organism` | Forwarded to `run_interpretation`. |

**Returns** `{dataset_name: interpretation_result}` and prints a per-dataset summary table.

---

#### `print_summary() → None`

Delegates to `ConfigLoader.print_summary()`.

---

#### `_run_feature_selection(X, y, feature_names, sample_ids, dataset_name, fs_method) → dict`

`BenchmarkMixin` method (`src/pipeline/benchmark.py`). No longer dispatches itself —
it's a thin wrapper that fetches `params = ConfigLoader.get_fs_method_params(fs_method)`
and `source_path` (dataset config path, only for `fs_method == "raw"`), then delegates to
`src/feature_selection/dispatch.py`'s `run_feature_selection()` (§6), which does the actual
`raw` / `mrmr*` / `boruta` dispatch. `src/pipeline/baseline_split.py` calls the same
`run_feature_selection()` directly (no per-mixin wrapper there).

---

### CLI entry point

There is no command-line entry point for the pipeline package — `main.py` at the
repo root is an unused stub, and `src/pipeline/` has no `__main__` block or argparse
CLI. Everything is driven from notebooks (see `notebooks/`) or the FastAPI backend
(`src/api/`), e.g.:

```python
from src.pipeline import GeneExpressionPipeline

pipe = GeneExpressionPipeline()
pipe.run_benchmark(['GEO-Breast-20711'])
pipe.run_interpretation('GEO-Breast-20711', 'mrmr_k50', 'rf', 1, platform='HG-U133_Plus_2', run_go=False)
```

---

## 15. configs/

### configs/datasets.yaml

Controls which datasets are active and where they live.

```yaml
datasets:
  DatasetName:
    enabled: true | false
    path: "Dataset/GEO/..."       # relative to project root
    type: "auto" | "matrix" | "cumida"
    description: "..."
    n_classes: 2
    class_labels: ["Normal", "Disease"]
```

> The former top-level `settings:` block (`random_seed`, `missing_strategy`,
> `remove_rare_classes`, `min_samples_per_class`) has been **removed** — no code ever read
> it. Those knobs live where they are actually consumed: `load_dataset()` arguments (§5)
> and `configs/holdout.yaml` / `configs/feature_selection.yaml` for the rare-class `k`.

**Supported types:**

| `type` | Dataset format |
|--------|---------------|
| `"matrix"` | GEO-D: last column = label |
| `"cumida"` | CuMiDa: col 0 = sample ID, col 1 = label |
| `"auto"` | Auto-detect |

---

### configs/models.yaml

Controls which models run and their hyperparameters.

```yaml
models:
  nb:     { enabled: true, hyperparams: { var_smoothing: 1.0e-9 } }
  knn:    { enabled: true, hyperparams: { n_neighbors: 5, weights: distance, ... } }
  svm:    { enabled: true, hyperparams: { kernel: rbf, C: 1.0, gamma: scale } }
  rf:     { enabled: true, hyperparams: { n_estimators: 200, max_depth: 20, ... } }
  dt:     { enabled: true, hyperparams: { max_depth: 20, min_samples_split: 5, ... } }
  xgboost: { ... }
  ann:    { ... hidden_dims, dropout, learning_rate, batch_size, max_epochs, patience }

# Which models get a StandardScaler fitted on the training fold (see src/helper/scaling.py).
# Source of truth; DEFAULT_NEEDS_SCALING is the fallback for callers without a
# ConfigLoader. Models listed in neither default to true (safe).
scaling:
  nb: true
  knn: true
  svm: true
  ann: true
  rf: false
  dt: false
  xgboost: false

cross_validation:
  n_splits: 5          # upper bound; actual = min(5, min_class_count)
  stratified: true
  shuffle: true
  random_state: 42

training:
  use_wandb: false
  wandb_project: "transcriptomic_xai"
  verbose: true
```

---

### configs/feature_selection.yaml

Controls which feature-selection methods run in the benchmark.

```yaml
# Rare-class thresholds swept by run_baseline_split (superset of holdout.batch's [3,4,5]).
rare_class_k_values: [2, 3, 4, 5, 6]

feature_selection:
  benchmark:
    methods:
      raw:        { enabled: true, params: {} }
      mrmr_k50:   { enabled: true, params: { criterion: MIQ, K: 50, n_bins: 3 } }
      mrmr_k75:   { enabled: true, params: { criterion: MIQ, K: 75, n_bins: 3 } }
      mrmr_mid:   { enabled: false, params: { criterion: MID, K: 50, n_bins: 3 } }  # legacy
      boruta:     { enabled: true, params: {
                      n_estimators: "auto",
                      rf_n_estimators: 500,
                      max_depth: null,
                      max_iter: 100,
                      perc: 100,
                      alpha: 0.05,
                      class_weight: balanced,
                      random_state: 42,
                      selection_mode: "auto",   # confirmed | confirmed_tentative | top_k | auto
                      dynamic_threshold: 50,    # "auto": confirmed if len(confirmed) >= this
                      k: null                   # required when selection_mode == "top_k"
                   }}
```

The two active mRMR entries differ only in `K` (50 vs 75); both use `criterion: MIQ`.
There is no longer a single configurable-K `mrmr_miq` method — see §6.3.

### configs/holdout.yaml

Controls the two split-based sweeps (`run_baseline_split`, `run_rule_extraction_holdout`)
and the live split-preview UI screen. Restructured into three named sub-blocks plus two
top-level keys — each block is read by exactly one consumer, so changing one cannot
silently perturb the others.

```yaml
holdout:
  enabled: true
  output_root: "outputs_holdout"        # base for run_rule_extraction_holdout
  active_min_samples_per_class: 4       # which k the API/UI currently serves

  # Read by run_baseline_split() to CREATE the canonical split.
  split_baseline:
    baseline_split_root: "outputs_baseline_split"
    test_size: 0.15
    split_random_state: 42
    # Repeated training of every enabled model — same pattern as holdout.batch
    # below, applied to the full model roster instead of just rf/decisiontree.
    n_repeats: 10
    repeat_base_seed: 0
    select_metric: "f1_macro"

  # Read by run_rule_extraction_holdout() (and the live UI job in src/api/jobs.py).
  batch:
    rare_class_k_values: [3, 4, 5]
    fs_methods: ["boruta", "mrmr_k50", "mrmr_k75"]
    models: ["rf", "decisiontree"]
    n_repeats: 10
    repeat_base_seed: 0
    select_metric: "f1_macro"

  # Read ONLY by src/api/registry.py -> compute_holdout_split()
  # (the on-demand split-preview UI screen). Not used by either batch flow.
  live_ui_defaults:
    min_samples_per_class: 4
    test_size: 0.15
    split_random_state: 42
```

| Key | Consumer | Meaning |
|-----|----------|---------|
| `holdout.output_root` | `run_rule_extraction_holdout` | Base dir; results land in `{output_root}/k{k}/{dataset}/`. |
| `holdout.active_min_samples_per_class` | `src/api/registry.py` | Default `k` served by the API/UI. Resolved by `holdout_root_for(k=None)`, which returns `outputs_holdout/k{k}/` (this function replaced the old flat module-level `HOLDOUT_ROOT` constant; `src/api/main.py`, `src/api/jobs.py`, `src/api/inference.py` and `scripts/generate_bio_descriptions.py` all call it now). |
| `holdout.split_baseline` | `run_baseline_split` | Where to write, and the split parameters used to **create** the canonical train/test split + FS artifacts. |
| `holdout.batch` | `run_rule_extraction_holdout`, live UI job | Which `k` values / FS methods / models to iterate, and the repeated-training parameters (`n_repeats`, `repeat_base_seed`, `select_metric` — the metric used to pick the winning run). |
| `holdout.live_ui_defaults` | `src/api/registry.py -> compute_holdout_split()` only | Defaults for the interactive split-preview screen. |

Note `batch.rare_class_k_values` (`[3,4,5]`) is a **subset** of
`feature_selection.yaml -> rare_class_k_values` (`[2,3,4,5,6]`) — the holdout flow can only
run for `k` values the baseline split has already produced artifacts for.

---

### configs/interpretation.yaml

Controls all explainability: SHAP, tree-based rule extraction, GO enrichment.
SHAP is a tool pointed at trained models (not a stage) — `on_best_benchmark`
explains the winning `(fs × model)`, `on_rule_model` explains the rule model,
and `run_shap(...)` explains any combo on demand.

```yaml
interpretation:
  shap:
    enabled: true
    top_k: 20
    select_metric: f1_macro            # picks the best (fs × model) to explain
    on_best_benchmark: true            # SHAP the winning combo after benchmarking
    on_rule_model: true                # SHAP the rule-extraction model
  rules:
    enabled: true
    models: ["rf", "decisiontree"]     # >=2 enables the confirmed-by-both cross-check
    select_metric: balanced_accuracy
    model_params: { rf: {...}, decisiontree: {...} }
    filter: { min_confidence: 0.80, min_support: 0.05, max_conditions: 5, ... }
    discretize: false
    export: [json, csv, csv_probes, human, summary, genes]
  go_enrichment:
    enabled: true
    organism: hsapiens
    sources: [GO:BP, GO:MF, GO:CC]
    significance_threshold: 0.1
    correction_method: "g:SCS"
```

---

## 16. Output directory structure

There are **five** output roots. Three of them are `k`-nested (one sub-tree per rare-class
threshold), two are flat:

| Root | Written by | `k`-nested? |
|------|-----------|-------------|
| `outputs/` | `run_rule_extraction`, `run_interpretation`, and any non-swept `run_benchmark` call | no |
| `outputs_baseline_full/k{N}/` | `run_benchmark(k_values=...)` — CV / full-data sweep | yes |
| `outputs_baseline_split/k{N}/` | `run_baseline_split()` — train/test split sweep, **all** models | yes |
| `outputs_holdout/k{N}/` | `run_rule_extraction_holdout()` — `boruta`/`mrmr_k50`/`mrmr_k75` × `rf`/`decisiontree` only, reusing the split baseline's artifacts | yes |
| `outputs_live/<job_id>/` | Live UI training jobs (`src/api/jobs.py`) | no |

```
outputs_baseline_full/
└── k{N}/                                   N ∈ feature_selection.yaml rare_class_k_values
    └── {dataset}/                          same inner layout as outputs/{dataset}/ below
        └── {fs_method}/{model}/            cv_results.csv · cv_summary.csv ·
                                            per_class_metrics.json · params_des.json · models/

outputs_baseline_split/
└── k{N}/
    ├── {dataset}/
    │   ├── split_info.json                 ← train/test row indices (CANONICAL split;
    │   │                                      read back by run_rule_extraction_holdout)
    │   ├── feature_selection/
    │   │   ├── boruta/selected_features/selected_features.json
    │   │   ├── mrmr_k50/selected_features/selected_features.json
    │   │   └── mrmr_k75/selected_features/selected_features.json
    │   ├── {fs_method}/{model}/            every enabled model (nb, knn, svm, rf, dt,
    │   │                                    xgboost, ann); winning repeat of n_repeats
    │   │   ├── metrics.json                train + test metrics, best_seed, n_repeats
    │   │   ├── per_class_metrics.json      {"train":..., "test":...} per-class P/R/F1
    │   │   └── models/
    │   │       ├── {model}.joblib          winning repeat's {"model", "scaler"}
    │   │       ├── repeats.csv             one row per repeat (seed + test_* metrics)
    │   │       ├── repeats_summary.csv     mean ± std per metric across repeats
    │   │       └── confusion_matrix_test.{png,csv}
    │   └── visualizations/                  per-dataset figures
    └── visualizations/                     ← combined cross-dataset figures for this k
        ├── all_datasets_class_distribution.png    (plot_all_class_distributions)
        └── all_datasets_metric_comparison.png     (plot_all_datasets_metric_comparison)

outputs_holdout/
└── k{N}/                                   N ∈ holdout.yaml holdout.batch.rare_class_k_values
    └── {dataset}/
        └── {fs_method}/                    boruta | mrmr_k50 | mrmr_k75
            ├── compare/ crosscheck.json    (rules confirmed by BOTH models)
            └── {rf|dt}/                    "decisiontree" → dt/ on disk
                ├── models/
                │   ├── cv_results.csv          one row per repeat
                │   ├── cv_summary.csv          mean ± std across repeats
                │   ├── best_run.json           winning run's row (TEST metrics)
                │   ├── best_run_train.json     winning run's TRAIN metrics
                │   ├── best_run_per_class.json {"train": ..., "test": ...} per-class P/R/F1 (§8)
                │   ├── params_des.json         n_repeats · repeat_base_seed · best_seed · ...
                │   ├── model_best.joblib       {"model", "scaler"}
                │   └── confusion_matrix_test.png
                └── rules/                  rules.json · rules.csv · rules_human.txt · ...

outputs_live/
└── <job_id>/                               live UI training job artifacts
```

The flat `outputs/` tree (below) is still produced verbatim by `run_rule_extraction`,
`run_interpretation` and non-swept `run_benchmark` calls:

```
outputs/
│
├── feature_selection/
│   └── {dataset}/
│       ├── raw/                          (zero-copy: no processed dataset on disk)
│       │   ├── selected_features/
│       │   │   └── selected_features.json
│       │   └── params/
│       │       ├── metadata.json          (source_path pointer, materialized: false)
│       │       └── params_des.json
│       │
│       ├── mrmr_k50/                       (likewise mrmr_k75/)
│       │   ├── selected_features/
│       │   │   ├── ranking.csv            ← mRMR selection order
│       │   │   ├── ranking_explained.csv  ← sorted by MI score
│       │   │   └── selected_features.json
│       │   ├── processed_datasets/
│       │   │   └── {dataset}_mrmr_k50_K50.csv
│       │   └── params/
│       │
│       └── boruta/
│           ├── selected_features/
│           │   ├── ranking.csv
│           │   ├── confirmed.csv
│           │   ├── tentative.csv
│           │   ├── rejected.csv
│           │   ├── support.csv
│           │   └── selected_features.json
│           └── processed_datasets/
│               ├── {dataset}_boruta_confirmed.csv
│               └── {dataset}_boruta_confirmed_tentative.csv
│
├── {dataset}/
│   ├── {fs_method}/
│   │   ├── {model}/
│   │   │   ├── cv_results.csv     ← per-fold metrics        (benchmark)
│   │   │   ├── cv_summary.csv     ← mean ± std              (benchmark)
│   │   │   ├── per_class_metrics.json ← per-class P/R/F1    (benchmark)
│   │   │   ├── params_des.json    ← experiment config       (benchmark)
│   │   │   ├── models/
│   │   │   │   ├── {model}_fold_1.joblib
│   │   │   │   ├── {model}_fold_2.joblib
│   │   │   │   └── ...
│   │   │   └── shap/              ← explain (best combo auto; any via run_shap)
│   │   │       ├── plots/                  (gene-symbol axes)
│   │   │       │   ├── shap_bar.png
│   │   │       │   ├── shap_beeswarm.png
│   │   │       │   └── shap_bar_class{i}.png  (multi-class)
│   │   │       └── text/
│   │   │           ├── shap_feature_importance_matrix.csv
│   │   │           ├── top_shap_genes_by_class.txt
│   │   │           └── probe_to_gene_metadata.csv   [Probe_ID, Gene_Symbol, Status]
│   │   └── interpretation/        ← explain (run_interpretation)
│   │       └── go_enrichment/
│   │           ├── go_enrichment.csv
│   │           └── go_enrichment_params.json
│   │
│   └── rules/                     ← find rule (run_rule_extraction; dataset-level)
│       ├── rf/    rules.json · rules.csv · rules_with_probes.csv · rules_human.txt ·
│       │          rules_summary.json · genes_in_rules.csv  · shap/ (if on_rule_model)
│       ├── dt/    (same artifacts as rf/)
│       └── compare/ crosscheck.json   (rules / literals confirmed by BOTH models)
│
└── (optional)
    └── pipeline_report.json
```

---

## 17. How to run

### From a notebook

```python
# At the top of any notebook in notebooks/
import sys
sys.path.insert(0, '../src')

from src.pipeline import GeneExpressionPipeline

pipe = GeneExpressionPipeline(
    config_root='../configs',
    output_root='../outputs',
    use_wandb=False,
)
pipe.print_summary()

# Benchmark
results = pipe.run_benchmark(['GEO-Breast-20711'])

# Interpretation
interp = pipe.run_interpretation(
    dataset_name='GEO-Breast-20711',
    fs_method='mrmr_k50',
    model_name='rf',
    fold=1,
    platform='HG-U133_Plus_2',   # optional
)
```

### From CLI (run from `LV_code/` directory)

```bash
# Benchmark — all enabled datasets
python -m src.pipeline.pipeline benchmark

# Benchmark — specific dataset
python -m src.pipeline.pipeline benchmark GEO-Breast-20711

# Benchmark — without W&B
python -m src.pipeline.pipeline benchmark GEO-Breast-20711 --no-wandb

# Interpretation — positional args: dataset fs_method model fold
python -m src.pipeline.pipeline interpret GEO-Breast-20711 mrmr_k50 rf 1

# Interpretation — with annotation, skip GO
python -m src.pipeline.pipeline interpret GEO-Breast-20711 boruta xgboost 2 --platform GPL570 --no-go

# Interpretation (batch) — interpret several datasets, best model per dataset
python -m src.pipeline.pipeline interpret-all                       # all enabled datasets
python -m src.pipeline.pipeline interpret-all GEO-Breast-20711 GEO-Mesothelioma-29354
```

### Run individual modules

```python
# Feature selection only
from feature_selection.mrmr_fs import run_mrmr_selection
result = run_mrmr_selection(X, y, feature_names, 'MyDataset', 'outputs/', K=50)

# Train one model
from src.benchmark.trainer import BenchmarkTrainer
trainer = BenchmarkTrainer('MyDataset', 'rf', 'mrmr_k50', 'outputs/')
summary = trainer.run_cv(X_sel, y, feature_names)

# SHAP on any model
from src.interpretation.shap_utils import explain_model
shap_values, top_features = explain_model(model, X_train_s, X_test_s, feature_names, 'outputs/shap/')

# GO enrichment
from src.interpretation.go_enrichment import run_go_enrichment
go_df = run_go_enrichment(top_genes, organism='hsapiens', output_dir='outputs/go/')
```

### Add a new dataset

1. Copy the file to `Dataset/GEO/` or `Dataset/Cumida/`.
2. Add an entry in `configs/datasets.yaml` with `enabled: true`.
3. Run `pipe.run_benchmark(['NewDataset'])`.

---

## 18. Key design decisions

| Decision | Rationale |
|----------|-----------|
| Feature selection on full dataset | Produces a single reproducible reduced CSV per method. Leakage prevention is the scaler's job (fitted on train fold only). |
| Adaptive n_splits | `min(5, min_class_count)` — prevents `StratifiedKFold` errors on imbalanced small datasets. |
| mrmr-selection (mrmr_classif) | F-statistic relevance with Pearson-correlation redundancy. No discretisation required. |
| Boruta rf_n_estimators=500 | Separate from BorutaPy's `n_estimators` (shadow iterations). The internal RF uses 500 trees for stable importance estimates. |
| SHAP TreeExplainer for RF/XGBoost | Exact, fast. KernelExplainer for everything else — approximate but model-agnostic. |
| W&B lazy import | `import wandb` is deferred until first use. The whole pipeline runs without W&B installed. |
| Conditional `StandardScaler` inside CV fold | Scaling is per-model, driven by `DEFAULT_NEEDS_SCALING` in `src/helper/scaling.py` (overridable via `configs/models.yaml -> scaling:`): `nb`/`knn`/`svm`/`ann` are scaled, `rf`/`dt`/`xgboost` are not — tree and boosting splits are invariant to monotone rescaling, so scaling them adds no accuracy but does add a fitted transform that must be carried around and re-applied at interpretation time (and it distorts rule thresholds, which must be read back in raw expression units). When a scaler *is* used it is fitted on the training fold only and stored in the joblib artifact alongside the model, so interpretation uses the exact scaler that produced the CV metrics; for unscaled models the stored scaler is `None`. |
| `dt` (single decision tree) as a first-class benchmark model | Previously a decision tree existed only as a rule-extraction model. Adding `dt` to the general factory lets the fully interpretable single-tree baseline be scored on the same CV/holdout footing as `rf`/`xgboost`, quantifying what accuracy the interpretability actually costs. |
| Two fixed-K mRMR variants (`mrmr_k50`, `mrmr_k75`) instead of one configurable K | K is not a neutral knob — it changes the feature set, hence every downstream model, rule and figure. Registering each K as its own named FS method gives each one its own output subtree, so both live side by side in one run and are directly comparable, instead of one overwriting the other whenever K is edited. Both use `criterion: MIQ`; the old MID entry is kept but disabled. |
| Boruta `selection_mode: "auto"` | On hard/small datasets Boruta sometimes confirms only a handful of features, making a fixed `"confirmed"` mode yield a near-empty feature set; a fixed `"confirmed_tentative"` conversely dilutes the set when Boruta *did* decide cleanly. `"auto"` picks per dataset — `"confirmed"` if `len(confirmed) >= dynamic_threshold` (50), else `"confirmed_tentative"` — and records `resolved_selection_mode` in metadata so the choice stays auditable. |
| Per-class metrics in a separate JSON, not the flat CSVs | `compute_metrics()["per_class"]` is essential for imbalanced multi-class datasets (a good `f1_macro` can still hide one collapsed class), but its width varies with class count. Keeping it out of `METRIC_COLUMNS` and writing it to `per_class_metrics.json` / `best_run_per_class.json` preserves the fixed, uniform schema of `cv_results.csv` / `cv_summary.csv`, which the summary readers and `select_best_model()` concatenate across datasets. |
| Holdout **reuses** `run_baseline_split`'s split + features | `run_rule_extraction_holdout()` reads `split_info.json` and `selected_features.json` out of `outputs_baseline_split/k{k}/{dataset}/` rather than re-deriving its own split and re-running FS. Two payoffs: (1) it avoids duplicating the most expensive step (Boruta/mRMR on tens of thousands of probes) for every `k`; (2) it *guarantees* the two flows are directly comparable — the rule models and the full model roster trained on the byte-identical split and feature set, so any metric gap is attributable to the model, not to a different random split or a differently-converged FS run. |
| `run_baseline_split` kept out of `HoldoutMixin` | `HoldoutMixin`'s per-model method `_extract_rules_for_model_holdout()` is also called directly by the live UI training job (`src/api/jobs.py`). `run_baseline_split` is a reporting-only batch sweep; giving it its own `BaselineSplitMixin` in `src/pipeline/baseline_split.py` means changes to the batch sweep can never destabilise the user-facing serving path. |
| Three separate `k`-nested output roots | `outputs_baseline_full/`, `outputs_baseline_split/` and `outputs_holdout/` are distinct trees rather than variants inside one, because they answer different questions (CV on full data; train/test split across all models; rules on the holdout split). Nesting `k{N}` directly under each root makes a `k` sweep additive — a new `k` never overwrites an existing one — and lets the API serve one chosen `k` via `holdout_root_for(k=None)`. |
| All FS methods independent | Each method produces its own output subtree and its own set of model results. Disabled methods are skipped entirely. |
