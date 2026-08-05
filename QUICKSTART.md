# Quickstart — Classification Transcriptomic with XAI

Get from zero to results in 5 minutes.

---

## 1. Install dependencies

This project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

(Installs scikit-learn, xgboost, `mrmr-selection`, `boruta`, torch + lightning,
shap, `rich`, and the optional wandb / mygene.)

---

## 2. Enable a dataset

Open `configs/datasets.yaml` and set `enabled: true` for the dataset you want to run.  
Make sure the `path` points to an existing file (relative to `LV_code/`).

```yaml
datasets:
  GEO-Breast-20711:
    enabled: true                             # ← flip this
    path: "Dataset/GEO_Cumida/20711/20711/Breast_GEOD20711.csv"
    type: "cumida"
    platform: "A-AFFY-44"                    # from description.txt — used by annotation.py
    n_classes: 5
    class_labels: ["basal", "HER2", "LumA", "LumB", "normal"]
```

To add your own dataset, append a new entry and copy the CuMiDa-format CSV into `Dataset/Cumida/`.

---

## 3. Run from a Jupyter notebook

There is no command-line entry point — `main.py` is an unused stub. Everything
runs from a notebook (or headlessly via `jupyter nbconvert --execute`, see
below).

```python
import sys
sys.path.insert(0, '..')              # run from notebooks/ directory (repo root)

from src.pipeline import GeneExpressionPipeline

pipe = GeneExpressionPipeline(
    config_root='../configs',
    output_root='../outputs_baseline_full',
    use_wandb=False,
)
pipe.print_summary()                  # verify config looks right

# CV/full-data baseline (single run, auto-SHAP on the best combo)
results = pipe.run_benchmark(['GEO-Breast-20711'])

# Optional: explain any other trained combo on demand
pipe.run_shap('GEO-Breast-20711', 'mrmr_k50', 'svm')

# Full-dataset tree-based rules (needs the run_benchmark call above)
pipe.run_rule_extraction(['GEO-Breast-20711'])
```

To sweep the rare-class threshold k across both baseline flows (produces the
inputs `run_rule_extraction_holdout` needs):

```python
K_VALUES = pipe.config_loader.load_feature_selection_config().get('rare_class_k_values', [2, 3, 4, 5, 6])

pipe_full = GeneExpressionPipeline(config_root='../configs', output_root='../outputs_baseline_full', use_wandb=False)
pipe_full.run_benchmark(['GEO-Breast-20711'], k_values=K_VALUES)            # -> outputs_baseline_full/k{N}/

pipe.run_baseline_split(['GEO-Breast-20711'], k_values=K_VALUES)            # -> outputs_baseline_split/k{N}/, all models

# Now that outputs_baseline_split/k{3,4,5}/ exists, rule extraction on a split can run:
pipe.run_rule_extraction_holdout(['GEO-Breast-20711'], k_values=[3, 4, 5])  # -> outputs_holdout/k{N}/
```

Open `notebooks/02_Benchmark.ipynb` (baseline + k-sweep), `03_Rules_Extraction.ipynb`
(full-dataset rules), or `04_Rule_Extraction_Holdout.ipynb` (rules on a train/test
split — run 02's k-sweep first) for complete, runnable examples.

To run any notebook headlessly from a terminal:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/02_Benchmark.ipynb
```

---

## 4. Read your results

```
outputs_baseline_full/k4/GEO-Breast-20711/                      ← run_benchmark(k_values=[4,...])
├── feature_selection/GEO-Breast-20711/
│   ├── mrmr_k50/processed_datasets/GEO-Breast-20711_mrmr_k50_K50.csv   ← reduced dataset
│   ├── mrmr_k50/selected_features/mrmr_k50_feature_rankings.csv         ← genes ranked by mRMR score
│   └── boruta/selected_features/ranking.csv                             ← confirmed/tentative genes
│
└── mrmr_k50/rf/                            ← benchmark
    ├── cv_results.csv          ← accuracy, F1, AUC per fold
    ├── cv_summary.csv          ← mean ± std summary
    ├── per_class_metrics.json  ← per-class precision/recall/f1 per fold
    └── shap/plots/shap_bar.png             ← explain (best combo / run_shap)
        └── shap/text/shap_feature_importance_matrix.csv

outputs_holdout/k4/GEO-Breast-20711/mrmr_k50/                    ← run_rule_extraction_holdout
├── rf/models/{cv_results.csv, best_run.json, best_run_train.json}
├── rf/rules/rules.csv          ← IF-THEN rules (strength_score ↓)
├── dt/rules/rules.csv
└── compare/crosscheck.json     ← rules confirmed by BOTH models

outputs/GEO-Breast-20711/rules/                                   ← run_rule_extraction (no split)
├── rf/rules.csv
├── dt/rules.csv
└── compare/crosscheck.json
```

---

## 5. Tune what runs

| Want to change | Edit |
|----------------|------|
| Which datasets | `configs/datasets.yaml` → `enabled: true/false` |
| Which models | `configs/models.yaml` → `enabled: true/false`, edit `hyperparams:` (models: nb, knn, svm, rf, dt, xgboost, ann) |
| Which models get StandardScaler | `configs/models.yaml` → `scaling:` map (tree models rf/dt/xgboost default to `false`) |
| Which FS methods | `configs/feature_selection.yaml` → `enabled: true/false` |
| mRMR K (# features) | `configs/feature_selection.yaml` → `mrmr_k50.params.K` / `mrmr_k75.params.K` |
| Boruta iterations / selection_mode | `configs/feature_selection.yaml` → `boruta.params.max_iter` / `.selection_mode` (`"auto"`, `"confirmed"`, `"confirmed_tentative"`, `"top_k"`) |
| Rare-class k sweep (baseline) | `configs/feature_selection.yaml` → `rare_class_k_values` |
| Rare-class k sweep (rule extraction) | `configs/holdout.yaml` → `holdout.batch.rare_class_k_values` |
| Which k the API/UI serves | `configs/holdout.yaml` → `holdout.active_min_samples_per_class` (restart the API after changing) |
| CV folds | `configs/models.yaml` → `cross_validation.n_splits` |
| SHAP top-K / where it runs | `configs/interpretation.yaml` → `interpretation.shap.*` |
| Rule thresholds | `configs/interpretation.yaml` → `interpretation.rules.filter.*` |
| Enable W&B | `configs/models.yaml` → `training.use_wandb: true` |

---

## 6. Common issues

| Error | Fix |
|-------|-----|
| `FileNotFoundError: Dataset not found` | Check `path:` in datasets.yaml. Must be relative to `LV_code/`, not `configs/`. |
| `ValueError: Class with only 1 sample` | Pass a higher `min_samples_per_class` — either `load_dataset(..., min_samples_per_class=k)` directly, or via a k-sweep call (`run_benchmark`/`run_baseline_split(k_values=[...])`). There is no `datasets.yaml -> settings:` block anymore (it was dead/unread code and has been removed). |
| `ModuleNotFoundError: mrmr` | Run `uv sync` (installs `mrmr-selection`). |
| `Model not found` for interpretation | Run the benchmark first: `pipe.run_benchmark(...)`. |
| `no selected_features.json ... run run_baseline_split() first` | `run_rule_extraction_holdout` reuses `run_baseline_split`'s split+FS artifacts — run the k-sweep in notebook 02 for that k before running notebook 04. |
| `ImportError: wandb` | W&B is optional. Pass `use_wandb=False` or `pip install wandb`. |
| Notebook can't find `src` | Make sure `sys.path.insert(0, '..')` is the first cell (if running from `notebooks/`). |

---

## Quick reference: key functions

```python
# Load data manually
from src.helper.data_loader import load_dataset
data = load_dataset('Dataset/GEO_Cumida/20711/20711/Breast_GEOD20711.csv')
X, y, feature_names = data['X'], data['y'], data['feature_names']

# Feature selection
from src.feature_selection.mrmr_fs import run_mrmr_selection
res = run_mrmr_selection(X, y, feature_names, 'MyDS', 'outputs/', criterion='MIQ', K=50)

from src.feature_selection.boruta_fs import run_boruta_selection
res = run_boruta_selection(X, y, feature_names, 'MyDS', 'outputs/', selection_mode='auto')

# Train one model (models: nb, knn, svm, rf, dt, xgboost, ann)
from src.benchmark.trainer import BenchmarkTrainer
trainer = BenchmarkTrainer('MyDS', 'rf', 'mrmr_k50', 'outputs/')
summary = trainer.run_cv(res['X_selected'], y, res['selected_features'])
# Pass needs_scaling_map=... to override which models get StandardScaler
# (defaults to src/helper/scaling.py's DEFAULT_NEEDS_SCALING if omitted).

# SHAP
import joblib
from src.interpretation.shap_utils import explain_model
artifact = joblib.load('outputs/MyDS/mrmr_k50/rf/models/rf_fold_1.joblib')
explain_model(
    artifact['model'], X_scaled, feature_names, class_labels,
    probe_gene_map, output_dir='outputs/shap/',
)
```

---

Full documentation: **[REFERENCE.md](REFERENCE.md)**
