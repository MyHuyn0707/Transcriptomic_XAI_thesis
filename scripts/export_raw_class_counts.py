"""
scripts/export_raw_class_counts.py

One-shot export: for every dataset with cached holdout results, computes
``raw_class_counts`` (class distribution BEFORE rare-class removal — the raw
CSV never changes at runtime) and ``dropped_classes`` (diffed against
split_info.json's surviving class_labels), and writes both to
``outputs_holdout/{dataset}/raw_class_counts.json``.

The API (src/api/registry.load_raw_class_counts) reads this cache file
instead of re-reading the raw dataset CSV on every /overview request — it's
also self-healing (computes + writes on first miss), so running this script
is an optimization, not a requirement; but running it once means the very
first request after a server restart doesn't pay the CSV-read cost either.

Run:
    uv run python scripts/export_raw_class_counts.py

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api import registry  # noqa: E402


def main() -> None:
    datasets = registry.list_datasets()
    print(f"{len(datasets)} dataset(s) to export.")

    for ds in datasets:
        dataset_id = ds["id"]
        cache_path = registry._raw_class_counts_cache_path(dataset_id)
        cache_path.unlink(missing_ok=True)  # force a fresh read of the raw CSV, not a stale cache file
        registry.load_raw_class_counts.cache_clear()  # and don't serve a stale in-memory hit either
        result = registry.load_raw_class_counts(dataset_id)
        n_classes = len(result["raw_class_counts"])
        n_dropped = len(result["dropped_classes"])
        print(f"[{dataset_id}] {n_classes} class(es), {n_dropped} dropped -> {cache_path.relative_to(PROJECT_ROOT)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
