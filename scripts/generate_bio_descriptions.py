"""
scripts/generate_bio_descriptions.py

Offline, one-shot script: for every dataset x fs_method x rule model under
outputs_holdout/, calls Gemini ONCE PER CHUNK of genes and ONCE PER CHUNK of
rules (see src/api/llm.py: describe_genes_batch / describe_rules_batch —
up to 40 items per request, not one request per item), and writes the
results next to the source files as gene_description_llm.json
({gene: text}) and rules_llm.json ({rule_id: text}).

The API's "Tai log cu" path only ever reads these cached files — it never
calls Gemini live for gene/rule descriptions (see src/api/llm.py docstring
for the one live call site, at predict time). The same generation function
also runs (best-effort) right after a live "Huan luyen" training job
finishes (src/api/jobs.py), so this script only needs to cover the cached
outputs_holdout/ runs.

Run once before the defense:
    uv run python scripts/generate_bio_descriptions.py
    uv run python scripts/generate_bio_descriptions.py --dataset CuMiDa-Brain-15824

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from src.api import llm  # noqa: E402
from src.api.registry import holdout_root_for  # noqa: E402

HOLDOUT_ROOT = holdout_root_for()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="Limit to one dataset (default: all).")
    args = parser.parse_args()

    rules_dirs = sorted(HOLDOUT_ROOT.glob("*/*/*/rules"))
    if args.dataset:
        rules_dirs = [d for d in rules_dirs if d.parts[len(HOLDOUT_ROOT.parts)] == args.dataset]

    if not rules_dirs:
        print("No rules/ directories found under outputs_holdout/.")
        return

    total_genes = total_rules = 0
    for rules_dir in rules_dirs:
        label = "/".join(rules_dir.relative_to(HOLDOUT_ROOT).parts[:-1])
        print(f"[{label}]")
        stats = llm.generate_bio_descriptions_for_rules_dir(rules_dir, log=print)
        total_genes += stats["genes"]
        total_rules += stats["rules"]

    print(f"\nDone. New gene descriptions: {total_genes}. New rule descriptions: {total_rules}.")


if __name__ == "__main__":
    main()
