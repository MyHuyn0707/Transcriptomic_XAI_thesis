"""
scripts/merge_annotation_into_description.py

One-shot (re-runnable) migration: bakes the GEO/CuMiDa build-time annotation
report (Dataset/GSE_*/*_annotation_report.txt — accession, platform, title,
class_characteristic, raw class counts, probe/gene annotation coverage, AFFX
probes removed, unique probes/genes, samples kept) directly into each
Dataset_Description/*.json file as a new top-level "origin" key.

Why: previously the API read the annotation report and the curated
disease-context content from two different places at request time
(src/api/registry.py). Baking "origin" into the SAME file Dataset_Description
already ships means the API only ever reads one file per dataset, and
Dataset_Description/ becomes fully self-contained — hand it to someone else
and they get everything, no dependency on Dataset/ being present too.

Matches each Dataset_Description/*.json to its annotation report via GEO
accession (configs/dataset_origin.yaml -> output_dir), same as before — not
by filename, since naming conventions differ (Brain_GSE15824_probe.json has
no relation to Dataset/GSE_Dataset_from_Cumida/GSE_15824/).

Re-run any time Dataset_Description/ is replaced with a fresh zip export —
safe to re-run, always overwrites "origin" with a freshly re-parsed report
(everything else in the file is left untouched).

Run:
    uv run python scripts/merge_annotation_into_description.py

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.helper.config_loader import ConfigLoader  # noqa: E402

DESCRIPTION_ROOT = PROJECT_ROOT / "Dataset_Description"
SKIP_FILES = {"index.json", "schema_notes_vi.json", "source_audit.json", "common_provenance_notes.json"}


def _to_int(s: str) -> int:
    return int(s.split()[0].replace(",", ""))


def parse_annotation_report(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    raw: Dict[str, str] = {}
    classes_raw: Dict[str, int] = {}

    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("=") or s.startswith("-"):
            continue
        if s.startswith("Classes ("):
            _, val = s.split(":", 1)
            val_clean = re.sub(r"np\.int64\((\d+)\)", r"\1", val.strip())
            classes_raw = {k: int(v) for k, v in ast.literal_eval(val_clean).items()}
            continue
        if ":" in s:
            key, val = s.split(":", 1)
            raw[key.strip()] = val.strip()

    samples_kept = na_dropped = None
    m = re.match(r"([\d,]+)\s*\(NA dropped:\s*(\d+)\)", raw.get("Samples kept", ""))
    if m:
        samples_kept = int(m.group(1).replace(",", ""))
        na_dropped = int(m.group(2))

    return {
        "geo_accession": raw.get("GEO accession"),
        "platform": raw.get("Platform"),
        "title": raw.get("Title"),
        "source_cumida_csv": raw.get("Source CuMiDa CSV"),
        "class_characteristic": raw.get("Class characteristic"),
        "classes_raw": classes_raw,
        "annotation_rows": _to_int(raw["Annotation rows"]) if "Annotation rows" in raw else None,
        "probes_with_gene_symbol": _to_int(raw["Probes with gene symbol"]) if "Probes with gene symbol" in raw else None,
        "probes_from_annotation_file": _to_int(raw["from annotation file"]) if "from annotation file" in raw else None,
        "probes_from_mygene": _to_int(raw["from MyGene.info"]) if "from MyGene.info" in raw else None,
        "probes_without_gene_symbol": _to_int(raw["Probes without"]) if "Probes without" in raw else None,
        "affx_control_removed": _to_int(raw["AFFX control probes removed"]) if "AFFX control probes removed" in raw else None,
        "unique_probes": _to_int(raw["Unique probes (Dataset 2)"]) if "Unique probes (Dataset 2)" in raw else None,
        "unique_genes": _to_int(raw["Unique genes (Dataset 1)"]) if "Unique genes (Dataset 1)" in raw else None,
        "samples_kept": samples_kept,
        "na_dropped": na_dropped,
    }


def find_annotation_report(loader: ConfigLoader, gse_id: str) -> Optional[Path]:
    for getter in (loader.get_origin_config, loader.get_cumida_origin_config):
        try:
            cfg = getter(gse_id)
        except ValueError:
            continue
        matches = list(Path(cfg["output_dir"]).glob("*_annotation_report.txt"))
        if matches:
            return matches[0]
    return None


def main() -> None:
    loader = ConfigLoader(PROJECT_ROOT / "configs")

    updated = skipped = 0
    for path in sorted(DESCRIPTION_ROOT.glob("*.json")):
        if path.name in SKIP_FILES:
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        gse_id = data.get("accession")
        if not gse_id:
            print(f"[{path.name}] SKIP: no 'accession' field.")
            skipped += 1
            continue

        report_path = find_annotation_report(loader, gse_id)
        if not report_path:
            print(f"[{path.name}] WARN: no annotation_report.txt found for {gse_id}.")
            skipped += 1
            continue

        data["origin"] = parse_annotation_report(report_path)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{path.name}] OK (origin from {report_path.relative_to(PROJECT_ROOT)})")
        updated += 1

    print(f"\nDone. {updated} updated, {skipped} skipped.")


if __name__ == "__main__":
    main()
