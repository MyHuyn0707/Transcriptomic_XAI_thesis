"""
src/dataset_builder/cumida.py

Re-annotate an existing CuMiDa-format probe-level CSV the same way
``build_geo_dataset`` processes a raw GEO series matrix.

CuMiDa (https://sbcb.inf.ufrgs.br/cumida) already ships analysis-ready
``samples, type, probe_1, probe_2, …`` CSVs with curated class labels, so
there's no series-matrix parsing or class-characteristic discovery to do —
only probe → gene annotation, AFFX control-probe removal, and gene-level
collapse, producing the same five output files as the GEO builder:

  <Tissue>_<GEO>_probe.csv      Dataset 2 — probe-level (AFFX-* controls removed)
  <Tissue>_<GEO>_gene.csv       Dataset 1 — gene-level (probes collapsed to gene symbols)
  <GEO>_metadata.csv            sample_id | type
  <GEO>_mapping.csv             probID | gene_symbol | source | dedup_status
  <GEO>_annotation_report.txt   coverage stats (annotation rows / mapped / unmapped / AFFX removed)

Probe → gene resolution and AFFX handling reuse the same primitives as
``build_geo_dataset`` (:func:`build_mapping_table`, :func:`_build_cumida`,
``collapse_probe_duplicates``) so both builders stay in lockstep.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.helper import report
from .builder import _build_cumida, _sanitize_path_component, build_mapping_table
from .mapping import collapse_probe_duplicates


def build_cumida_dataset(
    probe_csv_path: str | Path,
    *,
    annotation_path: Optional[str | Path] = None,
    output_dir: str | Path,
    tissue: Optional[str] = None,
    geo_id: Optional[str] = None,
    platform: Optional[str] = None,
    dedup_strategy: str = "mean",
    use_mygene_fallback: bool = False,
    species: str = "human",
) -> Dict[str, Any]:
    """
    Build a gene-level CuMiDa dataset (+ mapping/report) from an existing
    CuMiDa probe-level CSV.

    Parameters
    ----------
    probe_csv_path : path to the existing ``<Tissue>_<GEO>.csv`` CuMiDa file
        (``samples, type, probe_1, probe_2, …``).
    annotation_path : local platform annotation file (GEO SOFT or CuMiDa's
        flat ``GPL*_limpo.txt.gz``) — auto-detected format.
    output_dir : directory to write all output files into.
    tissue : prefix for the dataset filenames; defaults to the text before
        the first ``_`` in ``probe_csv_path``'s stem (e.g. ``Breast`` from
        ``Breast_GSE45827.csv``).
    geo_id : dataset accession; defaults to the text after the last ``_``
        in the stem (e.g. ``GSE45827``).
    platform : GPL accession, recorded in the report/return value only
        (CuMiDa CSVs carry no embedded platform metadata).
    dedup_strategy : passed to ``collapse_probe_duplicates`` ('mean' | 'max_var').
    use_mygene_fallback : query MyGene.info for probes the annotation file misses.

    Returns
    -------
    dict — status summary (counts + output file paths), same shape as
    :func:`build_geo_dataset`'s return value.
    """
    probe_csv_path = Path(probe_csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem_parts = probe_csv_path.stem.split("_")
    tissue = _sanitize_path_component(tissue or stem_parts[0])
    geo_id = _sanitize_path_component(geo_id or stem_parts[-1])

    report.section("CuMiDa dataset re-annotator", probe_csv_path.name)

    # 1. Load ------------------------------------------------------------------
    report.step("Loading CuMiDa probe-level CSV")
    df = pd.read_csv(probe_csv_path, low_memory=False)
    if list(df.columns[:2]) != ["samples", "type"]:
        raise ValueError(
            f"{probe_csv_path} is not CuMiDa layout: expected columns starting "
            f"with ['samples', 'type'], got {list(df.columns[:2])}"
        )

    sample_ids = df["samples"].astype(str)
    class_by_sample = pd.Series(df["type"].astype(str).values, index=sample_ids)
    probe_ids = df.columns[2:].tolist()
    n_probes_total = len(probe_ids)
    report.ok(
        f"{geo_id}: {n_probes_total:,} probes × {len(df)} samples"
        + (f" (platform {platform})" if platform else "")
    )

    # Index by the string-cast sample_ids (not the raw "samples" column) so
    # this matches class_by_sample's index dtype even when sample IDs are
    # numeric (e.g. GSE45827/GSE50161 use bare integers, not GSM accessions) —
    # a dtype mismatch here would silently reindex every class label to NaN.
    probe_matrix_full = df.set_index(sample_ids)[probe_ids]
    probe_matrix_full.index.name = "sample_id"

    # 2. Probe → gene mapping ----------------------------------------------------
    mapping_df, mapping_dict = build_mapping_table(
        probe_ids, annotation_path,
        use_mygene_fallback=use_mygene_fallback, species=species,
    )
    n_mapped = int((mapping_df["source"] != "unmapped").sum())
    n_from_annot = int((mapping_df["source"] == "annot").sum())
    n_from_mygene = int((mapping_df["source"] == "mygene").sum())
    n_unmapped = n_probes_total - n_mapped

    # 3. Drop Affymetrix control probes (AFFX-*) that never resolved to a gene —
    # same rule as build_geo_dataset(): excluded from the matrices, but kept as
    # rows in mapping_df (dedup_status="Removed_AFFX") for a full audit trail.
    is_affx_removed = (
        mapping_df["probID"].str.upper().str.startswith("AFFX")
        & (mapping_df["source"] == "unmapped")
    )
    n_affx_removed = int(is_affx_removed.sum())
    if n_affx_removed:
        report.info(f"Removing {n_affx_removed:,} unmapped AFFX-* control probe(s) from the matrix")
        keep_probe_ids = mapping_df.loc[~is_affx_removed, "probID"]
        probe_matrix = probe_matrix_full[keep_probe_ids]
    else:
        probe_matrix = probe_matrix_full

    # 4. Probe-level CuMiDa (Dataset 2) ------------------------------------------
    probe_cumida = _build_cumida(probe_matrix, class_by_sample)

    # 5. Gene-level CuMiDa (Dataset 1) -------------------------------------------
    report.step(f"Collapsing probes → genes (strategy={dedup_strategy})")
    gene_matrix, provenance = collapse_probe_duplicates(
        probe_matrix, mapping_dict, strategy=dedup_strategy,
    )
    gene_cumida = _build_cumida(gene_matrix, class_by_sample)

    status_by_probe = provenance.set_index("Probe_ID")["Status"].to_dict()
    mapping_df["dedup_status"] = mapping_df["probID"].map(status_by_probe)
    mapping_df.loc[is_affx_removed, "dedup_status"] = "Removed_AFFX"
    mapping_df["dedup_status"] = mapping_df["dedup_status"].fillna("unmapped")

    # 6. Write outputs -------------------------------------------------------
    probe_path = output_dir / f"{tissue}_{geo_id}_probe.csv"
    gene_path = output_dir / f"{tissue}_{geo_id}_gene.csv"
    meta_path = output_dir / f"{geo_id}_metadata.csv"
    mapping_path = output_dir / f"{geo_id}_mapping.csv"
    report_path = output_dir / f"{geo_id}_annotation_report.txt"

    probe_cumida.to_csv(probe_path, index=False)
    gene_cumida.to_csv(gene_path, index=False)
    pd.DataFrame({"sample_id": sample_ids.values, "type": class_by_sample.values}).to_csv(
        meta_path, index=False
    )
    mapping_df.to_csv(mapping_path, index=False)

    report_text = (
        f"CuMiDa dataset annotation report\n"
        f"{'=' * 40}\n"
        f"GEO accession            : {geo_id}\n"
        f"Source CuMiDa CSV        : {probe_csv_path.name}\n"
        f"Platform                 : {platform or '?'}\n"
        f"Class characteristic     : type (pre-labelled by CuMiDa)\n"
        f"Classes ({class_by_sample.nunique()})              : "
        f"{dict(class_by_sample.value_counts())}\n"
        f"{'-' * 40}\n"
        f"Annotation rows          : {n_probes_total:,}\n"
        f"Probes with gene symbol  : {n_mapped:,}\n"
        f"  from annotation file   : {n_from_annot:,}\n"
        f"  from MyGene.info       : {n_from_mygene:,}\n"
        f"Probes without           : {n_unmapped:,}\n"
        f"AFFX control probes removed : {n_affx_removed:,}\n"
        f"{'-' * 40}\n"
        f"Unique probes (Dataset 2): {probe_matrix.shape[1]:,}\n"
        f"Unique genes (Dataset 1) : {gene_matrix.shape[1]:,}\n"
        f"Samples kept             : {len(sample_ids)} (NA dropped: 0)\n"
    )
    report_path.write_text(report_text, encoding="utf-8")

    report.ok(f"Wrote 5 files → {output_dir}")
    report.output_tree(output_dir, title=f"{geo_id} outputs")

    return {
        "geo_id": geo_id,
        "platform": platform,
        "class_characteristic": "type",
        "annotation_rows": n_probes_total,
        "mapped_total": n_mapped,
        "mapped_from_annot": n_from_annot,
        "mapped_from_mygene": n_from_mygene,
        "unmapped": n_unmapped,
        "affx_removed": n_affx_removed,
        "unique_probes": int(probe_matrix.shape[1]),
        "unique_genes": int(gene_matrix.shape[1]),
        "samples_kept": int(len(sample_ids)),
        "na_samples": 0,
        "class_distribution": dict(class_by_sample.value_counts()),
        "output_files": {
            "probe_dataset": str(probe_path),
            "gene_dataset": str(gene_path),
            "metadata": str(meta_path),
            "mapping": str(mapping_path),
            "report": str(report_path),
        },
    }
