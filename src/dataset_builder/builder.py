"""
src/dataset_builder.py

Build CuMiDa-format datasets from a raw NCBI GEO **series matrix** file
(``GSE*_series_matrix.txt.gz``).

A GEO series matrix bundles three things in one file:
  1. ``!Series_*``  — study-level metadata (title, platform, summary, …)
  2. ``!Sample_*``  — per-sample metadata, incl. ``!Sample_characteristics_ch1``
  3. an expression table (rows = probe IDs, columns = sample GSM IDs)

This module turns that into two analysis-ready CSVs in CuMiDa layout
(``samples, type, feature_1, feature_2, …``) plus provenance files:

  <Tissue>_<GEO>_probe.csv      Dataset 2 — probe-level (feeds the Phase-1 pipeline directly)
  <Tissue>_<GEO>_gene.csv       Dataset 1 — gene-level (probes collapsed to gene symbols)
  <GEO>_metadata.csv            all per-sample characteristics
  <GEO>_mapping.csv             probID | gene_symbol | source | dedup_status
  <GEO>_annotation_report.txt   coverage stats (annotation rows / mapped / unmapped)

Probe → gene symbol resolution reuses ``src/annotation.py`` primitives:
  1. local GPL annotation file  (``.annot`` / ``.annot.gz``) — preferred, offline
  2. MyGene.info API            — fallback for probes the file did not cover

Gene-level collapse reuses ``src/mapping.collapse_probe_duplicates``.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import gzip
import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.helper import report


def _sanitize_path_component(value: str, fallback: str = "Dataset") -> str:
    """Collapse anything but [A-Za-z0-9_-] so a client-supplied tissue/GEO-id
    can't inject a path separator (or "..") into an output filename."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return cleaned or fallback
from .annotation import (
    extract_probe_gene_map,
    extract_probe_gene_map_multi,
    parse_platform_annotation,
    _mygene_lookup,
)
from .mapping import collapse_probe_duplicates


# =============================================================================
# Series matrix parsing
# =============================================================================

_TABLE_BEGIN = "!series_matrix_table_begin"
_TABLE_END = "!series_matrix_table_end"


def _strip_quotes(value: str) -> str:
    """Strip surrounding double quotes and whitespace from one cell."""
    return value.strip().strip('"').strip()


def _split_row(line: str) -> List[str]:
    """Tab-split a metadata line and de-quote each cell."""
    return [_strip_quotes(c) for c in line.rstrip("\n").split("\t")]


def parse_series_matrix(
    path: str | Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Parse a GEO series matrix file.

    Returns
    -------
    expr_df : pd.DataFrame
        Expression matrix, index = probe IDs (``probID``), columns = sample GSM IDs.
    sample_meta : pd.DataFrame
        One row per sample; columns = ``sample_id`` plus every characteristic
        found in ``!Sample_characteristics_ch1`` (sanitised column names) and a
        ``title`` / ``source`` column when present.
    series_meta : dict
        Selected study-level fields (geo_accession, title, platform_id, summary).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Series matrix not found: {path}")

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    # --- locate the expression table boundaries ---------------------------
    try:
        begin = next(i for i, l in enumerate(lines) if l.strip() == _TABLE_BEGIN)
        end = next(i for i, l in enumerate(lines) if l.strip() == _TABLE_END)
    except StopIteration:
        raise ValueError(
            f"No expression table ({_TABLE_BEGIN} … {_TABLE_END}) found in {path}. "
            "The file may be truncated."
        )

    meta_lines = lines[:begin]
    header_idx = begin + 1  # ID_REF + GSM columns

    # --- study-level metadata --------------------------------------------
    def _first(prefix: str) -> str:
        for l in meta_lines:
            if l.startswith(prefix):
                cells = _split_row(l)
                return cells[1] if len(cells) > 1 else ""
        return ""

    series_meta = {
        "geo_accession": _first("!Series_geo_accession"),
        "title": _first("!Series_title"),
        "platform_id": _first("!Series_platform_id"),
        "summary": _first("!Series_summary"),
    }

    # --- per-sample metadata ---------------------------------------------
    sample_ids = _split_row(next(l for l in meta_lines
                                 if l.startswith("!Sample_geo_accession")))[1:]
    meta_cols: Dict[str, List[str]] = {"sample_id": sample_ids}

    def _row_values(prefix: str) -> Optional[List[str]]:
        for l in meta_lines:
            if l.startswith(prefix):
                return _split_row(l)[1:]
        return None

    titles = _row_values("!Sample_title")
    if titles:
        meta_cols["title"] = titles
    source = _row_values("!Sample_source_name_ch1")
    if source:
        meta_cols["source"] = source

    # Each characteristic line normally holds "key: value" cells with the SAME
    # key for every sample — but GEO aligns characteristics by slot/position,
    # not by name, so a heterogeneous sample set (e.g. human tumor samples vs.
    # mouse tumorgraft samples) can put a DIFFERENT key in the same slot for
    # different samples (seen in GSE36895: one characteristics_ch1 line
    # alternates "bap1 status: ..." / "pathologic tnm staging: ..." cell by
    # cell). Deriving one column from just the first cell's key — as a naive
    # per-line approach would — silently folds the other key's values into
    # that column as garbage, and that key never gets its own column at all.
    # Split per line by the key actually on EACH cell instead; a normal
    # (homogeneous) line still yields exactly one column as before.
    seen_keys: Dict[str, int] = {}
    for l in meta_lines:
        if not l.startswith("!Sample_characteristics_ch1"):
            continue
        cells = _split_row(l)[1:]
        per_key_values: Dict[str, List[Optional[str]]] = {}
        for i, c in enumerate(cells):
            if ":" not in c:
                continue
            k, v = c.split(":", 1)
            k = k.strip()
            per_key_values.setdefault(k, [None] * len(cells))[i] = v.strip()
        if not per_key_values:
            per_key_values = {"characteristic": list(cells)}

        for key, values in per_key_values.items():
            col = re.sub(r"[^0-9a-z]+", "_", key.lower()).strip("_") or "characteristic"
            if col in seen_keys:
                seen_keys[col] += 1
                col = f"{col}_{seen_keys[col]}"
            else:
                seen_keys[col] = 0
            meta_cols[col] = values

    sample_meta = pd.DataFrame(meta_cols)

    # --- expression table -------------------------------------------------
    table_text = "".join(lines[header_idx:end])
    expr_df = pd.read_csv(io.StringIO(table_text), sep="\t", index_col=0, quotechar='"')
    expr_df.index.name = "probID"

    return expr_df, sample_meta, series_meta


# =============================================================================
# Class characteristic discovery (point 4 & 6: pick a clean class column)
# =============================================================================

_NA_TOKENS = {"", "na", "n/a", "nan", "none", "null", "--", "?"}


def discover_class_characteristics(sample_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Rank every candidate class column by suitability.

    A good class column has few NA-like values, a small number of distinct
    classes, and no singleton classes. Returns a ranked DataFrame so the
    caller can pick a column that avoids dropping samples.
    """
    rows: List[Dict[str, Any]] = []
    for col in sample_meta.columns:
        if col == "sample_id":
            continue
        col_raw = sample_meta[col]
        # .isna() on the RAW column first — see build_geo_dataset's identical
        # fix: under pandas's string dtype, a true-missing cell survives
        # .astype(str) as the dtype's own NA marker rather than the literal
        # text "nan"/"None", so token-matching the stringified value alone
        # misses every real (non-token) NA.
        is_missing = col_raw.isna()
        s = col_raw.astype(str).str.strip()
        is_na = is_missing | s.str.lower().isin(_NA_TOKENS)
        valid = s[~is_na]
        counts = valid.value_counts()
        rows.append({
            "characteristic": col,
            "n_classes": int(counts.size),
            "na_count": int(is_na.sum()),
            "min_class_size": int(counts.min()) if counts.size else 0,
            "classes": list(counts.index),
        })

    rank = pd.DataFrame(rows)
    if rank.empty:
        return rank
    # Prefer: 0 NA, then >=2 classes, then larger smallest-class, then fewer classes
    rank["_usable"] = (rank["n_classes"] >= 2).astype(int)
    rank = rank.sort_values(
        by=["_usable", "na_count", "min_class_size", "n_classes"],
        ascending=[False, True, False, True],
    ).drop(columns="_usable").reset_index(drop=True)
    return rank


# =============================================================================
# Probe → gene mapping with source tracking
# =============================================================================


def build_mapping_table(
    probe_ids: List[str],
    annotation_path: Optional[str | Path],
    *,
    use_mygene_fallback: bool = True,
    species: str = "human",
    probe_col: str = "ID",
    gene_col: str = "Gene Symbol",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Resolve probe IDs to gene symbols, recording the *source* of each mapping.

    Resolution order
    ----------------
    1. Local GPL annotation file (``.annot`` / ``.annot.gz`` / ``.soft.gz``).
    2. MyGene.info — queried only for probes the file did not cover.

    Returns
    -------
    mapping_df : pd.DataFrame  columns ``probID | gene_symbol | source``
        ``source`` ∈ {"annot", "mygene", "unmapped"}.
    mapping_dict : dict  probe_id -> gene_symbol  (only resolved probes).
    """
    mapping_dict: Dict[str, str] = {}
    source: Dict[str, str] = {}

    # 1. Local annotation file -------------------------------------------------
    if annotation_path is not None:
        annotation_path = Path(annotation_path)
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        report.step(f"Parsing annotation file: {annotation_path.name}")
        platform_df = parse_platform_annotation(annotation_path)  # SOFT or flat, auto-detected
        # Try the primary probe-ID column first, then other common identifier
        # columns some platforms expose (e.g. Agilent "_limpo" exports mix
        # design IDs with RefSeq/Ensembl accessions across features).
        id_cols = list(dict.fromkeys([probe_col, "GB_ACC", "REFSEQ", "ENSEMBL_ID"]))
        annot_map = extract_probe_gene_map_multi(platform_df, id_cols, gene_col=gene_col)
        for p in probe_ids:
            sym = annot_map.get(p)
            if sym:
                mapping_dict[p] = sym
                source[p] = "annot"
        report.ok(f"Annotation file mapped {len(mapping_dict):,}/{len(probe_ids):,} probes")

    # 2. MyGene.info fallback for the remainder -------------------------------
    missing = [p for p in probe_ids if p not in mapping_dict]
    if missing and use_mygene_fallback:
        report.step(f"MyGene.info fallback for {len(missing):,} unmapped probes")
        try:
            mg_map = _mygene_lookup(missing, species=species)
            for p, sym in mg_map.items():
                if sym:
                    mapping_dict[p] = sym
                    source[p] = "mygene"
            report.ok(f"MyGene.info recovered {len(mg_map):,} additional probes")
        except Exception as e:  # network/API failure must not abort the build
            report.warn(f"MyGene.info fallback failed: {e}")

    mapping_df = pd.DataFrame({
        "probID": probe_ids,
        "gene_symbol": [mapping_dict.get(p, "") for p in probe_ids],
        "source": [source.get(p, "unmapped") for p in probe_ids],
    })
    return mapping_df, mapping_dict


# =============================================================================
# CuMiDa assembly
# =============================================================================


def _build_cumida(
    feature_matrix: pd.DataFrame,   # index = sample_id, columns = features
    class_by_sample: "pd.Series",   # index = sample_id -> class string
) -> pd.DataFrame:
    """Assemble a CuMiDa-format frame: samples, type, <features…>."""
    df = feature_matrix.copy()
    df.insert(0, "type", class_by_sample.reindex(df.index).values)
    df.insert(0, "samples", df.index)
    return df.reset_index(drop=True)


# =============================================================================
# High-level builder
# =============================================================================


def build_geo_dataset(
    series_matrix_path: str | Path,
    *,
    class_characteristic: str,
    annotation_path: Optional[str | Path] = None,
    output_dir: str | Path,
    tissue: str = "Dataset",
    dedup_strategy: str = "mean",
    na_class_action: str = "exclude",
    na_class_label: str = "Unknown",
    use_mygene_fallback: bool = True,
    species: str = "human",
) -> Dict[str, Any]:
    """
    Build probe-level and gene-level CuMiDa datasets from a GEO series matrix.

    Parameters
    ----------
    series_matrix_path : path to ``GSE*_series_matrix.txt.gz``.
    class_characteristic : sample characteristic to use as the class label
        (e.g. ``"subtypeihc"``). Use :func:`discover_class_characteristics` to
        choose one that minimises NA.
    annotation_path : local GPL annotation file (preferred mapping source).
    output_dir : directory to write all output files into.
    tissue : prefix for the dataset filenames (e.g. ``"Breast"``).
    dedup_strategy : passed to ``collapse_probe_duplicates`` ('mean' | 'max_var').
    na_class_action : 'exclude' (drop NA-class samples) | 'keep_as_unknown'.
    na_class_label : class name used for NA samples when na_class_action is
        'keep_as_unknown' (e.g. ``"NON_TNM"`` for a staging characteristic
        where "no stage" is itself a meaningful class, not generic "Unknown").
    use_mygene_fallback : query MyGene.info for probes the annotation file misses.

    Returns
    -------
    dict — status summary (counts + output file paths).
    """
    series_matrix_path = Path(series_matrix_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tissue = _sanitize_path_component(tissue)

    report.section("GEO dataset builder", series_matrix_path.name)

    # 1. Parse -----------------------------------------------------------------
    report.step("Parsing series matrix")
    expr_df, sample_meta, series_meta = parse_series_matrix(series_matrix_path)
    geo_id = _sanitize_path_component(
        series_meta.get("geo_accession") or series_matrix_path.stem.split("_")[0]
    )
    n_probes_total = expr_df.shape[0]
    report.ok(f"{geo_id}: {expr_df.shape[0]:,} probes × {expr_df.shape[1]} samples "
              f"(platform {series_meta.get('platform_id', '?')})")

    if class_characteristic not in sample_meta.columns:
        ranking = discover_class_characteristics(sample_meta)
        report.dataframe_table(
            ranking.drop(columns="classes"),
            title="Available class characteristics (ranked)",
        )
        raise ValueError(
            f"class_characteristic '{class_characteristic}' not found. "
            f"Choose one of: {[c for c in sample_meta.columns if c != 'sample_id']}"
        )

    # 2. Class labels ----------------------------------------------------------
    # Check .isna() on the ORIGINAL column before stringifying: under pandas's
    # string dtype (default since pandas 3.x), a true-missing cell survives
    # .astype(str) as the dtype's own NA marker, not the literal text "nan"/
    # "None" it would become on legacy object dtype — so relying on the
    # stringified value alone silently misses every real (non-token) NA.
    class_col = sample_meta.set_index("sample_id")[class_characteristic]
    is_missing = class_col.isna()
    class_raw = class_col.astype(str).str.strip()
    is_na = is_missing | class_raw.str.lower().isin(_NA_TOKENS)
    n_na = int(is_na.sum())

    if na_class_action == "exclude":
        keep_samples = class_raw.index[~is_na]
        class_by_sample = class_raw.loc[keep_samples]
    elif na_class_action == "keep_as_unknown":
        class_by_sample = class_raw.mask(is_na, na_class_label)
        keep_samples = class_by_sample.index
    else:
        raise ValueError(f"Unknown na_class_action: {na_class_action!r}")

    report.info(f"Class '{class_characteristic}': {class_by_sample.nunique()} classes, "
                f"{n_na} NA sample(s) → action={na_class_action}")

    # 3. Probe → gene mapping --------------------------------------------------
    probe_ids = expr_df.index.tolist()
    mapping_df, mapping_dict = build_mapping_table(
        probe_ids, annotation_path,
        use_mygene_fallback=use_mygene_fallback, species=species,
    )
    n_mapped = int((mapping_df["source"] != "unmapped").sum())
    n_from_annot = int((mapping_df["source"] == "annot").sum())
    n_from_mygene = int((mapping_df["source"] == "mygene").sum())
    n_unmapped = n_probes_total - n_mapped

    # 3.5 Drop Affymetrix control probes (AFFX-*) that never resolved to a
    # gene — they are chip-QC artifacts, not biology, and would otherwise
    # pollute the probe-level dataset. Only probes that are BOTH AFFX-prefixed
    # AND unmapped are excluded from the expression matrices; they stay listed
    # in mapping_df (dedup_status="Removed_AFFX") for a full audit trail
    # instead of silently disappearing from GSE*_mapping.csv.
    is_affx_removed = (
        mapping_df["probID"].str.upper().str.startswith("AFFX")
        & (mapping_df["source"] == "unmapped")
    )
    n_affx_removed = int(is_affx_removed.sum())
    if n_affx_removed:
        report.info(f"Removing {n_affx_removed:,} unmapped AFFX-* control probe(s) from the matrix")
        keep_probe_ids = mapping_df.loc[~is_affx_removed, "probID"]
        expr_df = expr_df.loc[keep_probe_ids]

    # 4. Probe-level CuMiDa (Dataset 2) ---------------------------------------
    # expr_df: probes × samples  → transpose to samples × probes
    probe_matrix = expr_df[[s for s in keep_samples if s in expr_df.columns]].T
    probe_matrix.index.name = "sample_id"
    probe_cumida = _build_cumida(probe_matrix, class_by_sample)

    # 5. Gene-level CuMiDa (Dataset 1) ----------------------------------------
    report.step(f"Collapsing probes → genes (strategy={dedup_strategy})")
    gene_matrix, provenance = collapse_probe_duplicates(
        probe_matrix, mapping_dict, strategy=dedup_strategy,
    )
    gene_cumida = _build_cumida(gene_matrix, class_by_sample)

    # Enrich mapping_df with dedup status from the collapse provenance.
    # AFFX-removed probes never entered the collapse (excluded from
    # probe_matrix), so they'd otherwise fall back to "unmapped" — mark them
    # "Removed_AFFX" instead so the audit trail records why they're absent.
    status_by_probe = provenance.set_index("Probe_ID")["Status"].to_dict()
    mapping_df["dedup_status"] = mapping_df["probID"].map(status_by_probe)
    mapping_df.loc[is_affx_removed, "dedup_status"] = "Removed_AFFX"
    mapping_df["dedup_status"] = mapping_df["dedup_status"].fillna("unmapped")

    # 6. Write outputs ---------------------------------------------------------
    probe_path = output_dir / f"{tissue}_{geo_id}_probe.csv"
    gene_path = output_dir / f"{tissue}_{geo_id}_gene.csv"
    meta_path = output_dir / f"{geo_id}_metadata.csv"
    mapping_path = output_dir / f"{geo_id}_mapping.csv"
    report_path = output_dir / f"{geo_id}_annotation_report.txt"

    probe_cumida.to_csv(probe_path, index=False)
    gene_cumida.to_csv(gene_path, index=False)
    sample_meta.to_csv(meta_path, index=False)
    mapping_df.to_csv(mapping_path, index=False)

    report_text = (
        f"GEO dataset annotation report\n"
        f"{'=' * 40}\n"
        f"GEO accession            : {geo_id}\n"
        f"Platform                 : {series_meta.get('platform_id', '?')}\n"
        f"Title                    : {series_meta.get('title', '')}\n"
        f"Class characteristic     : {class_characteristic}\n"
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
        f"Samples kept             : {len(keep_samples)} (NA dropped: "
        f"{n_na if na_class_action == 'exclude' else 0})\n"
    )
    report_path.write_text(report_text, encoding="utf-8")

    report.ok(f"Wrote 5 files → {output_dir}")
    report.output_tree(output_dir, title=f"{geo_id} outputs")

    return {
        "geo_id": geo_id,
        "platform": series_meta.get("platform_id"),
        "class_characteristic": class_characteristic,
        "annotation_rows": n_probes_total,
        "mapped_total": n_mapped,
        "mapped_from_annot": n_from_annot,
        "mapped_from_mygene": n_from_mygene,
        "unmapped": n_unmapped,
        "affx_removed": n_affx_removed,
        "unique_probes": int(probe_matrix.shape[1]),
        "unique_genes": int(gene_matrix.shape[1]),
        "samples_kept": int(len(keep_samples)),
        "na_samples": n_na,
        "class_distribution": dict(class_by_sample.value_counts()),
        "output_files": {
            "probe_dataset": str(probe_path),
            "gene_dataset": str(gene_path),
            "metadata": str(meta_path),
            "mapping": str(mapping_path),
            "report": str(report_path),
        },
    }
