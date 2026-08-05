"""
src/annotation.py

Probe-to-gene annotation for microarray data.

Features
--------
- Automatic GPL download from NCBI GEO FTP with local caching.
- Parse GPL SOFT files to extract probe → gene symbol mappings.
- ADF (ArrayExpress Array Design Format) download and parsing.
- SDRF auto-detection of platform via Array Design REF column.
- Platform alias resolution (common chip names / ADF accessions → GPL ID).
- Duplicate probe handling (max_mean / first / mean strategies).
- Fallback: load from a user-supplied local annotation file.
- Fallback: query MyGene.info API.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import gzip
import io
import re
import urllib.request
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# =============================================================================
# Platform → GPL ID mapping
# =============================================================================

PLATFORM_TO_GPL: Dict[str, str] = {
    # Affymetrix Human
    "HG-U133_Plus_2": "GPL570",
    "HG-U133A": "GPL96",
    "HG-U133A_2": "GPL571",
    "HG-U133B": "GPL97",
    "HG-U95A": "GPL8300",
    "HuGene-1_0-st-v1": "GPL6244",
    "HuGene-2_0-st": "GPL16686",
    "HuEx-1_0-st-v2": "GPL5188",
    "Human_Genome_U133_Plus_2.0": "GPL570",
    # Illumina Human
    "HumanHT-12_V4_0_R1": "GPL10558",
    "HumanWG-6_V3_0_R3": "GPL6947",
    "HumanRef-8_V3_0_R3": "GPL6883",
    # Aliases (chip descriptions)
    "Affymetrix GeneChip Human Genome U133 Plus 2.0": "GPL570",
    "[HG-U133_Plus_2]": "GPL570",
    "[HG-U133A]": "GPL96",
    "[Mouse430_2]": "GPL1261",
}

# ADF accession → GPL ID (ArrayExpress array design → NCBI GEO platform)
ADF_TO_GPL: Dict[str, str] = {
    "A-AFFY-44": "GPL570",   # HG-U133 Plus 2.0
    "A-AFFY-33": "GPL96",    # HG-U133A
    "A-AFFY-34": "GPL571",   # HG-U133A_2
    "A-AFFY-37": "GPL97",    # HG-U133B
    "A-AFFY-1":  "GPL91",    # HG-U95Av2
    "A-AFFY-2":  "GPL8300",  # HG-U95A
    "A-AFFY-141": "GPL6244", # HuGene-1_0-st-v1
    "A-AFFY-176": "GPL16686",# HuGene-2_0-st
    "A-AFFY-72":  "GPL5188", # HuEx-1_0-st-v2
    # NOTE: A-AGIL-11 (Agilent Whole Human Genome G4112A, 22K) is intentionally
    # omitted — its features are RefSeq IDs (NM_*), not probe IDs, so MyGene.info
    # (the default fallback) handles them directly without a GPL SOFT file.
}

# EBI FTP base URL for ADF files
_EBI_ADF_BASE = "https://ftp.ebi.ac.uk/pub/databases/arrayexpress/data/array"

# Default cache directory
DEFAULT_CACHE_DIR = Path("cache") / "gpl"


# =============================================================================
# GPL ID resolution
# =============================================================================


def resolve_gpl_id(platform: str) -> Optional[str]:
    """
    Resolve a platform name, alias, or ADF accession to a GPL accession.

    Accepts:
    - Direct GPL ID ('GPL570', 'gpl570')
    - ADF accession ('A-AFFY-44')
    - Common chip names from PLATFORM_TO_GPL
    - Partial matches (case-insensitive substring search)

    Returns
    -------
    str | None
    """

    platform = platform.strip()

    # Direct GPL format
    if re.match(r"(?i)^gpl\d+$", platform):
        return "GPL" + re.sub(r"(?i)^gpl", "", platform)

    # ADF accession (e.g. A-AFFY-44)
    if re.match(r"(?i)^A-[A-Z]+-\d+$", platform):
        upper = platform.upper()
        if upper in ADF_TO_GPL:
            return ADF_TO_GPL[upper]
        return None

    # Exact match
    if platform in PLATFORM_TO_GPL:
        return PLATFORM_TO_GPL[platform]

    # Case-insensitive match
    platform_lower = platform.lower()
    for alias, gpl_id in PLATFORM_TO_GPL.items():
        if platform_lower == alias.lower():
            return gpl_id

    # Substring match (e.g. user passes the full chip description)
    for alias, gpl_id in PLATFORM_TO_GPL.items():
        if platform_lower in alias.lower() or alias.lower() in platform_lower:
            return gpl_id

    return None


# =============================================================================
# SDRF platform detection
# =============================================================================


def detect_platform_from_sdrf(sdrf_path: str | Path) -> Optional[str]:
    """
    Read an SDRF file and extract the array design accession from the
    'Array Design REF' column.

    Returns the raw value (e.g. 'A-AFFY-44') or None if not found.
    """
    sdrf_path = Path(sdrf_path)
    if not sdrf_path.exists():
        return None

    try:
        df = pd.read_csv(sdrf_path, sep="\t", dtype=str, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
    except Exception:
        return None

    for col in df.columns:
        if "array design" in col.lower() and "ref" in col.lower():
            vals = df[col].dropna().unique()
            if len(vals) > 0:
                return str(vals[0]).strip()

    return None


# =============================================================================
# GPL SOFT download & cache
# =============================================================================


def _gpl_prefix(gpl_id: str) -> str:
    """
    Compute the NCBI FTP thousand-range directory for a GPL ID.

    NCBI organises GPL files into directories by thousands range:
      GPL1–GPL999   → GPLnnn/
      GPL1000–1999  → GPL1nnn/
      GPL6000–6999  → GPL6nnn/
      GPL10000–10999 → GPL10nnn/
    """
    gpl_num = int(gpl_id[3:])
    if gpl_num < 1000:
        return "GPLnnn"
    return f"GPL{gpl_num // 1000}nnn"


def download_gpl(
    platform: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_download: bool = False,
) -> Path:
    """
    Download a GPL SOFT file from NCBI GEO FTP and cache it locally.

    Parameters
    ----------
    platform : str
        Platform name/alias or GPL/ADF ID (e.g. 'HG-U133_Plus_2', 'GPL570',
        'A-AFFY-44').
    cache_dir : Path
        Local cache directory.
    force_download : bool
        Re-download even if a cached file exists.

    Returns
    -------
    Path to cached .soft.gz file.
    """

    gpl_id = resolve_gpl_id(platform)
    if gpl_id is None:
        raise ValueError(
            f"Cannot resolve platform '{platform}' to a GPL ID. "
            "Pass platform='GPL570' explicitly, or add it to PLATFORM_TO_GPL."
        )

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{gpl_id}_family.soft.gz"

    if cache_path.exists() and not force_download:
        return cache_path

    prefix = _gpl_prefix(gpl_id)
    url = (
        f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{prefix}/{gpl_id}/soft/"
        f"{gpl_id}_family.soft.gz"
    )

    print(f"  Downloading {gpl_id} from NCBI GEO FTP...")
    try:
        urllib.request.urlretrieve(url, cache_path)
        print(f"  Cached -> {cache_path}")
    except Exception as e:
        if cache_path.exists():
            cache_path.unlink()
        raise ConnectionError(
            f"Failed to download GPL file for {gpl_id}: {e}\n"
            "Check your internet connection or provide a local annotation file."
        ) from e

    return cache_path


# =============================================================================
# SOFT file parser
# =============================================================================


def parse_gpl_soft(soft_path: str | Path) -> pd.DataFrame:
    """
    Parse a GPL SOFT file and extract the probe-to-gene mapping table.

    Looks for the data table section (between !platform_table_begin /
    !platform_table_end) and returns it as a DataFrame.

    Returns
    -------
    pd.DataFrame with at least 'ID' and 'Gene Symbol' columns when present.
    """

    soft_path = Path(soft_path)

    opener = gzip.open if soft_path.suffix == ".gz" else open

    lines = []
    in_table = False

    with opener(soft_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!platform_table_begin"):
                in_table = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if in_table:
                lines.append(line)

    if not lines:
        raise ValueError(
            f"No platform data table found in {soft_path}. "
            "The file may be truncated or use a non-standard format."
        )

    df = pd.read_csv(io.StringIO("".join(lines)), sep="\t", dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    return df


def parse_platform_annotation(path: str | Path) -> pd.DataFrame:
    """
    Parse a local platform annotation file, auto-detecting its format.

    Supports two layouts sharing an ``ID`` / ``Gene Symbol`` (or equivalent)
    header, so both feed :func:`extract_probe_gene_map` unchanged:

    - GEO GPL **SOFT** files (``.annot`` / ``.annot.gz`` / ``.soft.gz``) —
      delimited by ``!platform_table_begin`` / ``!platform_table_end``.
    - Flat delimited tables with no SOFT markers, e.g. CuMiDa's
      ``GPL*_limpo.txt.gz`` platform exports — read as-is.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        head = fh.read(8192)

    if "!platform_table_begin" in head:
        return parse_gpl_soft(path)

    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False, comment="#")
    df.columns = [c.strip() for c in df.columns]
    return df


# =============================================================================
# ADF download & parser
# =============================================================================


def _adf_url(adf_accession: str) -> str:
    """
    Build the EBI FTP URL for an ADF file.

    URL pattern:
      {_EBI_ADF_BASE}/{ORGANISM_DIR}/{accession}/{accession}.adf.txt

    The organism directory is derived from the second part of the accession,
    e.g. A-AFFY-44 → AFFY/, A-AGIL-1 → AGIL/.
    """
    parts = adf_accession.upper().split("-")  # ['A', 'AFFY', '44']
    if len(parts) < 3:
        raise ValueError(f"Invalid ADF accession: {adf_accession!r}")
    organism_dir = parts[1]  # AFFY
    return f"{_EBI_ADF_BASE}/{organism_dir}/{adf_accession}/{adf_accession}.adf.txt"


def download_adf(
    adf_accession: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_download: bool = False,
) -> Path:
    """
    Download an ADF file from EBI ArrayExpress FTP and cache it locally.

    Parameters
    ----------
    adf_accession : str
        ADF accession, e.g. 'A-AFFY-44'.
    cache_dir : Path
        Local cache directory.
    force_download : bool
        Re-download even if a cached file exists.

    Returns
    -------
    Path to the cached ADF .txt file.
    """
    adf_accession = adf_accession.upper()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{adf_accession}.adf.txt"

    if cache_path.exists() and not force_download:
        return cache_path

    url = _adf_url(adf_accession)
    print(f"  Downloading {adf_accession} from EBI ArrayExpress FTP...")
    try:
        urllib.request.urlretrieve(url, cache_path)
        print(f"  Cached -> {cache_path}")
    except Exception as e:
        if cache_path.exists():
            cache_path.unlink()
        raise ConnectionError(
            f"Failed to download ADF file for {adf_accession}: {e}\n"
            "Check your internet connection or provide the GPL ID directly."
        ) from e

    return cache_path


def parse_adf(adf_path: str | Path) -> pd.DataFrame:
    """
    Parse an ADF (Array Design Format) file from EBI ArrayExpress.

    ADF files have a header section followed by a [main] section.
    The [main] section is a tab-separated table where the first column
    'Composite Element Name' contains the probe IDs.

    Returns
    -------
    pd.DataFrame  (rows = probes; columns include 'Composite Element Name'
                   and various database cross-reference columns)
    """
    adf_path = Path(adf_path)

    lines = []
    in_main = False

    with open(adf_path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped == "[main]":
                in_main = True
                continue
            if in_main:
                lines.append(line)

    if not lines:
        raise ValueError(
            f"No [main] data section found in {adf_path}. "
            "The file may be empty or use a non-standard format."
        )

    df = pd.read_csv(
        io.StringIO("".join(lines)), sep="\t", dtype=str, low_memory=False
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def extract_probe_ids_from_adf(adf_df: pd.DataFrame) -> List[str]:
    """
    Extract the list of probe IDs from a parsed ADF DataFrame.

    The probe ID column is 'Composite Element Name'.
    """
    col = "Composite Element Name"
    if col not in adf_df.columns:
        raise ValueError(
            f"Column '{col}' not found in ADF. "
            f"Available: {list(adf_df.columns)}"
        )
    return adf_df[col].dropna().str.strip().tolist()


# =============================================================================
# Probe → gene symbol extraction
# =============================================================================


def extract_probe_gene_map(
    platform_df: pd.DataFrame,
    probe_col: str = "ID",
    gene_col: str = "Gene Symbol",
) -> Dict[str, str]:
    """
    Build a probe_id -> gene_symbol dict from a platform DataFrame.

    Handles:
    - Multiple gene symbols per probe separated by ' /// ' (Affymetrix convention).
    - NaN / empty entries are dropped.

    Returns only the first gene symbol when multiple are listed.
    """

    if probe_col not in platform_df.columns:
        raise ValueError(
            f"Column '{probe_col}' not found. "
            f"Available columns: {list(platform_df.columns)}"
        )

    if gene_col not in platform_df.columns:
        # Try common alternatives
        for alt in ("Gene symbol", "gene_symbol", "GENE_SYMBOL", "Symbol"):
            if alt in platform_df.columns:
                gene_col = alt
                break
        else:
            raise ValueError(
                f"Gene symbol column not found. "
                f"Available columns: {list(platform_df.columns)}"
            )

    df = platform_df[[probe_col, gene_col]].dropna(subset=[probe_col, gene_col])
    df = df[df[gene_col].str.strip() != ""]

    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        probe = str(row[probe_col]).strip()
        gene = str(row[gene_col]).strip().split("///")[0].strip()
        if probe and gene:
            mapping[probe] = gene

    return mapping


def extract_probe_gene_map_multi(
    platform_df: pd.DataFrame,
    probe_cols: List[str],
    gene_col: str = "Gene Symbol",
) -> Dict[str, str]:
    """
    Build a probe_id -> gene_symbol dict by trying several identifier columns.

    Some platform annotation exports don't use one consistent probe-ID
    namespace — e.g. CuMiDa's Agilent ``GPL6848``/``GPL6480`` "_limpo" files:
    a dataset's CSV header may mix the platform's design ID (``ID``) with
    RefSeq (``GB_ACC``/``REFSEQ``) or Ensembl (``ENSEMBL_ID``) accessions for
    different features. Each candidate column present in ``platform_df`` is
    tried in order; the first column to resolve a given probe ID wins.

    Returns
    -------
    dict  probe_id -> gene_symbol, unioned across every column tried.
    """
    mapping: Dict[str, str] = {}
    for col in probe_cols:
        if col not in platform_df.columns:
            continue
        col_map = extract_probe_gene_map(platform_df, probe_col=col, gene_col=gene_col)
        for probe, gene in col_map.items():
            mapping.setdefault(probe, gene)
    return mapping


# =============================================================================
# Duplicate probe strategy
# =============================================================================


def deduplicate_probes(
    X: "pd.DataFrame",
    mapping: Dict[str, str],
    strategy: str = "max_mean",
) -> "pd.DataFrame":
    """
    When multiple probes map to the same gene symbol, aggregate them.

    Parameters
    ----------
    X : pd.DataFrame
        Rows = samples, columns = probe IDs.
    mapping : dict
        probe_id -> gene_symbol.
    strategy : str
        'max_mean' : keep the probe with the highest mean expression across samples.
        'mean'     : average all probes per gene.
        'first'    : keep the first encountered probe per gene.

    Returns
    -------
    pd.DataFrame with gene symbols as columns.
    """

    gene_names = [mapping.get(p, p) for p in X.columns]
    X_renamed = X.copy()
    X_renamed.columns = gene_names

    if strategy == "mean":
        return X_renamed.groupby(level=0, axis=1).mean()

    elif strategy == "first":
        seen = set()
        keep_cols = []
        for col in X_renamed.columns:
            if col not in seen:
                seen.add(col)
                keep_cols.append(col)
        return X_renamed[keep_cols]

    else:  # max_mean (default)
        # For each gene, keep the probe (column index) with the highest mean.
        # Iterate by integer position so duplicate gene names don't cause
        # col_means[name] to return a Series instead of a scalar.
        col_means = X_renamed.mean(axis=0)

        best_idx: Dict[str, int] = {}  # gene_name -> column index
        for idx, gene in enumerate(X_renamed.columns):
            mean_val = float(col_means.iloc[idx])
            if gene not in best_idx or mean_val > float(col_means.iloc[best_idx[gene]]):
                best_idx[gene] = idx

        return X_renamed.iloc[:, list(best_idx.values())]


# =============================================================================
# High-level annotate function
# =============================================================================


def annotate_features(
    feature_names: List[str],
    platform: Optional[str] = None,
    annotation_path: Optional[str | Path] = None,
    sdrf_path: Optional[str | Path] = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    probe_col: str = "ID",
    gene_col: str = "Gene Symbol",
    force_download: bool = False,
    species: str = "human",
) -> Dict[str, str]:
    """
    Build a probe -> gene symbol mapping for a list of feature names.

    Resolution order
    ----------------
    1. Local annotation file (if annotation_path is provided).
    2. MyGene.info API  — default primary source, no large file download.
    3. GPL SOFT file    — only if platform is explicitly provided
                          (accepts GPL ID, chip name, or ADF accession).

    MyGene.info is the default because it requires no file download and
    covers all major Affymetrix, Illumina, and Agilent probe ID spaces.
    Pass platform='GPL570' (or an ADF accession) to opt into the NCBI
    SOFT file route instead.

    Parameters
    ----------
    feature_names : list
    platform : str, optional
        Platform name, GPL ID, or ADF accession
        (e.g. 'HG-U133_Plus_2', 'GPL570', 'A-AFFY-44').
        When provided, skips MyGene.info and uses GPL SOFT instead.
    annotation_path : path, optional
        Local annotation CSV/TSV file (highest priority if provided).
    sdrf_path : path, optional
        SDRF file — used only to log the detected platform; does NOT
        trigger GPL download when platform is None.
    cache_dir : path
        Cache directory for downloaded GPL files.
    probe_col : str
        Column name for probe IDs in local annotation file.
    gene_col : str
        Column name for gene symbols in local annotation file.
    force_download : bool
        Force re-download of GPL file (only relevant when platform is set).
    species : str
        Species for MyGene.info queries (default 'human').

    Returns
    -------
    dict  probe_id -> gene_symbol
    """

    mapping: Dict[str, str] = {}

    # 1. Local annotation file (always highest priority)
    if annotation_path is not None:
        annotation_path = Path(annotation_path)
        if annotation_path.exists():
            sep = "\t" if annotation_path.suffix in (".tsv", ".txt") else ","
            df = pd.read_csv(
                annotation_path, sep=sep, comment="#", dtype=str, low_memory=False
            )
            mapping = extract_probe_gene_map(df, probe_col, gene_col)
            print(
                f"  Annotation loaded from file: {len(mapping)} probe->gene entries."
            )
            return mapping
        else:
            warnings.warn(f"Annotation file not found: {annotation_path}", stacklevel=2)

    # Log SDRF-detected platform (informational only — does not trigger GPL download)
    if sdrf_path is not None and platform is None:
        adf_accession = detect_platform_from_sdrf(sdrf_path)
        if adf_accession:
            print(
                f"  SDRF platform: {adf_accession} "
                f"(pass platform='{adf_accession}' to use GPL SOFT instead of MyGene.info)"
            )

    # 2. GPL SOFT — only when caller explicitly requests it via platform=
    if platform is not None:
        try:
            soft_path = download_gpl(platform, cache_dir, force_download)
            platform_df = parse_gpl_soft(soft_path)
            mapping = extract_probe_gene_map(platform_df, probe_col, gene_col)
            print(
                f"  GPL annotation: {len(mapping)} probe->gene entries "
                f"(platform={platform})."
            )
            return mapping
        except Exception as e:
            warnings.warn(
                f"GPL download/parse failed: {e}. Falling back to MyGene.info.",
                stacklevel=2,
            )

    # 3. MyGene.info — default primary source
    try:
        mapping = _mygene_lookup(feature_names, species=species)
    except Exception as e:
        warnings.warn(
            f"MyGene.info lookup failed: {e}. "
            "Feature names will remain as probe IDs.",
            stacklevel=2,
        )

    return mapping


def _mygene_lookup(
    probe_ids: List[str],
    species: str = "human",
    batch_size: int = 1000,
) -> Dict[str, str]:
    """
    Query MyGene.info for probe -> gene symbol mapping.

    Sends probe IDs in batches to avoid timeouts on large feature sets.
    Uses 'reporter' scope which covers Affymetrix, Illumina, and other
    microarray probe IDs directly.
    """
    try:
        import mygene
    except ImportError:
        raise ImportError(
            "Install mygene to use MyGene.info annotation:  pip install mygene"
        )

    mg = mygene.MyGeneInfo()
    mapping: Dict[str, str] = {}
    total = len(probe_ids)

    for start in range(0, total, batch_size):
        batch = probe_ids[start : start + batch_size]
        end = min(start + batch_size, total)
        print(f"  MyGene.info: querying probes {start + 1}-{end} / {total}...")

        results = mg.querymany(
            batch,
            scopes="reporter,symbol,accession",
            fields="symbol",
            species=species,
            verbose=False,
        )
        for res in results:
            if res.get("notfound"):
                continue
            qid = res.get("query")
            sym = res.get("symbol")
            if qid and sym:
                mapping[qid] = sym

    print(f"  MyGene.info: mapped {len(mapping)}/{total} probes.")
    return mapping


# =============================================================================
# Apply mapping to a feature list
# =============================================================================


def map_features(
    feature_names: List[str],
    mapping: Dict[str, str],
) -> List[str]:
    """
    Replace probe IDs with gene symbols where available.
    Unmapped probes retain their original ID.
    """

    mapped = []
    unmapped = 0
    for f in feature_names:
        gene = mapping.get(f, "")
        if gene:
            mapped.append(gene)
        else:
            mapped.append(f)
            unmapped += 1

    if unmapped > 0:
        warnings.warn(
            f"{unmapped}/{len(feature_names)} features could not be mapped "
            "to gene symbols. Original probe IDs retained.",
            stacklevel=2,
        )

    return mapped
