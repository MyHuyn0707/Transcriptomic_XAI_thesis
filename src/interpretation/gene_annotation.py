"""
src/interpretation/gene_annotation.py

Enrich the genes that appear in extracted rules with biological annotation
pulled from the dataset's platform file (Platform_Annotations/…).

Two platform formats are supported, both keyed by probe ``ID``:

- NCBI ``.annot`` (Platform_Annotations/NBCI_platform/GPL*.annot.gz)
    A SOFT block between ``!platform_table_begin`` / ``!platform_table_end`` with
    columns: Gene title, Gene symbol, Gene ID, GenBank Accession, Chromosome
    location, GO:Function/Process/Component (+ ID) …
- CuMiDa ``_limpo`` (Platform_Annotations/Cumida_platform/GPL*_limpo.txt.gz)
    A plain TSV with columns: Gene Title, Gene Symbol, ENTREZ_GENE_ID, GB_ACC,
    RefSeq Transcript ID, Gene Ontology Biological Process / Cellular Component
    / Molecular Function.

Both are normalised into one schema (gene_symbol, gene_title, entrez_id,
genbank_acc, chromosome, go_biological_process, go_cellular_component,
go_molecular_function). The per-gene description joins that schema to each gene
in a ``genes_in_rules.csv`` via the gene's backing probe IDs.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from src.helper import report


# =============================================================================
# Normalised schema
# =============================================================================

# Superset of biological columns carried into the description, in display order.
# A column a given platform format does not provide is left blank (per user:
# "phần nào ở cumida không có thì bỏ trống") — the goal is to capture EVERY
# biologically useful field so an LLM has maximal context.
BIO_COLUMNS: List[str] = [
    "gene_symbol",
    "gene_title",
    "entrez_id",
    "genbank_acc",
    "refseq",                 # CuMiDa only
    "unigene_symbol",         # NBCI only
    "unigene_id",             # NBCI only
    "gi",                     # NBCI only (GenBank identifier)
    "nucleotide_title",       # NBCI only
    "target_description",     # CuMiDa only
    "representative_public_id",  # CuMiDa only
    "species",                # CuMiDa only
    "chromosome",             # NBCI only (Chromosome location)
    "chromosome_annotation",  # NBCI only
    "go_biological_process",
    "go_cellular_component",
    "go_molecular_function",
    "go_terms",                # CuMiDa "raw GEO" only (GPL6480/GPL6848 _limpo — BP/CC/MF not separated)
]

# Multi-value cells (``///``-separated) collapsed to a "; " list.
_MULTI_COLS = {
    "gene_symbol", "gene_title", "entrez_id", "genbank_acc", "refseq",
    "unigene_symbol", "unigene_id", "gi",
}
# GO cells parsed to term lists (format-specific).
_GO_COLS = {"go_biological_process", "go_cellular_component", "go_molecular_function", "go_terms"}

# Source-column → normalised-column maps for each platform format.
_NBCI_MAP = {
    "Gene symbol": "gene_symbol",
    "Gene title": "gene_title",
    "Gene ID": "entrez_id",
    "GenBank Accession": "genbank_acc",
    "UniGene symbol": "unigene_symbol",
    "UniGene ID": "unigene_id",
    "GI": "gi",
    "Nucleotide Title": "nucleotide_title",
    "Chromosome location": "chromosome",
    "Chromosome annotation": "chromosome_annotation",
    "GO:Process": "go_biological_process",
    "GO:Component": "go_cellular_component",
    "GO:Function": "go_molecular_function",
}
_CUMIDA_MAP = {
    "Gene Symbol": "gene_symbol",
    "Gene Title": "gene_title",
    "ENTREZ_GENE_ID": "entrez_id",
    "GB_ACC": "genbank_acc",
    "RefSeq Transcript ID": "refseq",
    "Target Description": "target_description",
    "Representative Public ID": "representative_public_id",
    "Species Scientific Name": "species",
    "Gene Ontology Biological Process": "go_biological_process",
    "Gene Ontology Cellular Component": "go_cellular_component",
    "Gene Ontology Molecular Function": "go_molecular_function",
}
# A second CuMiDa _limpo variant (GPL6480/GPL6848 — Agilent) ships GEO's raw
# platform-table column names instead of the curated ones _CUMIDA_MAP expects
# (e.g. "GENE_SYMBOL" not "Gene Symbol"), so none of the above ever matched
# and every biological field silently came back blank even for well-matched
# probes. GO terms aren't split into BP/CC/MF here — they land in the single
# combined "go_terms" column instead of being mis-labelled into one category.
_CUMIDA_RAW_MAP = {
    "GENE_SYMBOL": "gene_symbol",
    "GENE_NAME": "gene_title",
    "GENE": "entrez_id",
    "GB_ACC": "genbank_acc",
    "REFSEQ": "refseq",
    "DESCRIPTION": "target_description",
    "CHROMOSOMAL_LOCATION": "chromosome",
    "CYTOBAND": "chromosome_annotation",
    "GO_ID": "go_terms",
}


# =============================================================================
# Platform annotation loading
# =============================================================================


_SPECIES_PREFIX_RE = None  # set below, after `re` import


def _extract_species(description: object) -> str:
    """Pull the leading ``"Genus species"`` off a GEO target description.

    The raw CuMiDa platform files (GPL6480/GPL6848) have no dedicated species
    column, but every ``DESCRIPTION`` value observed starts with it anyway
    (e.g. ``"Homo sapiens family with sequence similarity ... mRNA [NM_...]"``)
    — extracted here rather than left blank, since the information is present,
    just not in its own column.
    """
    global _SPECIES_PREFIX_RE
    if _SPECIES_PREFIX_RE is None:
        import re
        _SPECIES_PREFIX_RE = re.compile(r"^([A-Z][a-z]+ [a-z]+)\b")
    if not isinstance(description, str) or not description:
        return ""
    m = _SPECIES_PREFIX_RE.match(description.strip())
    return m.group(1) if m else ""


def _read_text(path: Path) -> str:
    """Read a plain or gzip-compressed text file as UTF-8 (errors replaced)."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _clean_multi(value: object) -> str:
    """Collapse a ``///``-separated multi-value cell into a ``; `` list."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    parts = [p.strip() for p in str(value).split("///")]
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return "; ".join(out)


def _clean_go(value: object, fmt: str) -> str:
    """Extract GO *term* text into a ``; `` list, dropping IDs/evidence codes.

    NBCI      : ``termA///termB`` → term is the whole ``///`` part.
    CuMiDa    : ``GOID // term // evidence /// …`` → term is the middle ``//`` field.
    CuMiDa raw: ``GO:0016020(membrane)|GO:0016021(integral to membrane)`` →
                term is the text inside the parentheses, entries ``|``-separated.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if fmt == "cumida_raw":
        terms = []
        for part in str(value).split("|"):
            part = part.strip()
            if "(" in part and part.endswith(")"):
                term = part[part.index("(") + 1:-1].strip()
            else:
                term = part
            if term:
                terms.append(term)
    else:
        terms = []
        for part in str(value).split("///"):
            part = part.strip()
            if not part:
                continue
            if fmt == "cumida" and "//" in part:
                fields = [f.strip() for f in part.split("//")]
                term = fields[1] if len(fields) > 1 else fields[0]
            else:
                term = part
            if term:
                terms.append(term)
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return "; ".join(out)


def load_platform_annotation(annotation_path: str | Path) -> pd.DataFrame:
    """Load a platform annotation file into a probe-indexed, normalised table.

    Parameters
    ----------
    annotation_path : str | Path
        A ``GPL*.annot(.gz)`` (NBCI) or ``GPL*_limpo.txt(.gz)`` (CuMiDa) file.

    Returns
    -------
    pd.DataFrame
        Indexed by probe ID, columns = :data:`BIO_COLUMNS` (missing source
        columns become empty strings). Empty DataFrame if the file is unusable.
    """
    annotation_path = Path(annotation_path)
    if not annotation_path.exists():
        report.warn(f"Platform annotation not found: {annotation_path}")
        return pd.DataFrame(columns=["ID"] + BIO_COLUMNS).set_index("ID")

    text = _read_text(annotation_path)
    lines = text.splitlines()

    # NBCI .annot wraps the table in a SOFT block; CuMiDa _limpo is a plain TSV.
    if any(ln.strip() == "!platform_table_begin" for ln in lines):
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "!platform_table_begin") + 1
        end = next(
            (i for i, ln in enumerate(lines) if ln.strip() == "!platform_table_end"),
            len(lines),
        )
        table_text = "\n".join(lines[start:end])
        fmt = "nbci"
    else:
        table_text = text
        fmt = "cumida"

    try:
        df = pd.read_csv(io.StringIO(table_text), sep="\t", dtype=str, low_memory=False)
    except Exception as e:  # noqa: BLE001
        report.warn(f"Could not parse platform annotation {annotation_path.name}: {e}")
        return pd.DataFrame(columns=["ID"] + BIO_COLUMNS).set_index("ID")

    if fmt == "cumida" and "GENE_SYMBOL" in df.columns and "Gene Symbol" not in df.columns:
        # GPL6480/GPL6848 (Agilent) _limpo files ship GEO's raw platform-table
        # column names, not the curated CuMiDa naming _CUMIDA_MAP expects.
        fmt = "cumida_raw"

    colmap = _NBCI_MAP if fmt == "nbci" else (_CUMIDA_RAW_MAP if fmt == "cumida_raw" else _CUMIDA_MAP)
    if "ID" not in df.columns:
        report.warn(f"{annotation_path.name}: no 'ID' column; cannot map probes.")
        return pd.DataFrame(columns=["ID"] + BIO_COLUMNS).set_index("ID")

    out = pd.DataFrame({"ID": df["ID"].astype(str)})
    for src, norm in colmap.items():
        if src not in df.columns:
            out[norm] = ""
            continue
        if norm in _GO_COLS:
            out[norm] = df[src].map(lambda v: _clean_go(v, fmt))
        elif norm in _MULTI_COLS:
            out[norm] = df[src].map(_clean_multi)
        else:
            out[norm] = df[src].fillna("").astype(str).str.strip()
    if fmt == "cumida_raw" and "DESCRIPTION" in df.columns:
        out["species"] = df["DESCRIPTION"].map(_extract_species)

    for norm in BIO_COLUMNS:  # ensure all present, ordered
        if norm not in out.columns:
            out[norm] = ""

    out = out.drop_duplicates(subset="ID").set_index("ID")
    report.ok(
        f"Loaded platform annotation: {len(out):,} probes "
        f"({fmt}) from {annotation_path.name}"
    )
    result = out[BIO_COLUMNS]
    # Self-describing metadata for the UI: whether GO terms are split into
    # BP/CC/MF (nbci, curated cumida) or only available combined (cumida_raw,
    # e.g. GPL6480/GPL6848) — read via build_gene_descriptions below so the
    # frontend can switch display mode per platform instead of guessing from
    # which fields happen to be empty.
    result.attrs["go_format"] = "combined" if fmt == "cumida_raw" else "split"
    return result


# =============================================================================
# Gene description builder
# =============================================================================


def _first_nonempty(series: pd.Series) -> str:
    for v in series:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _union_terms(series: pd.Series) -> str:
    seen, out = set(), []
    for v in series:
        if not isinstance(v, str):
            continue
        for t in v.split("; "):
            t = t.strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return "; ".join(out)


def _build_refseq_index(annotation_df: pd.DataFrame) -> Dict[str, str]:
    """``{RefSeq accession: probe ID}`` reverse index, first match wins.

    Some CuMiDa series (e.g. GSE26304/GSE41657, Agilent GPL6480/GPL6848) report
    their expression matrix columns as RefSeq accessions (``NM_000095``)
    instead of the platform's own probe ID (``A_23_P90436``) — those never
    match the annotation table's ``ID`` index directly, even though the
    accession IS present in the table's ``REFSEQ`` column under a different
    probe. This lets :func:`build_gene_descriptions` fall back to a
    RefSeq-keyed lookup for exactly that case.
    """
    if "refseq" not in annotation_df.columns:
        return {}
    index: Dict[str, str] = {}
    for probe_id, value in annotation_df["refseq"].items():
        if not isinstance(value, str) or not value:
            continue
        for acc in value.split("; "):
            acc = acc.strip()
            if acc and acc not in index:
                index[acc] = probe_id
    return index


def build_gene_descriptions(
    genes_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join biological annotation onto each gene in a ``genes_in_rules`` table.

    Parameters
    ----------
    genes_df : pd.DataFrame
        A ``genes_in_rules.csv`` (columns: gene, probes, n_rules, classes).
    annotation_df : pd.DataFrame
        Probe-indexed normalised annotation from :func:`load_platform_annotation`.

    Returns
    -------
    pd.DataFrame
        One row per gene: rule context (gene, n_rules, classes, probes) plus the
        biological columns, resolved by looking up the gene's backing probes.
        Gene-level fields take the first non-empty probe value; GO terms are the
        union across the gene's probes.
    """
    if genes_df is None or genes_df.empty:
        return pd.DataFrame()

    refseq_index = _build_refseq_index(annotation_df)
    go_format = annotation_df.attrs.get("go_format", "split")

    rows: List[Dict[str, object]] = []
    for _, g in genes_df.iterrows():
        probes = [p.strip() for p in str(g.get("probes", "")).split(";") if p.strip()]
        present = [p for p in probes if p in annotation_df.index]
        if not present:
            # Fall back to RefSeq-keyed lookup — the probe string itself IS
            # a RefSeq accession for these series, not the platform probe ID.
            mapped = [refseq_index[p] for p in probes if p in refseq_index]
            present = list(dict.fromkeys(mapped))
        sub = annotation_df.loc[present] if present else annotation_df.iloc[0:0]

        row: Dict[str, object] = {
            "gene": g.get("gene", ""),
            "n_rules": int(g.get("n_rules", 0)) if pd.notna(g.get("n_rules")) else 0,
            "classes": g.get("classes", ""),
            "probes": "; ".join(probes),
            "annotated": bool(present),
            "go_format": go_format,
        }
        for col in BIO_COLUMNS:
            if col == "gene_symbol":
                # Prefer the rule's own gene label; fall back to annotation.
                row[col] = str(g.get("gene", "")) or (
                    _first_nonempty(sub[col]) if not sub.empty else ""
                )
            elif col.startswith("go_"):
                row[col] = _union_terms(sub[col]) if not sub.empty else ""
            else:
                row[col] = _first_nonempty(sub[col]) if not sub.empty else ""
        rows.append(row)

    # gene (rule label) first, then every biological column, then rule context.
    # gene_symbol duplicates the rule's `gene` label, so it is dropped here.
    bio = [c for c in BIO_COLUMNS if c != "gene_symbol"]
    cols = ["gene"] + bio + ["n_rules", "classes", "probes", "annotated", "go_format"]
    df = pd.DataFrame(rows)
    return df[[c for c in cols if c in df.columns]]


# =============================================================================
# Save (CSV table + readable Markdown)
# =============================================================================


def _truncate_terms(value: str, max_terms: int) -> str:
    if not value:
        return "—"
    terms = value.split("; ")
    if len(terms) <= max_terms:
        return ", ".join(terms)
    return ", ".join(terms[:max_terms]) + f"  _(+{len(terms) - max_terms} more)_"


def save_gene_descriptions(
    desc_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    dataset_name: str = "",
    model_name: str = "",
    platform: str = "",
    max_go_terms: int = 8,
) -> Dict[str, Path]:
    """Write ``gene_description.csv`` (full table) and ``gene_description.md``.

    The CSV keeps every GO term; the Markdown caps GO lists to ``max_go_terms``
    per category for readability.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    csv_path = output_dir / "gene_description.csv"
    desc_df.to_csv(csv_path, index=False)
    paths["csv"] = csv_path

    lines: List[str] = [
        f"# Gene descriptions — {dataset_name} · {model_name}".rstrip(" ·"),
        "",
        f"> Genes appearing in the extracted rules, enriched from platform "
        f"`{platform or 'annotation'}`. {len(desc_df)} gene(s).",
        "",
    ]
    if desc_df.empty:
        lines.append("_No genes to describe._\n")
    else:
        n_missing = int((~desc_df["annotated"]).sum()) if "annotated" in desc_df else 0
        if n_missing:
            lines.append(f"> ⚠️ {n_missing} gene(s) had no probe match in the platform file.\n")
        # (key, label, is_go) in display order. A field is printed only when
        # non-empty, so a format that lacks it (e.g. chromosome on CuMiDa) is
        # simply omitted rather than showing a blank line.
        FIELDS = [
            ("gene_title", "Title", False),
            ("entrez_id", "Entrez Gene ID", False),
            ("genbank_acc", "GenBank", False),
            ("refseq", "RefSeq", False),
            ("unigene_symbol", "UniGene symbol", False),
            ("unigene_id", "UniGene ID", False),
            ("gi", "GenBank GI", False),
            ("nucleotide_title", "Nucleotide title", False),
            ("target_description", "Target description", False),
            ("representative_public_id", "Representative public ID", False),
            ("species", "Species", False),
            ("chromosome", "Chromosome location", False),
            ("chromosome_annotation", "Chromosome annotation", False),
            ("go_biological_process", "GO Biological Process", True),
            ("go_cellular_component", "GO Cellular Component", True),
            ("go_molecular_function", "GO Molecular Function", True),
            ("go_terms", "GO Terms", True),
        ]
        for _, r in desc_df.iterrows():
            lines.append(f"## {r['gene']}")
            lines.append("")
            for key, label, is_go in FIELDS:
                val = str(r.get(key, "") or "").strip()
                if not val:
                    continue
                shown = _truncate_terms(val, max_go_terms) if is_go else val
                lines.append(f"- **{label}:** {shown}")
            lines.append(
                f"- **Appears in:** {r.get('n_rules', 0)} rule(s) · "
                f"classes: {r.get('classes') or '—'} · probes: `{r.get('probes') or '—'}`"
            )
            lines.append("")

    md_path = output_dir / "gene_description.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    paths["md"] = md_path

    report.ok(f"Gene descriptions → {md_path.parent} (csv + md, {len(desc_df)} genes)")
    return paths


# =============================================================================
# Batch — describe every genes_in_rules.csv found under a rules root
# =============================================================================


def describe_gene_files(
    gene_files: Sequence[Path],
    annotation_df: pd.DataFrame,
    dataset_name: str,
    platform: str,
    label_fn: Callable[[Path], str] = lambda p: p.parent.name,
) -> Dict[str, Dict[str, Any]]:
    """
    Build + save ``gene_description.{csv,md}`` next to each ``genes_in_rules.csv``.

    Shared by the full-dataset rule extraction (``rules/{rf,dt}/genes_in_rules.csv``,
    ``label_fn`` defaults to the rf/dt folder name) and the holdout variant
    (``{fs_method}/{rf,dt}/rules/genes_in_rules.csv``, callers pass a
    ``label_fn`` that includes the fs_method).

    Returns ``{label: {"n_genes", "n_annotated", "paths"}}``.
    """
    result: Dict[str, Dict[str, Any]] = {}
    for gf in gene_files:
        label = label_fn(gf)
        try:
            genes_df = pd.read_csv(gf)
            desc = build_gene_descriptions(genes_df, annotation_df)
            paths = save_gene_descriptions(
                desc, gf.parent,
                dataset_name=dataset_name, model_name=label, platform=platform,
            )
            n_ann = int(desc["annotated"].sum()) if "annotated" in desc else 0
            result[label] = {
                "n_genes": int(len(desc)),
                "n_annotated": n_ann,
                "paths": {k: str(v) for k, v in paths.items()},
            }
        except Exception as e:
            report.warn(f"{dataset_name}/{label}: gene description failed: {e}")
    return result
