"""
src/mapping.py

Probe-level -> gene-level collapse for microarray expression data.
Probe -> Gene Symbol resolution itself lives in src/dataset_builder/annotation.py
(local GPL annotation file + MyGene.info fallback); this module only handles
merging probes that resolve to the same gene.

Author: Thesis Framework
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# Probe-level collapse to gene-level
# =============================================================================


def collapse_probe_duplicates(
    X_df: pd.DataFrame,
    mapping_dict: Dict[str, str],
    strategy: str = "mean",
    output_dir: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collapse probe-level expression DataFrame to gene-level.

    Multiple probes mapping to the same Gene Symbol are merged to eliminate
    collinearity before downstream SHAP / FP-Growth analysis.

    Parameters
    ----------
    X_df         : pd.DataFrame (n_samples, n_probes).  Columns = probe IDs.
    mapping_dict : Dict[str, str]  {probe_id -> gene_symbol}.
                   Probes absent from the dict retain their original ID.
    strategy     : 'mean'    — per-sample mean across all co-mapped probes.
                   'max_var' — keep only the probe with the highest variance.
    output_dir   : If given, writes probe_to_gene_metadata.csv here.

    Returns
    -------
    collapsed_df  : pd.DataFrame (n_samples, n_unique_genes)
    provenance_df : pd.DataFrame [Probe_ID, Gene_Symbol, Status]
                    Status ∈ {Kept, Collapsed_Mean, Unmapped} for strategy='mean'
                    (max_var additionally uses Kept_MaxVar / Dropped_MaxVar).
                    'Kept'      — single probe that mapped to a gene symbol.
                    'Unmapped'  — probe had no gene symbol; kept under its probe ID.
                    'Collapsed_Mean' — one of several probes merged into a gene.
    """
    probe_ids: List[str] = list(X_df.columns)

    # Which probes actually resolved to a (non-empty) gene symbol?
    mapped: Dict[str, bool] = {
        p: bool(str(mapping_dict.get(p, "")).strip()) for p in probe_ids
    }

    # Map every probe to its gene (fallback: keep probe ID as-is)
    probe_to_gene: Dict[str, str] = {
        p: (str(mapping_dict.get(p, "")).strip() or p) for p in probe_ids
    }

    # Group probes by resolved gene symbol
    gene_to_probes: Dict[str, List[str]] = {}
    for probe, gene in probe_to_gene.items():
        gene_to_probes.setdefault(gene, []).append(probe)

    collapsed_cols: Dict[str, pd.Series] = {}
    provenance_rows: List[Dict[str, str]] = []

    for gene, probes in gene_to_probes.items():
        if len(probes) == 1:
            collapsed_cols[gene] = X_df[probes[0]]
            provenance_rows.append(
                {
                    "Probe_ID": probes[0],
                    "Gene_Symbol": gene,
                    "Status": "Kept" if mapped[probes[0]] else "Unmapped",
                }
            )
        elif strategy == "max_var":
            variances = X_df[probes].var(axis=0)
            best = str(variances.idxmax())
            collapsed_cols[gene] = X_df[best]
            for p in probes:
                provenance_rows.append({
                    "Probe_ID": p,
                    "Gene_Symbol": gene,
                    "Status": "Kept_MaxVar" if p == best else "Dropped_MaxVar",
                })
        else:  # mean (default)
            collapsed_cols[gene] = X_df[probes].mean(axis=1)
            for p in probes:
                provenance_rows.append(
                    {"Probe_ID": p, "Gene_Symbol": gene, "Status": "Collapsed_Mean"}
                )

    collapsed_df = pd.DataFrame(collapsed_cols, index=X_df.index)
    provenance_df = pd.DataFrame(provenance_rows)

    n_collapsed = int((provenance_df["Status"] == "Collapsed_Mean").sum())
    n_unmapped  = int((provenance_df["Status"] == "Unmapped").sum())
    parts = []
    if n_collapsed:
        parts.append(f"{n_collapsed} probes collapsed to genes")
    if n_unmapped:
        parts.append(f"{n_unmapped} unmapped (kept as probe IDs)")
    detail = ", ".join(parts) if parts else "all probes mapped 1-to-1"
    print(
        f"  Probe collapse: {len(probe_ids)} probes -> "
        f"{len(collapsed_df.columns)} unique genes ({detail})"
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        prov_path = out / "probe_to_gene_metadata.csv"
        provenance_df.to_csv(prov_path, index=False)
        print(f"  Provenance log -> {prov_path}")

    return collapsed_df, provenance_df
