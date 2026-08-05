"""
src/report.py

Shared presentation layer for console and notebook output, built on `rich`.

All user-facing console output in the pipeline routes through this module so the
terminal and Jupyter look consistent and clean. `rich.console.Console`
auto-detects Jupyter and renders HTML there; in a terminal it renders ANSI.

Typical use
-----------
    from report import section, subsection, ok, warn, err, step, info
    from report import dataframe_table, output_tree, progress

    section("Benchmark — Feature Selection & Model Comparison")
    subsection("Dataset: GEO-Breast-20711")
    step("Training RF on mrmr_mid")
    ok("rf complete")
    dataframe_table(summary_df, title="CV summary")

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Force UTF-8 output on Windows so Unicode glyphs in rich don't crash.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover - very old interpreters
        pass

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# Shared console. force_jupyter is left to auto-detection.
console = Console()

__all__ = [
    "console",
    "section",
    "subsection",
    "ok",
    "warn",
    "err",
    "step",
    "info",
    "dataframe_table",
    "output_tree",
    "progress",
]


# =============================================================================
# Headers
# =============================================================================


def section(title: str, subtitle: Optional[str] = None) -> None:
    """Render a prominent section header (replaces ``print('=' * 70)`` blocks)."""
    body = Text(title, style="bold cyan")
    if subtitle:
        body.append("\n")
        body.append(subtitle, style="dim")
    console.print(Panel(body, border_style="cyan", expand=True))


def subsection(title: str) -> None:
    """Render a lighter divider with a title (replaces ``'─' * 70`` lines)."""
    console.print(Rule(Text(title, style="bold"), style="cyan"))


# =============================================================================
# Status lines
# =============================================================================


def ok(msg: str) -> None:
    console.print(Text("  ✓ ", style="bold green") + Text(msg))


def warn(msg: str) -> None:
    console.print(Text("  ⚠ ", style="bold yellow") + Text(msg, style="yellow"))


def err(msg: str) -> None:
    console.print(Text("  ✗ ", style="bold red") + Text(msg, style="red"))


def step(msg: str) -> None:
    """A neutral progress line (e.g. 'Training RF...')."""
    console.print(Text("  → ", style="bold blue") + Text(msg))


def info(msg: str) -> None:
    """A dim, secondary line (shapes, paths, counts)."""
    console.print(Text("    " + msg, style="dim"))


# =============================================================================
# DataFrame rendering
# =============================================================================


def dataframe_table(
    df: pd.DataFrame,
    title: Optional[str] = None,
    index: bool = True,
    max_rows: Optional[int] = None,
    float_format: str = "{:.4f}",
) -> None:
    """
    Render a pandas DataFrame as a clean ``rich`` table.

    Parameters
    ----------
    df : pd.DataFrame
    title : str, optional        Table caption shown above the grid.
    index : bool                 Show the DataFrame index as the first column.
    max_rows : int, optional     Truncate to the first ``max_rows`` rows (a note
                                 is appended when rows are hidden).
    float_format : str           Format string applied to float cells.
    """
    if df is None or len(df) == 0:
        warn(f"{title or 'Table'}: no rows to display")
        return

    shown = df if max_rows is None else df.head(max_rows)

    table = Table(title=title, header_style="bold magenta", border_style="dim")

    if index:
        table.add_column(str(df.index.name or ""), style="cyan", no_wrap=True)
    for col in shown.columns:
        table.add_column(str(col), justify="right")

    def fmt(v: object) -> str:
        if isinstance(v, float):
            return float_format.format(v)
        return str(v)

    for idx, row in shown.iterrows():
        cells = [fmt(v) for v in row.tolist()]
        if index:
            cells = [str(idx)] + cells
        table.add_row(*cells)

    console.print(table)
    if max_rows is not None and len(df) > max_rows:
        info(f"… {len(df) - max_rows} more row(s) not shown")


# =============================================================================
# Output file tree
# =============================================================================


def output_tree(root: str | Path, title: Optional[str] = None) -> None:
    """
    Render the files under ``root`` as a ``rich`` tree (replaces flat rglob dumps).
    """
    root = Path(root)
    if not root.exists():
        warn(f"Path does not exist: {root}")
        return

    tree = Tree(Text(title or str(root), style="bold cyan"))

    def add(node: Tree, directory: Path) -> None:
        entries = sorted(
            directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
        )
        for entry in entries:
            if entry.is_dir():
                branch = node.add(Text(f"{entry.name}/", style="bold blue"))
                add(branch, entry)
            else:
                size = entry.stat().st_size
                label = Text(entry.name)
                label.append(f"  ({_human_size(size)})", style="dim")
                node.add(label)

    add(tree, root)
    console.print(tree)


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# =============================================================================
# Progress
# =============================================================================


def progress() -> Progress:
    """
    Return a preconfigured ``rich.progress.Progress`` for fan-out loops.

    Usage
    -----
        with progress() as p:
            task = p.add_task("Datasets", total=len(datasets))
            for d in datasets:
                ...
                p.advance(task)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    )


# =============================================================================
# Markdown provenance helpers
#
# Shared by _append_benchmark_provenance (src/pipeline/benchmark.py) and
# _append_holdout_provenance (src/pipeline/holdout.py), which both write a
# fs_method × model comparison section into a DATASET_PROVENANCE*.md file.
# =============================================================================


def fmt_metric(x: float) -> str:
    """NaN-safe metric formatter for markdown tables."""
    return f"{x:.4f}" if x == x else "—"


def markdown_metric_table(
    detail: List[Tuple[str, str, Dict[str, float]]],
    pivot: Dict[str, Dict[str, float]],
    metric_cols: List[str],
    metric: str,
    detail_title: str = "Each fs_method × model",
) -> List[str]:
    """
    Build markdown lines for a (fs_method × model) detail table, sorted by
    ``metric`` descending, followed by a fs_method × model pivot on that
    same metric. Returns a list of lines (no leading/trailing section
    header) — caller wraps with whatever surrounding context it needs.
    """
    if not detail:
        return ["_No successful (fs × model) combinations._"]

    ordered = sorted(
        detail,
        key=lambda r: (
            r[2].get(metric, float("-inf"))
            if r[2].get(metric) == r[2].get(metric) else float("-inf")
        ),
        reverse=True,
    )

    lines: List[str] = [f"### {detail_title}"]
    lines.append("| fs_method | model | " + " | ".join(metric_cols) + " |")
    lines.append("|:---|:---|" + "".join([":---:|" for _ in metric_cols]))
    for fs_method, model_name, vals in ordered:
        lines.append(
            f"| {fs_method} | {model_name} | "
            + " | ".join(fmt_metric(vals.get(c, float("nan"))) for c in metric_cols) + " |"
        )
    lines.append("")

    models_all = sorted({m for d in pivot.values() for m in d})
    lines.append(f"### fs_method × model pivot (`{metric}`)")
    lines.append("| fs_method \\ model | " + " | ".join(models_all) + " |")
    lines.append("|:---|" + "".join([":---:|" for _ in models_all]))
    for fs_method in sorted(pivot):
        lines.append(
            f"| {fs_method} | "
            + " | ".join(fmt_metric(pivot[fs_method].get(m, float("nan"))) for m in models_all)
            + " |"
        )
    lines.append("")
    return lines


def upsert_markdown_block(
    path: Path,
    block: str,
    start_marker: str,
    end_marker: str,
) -> None:
    """
    Idempotently write ``block`` into the markdown file at ``path``, between
    ``start_marker``/``end_marker``. If the file already has a marked block
    (e.g. from a previous run), it's replaced in place and any content
    before/after the block is preserved; otherwise the block is appended.
    """
    existing = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    if start_marker in existing and end_marker in existing:
        pre = existing.split(start_marker)[0].rstrip()
        post = existing.split(end_marker, 1)[1].lstrip()
        new_text = (pre + "\n\n" + block + ("\n\n" + post if post else "\n")).rstrip() + "\n"
    else:
        new_text = (existing.rstrip() + "\n\n" + block + "\n") if existing else block + "\n"
    path.write_text(new_text, encoding="utf-8")
