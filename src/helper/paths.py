"""
src/helper/paths.py

Small path helpers shared across config_loader.py, pipeline/baseline_split.py,
pipeline/holdout.py, and api/inference.py — previously each re-implemented
these inline.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

from pathlib import Path
from typing import Union


def resolve_path(project_root: Union[str, Path], value: Union[str, Path]) -> str:
    """Resolve ``value`` against ``project_root`` if it isn't already
    absolute; return it unchanged (as a string) otherwise."""
    path = Path(value)
    if not path.is_absolute():
        path = Path(project_root) / path
    return str(path)


def safe_filename(sample_id: str) -> str:
    """Replace path separators in a sample id so it's safe to use as a
    filename (e.g. ``"GSM123/foo"`` -> ``"GSM123_foo"``)."""
    return sample_id.replace("/", "_").replace("\\", "_")
