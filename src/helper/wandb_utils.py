"""
src/helper/wandb_utils.py

Shared W&B logging helpers used by both BaselineSplitMixin (baseline_split.py)
and HoldoutMixin (holdout.py) — kept here so a fix to per-class logging only
has to happen once.

Framework: Classification Transcriptomic with XAI
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List

_REQUIRED_ATTRS = ("init", "login", "data_types", "Table")


def _wandb_module_is_usable(wandb_module: Any) -> bool:
    """A module left behind by an `import wandb` interrupted mid-init (e.g.
    a Jupyter cell that got a KeyboardInterrupt while wandb was loading)
    stays in ``sys.modules`` and imports again without raising, but is
    missing attributes wandb's own code expects to already be set — calling
    into it later fails with errors like "partially initialized module
    'wandb' has no attribute 'data_types'". Checking for a few of those
    attributes up front lets callers detect and heal this instead of
    treating "the import statement didn't raise" as "wandb works"."""
    return all(hasattr(wandb_module, attr) for attr in _REQUIRED_ATTRS)


def import_wandb():
    """Import wandb, self-healing a half-initialized module left behind by
    an earlier interrupted import in the same process/kernel (see
    ``_wandb_module_is_usable``) instead of silently staying broken — and
    disabling W&B logging with no further warning — for the rest of that
    session. Raises ``ImportError`` if wandb genuinely isn't installed."""
    import wandb

    if not _wandb_module_is_usable(wandb):
        for name in list(sys.modules):
            if name == "wandb" or name.startswith("wandb."):
                del sys.modules[name]
        import wandb  # fresh import, no leftover partial state to inherit

    return wandb


def wandb_log_per_class(run: Any, per_class: Dict[str, Any], class_labels: List[str]) -> None:
    """Log per-class precision/recall/f1 (test split) as BOTH a
    wandb.Table (a readable snapshot table in the run's Tables panel) and
    flat scalars ``test_per_class_{metric}/{class}`` (so a single class's
    f1 can be charted/filtered/compared across runs, same as any other
    scalar metric)."""
    import wandb

    test_per_class = (per_class or {}).get("test", {})
    rows = []
    flat: Dict[str, float] = {}
    for cls in class_labels:
        m = test_per_class.get(cls, {})
        p, r, f = m.get("precision"), m.get("recall"), m.get("f1")
        rows.append([cls, p, r, f])
        for metric_name, val in (("precision", p), ("recall", r), ("f1", f)):
            if val is not None:
                flat[f"test_per_class_{metric_name}/{cls}"] = val
    if rows:
        table = wandb.Table(columns=["class", "precision", "recall", "f1"], data=rows)
        run.log({"per_class_metrics": table})
    if flat:
        run.log(flat)
