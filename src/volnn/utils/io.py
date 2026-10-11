"""I/O helpers: array persistence, metrics records and artefact paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (recursively) and return it as a :class:`Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_array(path: str | Path, array: np.ndarray) -> Path:
    """Save a 2D array as a plain-text (``np.savetxt``) file, creating parents."""
    p = Path(path)
    ensure_dir(p.parent)
    np.savetxt(p, array)
    return p


def load_array(path: str | Path) -> np.ndarray:
    """Load a plain-text array previously written by :func:`save_array`."""
    return np.loadtxt(path)


def save_metrics(path: str | Path, metrics: dict[str, Any]) -> Path:
    """Persist a metrics/history mapping as pretty-printed JSON."""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(metrics, indent=2, default=_json_default))
    return p


def load_metrics(path: str | Path) -> dict[str, Any]:
    """Load a metrics JSON file."""
    return json.loads(Path(path).read_text())


def _json_default(value: Any) -> Any:
    """Best-effort coercion of numpy scalars/arrays for JSON serialisation."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)
