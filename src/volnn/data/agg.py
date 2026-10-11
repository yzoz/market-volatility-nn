"""Merge per-month klines CSVs into a single chronologically sorted file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from volnn.config import DataConfig


def merge_monthly_csvs(cfg: DataConfig, dest: str | Path | None = None) -> Path:
    """Merge all monthly CSVs under ``data_dir`` into one sorted CSV.

    Files are discovered recursively and concatenated. The result is sorted by
    the open time (first column) so downstream code is order-independent —
    fixing the original implementation's reliance on directory-walk ordering.
    """
    root = Path(cfg.data_dir)
    files = sorted(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files found under {root}")

    frames = [pd.read_csv(f, header=None) for f in files]
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(by=merged.columns[0]).drop_duplicates(subset=[merged.columns[0]])

    out = Path(dest) if dest is not None else root / f"{cfg.ticker}-{cfg.interval}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False, header=False)
    return out
