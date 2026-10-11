"""Tests for the CSV merge helper (uses a small on-disk fixture)."""

from __future__ import annotations

import numpy as np

from volnn.config import DataConfig
from volnn.data.agg import merge_monthly_csvs


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.array(rows, dtype=float), delimiter=",")


def test_merge_sorts_and_deduplicates(tmp_path):
    # Two monthly files whose first column (open time) is out of order / duplicated.
    _write(tmp_path / "monthly" / "a.csv", [[3, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    _write(tmp_path / "monthly" / "b.csv", [[2, 1, 1, 1, 1], [1, 9, 9, 9, 9]])

    cfg = DataConfig(ticker="T", interval="5m", data_dir=tmp_path)
    out = merge_monthly_csvs(cfg, dest=tmp_path / "T-5m.csv")

    merged = np.loadtxt(out, delimiter=",")
    times = merged[:, 0]
    assert list(times) == sorted(times)
    # Duplicate open time (1) collapsed to a single row.
    assert len(np.unique(times)) == len(times)
