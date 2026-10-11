"""Tests for chronological splitting and baseline models."""

from __future__ import annotations

import numpy as np

from volnn.baselines import majority_baseline, persistence_baseline
from volnn.models.common import chronological_split


def test_chronological_split_preserves_order_and_sizes():
    x = np.arange(100).reshape(100, 1)
    y = np.arange(100)
    x_train, x_val, y_train, y_val = chronological_split(x, y, 0.8)
    assert x_train.shape[0] == 79
    assert x_val.shape[0] == 20
    assert x_train[0, 0] < x_val[0, 0]  # strictly time-ordered, not shuffled


def test_persistence_baseline_returns_last_hv():
    hv = np.arange(50, dtype=float)
    past, future = 5, 3
    preds = persistence_baseline(hv, past, future)
    assert preds.shape[0] == len(hv) - past - future
    assert preds[0] == hv[past - 1]


def test_majority_baseline_picks_most_common():
    y = np.array([1, 1, 1, 0, 0])
    out = majority_baseline(y)
    assert out.shape == y.shape
    assert set(out) == {1}
