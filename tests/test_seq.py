"""Tests for the sliding-window helpers and the historical reshape bug guard."""

from __future__ import annotations

import numpy as np
import pytest

from volnn.utils.seq import make_windows, reshape_windows


def test_make_windows_shape():
    arr = np.arange(100 * 4, dtype=float).reshape(100, 4)
    windows = make_windows(arr, past=10, n_features=4)
    assert windows.shape == (90, 10, 4)


def test_make_windows_rejects_wrong_feature_count():
    arr = np.zeros((100, 4))
    with pytest.raises(ValueError):
        make_windows(arr, past=10, n_features=7)


def test_reshape_windows_derives_width_from_features():
    flat = np.zeros((5, 40))
    out = reshape_windows(flat, n_features=4)
    assert out.shape == (5, 10, 4)


def test_reshape_windows_rejects_non_divisible():
    flat = np.zeros((5, 41))
    with pytest.raises(ValueError):
        reshape_windows(flat, n_features=4)
