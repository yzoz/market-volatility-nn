"""Smoke test for the synthetic data generator (schema only)."""

from __future__ import annotations

import numpy as np

from volnn.data.synthetic import N_COLUMNS, generate_klines


def test_generate_klines_schema():
    data = generate_klines(n_steps=100, seed=1)
    assert data.shape == (100, N_COLUMNS)


def test_generate_klines_deterministic():
    a = generate_klines(n_steps=50, seed=3)
    b = generate_klines(n_steps=50, seed=3)
    assert np.allclose(a, b)


def test_generate_klines_prices_positive():
    data = generate_klines(n_steps=200, seed=5)
    assert (data[:, 4] > 0).all()  # close prices
