"""Tests for realised-volatility and feature-frame construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volnn.config import FeatureConfig
from volnn.data.features import build_feature_frame, compute_realised_volatility


def test_hv_constant_prices_is_zero():
    prices = np.full(50, 100.0)
    hv = compute_realised_volatility(prices, window=4, annualisation=365.0)
    assert np.allclose(hv, 0.0)


def test_hv_length_and_positivity():
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=100)))
    hv = compute_realised_volatility(prices, window=4, annualisation=365.0)
    assert hv.shape[0] == 100 - 4
    assert np.nanmax(hv) > 0


def test_feature_frame_shape(frame, feature_cfg):
    assert frame.ndim == 2
    assert frame.shape[1] == feature_cfg.n_features


def test_feature_frame_rejects_wrong_feature_count(klines):
    bad = FeatureConfig(ticker="T", interval="5m", n_features=7)
    with pytest.raises(ValueError):
        build_feature_frame(pd.DataFrame(klines), bad)


def test_feature_frame_has_no_nans(frame):
    assert np.isfinite(frame).all()
