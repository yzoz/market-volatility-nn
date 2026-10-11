"""Shared pytest fixtures — synthetic data only, no network, no training."""

from __future__ import annotations

import numpy as np
import pytest

from volnn.config import FeatureConfig
from volnn.data.features import build_feature_frame
from volnn.data.synthetic import generate_klines


@pytest.fixture
def feature_cfg() -> FeatureConfig:
    return FeatureConfig(ticker="TEST", interval="5m", past=16, future=4, vol_window=4)


@pytest.fixture
def klines() -> np.ndarray:
    return generate_klines(n_steps=2000, seed=7)


@pytest.fixture
def frame(klines, feature_cfg) -> np.ndarray:
    import pandas as pd

    return build_feature_frame(pd.DataFrame(klines), feature_cfg)
