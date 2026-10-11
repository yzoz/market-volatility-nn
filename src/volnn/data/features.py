"""Feature engineering: realised volatility and the unified 4-feature frame.

Feature schema (order matters): ``[hv, ac, vol, count]`` where

* ``hv``    — realised volatility (rolling std of log returns, annualised);
* ``ac``    — taker buy base-asset volume (klines column 9);
* ``vol``   — base-asset volume (klines column 5);
* ``count`` — number of trades (klines column 8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volnn.config import FeatureConfig

# Column positions in a raw Binance klines CSV (no header).
COL_CLOSE = 4
COL_VOLUME = 5
COL_TRADES = 8
COL_TAKER_BUY = 9

FEATURE_NAMES = ("hv", "ac", "vol", "count")


def compute_realised_volatility(
    prices: pd.Series | np.ndarray,
    window: int = 4,
    annualisation: float = 365.0,
) -> np.ndarray:
    """Compute annualised realised volatility from a price series.

    ``hv = std(log(p / p.shift(1))).rolling(window) * sqrt(annualisation)``

    Returns the series with the first ``window`` entries dropped, matching the
    original legacy behaviour.
    """
    series = pd.Series(np.asarray(prices, dtype=float))
    returns = np.log(series / series.shift(1))
    hv = returns.rolling(window=window).std() * np.sqrt(annualisation)
    return hv.iloc[window:].to_numpy()


def build_feature_frame(df: pd.DataFrame, cfg: FeatureConfig) -> np.ndarray:
    """Build the aligned ``(timesteps, n_features)`` feature matrix.

    All four series are aligned to the same length after the volatility
    window is dropped. The trailing dimension is validated against
    ``cfg.n_features``.
    """
    close = df.iloc[:, COL_CLOSE]
    hv = compute_realised_volatility(close, cfg.vol_window, cfg.annualisation)
    offset = cfg.vol_window
    ac = df.iloc[offset:, COL_TAKER_BUY].to_numpy(dtype=float)
    vol = df.iloc[offset:, COL_VOLUME].to_numpy(dtype=float)
    count = df.iloc[offset:, COL_TRADES].to_numpy(dtype=float)

    frame = np.column_stack([hv, ac, vol, count])
    if frame.shape[1] != cfg.n_features:
        raise ValueError(
            f"built {frame.shape[1]} features but FeatureConfig.n_features={cfg.n_features}"
        )
    return frame


def buys(ac: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Combine taker-buy volume and total volume into a single buy-pressure proxy."""
    return (np.asarray(ac) + np.asarray(vol)) / 2.0
