"""Deterministic synthetic OHLCV generator for demos and tests.

Output mimics the raw Binance klines CSV schema (12 columns, no header) so the
whole pipeline can run without network access or real market data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Binance klines column count (open time, o,h,l,c,v, close time, quote vol,
# trades, taker base, taker quote, ignore).
N_COLUMNS = 12


def generate_klines(
    n_steps: int = 20000, seed: int = 42, start_price: float = 30000.0
) -> np.ndarray:
    """Generate a synthetic klines matrix of shape ``(n_steps, 12)``.

    The price follows a random walk with stochastic volatility so that realised
    volatility has meaningful structure for both regression and classification.
    """
    rng = np.random.default_rng(seed)
    timestamps = np.arange(n_steps, dtype=float) * 300_000  # 5-minute steps in ms
    timestamps += 1_577_836_800_000  # 2020-01-01 UTC-ish

    vol = 0.001 + 0.002 * np.abs(np.sin(np.arange(n_steps) / 300.0))
    shocks = rng.normal(0.0, 1.0, size=n_steps) * vol
    log_price = np.log(start_price) + np.cumsum(shocks)
    close = np.exp(log_price)
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.0005, size=n_steps)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.0005, size=n_steps)))

    base_vol = np.abs(rng.normal(50.0, 10.0, size=n_steps))
    quote_vol = base_vol * close
    trades = np.abs(rng.normal(500, 100, size=n_steps)).astype(float)
    taker_base = base_vol * rng.uniform(0.3, 0.7, size=n_steps)
    taker_quote = taker_base * close

    matrix = np.column_stack(
        [
            timestamps,
            open_,
            high,
            low,
            close,
            base_vol,
            timestamps,
            quote_vol,
            trades,
            taker_base,
            taker_quote,
            trades,
        ]
    )
    return matrix


def write_klines_csv(path: str, n_steps: int = 20000, seed: int = 42) -> str:
    """Write a synthetic klines CSV (no header), creating parents, and return its path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, generate_klines(n_steps=n_steps, seed=seed), delimiter=",")
    return str(out)
