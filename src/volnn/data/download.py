"""Download historical klines from Binance.

Requires the optional ``binance-historical-data`` dependency:
``pip install market-volatility-nn[data]``.
"""

from __future__ import annotations

import datetime as dt

from volnn.config import DataConfig


def download_klines(cfg: DataConfig) -> None:
    """Download monthly klines for ``cfg.ticker`` into ``cfg.data_dir``.

    The operation is idempotent and updates existing files when needed. No
    network access is performed at import time.
    """
    try:
        from binance_historical_data import BinanceDataDumper
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            "binance-historical-data is not installed. Install the data extra:\n"
            "    pip install 'market-volatility-nn[data]'"
        ) from exc

    year, month, day = (int(part) for part in cfg.start_date.split("-"))
    dumper = BinanceDataDumper(
        path_dir_where_to_dump=str(cfg.data_dir),
        asset_class=cfg.asset_class,
        data_type=cfg.data_type,
        data_frequency=cfg.interval,
    )
    dumper.dump_data(
        tickers=[cfg.ticker],
        date_start=dt.date(year=year, month=month, day=day),
        date_end=None,
        is_to_update_existing=True,
        tickers_to_exclude=None,
    )
