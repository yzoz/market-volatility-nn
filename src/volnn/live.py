"""Streaming (online) inference from InfluxDB.

Optional feature: secrets are read from the environment (see ``.env.example``).
No credentials are ever hard-coded.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from volnn.config import Config, LiveConfig
from volnn.utils.scaling import scale_series


def _client(cfg: LiveConfig):
    from influxdb_client import InfluxDBClient

    return InfluxDBClient(url=cfg.influx_url, token=cfg.influx_token, org=cfg.influx_org)


def query_series(cfg: LiveConfig, field: str, period: str, aggregate: str, fn: str) -> list[float]:
    """Query an aggregated window of a single field from InfluxDB."""
    client = _client(cfg)
    query = f'''from(bucket:"{cfg.influx_bucket}")
        |> range(start: -{period})
        |> filter(fn: (r) => r["_measurement"] == "{cfg.measurement}")
        |> filter(fn: (r) => r["_field"] == "{field}")
        |> filter(fn: (r) => r["unit"] == "USDT")
        |> aggregateWindow(every: {aggregate}, fn: {fn}, createEmpty: false)
        |> yield(name: "{fn}")'''
    tables = client.query_api().query(query)
    return [record.get_value() for table in tables for record in table.records]


def build_live_input(cfg: LiveConfig, vol_window: int = 4) -> np.ndarray:
    """Assemble a single ``(1, 24, 4)`` input tensor from live Influx queries."""
    price = query_series(cfg, "price", "27h", "1h", "last")
    returns = np.log(pd.Series(price) / pd.Series(price).shift(1))
    hv = (returns.rolling(window=vol_window).std() * np.sqrt(365)).iloc[vol_window:].to_numpy()
    vol = np.asarray(query_series(cfg, "vol", "23h", "1h", "sum"))
    ac = np.asarray(query_series(cfg, "ac", "23h", "1h", "sum"))
    count = np.asarray(query_series(cfg, "count", "23h", "1h", "sum"))
    frame = np.column_stack(
        [scale_series(hv), scale_series((vol + ac) / 2), scale_series(vol), scale_series(count)]
    )
    return frame.reshape(1, frame.shape[0], frame.shape[1])


def run(cfg: Config, model_path: str | Path, iterations: int | None = None) -> None:
    """Poll InfluxDB, run inference and print predictions.

    ``iterations=None`` streams forever (until interrupted); a finite value is
    useful for smoke testing.
    """
    from tensorflow import keras

    live = LiveConfig.from_env()
    model = keras.models.load_model(str(model_path))
    count = 0
    while iterations is None or count < iterations:
        try:
            x = build_live_input(live)
            print(model.predict(x)[0])
        except Exception as exc:  # keep the loop alive on transient errors
            print(f"live inference error: {exc}")
        count += 1
        if iterations is None or count < iterations:
            time.sleep(live.poll_seconds)
