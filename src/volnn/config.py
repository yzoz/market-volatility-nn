"""Configuration objects for market-volatility-nn.

Precedence (lowest to highest): dataclass defaults -> YAML file -> environment
variables -> CLI flags.  All original hyperparameters are preserved as defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

try:  # optional dependency
    import yaml
except ImportError:  # pragma: no cover - yaml is a declared dependency
    yaml = None  # type: ignore[assignment]

try:  # optional dependency
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover

    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:  # type: ignore
        return False


DATA_DIR = Path("data")
ARTIFACTS_DIR = Path("artifacts")


@dataclass(frozen=True)
class DataConfig:
    """Where the raw market data comes from and how it is fetched."""

    ticker: str = "BTCUSDT"
    interval: str = "5m"
    asset_class: str = "spot"  # spot, um, cm
    data_type: str = "klines"  # aggTrades, klines, trades
    start_date: str = "2020-01-01"
    data_dir: Path = DATA_DIR


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering parameters shared by both model branches."""

    ticker: str = "BTCUSDT"
    interval: str = "5m"
    # Sliding window length fed to the network.
    past: int = 288
    # Forecast horizon (in steps).
    future: int = 24
    # Rolling window for realised volatility. Unified for both branches.
    vol_window: int = 4
    # Annualisation factor (crypto trades 365 days a year).
    annualisation: float = 365.0
    # Number of input features. Must match the feature builder output.
    n_features: int = 4
    # Direction threshold for classification labels, keyed by interval.
    threshold_1m: float = 0.0025
    threshold_5m: float = 0.5
    threshold_1h: float = 0.5
    # Train/validation split fraction (chronological, never shuffled).
    split_fraction: float = 0.8

    def threshold(self) -> float:
        """Return the classification threshold for the configured interval."""
        return {
            "1m": self.threshold_1m,
            "5m": self.threshold_5m,
            "1h": self.threshold_1h,
        }.get(self.interval, self.threshold_5m)

    def tag(self) -> str:
        """Return the artefact naming tag, e.g. ``BTCUSDT_5m_288p_24f``."""
        return f"{self.ticker}_{self.interval}_{self.past}p_{self.future}f"


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters. Defaults mirror the original scripts."""

    model: str = "rnn"  # rnn | cnn
    batch_size: int = 256
    epochs: int = 1000
    learning_rate: float = 1e-3
    dropout: float = 0.25
    l2: float = 1e-5
    seed: int = 42
    artifacts_dir: Path = ARTIFACTS_DIR


@dataclass(frozen=True)
class LiveConfig:
    """Streaming inference configuration (optional feature)."""

    influx_url: str = "http://localhost:8086"
    influx_token: str = ""
    influx_org: str = ""
    influx_bucket: str = ""
    measurement: str = "BTC"
    poll_seconds: int = 60

    @classmethod
    def from_env(cls) -> LiveConfig:
        """Build a :class:`LiveConfig` from environment variables (see .env.example)."""
        load_dotenv()
        return cls(
            influx_url=os.getenv("INFLUX_URL", cls.influx_url),
            influx_token=os.getenv("INFLUX_TOKEN", ""),
            influx_org=os.getenv("INFLUX_ORG", ""),
            influx_bucket=os.getenv("INFLUX_BUCKET", ""),
        )


@dataclass(frozen=True)
class Config:
    """Aggregate configuration passed around the application."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def merged(self, **overrides: Any) -> Config:
        """Return a copy with top-level nested values overridden.

        ``overrides`` keys may be prefixed by section, e.g. ``features__past=96``.
        ``None`` values are ignored so CLI defaults compose cleanly.
        """
        data_kwargs: dict[str, Any] = {}
        feat_kwargs: dict[str, Any] = {}
        train_kwargs: dict[str, Any] = {}
        for key, value in overrides.items():
            if value is None:
                continue
            if "__" in key:
                section, name = key.split("__", 1)
                target = {
                    "data": data_kwargs,
                    "features": feat_kwargs,
                    "train": train_kwargs,
                }.get(section)
                if target is not None:
                    target[name] = value
            elif key in {"ticker", "interval", "past", "future", "n_features"}:
                feat_kwargs[key] = value
                if key in {"ticker", "interval"}:
                    data_kwargs[key] = value
            elif key in {"batch_size", "epochs", "learning_rate", "model", "seed"}:
                train_kwargs[key] = value
        return Config(
            data=replace(self.data, **data_kwargs),
            features=replace(self.features, **feat_kwargs),
            train=replace(self.train, **train_kwargs),
        )


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load configuration from an optional YAML file and apply overrides."""
    config = Config()
    if path is not None:
        yaml_path = Path(path)
        if yaml_path.exists() and yaml is not None:
            raw = yaml.safe_load(yaml_path.read_text()) or {}
            if "data" in raw:
                config = Config(
                    data=replace(config.data, **raw["data"]),
                    features=config.features,
                    train=config.train,
                )
            if "features" in raw:
                config = Config(
                    data=config.data,
                    features=replace(config.features, **raw["features"]),
                    train=config.train,
                )
            if "train" in raw:
                config = Config(
                    data=config.data,
                    features=config.features,
                    train=replace(config.train, **raw["train"]),
                )
    return config.merged(**overrides)
