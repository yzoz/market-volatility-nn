"""Unified training / evaluation entry points."""

from __future__ import annotations

from typing import Any

from volnn.config import Config
from volnn.data.prepare import load_dataset


def train_model(cfg: Config, dataset=None) -> dict[str, Any]:
    """Train the configured branch (``cfg.train.model``)."""
    dataset = dataset or load_dataset(cfg)
    if cfg.train.model == "cnn":
        from volnn.models import cnn

        return cnn.train(cfg, dataset)
    from volnn.models import rnn

    return rnn.train(cfg, dataset)


def evaluate_model(cfg: Config, dataset=None) -> dict[str, float]:
    """Evaluate the configured branch on the validation split."""
    dataset = dataset or load_dataset(cfg)
    if cfg.train.model == "cnn":
        from volnn.models import cnn

        return cnn.evaluate(cfg, dataset)
    from volnn.models import rnn

    return rnn.evaluate(cfg, dataset)
