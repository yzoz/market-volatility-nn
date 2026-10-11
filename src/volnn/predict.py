"""Inference and evaluation with human-readable reports."""

from __future__ import annotations

from pathlib import Path

from volnn.config import Config
from volnn.train import evaluate_model
from volnn.utils.io import ensure_dir, save_metrics


def run(cfg: Config) -> dict[str, float]:
    """Evaluate the configured branch, persist metrics and print a report."""
    metrics = evaluate_model(cfg)
    name = f"{cfg.train.model}_{cfg.features.tag()}"
    reports = ensure_dir(Path(cfg.train.artifacts_dir) / "reports")
    save_metrics(reports / f"{name}_metrics.json", metrics)

    print(f"=== {name} ===")
    for key, value in metrics.items():
        print(f"{key:>10}: {value:.6f}")
    return metrics
