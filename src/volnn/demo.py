"""End-to-end demo on synthetic data.

Generates synthetic klines, prepares features/labels, trains (for a couple of
epochs) and produces report plots — all on CPU with no network access.
"""

from __future__ import annotations

from volnn.config import ARTIFACTS_DIR, DATA_DIR, Config, DataConfig, FeatureConfig, TrainConfig
from volnn.data.prepare import prepare
from volnn.data.synthetic import write_klines_csv
from volnn.utils.io import ensure_dir


def run_demo(seed: int = 42, steps: int = 20000, epochs: int = 1) -> dict:
    """Execute the demo pipeline and return the training history."""
    from volnn import viz
    from volnn.train import train_model

    data_dir = ensure_dir(DATA_DIR)
    artifacts = ensure_dir(ARTIFACTS_DIR)
    reports = ensure_dir(artifacts / "reports")

    past, future = 64, 8
    klines_path = write_klines_csv(str(data_dir / "DEMO-5m.csv"), n_steps=steps, seed=seed)

    cfg = Config(
        data=DataConfig(ticker="DEMO", interval="5m"),
        features=FeatureConfig(
            ticker="DEMO", interval="5m", past=past, future=future, threshold_5m=0.05
        ),
        train=TrainConfig(model="rnn", epochs=epochs, batch_size=128, seed=seed),
    )
    prepare(cfg, klines_path)
    history = train_model(cfg)
    viz.plot_history(history, "demo training", reports / "demo_history.png")
    print(f"demo artefacts written under {artifacts}")
    return {"history": history, "artifacts": str(artifacts)}


if __name__ == "__main__":  # pragma: no cover
    run_demo()
