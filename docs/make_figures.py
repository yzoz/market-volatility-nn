"""Generate the figures embedded in README.md.

This is a development utility, not part of the installable package. It trains
both branches for a couple of epochs on synthetic data (CPU, no network) and
writes PNGs into ``docs/images/``.

Run from the repository root:

    python docs/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from volnn import viz  # noqa: E402
from volnn.config import Config, DataConfig, FeatureConfig, TrainConfig  # noqa: E402
from volnn.data.features import build_feature_frame  # noqa: E402
from volnn.data.prepare import load_dataset, prepare  # noqa: E402
from volnn.data.synthetic import write_klines_csv  # noqa: E402
from volnn.models.common import chronological_split, make_dataset  # noqa: E402
from volnn.train import train_model  # noqa: E402

import pandas as pd  # noqa: E402

IMAGES = ROOT / "docs" / "images"
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

PAST, FUTURE, STEPS, EPOCHS = 48, 6, 6000, 15


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    klines_path = write_klines_csv(str(DATA / "DOCS-5m.csv"), n_steps=STEPS, seed=11)
    klines = pd.read_csv(klines_path, header=None)

    features = FeatureConfig(ticker="DOCS", interval="5m", past=PAST, future=FUTURE, threshold_5m=0.05)
    frame = build_feature_frame(klines, features)

    # --- Volatility series and feature view ---------------------------------
    viz.plot_series(frame[:, 0], "Realised volatility (synthetic)", IMAGES / "volatility_series.png")

    # --- RNN regression -----------------------------------------------------
    rnn_cfg = Config(
        data=DataConfig(ticker="DOCS", interval="5m"),
        features=features,
        train=TrainConfig(model="rnn", epochs=EPOCHS, batch_size=128, seed=11, artifacts_dir=ARTIFACTS),
    )
    prepare(rnn_cfg, klines_path)
    rnn_history = train_model(rnn_cfg)
    viz.plot_history(rnn_history, "RNN training (synthetic)", IMAGES / "rnn_loss_curves.png")

    dataset = load_dataset(rnn_cfg)
    _, x_val, _, y_val = chronological_split(dataset.x, dataset.y, features.split_fraction)
    from tensorflow import keras

    rnn_model = keras.models.load_model(str(ARTIFACTS / "models" / f"rnn_{features.tag()}.keras"))
    y_pred = rnn_model.predict(x_val, verbose=0).reshape(-1)
    viz.plot_predicted_vs_actual(y_val, y_pred, IMAGES / "rnn_predicted_vs_actual.png")

    # --- CNN classification -------------------------------------------------
    cnn_cfg = Config(
        data=DataConfig(ticker="DOCS", interval="5m"),
        features=features,
        train=TrainConfig(model="cnn", epochs=EPOCHS, batch_size=128, seed=11, artifacts_dir=ARTIFACTS),
    )
    prepare(cnn_cfg, klines_path)
    cnn_history = train_model(cnn_cfg)
    viz.plot_history(cnn_history, "CNN training (synthetic)", IMAGES / "cnn_loss_curves.png")

    cnn_dataset = load_dataset(cnn_cfg)
    _, cx_val, _, cy_val = chronological_split(cnn_dataset.x, cnn_dataset.y, features.split_fraction)
    cnn_model = keras.models.load_model(str(ARTIFACTS / "models" / f"cnn_{features.tag()}.keras"))
    proba = cnn_model.predict(cx_val[..., np.newaxis], verbose=0).reshape(-1)
    preds = (proba >= 0.5).astype(int)
    viz.plot_confusion(cy_val, preds, IMAGES / "cnn_confusion_matrix.png")
    try:
        viz.plot_roc(cy_val, proba, IMAGES / "cnn_roc_curve.png")
    except ValueError:
        pass

    print(f"figures written to {IMAGES}")


if __name__ == "__main__":
    main()
