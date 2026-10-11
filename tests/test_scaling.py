"""Tests for the train-only scaler."""

from __future__ import annotations

import numpy as np

from volnn.utils.scaling import TrainFitScaler, scale_frame, scale_series


def test_scaler_fits_on_train_only():
    train = np.array([10.0, 20.0, 30.0])
    scaler = TrainFitScaler().fit(train)
    # Unseen value above the training max scales above 1 (proves train-only fit).
    out = scaler.transform(np.array([40.0]))
    assert out[0] > 1.0


def test_scaler_output_range_and_rounding():
    data = np.linspace(0, 1, 11)
    scaled = scale_series(data, round_to=5)
    assert scaled.min() == 0.0 and scaled.max() == 1.0
    assert np.allclose(scaled, np.round(scaled, 5))


def test_scale_frame_fits_on_train_portion_only():
    # Column 0: train rows in [0, 10), val rows far outside that range.
    col0 = np.concatenate([np.arange(0, 10, dtype=float), np.array([1000.0, 2000.0])])
    col1 = np.arange(12, dtype=float)
    frame = np.column_stack([col0, col1])
    out = scale_frame(frame, split_fraction=10 / 12)
    # Train part of column 0 spans [0, 1]; the unseen 1000/2000 map well above 1.
    assert np.isclose(out[0, 0], 0.0)
    assert out[-2, 0] > 1.0 and out[-1, 0] > 1.0
    assert out.shape == frame.shape
