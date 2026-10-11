"""Tests for label construction (regression and classification)."""

from __future__ import annotations

import numpy as np
import pytest

from volnn.data.labels import build_classification_dataset, build_regression_dataset


def test_regression_dataset_alignment(frame, feature_cfg):
    hv = frame[:, 0]
    ds = build_regression_dataset(frame, hv, feature_cfg.past, feature_cfg.future)
    expected = frame.shape[0] - feature_cfg.past - feature_cfg.future
    assert ds.x.shape == (expected, feature_cfg.past, feature_cfg.n_features)
    assert ds.y.shape[0] == ds.x.shape[0]
    assert np.isclose(ds.y[0], hv[feature_cfg.past + feature_cfg.future])


def test_classification_dataset_labels_are_binary(frame, feature_cfg):
    hv = frame[:, 0]
    ds = build_classification_dataset(
        frame, hv, feature_cfg.past, feature_cfg.future, threshold=0.001
    )
    assert set(np.unique(ds.y)).issubset({0.0, 1.0})
    assert ds.x.shape[1:] == (feature_cfg.past, feature_cfg.n_features)


def test_classification_impossible_threshold_raises(frame, feature_cfg):
    hv = frame[:, 0]
    with pytest.raises(ValueError):
        build_classification_dataset(frame, hv, feature_cfg.past, feature_cfg.future, threshold=1e9)
