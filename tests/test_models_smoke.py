"""Smoke tests for model builders: shapes only, no training.

TensorFlow is an optional import for the test suite; the tests are skipped when
it is not installed so lint/unit tests still pass in lightweight environments.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="TensorFlow not installed")

from volnn.models.builders import build  # noqa: E402


@pytest.mark.parametrize("arch", ["rnn", "cnn"])
def test_builder_output_shape(arch):
    past, n_features = 16, 4
    model = build(arch, input_shape=(past, n_features))
    if arch == "cnn":
        x = np.zeros((2, past, n_features, 1), dtype="float32")
    else:
        x = np.zeros((2, past, n_features), dtype="float32")
    y = model(x, training=False)
    assert y.shape == (2, 1)


def test_builder_rejects_unknown_arch():
    with pytest.raises(ValueError):
        build("mlp", input_shape=(16, 4))
