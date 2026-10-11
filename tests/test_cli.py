"""CLI smoke tests using Typer's CliRunner (no training, no network)."""

from __future__ import annotations

import re

import pytest

typer = pytest.importorskip("typer")

from typer.testing import CliRunner  # noqa: E402

from volnn.cli import app  # noqa: E402

runner = CliRunner()

# Typer/Rich may emit ANSI colour codes and wrap text to the terminal width,
# so assertions run against clean, unwrapped output in a wide "terminal".
ENV = {"COLUMNS": "200", "NO_COLOR": "1", "TERM": "dumb"}
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(output: str) -> str:
    """Strip ANSI escapes and collapse whitespace for stable assertions."""
    return re.sub(r"\s+", " ", ANSI.sub("", output))


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"], env=ENV)
    assert result.exit_code == 0
    out = _plain(result.stdout)
    for cmd in ("download", "prepare", "train", "predict", "live", "demo"):
        assert cmd in out


def test_train_help_exposes_model_option():
    result = runner.invoke(app, ["train", "--help"], env=ENV)
    assert result.exit_code == 0
    out = _plain(result.stdout)
    assert "--model" in out
    assert "rnn" in out and "cnn" in out


def test_unknown_command_fails():
    result = runner.invoke(app, ["frobnicate"], env=ENV)
    assert result.exit_code != 0
