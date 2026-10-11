"""Command-line interface for market-volatility-nn (Typer).

Entry point is declared as ``volnn = "volnn.cli:app"`` in ``pyproject.toml``.
Also runnable as ``python -m volnn``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from volnn.config import load_config

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Deep learning volatility forecasting toolkit (RNN regression + CNN classification).",
)


def _config(
    config: Path | None,
    ticker: str | None,
    interval: str | None,
    past: int | None,
    future: int | None,
    model: str,
    epochs: int | None,
    batch_size: int | None,
    seed: int | None,
):
    """Build a Config from CLI options, layering over YAML and defaults."""
    return load_config(
        config,
        ticker=ticker,
        interval=interval,
        past=past,
        future=future,
        model=model,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )


@app.command()
def download(
    config: Path | None = typer.Option(None, "--config", help="YAML config file."),
    ticker: str = typer.Option("BTCUSDT", help="Trading pair symbol."),
    interval: str = typer.Option("5m", help="Kline interval (1m, 5m, 1h ...)."),
) -> None:
    """Download historical klines into the data directory."""
    from volnn.config import DataConfig
    from volnn.data.download import download_klines

    cfg = _config(config, ticker, interval, None, None, "rnn", None, None, None)
    download_klines(DataConfig(ticker=cfg.features.ticker, interval=cfg.features.interval))
    typer.echo(f"download complete -> {cfg.data.data_dir}")


@app.command()
def prepare(
    config: Path | None = typer.Option(None, "--config", help="YAML config file."),
    model: str = typer.Option("rnn", help="Branch to prepare labels for: rnn | cnn."),
    ticker: str | None = typer.Option(None),
    interval: str | None = typer.Option(None),
    past: int | None = typer.Option(None),
    future: int | None = typer.Option(None),
    source: Path | None = typer.Option(None, help="Raw klines CSV to prepare from."),
) -> None:
    """Build feature and label arrays from raw klines."""
    from volnn.data.prepare import prepare as prepare_data

    cfg = _config(config, ticker, interval, past, future, model, None, None, None)
    src = source or Path(cfg.data.data_dir) / f"{cfg.features.ticker}-{cfg.features.interval}.csv"
    written = prepare_data(cfg, src)
    for kind, path in written.items():
        typer.echo(f"wrote {kind}: {path}")


@app.command()
def train(
    config: Path | None = typer.Option(None, "--config", help="YAML config file."),
    model: str = typer.Option("rnn", help="Branch to train: rnn | cnn."),
    ticker: str | None = typer.Option(None),
    interval: str | None = typer.Option(None),
    past: int | None = typer.Option(None),
    future: int | None = typer.Option(None),
    epochs: int | None = typer.Option(None),
    batch_size: int | None = typer.Option(None),
    seed: int | None = typer.Option(None),
) -> None:
    """Train the configured branch and save artefacts."""
    from volnn.train import train_model

    cfg = _config(config, ticker, interval, past, future, model, epochs, batch_size, seed)
    history = train_model(cfg)
    typer.echo(f"training finished; final loss={history.get('loss', ['n/a'])[-1]}")


@app.command()
def predict(
    config: Path | None = typer.Option(None, "--config", help="YAML config file."),
    model: str = typer.Option("rnn", help="Branch to evaluate: rnn | cnn."),
    ticker: str | None = typer.Option(None),
    interval: str | None = typer.Option(None),
    past: int | None = typer.Option(None),
    future: int | None = typer.Option(None),
) -> None:
    """Evaluate the configured branch and produce a metrics report."""
    from volnn import predict as predict_mod

    cfg = _config(config, ticker, interval, past, future, model, None, None, None)
    predict_mod.run(cfg)


@app.command()
def live(
    config: Path | None = typer.Option(None, "--config", help="YAML config file."),
    model_path: Path | None = typer.Option(None, help="Path to a saved Keras model."),
) -> None:
    """Stream predictions from InfluxDB (requires INFLUX_* env vars)."""
    from volnn import live as live_mod

    cfg = _config(config, None, None, None, None, "rnn", None, None, None)
    target = (
        model_path or Path(cfg.train.artifacts_dir) / "models" / f"rnn_{cfg.features.tag()}.keras"
    )
    live_mod.run(cfg, target)


@app.command()
def demo(
    seed: int = typer.Option(42, help="Random seed for synthetic data."),
    steps: int = typer.Option(20000, help="Number of synthetic timesteps."),
    epochs: int = typer.Option(1, help="Training epochs for the demo."),
) -> None:
    """Run the full pipeline end-to-end on synthetic data (no network, CPU-only)."""
    from volnn.demo import run_demo

    run_demo(seed=seed, steps=steps, epochs=epochs)


if __name__ == "__main__":  # pragma: no cover
    app()
