# market-volatility-nn v1.0.0 — Release Notes

**Deep learning for cryptocurrency volatility forecasting: an RNN regression
branch and a CNN direction-classification branch.**

This is the first public release. It turns an early research prototype into a
structured, installable, tested Python package.

## Highlights

- **Two modelling branches**
  - *RNN regression* — a stacked (Bi)LSTM forecasting the future value of
    realised volatility (`MSE` / `MAE`).
  - *CNN classification* — a Conv2D network predicting the direction of the
    volatility change (`binary cross-entropy`, accuracy / precision / recall /
    ROC-AUC).
- **Typer CLI** `volnn` with `download`, `prepare`, `train`, `predict`, `live`
  and `demo` commands (`python -m volnn` works too).
- **One command demo**: `volnn demo` runs the whole pipeline end-to-end on
  synthetic data, on CPU, with no network access — ideal for a quick look.
- **Baselines** (naive persistence, majority class, logistic regression) for a
  meaningful model-vs-baseline comparison.
- **Reports & plots** saved to `artifacts/reports/`.
- **Unit tests + CI** (ruff + pytest) that pass without a GPU or network.

## Installation

```bash
git clone <your-repo-url>
cd market-volatility-nn
pip install -e ".[dev]"
```

## Quickstart

```bash
# End-to-end on synthetic data (no network, CPU only)
volnn demo

# With real data
pip install -e ".[data,viz]"
volnn download --ticker BTCUSDT --interval 5m
volnn prepare --model rnn
volnn train   --model rnn
volnn predict --model rnn
```

## Bug fixes

- CNN feature reshape now derives its width from the configured feature count
  (previously hard-coded to 7 while the stack had 4 channels).
- Weight loading is guarded, so a clean run no longer crashes.
- CSV aggregation sorts by open time and de-duplicates.
- A single realised-volatility window is shared by both branches (previously
  inconsistent between CNN and RNN).

## Notes

- Model architecture and hyperparameters from the original prototype are
  preserved as defaults.
- Secrets are read only from environment variables; see `.env.example`.
- **This project is research software and not financial advice.**

See `CHANGELOG.md` for the full list of changes.
