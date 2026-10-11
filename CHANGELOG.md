# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-11

### Added
- Packaged the project as an installable Python distribution with a Typer-based
  CLI (`volnn`) exposing `download`, `prepare`, `train`, `predict`, `live` and
  `demo` commands (`python -m volnn` also works).
- Dataclass-based configuration (`DataConfig`, `FeatureConfig`, `TrainConfig`,
  `LiveConfig`) with YAML and environment/CLI overrides.
- Unified feature/label pipeline for both branches with an explicit
  feature-count guard.
- Baseline models (persistence for regression; majority and logistic
  regression for classification) for comparison against the learned models.
- Synthetic OHLCV generator and an end-to-end `volnn demo` pipeline that runs
  on CPU without network access.
- Visualisation helpers (loss curves, predicted-vs-actual, confusion matrix,
  ROC curve) writing PNG reports to `artifacts/reports/`.
- Unit tests for features, labels, sliding windows, scaling, CSV aggregation
  and synthetic data, plus TensorFlow smoke tests for the model builders.
- Exploratory (`01_eda`) and training (`02_training`) notebooks.
- Example figures under `docs/images/` (regenerable via `docs/make_figures.py`).
- GitHub Actions CI (ruff + pytest) and a `Makefile` with common targets.
- README, MIT `LICENSE`, `.env.example` and release notes.

### Changed
- Reorganised the flat script collection into the `src/volnn` package
  (`data/`, `models/`, `utils/` subpackages).
- Introduced one shared realised-volatility window for both branches
  (previously `h=4` for CNN and `h=12` for RNN).
- CSV aggregation now sorts by open time and de-duplicates, removing reliance
  on directory-walk ordering.
- Scaling is now fit on the training split only, avoiding look-ahead leakage.
- Secrets are read exclusively from environment variables (`.env`).

### Fixed
- CNN input reshape derived its width from the feature count instead of a
  hard-coded `7`, matching the actual 4-channel feature stack.
- Guarded weight loading so a clean run no longer attempts to load a
  non-existent checkpoint.

### Removed
- Tutorial scratch scripts (`check.py`) and unrelated tutorial/interview code.
