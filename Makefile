.PHONY: setup lint test download prepare train-rnn train-cnn predict demo clean

PY ?= python3

setup: ## Install the package with dev extras
	$(PY) -m pip install -e ".[dev]"

lint: ## Run ruff lint and format checks
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

test: ## Run the unit test suite
	$(PY) -m pytest

download: ## Download historical klines (requires the data extra)
	volnn download

prepare: ## Build features/labels (MODEL=rnn|cnn)
	volnn prepare --model $(or $(MODEL),rnn)

train-rnn: ## Train the RNN regression branch
	volnn train --model rnn

train-cnn: ## Train the CNN classification branch
	volnn train --model cnn

predict: ## Evaluate the configured branch (MODEL=rnn|cnn)
	volnn predict --model $(or $(MODEL),rnn)

demo: ## Run the full pipeline on synthetic data (no network, CPU only)
	volnn demo

clean: ## Remove generated data, artefacts and caches
	rm -rf data artifacts .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
