.DEFAULT_GOAL := help
PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help install install-full lint format typecheck test check data train benchmark pipeline dashboard clean

help: ## Show available targets
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

install: $(BIN)/python ## Install the package with dashboard and dev tools
	$(BIN)/pip install -e ".[dev,dashboard]"

install-full: $(BIN)/python ## Also install Qiskit and PyTorch (large download)
	$(BIN)/pip install -e ".[dev,dashboard,quantum,lstm]"

lint: ## Run ruff lint and format checks
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format: ## Auto-format and fix lint issues
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

typecheck: ## Run mypy
	$(BIN)/mypy

test: ## Run the test suite with coverage
	$(BIN)/pytest --cov --cov-report=term-missing

check: lint typecheck test ## Run every check CI runs

data: ## Simulate the labelled dataset
	$(BIN)/adaptive-qkd generate

train: ## Train and evaluate the classifiers
	$(BIN)/adaptive-qkd train

benchmark: ## Compare defence policies
	$(BIN)/adaptive-qkd benchmark

pipeline: ## Generate data, train, and benchmark
	$(BIN)/adaptive-qkd pipeline

dashboard: ## Launch the Streamlit dashboard
	$(BIN)/streamlit run dashboard/app.py

clean: ## Remove caches and generated data/models (keeps results/)
	rm -rf data models .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
