.PHONY: setup preprocess train optimize run-all test lint clean

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

preprocess:
	$(PYTHON) -m cement_opt.cli preprocess

train:
	$(PYTHON) -m cement_opt.cli train

optimize:
	$(PYTHON) -m cement_opt.cli optimize

run-all:
	$(PYTHON) -m cement_opt.cli run-all

test:
	$(PYTEST) -v

lint:
	$(RUFF) check src tests

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
