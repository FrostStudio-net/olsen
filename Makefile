install:
	python3 -m venv .venv
	.venv/bin/pip install --no-build-isolation -e '.[dev]'

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .

fetch:
	.venv/bin/olsen fetch

train:
	.venv/bin/olsen train

backtest:
	.venv/bin/olsen backtest

walk-forward:
	.venv/bin/olsen walk-forward --config configs/v0.2.json

paper:
	.venv/bin/olsen paper
