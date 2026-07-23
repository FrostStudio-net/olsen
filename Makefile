install:
	python3 -m venv --upgrade-deps .venv
	.venv/bin/pip install -e '.[dev]'

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check .

fetch:
	.venv/bin/olsen fetch

sync-history:
	.venv/bin/olsen sync-history

train:
	.venv/bin/olsen train

backtest:
	.venv/bin/olsen backtest

walk-forward:
	.venv/bin/olsen walk-forward --config configs/v0.2.json

paper:
	.venv/bin/olsen paper
