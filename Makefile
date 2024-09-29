all: lint type-check test

# lint + format(fix)
lint:
	uv run ruff format .
	uv run ruff check --fix .
# lint + format(not fix)
lint-check:
	uv run ruff format --check .
	uv run ruff check .
type-check:
	uv run mypy --install-types --non-interactive src
	uv run mypy src
test:
	uv run pytest -s