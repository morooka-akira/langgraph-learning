all: lint type-check test

# lint + format(fix)
lint:
	poetry run ruff format .
	poetry run ruff check --fix .
# lint + format(not fix)
lint-check:
	poetry run ruff format --check .
	poetry run ruff check .
type-check:
	poetry run mypy --install-types --non-interactive src
	poetry run mypy src
test:
	poetry run pytest -s