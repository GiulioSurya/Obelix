.PHONY: setup lint format test

setup: ## Setup dev environment + git hooks
	uv sync --all-extras --group dev
	uv run pre-commit install --hook-type pre-commit --hook-type pre-push

lint: ## Run linter
	uv run ruff check .

format: ## Run formatter
	uv run ruff format .

test: ## Run tests
	uv run pytest tests/ -x -q
