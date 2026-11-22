# Developer entry points. CI runs `install`, `check`, then the commit-stage hooks.
# Match the virtual environment used by the Git hooks and CI.
override BIN := .venv/bin
ENGINE := $(wildcard castle_steward_engine)
TESTS := $(wildcard tests)

.PHONY: bootstrap install hooks hooks-check fmt check check-python

bootstrap: hooks           ## one-time setup in a clone

install:                   ## create the virtual environment and install development dependencies
	uv venv --allow-existing .venv
	uv pip install --python $(BIN)/python --group dev $(if $(ENGINE),-e .)

hooks: install             ## install the pre-commit and pre-push git hooks
	$(BIN)/pre-commit install --install-hooks

hooks-check:               ## both hook stages on every project file, tracked or not
	git ls-files -co --exclude-standard -z | xargs -0 $(BIN)/pre-commit run --show-diff-on-failure --hook-stage pre-commit --files
	git ls-files -co --exclude-standard -z | xargs -0 $(BIN)/pre-commit run --hook-stage pre-push --files

fmt:                       ## auto-fix lint, then format
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

check:                     ## non-mutating: ruff format --check, ruff check, mypy, pytest
	$(BIN)/ruff format --check .
	$(BIN)/ruff check .
	$(MAKE) check-python

check-python:              ## type-check the engine and run tests when present
ifneq ($(ENGINE),)
	$(BIN)/mypy
else
	@echo "Skipping mypy because the engine package is absent."
endif
ifneq ($(TESTS),)
	$(BIN)/pytest -q tests
else
	@echo "Skipping pytest because the tests directory is absent."
endif
