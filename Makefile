# llmrun -- developer entrypoints.
#
# Everything below assumes exactly two things are installed: Docker and uv.
# `make demo` must work on a clean clone with nothing else. If it does not, that is a bug.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# Optional local overrides (LLMRUN_DATABASE_URL, LLMRUN_PROVIDER, ANTHROPIC_API_KEY).
# See .env.example. Plain KEY=value lines only.
-include .env

COMPOSE ?= docker compose
UV      ?= uv
# The `dev` dependency group (PEP 735) is installed by `uv run` automatically, so there
# is no extra to name and no way for a target to forget one.
UV_RUN  := $(UV) run

DB_SERVICE      ?= db
PG_USER         ?= llmrun
PG_DB           ?= llmrun
PG_HOST_PORT    ?= 55432
PG_WAIT_SECONDS ?= 60

LLMRUN_DATABASE_URL ?= postgresql://llmrun:llmrun@localhost:$(PG_HOST_PORT)/$(PG_DB)
LLMRUN_PROVIDER     ?= echo
export LLMRUN_DATABASE_URL
export LLMRUN_PROVIDER
export ANTHROPIC_API_KEY

# Put the repo root on the import path so `examples/` is importable by dotted name too.
export PYTHONPATH := $(CURDIR)$(if $(PYTHONPATH),:$(PYTHONPATH))

DEMO_PIPELINE ?= summarise_thread
DEMO_MODULE   ?= examples/summarise_thread.py
DEMO_TENANT   ?= demo
DEMO_INPUT    ?= {"thread_id": "thread-42"}

.PHONY: help up down demo test chaos fmt lint clean

help: ## Show this help
	@printf '\nllmrun -- make targets\n\n'
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'
	@printf '\nDatabase: %s\n\n' "$(LLMRUN_DATABASE_URL)"

up: ## Start Postgres, wait until it accepts connections, apply migrations
	@$(COMPOSE) up -d
	@printf '==> waiting for postgres (max %ss)' "$(PG_WAIT_SECONDS)"; \
	for i in $$(seq 1 $(PG_WAIT_SECONDS)); do \
		if $(COMPOSE) exec -T $(DB_SERVICE) \
			pg_isready -q -h 127.0.0.1 -p 5432 -U $(PG_USER) -d $(PG_DB) >/dev/null 2>&1; then \
			printf ' ready after %ss\n' "$$i"; \
			ready=1; \
			break; \
		fi; \
		printf '.'; \
		sleep 1; \
	done; \
	if [ "$${ready:-0}" != "1" ]; then \
		printf '\n'; \
		echo "postgres never became ready; last 50 log lines follow" >&2; \
		$(COMPOSE) logs --tail=50 $(DB_SERVICE) >&2; \
		exit 1; \
	fi
	@echo "==> applying migrations"
	@$(UV_RUN) llmrun migrate --quiet

down: ## Stop Postgres (keeps the data volume)
	$(COMPOSE) down --remove-orphans

demo: up ## Run the example pipeline end to end on the echo provider and show the run
	@printf '\n==> enqueueing %s for tenant %s\n' "$(DEMO_PIPELINE)" "$(DEMO_TENANT)"
	@run_id="$$($(UV_RUN) llmrun enqueue $(DEMO_PIPELINE) \
			--tenant $(DEMO_TENANT) \
			--json '$(DEMO_INPUT)' \
			--import $(DEMO_MODULE) --quiet \
		| grep -Eio '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' \
		| tail -n 1 || true)"; \
	if [ -z "$$run_id" ]; then \
		echo "make demo: no run id in 'llmrun enqueue' output -- see the error above" >&2; \
		exit 1; \
	fi; \
	printf '==> queued run %s\n' "$$run_id"; \
	printf '==> draining the queue with the echo provider (no API key, no network)\n'; \
	for _ in 1 2 3; do \
		$(UV_RUN) llmrun worker --once --provider echo --import $(DEMO_MODULE) --quiet; \
	done; \
	printf '\n==> llmrun runs show %s\n\n' "$$run_id"; \
	$(UV_RUN) llmrun runs show "$$run_id"

test: up ## Run the unit and integration suite (everything except chaos)
	$(UV_RUN) pytest -m "not chaos"

# -v so the output names the test rather than printing a single dot. The claim being
# made is specific enough that a reader should see which assertion passed.
chaos: up ## Run the crash-and-resume crash test (slow, kills a real worker)
	$(UV_RUN) pytest -m chaos -v

fmt: ## Format and autofix with ruff
	$(UV_RUN) ruff format .
	$(UV_RUN) ruff check --fix .

lint: ## Check formatting and lint rules without changing anything
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .

clean: ## Remove the database volume, the venv, caches and build output
	-$(COMPOSE) down --volumes --remove-orphans
	rm -rf .venv .pytest_cache .ruff_cache .mypy_cache build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
