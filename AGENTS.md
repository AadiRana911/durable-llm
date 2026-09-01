# llmrun

## Purpose and authority

`llmrun` is a small Python library and CLI for crash-safe, budget-capped,
idempotent multi-step LLM pipelines with all durable state in PostgreSQL. A
worker may be killed and resumed without re-running completed steps or rebilling
a provider call already present in the ledger.

`docs/DESIGN.md` is the binding implementation contract. If code, README text,
or this file conflicts with it, follow the design document and update the stale
guidance. Keep the intended scope small: a CLI/library, four database tables,
and exactly two providers (deterministic offline `echo` and Anthropic), with no
agent framework, prompt-management layer, web UI, ORM, or Alembic.

## Stack and layout

- Python 3.11+ packaged with Hatchling; `uv.lock` is the dependency lock.
- `psycopg` is the only core runtime dependency. The Anthropic SDK is optional.
- The CLI uses stdlib `argparse`; do not add Click or Typer.
- PostgreSQL 17 is the state store; SQL migrations are additive numbered files
  in `migrations/` and are copied into the built wheel.
- `src/llmrun/`: public types/errors, cost/cache/budget logic, database and
  migration code, registry/pipeline/context/worker/CLI, and provider adapters.
- `examples/`: registered runnable pipelines. `tests/`: unit, integration, CLI,
  provider-contract, and `tests/chaos/` crash-resume coverage.
- `src/llmrun/__init__.py` defines the public import surface; preserve
  `py.typed` and update `CHANGELOG.md` for user-visible API behavior.

## Setup and commands

Requirements are Docker and `uv`:

```bash
uv sync --locked
make up       # start PostgreSQL on 55432 and apply migrations
make demo     # deterministic echo-provider demonstration
make test     # unit/integration suite, excluding chaos
make chaos    # real SIGKILL-and-resume proof
make lint     # ruff lint and format checks; read-only
make fmt      # writes Ruff formatting and autofixes
uv build      # wheel and source distribution
make down     # stop containers; preserve the database volume
```

Tests use a real PostgreSQL instance and may skip when it is unavailable. The
echo provider is the default and needs no key or model network. For Anthropic:

```bash
uv sync --locked --extra anthropic
# Set ANTHROPIC_API_KEY in the environment or a local ignored .env file.
```

CI tests Python 3.11, 3.12, and 3.13 with locked dependencies, Ruff,
migrations, the full pytest suite including chaos, the Anthropic SDK contract,
package builds, and dependency auditing.

## Durability and billing invariants

- Use `Decimal` for money and token-cost arithmetic; never introduce floats.
  Store and compare timestamps in UTC using PostgreSQL time.
- Parameterize every SQL statement. Never interpolate values into SQL with
  f-strings or string concatenation.
- `llm_calls` is both the append-only billing ledger and the idempotency record.
  Preserve the unique billed-call identity based on `(run_id, input_hash)`.
- Before a provider call, enforce the budget. After it returns, write the call
  ledger entry and spend in the same transaction before issuing another call,
  then enforce the post-call check described in `docs/DESIGN.md`.
- Cache identity must cover all request semantics, including prompt, payload,
  `max_tokens`, and repair round. Keep canonical JSON and stored prompt JSON as
  separate concepts.
- `DISPOSITIONS` is the single source for error taxonomy and retry behavior.
  Preserve the distinction between transient errors and malformed, policy,
  budget, and permanent failures.
- Worker claiming uses leases, `FOR UPDATE SKIP LOCKED`, and fencing. Attempts
  and failures are different counters; do not collapse them.
- Never edit an already-applied migration. Add the next numbered SQL file and
  ensure both source and packaged migrations remain discoverable.

## Code and test conventions

- Use `from __future__ import annotations`, typed public APIs, logging rather
  than `print` in library modules, and Ruff's 100-character line limit.
- Keep the core provider-free. Provider SDK imports belong only in their
  optional adapter and tests.
- Treat `tests/chaos/test_crash_resume.py` as the executable credibility claim:
  it must kill a real worker in the ledger-dependent window, resume it, compare
  output and billed-call count to a baseline, and prove at least one replay.
- Test observable durability, transaction, CLI, and provider behavior. Do not
  replace real PostgreSQL integration tests with mocks that bypass the contract.

## Safety

- `make clean` is destructive: it drops the PostgreSQL volume and removes the
  virtual environment, caches, and build output. Never run it without explicit
  authorization. `make down` is the non-destructive default.
- Never commit `.env`, provider keys, database dumps, or captured customer
  prompts/responses. The database stores prompt and response text in cleartext;
  keep the development database loopback-only and use synthetic test data.
- Do not call a paid provider during tests or validation unless the user
  explicitly requests it and accepts the budget/network impact.
