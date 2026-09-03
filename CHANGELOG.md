# Changelog

All notable changes to durable-llm are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Scope note: the version promise covers the names exported from `durable_llm/__init__.py` and the
database schema. Anything else — the worker loop, the CLI's exact output, the migration
runner — may change in a patch release without an entry here, and entries that mention those
modules are informational rather than contractual.

## [Unreleased]

First round of fixes following a full code review. Nothing has been released yet, so the
breaking changes below are recorded for anyone tracking `main` rather than a tag.

### Breaking

- **The project is now `durable-llm`.** Install the `durable-llm` distribution, import the
  `durable_llm` package, run the `durable-llm` command, and configure it with
  `DURABLE_LLM_*` environment variables. No legacy aliases are shipped because no version of
  the old name was published. The three existing migration files remain byte-for-byte
  unchanged so databases that already recorded their checksums continue to validate.

- **Idempotency keys changed for every call, and `cache` now has two renderers.**
  `cache.canonical_json` no longer falls back to `default=str` for values JSON cannot
  represent natively. `str()` is type-blind, so `Decimal("1")`, `"1"` and a `UUID` with
  matching text all hashed identically and could be served each other's stored responses — a
  wrong answer, not a missed cache. The fallback is now `{"__type__": ..., "__str__": ...}`.
  Existing databases are not corrupted and need no migration, but every previously cached call
  misses once and is re-billed, after which the new keys are in the ledger and replay resumes.

  The blast radius was wider than the ledger, because `canonical_json` had a second caller:
  `context` rendered the payload a model reads with the same function. Tagging it therefore
  also rewrote **prompt text** — `{"amount":"1.50"}` became
  `{"amount":{"__str__":"1.50","__type__":"Decimal"}}` — and lengthened the string
  `budget.project_cost` measures, inflating the **pre-call cost projection** for any payload
  carrying a non-JSON type. Both effects have since been split back out: `canonical_json` is
  tagged and is key material only, and a new `cache.prompt_json` keeps the old `default=str`
  rendering, which `context` uses to build `payload_text`. So prompts and projections end up
  where they were before the fix, and the keys are the only thing that moved.
- **`worker.DEFAULT_MODELS` removed.** The name-keyed fallback table let a typo'd or unknown
  provider silently resolve to a model nobody chose. `worker.resolve_default_model` now asks
  the provider.
- **`context.FALLBACK_MODEL` removed.** A provider that names no model used to fall back to
  `echo`, which is priced at zero — so a misconfiguration produced runs that appeared to cost
  nothing while doing no real work.
- **A provider must name a model.** `context._provider_default_model` raises `PermanentError`
  instead of substituting a default. This is permanent by classification because retrying a
  configuration mistake five times only costs five times as much to learn the same thing.

### Added

- **Run-level backoff.** `runs.next_attempt_at` plus a partial index on it; the claim query
  honours it, so a run that fails transiently waits before a worker picks it up again instead
  of being retried as fast as the poll loop can spin.
- **Migration checksums.** `schema_migrations.checksum` records a hash of the SQL that was
  applied, so an edited migration file is detected rather than silently skipped as
  "already applied".
- **Schema constraints.** Migration `0003_integrity_and_pacing.sql` adds a `UNIQUE` index on
  `llm_calls (run_id, input_hash) WHERE cached = false` — the idempotency ledger's core
  invariant, previously enforced only by application code — along with `CHECK` constraints on
  `runs`, `steps` and `llm_calls` covering statuses, error classes, non-negative money and
  token counters, and the rule that a cached call costs zero.
- **Provider latency.** `llm_calls.duration_ms` records how long each provider call took.
- **Built-package CI gate.** CI now builds the wheel and source distribution, installs the
  wheel outside the checkout, applies its bundled migrations to a clean Postgres service, and
  runs `durable-llm health`. This catches source-only files that an editable checkout can hide.
- **`Provider`, `get_provider`, `Price` and `register_price` are now part of the top-level
  facade.** Writing a custom provider and refreshing the dated price snapshot are ordinary
  configuration, so they belong inside the stability promise. `import durable_llm` still pulls in
  no provider SDK: `get_provider` imports `anthropic` inside its own body.
- **`CHANGELOG.md`** — this file.

### Fixed

- **Bundled Anthropic prices now match the published standard global API rates verified on
  2026-09-03.** Sonnet 5 is `$2/$10` per million input/output tokens; Anthropic made its
  introductory price permanent instead of applying the previously announced `$3/$15` rate.
  Opus 5 remains `$5/$25` and Haiku 4.5 remains `$1/$5`. The code and README now identify
  these as a dated snapshot and name the pricing modes they do not represent.
- **The binding design and README describe the implementation that ships.** They now include
  migration `0003`, checksums, retry pacing, database constraints, progress-based lease renewal,
  the current CLI surface, UTC log metadata, and the loopback-only development database bind.
- **`enqueue` rejects NaN and Infinity.** `pipeline._encode_input` passes `allow_nan=False`,
  so a `float("nan")` argument raises `TypeError` at the call site — as the function's
  docstring already promised — instead of emitting the bare token `NaN`, which is not JSON,
  and dying two layers down as a raw `psycopg.errors.InvalidTextRepresentation` on the INSERT.
- **Both documented spellings of `@pipeline` type-check.** The decorator's
  `Pipeline | Callable[..., Pipeline]` return type left a checker unable to tell `@pipeline`
  from `@pipeline(...)`, so on a package shipping `py.typed` every user of either form saw
  errors. `@overload` declarations now discriminate on whether `fn` was supplied; the union
  survives only on the implementation signature.
- **Error details are truncated in one place.** `durable_llm.errors` exports
  `error_detail(exc, limit=MAX_ERROR_DETAIL)`; the private copies in `worker.py` and
  `context.py` are gone, so a provider that returns a megabyte of HTML cannot land a megabyte
  in `runs.error_detail` by whichever path happened to catch it.
- **`RunRecord` carries `max_attempts`.** The worker's retry decision now has the ceiling the
  run was enqueued under available to it, and takes the lowest of that, the disposition's
  attempt budget, and the pipeline's current setting. Lowest-wins in both directions: raising
  the limit today does not retroactively grant attempts to a run that is already failing, and
  `@pipeline(max_attempts=1)` still means "never retry this" for a run enqueued under a more
  generous ceiling.
