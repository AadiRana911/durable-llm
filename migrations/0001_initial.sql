-- 0001_initial — the whole schema. Four tables, and that is the budget: a fifth table
-- means the scope crept.
--
-- Conventions enforced here rather than in application code:
--   * ids are uuid DEFAULT gen_random_uuid() — built into Postgres 13+, no pgcrypto needed
--   * money is numeric(12,6), never float; the application always reads it as Decimal
--   * timestamps are timestamptz defaulted to now(), so the database owns the clock and
--     a worker with a skewed system time cannot corrupt the ordering of the ledger

-- Applied-migration bookkeeping. The runner (llmrun.migrate) creates this table itself
-- before applying anything, because a migration cannot be tracked by a table it is also
-- responsible for creating. It is repeated here, guarded by IF NOT EXISTS, so that this
-- file remains a complete description of the schema for anyone reading it cold.
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);

-- One row per pipeline execution: the unit of work a worker claims, the place the budget
-- cap and the running spend live, and the record of how the execution ended. `locked_by` /
-- `locked_at` implement the worker lease, so a run abandoned by a killed worker is
-- reclaimed after the lease expires instead of being stranded in 'running' forever.
CREATE TABLE runs (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    pipeline     text        NOT NULL,
    tenant_id    text        NOT NULL,
    status       text        NOT NULL,          -- see RunStatus
    input        jsonb       NOT NULL DEFAULT '{}'::jsonb,
    output       jsonb,
    budget_usd   numeric(12,6),                 -- NULL = unlimited
    spent_usd    numeric(12,6) NOT NULL DEFAULT 0,
    error_class  text,
    error_detail text,
    attempt      integer     NOT NULL DEFAULT 0,
    locked_by    text,
    locked_at    timestamptz,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now()
);

-- Supports the worker's claim query: find the oldest queued run (or the oldest run whose
-- lease has expired) without scanning the finished ones.
CREATE INDEX runs_claim_idx  ON runs (status, created_at);

-- Supports `llmrun runs list --tenant T` and the per-tenant cost report.
CREATE INDEX runs_tenant_idx ON runs (tenant_id, created_at DESC);

-- The resumption checkpoint: one row per named step of a run, holding that step's stored
-- output. A restarted worker replays a run by reading these rows instead of re-executing
-- the work. `input_hash` is what makes the replay honest — if a step's inputs changed the
-- hash differs and the step is re-executed rather than being served a stale output.
-- Keyed by (run_id, name) because a step name is unique within its run.
CREATE TABLE steps (
    run_id       uuid        NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name         text        NOT NULL,
    status       text        NOT NULL,          -- see StepStatus
    input_hash   text        NOT NULL,
    output       jsonb,
    attempt      integer     NOT NULL DEFAULT 0,
    error_class  text,
    error_detail text,
    started_at   timestamptz NOT NULL DEFAULT now(),
    ended_at     timestamptz,
    PRIMARY KEY (run_id, name)
);

-- BOTH the cost ledger AND the idempotency store — this table is why a crash cannot
-- double-bill you.
--
--   * cached = false: a real, billed provider call. It carries the true cost and the
--     stored response, and it is written in the same transaction as the spend it caused.
--   * cached = true:  a replay of an earlier identical call. It costs 0 and exists purely
--     to record that a retry was absorbed rather than paid for a second time.
--
-- Nothing is ever deleted or updated here; the ledger is append-only, so `SUM(cost_usd)`
-- is the authoritative answer to "what did this run cost".
CREATE TABLE llm_calls (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id      uuid        NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step_name   text        NOT NULL,
    attempt     integer     NOT NULL DEFAULT 0,
    input_hash  text        NOT NULL,
    model       text        NOT NULL,
    tokens_in   integer     NOT NULL DEFAULT 0,
    tokens_out  integer     NOT NULL DEFAULT 0,
    cost_usd    numeric(12,6) NOT NULL DEFAULT 0,
    cached      boolean     NOT NULL DEFAULT false,
    response    jsonb,                          -- {"text": ..., "model": ...}
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- The idempotency lookup. Partial on `cached = false` because only real calls can satisfy
-- a cache hit, and excluding the replays keeps the index small no matter how many times a
-- run is retried.
CREATE INDEX llm_calls_cache_idx ON llm_calls (run_id, input_hash) WHERE cached = false;

-- The per-run cost ledger in call order, as printed by `llmrun runs show`.
CREATE INDEX llm_calls_run_idx   ON llm_calls (run_id, created_at);
