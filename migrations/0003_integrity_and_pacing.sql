-- 0003_integrity_and_pacing — make the database enforce the invariants the code assumes,
-- give the claim query an index that actually serves it, and add the one column that lets a
-- failed run be retried on a schedule instead of instantly.
--
-- Still four tables. Everything here is a constraint, an index, or a column that closes a
-- gap review found between what a module's docstring promises and what the schema permits.
--
-- Three themes:
--
--   * **Constraints.** 0001 and 0002 declared no CHECK and no UNIQUE at all, so every
--     invariant the application maintains -- a legal status, a non-negative balance, at most
--     one billed call per idempotency key -- held only for as long as every writer stayed
--     correct. A ledger whose uniqueness is enforced by the application is a ledger that is
--     one bug away from double-billing, and the point of this project is that it does not.
--
--   * **Indexes.** `runs_claim_idx (status, created_at)` could not serve the claim query:
--     its OR across two status values defeats the ordering, so `ORDER BY created_at LIMIT 1`
--     fell back to scanning and sorting every queued and running row on every poll of every
--     worker. Leading with `created_at` and making the index partial fixes both halves --
--     the scan is ordered, and terminal rows (eventually almost the whole table) are not in
--     the index at all.
--
--   * **Pacing.** A requeued run was re-claimed immediately, usually by the worker that just
--     failed it, so run-level retries had no backoff -- the in-step loop paced its retries
--     and the run-level loop did not. `next_attempt_at` is where the wake-up time goes.

-- ======================================================================================
--  llm_calls: the idempotency key becomes a real constraint
-- ======================================================================================

-- Promote the idempotency lookup to a UNIQUE index. It is the same (run_id, input_hash)
-- WHERE cached = false predicate 0001 declared, so it still serves `_SELECT_RECORDED_CALL`
-- exactly as before and nothing is lost by replacing it -- but now a second billed row for
-- one key is rejected by Postgres rather than merely not written by careful code.
--
-- This is the constraint the whole no-double-billing claim rests on. `_billed_call` already
-- re-reads the ledger inside the row lock it spends behind, so a duplicate should be
-- unreachable; this makes "should be" into "cannot be", including for a future writer that
-- forgets the lock, and including for two workers racing through a window nobody has thought
-- of yet.
--
-- If this statement fails with a uniqueness violation, do not weaken it: it means duplicate
-- billed rows already exist, which is the bug it is here to prevent. Find them with
--   SELECT run_id, input_hash, count(*) FROM llm_calls WHERE cached = false
--   GROUP BY 1, 2 HAVING count(*) > 1;
DROP INDEX IF EXISTS llm_calls_cache_idx;
CREATE UNIQUE INDEX llm_calls_cache_idx
    ON llm_calls (run_id, input_hash) WHERE cached = false;

-- How long the provider took to answer, in milliseconds. The only external I/O in the system
-- was previously untimed, which meant a run that got slower could not be distinguished from a
-- provider that got slower -- in the logs or in SQL. Nullable because every row written
-- before this migration genuinely does not know, and because a replay (cached = true) never
-- called anybody.
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS duration_ms integer;

ALTER TABLE llm_calls DROP CONSTRAINT IF EXISTS llm_calls_tokens_chk;
ALTER TABLE llm_calls ADD CONSTRAINT llm_calls_tokens_chk
    CHECK (tokens_in >= 0 AND tokens_out >= 0 AND cost_usd >= 0);

ALTER TABLE llm_calls DROP CONSTRAINT IF EXISTS llm_calls_duration_chk;
ALTER TABLE llm_calls ADD CONSTRAINT llm_calls_duration_chk
    CHECK (duration_ms IS NULL OR duration_ms >= 0);

-- A replay costs nothing, by definition: `_replay_with_cursor` writes Decimal(0) and the
-- savings figure in `llmrun costs` is computed on the assumption that it did. State it here
-- so the assumption cannot quietly stop being true.
ALTER TABLE llm_calls DROP CONSTRAINT IF EXISTS llm_calls_cached_free_chk;
ALTER TABLE llm_calls ADD CONSTRAINT llm_calls_cached_free_chk
    CHECK (NOT cached OR cost_usd = 0);

-- ======================================================================================
--  runs: legal statuses, non-negative money, and a wake-up time
-- ======================================================================================

-- The status vocabulary is `llmrun.types.RunStatus`, and all six members belong here.
-- `halted_budget` in particular: it is the terminal status a budget stop writes, and a CHECK
-- that omitted it would turn every enforced cap into a failed INSERT.
ALTER TABLE runs DROP CONSTRAINT IF EXISTS runs_status_chk;
ALTER TABLE runs ADD CONSTRAINT runs_status_chk CHECK (
    status IN ('queued', 'running', 'succeeded', 'failed', 'pending_review', 'halted_budget')
);

-- `llmrun.errors.ErrorClass`. NULL is legal and means "no failure recorded".
ALTER TABLE runs DROP CONSTRAINT IF EXISTS runs_error_class_chk;
ALTER TABLE runs ADD CONSTRAINT runs_error_class_chk CHECK (
    error_class IS NULL
    OR error_class IN ('transient', 'malformed', 'policy', 'budget', 'permanent')
);

-- Money never goes backwards. `record_spend` already refuses a negative amount in Python;
-- this is the same rule stated where every writer has to obey it.
ALTER TABLE runs DROP CONSTRAINT IF EXISTS runs_money_chk;
ALTER TABLE runs ADD CONSTRAINT runs_money_chk CHECK (
    spent_usd >= 0 AND (budget_usd IS NULL OR budget_usd >= 0)
);

-- `max_attempts >= 1` is the one that matters: a row with max_attempts = 0 is unclaimable the
-- moment it is inserted, which is a stranded run created at enqueue time. Pipeline validates
-- this already; a row inserted by hand in psql did not.
ALTER TABLE runs DROP CONSTRAINT IF EXISTS runs_counters_chk;
ALTER TABLE runs ADD CONSTRAINT runs_counters_chk CHECK (
    attempt >= 0 AND failures >= 0 AND max_attempts >= 1
);

-- When a requeued run becomes claimable again. NULL means "now" -- the ordinary case, and the
-- value every row written before this migration carries.
--
-- Without it a retryable failure was requeued and re-claimed on the very next poll, almost
-- always by the worker that had just failed it: a provider having a bad thirty seconds got
-- answered with the whole retry budget spent inside those thirty seconds. The in-step loop
-- has paced, jittered backoff (`errors.backoff_delay`); this column is what lets the
-- run-level loop use the same function instead of hammering.
ALTER TABLE runs ADD COLUMN IF NOT EXISTS next_attempt_at timestamptz;

-- ======================================================================================
--  steps
-- ======================================================================================

ALTER TABLE steps DROP CONSTRAINT IF EXISTS steps_status_chk;
ALTER TABLE steps ADD CONSTRAINT steps_status_chk
    CHECK (status IN ('running', 'succeeded', 'failed'));

ALTER TABLE steps DROP CONSTRAINT IF EXISTS steps_attempt_chk;
ALTER TABLE steps ADD CONSTRAINT steps_attempt_chk CHECK (attempt >= 0);

-- ======================================================================================
--  indexes for the queries that actually run
-- ======================================================================================

-- The claim index, rebuilt. The old `(status, created_at)` could not serve
--
--     WHERE attempt < max_attempts
--       AND (status='queued' OR (status='running' AND locked_at < ...))
--     ORDER BY created_at LIMIT 1
--
-- because the OR spans two status values: Postgres can bitmap-OR two index scans, but a
-- BitmapOr destroys ordering, so the LIMIT 1 had to be fed by a sort of the whole matched
-- set. Leading with `created_at` gives the planner the ordering for free and lets it stop at
-- the first row that passes the residual filter.
--
-- Partial on the two non-terminal statuses because that is the whole point: a queue table
-- accumulates finished rows forever, and the working set is only ever the handful that are
-- queued or running. A succeeded run from last March should not be in this index at all.
DROP INDEX IF EXISTS runs_claim_idx;
CREATE INDEX runs_claim_idx ON runs (created_at)
    WHERE status IN ('queued', 'running');

-- `llmrun runs list` with no --tenant orders by created_at DESC over the whole table. The
-- tenant-scoped path has runs_tenant_idx; this is its unfiltered counterpart.
CREATE INDEX IF NOT EXISTS runs_created_idx ON runs (created_at DESC);

-- `llmrun runs show <prefix>` matches on `id::text LIKE 'abc%'`. A btree on the uuid cannot
-- serve a prefix match on its text rendering, so that lookup was a sequential scan -- with a
-- comment above it claiming otherwise. text_pattern_ops is what makes an anchored LIKE
-- indexable; the expression is immutable, so it is legal to index.
CREATE INDEX IF NOT EXISTS runs_id_prefix_idx ON runs ((id::text) text_pattern_ops);

-- Supports the claim query's `next_attempt_at IS NULL OR next_attempt_at <= now()` on the
-- backed-off rows, and lets an operator answer "what is waiting, and until when?" cheaply.
CREATE INDEX IF NOT EXISTS runs_next_attempt_idx ON runs (next_attempt_at)
    WHERE next_attempt_at IS NOT NULL;

-- ======================================================================================
--  schema_migrations: detect an edited migration instead of silently diverging
-- ======================================================================================

-- sha256 of the file's bytes at the moment it was applied. Migrations are forward-only with
-- no down path, so the failure mode is not a bad rollback -- it is someone editing an
-- already-applied .sql file, the runner skipping it because the version is recorded, and the
-- database quietly not matching the checkout that claims to describe it. Storing the digest
-- turns that into a startup error naming the file.
--
-- NULL for every row applied before this migration: those were recorded by a runner that did
-- not compute a digest, and inventing one now would assert something nobody checked.
ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS checksum text;
