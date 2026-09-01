-- 0002_attempt_ceiling — make `max_attempts` a real ceiling, and stop lease churn from
-- spending the retry budget.
--
-- Still four tables. Two columns on `runs`, no new table, no new index: the scope did not
-- creep, one number just turned out to be doing two jobs badly.
--
-- The problem this fixes:
--
--   * `runs.attempt` is incremented by the worker's claim query, so it counts *claims* —
--     including the ones where a lease simply expired while its holder was still working.
--     Nothing read it back at claim time, so a run that reliably killed its worker was
--     re-claimed forever, and every cycle after a billed call bought that call again.
--   * The same counter was compared against the pipeline's `max_attempts` when deciding
--     whether to requeue a failed run. A pipeline whose steps legitimately outlive the
--     lease therefore arrived at its first genuine transient error with its retry budget
--     already spent, and was failed terminally without ever being retried once.
--
-- One counter cannot be both. So:
--
--   * `attempt` keeps counting claims, and `max_attempts` bounds it *in the claim query* —
--     that is the ceiling on executions, and it is now enforced where it has to be, since
--     a SIGKILLed worker never reaches the code that writes a verdict.
--   * `failures` counts only recorded failures, and it is what the requeue decision reads.
--     Lease churn can no longer consume a retry budget it never used.

-- Copied from the pipeline at enqueue time. A column rather than a lookup because the
-- claim query is SQL and has no way to ask a Python object how many attempts it allows.
-- The default matches llmrun.pipeline.DEFAULT_MAX_ATTEMPTS, so a row inserted by hand
-- (a fixture, a psql session) still gets a sane ceiling instead of an unbounded one.
ALTER TABLE runs ADD COLUMN IF NOT EXISTS max_attempts integer NOT NULL DEFAULT 5;

-- Incremented only by the worker's verdict statement, never by the claim.
ALTER TABLE runs ADD COLUMN IF NOT EXISTS failures integer NOT NULL DEFAULT 0;
