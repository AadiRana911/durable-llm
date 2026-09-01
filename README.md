# llmrun

Your 40-step enrichment pipeline dies at step 38. A worker OOMs, a pod is evicted, the
database fails over — it does not matter which. You restart it, and it starts again at step 1:
it re-runs all 37 steps that had already succeeded, re-issues every model call inside them, and
your provider bills you for all of them a second time. The output is eventually correct. The
invoice is not. This happens for one boring reason: the pipeline's progress lived in a Python
stack frame, and stack frames do not survive `SIGKILL`. llmrun moves that progress into
Postgres. Every step is checkpointed before it returns, every model call is written to a ledger
before the next one is issued, and a restarted worker replays from the ledger instead of from
the provider. Four tables, one runtime dependency, a CLI, and no new infrastructure.

```python
@pipeline(name="summarise_thread", budget_usd="2.50", max_attempts=5)
def summarise_thread(ctx: RunContext, thread_id: str) -> dict:
    thread = ctx.step("fetch_thread", load_thread, thread_id)          # checkpointed
    facts = ctx.llm("extract_facts", prompt=EXTRACT, payload=thread,   # billed once, ever
                    schema=Facts.model_validate)
    return ctx.llm("write_summary", prompt=WRITE, payload=facts)
```

Kill the worker anywhere in that function and restart it. `fetch_thread` is not re-run,
`extract_facts` is not re-billed, and the run finishes with the same output it would have
produced uninterrupted. There is a test in this repo that asserts exactly that against a real
Postgres and a real `SIGKILL`, and it is one command to run.

Built by Asad Imtiaz. I do this kind of plumbing — durability, cost control and failure
handling around LLM features — for B2B SaaS teams; details at the bottom.

---

## Quickstart: 60 seconds, no API key

You need Docker and [uv](https://docs.astral.sh/uv/). Nothing else — in particular, **no
provider API key and no network access to any model**. llmrun ships a deterministic `echo`
provider whose responses are a pure function of `sha256(request)`. It is not a mock bolted on
for tests; it is the provider the demo and the crash test run against, which is what
makes this a sixty-second experience rather than a signup flow.

```bash
git clone https://github.com/asadimtiaz/llmrun.git
cd llmrun
make up      # Postgres 17 in Docker on port 55432, then applies migrations
make demo    # enqueues a 12-step pipeline, drains it, prints the run
```

`make demo` depends on `make up`, so the second command alone is enough from a clean clone.

<sub>*Real output from `make demo` on a clean clone. `docker compose`'s own progress lines
and the run's INPUT/OUTPUT sections are cut for length; `…` marks where table rows repeat.
Ids, timings and digests will differ on your machine.*</sub>

```console
$ make demo
==> waiting for postgres (max 60s). ready after 2s
==> applying migrations
applied 0001_initial
applied 0002_attempt_ceiling
2 migration(s) applied

==> enqueueing summarise_thread for tenant demo
==> queued run 8d1f8f6d-b7d8-401f-b132-c68b8cd19257
==> draining the queue with the echo provider (no API key, no network)
8d1f8f6d-b7d8-401f-b132-c68b8cd19257

==> llmrun runs show 8d1f8f6d-b7d8-401f-b132-c68b8cd19257

RUN 8d1f8f6d-b7d8-401f-b132-c68b8cd19257
============================================================================================
  pipeline    summarise_thread
  tenant      demo
  status      SUCCEEDED
  attempts    1
  created     2026-08-12 15:21:08 UTC
  updated     2026-08-12 15:21:08 UTC
  wall clock  225ms
  budget      $2.500000
  spent       $0.000000  (0.0% of budget)
  cache       0 of 11 calls replayed

STEPS
  STEP                          STATUS     ATT  DURATION  CALLS  REPLAYS  COST USD  ERROR
  ----------------------------  ---------  ---  --------  -----  -------  --------  -----
  fetch_messages                succeeded    0       1ms      0        0  0.000000  -
  extract_participants          succeeded    0       6ms      1        0  0.000000  -
  write_opening                 succeeded    0       5ms      1        0  0.000000  -
  …
  write_executive_summary       succeeded    0       5ms      1        0  0.000000  -

LLM CALLS  (cost ledger)
     #  STEP                          ATT  MODEL  TOK IN  TOK OUT  COST USD  BILLING  SAVED USD  AT (UTC)
    --  ----------------------------  ---  -----  ------  -------  --------  -------  ---------  --------
     1  extract_participants            0  echo      461       51  0.000000  billed              15:21:08
     2  write_opening                   0  echo       68       46  0.000000  billed              15:21:08
     …
    11  write_executive_summary         0  echo      106       55  0.000000  billed              15:21:08

  no replays: every call in this run was billed exactly once.
```

Two columns are doing the work here. `COST USD` is `0.000000` because `echo` is priced at zero
— it makes no billed calls — and the arithmetic that produced those zeros is the identical
`Decimal` path that runs against Anthropic. And every row reads `billed`, because nothing
crashed, so nothing had to be replayed. The interesting version of this table is one section
down, where a worker was killed mid-run: the billed count is still 11, and one row reads
`CACHED`.

---

## Prove it: LLM pipeline retry without double billing

Claims about durability are cheap. This one is executable:

```bash
make chaos
```

<sub>*Real output. `make chaos` depends on `make up`, so its Postgres preamble and pytest's
session header come first and are cut here. The duration is machine-dependent.*</sub>

```console
$ make chaos
tests/chaos/test_crash_resume.py::test_a_sigkilled_worker_resumes_without_paying_twice PASSED [100%]

====================== 1 passed, 116 deselected in 9.54s =======================
```

What that test actually does:

1. Enqueue the 12-step example pipeline and run it to completion, uninterrupted. Record the
   output and `SELECT count(*) FROM llm_calls WHERE run_id = %s AND cached = false` — the number
   of real, billed provider calls. That is the **baseline**.
2. Enqueue an identical run. Start `llmrun worker` as a real subprocess.
3. Poll until roughly half the steps are `succeeded` **and** a provider call has committed to
   the ledger whose step has not yet been checkpointed, then `SIGKILL` the worker's process
   group. No `finally` block runs. No cleanup happens. The run is stranded in `running` with a
   stale lease.
4. Start a new worker. It reclaims the run once the lease expires and finishes it.
5. Assert the final output is identical to the baseline.
6. Assert the count of billed calls is **identical to the baseline**.
7. Assert at least one call was replayed from the ledger.

Assertion 6 is the one that matters. It is the difference between "we retry safely" as a
bullet on a landing page and a claim a skeptical reader executes themselves, from a clean
clone, in one command, before believing it.

Step 3 and assertion 7 exist because assertion 6 has a blind spot. A crash *between* steps is
recovered by the `steps` table alone, and a run killed there keeps the billed count equal
without the `llm_calls` ledger being consulted once — the test would pass with idempotency
deleted. So the kill is aimed at the only window where the ledger is load-bearing: a call paid
for, its step not yet checkpointed. Assertion 7 then insists the resumed worker actually
replayed it rather than getting lucky.

The killed run's ledger looks like this:

<sub>*Real output from a `make chaos` run. The run's INPUT/OUTPUT sections and the step
table are cut for length; `…` marks where ledger rows repeat.*</sub>

```console
$ llmrun runs show 6abe7fd6

RUN 6abe7fd6-b687-4adc-a655-75f07d0bae7d
============================================================================================
  pipeline    summarise_thread
  tenant      chaos-crash
  status      SUCCEEDED
  attempts    2
  created     2026-08-12 15:22:15 UTC
  updated     2026-08-12 15:22:24 UTC
  wall clock  9.02s
  budget      $2.500000
  spent       $0.000000  (0.0% of budget)
  cache       1 of 12 calls replayed, $0.000000 not spent again

LLM CALLS  (cost ledger)
      #  STEP                          ATT  MODEL  TOK IN  TOK OUT  COST USD  BILLING  SAVED USD  AT (UTC)
  -  --  ----------------------------  ---  -----  ------  -------  --------  -------  ---------  --------
      1  extract_participants            0  echo      462       47  0.000000  billed              15:22:17
      …
      6  write_action_items_section      0  echo       95       53  0.000000  billed              15:22:24
  *   7  write_action_items_section      1  echo       95       53  0.000000  CACHED    0.000000  15:22:24
      8  extract_risks                   0  echo      460       46  0.000000  billed              15:22:24
      …
     12  write_executive_summary         0  echo      105       54  0.000000  billed              15:22:24

  * 1 of 12 calls were replayed from this run's ledger: billed $0.000000, where
    an unguarded retry would have spent $0.000000 re-buying tokens it already had.
```

`attempts 2` is the crash and the resume. The run id is passed as a prefix above because
`runs show` accepts any unambiguous one — you rarely need the whole UUID.

Eleven billed calls, exactly as in the baseline, and twelve rows. Three different mechanisms
produced that result, and it is worth being precise about which did what:

- **Steps 1–6 were never re-executed.** They had committed to the `steps` table with their
  outputs, so the resumed run read them straight back and never reached the provider at all.
  This is resumption, and it is where nearly all of the savings come from.
- **Step 7, `write_action_items_section`, was in flight when the kill landed.** Its provider
  call had completed and its `llm_calls` row had committed, but the step had not yet been
  marked `succeeded`. On resume the step re-executed, computed the same `input_hash`, found the
  existing billed row, wrote a `cached=true` row costing zero, and returned the stored
  response — the `*` row above, and the reason its `ATT` column reads `1` where every other row
  reads `0`. This is idempotency, and it covers the one window that checkpointing alone cannot.
- **Steps 8–12 ran normally**, because they genuinely had not run yet.

**The honest caveat.** If a process dies in the microseconds between the provider's HTTP
response arriving and the local `INSERT` committing, that call was billed by the provider and
llmrun has no record of it — nothing outside the provider can fix that without provider-side
idempotency keys. What llmrun guarantees is narrower and still worth having: it never issues a
second call for work that is already durably recorded, and it never loses a response you were
billed for. The sequencing rule that makes this true is that a completed call is committed
**before** the next one is issued, so the exposure is one call wide, not one pipeline wide.

---

## The five parts of a durable LLM pipeline

### 1. Checkpointed steps — resume an LLM pipeline after a crash

`ctx.step(name, fn, *args, **kwargs)` runs `fn` once per run, ever. Before executing it, llmrun
hashes `(name, args, kwargs)` and looks for a `steps` row:

- row exists, `status='succeeded'`, hash matches → return the stored output, do not execute
- row exists, hash differs → the inputs changed, so re-execute and overwrite
- no row → execute, then **commit** `succeeded` with the serialised output before returning

That hash check is what keeps resumption honest. A resumed run that served a stale output for
changed inputs would be worse than one that re-ran everything, because it would be wrong
quietly. And the commit happens before the return, not at the end of the pipeline: an
uncommitted step is a lost step, and a step you have to redo is a step you have to pay for.

`ctx.llm(...)` gets the same treatment plus everything in the next section. A durable LLM pipeline
is, mechanically, nothing more than a pipeline in which every unit of work has somewhere durable
to write "done" — and somewhere durable to leave what it produced.

### 2. Idempotent LLM retry

Here is the full sequence of one `ctx.llm()` call, which is the whole mechanism in one place:

```
ctx.llm("extract_4", prompt=..., payload=..., schema=Facts.model_validate)
  │
  ├─ steps row succeeded with this input_hash? ────────yes─▶ return stored output (no call, no row)
  │
  ├─ llm_calls row (run_id, input_hash, cached=false)? ─yes─▶ INSERT cached=true, cost 0, return it
  │
  ├─ check_budget(projected worst-case cost) ──over cap──▶ BudgetExceededError → run halts
  │
  ├─ provider.complete(...) ────────────────────transient──▶ exponential backoff + jitter, ≤5 tries
  │
  ├─ INSERT llm_calls (cached=false) + UPDATE runs.spent_usd ── one transaction ──▶ COMMIT
  │
  └─ schema(json.loads(text)) ──────────────────── raises ──▶ repair prompt, ≤2 attempts
```

The cache key is `input_hash(step, payload, model, params)` — a SHA-256 over canonical JSON
(`sort_keys=True`, tight separators), so it is stable across processes, machines and days. Key
order and whitespace do not change it; a different model, a different `max_tokens`, or a
different **prompt** does. The prompt is in the key because it is the input that most decides
the answer: without it, editing an instruction and restarting an in-flight run serves the old
instruction's answer back, silently. Which round of the repair loop a call belongs to is in
the key for the same reason — a malformed answer and the repair bought to replace it are two
different calls, and filing them under one key means a resumed run replays the answer already
known to be unusable and re-buys the repair.

Lookups are scoped to the run, because idempotency here means "do not pay twice for this run's
work", not "share a global response cache" — the latter is a different feature with different
correctness and privacy properties, and it is not this one.

The budget check happens **before** the network call, and the ledger write happens **before** the
next one. Both orderings are the point.

### 3. The failure taxonomy

Most production LLM code contains some version of this:

```python
try:
    result = call_the_model(prompt)
except Exception:
    retry()
```

That `except` is the most expensive line in the file, because "the call failed" is not one
situation. It is at least five, and they want opposite responses. Flattening them into one
handler is the most common production mistake in this space, and it has two signature failure
modes: a refusal loop that quietly spends a month's budget in an afternoon, and a
malformed-output loop that retries an unchanged prompt until it gives up.

Retrying a **policy refusal** is the purest way to burn money: every attempt is billed, every
attempt is refused, and the loop only terminates when the budget does. The model is not going to
change its mind because you asked a fourth time. That is a routing decision — a human needs to
see it — not an error to paper over.

A **malformed response** is the opposite trap. You were billed, you got an answer, and it did not
satisfy your schema. A plain retry sends the identical prompt and tends to get an
identically-shaped failure, so you pay twice for the same mistake. What works is a *repair*
attempt: feed the validation error back into the prompt so the model can correct itself. Two
attempts, then stop, because a model that has failed the same schema twice is usually being
asked for something the schema does not permit.

A **transient** failure — 429, 5xx, timeout, reset connection — is the only one where a plain
retry is correct, and it is the only one that is free, because a failed call was never billed.
Backoff needs jitter: an unjittered retry synchronises every worker that got the same 429 onto
the same instant and reproduces the overload that caused it.

The policy lives in `errors.py` as **data**, not as `if` statements, and both the worker
(run-level retries) and the context (in-step retries) read the same table — so the two can never
drift apart, and the table can be asserted against this document in a three-line test:

| class       | retry? | strategy                     | max attempts | terminal run status |
|-------------|--------|------------------------------|--------------|---------------------|
| `transient` | yes    | exponential backoff + jitter | 5            | `failed`            |
| `malformed` | yes    | repair prompt (in-step)      | 2            | `pending_review`    |
| `policy`    | no     | —                            | 0            | `pending_review`    |
| `budget`    | no     | —                            | 0            | `halted_budget`     |
| `permanent` | no     | —                            | 0            | `failed`            |

Note that the three non-retryable classes end in three *different* terminal statuses.
`pending_review` is not `failed`: a refusal is a run that needs a human, not a run that broke.
`halted_budget` is not `failed` either: "we stopped you" should never be confused with "it broke"
when you are reading a dashboard at 2am. And anything llmrun does not recognise — a `KeyError`, a
typo in a model name, any bug in your own pipeline code — classifies as `permanent`. The
conservative direction is deliberate: mislabelling a real bug as transient buys you five slow,
billed repetitions of the same crash, while mislabelling a blip as permanent buys you one honest
failure that a human can requeue.

`ctx.escalate("reason")` raises a policy refusal from your own code, so "this input needs a
human" is a first-class outcome rather than an exception you invent.

### 4. Cost accounting — LLM cost tracking per tenant

Every billed call writes a row: model, tokens in, tokens out, cost, and the response. Nothing is
ever updated or deleted, so `SUM(cost_usd)` is the authoritative answer to "what did this run
cost" rather than a derived guess. Money is `decimal.Decimal` end to end, quantised to six
decimal places with `ROUND_HALF_UP` to match the `numeric(12,6)` columns — never `float`. Not
pedantry: those costs are summed thousands of times into an accumulator that a budget is then
compared against, and binary rounding error in that accumulator does not stay small.

Because every row carries `run_id` and every run carries `tenant_id`, attributing spend to a
tenant is a `GROUP BY` rather than a project:

<sub>*Illustrative output. See the warning about the price table below — those figures are
placeholders, not a quote.*</sub>

```console
$ llmrun costs --tenant acme

COSTS  (tenant: acme)
============================================================================================
  PIPELINE          MODEL             RUNS  BILLED  REPLAYED     TOK IN  TOK OUT  SPEND USD  SAVED USD
  ----------------  ----------------  ----  ------  --------  ---------  -------  ---------  ---------
  summarise_thread  claude-sonnet-5    142   1,562       118  4,912,004  611,233  23.904507   1.806133
  enrich_account    claude-haiku-4-5    89     712        31  1,204,880   96,417   1.686965   0.073451
  ----------------  ----------------  ----  ------  --------  ---------  -------  ---------  ---------
  TOTAL                                231   2,274       149  6,116,884  707,650  25.591472   1.879584

  149 replayed call(s) cost nothing: $1.879584 that a retry would have spent again.
```

`REPLAYED` is the retry absorption rate and `SAVED USD` prices it. Those are the numbers that
tell you whether your crash-and-resume story is working, and they are the numbers nobody has
when the state lives in a stack frame. Note that `TOK IN`/`TOK OUT` count only *billed* calls —
a replay copies the original's token counts so the saving can be priced, and adding those into
the totals would report tokens nobody bought.

### 5. Budget caps that halt

A budget is per run and set at the pipeline: `@pipeline(budget_usd="2.50")`, `NULL` for
unlimited. Before every call, llmrun projects the worst case — input tokens estimated from
prompt length, output assumed to be the full `max_tokens` — and takes `SELECT budget_usd,
spent_usd ... FOR UPDATE` on the run. If `spent + projected > budget`, it raises before the
network call, and the run terminates as `halted_budget`.

Checking *only* after the call would make the cap a report of an overrun rather than a
prevention of one. But checking only before it has a gap worth naming: the output side is
genuinely bounded by `max_tokens`, while the input side is an estimate — `prompt_chars / 4`,
which under-counts CJK, Cyrillic, base64 and minified JSON — and the provider can report a
pricier model than the one that was projected. So the recorded cost is compared against the
cap again *after* the call, in the same transaction as the ledger row, and the run halts
before the next one. The receipt is always committed, even when it is the call that crossed
the line: a cap that loses the record of what it failed to stop is worse than one that
reports it a call late.

**So the honest guarantee is bounded, not absolute.** llmrun stops a run one call after it
crosses its cap, not zero. Making that zero requires a real tokeniser on the input side —
worth doing, not done here. And a `budget` failure is non-retryable by construction, because
retrying past a cap is precisely the behaviour the cap exists to stop.

---

## Writing a pipeline

The whole API is a decorator and three methods on the context. There is no DSL, no graph object
and no builder — a pipeline is a Python function, and it stays callable as one. `load_thread` and
`Facts` below are your own code; llmrun has no opinion about either.

```python
from __future__ import annotations

from llmrun import RunContext, pipeline

EXTRACT = "Extract every decision and its owner. Reply with JSON only."


@pipeline(name="summarise_thread", budget_usd="2.50", max_attempts=5)
def summarise_thread(ctx: RunContext, thread_id: str) -> dict:
    # A plain function call, checkpointed. Runs once per run, even across crashes.
    thread = ctx.step("fetch_thread", load_thread, thread_id)

    if thread["visibility"] == "restricted":
        # A first-class outcome, not an exception you invented: the run ends
        # `pending_review` and a human sees it.
        ctx.escalate(f"thread {thread_id} is restricted")

    # `schema` is any callable that takes decoded JSON and raises on invalid input.
    # Pydantic users pass Model.model_validate; there is no pydantic dependency.
    facts = ctx.llm(
        "extract_facts",
        prompt=EXTRACT,
        payload=thread["messages"],
        schema=Facts.model_validate,
        max_tokens=800,
    )

    summary = ctx.llm("write_summary", prompt="Five bullets, no preamble.", payload=facts)
    return {"thread_id": thread_id, "summary": summary, "spent_usd": str(ctx.spent_usd)}
```

Queue a run from Python, or from the shell:

```python
run_id = summarise_thread.enqueue(tenant_id="acme", thread_id="t-42")
```

```bash
llmrun enqueue summarise_thread --tenant acme \
    --json '{"thread_id": "t-42"}' \
    --import examples/summarise_thread.py

llmrun worker --import examples/summarise_thread.py     # claims it and runs it
```

`examples/summarise_thread.py` in this repo is the twelve-step version the demo and the chaos
suite use — long enough that a mid-run kill is visibly expensive.

---

## How it works

Four tables. If a fifth were needed, the scope crept.

| table | one row per | why it exists |
|---|---|---|
| `runs` | pipeline execution | the unit of work a worker claims; holds the budget, the running spend, the terminal status and the lease |
| `steps` | named step of a run | the resumption checkpoint: stored output plus the `input_hash` that proves the output still matches its inputs |
| `llm_calls` | provider call *and* replay | the cost ledger and the idempotency store, at once |
| `schema_migrations` | applied migration | plain filename bookkeeping so `llmrun migrate` is idempotent |

**Why `llm_calls` is both ledger and idempotency store.** These are usually built as two systems:
a spend table for finance, and a response cache keyed by request hash for retries. Keeping them
as one table removes the failure mode where they disagree. A row with `cached=false` is a real,
billed call — it carries the true cost and the stored response, and it is written in the same
transaction as the spend it caused. A row with `cached=true` is a replay: it costs zero, and it
exists purely to record that a retry was *absorbed* rather than paid for a second time. So "what
did this cost" and "have I already made this call" are answered by the same append-only rows,
and there is no reconciliation job, no cache that can be warm while the ledger is cold, and no
way to serve a cached response without leaving evidence that you did.

**How work is claimed.** One SQL statement, no broker, no scheduler process:

```sql
UPDATE runs SET status='running', locked_by=%s, locked_at=now(), attempt=attempt+1
WHERE id = (
    SELECT id FROM runs
    WHERE attempt < max_attempts
      AND (status='queued'
           OR (status='running' AND locked_at < now() - make_interval(secs => %s)))
    ORDER BY created_at
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
RETURNING *;
```

The `status='running'` clause is crash recovery: a run whose worker was `SIGKILL`ed sits in
`running` with a stale `locked_at`, and the next worker to come along steals it. Nothing else in
the system notices that a process died, and nothing needs to. `FOR UPDATE SKIP LOCKED` is what
makes concurrent *claims* safe — each worker's `SELECT` locks the row it takes and other workers
skip it rather than blocking behind it. Without `SKIP LOCKED`, ten workers serialise on the
head-of-queue row and your pool has an effective concurrency of one.

`attempt < max_attempts` is the ceiling on executions, and it has to live here: a SIGKILLed
worker never reaches the code that writes a verdict, so a run that reliably kills its worker
would otherwise be re-claimed forever. A sweep run before each claim moves any run past the
ceiling whose lease has also expired to `failed`, rather than leaving it stranded in `running`
where it reads as work in progress.

Note what the lease is *not*: it is not proof the first worker is dead, only that it has been
quiet for longer than the lease. `SKIP LOCKED` stops two workers claiming the same row in the
same instant; it says nothing about a lease that expires while its holder is still working,
which is the ordinary case. Two things handle that. Every billed call re-reads `runs.locked_by`
under the row lock it is about to spend behind, and abandons the run if the lease has moved on;
and it re-reads the `llm_calls` ledger under that same lock, so a worker that queued behind
another one replays its answer instead of buying a second copy. Neither is a substitute for the
checkpoints and the ledger — they are what make an overlap cost nothing rather than cost double.

Every SQL statement in the codebase is parameterised. There is no f-string SQL anywhere.

---

## What this is not

Reproduced verbatim from [`docs/DESIGN.md`](docs/DESIGN.md), which is the binding interface
contract for this repository:

> - Not an agent framework, no orchestration DSL, no prompt management
> - No web UI (CLI only)
> - Exactly two provider adapters: one real (Anthropic), one deterministic fake (echo)
> - Four tables. If a fifth is needed, the scope crept.

### Why not just use LangChain?

Because they are not competitors, and a fair reading of both makes that obvious. LangChain,
LlamaIndex and the rest are **composition** tools: prompt templates, retrievers, tool loops,
output parsers, a common interface across model vendors. llmrun composes nothing. It has no
prompt management, no agent loop, no vector anything, and no opinion whatsoever about what your
prompts say. It makes composition **durable**: checkpoint, ledger, budget, taxonomy.

They stack cleanly. Call a LangChain chain inside `ctx.step("classify", chain.invoke, doc)` and
it becomes resumable and checkpointed without llmrun knowing what a chain is. The only surface
that overlaps is `ctx.llm()`, and you are free not to use it — you lose per-call idempotency and
cost attribution for those calls and keep everything else.

The more honest comparison is with durable execution engines: Temporal, Restate, DBOS. Those are
better than this at the general problem, and if you already run one, use it — the part of this
repository worth stealing is then the failure taxonomy in `errors.py` and the dual-purpose
ledger, both of which are LLM-specific and neither of which a general workflow engine gives you.
llmrun exists for teams who want crash-safe LLM pipelines without adding a workflow cluster to
their runbook, and who already have a Postgres.

---

## Requirements

- Python 3.11 or newer
- Postgres 13 or newer (the schema uses built-in `gen_random_uuid()`, so `pgcrypto` is not
  required). `docker-compose.yml` ships Postgres 17 on host port **55432** to avoid colliding
  with a local 5432.
- Docker and [uv](https://docs.astral.sh/uv/) for the `make` targets. Neither is required to use
  the library — they are how the demo and tests get a database.

The core has **zero provider dependencies**. `psycopg[binary]` is the only runtime dependency:
no ORM, no migration framework, no HTTP client, no model SDK. `anthropic` is an optional extra.

## Install

v0.1 is not on PyPI yet. Install from a checkout:

```bash
git clone https://github.com/asadimtiaz/llmrun.git
cd llmrun

uv sync                    # llmrun + psycopg, plus pytest and ruff (the `dev` group)
uv sync --extra anthropic  # add the real provider adapter
uv sync --no-dev           # runtime only, as a consumer would install it
```

Then apply the schema (idempotent — re-running is a no-op):

```bash
llmrun migrate
```

## Configuration

Everything is environment variables. Copy `.env.example` to `.env`; the Makefile loads it
automatically, and the library itself only ever reads real environment variables, so outside of
`make` you export them yourself.

| variable | default | what it does |
|---|---|---|
| `LLMRUN_DATABASE_URL` | `postgresql://llmrun:llmrun@localhost:55432/llmrun` | connection string for every command and for the test suite |
| `LLMRUN_PROVIDER` | `echo` | default adapter when `--provider` is not passed |
| `ANTHROPIC_API_KEY` | *(empty)* | only read when the Anthropic adapter is active |

Nothing in the default demo, test or chaos paths touches the network.

## Running against Anthropic instead of echo

```bash
uv sync --extra anthropic
export ANTHROPIC_API_KEY=sk-ant-...
llmrun worker --provider anthropic --import examples/summarise_thread.py
```

The adapter defaults to `claude-sonnet-5`, reads token counts from `response.usage`, and maps SDK
failures onto the taxonomy: rate limits, connection errors, timeouts and 5xx become `transient`;
a refusal stop reason becomes `policy`; everything else becomes `permanent`. It deliberately does
**not** retry internally — retry policy belongs to one place, and that place is the disposition
table.

It also bounds its own wait: **requests time out after 45 seconds** unless you pass `timeout=`,
rather than inheriting the SDK's ten-minute read timeout. That is a database number before it is
a latency one. The request is issued inside the transaction holding the run row's lock, so a
request the server never answers would pin that lock — and a pooled connection — for ten minutes,
nine of them after the 60-second lease had already expired. A timeout classifies as `transient`,
so hitting it hands the call straight to the retry policy with every attempt written down, which
beats stalling behind a lock. A step that genuinely needs longer needs a longer lease too: raise
`timeout=` and `--lease-seconds` together, not either alone.

Request options are an allowlist, not a passthrough: `metadata`, `output_config`,
`stop_sequences` and `thinking` are forwarded, and anything else is dropped and **logged at
WARNING**, naming the parameter that was dropped. A dropped option is your request quietly not
happening — you still hashed it into the ledger key and still paid for the call — so it is worth
a line at the level the CLI actually ships at. The one exception is `step`, which the framework
injects into every single call for the echo provider's benefit: that one is expected and drops
silently, because a warning on every call is a warning nobody reads.

The dropped set includes `temperature`, `top_p` and `top_k`, which current models reject with a
400 — which this adapter would classify, correctly and unhelpfully, as a permanent failure of
your run. They still change the cache key, so two calls that differ only by a dropped parameter
remain two different ledger entries.

**Check the prices before you trust the numbers.** `cost.py` ships a placeholder price table so
the project runs out of the box. Those figures are illustrative, are not maintained in lockstep
with any vendor, and vendor pricing changes without notice. Before using llmrun's cost output for
billing, budgeting or chargeback, check each model against the provider's current published
pricing and override what you find:

```python
from decimal import Decimal
from llmrun.cost import Price, register_price

register_price("claude-sonnet-5", Price(Decimal("3"), Decimal("15")))  # USD per 1M tokens
```

An unknown model raises `KeyError` rather than defaulting to zero. A silent zero would be the
worst possible failure here: the run would look free, the budget check would always pass, and the
discrepancy would surface as an invoice.

## CLI

| command | what it does |
|---|---|
| `llmrun migrate` | apply `migrations/*.sql` in filename order, idempotently |
| `llmrun worker [--once] [--import M] [--provider echo\|anthropic] [--lease-seconds N]` | claim and execute runs; `SIGTERM` finishes the current run and exits |
| `llmrun enqueue PIPELINE --tenant T --json '{"thread_id": "t1"}' [--import M]` | queue a run, print its id |
| `llmrun runs list [--tenant T] [--status S] [--limit N]` | recent runs and their statuses |
| `llmrun runs show RUN_ID` | run header, step table, per-call cost ledger with the `cached` column; `RUN_ID` may be any unambiguous prefix |
| `llmrun costs [--tenant T]` | spend grouped by pipeline and model, including cache savings |

Pipelines are registered by importing the module that defines them, which is what `--import`
does; it accepts a dotted path or a file path.

## Status

**v0.1.** CI runs lint plus the *full* suite on every push — including the crash test, against a
real Postgres service container, with no `-m "not chaos"` escape hatch. That is deliberate: the
no-double-billing assertion is the claim this repository exists to make, so it is not allowed to
be optional. What is not built yet, stated plainly:

- **Not published to PyPI.** Install from a checkout.
- **Two provider adapters only**, by design. No OpenAI, Gemini, Bedrock or Vertex adapter. The
  `Provider` protocol is one method, so a third-party adapter is a small file — but it will not
  live in this repository.
- **Budgets are per run**, not per tenant per month. Aggregate spend across runs is
  reporting-only today (`llmrun costs`); there is no rolling tenant cap that refuses to enqueue.
- **Steps within a run are sequential.** Concurrency comes from running multiple workers across
  runs. There is no fan-out primitive.
- **Leases are not renewed mid-run.** A single step that legitimately runs longer than
  `--lease-seconds` can have its run re-claimed by another worker while it is still working.
  That overlap does not double-bill — the ledger is re-read under the run's row lock, and the
  displaced worker stops when it sees the lease has moved — but the in-flight step may still be
  *executed* twice, so any side effect outside Postgres happens twice. Set the lease above your
  slowest step.
- **No metrics exporter, no tracing, no web UI.** Levelled logging under `llmrun.*` loggers —
  plain human-readable lines to stderr, with the run id written into the message text so one
  run can be grepped out of a busy worker's output — and SQL. Not structured: no JSON, no
  key/value fields, and the timestamps carry local time only, so a log pipeline has nothing to
  index and correlating across days means adding your own handler. That is the whole
  observability story at v0.1.
- **The price table is a placeholder.** See above.
- **Python and Postgres only.** No SQLite fallback, no MySQL.

Interfaces may still change before v1. `docs/DESIGN.md` is the contract; if the code and that
document ever disagree, the document wins and the code is wrong.

## Contributing

Issues and pull requests are welcome. `make lint` and `make test` need to pass, plus `make chaos`
for anything touching the worker, the context or the ledger. New behaviour needs a test that
fails without it. If a change requires a fifth table, open an issue first — that is a design
conversation, not a patch.

## Licence

MIT. See [LICENSE](LICENSE).

**Dependency licences.** The MIT header covers llmrun's own code and nothing else — the licence
that governs a *distribution* is the union of everything shipped inside it. `psycopg` and
`psycopg-binary` — the two distributions behind `psycopg[binary]`, the only *required* runtime
dependency — are **LGPL-3.0-only**, so anyone who redistributes llmrun with them (a wheel that
vendors them, a container image, an installer) inherits the LGPL obligations for those parts:
keep them replaceable, and make their source available on request.
Importing them at run time from a normal `uv sync` is the ordinary dynamic-linking case and is
why this is a note rather than a warning. The optional `anthropic` extra pulls a wider but
permissive closure of its own; see [NOTICE](NOTICE) at the repo root for the
dependency-by-dependency picture, including the native libraries inside the binary wheel. None
of this is legal advice; if you ship binaries, ask someone whose job that is.

---

Written by Asad Imtiaz. I build this kind of plumbing — durability, cost control and failure
handling around LLM features — for B2B SaaS teams whose model spend became a line item before
anyone had planned for it. If the failure at the top of this page is one you have actually had,
I am happy to talk about it either way —
[open an issue](https://github.com/asadimtiaz/llmrun/issues), or reach me at the address on my
GitHub profile.

<!-- BEFORE PUBLISHING: replace the line above with a direct contact — an email address or a
     LinkedIn URL — and put the same address in pyproject.toml's `authors` entry so it survives
     into the package metadata. "Reach me via my profile" is one hop more than a reader who has
     just finished 600 lines should have to make. Also verify the clone URL used throughout
     this file (`github.com/asadimtiaz/llmrun`) actually resolves; it is currently a
     placeholder, and `git ls-remote` on it returns "Repository not found". -->
