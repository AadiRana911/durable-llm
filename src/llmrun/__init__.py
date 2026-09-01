"""llmrun — multi-step LLM pipelines that survive crashes and cannot silently overspend.

All state lives in Postgres: four tables, no broker, no scheduler, no new infrastructure.
A pipeline is an ordinary Python function whose steps are checkpointed, whose provider calls
are recorded in an idempotency ledger before they can be repeated, and whose spend is
checked *before* each call rather than reconciled afterwards.

    from llmrun import pipeline, RunContext

    @pipeline(name="summarise_thread", budget_usd="2.50")
    def summarise_thread(ctx: RunContext, thread_id: str) -> dict:
        thread = ctx.step("fetch", load_thread, thread_id)
        points = ctx.llm("extract", "List the decisions.", thread, schema=Decisions.model_validate)
        return {"summary": ctx.llm("write", "Write a summary.", points)}

Then::

    llmrun migrate
    llmrun enqueue summarise_thread --tenant acme --json '{"thread_id": "t1"}'
    llmrun worker
    llmrun runs show <run-id>

Kill the worker at any point and start it again: completed steps replay from the ``steps``
table and completed provider calls replay from the ``llm_calls`` ledger, so the retry
re-executes the pipeline without re-buying a single token. ``llmrun runs show`` prints the
per-call ledger with the replays marked, which is the claim made auditable.

What is exported here is the whole public API. Everything else — the worker loop, the SQL
migration runner, the registry, the CLI — is reachable at ``llmrun.<module>`` for the people
who need it, but is not part of the promise:

* :func:`~llmrun.pipeline.pipeline` — the decorator that defines and registers a pipeline.
* :class:`~llmrun.context.RunContext` — the handle a pipeline body is given: ``ctx.step()``,
  ``ctx.llm()``, ``ctx.escalate()``.
* :class:`~llmrun.types.LLMResult` — one provider completion, with its token counts.
* :class:`~llmrun.errors.ErrorClass` and the five error types — the failure taxonomy that
  decides whether something is retried, repaired, escalated, or halted.
* :class:`~llmrun.providers.base.Provider` and :func:`~llmrun.providers.base.get_provider` —
  the seam you implement to talk to a model llmrun has never heard of.
* :class:`~llmrun.cost.Price` and :func:`~llmrun.cost.register_price` — how you correct the
  shipped price table, which is placeholder data and *will* be wrong for your account.

The last four are exported rather than left at ``llmrun.providers`` / ``llmrun.cost``
because both are things a working installation routinely needs: a custom provider and a
corrected price are ordinary configuration, not internals, and a stability promise that
excluded them would be a promise about the parts nobody has to touch. Naming them here is
free — :func:`~llmrun.providers.base.get_provider` imports the ``anthropic`` SDK inside the
function body, so ``import llmrun`` still pulls in no provider SDK on a core install.
"""

from __future__ import annotations

from llmrun.context import RunContext
from llmrun.cost import Price, register_price
from llmrun.errors import (
    BudgetExceededError,
    ErrorClass,
    LLMRunError,
    MalformedOutputError,
    PermanentError,
    PolicyRefusalError,
    TransientError,
)
from llmrun.pipeline import Pipeline, pipeline

# Imported from the package facade rather than from `llmrun.providers.base` so this line
# keeps working if the seam ever moves modules. It costs nothing at import time: the
# subpackage's own __init__ deliberately touches no provider SDK, and the `anthropic`
# import lives inside get_provider's body. `import llmrun` on a core install therefore
# still leaves "anthropic" absent from sys.modules, which is the property that lets someone
# clone this repo with no API key and run the whole suite.
from llmrun.providers import Provider, get_provider
from llmrun.types import LLMResult

__version__ = "0.1.0"

__all__ = [
    "BudgetExceededError",
    "ErrorClass",
    "LLMResult",
    "LLMRunError",
    "MalformedOutputError",
    "PermanentError",
    "Pipeline",
    "PolicyRefusalError",
    "Price",
    "Provider",
    "RunContext",
    "TransientError",
    "__version__",
    "get_provider",
    "pipeline",
    "register_price",
]
