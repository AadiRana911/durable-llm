"""The stability promise: what ``from llmrun import *`` gives you, and what it costs.

``llmrun/__init__.py``'s docstring says "what is exported here is the whole public API" and
then lists fifteen names. Nothing checked that sentence, so the surface this package promises
not to break could be widened by an errant import or narrowed by a rename, and the first
report would come from a downstream ``ImportError`` rather than from CI.

Two properties, and they fail in opposite directions:

* **The names.** :data:`GOLDEN_API` below is a copy of ``__all__`` written down on purpose.
  Duplicating it is the entire point — a test that asserted ``llmrun.__all__ == llmrun.__all__``
  would pass for any value of it. Changing the exported surface now means editing this list,
  which is a deliberate act somebody has to justify in a diff, rather than a side effect.

* **The cost of importing it.** ``__init__`` exports :func:`~llmrun.providers.base.get_provider`
  and :class:`~llmrun.providers.base.Provider` from the package facade, and the docstring
  claims that costs nothing because the ``anthropic`` SDK is imported inside
  ``get_provider``'s body. That claim is load-bearing: it is why ``pip install llmrun`` with
  no extras works, why the suite runs with no API key, and why ``import llmrun`` in a Lambda
  does not pay for an SDK nobody called. Nothing guarded it, and it is exactly the kind of
  property a tidy-up commit ("move the imports to the top") deletes without noticing.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import llmrun

#: Every name ``llmrun`` promises. Kept sorted because ``__all__`` is, and compared as a
#: list rather than a set so a duplicated entry is a failure too.
#:
#: Adding a name here is adding it to the stability promise. Removing one is a breaking
#: change for anyone who imported it. Neither should ever be an accident, which is why this
#: list is hand-written rather than derived.
GOLDEN_API: list[str] = [
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

#: Run in a *fresh* interpreter rather than asserting on this process's ``sys.modules``.
#:
#: In-process the assertion would be "no test that happened to run before this one imported
#: the SDK", which is a fact about pytest's collection order and not about llmrun. It would
#: pass today, pass for the wrong reason tomorrow, and fail on ``-p no:randomly`` being
#: dropped. A subprocess asks the actual question — "does importing llmrun, and nothing else,
#: pull in anthropic?" — and answers it the same way every time, on any machine, whatever
#: else the suite is doing. It is the one place in this file worth ~200ms.
_LAZY_PROVIDER_PROBE = textwrap.dedent(
    """
    import sys

    import llmrun

    # Sanity: the export the claim is *about* is present, so a future refactor that deletes
    # get_provider cannot make this probe pass by making it vacuous.
    assert llmrun.get_provider is not None
    print("anthropic" in sys.modules)
    """
)


def test_the_exported_api_is_exactly_the_golden_list() -> None:
    """Pin the stability surface so widening or narrowing it has to be deliberate.

    Delete this and ``llmrun.__all__`` becomes whatever the last import statement left
    behind. Both directions are real: a helper imported for internal convenience and added
    to ``__all__`` "while we are here" becomes a name the project owes compatibility to
    forever, and a rename that misses ``__all__`` silently removes an export that downstream
    code imports by name. Neither shows up in any other test, because every other test in
    this suite imports from the *module* (``llmrun.pipeline``, ``llmrun.errors``) rather than
    from the package facade — so the facade could be empty and the suite would still be
    green.
    """
    assert list(llmrun.__all__) == GOLDEN_API


def test_every_exported_name_actually_resolves() -> None:
    """``__all__`` is a list of strings; nothing makes them true.

    A name in ``__all__`` with no matching attribute is not a syntax error, not an import
    error, and not visible until somebody writes ``from llmrun import X`` and gets an
    ``ImportError`` — or worse, ``from llmrun import *``, which raises ``AttributeError`` at
    import time and takes their whole application down. The previous test pins the *list*;
    this one pins that the list is not fiction.
    """
    missing = [name for name in llmrun.__all__ if not hasattr(llmrun, name)]
    assert not missing, f"names promised by llmrun.__all__ that do not exist: {missing}"


def test_importing_llmrun_does_not_import_the_optional_anthropic_sdk() -> None:
    """The lazy provider import, which is a design property and not an implementation detail.

    ``providers/base.py`` imports the SDK inside :func:`get_provider`'s body, with a comment
    saying why; ``__init__.py`` repeats the promise; ``pyproject.toml`` makes ``anthropic`` an
    optional extra on the strength of it. Move that import to module scope and every one of
    those statements becomes false at once — ``import llmrun`` starts paying for an HTTP
    client and a large SDK, and a core install with the extra uninstalled stops importing at
    all. Nothing else in the suite would notice: ``tests/test_providers.py`` patches
    ``_import_sdk`` precisely so it can exercise the adapter without the SDK, so it proves
    the adapter is tolerant, not that the core is clean.

    The SDK *is* installed in this environment (it is in the dev lock), so the assertion has
    something real to catch rather than passing because the import could not have succeeded.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _LAZY_PROVIDER_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        f"the probe interpreter failed to import llmrun at all:\n{completed.stderr}"
    )
    assert completed.stdout.strip() == "False", (
        "importing llmrun pulled the optional `anthropic` SDK into sys.modules. The lazy "
        "import inside get_provider is what makes `pip install llmrun` provider-free and "
        "what lets this suite run with no API key — put the import back inside the function."
    )


def test_the_probe_would_notice_an_eager_sdk_import() -> None:
    """Guard the guard: prove the subprocess above can actually report ``True``.

    The probe's whole value is that it distinguishes "anthropic was not imported" from
    "anthropic could not have been imported". If the SDK were absent from the environment —
    a lockfile change, a slimmer CI image — the previous test would print ``False`` forever
    and assert nothing, which is the failure mode this file exists to avoid in the first
    place. So import the SDK explicitly and confirm the same probe flips.
    """
    pytest.importorskip("anthropic", reason="the lazy-import guard has nothing to detect")

    completed = subprocess.run(
        [sys.executable, "-c", "import anthropic\n" + _LAZY_PROVIDER_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "True", (
        "the probe reported False even with `import anthropic` executed first, so it cannot "
        "detect an eager import and the test above proves nothing"
    )
