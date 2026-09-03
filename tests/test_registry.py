"""Name → callable, and the three ways that mapping can lie to you.

``runs.pipeline`` is a string. An hour after a run is enqueued, a worker on a different
machine has to turn that string back into the function that was meant to execute it, and
everything durable_llm promises about correctness is downstream of getting that right. Three
properties, one section each:

* **The conflict.** Two different functions registered under one name is the failure nobody
  sees: the run is claimed, the pipeline is found, the worker reports success, and the output
  came from code the enqueuer never asked for. ``register`` raises instead — documented as
  the thing preventing exactly that, and, until this file, never tested.

* **The miss.** An unknown name is almost never "that pipeline does not exist"; it is "this
  process never imported the module that defines it". The message has to say so, because the
  operator reading it is one ``--import`` away from a working worker.

* **The loader.** ``--import`` accepts a dotted module and a file path, has to be idempotent
  in both forms (a module reachable two ways would otherwise trip the conflict check against
  itself), and must not leave a half-executed module in ``sys.modules`` pretending to be
  importable.

Every test here runs against an empty registry and puts the real one back afterwards. The
registry is process-global module state populated at *import* time — ``tests/test_worker.py``
and ``tests/test_pipeline.py`` register pipelines when pytest collects them — so a bare
``registry.clear()`` in this file would delete other files' fixtures and fail them with
"no pipeline named ... is registered", minutes later, in an unrelated test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from durable_llm import registry
from durable_llm.pipeline import Pipeline

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Source for a throwaway module that registers exactly one pipeline when executed.
#:
#: Written to disk per-test rather than kept as a real file in ``tests/`` because the loader's
#: two interesting behaviours — dotted-vs-path dispatch and idempotency — are about *how* a
#: module is reached, and a module pytest has already imported for its own reasons would have
#: been reached before the test started.
_PIPELINE_MODULE = '''\
"""Throwaway module for tests/test_registry.py. Registers one pipeline on import."""

from durable_llm.pipeline import pipeline


@pipeline(name={name!r})
def probe(ctx, value):
    """A pipeline body nothing ever executes; only its registration is under test."""
    return {{"value": value}}
'''

#: A module that raises while executing. The failure is deliberately *not* an ImportError,
#: because the realistic case is a pipeline module whose top level does real work — reads a
#: config file, builds a client, validates an environment variable — and blows up on it.
_EXPLODING_MODULE = '''\
"""Throwaway module for tests/test_registry.py. Raises while executing, on purpose."""

raise RuntimeError("this module refuses to import")
'''


# --------------------------------------------------------------------------------------
# isolation
# --------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_registry() -> Iterator[None]:
    """Run each test against an empty registry, then restore the process-wide one.

    Reaching into ``registry._REGISTRY`` is deliberate and is the only way to do this: the
    module's public surface can ``clear()`` but cannot snapshot, and every other test file in
    this suite registers its pipelines at import time. Without the restore, the first test
    here would silently uninstall ``test_worker_flaky``, ``summarise_thread`` and the rest for
    the remainder of the session.
    """
    saved = dict(registry._REGISTRY)
    registry.clear()
    try:
        yield
    finally:
        registry.clear()
        registry._REGISTRY.update(saved)


@pytest.fixture(autouse=True)
def isolated_import_state() -> Iterator[None]:
    """Put ``sys.modules`` and ``sys.path`` back exactly as they were.

    ``registry._load_from_file`` writes a ``sys.modules`` entry and prepends the file's
    directory to ``sys.path`` — permanently, by design, so a single-file pipeline can import
    the helpers sitting next to it. Both are global. A leaked ``sys.path`` entry pointing at a
    deleted ``tmp_path`` changes how *every* later import in the session resolves, and a
    leaked ``sys.modules`` entry would make this file's own idempotency assertions pass for
    the wrong reason on a re-run.
    """
    modules = dict(sys.modules)
    path = list(sys.path)
    try:
        yield
    finally:
        for name in set(sys.modules) - set(modules):
            del sys.modules[name]
        sys.modules.update(modules)
        sys.path[:] = path


def _write_module(directory: Path, stem: str, *, pipeline_name: str) -> Path:
    """Drop a pipeline-defining module on disk and return its path."""
    path = directory / f"{stem}.py"
    path.write_text(_PIPELINE_MODULE.format(name=pipeline_name), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# register: the duplicate-name conflict
# --------------------------------------------------------------------------------------


def _named(label: str) -> Pipeline:
    """Build an *unregistered* pipeline whose function is identifiable in a message.

    Constructed straight from :class:`Pipeline` rather than through ``@pipeline``, because
    the decorator registers as a side effect and this section needs to control exactly when
    and what ``register`` is called with.
    """

    def body(ctx: Any) -> dict[str, str]:
        return {"label": label}

    body.__qualname__ = f"pipeline_{label}"
    return Pipeline(body, name="collides")


def test_re_registering_the_same_object_is_a_no_op() -> None:
    """A module reached twice must not be punished for it.

    ``--import myapp.pipelines`` and ``--import ./myapp/pipelines.py`` are the same file by
    two routes, and a deployment that names both — or a conftest that imports a module the
    CLI then loads — hits ``register`` twice with the identical object. Raising there would
    make ``--import`` a minefield for no safety gain, because the second registration is
    asking for precisely what the first one already installed.

    Delete this and the tolerance is free to disappear: turning the ``existing is pipeline``
    branch into an unconditional raise would still satisfy the conflict test below, and the
    loader idempotency tests further down would start failing with an error message about the
    wrong problem.
    """
    first = _named("alpha")

    registry.register("collides", first)
    registry.register("collides", first)

    assert registry.get("collides") is first
    assert registry.names() == ["collides"], "a no-op registration created a second entry"


def test_two_different_pipelines_under_one_name_are_refused_and_both_are_named() -> None:
    """The conflict that would otherwise execute the wrong function for a queued run.

    ``runs.pipeline`` stores only a name. If the second registration silently won, a run
    enqueued against one function would be executed by another — no exception, no wrong
    status, just wrong output, discovered (if ever) by whoever reads it. That is the single
    worst failure this module can produce, and the guard against it lives in four lines of
    ``register``.

    The message has to name *both* sides. "already registered" alone tells the operator that
    two things collided and nothing about which two, and the usual cause is a copied-and-
    pasted pipeline that kept the original's function name — so ``module.qualname`` for each
    is the difference between a five-second fix and a grep.
    """
    first = _named("alpha")
    second = _named("beta")
    registry.register("collides", first)

    with pytest.raises(ValueError) as caught:
        registry.register("collides", second)

    message = str(caught.value)
    assert "collides" in message
    assert "pipeline_alpha" in message, "the conflict message does not identify the incumbent"
    assert "pipeline_beta" in message, "the conflict message does not identify the newcomer"
    assert registry.get("collides") is first, "the loser of a conflict overwrote the winner"


def test_an_empty_name_is_refused() -> None:
    """A pipeline with no name is a row nothing can ever look up.

    ``@pipeline`` derives the name from ``fn.__name__`` when none is given, and a callable
    without one (a ``functools.partial``, a class instance with ``__call__``) yields the empty
    string. Registering it would put an unreachable entry in the map and enqueue runs whose
    ``pipeline`` column is ``''`` — permanently unclaimable by any worker.
    """
    with pytest.raises(ValueError, match="non-empty"):
        registry.register("", _named("alpha"))


# --------------------------------------------------------------------------------------
# get: the miss
# --------------------------------------------------------------------------------------


def test_an_unknown_name_explains_import_rather_than_just_failing() -> None:
    """The error an operator actually meets, and the fix has to be inside it.

    Nine times out of ten the pipeline exists and this process simply never imported the
    module defining it — a worker started without ``--import``. A bare ``KeyError('name')``
    sends that person to the source; a message naming the flag, giving an example, and listing
    what *is* registered ends the incident at the terminal. The registry has no other way to
    communicate: it is a dict, and this string is its entire diagnostics surface.
    """
    registry.register("collides", _named("alpha"))

    with pytest.raises(KeyError) as caught:
        registry.get("nowhere")

    message = str(caught.value)
    assert "nowhere" in message
    assert "--import" in message, "the miss message does not name the flag that fixes it"
    assert "collides" in message, "the miss message does not say what IS registered"


def test_the_miss_message_says_so_when_nothing_at_all_is_registered() -> None:
    """The empty-registry case, which is the common one and the least informative.

    A worker started with no ``--import`` at all fails on its first claim with this message,
    and "Currently registered: " followed by nothing reads like a truncated string rather than
    an answer. The placeholder is what turns it into a diagnosis.
    """
    with pytest.raises(KeyError, match="nothing imported yet"):
        registry.get("nowhere")


# --------------------------------------------------------------------------------------
# load_module: dotted names, file paths, and failures
# --------------------------------------------------------------------------------------


def test_load_module_imports_a_dotted_name_and_is_idempotent(tmp_path: Path) -> None:
    """``--import myapp.pipelines`` must resolve through ``sys.path``, not the filesystem.

    Two things are pinned. The first is dispatch: ``_looks_like_path`` sends anything without
    a ``.py`` suffix or a separator down the ``importlib.import_module`` branch, so an
    unqualified name is imported as a *module* rather than being looked for as a file in the
    current working directory — which is what makes ``--import`` behave the same from any
    directory. Make that predicate answer "path" and this test fails immediately with
    ``FileNotFoundError``.

    The second is idempotency. ``--import`` is repeatable and a deployment can easily name the
    same module twice; re-executing it would re-run the ``@pipeline`` decorator, hand
    ``register`` a *different* Pipeline object under the same name, and turn a harmless
    duplicate flag into the conflict ValueError from the section above.
    """
    _write_module(tmp_path, "registry_probe_dotted", pipeline_name="probe_dotted")
    sys.path.insert(0, str(tmp_path))

    registry.load_module("registry_probe_dotted")
    assert registry.names() == ["probe_dotted"]
    first = registry.get("probe_dotted")

    registry.load_module("registry_probe_dotted")
    assert registry.names() == ["probe_dotted"], "the second import registered a second entry"
    assert registry.get("probe_dotted") is first, "the module was re-executed"


def test_load_module_imports_a_py_path_and_is_idempotent(tmp_path: Path) -> None:
    """``--import ./examples/summarise_thread.py`` — the form the README and demo use.

    The file branch cannot lean on ``importlib``'s cache the way the dotted branch does: it
    builds a spec from a location every time, so idempotency is an explicit early return
    keyed on ``(module name, __file__)``. Remove it and the second ``--import`` of the same
    path re-executes the module, producing a second Pipeline object under a name the first one
    already holds — ``ValueError: a different pipeline is already registered`` on a command
    line that asked for nothing unusual.

    This form matters more than it looks: the examples and the chaos suite live outside any
    installed package, so it is the only way to run durable-llm without packaging your pipelines
    first.
    """
    path = _write_module(tmp_path, "registry_probe_file", pipeline_name="probe_file")

    registry.load_module(str(path))
    assert registry.names() == ["probe_file"]
    first = registry.get("probe_file")
    module = sys.modules["registry_probe_file"]

    registry.load_module(str(path))
    assert registry.names() == ["probe_file"], "the second import registered a second entry"
    assert registry.get("probe_file") is first, "the file was re-executed"
    assert sys.modules["registry_probe_file"] is module


def test_loading_a_py_path_puts_its_directory_on_sys_path(tmp_path: Path) -> None:
    """A single-file pipeline has to be able to import the helpers beside it.

    Without this, ``--import ./pipelines/summarise.py`` loads a module that cannot do
    ``import helpers`` even though ``helpers.py`` is in the same folder — because the file was
    executed from a spec, not resolved from ``sys.path``, so nothing ever put its directory
    there. The failure surfaces as a ``ModuleNotFoundError`` from inside the user's own file,
    which reads like their bug and is not.
    """
    path = _write_module(tmp_path, "registry_probe_neighbour", pipeline_name="probe_neighbour")
    assert str(tmp_path) not in sys.path, "precondition: the directory is not already reachable"

    registry.load_module(str(path))

    assert str(tmp_path.resolve()) in sys.path


def test_a_module_that_raises_while_importing_leaves_no_entry_in_sys_modules(
    tmp_path: Path,
) -> None:
    """A half-executed module must not be left behind claiming to be importable.

    ``_load_from_file`` inserts into ``sys.modules`` *before* executing, deliberately, so a
    module that imports itself or defines dataclasses resolving by ``__module__`` sees the
    right object. The price of that is a corpse when execution raises: an entry whose top
    level never finished, with whatever names happened to be bound before the exception.

    Leaving it is worse than the original failure, because ``import`` is now a silent no-op
    that hands out the half-built module. Concretely: a worker started with ``--import
    broken.py`` fails loudly once, and then every later import of that name in the same
    process succeeds and returns a module with no pipelines in it — so the operator's next
    move ("run it again, it must have been a blip") produces a completely different and much
    more confusing error.
    """
    path = tmp_path / "registry_probe_boom.py"
    path.write_text(_EXPLODING_MODULE, encoding="utf-8")
    assert "registry_probe_boom" not in sys.modules, "precondition: the name is free"

    with pytest.raises(RuntimeError, match="refuses to import"):
        registry.load_module(str(path))

    assert "registry_probe_boom" not in sys.modules, (
        "a module that raised mid-execution was left in sys.modules, so importing it again "
        "would silently return the half-built object instead of raising"
    )
    assert registry.names() == [], "a failed import registered something"


def test_load_module_refuses_an_empty_specification() -> None:
    """``--import ''`` is a shell accident, and it must not be read as "import nothing".

    An empty string reaches ``importlib.import_module('')``, which raises ``ValueError`` from
    deep inside the import machinery with a message about relative imports and no package —
    accurate, and about a mechanism the operator never used. Refusing at the boundary keeps
    the flag's own vocabulary in the error.
    """
    with pytest.raises(ValueError, match="module name or a file path"):
        registry.load_module("   ")


def test_a_file_path_that_does_not_exist_says_so(tmp_path: Path) -> None:
    """The typo case, which must not be mistaken for a dotted module name.

    ``./piplines.py`` has a ``.py`` suffix, so ``_looks_like_path`` sends it down the file
    branch and the answer is "no such file" naming the path. Were the dispatch the other way
    round it would be an opaque ``ModuleNotFoundError: No module named 'piplines.py'``.
    """
    missing = tmp_path / "registry_probe_absent.py"

    with pytest.raises(FileNotFoundError, match="no such file"):
        registry.load_module(str(missing))
