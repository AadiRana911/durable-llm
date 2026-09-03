"""Pipeline name → callable, and the module loader behind ``--import``.

A run is a row in a table, and the only thing that row knows about the code that produced
it is a string: ``runs.pipeline``. Something has to turn that string back into a function
when a worker on a different machine picks the row up an hour later. There are two ways to
do that, and only one of them is honest.

The dishonest way is to scan the filesystem for anything that looks like a pipeline. It
demos well and then fails in production, because "anything that looks like a pipeline" is
a guess about someone else's project layout, and importing a module has side effects.

The honest way is to make the caller say where the code lives — ``--import
myapp.pipelines`` — and to fail loudly when they have not. So :func:`get` never guesses:
an unknown name raises ``KeyError`` with a message that names the flag, because the single
most common way to meet this error is to start a worker that was never told what it can
run, and the fix should be in the error rather than in the documentation.

Registration is process-local by design. The registry is not shared state between workers;
it is just the set of pipelines *this* process has imported.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular import at runtime
    from durable_llm.pipeline import Pipeline

__all__ = ["clear", "get", "load_module", "names", "register"]

logger = logging.getLogger("durable_llm.registry")

#: Name → pipeline for this process. Populated by the ``@pipeline`` decorator at import
#: time, which is why importing the defining module is the whole of "registration".
_REGISTRY: dict[str, Pipeline] = {}

#: Prefix for modules loaded from a file path whose stem is already taken in
#: ``sys.modules``. Loading ``./tasks.py`` must never shadow an installed ``tasks``
#: package for the rest of the process.
_FILE_MODULE_PREFIX: Final[str] = "durable_llm_imported_"


def register(name: str, pipeline: Pipeline) -> None:
    """Record ``pipeline`` under ``name`` for this process.

    Re-registering the *same* object is a no-op rather than an error: a module can
    legitimately be reached twice (imported by dotted path in one place and by file path in
    another), and punishing that would make ``--import`` fragile for no safety gain.

    Two *different* pipelines under one name is a genuine conflict and raises ``ValueError``.
    Silently letting the second win would mean a run enqueued against one function is
    executed by another — the kind of bug that only shows up as wrong output.
    """
    if not name:
        raise ValueError("pipeline name must be a non-empty string")

    existing = _REGISTRY.get(name)
    if existing is not None:
        if existing is pipeline:
            logger.debug("pipeline %r re-registered from the same object; ignoring", name)
            return
        raise ValueError(
            f"a different pipeline is already registered as {name!r} "
            f"({_describe(existing)} vs {_describe(pipeline)}); "
            f"pass name= to @pipeline to disambiguate them"
        )

    _REGISTRY[name] = pipeline
    logger.debug("registered pipeline %r (%s)", name, _describe(pipeline))


def get(name: str) -> Pipeline:
    """Look up a registered pipeline.

    Raises:
        KeyError: Nothing is registered under ``name``. The message names ``--import``,
            because in practice the pipeline almost always exists — this process simply
            never imported the module that defines it.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<nothing imported yet>"
        # `from None` so the traceback shows this explanation rather than the internal
        # dict lookup that produced an identical, useless KeyError.
        raise KeyError(
            f"no pipeline named {name!r} is registered in this process. "
            f"Pass --import with the module or file that defines it, for example: "
            f"durable-llm worker --import myapp.pipelines   (or --import ./examples/"
            f"summarise_thread.py). Currently registered: {known}"
        ) from None


def names() -> list[str]:
    """Every registered pipeline name, sorted. Used by the CLI to list what it can run."""
    return sorted(_REGISTRY)


def clear() -> None:
    """Forget every registration.

    Exists for tests, which import pipeline-defining modules repeatedly and would
    otherwise collide on the second import.
    """
    _REGISTRY.clear()


def load_module(dotted_or_path: str) -> None:
    """Import a module so its ``@pipeline`` decorators run.

    Accepts either a dotted module path (``myapp.pipelines``) or a filesystem path
    (``./examples/summarise_thread.py``). The file form matters more than it looks: the
    examples and the chaos tests live outside any installed package, and requiring users to
    package their pipelines before they can try the tool would be a poor first five minutes.

    Raises:
        ModuleNotFoundError: A dotted path that does not resolve.
        FileNotFoundError: A file path that does not exist.
        Exception: Whatever the module itself raised while executing, propagated
            unchanged. Deliberately not wrapped in ``ImportError``: the useful traceback is
            the one from inside the module, and re-raising as a different class buries the
            line the author actually needs to look at under an import failure that suggests
            the wrong problem. ``_load_from_file`` still removes the half-executed module
            from ``sys.modules`` on the way out, so a later import retries cleanly rather
            than finding a broken object already cached.
    """
    target = dotted_or_path.strip()
    if not target:
        raise ValueError("--import needs a module name or a file path")

    if _looks_like_path(target):
        _load_from_file(Path(target).expanduser())
    else:
        # importlib caches in sys.modules, so importing the same dotted path twice does not
        # re-run the decorators and cannot trip register()'s duplicate check.
        importlib.import_module(target)

    logger.debug("loaded %s; registered pipelines: %s", target, ", ".join(names()) or "<none>")


def _looks_like_path(target: str) -> bool:
    """Decide whether ``target`` is a file path rather than a dotted module name.

    A ``.py`` suffix or a path separator settles it. Everything else is treated as a dotted
    name, so an unqualified ``pipelines`` is imported from ``sys.path`` rather than being
    mistaken for a file in the current directory.
    """
    return target.endswith(".py") or "/" in target or "\\" in target


def _load_from_file(path: Path) -> None:
    """Execute a .py file as a module, without requiring it to be on ``sys.path``."""
    if not path.is_file():
        raise FileNotFoundError(f"cannot import {str(path)!r}: no such file")

    resolved = path.resolve()
    module_name = _module_name_for(resolved)

    already = sys.modules.get(module_name)
    if already is not None and getattr(already, "__file__", None) == str(resolved):
        # Same file, already executed. Returning early keeps `--import` idempotent instead
        # of re-running the decorators and raising a duplicate-registration ValueError.
        logger.debug("%s is already loaded; not re-executing it", resolved)
        return

    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:  # pragma: no cover - malformed file only
        raise ImportError(f"cannot import {str(resolved)!r}: not a loadable Python module")

    module = importlib.util.module_from_spec(spec)
    # Registered *before* execution so that a module importing itself, or defining
    # dataclasses that look themselves up by __module__, resolves to this object.
    sys.modules[module_name] = module

    # A single-file pipeline usually sits next to its helpers; without this, `import
    # helpers` inside it would fail even though the file it lives beside is right there.
    parent = str(resolved.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Do not leave a half-executed module in sys.modules pretending to be importable.
        sys.modules.pop(module_name, None)
        raise


def _module_name_for(resolved: Path) -> str:
    """Pick a ``sys.modules`` key for a file, avoiding collisions with real packages."""
    stem = resolved.stem
    candidate = stem if stem.isidentifier() else _FILE_MODULE_PREFIX + _sanitise(stem)

    occupant = sys.modules.get(candidate)
    if occupant is not None and getattr(occupant, "__file__", None) != str(resolved):
        # Something else owns that name — an installed package, most likely. Take a
        # namespaced key instead of shadowing it for the rest of the process.
        return _FILE_MODULE_PREFIX + _sanitise(stem)
    return candidate


def _sanitise(stem: str) -> str:
    """Turn a filename stem into a legal identifier."""
    cleaned = "".join(char if char.isalnum() else "_" for char in stem)
    return cleaned if cleaned and not cleaned[0].isdigit() else f"_{cleaned}"


def _describe(pipeline: Pipeline) -> str:
    """Locate a pipeline's function for a conflict message: ``module.qualname``."""
    fn = getattr(pipeline, "fn", pipeline)
    module = getattr(fn, "__module__", "?")
    qualname = getattr(fn, "__qualname__", getattr(fn, "__name__", "?"))
    return f"{module}.{qualname}"
