"""Postgres connection plumbing.

Two decisions are made here once so that no other module has to think about them again.

*Rows are dicts.* Every cursor produced by :func:`connect` yields ``dict``-like rows, so
call sites read ``row["spent_usd"]`` instead of counting columns. Column order then stops
being part of the interface between modules.

*Money is Decimal.* Everything financial in the schema is ``numeric(12,6)``. psycopg 3
already maps ``numeric`` to :class:`decimal.Decimal` in both directions, but "already does"
is not a guarantee anyone can point at during review, so :func:`register_decimal_adapters`
states it explicitly and :func:`connect` calls it. A float sneaking into the cost ledger
would silently corrupt every subsequent budget check.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from psycopg.abc import AdaptContext
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import DictRow, dict_row

__all__ = [
    "DEFAULT_DSN",
    "DEFAULT_DSN_ENV",
    "connect",
    "default_dsn",
    "describe_dsn",
    "register_decimal_adapters",
    "transaction",
]

logger = logging.getLogger("durable_llm.db")

DEFAULT_DSN_ENV = "DURABLE_LLM_DATABASE_URL"
"""Environment variable consulted by :func:`default_dsn`."""

DEFAULT_DSN = "postgresql://durable_llm:durable_llm@localhost:55432/durable_llm"
"""Matches docker-compose.yml. Port 55432 avoids colliding with a local Postgres on 5432."""

_APPLICATION_NAME = "durable-llm"

_adapters_registered = False


def default_dsn() -> str:
    """Return the DSN from the environment, falling back to the local compose database.

    An empty or whitespace-only environment variable is treated as unset: an exported but
    blank ``DURABLE_LLM_DATABASE_URL`` is far more likely to be a broken shell profile than a
    deliberate request to connect to nothing.
    """
    dsn = os.environ.get(DEFAULT_DSN_ENV, "").strip()
    return dsn or DEFAULT_DSN


def register_decimal_adapters(context: AdaptContext | None = None) -> None:
    """Guarantee that PostgreSQL ``numeric`` round-trips as :class:`decimal.Decimal`.

    Passing ``context=None`` registers on psycopg's global adapter map, which every
    connection created afterwards inherits; passing a connection scopes the registration to
    that connection.

    This restates psycopg's own defaults rather than replacing them, so it is a no-op on a
    stock install. It is worth the few lines anyway: it turns "numeric happens to load as
    Decimal" into a property this project asserts on purpose, and it gives a future psycopg
    reshuffle a single place to be fixed instead of a budget checker quietly handed floats.
    """
    adapters = context.adapters if context is not None else psycopg.adapters
    try:
        from psycopg.types.numeric import (
            DecimalBinaryDumper,
            DecimalDumper,
            NumericBinaryLoader,
            NumericLoader,
        )
    except ImportError:  # pragma: no cover - depends on the installed psycopg build
        logger.warning(
            "psycopg's numeric adapters could not be imported; relying on library defaults "
            "for Decimal round-tripping"
        )
        return

    # Text first, then binary: psycopg keys dumpers by wire format, and registering in this
    # order preserves the library's own preference for the text dumper.
    adapters.register_dumper("decimal.Decimal", DecimalDumper)
    adapters.register_dumper("decimal.Decimal", DecimalBinaryDumper)
    adapters.register_loader("numeric", NumericLoader)
    adapters.register_loader("numeric", NumericBinaryLoader)


def _ensure_global_decimal_adapters() -> None:
    """Register the Decimal adapters on the global map exactly once per process."""
    global _adapters_registered
    if _adapters_registered:
        return
    register_decimal_adapters()
    _adapters_registered = True


def connect(dsn: str | None = None, **kwargs: Any) -> psycopg.Connection[DictRow]:
    """Open a connection with the project's non-negotiable settings.

    ``autocommit=False`` is deliberate: durable_llm's correctness rests on writing a provider
    response, its cost row and the run's spend inside one transaction, so implicit
    per-statement commits would defeat the whole design. Use :func:`transaction` to commit.

    Extra keyword arguments are forwarded to :func:`psycopg.connect` (``connect_timeout``,
    ``sslmode``, ...); anything set there overrides the same key in the DSN.
    """
    conninfo = dsn if dsn is not None else default_dsn()
    _ensure_global_decimal_adapters()
    kwargs.setdefault("application_name", _APPLICATION_NAME)
    conn: psycopg.Connection[DictRow] = psycopg.connect(
        conninfo,
        row_factory=dict_row,
        autocommit=False,
        **kwargs,
    )
    logger.debug("connected to %s", describe_dsn(conninfo))
    return conn


@contextmanager
def transaction(conn: psycopg.Connection[DictRow]) -> Iterator[psycopg.Cursor[DictRow]]:
    """Run a block in one transaction: commit on clean exit, roll back on any exception.

    The cursor is closed before the commit, so results must be fetched inside the block.
    ``BaseException`` is caught rather than ``Exception`` because a ``KeyboardInterrupt`` or
    a worker's SIGTERM handler firing mid-block must not leave a half-written transaction
    open on the connection.

    Not re-entrant: nesting two of these on the same connection would let the inner block
    commit the outer block's work. Callers that need a nested scope should use psycopg's
    ``conn.transaction()`` savepoints instead.
    """
    cur = conn.cursor()
    try:
        try:
            yield cur
        finally:
            cur.close()
    except BaseException:
        conn.rollback()
        raise
    conn.commit()


def describe_dsn(dsn: str) -> str:
    """Summarise a DSN as ``host:port/dbname`` for logs.

    Never log a DSN directly — it carries the password. This exists so that the safe form
    is the convenient one.
    """
    try:
        params = conninfo_to_dict(dsn)
    except psycopg.Error:
        return "<unparseable dsn>"
    host = params.get("host") or os.environ.get("PGHOST") or "localhost"
    port = params.get("port") or os.environ.get("PGPORT") or "5432"
    dbname = params.get("dbname") or os.environ.get("PGDATABASE") or "?"
    return f"{host}:{port}/{dbname}"
