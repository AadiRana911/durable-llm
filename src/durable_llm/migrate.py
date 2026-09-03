"""SQL migration runner.

Deliberately about eighty lines of logic instead of a dependency: durable_llm has four tables
and forward-only migrations, which is well below the threshold where Alembic earns its
keep. What it does buy is the parts people actually get wrong — one transaction per file,
an advisory lock so two workers booting at once cannot both apply ``0001``, and a loud
warning when the database is ahead of the checkout.

Ordering is by filename, so migrations are named ``NNNN_description.sql`` with a
zero-padded prefix and the ``version`` recorded in ``schema_migrations`` is the stem.
"""

from __future__ import annotations

import hashlib
import logging
import os
from importlib import resources
from pathlib import Path

import psycopg
from psycopg.rows import DictRow, tuple_row

from durable_llm.db import transaction

__all__ = [
    "MIGRATIONS_DIR_ENV",
    "MigrationError",
    "default_migrations_dir",
    "migrate",
]

logger = logging.getLogger("durable_llm.migrate")

MIGRATIONS_DIR_ENV = "DURABLE_LLM_MIGRATIONS_DIR"
"""Escape hatch for deployments that ship the .sql files somewhere unusual."""

_PACKAGE = "durable_llm"

# How far up from this file to look for a sibling ``migrations/`` directory. Three levels
# covers ``src/durable_llm/`` -> ``src/`` -> repository root; going further risks matching an
# unrelated directory somewhere up the filesystem.
_SEARCH_DEPTH = 3

# Session-level advisory lock guarding the whole migrate() call, so concurrently starting
# workers serialise instead of racing to apply the same file. The key is arbitrary but must
# be stable and unlikely to collide. It was originally derived from
# ``sha256(b"llmrun.schema_migrations")`` and deliberately survives the pre-release rename:
# changing it would let an old and a renamed process take different locks against one database.
_ADVISORY_LOCK_KEY = 7026850041296117638

# The runner owns this table: it cannot be created by a migration whose application it is
# supposed to record. migrations/0001_initial.sql repeats it (guarded by IF NOT EXISTS) so
# that file stays a complete description of the schema — the two definitions must match.
_SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
)
"""

# The `checksum` column arrives in 0003, so the runner cannot assume it exists: on a database
# still at 0002 -- which is exactly the database that is about to apply 0003 -- selecting it
# would fail before the migration that adds it could run. Hence the probe rather than a plain
# SELECT, and hence every checksum path below tolerating None.
_HAS_CHECKSUM_COLUMN = """
SELECT 1 FROM information_schema.columns
WHERE table_schema = current_schema()
  AND table_name = 'schema_migrations'
  AND column_name = 'checksum'
"""

# Two spellings of the same read, picked by the probe above rather than assembled with string
# formatting -- a SELECT built by f-string is how a project ends up with one interpolated
# statement it cannot grep for later, and there are only ever two forms here.
_SELECT_APPLIED_WITH_CHECKSUM = "SELECT version, checksum FROM schema_migrations"
_SELECT_APPLIED_BARE = "SELECT version, NULL FROM schema_migrations"


class MigrationError(RuntimeError):
    """A migration could not be located, read, or applied."""


def default_migrations_dir() -> Path:
    """Locate the bundled ``migrations/`` directory.

    Resolution order, first hit wins:

    1. ``$DURABLE_LLM_MIGRATIONS_DIR``.
    2. ``durable_llm/migrations`` inside the installed package, via :mod:`importlib.resources` —
       this is the wheel case, and it works only if the build force-includes the .sql files
       into the package.
    3. A ``migrations/`` directory found by walking up from this file — the source-checkout
       case, where ``migrations/`` sits at the repository root next to ``src/``.

    Both filesystem strategies are needed because the contract puts ``migrations/`` at the
    repository root rather than inside the package, so neither one alone covers both a
    ``uv sync`` checkout and an installed wheel.
    """
    override = os.environ.get(MIGRATIONS_DIR_ENV, "").strip()
    if override:
        candidate = Path(override).expanduser()
        if not _contains_migrations(candidate):
            raise MigrationError(
                f"{MIGRATIONS_DIR_ENV}={override!r} does not contain any .sql migrations"
            )
        return candidate

    for candidate in _candidate_dirs():
        if _contains_migrations(candidate):
            return candidate

    raise MigrationError(
        "could not locate the migrations directory; pass migrations_dir= explicitly or set "
        f"{MIGRATIONS_DIR_ENV}"
    )


def migrate(conn: psycopg.Connection[DictRow], migrations_dir: Path | None = None) -> list[str]:
    """Apply every unapplied migration in order and return the versions applied.

    Idempotent: a second call on an up-to-date database returns ``[]`` without touching the
    schema. Each file runs in its own transaction, so a failure part-way through a batch
    leaves the earlier files applied and recorded, and the failing one applied not at all —
    Postgres rolls back DDL like anything else.
    """
    directory = Path(migrations_dir) if migrations_dir is not None else default_migrations_dir()
    if not directory.is_dir():
        raise MigrationError(f"migrations directory does not exist: {directory}")

    files = sorted(directory.glob("*.sql"), key=lambda path: path.name)
    if not files:
        logger.warning("no .sql files found in %s", directory)

    _acquire_lock(conn)
    try:
        _ensure_schema_migrations(conn)
        already_applied = _applied_versions(conn)
        _warn_if_database_is_ahead(already_applied, files)
        _verify_checksums(already_applied, files)

        applied: list[str] = []
        for path in files:
            version = path.stem
            if version in already_applied:
                logger.debug("migration %s already applied, skipping", version)
                continue
            _apply(conn, path, version)
            applied.append(version)
            logger.info("applied migration %s", version)

        if not applied:
            logger.debug("database is up to date (%d migrations)", len(already_applied))
        return applied
    finally:
        _release_lock(conn)


def _candidate_dirs() -> list[Path]:
    """Directories that might hold the .sql files, in priority order."""
    candidates: list[Path] = []

    try:
        anchor = resources.files(_PACKAGE)
    except (ImportError, TypeError):  # pragma: no cover - only when durable_llm is not importable
        anchor = None
    if isinstance(anchor, Path):
        # A zip-imported package yields a non-Path Traversable; durable_llm needs a real
        # directory here, and a zipped install is out of scope.
        candidates.append(anchor / "migrations")

    here = Path(__file__).resolve()
    candidates.extend(parent / "migrations" for parent in here.parents[:_SEARCH_DEPTH])
    return candidates


def _contains_migrations(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob("*.sql"))


def _acquire_lock(conn: psycopg.Connection[DictRow]) -> None:
    """Take the session-level advisory lock, blocking until it is free.

    Session-level rather than transaction-level because it has to outlive the individual
    per-migration transactions. It is released in migrate()'s ``finally``, and by the
    server anyway if the connection dies.
    """
    with transaction(conn) as cur:
        cur.execute("SELECT pg_advisory_lock(%s)", (_ADVISORY_LOCK_KEY,))


def _release_lock(conn: psycopg.Connection[DictRow]) -> None:
    try:
        with transaction(conn) as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (_ADVISORY_LOCK_KEY,))
    except psycopg.Error:
        # Losing the connection releases the lock server-side, so this is never fatal and
        # must not mask whatever error is already propagating.
        logger.warning("could not release the migration advisory lock", exc_info=True)


def _ensure_schema_migrations(conn: psycopg.Connection[DictRow]) -> None:
    with transaction(conn) as cur:
        cur.execute(_SCHEMA_MIGRATIONS_DDL)


def checksum_of(sql_text: str) -> str:
    """The digest recorded for a migration: ``sha256`` hex over its UTF-8 bytes.

    Public so a test, or an operator reconciling a database by hand, can compute the same
    number the runner does without reimplementing it slightly differently.
    """
    return hashlib.sha256(sql_text.encode("utf-8")).hexdigest()


def _applied_versions(conn: psycopg.Connection[DictRow]) -> dict[str, str | None]:
    """Read the recorded versions and their checksums.

    Uses an explicit ``tuple_row`` cursor so the result does not depend on the row factory
    the caller's connection happens to carry.

    The checksum is ``None`` for a row recorded before 0003 added the column, and for every
    row on a database that has not applied 0003 yet — including, necessarily, the database
    that is one statement away from applying it.
    """
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(_HAS_CHECKSUM_COLUMN)
        has_checksum = cur.fetchone() is not None
        cur.execute(_SELECT_APPLIED_WITH_CHECKSUM if has_checksum else _SELECT_APPLIED_BARE)
        versions = {row[0]: row[1] for row in cur.fetchall()}
    # End the implicit read transaction so each migration below starts from a clean slate.
    conn.commit()
    return versions


def _warn_if_database_is_ahead(applied: dict[str, str | None], files: list[Path]) -> None:
    unknown = sorted(set(applied) - {path.stem for path in files})
    if unknown:
        logger.warning(
            "database records %d migration(s) with no matching file (%s); this checkout may "
            "be older than the database",
            len(unknown),
            ", ".join(unknown),
        )


def _verify_checksums(applied: dict[str, str | None], files: list[Path]) -> None:
    """Refuse to run if an already-applied migration file has been edited since.

    Migrations are forward-only, so the dangerous edit is not a bad rollback — it is someone
    changing a ``.sql`` file that has already run. The runner skips it (the version is
    recorded), the database keeps the old shape, and the checkout now describes a schema that
    does not exist anywhere. Nothing surfaces until a query fails against a column the file
    says is there.

    Raising here converts that into a startup failure naming the file. A recorded ``None``
    is not a mismatch: it means the row predates the checksum column, and asserting anything
    about it would be inventing evidence.
    """
    drifted: list[str] = []
    for path in files:
        recorded = applied.get(path.stem)
        if recorded is None:
            continue
        try:
            current = checksum_of(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise MigrationError(f"could not read migration {path}: {exc}") from exc
        if current != recorded:
            drifted.append(f"{path.name} (recorded {recorded[:12]}…, on disk {current[:12]}…)")

    if drifted:
        raise MigrationError(
            "these migration files have changed since they were applied: "
            + "; ".join(drifted)
            + ". A forward-only runner cannot reconcile that — the database still has the "
            "old shape. Restore the file to what was applied and put the change in a new "
            "migration, or, if the edit is known to be cosmetic, update the recorded "
            "checksum by hand in schema_migrations."
        )


def _apply(conn: psycopg.Connection[DictRow], path: Path, version: str) -> None:
    """Run one migration file and record it, atomically."""
    try:
        sql_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MigrationError(f"could not read migration {path}: {exc}") from exc
    if not sql_text.strip():
        raise MigrationError(f"migration {path} is empty")

    try:
        with transaction(conn) as cur:
            # Passing no parameters makes psycopg use the simple query protocol, which is
            # what allows one file to contain several statements. It also means no
            # placeholder substitution happens, so a literal '%' in the DDL is safe.
            cur.execute(sql_text)
            # The digest goes in inside the same transaction as the DDL it describes, so a
            # rolled-back migration cannot leave behind a checksum for a schema change that
            # did not happen. 0003 is the migration that adds the column, so it is written
            # through a statement that tolerates the column not existing yet: the INSERT
            # below runs *after* cur.execute(sql_text), by which point 0003 has created it.
            _record_version(cur, version, checksum_of(sql_text))
    except psycopg.Error as exc:
        raise MigrationError(f"migration {path.name} failed and was rolled back: {exc}") from exc


def _record_version(cur: psycopg.Cursor[DictRow], version: str, checksum: str) -> None:
    """Insert the applied-migration row, with or without the checksum column."""
    cur.execute(_HAS_CHECKSUM_COLUMN)
    if cur.fetchone() is None:
        # Pre-0003 database applying 0001 or 0002: there is nowhere to put the digest yet.
        # The row is recorded without one and _verify_checksums treats it as unknown rather
        # than as a mismatch, which is the honest reading — nobody checked it.
        cur.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
        return
    cur.execute(
        "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)",
        (version, checksum),
    )
