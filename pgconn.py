"""Postgres connection plumbing shared by the ingest scripts and the backend.

Credentials come from the environment, never from the config file, so the same
`config.nvissues.pg.yaml` can be committed and used unchanged on a laptop, on
the DGX and in a Container App. `.pgsecrets` at the repo root is a convenience
for interactive use and is gitignored; in ACA the same names arrive as secrets.
"""
from __future__ import annotations

import os
import pathlib

DEFAULT_SECRETS = pathlib.Path(__file__).with_name(".pgsecrets")

# Azure Database for PostgreSQL terminates idle connections and sits behind a
# gateway, so a keepalive is worth more here than the default would suggest.
_KEEPALIVE = {
    "keepalives": "1",
    "keepalives_idle": "30",
    "keepalives_interval": "10",
    "keepalives_count": "5",
}


def load_secrets(path: str | os.PathLike | None = None) -> None:
    """Populate PG* environment variables from a dotenv-style file.

    Existing environment values win, so an ACA secret or an explicit export is
    never silently overridden by a stale file left in a working copy.
    """
    p = pathlib.Path(path) if path else DEFAULT_SECRETS
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def dsn(dbname: str | None = None, *, autoload: bool = True) -> str:
    """Build a libpq connection string from the environment."""
    if autoload:
        load_secrets()
    host = os.environ.get("PGHOST")
    if not host:
        raise RuntimeError(
            "PGHOST is unset. Export the PG* variables, or place them in "
            f"{DEFAULT_SECRETS.name} at the repo root."
        )
    parts = {
        "host": host,
        "port": os.environ.get("PGPORT", "5432"),
        "user": os.environ.get("PGUSER", "nvissues"),
        "password": os.environ.get("PGPASSWORD", ""),
        "dbname": dbname or os.environ.get("PGDATABASE", "nvissues"),
        # Azure requires TLS; `require` verifies the channel without pinning a
        # CA bundle we would then have to ship into the container image.
        "sslmode": os.environ.get("PGSSLMODE", "require"),
        "application_name": os.environ.get("PGAPPNAME", "nvissues"),
        **_KEEPALIVE,
    }
    return " ".join(f"{k}={v}" for k, v in parts.items() if v != "")


def connect(dbname: str | None = None, *, autocommit: bool = False):
    """One synchronous connection, used by the ingest and schema scripts."""
    import psycopg

    return psycopg.connect(dsn(dbname), connect_timeout=30, autocommit=autocommit)


def redacted(dbname: str | None = None) -> str:
    """The DSN with the password removed, for logging."""
    return " ".join(
        part for part in dsn(dbname).split() if not part.startswith("password=")
    )
