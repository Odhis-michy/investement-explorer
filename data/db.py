"""Database connection and schema.

Uses DATABASE_URL when set (e.g. a free Supabase / Neon Postgres URL for Streamlit Community
Cloud, whose disk is wiped on restart), otherwise a local SQLite file at data/app.db.

    DATABASE_URL=postgresql://user:password@host:5432/dbname
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

DEFAULT_URL = f"sqlite:///{Path(__file__).parent / 'app.db'}"

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT NOT NULL UNIQUE,
        email TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        last_login TEXT,
        failed_attempts INTEGER NOT NULL DEFAULT 0,
        locked_until TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS documents (
        user_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        data TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (user_id, kind)
    )""",
    """CREATE TABLE IF NOT EXISTS password_resets (
        user_id TEXT PRIMARY KEY,
        code_hash TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS login_codes (
        user_id TEXT PRIMARY KEY,
        purpose TEXT NOT NULL,
        code_hash TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0
    )""",
]

# Columns added after the first release; created on existing databases by engine().
ADDED_COLUMNS = {
    "users": {
        "twofa_method": "TEXT",
        "totp_secret": "TEXT",
        "twofa_phone": "TEXT",
        "recovery_codes": "TEXT",
        "last_totp_step": "INTEGER",
    },
}

_engine: Engine | None = None
_lock = threading.Lock()


def database_url() -> str:
    url = os.environ.get("DATABASE_URL") or DEFAULT_URL
    if url.startswith("postgres://"):  # Heroku/Supabase-style scheme SQLAlchemy doesn't accept
        url = "postgresql://" + url[len("postgres://"):]
    return url


def is_sqlite() -> bool:
    return database_url().startswith("sqlite")


def engine() -> Engine:
    """Create the engine and tables once per process."""
    global _engine
    with _lock:
        if _engine is None:
            url = database_url()
            kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
            _engine = create_engine(url, pool_pre_ping=True, **kwargs)
            with _engine.begin() as conn:
                for statement in SCHEMA:
                    conn.execute(text(statement))
                existing = {t: {c["name"] for c in inspect(conn).get_columns(t)} for t in ADDED_COLUMNS}
                for table, columns in ADDED_COLUMNS.items():
                    for name, kind in columns.items():
                        if name not in existing[table]:
                            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {kind}"))
        return _engine


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
