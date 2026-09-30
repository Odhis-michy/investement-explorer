"""Per-user document storage.

Each signed-in user's data (portfolio, watchlist, profile, chat, AI notes) is stored as a JSON
document in the `documents` table, keyed by (user_id, kind). The "current user" is a context
variable set at the start of every app run (and per user by the background alert worker), so
modules like data/portfolio.py keep their simple load_x() / save_x() API.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from sqlalchemy import text

from data.db import engine, now_iso

_current_user: ContextVar[str | None] = ContextVar("current_user", default=None)


def set_user(user_id: str | None) -> None:
    _current_user.set(user_id)


def current_user() -> str:
    user_id = _current_user.get()
    if not user_id:
        raise RuntimeError("No signed-in user for this request.")
    return user_id


@contextmanager
def as_user(user_id: str):
    token = _current_user.set(user_id)
    try:
        yield
    finally:
        _current_user.reset(token)


def get_doc(kind: str, user_id: str | None = None) -> Any | None:
    with engine().connect() as conn:
        row = conn.execute(
            text("SELECT data FROM documents WHERE user_id = :u AND kind = :k"),
            {"u": user_id or current_user(), "k": kind},
        ).first()
    return json.loads(row[0]) if row else None


def put_doc(kind: str, data: Any, user_id: str | None = None) -> None:
    with engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO documents (user_id, kind, data, updated_at) VALUES (:u, :k, :d, :t) "
                "ON CONFLICT (user_id, kind) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at"
            ),
            {"u": user_id or current_user(), "k": kind, "d": json.dumps(data), "t": now_iso()},
        )


def delete_doc(kind: str, user_id: str | None = None) -> None:
    with engine().begin() as conn:
        conn.execute(
            text("DELETE FROM documents WHERE user_id = :u AND kind = :k"),
            {"u": user_id or current_user(), "k": kind},
        )
