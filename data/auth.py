"""User accounts: registration, sign-in, sessions, password change/reset, account deletion.

Passwords are hashed with PBKDF2-HMAC-SHA256 (salted, 390k iterations); only hashes of session
tokens and reset codes are stored. After MAX_FAILED wrong passwords an account is locked for
LOCK_MINUTES.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import text

from data.db import engine, now_iso

ITERATIONS = 390_000
MAX_FAILED = 5
LOCK_MINUTES = 15
SESSION_DAYS_REMEMBER = 30
SESSION_HOURS_DEFAULT = 12
RESET_MINUTES = 15
COOKIE_NAME = "ki_session"

USERNAME_RE = re.compile(r"^[a-z0-9_.]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Files used before accounts existed; imported into the first account that is created.
LEGACY_FILES = {
    "portfolio": "portfolio.json",
    "watchlist": "watchlist.json",
    "profile": "profile.json",
    "chat": "chat_history.json",
    "ai_notes": "ai_notes.json",
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


# --- Passwords ----------------------------------------------------------------------------------


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        ITERATIONS, base64.b64encode(salt).decode(), base64.b64encode(digest).decode()
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt, digest = stored.split("$")
        check = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt), int(iterations))
        return hmac.compare_digest(check, base64.b64decode(digest))
    except (ValueError, TypeError):
        return False


def password_problem(password: str) -> str | None:
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return "Password must contain both letters and numbers."
    return None


# --- Users -------------------------------------------------------------------------------------------


def _user_row(conn, where: str, value: str):
    return conn.execute(
        text(f"SELECT id, username, email, password_hash, failed_attempts, locked_until FROM users WHERE {where} = :v"),
        {"v": value},
    ).first()


def _public(row) -> dict:
    return {"id": row[0], "username": row[1], "email": row[2]}


def find_user(identifier: str):
    """Look a user up by username or email (case-insensitive)."""
    ident = identifier.strip().lower()
    with engine().connect() as conn:
        return _user_row(conn, "email" if "@" in ident else "username", ident)


def user_count() -> int:
    with engine().connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM users")).scalar_one()


def list_user_ids() -> list[str]:
    with engine().connect() as conn:
        return [r[0] for r in conn.execute(text("SELECT id FROM users"))]


def register(username: str, email: str, password: str) -> tuple[bool, str, dict | None]:
    username, email = username.strip().lower(), email.strip().lower()
    if not USERNAME_RE.match(username):
        return False, "Username must be 3-30 characters: letters, numbers, dots or underscores.", None
    if not EMAIL_RE.match(email):
        return False, "Enter a valid email address.", None
    problem = password_problem(password)
    if problem:
        return False, problem, None
    first_account = user_count() == 0
    with engine().begin() as conn:
        if _user_row(conn, "username", username):
            return False, "That username is taken.", None
        if _user_row(conn, "email", email):
            return False, "An account with that email already exists — sign in instead.", None
        user_id = uuid.uuid4().hex
        conn.execute(
            text("INSERT INTO users (id, username, email, password_hash, created_at) VALUES (:i, :u, :e, :p, :t)"),
            {"i": user_id, "u": username, "e": email, "p": hash_password(password), "t": now_iso()},
        )
    user = {"id": user_id, "username": username, "email": email}
    import os

    import_ok = os.environ.get("IMPORT_LEGACY_DATA", "1") != "0"
    imported = import_legacy_files(user_id) if first_account and import_ok else []
    msg = "Account created."
    if imported:
        msg += " Your existing data (" + ", ".join(imported) + ") was moved into this account."
    return True, msg, user


def authenticate(identifier: str, password: str) -> tuple[bool, str, dict | None]:
    row = find_user(identifier)
    generic = "Incorrect username/email or password."
    if row is None:
        verify_password(password, hash_password("timing-equaliser"))  # similar timing for unknown users
        return False, generic, None
    user_id, _, _, pw_hash, failed, locked_until = row
    if locked_until and datetime.fromisoformat(locked_until) > _now():
        mins = max(1, int((datetime.fromisoformat(locked_until) - _now()).total_seconds() // 60) + 1)
        return False, f"Too many failed attempts. Try again in about {mins} minute(s).", None
    with engine().begin() as conn:
        if not verify_password(password, pw_hash):
            failed += 1
            lock = (_now() + timedelta(minutes=LOCK_MINUTES)).isoformat() if failed >= MAX_FAILED else None
            conn.execute(
                text("UPDATE users SET failed_attempts = :f, locked_until = :l WHERE id = :i"),
                {"f": 0 if lock else failed, "l": lock, "i": user_id},
            )
            if lock:
                return False, f"Too many failed attempts — account locked for {LOCK_MINUTES} minutes.", None
            return False, generic, None
        conn.execute(
            text("UPDATE users SET failed_attempts = 0, locked_until = NULL, last_login = :t WHERE id = :i"),
            {"t": now_iso(), "i": user_id},
        )
    return True, "Signed in.", _public(row)


def change_password(user_id: str, current: str, new: str) -> tuple[bool, str]:
    with engine().connect() as conn:
        row = _user_row(conn, "id", user_id)
    if row is None or not verify_password(current, row[3]):
        return False, "Your current password is incorrect."
    problem = password_problem(new)
    if problem:
        return False, problem
    with engine().begin() as conn:
        conn.execute(text("UPDATE users SET password_hash = :p WHERE id = :i"), {"p": hash_password(new), "i": user_id})
        conn.execute(text("DELETE FROM sessions WHERE user_id = :i"), {"i": user_id})  # sign out other devices
    return True, "Password changed. Other devices have been signed out."


def delete_account(user_id: str) -> None:
    with engine().begin() as conn:
        for table in ("documents", "sessions", "password_resets"):
            conn.execute(text(f"DELETE FROM {table} WHERE user_id = :i"), {"i": user_id})
        conn.execute(text("DELETE FROM users WHERE id = :i"), {"i": user_id})


# --- Sessions --------------------------------------------------------------------------------------


def create_session(user_id: str, remember: bool) -> tuple[str, int | None]:
    """Returns (token, cookie max-age in seconds or None for a browser-session cookie)."""
    token = secrets.token_urlsafe(32)
    lifetime = timedelta(days=SESSION_DAYS_REMEMBER) if remember else timedelta(hours=SESSION_HOURS_DEFAULT)
    with engine().begin() as conn:
        conn.execute(text("DELETE FROM sessions WHERE expires_at < :n"), {"n": _now().isoformat()})
        conn.execute(
            text("INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (:h, :u, :c, :e)"),
            {"h": _sha256(token), "u": user_id, "c": now_iso(), "e": (_now() + lifetime).isoformat()},
        )
    return token, int(lifetime.total_seconds()) if remember else None


def user_for_session(token: str | None) -> dict | None:
    if not isinstance(token, str) or not token:
        return None
    with engine().connect() as conn:
        row = conn.execute(
            text(
                "SELECT u.id, u.username, u.email, s.expires_at FROM sessions s JOIN users u ON u.id = s.user_id "
                "WHERE s.token_hash = :h"
            ),
            {"h": _sha256(token)},
        ).first()
    if row is None or datetime.fromisoformat(row[3]) < _now():
        return None
    return {"id": row[0], "username": row[1], "email": row[2]}


def end_session(token: str | None) -> None:
    if token:
        with engine().begin() as conn:
            conn.execute(text("DELETE FROM sessions WHERE token_hash = :h"), {"h": _sha256(token)})


# --- Password reset --------------------------------------------------------------------------------


def start_reset(identifier: str) -> tuple[bool, str]:
    """Email a 6-digit code. The reply is the same whether or not the account exists."""
    from data.notify import email_configured, send_email

    if not email_configured():
        return False, "Password reset by email isn't set up on this app yet — contact the app owner."
    generic = "If an account matches, a 6-digit code has been emailed to it. It expires in 15 minutes."
    row = find_user(identifier)
    if row is None:
        return True, generic
    code = f"{secrets.randbelow(10**6):06d}"
    with engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO password_resets (user_id, code_hash, expires_at, attempts) VALUES (:u, :c, :e, 0) "
                "ON CONFLICT (user_id) DO UPDATE SET code_hash = excluded.code_hash, "
                "expires_at = excluded.expires_at, attempts = 0"
            ),
            {"u": row[0], "c": _sha256(code), "e": (_now() + timedelta(minutes=RESET_MINUTES)).isoformat()},
        )
    ok, msg = send_email(
        row[2],
        "Your Kenya Invest password reset code",
        f"Your password reset code is {code}. It expires in {RESET_MINUTES} minutes.\n\n"
        "If you didn't ask to reset your password, you can ignore this email.",
    )
    return (True, generic) if ok else (False, msg)


def finish_reset(identifier: str, code: str, new_password: str) -> tuple[bool, str]:
    invalid = "That code is invalid or has expired. Request a new one."
    row = find_user(identifier)
    if row is None:
        return False, invalid
    problem = password_problem(new_password)
    if problem:
        return False, problem
    with engine().begin() as conn:
        reset = conn.execute(
            text("SELECT code_hash, expires_at, attempts FROM password_resets WHERE user_id = :u"), {"u": row[0]}
        ).first()
        if reset is None or datetime.fromisoformat(reset[1]) < _now() or reset[2] >= 5:
            return False, invalid
        if not hmac.compare_digest(reset[0], _sha256(code.strip())):
            conn.execute(text("UPDATE password_resets SET attempts = attempts + 1 WHERE user_id = :u"), {"u": row[0]})
            return False, invalid
        conn.execute(text("UPDATE users SET password_hash = :p, failed_attempts = 0, locked_until = NULL WHERE id = :u"),
                     {"p": hash_password(new_password), "u": row[0]})
        conn.execute(text("DELETE FROM password_resets WHERE user_id = :u"), {"u": row[0]})
        conn.execute(text("DELETE FROM sessions WHERE user_id = :u"), {"u": row[0]})
    return True, "Password updated — sign in with your new password."


# --- Legacy data import ------------------------------------------------------------------------------


def import_legacy_files(user_id: str) -> list[str]:
    """Move pre-account JSON files (data/*.json) into this user's documents; rename them *.imported."""
    import json

    from data.storage import put_doc

    folder = Path(__file__).parent
    imported = []
    for kind, filename in LEGACY_FILES.items():
        path = folder / filename
        if path.exists():
            try:
                put_doc(kind, json.loads(path.read_text()), user_id=user_id)
                path.rename(path.with_suffix(".json.imported"))
                imported.append(kind.replace("_", " "))
            except (ValueError, OSError):
                continue
    return imported
