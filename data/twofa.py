"""Two-step verification (2FA) for sign-in.

Methods:
- "totp": an authenticator app (Google/Microsoft Authenticator, Authy…) using RFC 6238 time-based
  6-digit codes. Codes are accepted for the current 30-second step ±1, and a step can't be reused.
- "sms":  a 6-digit code texted to a verified phone number (needs the SMS settings from
  data/notify.py). Codes are hashed, expire after 5 minutes and allow 5 tries.
Each account also gets 8 single-use recovery codes (stored hashed) for when the phone is lost.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import secrets
import struct
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

from sqlalchemy import text

from data.db import engine

ISSUER = "Kenya Invest"
STEP_SECONDS = 30
SMS_CODE_MINUTES = 5
MAX_CODE_ATTEMPTS = 5
RECOVERY_CODES = 8


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


# --- TOTP (RFC 6238) -------------------------------------------------------------------------------


def new_totp_secret() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode().rstrip("=")


def totp_code(secret: str, step: int, digits: int = 6) -> str:
    key = base64.b32decode(secret + "=" * (-len(secret) % 8), casefold=True)
    digest = hmac.new(key, struct.pack(">Q", step), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    value = (struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return str(value).zfill(digits)


def current_step(at: float | None = None) -> int:
    return int((at if at is not None else time.time()) // STEP_SECONDS)


def match_totp(secret: str, code: str, last_step: int | None, at: float | None = None) -> int | None:
    """The matching time step (for replay protection), or None."""
    code = code.strip().replace(" ", "")
    if not (code.isdigit() and len(code) == 6):
        return None
    now = current_step(at)
    for step in (now - 1, now, now + 1):
        if (last_step is None or step > last_step) and hmac.compare_digest(totp_code(secret, step), code):
            return step
    return None


def totp_uri(username: str, secret: str) -> str:
    label = quote(f"{ISSUER}:{username}")
    return f"otpauth://totp/{label}?secret={secret}&issuer={quote(ISSUER)}&digits=6&period={STEP_SECONDS}"


def qr_png(data: str) -> bytes:
    import qrcode

    img = qrcode.make(data, box_size=6, border=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- Recovery codes ----------------------------------------------------------------------------------


def _new_recovery_codes() -> list[str]:
    return [f"{secrets.token_hex(2)}-{secrets.token_hex(2)}".upper() for _ in range(RECOVERY_CODES)]


def _normalize_recovery(code: str) -> str:
    raw = code.strip().upper().replace(" ", "").replace("-", "")
    return f"{raw[:4]}-{raw[4:]}" if len(raw) == 8 else raw


# --- Account state ---------------------------------------------------------------------------------------


def status(user_id: str) -> dict:
    with engine().connect() as conn:
        row = conn.execute(
            text("SELECT twofa_method, twofa_phone, recovery_codes FROM users WHERE id = :u"), {"u": user_id}
        ).first()
    method, phone, recovery = row if row else (None, None, None)
    return {
        "enabled": bool(method),
        "method": method,
        "phone": phone,
        "recoveryLeft": len(json.loads(recovery)) if recovery else 0,
    }


def _save(user_id: str, **fields) -> None:
    sets = ", ".join(f"{k} = :{k}" for k in fields)
    with engine().begin() as conn:
        conn.execute(text(f"UPDATE users SET {sets} WHERE id = :uid"), {**fields, "uid": user_id})


def enable_totp(user_id: str, secret: str, code: str) -> tuple[bool, str, list[str]]:
    step = match_totp(secret, code, None)
    if step is None:
        return False, "That code didn't match. Check the time on your phone is automatic and try the newest code.", []
    codes = _new_recovery_codes()
    _save(user_id, twofa_method="totp", totp_secret=secret, twofa_phone=None, last_totp_step=step,
          recovery_codes=json.dumps([_sha256(c) for c in codes]))
    return True, "Two-step verification is on (authenticator app).", codes


def _store_code(user_id: str, purpose: str) -> str:
    code = f"{secrets.randbelow(10**6):06d}"
    with engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO login_codes (user_id, purpose, code_hash, expires_at, attempts) "
                "VALUES (:u, :p, :c, :e, 0) ON CONFLICT (user_id) DO UPDATE SET purpose = excluded.purpose, "
                "code_hash = excluded.code_hash, expires_at = excluded.expires_at, attempts = 0"
            ),
            {"u": user_id, "p": purpose, "c": _sha256(code),
             "e": (_now() + timedelta(minutes=SMS_CODE_MINUTES)).isoformat()},
        )
    return code


def _check_code(user_id: str, purpose: str, code: str) -> bool:
    with engine().begin() as conn:
        row = conn.execute(
            text("SELECT purpose, code_hash, expires_at, attempts FROM login_codes WHERE user_id = :u"),
            {"u": user_id},
        ).first()
        if row is None or row[0] != purpose or datetime.fromisoformat(row[2]) < _now() or row[3] >= MAX_CODE_ATTEMPTS:
            return False
        if not hmac.compare_digest(row[1], _sha256(code.strip())):
            conn.execute(text("UPDATE login_codes SET attempts = attempts + 1 WHERE user_id = :u"), {"u": user_id})
            return False
        conn.execute(text("DELETE FROM login_codes WHERE user_id = :u"), {"u": user_id})
    return True


def start_sms_setup(user_id: str, phone: str) -> tuple[bool, str]:
    from data.notify import normalize_phone, send_sms, sms_configured

    number = normalize_phone(phone)
    if not number:
        return False, "Enter a valid phone number (e.g. 0712 345 678)."
    if not sms_configured():
        return False, "SMS isn't set up on this app yet — use an authenticator app instead."
    code = _store_code(user_id, f"setup:{number}")
    ok, msg = send_sms(number, f"Kenya Invest: your code to turn on two-step verification is {code}. "
                               f"It expires in {SMS_CODE_MINUTES} minutes.")
    return (True, f"Code sent to {mask_phone(number)}.") if ok else (False, msg)


def enable_sms(user_id: str, phone: str, code: str) -> tuple[bool, str, list[str]]:
    from data.notify import normalize_phone

    number = normalize_phone(phone)
    if not number or not _check_code(user_id, f"setup:{number}", code):
        return False, "That code is invalid or has expired.", []
    codes = _new_recovery_codes()
    _save(user_id, twofa_method="sms", twofa_phone=number, totp_secret=None, last_totp_step=None,
          recovery_codes=json.dumps([_sha256(c) for c in codes]))
    return True, f"Two-step verification is on (SMS to {mask_phone(number)}).", codes


def disable(user_id: str) -> None:
    _save(user_id, twofa_method=None, totp_secret=None, twofa_phone=None, last_totp_step=None, recovery_codes=None)


def regenerate_recovery_codes(user_id: str) -> list[str]:
    codes = _new_recovery_codes()
    _save(user_id, recovery_codes=json.dumps([_sha256(c) for c in codes]))
    return codes


def mask_phone(number: str) -> str:
    return f"{number[:5]}•••{number[-3:]}" if number and len(number) > 8 else "•••"


# --- Sign-in challenge -------------------------------------------------------------------------------------


def send_login_code(user_id: str) -> tuple[bool, str]:
    from data.notify import send_sms

    st = status(user_id)
    if st["method"] != "sms" or not st["phone"]:
        return False, "SMS verification isn't set up for this account."
    code = _store_code(user_id, "login")
    ok, msg = send_sms(st["phone"], f"Kenya Invest sign-in code: {code}. It expires in {SMS_CODE_MINUTES} minutes. "
                                    "Don't share it with anyone.")
    return (True, f"Code sent to {mask_phone(st['phone'])}.") if ok else (False, msg)


def verify_login(user_id: str, code: str, recovery: bool = False) -> tuple[bool, str]:
    """Check a sign-in code (authenticator, SMS or recovery). Wrong codes count toward the account lockout."""
    from data.auth import record_failed_attempt

    with engine().connect() as conn:
        row = conn.execute(
            text("SELECT twofa_method, totp_secret, last_totp_step, recovery_codes FROM users WHERE id = :u"),
            {"u": user_id},
        ).first()
    if row is None or not row[0]:
        return True, "No two-step verification on this account."
    method, secret, last_step, recovery_json = row

    if recovery:
        hashes = json.loads(recovery_json or "[]")
        h = _sha256(_normalize_recovery(code))
        if h in hashes:
            hashes.remove(h)
            _save(user_id, recovery_codes=json.dumps(hashes))
            return True, f"Recovery code accepted — {len(hashes)} left. Generate new ones in the sidebar if you're low."
    elif method == "totp":
        step = match_totp(secret, code, last_step)
        if step is not None:
            _save(user_id, last_totp_step=step)
            return True, "Verified."
    elif method == "sms" and _check_code(user_id, "login", code):
        return True, "Verified."

    locked = record_failed_attempt(user_id)
    return False, locked or "That code is incorrect or has expired."
