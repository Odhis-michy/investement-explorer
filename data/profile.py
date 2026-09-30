"""Investor profile for the Kenya Investment Explorer.

Single local profile (no auth, no multi-user support), tracked in
data/profile.json — mirrors the pattern used by data/watchlist.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROFILE_PATH = Path(__file__).parent / "profile.json"

RISK_LEVELS = ["Conservative", "Moderate", "Aggressive"]
HORIZONS = ["Short term (< 1 year)", "Medium term (1–5 years)", "Long term (5+ years)"]
EXPERIENCE_LEVELS = ["Beginner", "Intermediate", "Advanced"]
GOALS = [
    "Wealth growth",
    "Regular income (dividends/interest)",
    "Capital preservation",
    "Retirement",
    "Education fund",
    "Emergency savings",
    "Buying property",
]

ID_TYPES = ["National ID", "Passport", "Alien ID"]

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
# Kenyan National ID: 6-9 digits · Passport: 6-12 letters/digits (e.g. AK1234567) · Alien ID: 6-10 digits
_ID_RULES = {
    "National ID": (re.compile(r"^\d{6,9}$"), "a National ID number is 6-9 digits"),
    "Passport": (re.compile(r"^[A-Z0-9]{6,12}$"), "a passport number is 6-12 letters/digits, e.g. AK1234567"),
    "Alien ID": (re.compile(r"^\d{6,10}$"), "an Alien ID number is 6-10 digits"),
}


@dataclass
class ProfileResult:
    ok: bool
    message: str


def _default_profile() -> dict:
    return {
        "surname": "",
        "firstName": "",
        "secondName": "",
        "fullName": "",  # derived: "Surname First Second"
        "idType": ID_TYPES[0],
        "idNumber": "",
        "email": "",
        "phone": "",
        "location": "",
        "experience": EXPERIENCE_LEVELS[0],
        "riskTolerance": RISK_LEVELS[1],
        "horizon": HORIZONS[1],
        "goals": [],
        "preferredSectors": [],
        "monthlyBudget": 0.0,
        "bio": "",
        "emailAlerts": False,
        "smsAlerts": False,
        "language": "en",
        "lastUpdated": None,
    }


def load_profile() -> dict:
    if not PROFILE_PATH.exists():
        return _default_profile()
    # Merge over defaults so profiles saved by older versions still get new fields.
    profile = {**_default_profile(), **json.loads(PROFILE_PATH.read_text())}
    if profile["fullName"] and not (profile["surname"] or profile["firstName"]):
        # Older profiles stored one "First Middle Last" name — split it into the separate fields.
        parts = profile["fullName"].split()
        profile["firstName"] = parts[0]
        profile["surname"] = parts[-1] if len(parts) > 1 else ""
        profile["secondName"] = " ".join(parts[1:-1])
    return profile


def compose_full_name(surname: str, first: str, second: str) -> str:
    return " ".join(x for x in (surname, first, second) if x)


def normalize_id(number: str) -> str:
    return re.sub(r"[\s-]", "", number or "").upper()


def validate_id(id_type: str, number: str) -> str | None:
    """Error message if `number` isn't a plausible document number for `id_type`, else None."""
    pattern, rule = _ID_RULES.get(id_type, _ID_RULES["National ID"])
    return None if pattern.match(normalize_id(number)) else f"That doesn't look right — {rule}."


def mask_id(number: str) -> str:
    """Show only the last 4 characters, e.g. '•••• 5678'."""
    n = normalize_id(number)
    return f"•••• {n[-4:]}" if n else ""


def _valid_name(name: str) -> bool:
    return all(ch.isalpha() or ch in " '-." for ch in name)


def save_profile(profile: dict) -> None:
    PROFILE_PATH.write_text(json.dumps(profile, indent=2))


def update_profile(profile: dict, updates: dict) -> ProfileResult:
    surname = updates.get("surname", profile.get("surname", "")).strip()
    first = updates.get("firstName", profile.get("firstName", "")).strip()
    second = updates.get("secondName", profile.get("secondName", "")).strip()
    email = updates.get("email", profile.get("email", "")).strip()
    if not surname or not first:
        return ProfileResult(False, "Surname and first name are required.")
    if not all(_valid_name(n) for n in (surname, first, second)):
        return ProfileResult(False, "Names can only contain letters, spaces, hyphens and apostrophes.")
    if "idNumber" in updates:
        updates["idNumber"] = normalize_id(updates["idNumber"])
        if updates["idNumber"]:
            error = validate_id(updates.get("idType", profile.get("idType", ID_TYPES[0])), updates["idNumber"])
            if error:
                return ProfileResult(False, error)
    if email and not _EMAIL_RE.match(email):
        return ProfileResult(False, f"'{email}' doesn't look like a valid email address.")
    if updates.get("emailAlerts") and not email:
        return ProfileResult(False, "Add your email address to receive email alerts.")
    if updates.get("smsAlerts"):
        from data.notify import normalize_phone

        if not normalize_phone(updates.get("phone", "")):
            return ProfileResult(False, "Add a valid phone number (e.g. 0712 345 678) to receive SMS alerts.")
    if updates.get("monthlyBudget", 0.0) < 0:
        return ProfileResult(False, "Monthly investment budget can't be negative.")

    profile.update({k: v.strip() if isinstance(v, str) else v for k, v in updates.items()})
    profile["fullName"] = compose_full_name(surname, first, second)
    profile["lastUpdated"] = datetime.now().isoformat(timespec="seconds")
    save_profile(profile)
    return ProfileResult(True, "Profile saved.")


def reset_profile() -> None:
    if PROFILE_PATH.exists():
        PROFILE_PATH.unlink()


def set_preference(key: str, value) -> None:
    """Save a single app preference (e.g. language) without the full-profile validation."""
    profile = load_profile()
    profile[key] = value
    save_profile(profile)
