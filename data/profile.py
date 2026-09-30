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

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass
class ProfileResult:
    ok: bool
    message: str


def _default_profile() -> dict:
    return {
        "fullName": "",
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
        "lastUpdated": None,
    }


def load_profile() -> dict:
    if not PROFILE_PATH.exists():
        return _default_profile()
    # Merge over defaults so profiles saved by older versions still get new fields.
    return {**_default_profile(), **json.loads(PROFILE_PATH.read_text())}


def save_profile(profile: dict) -> None:
    PROFILE_PATH.write_text(json.dumps(profile, indent=2))


def update_profile(profile: dict, updates: dict) -> ProfileResult:
    name = updates.get("fullName", "").strip()
    email = updates.get("email", "").strip()
    if not name:
        return ProfileResult(False, "Full name is required.")
    if email and not _EMAIL_RE.match(email):
        return ProfileResult(False, f"'{email}' doesn't look like a valid email address.")
    if updates.get("monthlyBudget", 0.0) < 0:
        return ProfileResult(False, "Monthly investment budget can't be negative.")

    profile.update({k: v.strip() if isinstance(v, str) else v for k, v in updates.items()})
    profile["lastUpdated"] = datetime.now().isoformat(timespec="seconds")
    save_profile(profile)
    return ProfileResult(True, "Profile saved.")


def reset_profile() -> None:
    if PROFILE_PATH.exists():
        PROFILE_PATH.unlink()
