"""Background price-alert job.

Runs independently of anyone having the app open:
- every ALERT_CHECK_MINUTES (default 2) it checks watchlist alerts against the
  latest prices and sends any newly triggered ones by email / SMS;
- every PRICE_REFRESH_MINUTES (default 15), during NSE trading hours only, it
  refreshes live prices first so alerts react to real market moves.

Used two ways:
- inside the Streamlit server as a daemon thread (see start_worker / app.py),
  controlled by BACKGROUND_ALERTS=1 (default) or 0;
- as a standalone process: `python alert_worker.py` (see that file).
Run only one of the two against the same data folder.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, time as dtime, timedelta, timezone
from pathlib import Path

EAT = timezone(timedelta(hours=3))  # Nairobi time (no daylight saving)
MARKET_OPEN, MARKET_CLOSE = dtime(9, 30), dtime(15, 0)
COMPANIES_PATH = Path(__file__).parent / "companies.json"

# One lock per process so the page-load check and the worker never send the same alert twice.
LOCK = threading.Lock()


def check_minutes() -> float:
    return float(os.environ.get("ALERT_CHECK_MINUTES", 2))


def refresh_minutes() -> float:
    return float(os.environ.get("PRICE_REFRESH_MINUTES", 15))


def market_open(now: datetime | None = None) -> bool:
    now = (now or datetime.now(EAT)).astimezone(EAT)
    return now.weekday() < 5 and MARKET_OPEN <= now.time() < MARKET_CLOSE


def current_prices() -> dict[str, float]:
    raw = json.loads(COMPANIES_PATH.read_text())
    return {c["company"]: float(c["marketPrice"]) for c in raw["companies"]}


def check_and_notify(prices: dict[str, float] | None = None) -> list[str]:
    """Send any newly triggered watchlist alerts to the channels the user turned on."""
    from data.notify import alert_targets, send_alerts
    from data.profile import load_profile
    from data.watchlist import check_alerts, load_watchlist, save_watchlist

    with LOCK:
        email_to, sms_to = alert_targets(load_profile())
        if not (email_to or sms_to):
            return []
        wl = load_watchlist()
        messages = send_alerts(wl, check_alerts(wl, prices or current_prices()), email_to, sms_to)
        if messages:
            save_watchlist(wl)
        return messages


def run_cycle(refresh: bool) -> tuple[list[str], bool]:
    """One pass: optionally refresh live prices (market hours only), then check alerts.

    Returns (messages, refreshed).
    """
    messages, refreshed = [], False
    if refresh and market_open():
        from data.fetch_prices import update_companies_json

        result = update_companies_json()
        messages.append(result.message)
        refreshed = True
    messages += check_and_notify()
    return messages, refreshed


def _stamp() -> str:
    return datetime.now(EAT).strftime("%Y-%m-%d %H:%M:%S EAT")


def worker_loop(status: dict, stop: threading.Event | None = None, on_message=None) -> None:
    """Run cycles forever (or until `stop` is set), recording progress in `status`.

    `on_message`, if given, is called with each timestamped message line (e.g. print).
    """
    last_refresh = 0.0
    while not (stop and stop.is_set()):
        due = time.monotonic() - last_refresh >= refresh_minutes() * 60
        try:
            messages, refreshed = run_cycle(refresh=due)
            if refreshed:
                last_refresh = time.monotonic()
                status["lastRefresh"] = _stamp()
            status["lastError"] = None
        except Exception as exc:  # noqa: BLE001 - keep the worker alive through any failure
            messages = [f"Background check failed: {exc}"]
            status["lastError"] = str(exc)
        status["lastRun"] = _stamp()
        status["runs"] = status.get("runs", 0) + 1
        status["marketOpen"] = market_open()
        lines = [f"{_stamp()} · {m}" for m in messages]
        if lines:
            status["recent"] = (lines + status.get("recent", []))[:20]
            if on_message:
                for line in lines:
                    on_message(line)
        wait = check_minutes() * 60
        status["nextRun"] = (datetime.now(EAT) + timedelta(seconds=wait)).strftime("%H:%M:%S EAT")
        if stop:
            stop.wait(wait)
        else:
            time.sleep(wait)


def start_worker() -> dict:
    """Start the daemon thread and return its live status dict."""
    status = {"running": True, "startedAt": _stamp(), "runs": 0, "recent": [], "lastRun": None,
              "lastRefresh": None, "nextRun": None, "lastError": None, "marketOpen": market_open()}
    threading.Thread(target=worker_loop, args=(status,), daemon=True, name="price-alert-worker").start()
    return status
