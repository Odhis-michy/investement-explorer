"""Watchlist and price-alert tracking for the Kenya Investment Explorer.

One watchlist per signed-in user, stored in the database (see data/storage.py).
"""

from __future__ import annotations

from data.storage import get_doc, put_doc


def _default_watchlist() -> dict:
    return {"watching": [], "alerts": []}


def load_watchlist() -> dict:
    wl = get_doc("watchlist")
    if wl is None:
        wl = _default_watchlist()
        save_watchlist(wl)
    return wl


def save_watchlist(wl: dict) -> None:
    put_doc("watchlist", wl)


def add_company(wl: dict, company: str) -> None:
    if company not in wl["watching"]:
        wl["watching"].append(company)
        save_watchlist(wl)


def remove_company(wl: dict, company: str) -> None:
    if company in wl["watching"]:
        wl["watching"].remove(company)
    wl["alerts"] = [a for a in wl["alerts"] if a["company"] != company]
    save_watchlist(wl)


def set_alert(wl: dict, company: str, target_price: float, direction: str) -> None:
    wl["alerts"] = [a for a in wl["alerts"] if a["company"] != company]
    wl["alerts"].append({"company": company, "target": target_price, "direction": direction})
    save_watchlist(wl)


def remove_alert(wl: dict, company: str) -> None:
    wl["alerts"] = [a for a in wl["alerts"] if a["company"] != company]
    save_watchlist(wl)


def check_alerts(wl: dict, price_by_company: dict[str, float]) -> list[dict]:
    triggered = []
    for a in wl["alerts"]:
        price = price_by_company.get(a["company"])
        if price is None:
            continue
        if a["direction"] == "above" and price >= a["target"]:
            triggered.append({**a, "current": price})
        elif a["direction"] == "below" and price <= a["target"]:
            triggered.append({**a, "current": price})
    return triggered
