"""Illustrative daily price history for the trading chart.

There is no free historical NSE price feed wired into this app, so the chart
history is *reconstructed*: yearly anchor prices are backed out from each
company's per-financial-year returns in data/companies.json, and the days in
between are filled with a deterministic random walk (a Brownian bridge, seeded
by the company name) so the path always passes exactly through the anchors and
ends at today's market price. It is for visual context only, not real data.
"""

from __future__ import annotations

import hashlib
from datetime import date

import numpy as np
import pandas as pd

DAILY_VOL = 0.018  # ~1.8% daily volatility for the simulated wiggle between anchors


def _seed(company: str) -> int:
    return int(hashlib.sha256(company.encode()).hexdigest()[:8], 16)


def _anchors(company: dict, return_years: list[str], today: date) -> list[tuple[pd.Timestamp, float]]:
    """(date, price) points, oldest first, ending at today's market price."""
    price = float(company["marketPrice"])
    points = [(pd.Timestamp(today), price)]

    # Treat the recorded previous price (if any) as the close of the last financial year.
    last_fy_year = int(return_years[-1].removeprefix("FY"))
    fy_close = float(company.get("previousPrice") or price)
    points.append((pd.Timestamp(last_fy_year, 12, 31), fy_close))

    # Walk backwards through the yearly returns: close(Y-1) = close(Y) / (1 + return(Y)).
    for fy in reversed(return_years):
        year = int(fy.removeprefix("FY"))
        fy_close = fy_close / (1 + company["returns"][fy] / 100)
        points.append((pd.Timestamp(year - 1, 12, 31), fy_close))

    return sorted(points)


def build_history(company: dict, return_years: list[str], today: date | None = None) -> pd.DataFrame:
    """Daily OHLC rows (business days) from the first anchor up to today."""
    today = today or date.today()
    anchors = _anchors(company, return_years, today)
    rng = np.random.default_rng(_seed(company["company"]))

    frames = []
    for (d0, p0), (d1, p1) in zip(anchors, anchors[1:]):
        days = pd.bdate_range(d0, d1)
        n = len(days)
        if n < 2:
            continue
        t = np.linspace(0, 1, n)
        steps = rng.normal(0, DAILY_VOL, n)
        walk = np.cumsum(steps) - steps[0]
        bridge = walk - t * walk[-1]  # pinned to 0 at both ends
        log_close = np.log(p0) + t * (np.log(p1) - np.log(p0)) + bridge
        frames.append(pd.DataFrame({"date": days, "close": np.exp(log_close)}))

    hist = pd.concat(frames).drop_duplicates("date", keep="last").reset_index(drop=True)
    hist["open"] = hist["close"].shift(1).fillna(hist["close"])
    spread = np.abs(rng.normal(0, DAILY_VOL / 2, len(hist)))
    hist["high"] = hist[["open", "close"]].max(axis=1) * (1 + spread)
    hist["low"] = hist[["open", "close"]].min(axis=1) * (1 - spread)
    return hist


def resample(hist: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate daily candles to weekly ('W') or monthly ('ME') candles."""
    out = (
        hist.set_index("date")
        .resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    return out
