"""Simulated order book and recent trades for the trading view.

There is no live NSE order book feed in this app, so bids/asks and the trade
tape are generated around the current price, seeded by company and minute so
they look stable between refreshes but move over time. Illustrative only.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def tick_size(price: float) -> float:
    return 0.01 if price < 10 else 0.05


def _rng(company: str, salt: str) -> np.random.Generator:
    minute = datetime.now().strftime("%Y%m%d%H%M")
    return np.random.default_rng(int(hashlib.sha256(f"{company}|{minute}|{salt}".encode()).hexdigest()[:8], 16))


def order_book(company: str, price: float, levels: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(bids, asks) with columns price, shares, total — best prices first."""
    rng = _rng(company, "book")
    tick = tick_size(price)
    best_bid = round(price - tick, 2)
    best_ask = round(price + tick, 2)
    lot = max(100, int(20_000 / max(price, 1)) // 100 * 100)

    def side(start: float, step: float) -> pd.DataFrame:
        prices = [round(start + step * i * rng.integers(1, 3), 2) for i in range(levels)]
        prices = sorted(set(prices), reverse=step < 0)[:levels]
        shares = (rng.lognormal(0, 0.8, len(prices)) * lot).round(-2).clip(min=100)
        return pd.DataFrame({"price": prices, "shares": shares, "total": np.cumsum(shares)})

    return side(best_bid, -tick), side(best_ask, tick)


def recent_trades(company: str, price: float, count: int = 12) -> pd.DataFrame:
    rng = _rng(company, "tape")
    tick = tick_size(price)
    now = datetime.now()
    times = sorted((now - timedelta(seconds=int(s)) for s in rng.integers(0, 900, count)), reverse=True)
    moves = rng.integers(-2, 3, count) * tick
    lot = max(100, int(10_000 / max(price, 1)) // 100 * 100)
    return pd.DataFrame(
        {
            "time": [t.strftime("%H:%M:%S") for t in times],
            "price": np.round(price + moves, 2),
            "shares": (rng.lognormal(0, 0.7, count) * lot).round(-2).clip(min=100),
            "side": np.where(moves >= 0, "BUY", "SELL"),
        }
    )
