"""Simulated paper-trading portfolio for the Kenya Investment Explorer.

Phase 2 of the live-data + trading roadmap: a single local virtual portfolio
(no auth, no multi-user support) that lets the user "buy" and "sell" shares
against the prices in data/companies.json, tracked in data/portfolio.json.

This is NOT real brokered trading — no order is ever sent anywhere, no real
money moves. See data/fetch_prices.py for where the prices this trades
against come from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PORTFOLIO_PATH = Path(__file__).parent / "portfolio.json"
STARTING_CASH = 1_000_000.0  # KES — arbitrary virtual starting balance


def _default_portfolio() -> dict:
    return {
        "startingCash": STARTING_CASH,
        "cash": STARTING_CASH,
        "netDeposits": 0.0,
        "holdings": [],
        "trades": [],
        "openOrders": [],
    }


def load_portfolio() -> dict:
    if not PORTFOLIO_PATH.exists():
        portfolio = _default_portfolio()
        save_portfolio(portfolio)
        return portfolio
    portfolio = json.loads(PORTFOLIO_PATH.read_text())
    if "netDeposits" not in portfolio:
        portfolio["netDeposits"] = 0.0
        save_portfolio(portfolio)
    if "openOrders" not in portfolio:
        portfolio["openOrders"] = []
        save_portfolio(portfolio)
    return portfolio


def save_portfolio(portfolio: dict) -> None:
    PORTFOLIO_PATH.write_text(json.dumps(portfolio, indent=2))


def reset_portfolio() -> dict:
    portfolio = _default_portfolio()
    save_portfolio(portfolio)
    return portfolio


def _find_holding(portfolio: dict, company: str) -> dict | None:
    for h in portfolio["holdings"]:
        if h["company"] == company:
            return h
    return None


@dataclass
class TradeResult:
    ok: bool
    message: str


def buy(portfolio: dict, company: str, shares: float, price: float) -> TradeResult:
    if shares <= 0:
        return TradeResult(False, "Shares must be greater than zero.")
    cost = shares * price
    if cost > portfolio["cash"]:
        return TradeResult(False, f"Insufficient cash: need KES {cost:,.2f}, have KES {portfolio['cash']:,.2f}.")

    holding = _find_holding(portfolio, company)
    if holding is None:
        portfolio["holdings"].append({"company": company, "shares": shares, "avgCost": price})
    else:
        total_shares = holding["shares"] + shares
        holding["avgCost"] = (holding["avgCost"] * holding["shares"] + cost) / total_shares
        holding["shares"] = total_shares

    portfolio["cash"] -= cost
    _log_trade(portfolio, company, "BUY", shares, price, cost)
    save_portfolio(portfolio)
    return TradeResult(True, f"Bought {shares:g} share(s) of {company} at KES {price:,.2f} for KES {cost:,.2f}.")


def sell(portfolio: dict, company: str, shares: float, price: float) -> TradeResult:
    if shares <= 0:
        return TradeResult(False, "Shares must be greater than zero.")
    holding = _find_holding(portfolio, company)
    if holding is None or shares > holding["shares"]:
        held = holding["shares"] if holding else 0
        return TradeResult(False, f"You only hold {held:g} share(s) of {company}.")

    proceeds = shares * price
    holding["shares"] -= shares
    if holding["shares"] <= 0:
        portfolio["holdings"].remove(holding)

    portfolio["cash"] += proceeds
    _log_trade(portfolio, company, "SELL", shares, price, proceeds)
    save_portfolio(portfolio)
    return TradeResult(True, f"Sold {shares:g} share(s) of {company} at KES {price:,.2f} for KES {proceeds:,.2f}.")


def deposit(portfolio: dict, amount: float) -> TradeResult:
    if amount <= 0:
        return TradeResult(False, "Deposit amount must be greater than zero.")
    portfolio["cash"] += amount
    portfolio["netDeposits"] = portfolio.get("netDeposits", 0.0) + amount
    _log_trade(portfolio, "CASH", "DEPOSIT", 0.0, 0.0, amount)
    save_portfolio(portfolio)
    return TradeResult(True, f"Deposited KES {amount:,.2f}.")


def withdraw(portfolio: dict, amount: float) -> TradeResult:
    if amount <= 0:
        return TradeResult(False, "Withdrawal amount must be greater than zero.")
    if amount > portfolio["cash"]:
        return TradeResult(False, f"Insufficient cash: have KES {portfolio['cash']:,.2f}.")
    portfolio["cash"] -= amount
    portfolio["netDeposits"] = portfolio.get("netDeposits", 0.0) - amount
    _log_trade(portfolio, "CASH", "WITHDRAWAL", 0.0, 0.0, amount)
    save_portfolio(portfolio)
    return TradeResult(True, f"Withdrew KES {amount:,.2f}.")


def _log_trade(portfolio: dict, company: str, action: str, shares: float, price: float, total: float) -> None:
    portfolio["trades"].append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "company": company,
            "action": action,
            "shares": shares,
            "price": price,
            "total": total,
            "cashAfter": portfolio["cash"],
        }
    )


def holdings_market_value(portfolio: dict, price_by_company: dict[str, float]) -> float:
    return sum(h["shares"] * price_by_company.get(h["company"], h["avgCost"]) for h in portfolio["holdings"])


# --- Limit orders ---------------------------------------------------------------
# A limit order waits in portfolio["openOrders"] until the market price reaches the
# limit: a BUY fills when price <= limit, a SELL when price >= limit. Fills happen
# at the (equal-or-better) market price whenever process_open_orders() runs (on every app refresh).


def place_limit_order(portfolio: dict, side: str, company: str, shares: float, limit: float) -> TradeResult:
    if shares <= 0 or limit <= 0:
        return TradeResult(False, "Shares and limit price must be greater than zero.")
    if side == "BUY" and shares * limit > portfolio["cash"]:
        return TradeResult(
            False, f"Insufficient cash: need KES {shares * limit:,.2f}, have KES {portfolio['cash']:,.2f}."
        )
    if side == "SELL":
        holding = _find_holding(portfolio, company)
        held = holding["shares"] if holding else 0
        if shares > held:
            return TradeResult(False, f"You only hold {held:g} share(s) of {company}.")

    order_id = max((o["id"] for o in portfolio["openOrders"]), default=0) + 1
    portfolio["openOrders"].append(
        {
            "id": order_id,
            "placedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "company": company,
            "side": side,
            "shares": shares,
            "limit": limit,
        }
    )
    save_portfolio(portfolio)
    return TradeResult(True, f"Limit {side.lower()} placed: {shares:g} {company} @ KES {limit:,.2f}.")


def cancel_order(portfolio: dict, order_id: int) -> TradeResult:
    before = len(portfolio["openOrders"])
    portfolio["openOrders"] = [o for o in portfolio["openOrders"] if o["id"] != order_id]
    if len(portfolio["openOrders"]) == before:
        return TradeResult(False, "Order not found.")
    save_portfolio(portfolio)
    return TradeResult(True, f"Order #{order_id} cancelled.")


def process_open_orders(portfolio: dict, price_by_company: dict[str, float]) -> list[TradeResult]:
    """Fill any open limit orders whose limit has been reached. Returns one result per fill/cancel."""
    results = []
    for order in list(portfolio["openOrders"]):
        price = price_by_company.get(order["company"])
        if price is None:
            continue
        reached = price <= order["limit"] if order["side"] == "BUY" else price >= order["limit"]
        if not reached:
            continue
        portfolio["openOrders"].remove(order)
        trade = buy if order["side"] == "BUY" else sell
        # Fill at the current market price, which is at or better than the limit.
        result = trade(portfolio, order["company"], order["shares"], price)
        if not result.ok:
            result = TradeResult(False, f"Limit order #{order['id']} cancelled — {result.message}")
        results.append(result)
    if results:
        save_portfolio(portfolio)
    return results
