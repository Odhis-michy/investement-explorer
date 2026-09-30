"""Simulated paper-trading portfolio for the Kenya Investment Explorer.

A virtual portfolio per signed-in user that lets them "buy" and "sell" shares against
the prices in data/companies.json, stored in the database (see data/storage.py). Covers:

- market orders, with NSE trading charges from data/fees.py
- limit, stop-loss, take-profit and stop-limit orders, day or good-till-cancelled,
  with a full order history (FILLED / CANCELLED / EXPIRED / REJECTED)
- auto-invest plans that buy a fixed KES amount weekly or monthly
- dividends credited from data/corporate_actions.json (net of withholding tax)

This is NOT real brokered trading — no order is ever sent anywhere, no real
money moves. See data/fetch_prices.py for where the prices this trades
against come from.
"""

from __future__ import annotations

import calendar
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from data.fees import TOTAL_RATE, total_fees
from data.storage import get_doc, put_doc

STARTING_CASH = 1_000_000.0  # KES — arbitrary virtual starting balance
DIVIDEND_WHT = 0.05  # withholding tax on dividends for Kenyan residents

ORDER_TYPES = {
    "MARKET": "Market",
    "LIMIT": "Limit",
    "STOP_LOSS": "Stop-loss",
    "TAKE_PROFIT": "Take-profit",
    "STOP_LIMIT": "Stop-limit",
}


def _default_portfolio() -> dict:
    return {
        "startingCash": STARTING_CASH,
        "cash": STARTING_CASH,
        "netDeposits": 0.0,
        "holdings": [],
        "trades": [],
        "openOrders": [],
        "orderHistory": [],
        "autoInvest": [],
        "dividendsCredited": [],
    }


def load_portfolio() -> dict:
    portfolio = get_doc("portfolio")
    if portfolio is None:
        portfolio = _default_portfolio()
        save_portfolio(portfolio)
        return portfolio
    # Upgrade portfolios saved by older versions of the app.
    for key, value in _default_portfolio().items():
        portfolio.setdefault(key, value)
    for o in portfolio["openOrders"]:
        o.setdefault("type", "LIMIT")
        o.setdefault("tif", "GTC")
        o.setdefault("stop", None)
        o.setdefault("placedDate", o["placedAt"][:10])
    return portfolio


def save_portfolio(portfolio: dict) -> None:
    put_doc("portfolio", portfolio)


def reset_portfolio() -> dict:
    portfolio = _default_portfolio()
    save_portfolio(portfolio)
    return portfolio


def _find_holding(portfolio: dict, company: str) -> dict | None:
    for h in portfolio["holdings"]:
        if h["company"] == company:
            return h
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TradeResult:
    ok: bool
    message: str


# --- Market execution ---------------------------------------------------------------


def buy(portfolio: dict, company: str, shares: float, price: float, note: str = "Market order") -> TradeResult:
    if shares <= 0:
        return TradeResult(False, "Shares must be greater than zero.")
    gross = shares * price
    fees = total_fees(gross)
    cost = gross + fees
    if cost > portfolio["cash"]:
        return TradeResult(
            False, f"Insufficient cash: need KES {cost:,.2f} incl. fees, have KES {portfolio['cash']:,.2f}."
        )

    holding = _find_holding(portfolio, company)
    if holding is None:
        portfolio["holdings"].append({"company": company, "shares": shares, "avgCost": cost / shares})
    else:
        total_shares = holding["shares"] + shares
        holding["avgCost"] = (holding["avgCost"] * holding["shares"] + cost) / total_shares
        holding["shares"] = total_shares

    portfolio["cash"] -= cost
    _log_trade(portfolio, company, "BUY", shares, price, gross, fees, cost, note)
    save_portfolio(portfolio)
    return TradeResult(
        True, f"Bought {shares:g} {company} at KES {price:,.2f} — KES {cost:,.2f} incl. KES {fees:,.2f} fees."
    )


def sell(portfolio: dict, company: str, shares: float, price: float, note: str = "Market order") -> TradeResult:
    if shares <= 0:
        return TradeResult(False, "Shares must be greater than zero.")
    holding = _find_holding(portfolio, company)
    if holding is None or shares > holding["shares"]:
        held = holding["shares"] if holding else 0
        return TradeResult(False, f"You only hold {held:g} share(s) of {company}.")

    gross = shares * price
    fees = total_fees(gross)
    proceeds = gross - fees
    holding["shares"] -= shares
    if holding["shares"] <= 0:
        portfolio["holdings"].remove(holding)

    portfolio["cash"] += proceeds
    _log_trade(portfolio, company, "SELL", shares, price, gross, fees, proceeds, note)
    save_portfolio(portfolio)
    return TradeResult(
        True, f"Sold {shares:g} {company} at KES {price:,.2f} — KES {proceeds:,.2f} after KES {fees:,.2f} fees."
    )


def deposit(portfolio: dict, amount: float) -> TradeResult:
    if amount <= 0:
        return TradeResult(False, "Deposit amount must be greater than zero.")
    portfolio["cash"] += amount
    portfolio["netDeposits"] = portfolio.get("netDeposits", 0.0) + amount
    _log_trade(portfolio, "CASH", "DEPOSIT", 0.0, 0.0, amount, 0.0, amount, "Cash deposit")
    save_portfolio(portfolio)
    return TradeResult(True, f"Deposited KES {amount:,.2f}.")


def withdraw(portfolio: dict, amount: float) -> TradeResult:
    if amount <= 0:
        return TradeResult(False, "Withdrawal amount must be greater than zero.")
    if amount > portfolio["cash"]:
        return TradeResult(False, f"Insufficient cash: have KES {portfolio['cash']:,.2f}.")
    portfolio["cash"] -= amount
    portfolio["netDeposits"] = portfolio.get("netDeposits", 0.0) - amount
    _log_trade(portfolio, "CASH", "WITHDRAWAL", 0.0, 0.0, amount, 0.0, amount, "Cash withdrawal")
    save_portfolio(portfolio)
    return TradeResult(True, f"Withdrew KES {amount:,.2f}.")


def _log_trade(
    portfolio: dict,
    company: str,
    action: str,
    shares: float,
    price: float,
    gross: float,
    fees: float,
    total: float,
    note: str,
) -> None:
    portfolio["trades"].append(
        {
            "timestamp": _now(),
            "company": company,
            "action": action,
            "shares": shares,
            "price": price,
            "gross": round(gross, 2),
            "fees": round(fees, 2),
            "total": round(total, 2),
            "cashAfter": round(portfolio["cash"], 2),
            "note": note,
        }
    )


def holdings_market_value(portfolio: dict, price_by_company: dict[str, float]) -> float:
    return sum(h["shares"] * price_by_company.get(h["company"], h["avgCost"]) for h in portfolio["holdings"])


def shares_held_on(portfolio: dict, company: str, day: str) -> float:
    """Shares of `company` held at the end of `day` (YYYY-MM-DD), rebuilt from the trade log."""
    held = 0.0
    for t in portfolio["trades"]:
        if t["company"] == company and t["timestamp"][:10] <= day:
            held += t["shares"] if t["action"] == "BUY" else -t["shares"] if t["action"] == "SELL" else 0.0
    return held


# --- Orders -------------------------------------------------------------------------------
# Resting orders wait in portfolio["openOrders"] and are checked on every app run:
#   LIMIT        BUY fills when price <= limit, SELL when price >= limit
#   STOP_LOSS    SELL at market when price <= stop
#   TAKE_PROFIT  SELL at market when price >= stop (the target)
#   STOP_LIMIT   when price crosses the stop (BUY: >= stop, SELL: <= stop) it becomes a LIMIT
# Fills happen at the current market price (equal to or better than the order's price).
# DAY orders expire if still open after the day they were placed; GTC orders stay open.


def _next_order_id(portfolio: dict) -> int:
    ids = [o["id"] for o in portfolio["openOrders"] + portfolio["orderHistory"]]
    return max(ids, default=0) + 1


def _close_order(portfolio: dict, order: dict, status: str, fill_price: float | None = None, note: str = "") -> None:
    if order in portfolio["openOrders"]:
        portfolio["openOrders"].remove(order)
    portfolio["orderHistory"].append(
        {**order, "status": status, "closedAt": _now(), "fillPrice": fill_price, "note": note}
    )


def place_order(
    portfolio: dict,
    side: str,
    company: str,
    shares: float,
    order_type: str,
    price_now: float,
    limit: float | None = None,
    stop: float | None = None,
    tif: str = "GTC",
) -> TradeResult:
    side = side.upper()
    if shares <= 0:
        return TradeResult(False, "Enter how many shares to trade.")

    order = {
        "id": _next_order_id(portfolio),
        "placedAt": _now(),
        "placedDate": date.today().isoformat(),
        "company": company,
        "side": side,
        "type": order_type,
        "shares": shares,
        "limit": limit,
        "stop": stop,
        "tif": tif,
    }

    if order_type == "MARKET":
        trade = buy if side == "BUY" else sell
        result = trade(portfolio, company, shares, price_now)
        _close_order(portfolio, order, "FILLED" if result.ok else "REJECTED",
                     price_now if result.ok else None, "" if result.ok else result.message)
        save_portfolio(portfolio)
        return result

    # Validate the order before it rests on the book.
    if order_type in ("STOP_LOSS", "TAKE_PROFIT") and side != "SELL":
        return TradeResult(False, f"{ORDER_TYPES[order_type]} orders are for selling shares you hold.")
    if order_type in ("LIMIT", "STOP_LIMIT") and not limit:
        return TradeResult(False, "Enter a limit price.")
    if order_type in ("STOP_LOSS", "TAKE_PROFIT", "STOP_LIMIT") and not stop:
        return TradeResult(False, "Enter a stop / trigger price.")
    if order_type == "STOP_LOSS" and stop >= price_now:
        return TradeResult(False, f"A stop-loss must be below the current price (KES {price_now:,.2f}).")
    if order_type == "TAKE_PROFIT" and stop <= price_now:
        return TradeResult(False, f"A take-profit target must be above the current price (KES {price_now:,.2f}).")
    if order_type == "STOP_LIMIT" and side == "BUY" and stop <= price_now:
        return TradeResult(False, f"A buy stop must be above the current price (KES {price_now:,.2f}).")
    if order_type == "STOP_LIMIT" and side == "SELL" and stop >= price_now:
        return TradeResult(False, f"A sell stop must be below the current price (KES {price_now:,.2f}).")

    if side == "BUY":
        est = shares * (limit or stop)
        if est + total_fees(est) > portfolio["cash"]:
            return TradeResult(False, f"Insufficient cash for this order (about KES {est + total_fees(est):,.2f}).")
    else:
        holding = _find_holding(portfolio, company)
        held = holding["shares"] if holding else 0
        if shares > held:
            return TradeResult(False, f"You only hold {held:g} share(s) of {company}.")

    portfolio["openOrders"].append(order)
    save_portfolio(portfolio)
    placed = TradeResult(
        True, f"{ORDER_TYPES[order_type]} {side.lower()} #{order['id']} placed ({tif}): {shares:g} {company}."
    )
    # A marketable order (e.g. a buy limit above the current price) fills straight away.
    fills = process_open_orders(portfolio, {company: price_now})
    return fills[-1] if fills else placed


def cancel_order(portfolio: dict, order_id: int) -> TradeResult:
    order = next((o for o in portfolio["openOrders"] if o["id"] == order_id), None)
    if order is None:
        return TradeResult(False, "Order not found.")
    _close_order(portfolio, order, "CANCELLED", note="Cancelled by you")
    save_portfolio(portfolio)
    return TradeResult(True, f"Order #{order_id} cancelled.")


def process_open_orders(
    portfolio: dict, price_by_company: dict[str, float], today: date | None = None
) -> list[TradeResult]:
    """Expire, trigger and fill resting orders. Returns one result per event."""
    today_iso = (today or date.today()).isoformat()
    results = []
    for order in list(portfolio["openOrders"]):
        label = f"{ORDER_TYPES[order['type']]} {order['side'].lower()} #{order['id']}"
        if order["tif"] == "DAY" and order["placedDate"] < today_iso:
            _close_order(portfolio, order, "EXPIRED", note="Day order not filled")
            results.append(TradeResult(False, f"{label} for {order['company']} expired unfilled."))
            continue

        price = price_by_company.get(order["company"])
        if price is None:
            continue

        side, otype = order["side"], order["type"]
        if otype == "STOP_LIMIT" and not order.get("triggered"):
            if (side == "BUY" and price >= order["stop"]) or (side == "SELL" and price <= order["stop"]):
                order["triggered"] = True
                results.append(TradeResult(True, f"{label} triggered at KES {price:,.2f} — now a limit order."))
            else:
                continue

        if otype in ("LIMIT", "STOP_LIMIT"):
            fill = price <= order["limit"] if side == "BUY" else price >= order["limit"]
        elif otype == "STOP_LOSS":
            fill = price <= order["stop"]
        else:  # TAKE_PROFIT
            fill = price >= order["stop"]
        if not fill:
            continue

        trade = buy if side == "BUY" else sell
        result = trade(portfolio, order["company"], order["shares"], price, note=label)
        if result.ok:
            _close_order(portfolio, order, "FILLED", price)
            results.append(TradeResult(True, f"{label} filled — {result.message}"))
        else:
            _close_order(portfolio, order, "REJECTED", note=result.message)
            results.append(TradeResult(False, f"{label} rejected — {result.message}"))

    if results:
        save_portfolio(portfolio)
    return results


# --- Auto-invest ---------------------------------------------------------------------------


def _add_period(day: date, frequency: str) -> date:
    if frequency == "Weekly":
        return day + timedelta(days=7)
    month = day.month % 12 + 1
    year = day.year + (day.month == 12)
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def add_auto_invest(portfolio: dict, company: str, amount: float, frequency: str, start: date) -> TradeResult:
    if amount <= 0:
        return TradeResult(False, "Amount must be greater than zero.")
    plan_id = max((p["id"] for p in portfolio["autoInvest"]), default=0) + 1
    portfolio["autoInvest"].append(
        {
            "id": plan_id,
            "company": company,
            "amount": amount,
            "frequency": frequency,
            "nextRun": start.isoformat(),
            "active": True,
            "lastRun": None,
        }
    )
    save_portfolio(portfolio)
    return TradeResult(True, f"Auto-invest set: KES {amount:,.0f} of {company} {frequency.lower()}, from {start}.")


def set_auto_invest_active(portfolio: dict, plan_id: int, active: bool) -> None:
    for p in portfolio["autoInvest"]:
        if p["id"] == plan_id:
            p["active"] = active
    save_portfolio(portfolio)


def remove_auto_invest(portfolio: dict, plan_id: int) -> None:
    portfolio["autoInvest"] = [p for p in portfolio["autoInvest"] if p["id"] != plan_id]
    save_portfolio(portfolio)


def run_auto_invest(portfolio: dict, price_by_company: dict[str, float], today: date | None = None) -> list[TradeResult]:
    """Run every active plan that is due (once per due date; missed periods are not back-filled)."""
    today = today or date.today()
    results = []
    for plan in portfolio["autoInvest"]:
        if not plan["active"] or date.fromisoformat(plan["nextRun"]) > today:
            continue
        price = price_by_company.get(plan["company"])
        if price:
            shares = math.floor(plan["amount"] / (price * (1 + TOTAL_RATE)))
            if shares >= 1:
                results.append(buy(portfolio, plan["company"], float(shares), price, note=f"Auto-invest #{plan['id']}"))
            else:
                results.append(TradeResult(False, f"Auto-invest #{plan['id']}: KES {plan['amount']:,.0f} "
                                                  f"is less than one share of {plan['company']}."))
        plan["lastRun"] = today.isoformat()
        nxt = date.fromisoformat(plan["nextRun"])
        while nxt <= today:
            nxt = _add_period(nxt, plan["frequency"])
        plan["nextRun"] = nxt.isoformat()
    if results:
        save_portfolio(portfolio)
    return results


# --- Dividends -------------------------------------------------------------------------------


def credit_dividends(portfolio: dict, events: list[dict], today: date | None = None) -> list[TradeResult]:
    """Pay any cash dividends whose payment date has passed, to shares held at book closure."""
    today_iso = (today or date.today()).isoformat()
    results = []
    changed = False
    for ev in events:
        if not ev.get("dps") or ev["id"] in portfolio["dividendsCredited"] or ev["paymentDate"] > today_iso:
            continue
        portfolio["dividendsCredited"].append(ev["id"])
        changed = True
        held = shares_held_on(portfolio, ev["company"], ev["bookClosure"])
        if held <= 0:
            continue
        gross = held * ev["dps"]
        tax = round(gross * DIVIDEND_WHT, 2)
        portfolio["cash"] += gross - tax
        _log_trade(portfolio, ev["company"], "DIVIDEND", held, ev["dps"], gross, tax, gross - tax, ev["type"])
        results.append(TradeResult(True, f"Dividend from {ev['company']}: KES {gross - tax:,.2f} after 5% tax."))
    if changed:
        save_portfolio(portfolio)
    return results
