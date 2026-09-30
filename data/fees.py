"""NSE equity trading charges applied to every simulated buy and sell.

Rates are typical published NSE retail equity charges (as a % of consideration).
They vary by broker — check your own broker's tariff before relying on them.
"""

from __future__ import annotations

# (label, rate as a fraction of consideration)
CHARGES = [
    ("Brokerage commission", 0.0150),
    ("NSE transaction levy", 0.0012),
    ("CMA levy", 0.0012),
    ("CDSC fee", 0.0008),
    ("Investor Compensation Fund", 0.0001),
]
VAT_ON_COMMISSION = 0.16  # VAT is charged on the brokerage commission only

TOTAL_RATE = sum(rate for _, rate in CHARGES) + CHARGES[0][1] * VAT_ON_COMMISSION


def fee_breakdown(consideration: float) -> list[tuple[str, float]]:
    rows = [(label, round(consideration * rate, 2)) for label, rate in CHARGES]
    rows.append(("VAT on commission (16%)", round(consideration * CHARGES[0][1] * VAT_ON_COMMISSION, 2)))
    return rows


def total_fees(consideration: float) -> float:
    return round(sum(amount for _, amount in fee_breakdown(consideration)), 2)
