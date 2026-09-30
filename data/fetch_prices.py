"""Live NSE price fetcher for the Kenya Investment Explorer.

Scrapes free public NSE price data from afx.kwayisi.org (a public NSE data
aggregator — the NSE itself has no free public real-time API) and merges it
into data/companies.json, the same file the app's "Update Market Prices" tab
edits by hand.

On any fetch/parse failure, the existing companies.json is left untouched so
the app falls back to the last-known-good prices. Every attempt (success or
failure) is recorded in the JSON's top-level "lastFetch" block so the UI can
show a staleness indicator.

Run directly to refresh once: `python data/fetch_prices.py`
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_PATH = Path(__file__).parent / "companies.json"
SOURCE_URL = "https://afx.kwayisi.org/nse/"
REQUEST_TIMEOUT = 15
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# Maps each company name in companies.json to its NSE ticker on afx.kwayisi.org.
# Kept explicit (rather than fuzzy name-matching) so a scrape never silently
# mismatches a price to the wrong company.
TICKER_BY_COMPANY = {
    "Safaricom PLC": "SCOM",
    "Equity Group Holdings": "EQTY",
    "KCB Group": "KCB",
    "Co-operative Bank of Kenya": "COOP",
    "Absa Bank Kenya": "ABSA",
    "Stanbic Holdings": "SBIC",
    "NCBA Group": "NCBA",
    "East African Breweries (EABL)": "EABL",
    "British American Tobacco Kenya": "BAT",
    "Unga Group": "UNGA",
    "Bamburi Cement (Portland)": "BAMB",
    "Crown Paints Kenya": "CRWN",
    "Kenya Airways (KQ)": "KQ",
    "Nairobi Securities Exchange PLC": "NSE",
    "TPS Eastern Africa (Serena Hotels)": "TPSE",
    "Britam Holdings": "BRIT",
    "Jubilee Holdings": "JUB",
    "CIC Insurance Group": "CIC",
    "KenGen": "KEGN",
    "Kenya Power (KPLC)": "KPLC",
    "Total Energies Kenya": "TOTL",
    "Centum Investment Company": "CTUM",
    "Kakuzi PLC": "KUKZ",
    "Sasini PLC": "SASN",
    "Williamson Tea Kenya": "WTK",
    "Car & General (K) Ltd": "CGEN",
    "Nation Media Group": "NMG",
    "Standard Group PLC": "SGL",
    # Not tracked by afx.kwayisi.org as of 2026-08-30 — left on last-known price:
    # "ICDC Investment Company", "ILAM Fahari I-REIT"
}


@dataclass
class FetchResult:
    ok: bool
    message: str
    updated_count: int = 0
    skipped: list[str] | None = None


def _fetch_html() -> str:
    resp = requests.get(SOURCE_URL, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


def _parse_ticker_prices(html: str) -> dict[str, float]:
    """Parse the 'Listed companies/securities' table into {ticker: price}."""
    # "html.parser" doesn't auto-close unclosed <tr>/<td> tags per HTML5 rules,
    # so this minified page's rows end up nested instead of sibling <tr>s — lxml
    # implements proper tree construction and parses it correctly.
    soup = BeautifulSoup(html, "lxml")
    prices: dict[str, float] = {}

    for row in soup.select("div.t table tr"):
        cells = row.find_all("td")
        if len(cells) < 4:
            continue  # header row
        ticker_link = cells[0].find("a")
        if not ticker_link:
            continue
        ticker = ticker_link.get_text(strip=True)
        price_text = cells[3].get_text(strip=True).replace(",", "")
        try:
            prices[ticker] = float(price_text)
        except ValueError:
            continue

    return prices


def fetch_live_prices() -> dict[str, float]:
    """Returns {ticker: price}. Raises on network/parse failure."""
    html = _fetch_html()
    prices = _parse_ticker_prices(html)
    if not prices:
        raise ValueError("Parsed 0 rows from NSE source page — page layout may have changed")
    return prices


def update_companies_json() -> FetchResult:
    """Fetch live prices and merge into companies.json.

    On failure, companies.json is left completely untouched (fallback to
    last-known-good data) — only the lastFetch status block is written so the
    UI can still show that a refresh was attempted and failed.
    """
    raw = json.loads(DATA_PATH.read_text())
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    try:
        ticker_prices = fetch_live_prices()
    except Exception as exc:  # noqa: BLE001 - any failure should fall back gracefully
        raw["lastFetch"] = {"attemptedAt": now, "status": "failed", "error": str(exc)}
        DATA_PATH.write_text(json.dumps(raw, indent=2))
        return FetchResult(ok=False, message=f"Fetch failed: {exc}")

    today = now[:10]
    updated_count = 0
    skipped: list[str] = []

    for c in raw["companies"]:
        ticker = TICKER_BY_COMPANY.get(c["company"])
        if ticker is None or ticker not in ticker_prices:
            skipped.append(c["company"])
            continue

        new_price = ticker_prices[ticker]
        old_price = float(c["marketPrice"])
        if new_price != old_price:
            # Keep "previous" as the last price from an earlier day, so frequent intraday refreshes
            # (e.g. the background alert worker) show today's change rather than the last few minutes'.
            if c.get("lastUpdated") != today:
                c["previousPrice"] = old_price
            c["marketPrice"] = new_price
            c["lastUpdated"] = today
            updated_count += 1

    raw["lastPriceUpdate"] = today
    raw["lastFetch"] = {
        "attemptedAt": now,
        "status": "ok",
        "source": SOURCE_URL,
        "updatedCount": updated_count,
        "skipped": skipped,
    }
    DATA_PATH.write_text(json.dumps(raw, indent=2))

    return FetchResult(
        ok=True,
        message=f"Updated {updated_count} price(s); {len(skipped)} company(ies) not found on source.",
        updated_count=updated_count,
        skipped=skipped,
    )


if __name__ == "__main__":
    result = update_companies_json()
    print(("[OK] " if result.ok else "[FAILED] ") + result.message)
    if result.skipped:
        print("Skipped (no ticker mapping / not on source):")
        for name in result.skipped:
            print(f"  - {name}")
