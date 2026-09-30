"""Kenya Investment Explorer.

A single Streamlit app covering:
- Companies across all major sectors of the Kenyan economy, their share returns,
  and current market price per share (editable — prices can be updated any time
  they change, and the update is persisted to data/companies.json).
- An "Asset Classes Explained" education section.
- A Groq-powered AI chat assistant that answers questions using the live dataset.

All figures are illustrative/sample data for a demo app, not live market data.
"""

import html
import json
import os
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

DATA_PATH = Path(__file__).parent / "data" / "companies.json"

PRICE_COL = "Market Price (KES/share)"
PREV_PRICE_COL = "Previous Price (KES/share)"
UPDATED_COL = "Price Last Updated"


def load_raw() -> dict:
    return json.loads(DATA_PATH.read_text())


@st.cache_data
def load_data(_version: int = 0) -> pd.DataFrame:
    """`_version` is bumped after a save so st.cache_data invalidates."""
    raw = load_raw()
    rows = []
    for c in raw["companies"]:
        row = {
            "Company": c["company"],
            "Sector": c["sector"],
            PRICE_COL: c["marketPrice"],
            PREV_PRICE_COL: c.get("previousPrice"),
            UPDATED_COL: c.get("lastUpdated", raw.get("lastPriceUpdate", "")),
            "Market Cap (KES Bn)": c["marketCap"],
            **c["returns"],
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df["Avg Return %"] = df[raw["returnYears"]].mean(axis=1).round(2)
    return df


def save_price_updates(edited_df: pd.DataFrame) -> bool:
    """Persist edited market prices back to data/companies.json."""
    raw = load_raw()
    today = date.today().isoformat()
    edited_by_name = edited_df.set_index("Company")

    changed = False
    for c in raw["companies"]:
        new_price = float(edited_by_name.loc[c["company"], PRICE_COL])
        old_price = float(c["marketPrice"])
        if new_price != old_price:
            c["previousPrice"] = old_price
            c["marketPrice"] = new_price
            c["lastUpdated"] = today
            changed = True

    if changed:
        raw["lastPriceUpdate"] = today
        DATA_PATH.write_text(json.dumps(raw, indent=2))
    return changed


def build_ai_context(df: pd.DataFrame) -> str:
    cols = ["Company", "Sector", PRICE_COL, "Market Cap (KES Bn)", "Avg Return %"]
    return df[cols].to_csv(index=False)


def ask_groq(question: str, context_csv: str, history: list[dict]) -> str:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    system_prompt = (
        "You are an investment research assistant focused on the Kenyan market (NSE) "
        "and covers all sectors of the Kenyan economy. Answer questions using ONLY the "
        "sample dataset (CSV) provided below as context, plus general knowledge about how "
        "asset classes (equities, bonds, money market funds, REITs, unit trusts, etc.) work "
        "when the user asks conceptual questions. "
        "The company data is illustrative/sample data for a demo app, not live market data — "
        "if the user seems to want financial advice, remind them this is not financial advice "
        "and figures are illustrative.\n\nDataset (CSV):\n" + context_csv
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=800,
    )
    return completion.choices[0].message.content


ASSET_CLASSES = [
    {
        "name": "Equities (Shares/Stocks)",
        "icon": "📈",
        "how": (
            "You buy a small ownership stake (a share) in a listed company, such as the "
            "companies on the Nairobi Securities Exchange (NSE) in the **Companies & Sectors** "
            "tab. Returns come from two places: **capital gains** (the market price per share "
            "rising) and **dividends** (a portion of profits paid out to shareholders). Value "
            "moves with company performance and market sentiment, so returns can be volatile "
            "year to year — as seen in the FY return columns in this app."
        ),
    },
    {
        "name": "Government Bonds & Treasury Bills",
        "icon": "🏛️",
        "how": (
            "You lend money to the Kenyan government via the Central Bank of Kenya (CBK). "
            "Treasury Bills (T-Bills) are short-term (91/182/364 days) and sold at a discount; "
            "Treasury Bonds run longer (2–30 years) and pay a fixed coupon (interest) every six "
            "months. Considered one of the lowest-risk assets in Kenya since they're backed by "
            "the government, returns are fixed and predictable."
        ),
    },
    {
        "name": "Corporate Bonds",
        "icon": "🏢",
        "how": (
            "Similar to government bonds, but you lend to a company instead of the state. "
            "Corporate bonds typically pay a higher coupon than government bonds to compensate "
            "for the extra risk that the company could default on payments."
        ),
    },
    {
        "name": "Money Market Funds (MMFs)",
        "icon": "💰",
        "how": (
            "A pooled fund (e.g. from a licensed fund manager) that invests in short-term, "
            "low-risk instruments like T-Bills, commercial paper, and bank deposits. You buy "
            "units and earn a variable daily interest rate. Highly liquid — withdrawals are "
            "usually available within a day or two — making it popular for emergency savings."
        ),
    },
    {
        "name": "Unit Trusts / Collective Investment Schemes (CIS)",
        "icon": "🧺",
        "how": (
            "A professionally managed fund that pools money from many investors to buy a "
            "diversified basket of assets (equities, bonds, or a balanced mix). You buy units "
            "whose price (NAV) moves with the value of the underlying portfolio. Lets small "
            "investors get diversification and professional management without picking "
            "individual stocks themselves."
        ),
    },
    {
        "name": "Real Estate Investment Trusts (REITs)",
        "icon": "🏗️",
        "how": (
            "A REIT owns and manages income-generating property (offices, malls, residential "
            "developments) and is listed on the exchange, like ILAM Fahari I-REIT in this app. "
            "You buy shares in the REIT rather than a physical building, and earn a share of "
            "rental income and any appreciation in property value — real estate exposure "
            "without needing the capital to buy property outright."
        ),
    },
    {
        "name": "Direct Real Estate",
        "icon": "🏠",
        "how": (
            "Buying physical land or property directly. Returns come from rental income and "
            "capital appreciation when the property is sold. Requires significant upfront "
            "capital, is illiquid (can take months to sell), but is a well-established store "
            "of value in Kenya."
        ),
    },
    {
        "name": "Fixed/Term Deposits",
        "icon": "🏦",
        "how": (
            "You deposit a lump sum with a bank for a fixed period (e.g. 3–12 months) at an "
            "agreed interest rate, and withdraw the principal plus interest at maturity. Low "
            "risk and simple, but usually lower returns than market-based instruments, with a "
            "penalty for early withdrawal."
        ),
    },
    {
        "name": "SACCOs (Savings & Credit Co-operatives)",
        "icon": "🤝",
        "how": (
            "A member-owned co-operative where members save together and can borrow against "
            "their savings. Returns come as annual dividends on shares and interest on "
            "deposits, set by the SACCO's board based on yearly performance."
        ),
    },
    {
        "name": "Pension Funds (e.g. NSSF, occupational schemes)",
        "icon": "👴",
        "how": (
            "Long-term retirement savings, often with employer and employee contributions, "
            "invested by professional fund managers across equities, bonds, and property. "
            "Grows tax-efficiently over an entire career and is paid out (or drawn down) at "
            "retirement."
        ),
    },
    {
        "name": "Commodities",
        "icon": "🌾",
        "how": (
            "Investing in physical goods such as gold, or agricultural produce, either "
            "directly or via a fund/derivative tracking their price. Prices are driven by "
            "global supply and demand, and commodities often move independently of stocks and "
            "bonds — useful for diversification."
        ),
    },
    {
        "name": "Derivatives (Futures/Options)",
        "icon": "📐",
        "how": (
            "Contracts whose value is derived from an underlying asset (a stock, index, or "
            "commodity), used to hedge risk or speculate on price moves. The NSE's derivatives "
            "market (NEXT) offers equity futures. Can amplify both gains and losses, so it's "
            "considered higher-risk and typically used by more experienced investors."
        ),
    },
]


def hide_default_chrome() -> None:
    st.markdown(
        "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>",
        unsafe_allow_html=True,
    )


HOME_CSS = """
<style>
.hero {background:#0f766e;color:#fff;border-radius:14px;padding:22px 26px;margin-bottom:14px;
  display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:12px}
.hero h1 {color:#fff !important;font-size:30px;margin:0;padding:0}
.hero p {margin:4px 0 0;color:#d1fae5;font-size:15px}
.hero .chips {margin-top:10px;display:flex;gap:6px;flex-wrap:wrap}
.hero .chip {background:rgba(255,255,255,.16);border-radius:999px;padding:3px 11px;font-size:13px}
.hero .stamp {font-size:13px;color:#d1fae5;text-align:right}
.kpis {display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:6px 0 18px}
.kpi {border:1px solid rgba(128,128,128,.25);border-radius:12px;padding:14px 16px;background:rgba(128,128,128,.06)}
.kpi .l {font-size:13px;opacity:.75}
.kpi .v {font-size:26px;font-weight:600;margin-top:2px}
.kpi .s {font-size:12px;opacity:.7;margin-top:2px}
.panel {border:1px solid rgba(128,128,128,.25);border-radius:12px;padding:14px 16px;margin-bottom:12px}
.panel h4 {margin:0 0 8px;font-size:16px;padding:0}
.mv-row {display:flex;justify-content:space-between;gap:10px;padding:7px 0;
  border-bottom:1px solid rgba(128,128,128,.18);font-size:14px}
.mv-row:last-child {border-bottom:none}
.mv-row .sub {opacity:.65;font-size:12px}
.up {color:#16a34a;font-weight:600}
.dn {color:#dc2626;font-weight:600}
.home-note {font-size:12px;opacity:.65;margin-top:18px}
</style>
"""


def _greeting() -> str:
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


def render_hero(raw: dict) -> None:
    from data.profile import load_profile

    profile = load_profile()
    name = html.escape(profile["fullName"])
    if name:
        headline = f"{_greeting()}, {name}"
        chips = [profile["riskTolerance"] + " risk", profile["horizon"], *profile["preferredSectors"][:3]]
        chips_html = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)
    else:
        headline = "Explore investment opportunities across the Kenyan economy"
        chips_html = '<span class="chip">Set up your profile in the 👤 Profile tab to personalise this page</span>'

    st.markdown(HOME_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="hero">
  <div>
    <h1>📈 Kenya Investment Explorer</h1>
    <p>{headline}</p>
    <div class="chips">{chips_html}</div>
  </div>
  <div class="stamp">NSE prices last updated<br><b>{html.escape(str(raw.get("lastPriceUpdate", "—")))}</b></div>
</div>
""",
        unsafe_allow_html=True,
    )


def _kpi_cards(cards: list[tuple[str, str, str]]) -> None:
    body = "".join(
        f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'
        for label, value, sub in cards
    )
    st.markdown(f'<div class="kpis">{body}</div>', unsafe_allow_html=True)


def _panel(title: str, rows: list[tuple[str, str, str, str]], empty: str = "Nothing to show yet.") -> None:
    """rows: (main label, sub label, value text, css class for value)."""
    body = "".join(
        f'<div class="mv-row"><div>{html.escape(main)}<div class="sub">{html.escape(sub)}</div></div>'
        f'<div class="{cls}">{val}</div></div>'
        for main, sub, val, cls in rows
    ) or f'<div class="mv-row"><div class="sub">{empty}</div></div>'
    st.markdown(f'<div class="panel"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)


def _signed(value: float) -> tuple[str, str]:
    return f"{value:+.2f}%", "up" if value >= 0 else "dn"


def render_overview(df: pd.DataFrame, raw: dict) -> None:
    from data.portfolio import holdings_market_value, load_portfolio
    from data.profile import load_profile
    from data.watchlist import check_alerts, load_watchlist

    movers = df.dropna(subset=[PREV_PRICE_COL]).copy()
    movers = movers[movers[PREV_PRICE_COL] > 0]
    movers["Change %"] = (movers[PRICE_COL] / movers[PREV_PRICE_COL] - 1) * 100
    gainers = int((movers["Change %"] > 0).sum())
    losers = int((movers["Change %"] < 0).sum())

    st.subheader("Market snapshot")
    _kpi_cards(
        [
            ("🏢 Companies tracked", f"{len(df)}", "Listed on the NSE"),
            ("🧩 Sectors covered", f"{df['Sector'].nunique()}", "Across the economy"),
            ("💰 Total market cap", f"{df['Market Cap (KES Bn)'].sum():,.1f} Bn", "KES"),
            (
                "📊 Gainers / losers",
                f'<span class="up">{gainers}</span> / <span class="dn">{losers}</span>',
                "Since previous price",
            ),
        ]
    )

    price_by_company = dict(zip(df["Company"], df[PRICE_COL]))
    portfolio = load_portfolio()
    total_value = portfolio["cash"] + holdings_market_value(portfolio, price_by_company)
    baseline = portfolio["startingCash"] + portfolio.get("netDeposits", 0.0)
    pnl_pct = (total_value - baseline) / baseline * 100 if baseline else 0.0
    wl = load_watchlist()
    triggered = check_alerts(wl, price_by_company)
    pnl_text, pnl_cls = _signed(pnl_pct)

    st.subheader("Your account")
    _kpi_cards(
        [
            ("💼 Paper portfolio", f"{total_value:,.0f}", "KES, cash + holdings"),
            ("📈 Total P&L", f'<span class="{pnl_cls}">{pnl_text}</span>', "Against money put in"),
            ("🔔 Alerts triggered", f"{len(triggered)}", "See the Watchlist tab"),
            ("⭐ Watching", f"{len(wl['watching'])}", "Companies on your watchlist"),
        ]
    )

    st.subheader("Market movers")
    g_col, l_col = st.columns(2)

    def mover_rows(frame: pd.DataFrame) -> list[tuple[str, str, str, str]]:
        return [
            (r["Company"], f"KES {r[PRICE_COL]:,.2f} · {r['Sector']}", *_signed(r["Change %"]))
            for _, r in frame.iterrows()
        ]

    with g_col:
        top_up = movers[movers["Change %"] > 0].nlargest(5, "Change %")
        _panel("🟢 Top gainers", mover_rows(top_up), "No gainers since the last update.")
    with l_col:
        top_down = movers[movers["Change %"] < 0].nsmallest(5, "Change %")
        _panel("🔴 Top losers", mover_rows(top_down), "No losers since the last update.")

    chart_col, perf_col = st.columns([3, 2])
    with chart_col:
        st.subheader("Average return by sector")
        import altair as alt

        sector_returns = df.groupby("Sector", as_index=False)["Avg Return %"].mean().round(2)
        chart = (
            alt.Chart(sector_returns)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("Avg Return %:Q", title="Average return (%)"),
                y=alt.Y("Sector:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
                color=alt.condition(alt.datum["Avg Return %"] >= 0, alt.value("#0f766e"), alt.value("#dc2626")),
                tooltip=["Sector", "Avg Return %"],
            )
            .properties(height=380)
        )
        st.altair_chart(chart, use_container_width=True)
    with perf_col:
        st.subheader("Long-run performers")

        def perf_rows(frame: pd.DataFrame) -> list[tuple[str, str, str, str]]:
            return [(r["Company"], r["Sector"], *_signed(r["Avg Return %"])) for _, r in frame.iterrows()]

        _panel("🏆 Best average return", perf_rows(df.nlargest(3, "Avg Return %")))
        _panel("⚠️ Weakest average return", perf_rows(df.nsmallest(3, "Avg Return %")))

    profile = load_profile()
    st.subheader("Picks in your sectors")
    if profile["preferredSectors"]:
        picks = df[df["Sector"].isin(profile["preferredSectors"])].nlargest(5, "Avg Return %")
        rows = [
            (r["Company"], f"{r['Sector']} · KES {r[PRICE_COL]:,.2f}", *_signed(r["Avg Return %"]))
            for _, r in picks.iterrows()
        ]
        _panel(f"Top companies for your {profile['riskTolerance'].lower()} profile, by average return", rows)
    else:
        st.info("Choose your preferred sectors in the 👤 Profile tab to see personalised picks here.", icon="👤")

    st.markdown(
        '<div class="home-note">⚠️ Market prices can be refreshed from a free public NSE data source '
        "(Update Market Prices tab); historical returns are illustrative demo data. "
        "Nothing on this page is financial advice.</div>",
        unsafe_allow_html=True,
    )


def render_companies(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    sectors = sorted(df["Sector"].unique())
    selected_sectors = st.sidebar.multiselect("Sector", sectors, default=sectors)

    max_price = float(df[PRICE_COL].max())
    price_cap = st.sidebar.slider("Max market price (KES/share)", 0.0, max_price, max_price, step=1.0)

    min_avg_return = st.sidebar.slider("Min average return (%)", -30.0, 30.0, -30.0, step=0.5)

    filtered = df[
        df["Sector"].isin(selected_sectors)
        & (df[PRICE_COL] <= price_cap)
        & (df["Avg Return %"] >= min_avg_return)
    ]

    st.subheader(f"Investment opportunities ({len(filtered)} companies)")
    display_cols = ["Sector", PRICE_COL, "Market Cap (KES Bn)", "Avg Return %", UPDATED_COL]
    st.dataframe(filtered.set_index("Company")[display_cols], use_container_width=True)

    if not filtered.empty:
        st.subheader("Average return by company")
        st.bar_chart(filtered.set_index("Company")["Avg Return %"])

    return filtered


def render_live_refresh(raw: dict) -> None:
    st.subheader("🔄 Live NSE prices")
    last_fetch = raw.get("lastFetch")

    if last_fetch is None:
        st.caption("No live refresh has been run yet. Prices below are the last manually-set values.")
    elif last_fetch["status"] == "ok":
        st.success(
            f"Last live refresh: **{last_fetch['attemptedAt']}** — "
            f"updated {last_fetch['updatedCount']} price(s) from {last_fetch['source']}.",
            icon="✅",
        )
        if last_fetch.get("skipped"):
            st.caption("Not on live source (kept last known price): " + ", ".join(last_fetch["skipped"]))
    else:
        st.warning(
            f"Last live refresh attempt **failed** at {last_fetch['attemptedAt']} "
            f"({last_fetch['error']}) — showing last known prices.",
            icon="⚠️",
        )

    if st.button("🔄 Refresh live prices now", type="primary"):
        from data.fetch_prices import update_companies_json

        with st.spinner("Fetching live NSE prices..."):
            result = update_companies_json()
        st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1
        if result.ok:
            st.success(result.message)
        else:
            st.error(result.message)
        st.rerun()


def render_trading(df: pd.DataFrame) -> None:
    from data.portfolio import (
        buy,
        deposit,
        holdings_market_value,
        load_portfolio,
        reset_portfolio,
        sell,
        withdraw,
    )

    portfolio = load_portfolio()
    price_by_company = dict(zip(df["Company"], df[PRICE_COL]))

    st.subheader("💼 Paper Trading Portfolio")
    st.caption(
        "Simulated trading only — no real money or real brokerage orders are involved. Buys "
        "and sells are priced off the current **Market Price** column and tracked locally in "
        "`data/portfolio.json`."
    )

    holdings_value = holdings_market_value(portfolio, price_by_company)
    total_value = portfolio["cash"] + holdings_value
    baseline = portfolio["startingCash"] + portfolio.get("netDeposits", 0.0)
    pnl = total_value - baseline
    pnl_pct = (pnl / baseline * 100) if baseline else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cash (KES)", f"{portfolio['cash']:,.2f}")
    c2.metric("Holdings value (KES)", f"{holdings_value:,.2f}")
    c3.metric("Total portfolio (KES)", f"{total_value:,.2f}")
    c4.metric("Total P&L (KES)", f"{pnl:,.2f}", f"{pnl_pct:+.2f}%")

    buy_col, sell_col = st.columns(2)

    with buy_col:
        st.markdown("**Buy shares**")
        company = st.selectbox("Company", df["Company"], key="buy_company")
        price = float(price_by_company[company])
        st.caption(f"Current price: KES {price:,.2f}/share")
        shares = st.number_input("Shares to buy", min_value=1, step=1, key="buy_shares")
        st.caption(f"Estimated cost: KES {shares * price:,.2f}")
        if st.button("🟢 Buy", key="buy_btn"):
            result = buy(portfolio, company, float(shares), price)
            (st.success if result.ok else st.error)(result.message)
            if result.ok:
                st.rerun()

    with sell_col:
        st.markdown("**Sell shares**")
        held = {h["company"]: h["shares"] for h in portfolio["holdings"]}
        if not held:
            st.info("You have no holdings to sell yet.")
        else:
            company_s = st.selectbox("Company", list(held.keys()), key="sell_company")
            held_shares = held[company_s]
            price_s = float(price_by_company.get(company_s, 0.0))
            st.caption(f"You hold {held_shares:g} share(s) — current price: KES {price_s:,.2f}/share")
            shares_s = st.number_input(
                "Shares to sell", min_value=1, max_value=int(held_shares), step=1, key="sell_shares"
            )
            st.caption(f"Estimated proceeds: KES {shares_s * price_s:,.2f}")
            if st.button("🔴 Sell", key="sell_btn"):
                result = sell(portfolio, company_s, float(shares_s), price_s)
                (st.success if result.ok else st.error)(result.message)
                if result.ok:
                    st.rerun()

    st.divider()
    st.markdown("**Cash management (deposit / withdraw)**")
    st.caption("Simulates moving cash in or out of your virtual brokerage account.")
    dep_col, wd_col = st.columns(2)
    with dep_col:
        dep_amount = st.number_input("Deposit amount (KES)", min_value=0.0, step=1000.0, key="deposit_amount")
        if st.button("⬆️ Deposit", key="deposit_btn"):
            result = deposit(portfolio, float(dep_amount))
            (st.success if result.ok else st.error)(result.message)
            if result.ok:
                st.rerun()
    with wd_col:
        wd_amount = st.number_input("Withdraw amount (KES)", min_value=0.0, step=1000.0, key="withdraw_amount")
        if st.button("⬇️ Withdraw", key="withdraw_btn"):
            result = withdraw(portfolio, float(wd_amount))
            (st.success if result.ok else st.error)(result.message)
            if result.ok:
                st.rerun()

    st.divider()
    st.markdown("**Current holdings**")
    holdings_rows = []
    for h in portfolio["holdings"]:
        cur_price = float(price_by_company.get(h["company"], h["avgCost"]))
        market_value = h["shares"] * cur_price
        cost_basis = h["shares"] * h["avgCost"]
        unrealized = market_value - cost_basis
        unrealized_pct = (unrealized / cost_basis * 100) if cost_basis else 0.0
        holdings_rows.append(
            {
                "Company": h["company"],
                "Shares": h["shares"],
                "Avg Cost (KES)": round(h["avgCost"], 2),
                "Current Price (KES)": round(cur_price, 2),
                "Market Value (KES)": round(market_value, 2),
                "Unrealized P&L (KES)": round(unrealized, 2),
                "Unrealized P&L %": round(unrealized_pct, 2),
            }
        )
    if holdings_rows:
        st.dataframe(pd.DataFrame(holdings_rows).set_index("Company"), use_container_width=True)
    else:
        st.caption("No holdings yet — place a buy order above to get started.")

    st.markdown("**Trade history**")
    if portfolio["trades"]:
        st.dataframe(pd.DataFrame(list(reversed(portfolio["trades"]))), use_container_width=True, hide_index=True)
    else:
        st.caption("No trades yet.")

    st.divider()
    st.markdown("**📄 Portfolio statement**")
    st.caption("Download your current holdings and full trade/cash history as CSV.")
    s1, s2 = st.columns(2)
    with s1:
        holdings_csv = pd.DataFrame(holdings_rows).to_csv(index=False) if holdings_rows else ""
        st.download_button(
            "⬇️ Download holdings (CSV)",
            data=holdings_csv,
            file_name="holdings_statement.csv",
            mime="text/csv",
            disabled=not holdings_rows,
        )
    with s2:
        trades_csv = (
            pd.DataFrame(list(reversed(portfolio["trades"]))).to_csv(index=False) if portfolio["trades"] else ""
        )
        st.download_button(
            "⬇️ Download trade history (CSV)",
            data=trades_csv,
            file_name="trade_history_statement.csv",
            mime="text/csv",
            disabled=not portfolio["trades"],
        )

    with st.expander("⚠️ Reset portfolio"):
        st.caption("Wipes cash, holdings, and trade history back to the starting balance. Cannot be undone.")
        if st.button("Reset portfolio to starting cash", key="reset_portfolio_btn"):
            reset_portfolio()
            st.success("Portfolio reset.")
            st.rerun()


def render_watchlist(df: pd.DataFrame) -> None:
    from data.watchlist import (
        add_company,
        check_alerts,
        load_watchlist,
        remove_alert,
        remove_company,
        set_alert,
    )

    wl = load_watchlist()
    price_by_company = dict(zip(df["Company"], df[PRICE_COL]))

    st.subheader("⭐ Watchlist & price alerts")
    st.caption(
        "Track companies you're interested in and set a target price alert — above or below "
        "a threshold. Checked live against the current Market Price."
    )

    for t in check_alerts(wl, price_by_company):
        arrow = "risen above" if t["direction"] == "above" else "fallen below"
        st.warning(
            f"🔔 **{t['company']}** has {arrow} your target of KES {t['target']:,.2f} — "
            f"current price KES {t['current']:,.2f}.",
            icon="🔔",
        )

    available = [c for c in df["Company"] if c not in wl["watching"]]
    if available:
        add_col1, add_col2 = st.columns([3, 1])
        new_company = add_col1.selectbox("Add company to watchlist", available, key="watch_add")
        add_col2.write("")
        add_col2.write("")
        if add_col2.button("➕ Add"):
            add_company(wl, new_company)
            st.rerun()
    else:
        st.caption("All companies are already on your watchlist.")

    st.divider()

    if not wl["watching"]:
        st.info("Your watchlist is empty — add a company above to get started.")
        return

    alerts_by_company = {a["company"]: a for a in wl["alerts"]}
    for company in wl["watching"]:
        price = price_by_company.get(company)
        label = f"{company} — KES {price:,.2f}" if price is not None else company
        with st.expander(label):
            existing = alerts_by_company.get(company)
            c1, c2, c3 = st.columns([2, 2, 1])
            direction = c1.selectbox(
                "Alert when price is",
                ["above", "below"],
                index=0 if not existing or existing["direction"] == "above" else 1,
                key=f"dir_{company}",
            )
            target = c2.number_input(
                "Target price (KES)",
                min_value=0.0,
                value=float(existing["target"]) if existing else float(price or 0.0),
                step=0.5,
                key=f"target_{company}",
            )
            with c3:
                st.write("")
                st.write("")
                if st.button("💾 Save alert", key=f"save_{company}"):
                    set_alert(wl, company, target, direction)
                    st.rerun()
            b1, b2 = st.columns(2)
            if existing and b1.button("🗑️ Remove alert", key=f"rm_alert_{company}"):
                remove_alert(wl, company)
                st.rerun()
            if b2.button("❌ Remove from watchlist", key=f"rm_watch_{company}"):
                remove_company(wl, company)
                st.rerun()


def load_news() -> list[dict]:
    path = Path(__file__).parent / "data" / "news.json"
    return json.loads(path.read_text())["items"]


def render_news() -> None:
    st.subheader("📰 Research & market news")
    st.caption(
        "Curated sample research notes and market commentary for this demo — not a live news "
        "feed, and not financial advice."
    )
    items = load_news()
    sectors = sorted({item["sector"] for item in items})
    selected = st.multiselect("Filter by sector", sectors, default=sectors, key="news_sector_filter")
    filtered = sorted(
        (item for item in items if item["sector"] in selected),
        key=lambda x: x["date"],
        reverse=True,
    )
    if not filtered:
        st.info("No news items match the selected sectors.")
    for item in filtered:
        with st.container(border=True):
            st.markdown(f"**{item['title']}**")
            st.caption(f"{item['date']} · {item['sector']} · {item['source']}")
            st.write(item["summary"])


def load_bonds_funds() -> dict:
    path = Path(__file__).parent / "data" / "bonds_funds.json"
    return json.loads(path.read_text())


def render_bonds_funds() -> None:
    data = load_bonds_funds()
    st.subheader("🏛️ Bonds, T-Bills & Funds")
    st.caption(
        "Illustrative sample rates for fixed-income and pooled-fund instruments available in "
        "Kenya — not live rates, not financial advice."
    )

    st.markdown("**Government securities (Treasury Bills & Bonds)**")
    st.dataframe(pd.DataFrame(data["governmentSecurities"]), use_container_width=True, hide_index=True)

    st.markdown("**Corporate bonds**")
    st.dataframe(pd.DataFrame(data["corporateBonds"]), use_container_width=True, hide_index=True)

    st.markdown("**Money market funds**")
    st.dataframe(pd.DataFrame(data["moneyMarketFunds"]), use_container_width=True, hide_index=True)

    st.markdown("**Unit trusts / collective investment schemes**")
    st.dataframe(pd.DataFrame(data["unitTrusts"]), use_container_width=True, hide_index=True)


def load_ipo_calendar() -> list[dict]:
    path = Path(__file__).parent / "data" / "ipo_calendar.json"
    return json.loads(path.read_text())["events"]


def render_ipo_calendar() -> None:
    st.subheader("🗓️ IPO & Rights Issue calendar")
    st.caption("Illustrative sample calendar of primary-market events — not a live feed, not financial advice.")
    events = load_ipo_calendar()
    status_order = {"Ongoing": 0, "Upcoming": 1, "Closed": 2}
    events = sorted(events, key=lambda e: (status_order.get(e["Status"], 9), e["openDate"]))
    st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)


def render_price_updates(df: pd.DataFrame) -> None:
    st.subheader("💹 Update current market price per share")
    st.caption(
        "Edit the **Market Price (KES/share)** column below to reflect a new price, then click "
        "**Save price updates**. Changes are written to `data/companies.json` and the previous "
        "price + update date are recorded automatically."
    )

    editable_cols = ["Company", "Sector", PRICE_COL, PREV_PRICE_COL, UPDATED_COL]
    edited = st.data_editor(
        df[editable_cols],
        use_container_width=True,
        hide_index=True,
        disabled=["Company", "Sector", PREV_PRICE_COL, UPDATED_COL],
        column_config={
            PRICE_COL: st.column_config.NumberColumn(PRICE_COL, min_value=0.0, step=0.05, format="%.2f"),
        },
        key="price_editor",
    )

    if st.button("💾 Save price updates", type="primary"):
        changed = save_price_updates(edited)
        if changed:
            st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1
            st.success("Prices updated and saved.")
            st.rerun()
        else:
            st.info("No price changes detected.")


def render_profile(df: pd.DataFrame) -> None:
    from data.profile import (
        EXPERIENCE_LEVELS,
        GOALS,
        HORIZONS,
        RISK_LEVELS,
        load_profile,
        reset_profile,
        update_profile,
    )

    profile = load_profile()

    st.subheader("👤 My investor profile")
    st.caption(
        "Keep your details and investment preferences up to date — you can edit and save this "
        "any time. Stored locally in `data/profile.json`."
    )

    if profile["lastUpdated"]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Name", profile["fullName"] or "—")
        c2.metric("Risk tolerance", profile["riskTolerance"])
        c3.metric("Monthly budget (KES)", f"{profile['monthlyBudget']:,.0f}")
        c4.metric("Last updated", profile["lastUpdated"].replace("T", " "))
    else:
        st.info("You haven't set up a profile yet — fill in the form below and save.")

    sectors = sorted(df["Sector"].unique())
    with st.form("profile_form"):
        st.markdown("**Personal details**")
        p1, p2 = st.columns(2)
        full_name = p1.text_input("Full name *", value=profile["fullName"])
        email = p2.text_input("Email", value=profile["email"])
        phone = p1.text_input("Phone", value=profile["phone"], placeholder="+254 7xx xxx xxx")
        location = p2.text_input("County / town", value=profile["location"])

        st.markdown("**Investment preferences**")
        q1, q2, q3 = st.columns(3)
        experience = q1.selectbox(
            "Experience level", EXPERIENCE_LEVELS, index=EXPERIENCE_LEVELS.index(profile["experience"])
        )
        risk = q2.selectbox("Risk tolerance", RISK_LEVELS, index=RISK_LEVELS.index(profile["riskTolerance"]))
        horizon = q3.selectbox("Investment horizon", HORIZONS, index=HORIZONS.index(profile["horizon"]))
        goals = st.multiselect("Investment goals", GOALS, default=[g for g in profile["goals"] if g in GOALS])
        preferred = st.multiselect(
            "Preferred sectors",
            sectors,
            default=[s for s in profile["preferredSectors"] if s in sectors],
        )
        budget = st.number_input(
            "Monthly investment budget (KES)",
            min_value=0.0,
            value=float(profile["monthlyBudget"]),
            step=1000.0,
        )
        bio = st.text_area("About me / notes", value=profile["bio"], max_chars=500)

        if st.form_submit_button("💾 Save profile", type="primary"):
            result = update_profile(
                profile,
                {
                    "fullName": full_name,
                    "email": email,
                    "phone": phone,
                    "location": location,
                    "experience": experience,
                    "riskTolerance": risk,
                    "horizon": horizon,
                    "goals": goals,
                    "preferredSectors": preferred,
                    "monthlyBudget": float(budget),
                    "bio": bio,
                },
            )
            (st.success if result.ok else st.error)(result.message)
            if result.ok:
                st.rerun()

    if profile["preferredSectors"]:
        st.markdown("**Companies in your preferred sectors**")
        matches = df[df["Sector"].isin(profile["preferredSectors"])]
        st.dataframe(
            matches.set_index("Company")[["Sector", PRICE_COL, "Avg Return %"]],
            use_container_width=True,
        )

    with st.expander("⚠️ Clear profile"):
        st.caption("Deletes all saved profile details. Cannot be undone.")
        if st.button("Clear my profile", key="reset_profile_btn"):
            reset_profile()
            st.success("Profile cleared.")
            st.rerun()


def render_asset_classes() -> None:
    st.subheader("🎓 Asset classes explained")
    st.caption("A quick primer on the main asset classes available to investors in Kenya, and how each one works.")
    for a in ASSET_CLASSES:
        with st.expander(f"{a['icon']} {a['name']}"):
            st.markdown(a["how"])


def render_chat(df: pd.DataFrame) -> None:
    st.subheader("🤖 Ask Groq AI about these opportunities")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if not GROQ_API_KEY:
        st.info(
            "Set `GROQ_API_KEY` in a `.env` file (see `.env.example`) to enable the AI assistant. "
            "The rest of the app works without it.",
            icon="🔑",
        )
        return

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("e.g. Which sectors had the best average returns?")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                answer = ask_groq(question, build_ai_context(df), st.session_state.chat_history[:-1])
            except Exception as exc:  # noqa: BLE001 - surface any API/config error to the user
                answer = f"Sorry, the AI assistant hit an error: `{exc}`"
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})


def main() -> None:
    st.set_page_config(page_title="Kenya Investment Explorer", page_icon="📈", layout="wide")
    hide_default_chrome()

    version = st.session_state.get("data_version", 0)
    df = load_data(version)
    raw = load_raw()
    render_hero(raw)

    (
        tab_overview,
        tab_companies,
        tab_watchlist,
        tab_trade,
        tab_bonds,
        tab_ipo,
        tab_news,
        tab_prices,
        tab_assets,
        tab_chat,
        tab_profile,
    ) = st.tabs(
        [
            "🏠 Overview",
            "📊 Companies & Sectors",
            "⭐ Watchlist",
            "💼 Trade / Portfolio",
            "🏛️ Bonds & Funds",
            "🗓️ IPO Calendar",
            "📰 Research & News",
            "💹 Update Market Prices",
            "🎓 Asset Classes",
            "🤖 Ask AI",
            "👤 Profile",
        ]
    )

    with tab_overview:
        render_overview(df, raw)

    with tab_companies:
        filtered = render_companies(df)

    with tab_watchlist:
        render_watchlist(df)

    with tab_trade:
        render_trading(df)

    with tab_bonds:
        render_bonds_funds()

    with tab_ipo:
        render_ipo_calendar()

    with tab_news:
        render_news()

    with tab_prices:
        render_live_refresh(raw)
        st.divider()
        render_price_updates(df)

    with tab_assets:
        render_asset_classes()

    with tab_chat:
        render_chat(filtered if not filtered.empty else df)

    with tab_profile:
        render_profile(df)


if __name__ == "__main__":
    main()
