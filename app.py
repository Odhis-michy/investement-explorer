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


ACCENT = "#F0B90B"
UP = "#0ECB81"
DOWN = "#F6465D"

HOME_CSS = """
<style>
.topbar {display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;
  padding:12px 4px 14px;margin-bottom:6px;border-bottom:1px solid #2B3139}
.brand {color:#F0B90B;font-size:24px;font-weight:700;letter-spacing:.5px}
.brand small {color:#848E9C;font-size:13px;font-weight:500;margin-left:10px;letter-spacing:0}
.topbar .right {text-align:right;font-size:13px;color:#848E9C}
.topbar .right b {color:#EAECEF}
.chip {display:inline-block;background:#2B3139;color:#EAECEF;border-radius:6px;padding:2px 9px;
  font-size:12px;margin:4px 0 0 5px}
.mk-cards {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:12px 0 22px}
@media (max-width:1100px) {.mk-cards {grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:640px) {.mk-cards {grid-template-columns:minmax(0,1fr)}}
.mk-card {border:1px solid #2B3139;border-radius:14px;padding:14px 14px}
.mk-card .hd {display:flex;justify-content:space-between;font-size:13px;font-weight:600;margin-bottom:10px}
.mk-card .hd a {color:#848E9C !important;text-decoration:none;font-weight:500}
.mk-card .hd a:hover {color:#F0B90B !important}
.mk-row {display:grid;grid-template-columns:24px minmax(0,1fr) auto 64px;align-items:center;gap:6px;
  padding:8px 0;font-size:13px}
.mk-row .nm {font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mk-row .px {text-align:right}
.mk-row .ch {text-align:right}
.ico {width:24px;height:24px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;
  font-size:10px;font-weight:700;color:#0B0E11;flex-shrink:0}
.ico.lg {width:30px;height:30px;font-size:12px}
.up {color:#0ECB81 !important;font-weight:600}
.dn {color:#F6465D !important;font-weight:600}
.mk-title {font-size:18px;font-weight:700;margin:6px 0 2px}
.mk-sub {font-size:12px;color:#848E9C;margin-bottom:10px}
.mk-table {width:100%;border-collapse:collapse;font-size:14px}
.mk-table th {color:#848E9C;font-weight:500;font-size:12px;text-align:right;padding:10px 8px;
  border-bottom:1px solid #2B3139}
.mk-table th:first-child, .mk-table td:first-child {text-align:left}
.mk-table td {text-align:right;padding:14px 8px;border-bottom:1px solid #1E2329}
.mk-table tr:hover td {background:#1E2329}
.mk-name {display:flex;align-items:center;gap:10px}
.mk-name b {font-size:15px}
.mk-name span {color:#848E9C;font-size:12px}
.mk-table .muted {color:#848E9C;font-size:12px}
.star {font-size:18px;text-decoration:none !important;color:#848E9C !important}
.star.on {color:#F0B90B !important}
.home-note {font-size:12px;color:#848E9C;margin-top:18px}
.sb-card {border:1px solid #2B3139;border-radius:12px;padding:14px;margin-bottom:12px;background:#181A20}
.sb-top {display:flex;align-items:center;gap:12px}
.sb-avatar {width:46px;height:46px;border-radius:50%;background:#F0B90B;color:#0B0E11;font-weight:700;
  display:flex;align-items:center;justify-content:center;font-size:17px;flex-shrink:0}
.sb-name {font-weight:600;font-size:16px;line-height:1.2}
.sb-sub {font-size:12px;color:#848E9C;word-break:break-all}
.sb-chips {display:flex;flex-wrap:wrap;gap:5px;margin-top:10px}
.sb-chip {font-size:11px;border-radius:6px;padding:2px 8px;background:rgba(240,185,11,.14);color:#F0B90B}
.sb-stats {display:grid;grid-template-columns:1fr 1fr;gap:8px}
.sb-stat {border:1px solid #2B3139;border-radius:10px;padding:8px 10px}
.sb-stat .l {font-size:11px;color:#848E9C}
.sb-stat .v {font-size:16px;font-weight:600}
</style>
"""

SECTOR_COLORS = ["#F0B90B", "#F7931A", "#627EEA", "#0ECB81", "#E84142", "#8247E5",
                 "#26A17B", "#F3BA2F", "#00AAE4", "#FF6B6B", "#C0A2FF", "#4FD1C5"]


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
        greeting = f"{_greeting()}, <b>{name}</b>"
        chips = [profile["riskTolerance"] + " risk", profile["horizon"]]
        chips_html = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)
    else:
        greeting = "Welcome, <b>guest investor</b>"
        chips_html = '<span class="chip">Set up your profile in the sidebar</span>'

    st.markdown(HOME_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="topbar">
  <div class="brand">📈 KENYA INVEST<small>NSE Investment Explorer</small></div>
  <div class="right">{greeting} · prices updated <b>{html.escape(str(raw.get("lastPriceUpdate", "—")))}</b>
    <div>{chips_html}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sidebar(df: pd.DataFrame) -> None:
    from data.portfolio import holdings_market_value, load_portfolio, reset_portfolio
    from data.profile import HORIZONS, RISK_LEVELS, load_profile, reset_profile, update_profile
    from data.watchlist import check_alerts, load_watchlist

    profile = load_profile()
    price_by_company = dict(zip(df["Company"], df[PRICE_COL]))
    portfolio = load_portfolio()
    total_value = portfolio["cash"] + holdings_market_value(portfolio, price_by_company)
    baseline = portfolio["startingCash"] + portfolio.get("netDeposits", 0.0)
    pnl_pct = (total_value - baseline) / baseline * 100 if baseline else 0.0
    wl = load_watchlist()
    triggered = check_alerts(wl, price_by_company)

    sb = st.sidebar
    sb.markdown("### 🧭 My dashboard")

    # --- Profile card -----------------------------------------------------------
    name = profile["fullName"]
    if name:
        initials = "".join(part[0] for part in name.split()[:2]).upper()
        contact = " · ".join(html.escape(x) for x in (profile["email"], profile["location"]) if x)
        chips = [profile["experience"], profile["riskTolerance"] + " risk", profile["horizon"]]
        sb.markdown(
            f"""
<div class="sb-card">
  <div class="sb-top">
    <div class="sb-avatar">{html.escape(initials)}</div>
    <div><div class="sb-name">{html.escape(name)}</div><div class="sb-sub">{contact or "No contact details yet"}</div></div>
  </div>
  <div class="sb-chips">{"".join(f'<span class="sb-chip">{html.escape(c)}</span>' for c in chips)}</div>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        sb.markdown(
            '<div class="sb-card"><div class="sb-top"><div class="sb-avatar">?</div>'
            '<div><div class="sb-name">Guest investor</div>'
            '<div class="sb-sub">Set up your profile under Account settings below.</div></div></div></div>',
            unsafe_allow_html=True,
        )

    # --- Account summary ------------------------------------------------------------
    sb.markdown("**💼 Account**")
    pnl_cls = "up" if pnl_pct >= 0 else "dn"
    stats = [
        ("Portfolio (KES)", f"{total_value:,.0f}"),
        ("P&L", f'<span class="{pnl_cls}">{pnl_pct:+.2f}%</span>'),
        ("Cash (KES)", f"{portfolio['cash']:,.0f}"),
        ("Holdings", f"{len(portfolio['holdings'])}"),
        ("Watching", f"{len(wl['watching'])}"),
        ("Alerts hit", f"{len(triggered)}"),
    ]
    sb.markdown(
        '<div class="sb-card sb-stats">'
        + "".join(f'<div class="sb-stat"><div class="l">{l}</div><div class="v">{v}</div></div>' for l, v in stats)
        + "</div>",
        unsafe_allow_html=True,
    )
    for t in triggered:
        sb.warning(f"🔔 {t['company']} is {t['direction']} KES {t['target']:,.2f}", icon="🔔")

    # --- Account settings -------------------------------------------------------------
    # Keys include lastUpdated so fields refresh after the profile is saved elsewhere.
    ver = profile["lastUpdated"] or "new"
    with sb.expander("⚙️ Account settings", expanded=not name):
        with st.form("sidebar_settings"):
            new_name = st.text_input("Full name *", value=profile["fullName"], key=f"sb_name_{ver}")
            new_email = st.text_input("Email", value=profile["email"], key=f"sb_email_{ver}")
            new_phone = st.text_input("Phone", value=profile["phone"], key=f"sb_phone_{ver}")
            new_risk = st.selectbox(
                "Risk tolerance", RISK_LEVELS, index=RISK_LEVELS.index(profile["riskTolerance"]), key=f"sb_risk_{ver}"
            )
            new_horizon = st.selectbox(
                "Investment horizon", HORIZONS, index=HORIZONS.index(profile["horizon"]), key=f"sb_hor_{ver}"
            )
            if st.form_submit_button("💾 Save settings", type="primary", use_container_width=True):
                result = update_profile(
                    profile,
                    {
                        "fullName": new_name,
                        "email": new_email,
                        "phone": new_phone,
                        "riskTolerance": new_risk,
                        "horizon": new_horizon,
                    },
                )
                if result.ok:
                    st.rerun()
                st.error(result.message)
        st.caption("Goals, sectors, budget and notes are in the 👤 Profile tab.")

    with sb.expander("🧹 Reset data"):
        st.caption("These can't be undone.")
        if st.button("Reset paper portfolio", key="sb_reset_portfolio", use_container_width=True):
            reset_portfolio()
            st.rerun()
        if st.button("Clear profile", key="sb_clear_profile", use_container_width=True):
            reset_profile()
            st.rerun()


def _signed(value: float) -> tuple[str, str]:
    return f"{value:+.2f}%", "up" if value >= 0 else "dn"


def _icon(company: str, sector: str, sectors: list[str], size: str = "") -> str:
    initials = "".join(w[0] for w in company.replace("(", " ").split()[:2]).upper()
    color = SECTOR_COLORS[sectors.index(sector) % len(SECTOR_COLORS)] if sector in sectors else ACCENT
    return f'<span class="ico {size}" style="background:{color}">{html.escape(initials)}</span>'


SORTS = {
    "mcap": ("Market cap", "Market Cap (KES Bn)", False),
    "gain": ("Top gainers", "Change %", False),
    "loss": ("Top losers", "Change %", True),
    "perf": ("5-yr average return", "Avg Return %", False),
    "price": ("Price", PRICE_COL, False),
    "name": ("Name (A–Z)", "Company", True),
}


def render_overview(df: pd.DataFrame, raw: dict) -> None:
    from urllib.parse import quote

    from data.watchlist import add_company, load_watchlist, remove_company

    wl = load_watchlist()

    # Star links in the table come back as ?watch=<company>; toggle and clean the URL.
    qp = st.query_params
    if "watch" in qp:
        company = qp["watch"]
        if company in wl["watching"]:
            remove_company(wl, company)
        elif company in set(df["Company"]):
            add_company(wl, company)
        del st.query_params["watch"]
        st.rerun()

    sort_key = qp.get("sort", "mcap")
    if sort_key not in SORTS:
        sort_key = "mcap"

    df = df.copy()
    prev = df[PREV_PRICE_COL].where(df[PREV_PRICE_COL] > 0)
    df["Change %"] = ((df[PRICE_COL] / prev - 1) * 100).fillna(0.0)
    sectors = sorted(df["Sector"].unique())

    # --- Summary cards ----------------------------------------------------------
    def card(title: str, sort: str, frame: pd.DataFrame, value_col: str) -> str:
        rows = ""
        for _, r in frame.iterrows():
            text, cls = _signed(r[value_col])
            rows += (
                f'<div class="mk-row">{_icon(r["Company"], r["Sector"], sectors)}'
                f'<span class="nm">{html.escape(r["Company"])}</span>'
                f'<span class="px">{r[PRICE_COL]:,.2f}</span><span class="ch {cls}">{text}</span></div>'
            )
        return (
            f'<div class="mk-card"><div class="hd"><span>{title}</span>'
            f'<a href="?sort={sort}" target="_self">More ›</a></div>{rows}</div>'
        )

    st.markdown(
        '<div class="mk-cards">'
        + card("🔥 Hot · largest", "mcap", df.nlargest(3, "Market Cap (KES Bn)"), "Change %")
        + card("🚀 Top gainer", "gain", df.nlargest(3, "Change %"), "Change %")
        + card("📉 Top loser", "loss", df.nsmallest(3, "Change %"), "Change %")
        + card("🏆 Top performer · 5-yr", "perf", df.nlargest(3, "Avg Return %"), "Avg Return %")
        + "</div>",
        unsafe_allow_html=True,
    )

    # --- Category bar + controls -----------------------------------------------------
    category = st.pills(
        "Category",
        ["⭐ Favorites", "All", *sectors],
        default="All",
        key="mk_category",
        label_visibility="collapsed",
    ) or "All"

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('<div class="mk-title">Top NSE companies by market capitalization</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="mk-sub">A snapshot of every NSE company tracked here: latest price, price change, '
            "5-year average return and market cap. Tap ☆ to add a company to your watchlist.</div>",
            unsafe_allow_html=True,
        )
    with c2:
        s1, s2 = st.columns(2)
        period = s1.selectbox(
            "Change", ["Since last price", "FY2025", "5-yr average"], key="mk_period", label_visibility="collapsed"
        )
        sort_label = s2.selectbox(
            "Sort by",
            [v[0] for v in SORTS.values()],
            index=list(SORTS).index(sort_key),
            key=f"mk_sort_{sort_key}",
            label_visibility="collapsed",
        )
    sort_key = next(k for k, v in SORTS.items() if v[0] == sort_label)
    _, sort_col, ascending = SORTS[sort_key]

    change_col = {"Since last price": "Change %", "FY2025": "FY2025", "5-yr average": "Avg Return %"}[period]

    if category == "⭐ Favorites":
        view = df[df["Company"].isin(wl["watching"])]
    elif category == "All":
        view = df
    else:
        view = df[df["Sector"] == category]
    view = view.sort_values(sort_col, ascending=ascending)

    # --- Markets table ----------------------------------------------------------------------
    if view.empty:
        msg = "Your favorites list is empty — tap ☆ next to a company to add it." if category == "⭐ Favorites" \
            else "No companies in this category."
        st.info(msg, icon="⭐")
    else:
        body = ""
        watching = set(wl["watching"])
        for _, r in view.iterrows():
            ch_text, ch_cls = _signed(r[change_col])
            prev_px = r[PREV_PRICE_COL]
            prev_html = f'<div class="muted">prev {prev_px:,.2f}</div>' if pd.notna(prev_px) else ""
            on = r["Company"] in watching
            star = (
                f'<a class="star {"on" if on else ""}" target="_self" title="{"Remove from" if on else "Add to"} '
                f'watchlist" href="?watch={quote(r["Company"])}&sort={sort_key}">{"★" if on else "☆"}</a>'
            )
            body += (
                "<tr>"
                f'<td><div class="mk-name">{_icon(r["Company"], r["Sector"], sectors, "lg")}'
                f'<div><b>{html.escape(r["Company"])}</b><br><span>{html.escape(r["Sector"])}</span></div></div></td>'
                f"<td>{r[PRICE_COL]:,.2f}{prev_html}</td>"
                f'<td class="{ch_cls}">{ch_text}</td>'
                f'<td>{r["Market Cap (KES Bn)"]:,.1f} Bn</td>'
                f'<td>{r["Avg Return %"]:+.2f}%</td>'
                f"<td>{star}</td>"
                "</tr>"
            )
        st.markdown(
            '<table class="mk-table"><thead><tr><th>Name</th><th>Price (KES)</th>'
            f"<th>Change · {html.escape(period)}</th><th>Market cap (KES)</th><th>5-yr avg</th><th>Watch</th>"
            f"</tr></thead><tbody>{body}</tbody></table>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="home-note">⚠️ Market prices can be refreshed from a free public NSE data source '
        "(Update Market Prices tab); historical returns are illustrative demo data. "
        "Nothing on this page is financial advice.</div>",
        unsafe_allow_html=True,
    )


def render_companies(df: pd.DataFrame) -> pd.DataFrame:
    sectors = sorted(df["Sector"].unique())
    max_price = float(df[PRICE_COL].max())
    with st.expander("🔎 Filters", expanded=True):
        selected_sectors = st.multiselect("Sector", sectors, default=sectors)
        f1, f2 = st.columns(2)
        price_cap = f1.slider("Max market price (KES/share)", 0.0, max_price, max_price, step=1.0)
        min_avg_return = f2.slider("Min average return (%)", -30.0, 30.0, -30.0, step=0.5)

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
    render_sidebar(df)

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
