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

def _load_streamlit_secrets() -> None:
    """On Streamlit Community Cloud keys live in st.secrets; mirror top-level ones into the environment
    so everything (Groq, SMTP email alerts) reads them the same way as a local .env file."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float)) and not os.environ.get(key):
                os.environ[key] = str(value)
    except Exception:  # noqa: BLE001 - no secrets.toml when running locally
        pass


_load_streamlit_secrets()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

DATA_PATH = Path(__file__).parent / "data" / "companies.json"

PRICE_COL = "Market Price (KES/share)"
PREV_PRICE_COL = "Previous Price (KES/share)"
UPDATED_COL = "Price Last Updated"

from data.i18n import SW  # noqa: E402

THEMES = {
    "dark": {"--ki-bg": "#0B0E11", "--ki-card": "#181A20", "--ki-hover": "#1E2329", "--ki-border": "#2B3139",
             "--ki-text": "#EAECEF", "--ki-muted": "#848E9C", "--ki-accent-text": "#F0B90B"},
    "light": {"--ki-bg": "#FFFFFF", "--ki-card": "#FAFAFA", "--ki-hover": "#F5F5F5", "--ki-border": "#EAECEF",
              "--ki-text": "#1E2329", "--ki-muted": "#707A8A", "--ki-accent-text": "#C99400"},
}
THEME = THEMES["dark"]  # replaced per run in main() from the active Streamlit theme


def t(text: str) -> str:
    """Translate a UI string to Kiswahili when that language is selected (falls back to English)."""
    return SW.get(text, text) if st.session_state.get("lang") == "sw" else text


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
  padding:12px 4px 14px;margin-bottom:6px;border-bottom:1px solid var(--ki-border)}
.brand {color:var(--ki-accent-text);font-size:24px;font-weight:700;letter-spacing:.5px}
.brand small {color:var(--ki-muted);font-size:13px;font-weight:500;margin-left:10px;letter-spacing:0}
.topbar .right {text-align:right;font-size:13px;color:var(--ki-muted)}
.topbar .right b {color:var(--ki-text)}
.chip {display:inline-block;background:var(--ki-border);color:var(--ki-text);border-radius:6px;padding:2px 9px;
  font-size:12px;margin:4px 0 0 5px}
.mk-cards {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:12px 0 22px}
@media (max-width:1100px) {.mk-cards {grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:640px) {.mk-cards {grid-template-columns:minmax(0,1fr)}}
.mk-card {border:1px solid var(--ki-border);border-radius:14px;padding:14px 14px}
.mk-card .hd {display:flex;justify-content:space-between;font-size:13px;font-weight:600;margin-bottom:10px}
.mk-card .hd a {color:var(--ki-muted) !important;text-decoration:none;font-weight:500}
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
.mk-sub {font-size:12px;color:var(--ki-muted);margin-bottom:10px}
.mk-table {width:100%;border-collapse:collapse;font-size:14px}
.mk-table th {color:var(--ki-muted);font-weight:500;font-size:12px;text-align:right;padding:10px 8px;
  border-bottom:1px solid var(--ki-border)}
.mk-table th:first-child, .mk-table td:first-child {text-align:left}
.mk-table td {text-align:right;padding:14px 8px;border-bottom:1px solid var(--ki-hover)}
.mk-table tr:hover td {background:var(--ki-hover)}
.mk-name {display:flex;align-items:center;gap:10px}
.mk-name b {font-size:15px}
.mk-name span {color:var(--ki-muted);font-size:12px}
.mk-table .muted {color:var(--ki-muted);font-size:12px}
.star {font-size:18px;text-decoration:none !important;color:var(--ki-muted) !important}
.star.on {color:var(--ki-accent-text) !important}
.home-note {font-size:12px;color:var(--ki-muted);margin-top:18px}
.mk-stats {display:flex;flex-wrap:wrap;gap:12px 36px;margin:10px 0 2px;padding:12px 16px;border:1px solid var(--ki-border);
  border-radius:14px}
.mk-stats .l {font-size:12px;color:var(--ki-muted)}
.mk-stats .v {font-size:17px;font-weight:700}
.sb-card {border:1px solid var(--ki-border);border-radius:12px;padding:14px;margin-bottom:12px;background:var(--ki-card)}
.sb-top {display:flex;align-items:center;gap:12px}
.sb-avatar {width:46px;height:46px;border-radius:50%;background:#F0B90B;color:#0B0E11;font-weight:700;
  display:flex;align-items:center;justify-content:center;font-size:17px;flex-shrink:0}
.sb-name {font-weight:600;font-size:16px;line-height:1.2}
.sb-sub {font-size:12px;color:var(--ki-muted);word-break:break-all}
.sb-chips {display:flex;flex-wrap:wrap;gap:5px;margin-top:10px}
.sb-chip {font-size:11px;border-radius:6px;padding:2px 8px;background:rgba(240,185,11,.14);color:#F0B90B}
.sb-stats {display:grid;grid-template-columns:1fr 1fr;gap:8px}
.sb-stat {border:1px solid var(--ki-border);border-radius:10px;padding:8px 10px}
.sb-stat .l {font-size:11px;color:var(--ki-muted)}
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
        greeting = f"{t(_greeting())}, <b>{name}</b>"
        chips = [t(profile["riskTolerance"]) + " · " + t("risk"), t(profile["horizon"])]
        chips_html = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)
    else:
        greeting = f'{t("Welcome")}, <b>{t("guest investor")}</b>'
        chips_html = f'<span class="chip">{t("Set up your profile in the sidebar")}</span>'

    st.markdown(HOME_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="topbar">
  <div class="brand">📈 KENYA INVEST<small>{t("NSE Investment Explorer")}</small></div>
  <div class="right">{greeting} · {t("prices updated")} <b>{html.escape(str(raw.get("lastPriceUpdate", "—")))}</b>
    <div>{chips_html}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )


def _save_language() -> None:
    from data.profile import set_preference

    st.session_state["lang"] = st.session_state["pref_lang"]
    set_preference("language", st.session_state["lang"])


def render_preferences(sb) -> None:
    """Theme (light/dark) and language (English/Kiswahili) switches."""
    import streamlit.components.v1 as components

    current = "light" if THEME is THEMES["light"] else "dark"
    c1, c2 = sb.columns(2)
    choice = c1.segmented_control(
        t("Theme"), ["dark", "light"], default=current, key="pref_theme", label_visibility="collapsed",
        format_func=lambda v: "🌙" if v == "dark" else "☀️",
    )
    if choice and choice != current:
        # Streamlit keeps the chosen theme in localStorage; set it and reload so the whole app re-themes.
        name = "Light" if choice == "light" else "Dark"
        components.html(
            "<script>const p = window.parent;"
            f"p.localStorage.setItem(`stActiveTheme-${{p.location.pathname}}-v2`, JSON.stringify('{name}'));"
            "p.location.reload();</script>",
            height=0,
        )
    if "pref_lang" not in st.session_state:
        st.session_state["pref_lang"] = st.session_state.get("lang", "en")
    c2.segmented_control(
        t("Language"), ["en", "sw"], key="pref_lang", label_visibility="collapsed", on_change=_save_language,
        format_func=lambda v: "EN" if v == "en" else "SW",
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
    render_preferences(sb)
    sb.markdown(t("### 🧭 My dashboard"))

    # --- Profile card -----------------------------------------------------------
    name = profile["fullName"]
    if name:
        initials = "".join(part[0] for part in name.split()[:2]).upper()
        contact = " · ".join(html.escape(x) for x in (profile["email"], profile["location"]) if x)
        chips = [t(profile["experience"]), t(profile["riskTolerance"]) + " · " + t("risk"), t(profile["horizon"])]
        sb.markdown(
            f"""
<div class="sb-card">
  <div class="sb-top">
    <div class="sb-avatar">{html.escape(initials)}</div>
    <div><div class="sb-name">{html.escape(name)}</div><div class="sb-sub">{contact or t("No contact details yet")}</div></div>
  </div>
  <div class="sb-chips">{"".join(f'<span class="sb-chip">{html.escape(c)}</span>' for c in chips)}</div>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        sb.markdown(
            '<div class="sb-card"><div class="sb-top"><div class="sb-avatar">?</div>'
            f'<div><div class="sb-name">{t("Guest investor")}</div>'
            f'<div class="sb-sub">{t("Set up your profile under Account settings below.")}</div></div></div></div>',
            unsafe_allow_html=True,
        )

    # --- Account summary ------------------------------------------------------------
    sb.markdown(t("**💼 Account**"))
    pnl_cls = "up" if pnl_pct >= 0 else "dn"
    stats = [
        (t("Portfolio (KES)"), f"{total_value:,.0f}"),
        (t("P&L"), f'<span class="{pnl_cls}">{pnl_pct:+.2f}%</span>'),
        (t("Cash (KES)"), f"{portfolio['cash']:,.0f}"),
        (t("Holdings"), f"{len(portfolio['holdings'])}"),
        (t("Watching"), f"{len(wl['watching'])}"),
        (t("Alerts hit"), f"{len(triggered)}"),
    ]
    sb.markdown(
        '<div class="sb-card sb-stats">'
        + "".join(f'<div class="sb-stat"><div class="l">{l}</div><div class="v">{v}</div></div>' for l, v in stats)
        + "</div>",
        unsafe_allow_html=True,
    )
    for hit in triggered:
        sb.warning(f"🔔 {hit['company']} is {hit['direction']} KES {hit['target']:,.2f}", icon="🔔")

    # --- Account settings -------------------------------------------------------------
    # Keys include lastUpdated so fields refresh after the profile is saved elsewhere.
    ver = profile["lastUpdated"] or "new"
    with sb.expander(t("⚙️ Account settings"), expanded=not name):
        with st.form("sidebar_settings"):
            new_name = st.text_input(t("Full name *"), value=profile["fullName"], key=f"sb_name_{ver}")
            new_email = st.text_input(t("Email"), value=profile["email"], key=f"sb_email_{ver}")
            new_phone = st.text_input(t("Phone"), value=profile["phone"], key=f"sb_phone_{ver}")
            new_risk = st.selectbox(t("Risk tolerance"), RISK_LEVELS, index=RISK_LEVELS.index(profile["riskTolerance"]), format_func=t, key=f"sb_risk_{ver}_{st.session_state.get('lang')}"
            )
            new_horizon = st.selectbox(t("Investment horizon"), HORIZONS, index=HORIZONS.index(profile["horizon"]), format_func=t, key=f"sb_hor_{ver}_{st.session_state.get('lang')}"
            )
            if st.form_submit_button(t("💾 Save settings"), type="primary", use_container_width=True):
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
        st.caption(t("Goals, sectors, budget and notes are in the 👤 Profile tab."))

    with sb.expander(t("🧹 Reset data")):
        st.caption(t("These can't be undone."))
        if st.button(t("Reset paper portfolio"), key="sb_reset_portfolio", use_container_width=True):
            reset_portfolio()
            st.rerun()
        if st.button(t("Clear profile"), key="sb_clear_profile", use_container_width=True):
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
            f'<a href="?sort={sort}" target="_self">{t("More ›")}</a></div>{rows}</div>'
        )

    # Market statistics (illustrative volume from the reconstructed price history).
    last_bars = pd.DataFrame(
        [_company_history(raw, c).iloc[-1][["volume", "close"]] for c in df["Company"]], index=df["Company"]
    )
    df["Volume"] = df["Company"].map(last_bars["volume"])
    turnover = float((last_bars["volume"] * last_bars["close"]).sum())
    stats = [
        (t("Volume (shares)"), f"{last_bars['volume'].sum() / 1e6:,.1f} M"),
        (t("Turnover"), f"KES {turnover / 1e9:,.2f} Bn"),
        (t("Deals"), f"{int(last_bars['volume'].sum() / 2_500):,}"),
        (t("Advancers / decliners"), f'<span class="up">{int((df["Change %"] > 0).sum())}</span> / '
                                  f'<span class="dn">{int((df["Change %"] < 0).sum())}</span>'),
        (t("Market cap"), f"KES {df['Market Cap (KES Bn)'].sum():,.0f} Bn"),
    ]
    st.markdown(
        '<div class="mk-stats">'
        + "".join(f'<div><div class="l">{l}</div><div class="v">{v}</div></div>' for l, v in stats)
        + f'<div class="l" style="align-self:end">{t("NSE today · illustrative")}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="mk-cards">'
        + card(t("🔥 Hot · largest"), "mcap", df.nlargest(3, "Market Cap (KES Bn)"), "Change %")
        + card(t("🚀 Top gainer"), "gain", df.nlargest(3, "Change %"), "Change %")
        + card(t("📉 Top loser"), "loss", df.nsmallest(3, "Change %"), "Change %")
        + card(t("🏆 Top performer · 5-yr"), "perf", df.nlargest(3, "Avg Return %"), "Avg Return %")
        + "</div>",
        unsafe_allow_html=True,
    )

    # --- Category bar + controls -----------------------------------------------------
    category = st.pills(t("Category"),
        ["⭐ Favorites", "All", *sectors],
        default="All",
        format_func=t, key="mk_category",
        label_visibility="collapsed",
    ) or "All"

    search = st.text_input(t("Search"), placeholder=t("🔎 Search companies or sectors…"), key="mk_search",
                           label_visibility="collapsed")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f'<div class="mk-title">{t("Top NSE companies by market capitalization")}</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="mk-sub">' + t("A snapshot of every NSE company tracked here: latest price, price change, "
            "5-year average return and market cap. Tap ☆ to add a company to your watchlist, or Trade to open it "
            "in the trading view.") + "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        s1, s2 = st.columns(2)
        period = s1.selectbox(t("Change"), ["Since last price", "FY2025", "5-yr average"], format_func=t, key=f"mk_period_{st.session_state.get('lang')}", label_visibility="collapsed"
        )
        sort_label = s2.selectbox(t("Sort by"),
            [v[0] for v in SORTS.values()],
            index=list(SORTS).index(sort_key),
            format_func=t, key=f"mk_sort_{sort_key}_{st.session_state.get('lang')}",
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
    if search:
        view = view[view["Company"].str.contains(search, case=False, regex=False)
                    | view["Sector"].str.contains(search, case=False, regex=False)]

    # --- Markets table ----------------------------------------------------------------------
    if view.empty:
        msg = t("Your favorites list is empty — tap ☆ next to a company to add it.") if category == "⭐ Favorites" \
            else t("No companies match.") if search else t("No companies in this category.")
        st.info(msg, icon="⭐")
    else:
        body = ""
        watching = set(wl["watching"])
        for _, r in view.iterrows():
            ch_text, ch_cls = _signed(r[change_col])
            prev_px = r[PREV_PRICE_COL]
            prev_html = f'<div class="muted">{t("prev")} {prev_px:,.2f}</div>' if pd.notna(prev_px) else ""
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
                f'<td>{r["Volume"] / 1e3:,.0f} K</td>'
                f'<td>{r["Avg Return %"]:+.2f}%</td>'
                f'<td>{star}<a class="trade-link" target="_self" href="?trade={quote(r["Company"])}">{t("Trade")}</a></td>'
                "</tr>"
            )
        st.markdown(
            f'<table class="mk-table"><thead><tr><th>{t("Name")}</th><th>{t("Price (KES)")}</th>'
            f"<th>{t('Change')} · {html.escape(t(period))}</th><th>{t('Market cap (KES)')}</th><th>{t('Volume')}</th>"
            f"<th>{t('5-yr avg')}</th><th>{t('Actions')}</th>"
            f"</tr></thead><tbody>{body}</tbody></table>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="home-note">' + t("⚠️ Market prices can be refreshed from a free public NSE data source "
        "(Update Market Prices tab); historical returns are illustrative demo data. "
        "Nothing on this page is financial advice.") + "</div>",
        unsafe_allow_html=True,
    )


TRADE_TAB = "📈 Trade"

TRADE_CSS = """
<style>
.td-head {display:flex;flex-wrap:wrap;align-items:center;gap:24px;border:1px solid var(--ki-border);border-radius:14px;
  padding:14px 18px;margin:4px 0 12px}
.td-head .who {display:flex;align-items:center;gap:12px}
.td-head .who b {font-size:18px}
.td-head .who span {color:var(--ki-muted);font-size:12px}
.td-head .big {font-size:24px;font-weight:700}
.td-stat .l {color:var(--ki-muted);font-size:12px}
.td-stat .v {font-size:14px;font-weight:600}
.td-note {color:var(--ki-muted);font-size:11px;margin-top:-6px}
.td-sum {display:flex;justify-content:space-between;font-size:13px;color:var(--ki-muted);margin:2px 0}
.td-sum b {color:var(--ki-text)}
.st-key-td_submit_buy button {background:#0ECB81 !important;border-color:#0ECB81 !important;color:#0B0E11 !important;
  font-weight:700}
.st-key-td_submit_sell button {background:#F6465D !important;border-color:#F6465D !important;color:#fff !important;
  font-weight:700}
.td-list {width:100%;border-collapse:collapse;font-size:13px}
.td-list td {padding:7px 4px;border-bottom:1px solid var(--ki-hover)}
.td-list td:not(:first-child) {text-align:right}
.td-list a {color:var(--ki-text) !important;text-decoration:none}
.td-list a:hover {color:#F0B90B !important}
.td-list tr.sel td {background:var(--ki-hover)}
.trade-link {color:var(--ki-accent-text) !important;text-decoration:none !important;font-weight:600;margin-left:12px}
.ob {width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
.ob th {color:var(--ki-muted);font-weight:500;text-align:right;padding:3px 4px}
.ob th:first-child, .ob td:first-child {text-align:left}
.ob td {text-align:right;padding:2px 4px;position:relative}
.ob .mid {font-size:17px;font-weight:700;padding:8px 4px}
.ob-h {font-size:13px;font-weight:600;margin:4px 0}
.st-tag {font-size:11px;border-radius:4px;padding:1px 6px;font-weight:600}
.st-FILLED {background:rgba(14,203,129,.15);color:#0ECB81}
.st-CANCELLED, .st-EXPIRED {background:var(--ki-border);color:var(--ki-muted)}
.st-REJECTED {background:rgba(246,70,93,.15);color:#F6465D}
.st-OPEN, .st-TRIGGERED {background:rgba(240,185,11,.15);color:#F0B90B}
</style>
"""

RANGES = {"1M": (21, None), "3M": (63, None), "6M": (126, None), "1Y": (252, "W"), "5Y": (None, "W")}
OVERLAYS = ["MA 7", "MA 25", "EMA 20", "Bollinger"]
LOWER_PANES = ["Volume", "RSI", "MACD", "None"]
ORDER_LABELS = {"Market": "MARKET", "Limit": "LIMIT", "Stop-loss": "STOP_LOSS",
                "Take-profit": "TAKE_PROFIT", "Stop-limit": "STOP_LIMIT"}


@st.cache_data
def _history(company_json: str, return_years: tuple[str, ...], today_iso: str) -> pd.DataFrame:
    from data.price_history import build_history

    return build_history(json.loads(company_json), list(return_years), date.fromisoformat(today_iso))


def _company_history(raw: dict, company: str) -> pd.DataFrame:
    c = next(x for x in raw["companies"] if x["company"] == company)
    return _history(json.dumps(c, sort_keys=True), tuple(raw["returnYears"]), date.today().isoformat())


def _price_chart(full: pd.DataFrame, start, kind: str, overlays: list[str], lower: str,
                 price: float, avg_cost: float | None):
    import altair as alt

    from data.indicators import bollinger, ema, macd, rsi, sma

    data = full.copy()
    data["MA 7"] = sma(data["close"], 7)
    data["MA 25"] = sma(data["close"], 25)
    data["EMA 20"] = ema(data["close"], 20)
    data["BB low"], data["BB mid"], data["BB high"] = bollinger(data["close"])
    data["RSI"] = rsi(data["close"])
    data["MACD"], data["Signal"], data["Hist"] = macd(data["close"])
    data = data[data["date"] >= start]

    domain = [data["date"].min().isoformat(), data["date"].max().isoformat()]
    x = alt.X("date:T", title=None, axis=alt.Axis(grid=False), scale=alt.Scale(domain=domain))
    tooltip = [alt.Tooltip("date:T", title="Date")] + [
        alt.Tooltip(f"{c}:Q", format=",.2f") for c in ("open", "high", "low", "close")
    ] + [alt.Tooltip("volume:Q", format=",.0f")]
    y = alt.Y("low:Q" if kind == "Candles" else "close:Q", scale=alt.Scale(zero=False), title="KES")

    if kind == "Candles":
        base = alt.Chart(data).encode(
            x=x, color=alt.condition("datum.open <= datum.close", alt.value(UP), alt.value(DOWN)), tooltip=tooltip
        )
        bar_width = max(1.5, min(10.0, 560 / max(len(data), 1) * 0.7))
        chart = base.mark_rule().encode(y=y, y2="high:Q") + base.mark_bar(size=bar_width).encode(
            y="open:Q", y2="close:Q"
        )
    else:
        base = alt.Chart(data).encode(x=x, tooltip=tooltip)
        chart = base.mark_area(color=ACCENT, opacity=0.12).encode(y=y) + base.mark_line(
            color=ACCENT, strokeWidth=2
        ).encode(y="close:Q")

    colors = {"MA 7": "#F0B90B", "MA 25": "#C0A2FF", "EMA 20": "#00AAE4"}
    for name in overlays:
        if name == "Bollinger":
            band = alt.Chart(data).encode(x=x)
            chart += band.mark_area(opacity=0.08, color="#848E9C").encode(y="BB low:Q", y2="BB high:Q")
            chart += band.mark_line(color="#848E9C", strokeWidth=1, strokeDash=[3, 3]).encode(y="BB mid:Q")
        else:
            chart += alt.Chart(data).mark_line(color=colors[name], strokeWidth=1.3).encode(x=x, y=f"{name}:Q")

    lines = [{"label": f"Last {price:,.2f}", "y": price, "c": ACCENT}]
    if avg_cost:
        lines.append({"label": f"Your avg cost {avg_cost:,.2f}", "y": avg_cost, "c": "#848E9C"})
    ref = alt.Chart(pd.DataFrame(lines))
    chart += ref.mark_rule(strokeDash=[4, 4]).encode(y="y:Q", color=alt.Color("c:N", scale=None))
    chart += ref.mark_text(align="left", dx=4, dy=-6, fontSize=11).encode(
        y="y:Q", x=alt.value(0), text="label:N", color=alt.Color("c:N", scale=None)
    )
    chart = chart.properties(height=330 if lower != "None" else 420)

    if lower == "Volume":
        pane = alt.Chart(data).mark_bar(size=max(1.0, min(8.0, 560 / max(len(data), 1) * 0.7))).encode(
            x=x, y=alt.Y("volume:Q", title="Volume", axis=alt.Axis(format="~s")),
            color=alt.condition("datum.open <= datum.close", alt.value(UP), alt.value(DOWN)), tooltip=tooltip,
        )
    elif lower == "RSI":
        pane = alt.Chart(data).mark_line(color="#C0A2FF").encode(
            x=x, y=alt.Y("RSI:Q", scale=alt.Scale(domain=[0, 100]), title="RSI 14")
        ) + alt.Chart(pd.DataFrame({"v": [30, 70]})).mark_rule(strokeDash=[3, 3], color="#848E9C").encode(y="v:Q")
    elif lower == "MACD":
        pane = alt.Chart(data).mark_bar().encode(
            x=x, y=alt.Y("Hist:Q", title="MACD"),
            color=alt.condition("datum.Hist >= 0", alt.value(UP), alt.value(DOWN)),
        ) + alt.Chart(data).mark_line(color=ACCENT).encode(x=x, y="MACD:Q") + alt.Chart(data).mark_line(
            color="#00AAE4"
        ).encode(x=x, y="Signal:Q")
    else:
        return chart, None
    return chart, pane.properties(height=110)


def _order_book_html(company: str, price: float) -> str:
    from data.market_depth import order_book, recent_trades

    bids, asks = order_book(company, price)
    top = max(bids["total"].max(), asks["total"].max())

    def rows(frame: pd.DataFrame, cls: str, color: str) -> str:
        out = ""
        for _, r in frame.iterrows():
            pct = r["total"] / top * 100
            out += (
                f'<tr style="background:linear-gradient(to left, {color} {pct:.0f}%, transparent {pct:.0f}%)">'
                f'<td class="{cls}">{r["price"]:,.2f}</td><td>{r["shares"]:,.0f}</td><td>{r["total"]:,.0f}</td></tr>'
            )
        return out

    book = (
        f'<div class="ob-h">{t("Order book")}</div><table class="ob"><tr><th>{t("Price")}</th>'
        f'<th>{t("Shares")}</th><th>{t("Total")}</th></tr>'
        + rows(asks.iloc[::-1], "dn", "rgba(246,70,93,.12)")
        + f'<tr><td class="mid" colspan="3">{price:,.2f}</td></tr>'
        + rows(bids, "up", "rgba(14,203,129,.12)")
        + "</table>"
    )
    tape = "".join(
        f'<tr><td class="{"up" if t.side == "BUY" else "dn"}">{t.price:,.2f}</td><td>{t.shares:,.0f}</td>'
        f"<td>{t.time}</td></tr>"
        for t in recent_trades(company, price).itertuples()
    )
    return (
        book
        + f'<div class="ob-h" style="margin-top:14px">{t("Recent trades")}</div>'
        + f'<table class="ob"><tr><th>{t("Price")}</th><th>{t("Shares")}</th><th>{t("Time")}</th></tr>{tape}</table>'
    )


def _set_qty(qty_key: str, pct_key: str, max_shares: int) -> None:
    pct = st.session_state.get(pct_key)
    if pct:
        st.session_state[qty_key] = int(max_shares * int(pct.rstrip("%")) / 100)


def render_trade_desk(df: pd.DataFrame, raw: dict) -> None:
    from urllib.parse import quote

    from data.fees import TOTAL_RATE, fee_breakdown, total_fees
    from data.portfolio import ORDER_TYPES, cancel_order, load_portfolio, place_order
    from data.price_history import resample
    from data.profile import load_profile
    from data.statements import contract_note_pdf

    st.markdown(TRADE_CSS, unsafe_allow_html=True)
    sectors = sorted(df["Sector"].unique())
    companies = set(df["Company"])

    # A ?trade=<company> link (from the markets table or the list below) selects that company once.
    wanted = st.query_params.get("trade")
    if wanted in companies and st.session_state.get("td_last_param") != wanted:
        st.session_state["td_last_param"] = wanted
        st.session_state["td_sector"] = "All sectors"
        st.session_state["td_company"] = wanted

    sector = st.pills(t("Sector"), ["All sectors", *sectors], default="All sectors", format_func=t, key="td_sector", label_visibility="collapsed"
    ) or "All sectors"
    options = sorted(df["Company"] if sector == "All sectors" else df.loc[df["Sector"] == sector, "Company"])
    if st.session_state.get("td_company") not in options:
        st.session_state["td_company"] = options[0]
    company = st.selectbox(t("🔎 Search or pick a company"), options, key="td_company")

    row = df.loc[df["Company"] == company].iloc[0]
    price = float(row[PRICE_COL])
    prev = row[PREV_PRICE_COL]
    change = (price / prev - 1) * 100 if pd.notna(prev) and prev else 0.0
    ch_text, ch_cls = _signed(change)

    portfolio = load_portfolio()
    holding = next((h for h in portfolio["holdings"] if h["company"] == company), None)
    held = holding["shares"] if holding else 0.0
    hist = _company_history(raw, company)
    today_bar = hist.iloc[-1]

    # --- Header strip with market statistics ------------------------------------------------
    stats = [
        (t("Change"), f'<span class="{ch_cls}">{ch_text}</span>'),
        (t("Day high"), f'{today_bar["high"]:,.2f}'),
        (t("Day low"), f'{today_bar["low"]:,.2f}'),
        (t("Volume"), f'{today_bar["volume"]:,.0f}'),
        (t("Turnover (KES)"), f'{today_bar["volume"] * today_bar["close"] / 1e6:,.1f} M'),
        (t("Market cap"), f'{row["Market Cap (KES Bn)"]:,.1f} Bn'),
        (t("You hold"), f"{held:g} {t('shares')}"),
    ]
    st.markdown(
        f'<div class="td-head"><div class="who">{_icon(company, row["Sector"], sectors, "lg")}'
        f'<div><b>{html.escape(company)}</b><br><span>{html.escape(row["Sector"])} · NSE · KES</span></div></div>'
        f'<div class="big {ch_cls}">{price:,.2f}</div>'
        + "".join(f'<div class="td-stat"><div class="l">{l}</div><div class="v">{v}</div></div>' for l, v in stats)
        + "</div>",
        unsafe_allow_html=True,
    )

    book_col, chart_col, order_col = st.columns([2.2, 5.3, 2.8], gap="small")

    with book_col:
        st.markdown(_order_book_html(company, price), unsafe_allow_html=True)

    # --- Chart ------------------------------------------------------------------------------
    with chart_col:
        r1, r2 = st.columns([3, 2])
        rng = r1.segmented_control(t("Range"), list(RANGES), default="6M", key="td_range",
                                   label_visibility="collapsed") or "6M"
        kind = r2.segmented_control(t("Chart"), ["Candles", "Line"], default="Candles", format_func=t, key="td_kind",
                                    label_visibility="collapsed") or "Candles"
        i1, i2 = st.columns([3, 2])
        overlays = i1.multiselect(t("Indicators"), OVERLAYS, default=["MA 7", "MA 25"], key="td_overlays",
                                  placeholder=t("Add indicators"), label_visibility="collapsed")
        lower = i2.segmented_control(t("Lower pane"), LOWER_PANES, default="Volume", format_func=t, key="td_lower",
                                     label_visibility="collapsed") or "None"
        n_days, rule = RANGES[rng]
        start = hist["date"].iloc[-n_days] if n_days else hist["date"].iloc[0]
        full = resample(hist, rule) if rule else hist
        main_chart, pane = _price_chart(full, start, kind, overlays, lower, price,
                                        holding["avgCost"] if holding else None)
        st.altair_chart(main_chart, use_container_width=True)
        if pane is not None:
            st.altair_chart(pane, use_container_width=True)
        st.markdown(
            '<div class="td-note">' + t("Illustrative price history, volume and order book — reconstructed or "
            "simulated, not real NSE trading data. The last candle ends at the current market price.") + "</div>",
            unsafe_allow_html=True,
        )

    # --- Order panel ------------------------------------------------------------------------
    with order_col:
        if msg := st.session_state.pop("td_msg", None):
            (st.success if msg[0] else st.error)(msg[1])

        side = st.segmented_control(t("Side"), ["Buy", "Sell"], default="Buy", format_func=t, key="td_side",
                                    label_visibility="collapsed") or "Buy"
        type_opts = ["Market", "Limit", "Stop-limit"] + (["Stop-loss", "Take-profit"] if side == "Sell" else [])
        type_key = f"td_type_{st.session_state.get('lang')}"
        if st.session_state.get(type_key) not in type_opts:
            st.session_state[type_key] = "Market"
        otype_label = st.selectbox(t("Order type"), type_opts, format_func=t, key=type_key)
        otype = ORDER_LABELS[otype_label]

        limit = stop = None
        if otype in ("STOP_LOSS", "TAKE_PROFIT", "STOP_LIMIT"):
            default_stop = price * (0.95 if (otype == "STOP_LOSS" or (otype == "STOP_LIMIT" and side == "Sell")) else 1.05)
            label = t({"STOP_LOSS": "Stop price (sell if at or below)", "TAKE_PROFIT": "Target price (sell if at or above)",
                       "STOP_LIMIT": "Stop / trigger price"}[otype])
            stop = st.number_input(label, min_value=0.01, value=round(default_stop, 2), step=0.05, format="%.2f",
                                   key=f"td_stop_{company}_{otype}_{side}")
        if otype in ("LIMIT", "STOP_LIMIT"):
            limit = st.number_input(t("Limit price (KES)"), min_value=0.01, value=float(stop or price), step=0.05,
                                    format="%.2f", key=f"td_limit_{company}_{otype}_{side}")
        tif = "GTC"
        if otype != "MARKET":
            tif_label = st.segmented_control(t("Time in force"), ["Good till cancelled", "Day only"],
                                             default="Good till cancelled", format_func=t, key="td_tif",
                                             label_visibility="collapsed") or "Good till cancelled"
            tif = "DAY" if tif_label == "Day only" else "GTC"

        exec_price = limit or stop or price
        max_shares = int(portfolio["cash"] // (exec_price * (1 + TOTAL_RATE))) if side == "Buy" else int(held)
        qty_key, pct_key = f"td_qty_{company}_{side}", f"td_pct_{company}_{side}"
        qty = st.number_input(t("Shares"), min_value=0, step=1, key=qty_key)
        st.segmented_control(t("Amount"), ["25%", "50%", "75%", "100%"], key=pct_key, label_visibility="collapsed",
                             on_change=_set_qty, args=(qty_key, pct_key, max_shares))

        consideration = qty * exec_price
        fees = total_fees(consideration)
        net = consideration + fees if side == "Buy" else consideration - fees
        avail = f"KES {portfolio['cash']:,.2f}" if side == "Buy" else f"{held:g} {t('shares')}"
        st.markdown(
            f'<div class="td-sum"><span>{t("Available")}</span><b>{avail}</b></div>'
            f'<div class="td-sum"><span>{t("Max " + side.lower())}</span><b>{max_shares:,} {t("shares")}</b></div>'
            f'<div class="td-sum"><span>{t("Consideration")}</span><b>KES {consideration:,.2f}</b></div>'
            f'<div class="td-sum"><span>{t("Fees")} (~{TOTAL_RATE * 100:.2f}%)</span><b>KES {fees:,.2f}</b></div>'
            f'<div class="td-sum"><span>{t("Total cost") if side == "Buy" else t("You receive")}</span>'
            f"<b>KES {net:,.2f}</b></div>",
            unsafe_allow_html=True,
        )
        with st.popover(t("Fee breakdown"), use_container_width=True):
            st.dataframe(pd.DataFrame(fee_breakdown(consideration), columns=["Charge", "KES"]),
                         hide_index=True, use_container_width=True)
            st.caption(t("Typical NSE retail charges; your broker's tariff may differ."))

        if st.button(f"{t(side)} {company}", key=f"td_submit_{side.lower()}", use_container_width=True):
            result = place_order(portfolio, side, company, float(qty), otype, price, limit=limit, stop=stop, tif=tif)
            st.session_state["td_msg"] = (result.ok, result.message)
            st.rerun()
        st.caption(t("Simulated trading with virtual cash — no real orders or money."))

        st.markdown(f"**{html.escape(sector)}**")
        rows = ""
        for _, r in df[df["Company"].isin(options)].sort_values("Market Cap (KES Bn)", ascending=False).iterrows():
            c_prev = r[PREV_PRICE_COL]
            c_chg = (r[PRICE_COL] / c_prev - 1) * 100 if pd.notna(c_prev) and c_prev else 0.0
            chg_txt, chg_cls = _signed(c_chg)
            sel = ' class="sel"' if r["Company"] == company else ""
            rows += (
                f'<tr{sel}><td><a target="_self" href="?trade={quote(r["Company"])}">'
                f'{html.escape(r["Company"])}</a></td><td>{r[PRICE_COL]:,.2f}</td><td class="{chg_cls}">{chg_txt}</td></tr>'
            )
        st.markdown(f'<table class="td-list">{rows}</table>', unsafe_allow_html=True)

    # --- Orders, history, holdings ------------------------------------------------------------
    price_by_company = dict(zip(df["Company"], df[PRICE_COL]))
    t_open, t_orders, t_hist, t_hold = st.tabs(
        [f"{t('Open orders')} ({len(portfolio['openOrders'])})", t("Order history"), t("Trade history"),
         t("Holdings")]
    )
    with t_open:
        if not portfolio["openOrders"]:
            st.caption(t("No open orders. Limit, stop and take-profit orders wait here until their price is reached."))
        for o in portfolio["openOrders"]:
            c1, c2 = st.columns([6, 1])
            color = "up" if o["side"] == "BUY" else "dn"
            status = "TRIGGERED" if o.get("triggered") else "OPEN"
            prices = " · ".join(
                p for p in (f"stop {o['stop']:,.2f}" if o.get("stop") else "",
                            f"limit {o['limit']:,.2f}" if o.get("limit") else "") if p
            )
            c1.markdown(
                f'<span class="st-tag st-{status}">{status}</span> <span class="{color}">{o["side"]}</span> '
                f'{ORDER_TYPES[o["type"]]} · {o["shares"]:g} × **{o["company"]}** · {prices} · {o["tif"]} · '
                f'now KES {price_by_company.get(o["company"], 0):,.2f} · placed {o["placedAt"][:16].replace("T", " ")}',
                unsafe_allow_html=True,
            )
            if c2.button(t("Cancel"), key=f"td_cancel_{o['id']}"):
                result = cancel_order(portfolio, o["id"])
                st.session_state["td_msg"] = (result.ok, result.message)
                st.rerun()
    with t_orders:
        if portfolio["orderHistory"]:
            hist_rows = [
                {
                    "#": o["id"],
                    "Status": o["status"],
                    "Placed": o["placedAt"][:16].replace("T", " "),
                    "Closed": o["closedAt"][:16].replace("T", " "),
                    "Company": o["company"],
                    "Side": o["side"],
                    "Type": ORDER_TYPES[o["type"]],
                    "Shares": o["shares"],
                    "Stop": o.get("stop"),
                    "Limit": o.get("limit"),
                    "Fill price": o.get("fillPrice"),
                    "TIF": o["tif"],
                    "Note": o.get("note", ""),
                }
                for o in reversed(portfolio["orderHistory"])
            ]
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)
        else:
            st.caption(t("No orders yet."))
    with t_hist:
        trades = [(i, t) for i, t in enumerate(portfolio["trades"]) if t["action"] in ("BUY", "SELL")]
        if trades:
            st.dataframe(pd.DataFrame([t for _, t in reversed(trades)]), use_container_width=True, hide_index=True)
            labels = {
                f'#{i + 1} · {t["timestamp"][:16].replace("T", " ")} · {t["action"]} {t["shares"]:g} {t["company"]}': i
                for i, t in reversed(trades)
            }
            n1, n2 = st.columns([3, 1])
            pick = n1.selectbox(t("Contract note for trade"), list(labels), key="td_note_pick")
            idx = labels[pick]
            n2.download_button(t("📄 Contract note (PDF)"),
                data=contract_note_pdf(portfolio["trades"][idx], load_profile()["fullName"], idx + 1),
                file_name=f"contract_note_{idx + 1:05d}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.caption(t("No trades yet."))
    with t_hold:
        if portfolio["holdings"]:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Company": h["company"],
                            "Shares": h["shares"],
                            "Avg cost (incl. fees)": round(h["avgCost"], 2),
                            "Price": price_by_company.get(h["company"], h["avgCost"]),
                            "Value (KES)": round(h["shares"] * price_by_company.get(h["company"], h["avgCost"]), 2),
                            "P&L %": round((price_by_company.get(h["company"], h["avgCost"]) / h["avgCost"] - 1) * 100, 2),
                        }
                        for h in portfolio["holdings"]
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption(t("No holdings yet."))


def render_companies(df: pd.DataFrame) -> pd.DataFrame:
    sectors = sorted(df["Sector"].unique())
    max_price = float(df[PRICE_COL].max())
    with st.expander(t("🔎 Filters"), expanded=True):
        selected_sectors = st.multiselect(t("Sector"), sectors, default=sectors)
        f1, f2 = st.columns(2)
        price_cap = f1.slider(t("Max market price (KES/share)"), 0.0, max_price, max_price, step=1.0)
        min_avg_return = f2.slider(t("Min average return (%)"), -30.0, 30.0, -30.0, step=0.5)

    filtered = df[
        df["Sector"].isin(selected_sectors)
        & (df[PRICE_COL] <= price_cap)
        & (df["Avg Return %"] >= min_avg_return)
    ]

    st.subheader(f"{t('Investment opportunities')} ({len(filtered)} {t('companies')})")
    display_cols = ["Sector", PRICE_COL, "Market Cap (KES Bn)", "Avg Return %", UPDATED_COL]
    st.dataframe(filtered.set_index("Company")[display_cols], use_container_width=True)

    if not filtered.empty:
        st.subheader(t("Average return by company"))
        st.bar_chart(filtered.set_index("Company")["Avg Return %"])

    return filtered


def render_live_refresh(raw: dict) -> None:
    st.subheader(t("🔄 Live NSE prices"))
    last_fetch = raw.get("lastFetch")

    if last_fetch is None:
        st.caption(t("No live refresh has been run yet. Prices below are the last manually-set values."))
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

    if st.button(t("🔄 Refresh live prices now"), type="primary"):
        from data.fetch_prices import update_companies_json

        with st.spinner(t("Fetching live NSE prices...")):
            result = update_companies_json()
        st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1
        if result.ok:
            st.success(result.message)
        else:
            st.error(result.message)
        st.rerun()


def _value_history(portfolio: dict, raw: dict) -> pd.DataFrame:
    """Daily portfolio value rebuilt from the trade log and (illustrative) price history."""
    trades = portfolio["trades"]
    if not trades:
        return pd.DataFrame(columns=["date", "value"])
    days = pd.bdate_range(trades[0]["timestamp"][:10], date.today())
    if len(days) == 0 or days[-1].date() != date.today():
        days = days.append(pd.DatetimeIndex([pd.Timestamp(date.today())]))
    held_companies = {t["company"] for t in trades if t["action"] in ("BUY", "SELL")}
    closes = {
        c: _company_history(raw, c).set_index("date")["close"].reindex(days, method="ffill")
        for c in held_companies
        if any(x["company"] == c for x in raw["companies"])
    }
    rows, cash, shares, i = [], portfolio["startingCash"], {}, 0
    for day in days:
        day_iso = day.date().isoformat()
        while i < len(trades) and trades[i]["timestamp"][:10] <= day_iso:
            t = trades[i]
            cash = t["cashAfter"]
            if t["action"] in ("BUY", "SELL"):
                shares[t["company"]] = shares.get(t["company"], 0) + (t["shares"] if t["action"] == "BUY" else -t["shares"])
            i += 1
        value = cash + sum(n * float(closes[c].get(day, 0) or 0) for c, n in shares.items() if c in closes)
        rows.append({"date": day, "value": round(value, 2)})
    return pd.DataFrame(rows)


def render_trading(df: pd.DataFrame, raw: dict) -> None:
    import altair as alt

    from data.portfolio import (
        add_auto_invest,
        deposit,
        holdings_market_value,
        load_portfolio,
        remove_auto_invest,
        reset_portfolio,
        DIVIDEND_WHT,
        set_auto_invest_active,
        shares_held_on,
        withdraw,
    )
    from data.profile import load_profile
    from data.statements import statement_pdf

    portfolio = load_portfolio()
    profile = load_profile()
    price_by_company = dict(zip(df["Company"], df[PRICE_COL]))
    sector_of = dict(zip(df["Company"], df["Sector"]))

    st.subheader(t("💼 Paper Trading Portfolio"))
    st.caption(t("Simulated trading only — no real money or real brokerage orders are involved. Trades include typical "
        "NSE charges and are tracked locally in `data/portfolio.json`. To buy or sell, use the **📈 Trade** tab.")
    )

    holdings_value = holdings_market_value(portfolio, price_by_company)
    total_value = portfolio["cash"] + holdings_value
    baseline = portfolio["startingCash"] + portfolio.get("netDeposits", 0.0)
    pnl = total_value - baseline
    pnl_pct = (pnl / baseline * 100) if baseline else 0.0
    dividends = sum(t["total"] for t in portfolio["trades"] if t["action"] == "DIVIDEND")
    fees_paid = sum(t.get("fees", 0) for t in portfolio["trades"] if t["action"] in ("BUY", "SELL"))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(t("Cash (KES)"), f"{portfolio['cash']:,.2f}")
    c2.metric(t("Holdings value (KES)"), f"{holdings_value:,.2f}")
    c3.metric(t("Total portfolio (KES)"), f"{total_value:,.2f}")
    c4.metric(t("Total P&L (KES)"), f"{pnl:,.2f}", f"{pnl_pct:+.2f}%")
    c5.metric(t("Dividends / fees paid"), f"{dividends:,.0f} / {fees_paid:,.0f}")

    # --- Portfolio analysis ------------------------------------------------------------------
    st.markdown(t("#### 📊 Portfolio analysis"))
    a1, a2 = st.columns([2, 3])
    with a1:
        alloc = {"Cash": portfolio["cash"]}
        for h in portfolio["holdings"]:
            sec = sector_of.get(h["company"], "Other")
            alloc[sec] = alloc.get(sec, 0) + h["shares"] * price_by_company.get(h["company"], h["avgCost"])
        alloc_df = pd.DataFrame({"Asset": list(alloc), "Value": list(alloc.values())})
        alloc_df["Share"] = alloc_df["Value"] / alloc_df["Value"].sum()
        palette = [ACCENT, *SECTOR_COLORS[1:]]
        donut = (
            alt.Chart(alloc_df)
            .mark_arc(innerRadius=60, stroke=THEME["--ki-bg"], strokeWidth=2)
            .encode(
                theta="Value:Q",
                color=alt.Color("Asset:N", scale=alt.Scale(range=palette), legend=alt.Legend(title=None, orient="bottom", columns=2)),
                tooltip=["Asset", alt.Tooltip("Value:Q", format=",.0f"), alt.Tooltip("Share:Q", format=".1%")],
            )
            .properties(height=300, title=t("Allocation by sector"))
        )
        st.altair_chart(donut, use_container_width=True)
    with a2:
        vh = _value_history(portfolio, raw)
        if len(vh) > 1:
            line = (
                alt.Chart(vh)
                .mark_line(color=ACCENT, strokeWidth=2)
                .encode(
                    x=alt.X("date:T", title=None),
                    y=alt.Y("value:Q", title="KES", scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip("date:T"), alt.Tooltip("value:Q", format=",.2f", title="Value")],
                )
                .properties(height=300, title=t("Portfolio value over time"))
            )
            st.altair_chart(line, use_container_width=True)
        else:
            st.info(t("Your value-over-time chart appears after your first trade."), icon="📈")

    if portfolio["holdings"]:
        by_sector = {}
        for h in portfolio["holdings"]:
            sec = sector_of.get(h["company"], "Other")
            val = h["shares"] * price_by_company.get(h["company"], h["avgCost"])
            cost = h["shares"] * h["avgCost"]
            v, c = by_sector.get(sec, (0.0, 0.0))
            by_sector[sec] = (v + val, c + cost)
        st.dataframe(
            pd.DataFrame(
                [{"Sector": s, "Value (KES)": round(v, 2), "Cost (KES)": round(c, 2),
                  "Unrealized P&L (KES)": round(v - c, 2), "P&L %": round((v / c - 1) * 100, 2) if c else 0.0}
                 for s, (v, c) in sorted(by_sector.items(), key=lambda kv: -kv[1][0])]
            ),
            use_container_width=True,
            hide_index=True,
        )

    # --- Auto-invest -------------------------------------------------------------------------
    st.markdown(t("#### 🔁 Auto-invest"))
    st.caption(t("Buy a fixed amount of a company every week or month. Plans run automatically when the app is "
               "opened on or after their due date (missed periods aren't back-filled)."))
    with st.form("auto_invest_form"):
        f1, f2, f3, f4 = st.columns([3, 2, 2, 2])
        ai_company = f1.selectbox(t("Company"), sorted(df["Company"]), key="ai_company")
        ai_amount = f2.number_input(t("Amount (KES)"), min_value=100.0, step=500.0,
                                    value=float(profile["monthlyBudget"] or 10_000), key="ai_amount")
        ai_freq = f3.selectbox(t("Frequency"), ["Monthly", "Weekly"], format_func=t, key=f"ai_freq_{st.session_state.get('lang')}")
        ai_start = f4.date_input(t("First run"), value=date.today(), min_value=date.today(), key="ai_start")
        if st.form_submit_button(t("➕ Add plan"), type="primary"):
            result = add_auto_invest(portfolio, ai_company, float(ai_amount), ai_freq, ai_start)
            st.session_state["pf_msg"] = (result.ok, result.message)
            st.rerun()
    if msg := st.session_state.pop("pf_msg", None):
        (st.success if msg[0] else st.error)(msg[1])
    for plan in portfolio["autoInvest"]:
        p1, p2, p3 = st.columns([6, 1, 1])
        state = t("Active") if plan["active"] else t("Paused")
        p1.markdown(
            f'**#{plan["id"]} {plan["company"]}** · KES {plan["amount"]:,.0f} {t(plan["frequency"]).lower()} · '
            f'{t("next run")} {plan["nextRun"]} · {t("last run")} {plan["lastRun"] or "—"} · '
            f'<span class="st-tag st-{"FILLED" if plan["active"] else "CANCELLED"}">{state}</span>',
            unsafe_allow_html=True,
        )
        if p2.button(t("Resume") if not plan["active"] else t("Pause"), key=f"ai_toggle_{plan['id']}"):
            set_auto_invest_active(portfolio, plan["id"], not plan["active"])
            st.rerun()
        if p3.button(t("Delete"), key=f"ai_del_{plan['id']}"):
            remove_auto_invest(portfolio, plan["id"])
            st.rerun()

    # --- Dividend income ------------------------------------------------------------------------
    st.markdown(t("#### 💰 Dividend income"))
    events = load_corporate_actions()
    today_iso = date.today().isoformat()
    upcoming = []
    for ev in events:
        if ev.get("dps") and ev["paymentDate"] > today_iso:
            # Before book closure, today's holding counts; after it, only what was held on that date.
            if ev["bookClosure"] >= today_iso:
                held = next((h["shares"] for h in portfolio["holdings"] if h["company"] == ev["company"]), 0)
            else:
                held = shares_held_on(portfolio, ev["company"], ev["bookClosure"])
            if held:
                upcoming.append({
                    "Company": ev["company"], "Type": ev["type"], "DPS (KES)": ev["dps"],
                    "Book closure": ev["bookClosure"], "Payment": ev["paymentDate"], "Qualifying shares": held,
                    "Expected net (KES)": round(held * ev["dps"] * (1 - DIVIDEND_WHT), 2),
                })
    received = [t for t in portfolio["trades"] if t["action"] == "DIVIDEND"]
    d1, d2 = st.columns(2)
    d1.metric(t("Dividends received (net)"), f"KES {dividends:,.2f}")
    d2.metric(t("Expected from upcoming payments"), f"KES {sum(u['Expected net (KES)'] for u in upcoming):,.2f}")
    if upcoming:
        st.dataframe(pd.DataFrame(upcoming), use_container_width=True, hide_index=True)
        st.caption(t("You must hold the shares at book closure to qualify. Paid automatically, less 5% withholding tax."))
    else:
        st.caption(t("No upcoming dividends on your holdings — see the 🗓️ Calendar tab for dividend-paying companies."))
    if received:
        st.dataframe(pd.DataFrame(received)[["timestamp", "company", "shares", "price", "gross", "fees", "total", "note"]]
                     .rename(columns={"price": "dps", "fees": "tax"}), use_container_width=True, hide_index=True)

    # --- Cash management ----------------------------------------------------------------------
    st.markdown(t("#### 🏦 Cash management"))
    st.caption(t("Simulates moving cash in or out of your virtual brokerage account."))
    dep_col, wd_col = st.columns(2)
    with dep_col:
        dep_amount = st.number_input(t("Deposit amount (KES)"), min_value=0.0, step=1000.0, key="deposit_amount")
        if st.button(t("⬆️ Deposit"), key="deposit_btn"):
            result = deposit(portfolio, float(dep_amount))
            (st.success if result.ok else st.error)(result.message)
            if result.ok:
                st.rerun()
    with wd_col:
        wd_amount = st.number_input(t("Withdraw amount (KES)"), min_value=0.0, step=1000.0, key="withdraw_amount")
        if st.button(t("⬇️ Withdraw"), key="withdraw_btn"):
            result = withdraw(portfolio, float(wd_amount))
            (st.success if result.ok else st.error)(result.message)
            if result.ok:
                st.rerun()

    # --- Holdings & history -----------------------------------------------------------------------
    st.markdown(t("#### 📋 Holdings"))
    holdings_rows = []
    for h in portfolio["holdings"]:
        cur_price = float(price_by_company.get(h["company"], h["avgCost"]))
        market_value = h["shares"] * cur_price
        cost_basis = h["shares"] * h["avgCost"]
        unrealized = market_value - cost_basis
        holdings_rows.append(
            {
                "Company": h["company"],
                "Sector": sector_of.get(h["company"], ""),
                "Shares": h["shares"],
                "Avg Cost incl. fees (KES)": round(h["avgCost"], 2),
                "Current Price (KES)": round(cur_price, 2),
                "Market Value (KES)": round(market_value, 2),
                "Unrealized P&L (KES)": round(unrealized, 2),
                "Unrealized P&L %": round(unrealized / cost_basis * 100, 2) if cost_basis else 0.0,
            }
        )
    if holdings_rows:
        st.dataframe(pd.DataFrame(holdings_rows).set_index("Company"), use_container_width=True)
    else:
        st.caption(t("No holdings yet — place a buy order in the 📈 Trade tab to get started."))

    st.markdown(t("#### 🧾 Transaction history"))
    if portfolio["trades"]:
        st.dataframe(pd.DataFrame(list(reversed(portfolio["trades"]))), use_container_width=True, hide_index=True)
    else:
        st.caption(t("No transactions yet."))

    # --- Statements ----------------------------------------------------------------------------
    st.markdown(t("#### 📄 Statements"))
    months = sorted({t["timestamp"][:7] for t in portfolio["trades"]}, reverse=True)
    s1, s2, s3, s4 = st.columns(4)
    period = s1.selectbox(t("Statement period"), ["All time", *months], format_func=t, key=f"stmt_period_{st.session_state.get('lang')}",
                          label_visibility="collapsed")
    with s2:
        st.download_button(t("📄 Statement (PDF)"),
            data=statement_pdf(portfolio, profile["fullName"], price_by_company, None if period == "All time" else period),
            file_name=f"statement_{period.replace(' ', '_').lower()}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with s3:
        st.download_button(t("⬇️ Holdings (CSV)"),
            data=pd.DataFrame(holdings_rows).to_csv(index=False) if holdings_rows else "",
            file_name="holdings_statement.csv",
            mime="text/csv",
            disabled=not holdings_rows,
            use_container_width=True,
        )
    with s4:
        st.download_button(t("⬇️ Transactions (CSV)"),
            data=pd.DataFrame(list(reversed(portfolio["trades"]))).to_csv(index=False) if portfolio["trades"] else "",
            file_name="trade_history_statement.csv",
            mime="text/csv",
            disabled=not portfolio["trades"],
            use_container_width=True,
        )
    st.caption(t("Contract notes for individual trades are in the 📈 Trade tab → Trade history."))

    with st.expander(t("⚠️ Reset portfolio")):
        st.caption(t("Wipes cash, holdings, orders, auto-invest plans and history back to the starting balance. "
                   "Cannot be undone."))
        if st.button(t("Reset portfolio to starting cash"), key="reset_portfolio_btn"):
            reset_portfolio()
            st.success(t("Portfolio reset."))
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

    st.subheader(t("⭐ Watchlist & price alerts"))
    st.caption(t("Track companies you're interested in and set a target price alert — above or below "
        "a threshold. Checked live against the current Market Price.")
    )
    from data.notify import email_configured
    from data.profile import load_profile

    prof = load_profile()
    if prof["emailAlerts"] and prof["email"] and email_configured():
        st.caption(f"📧 Triggered alerts are emailed to **{prof['email']}** (once per trigger).")
    elif prof["emailAlerts"]:
        st.caption(t("📧 Email alerts are on, but need an email in your profile and SMTP settings in `.env` "
                   "(see `.env.example`)."))
    else:
        st.caption(t("📧 Turn on **Email me price alerts** in the 👤 Profile tab to get alerts by email."))

    for hit in check_alerts(wl, price_by_company):
        arrow = "risen above" if hit["direction"] == "above" else "fallen below"
        st.warning(
            f"🔔 **{hit['company']}** has {arrow} your target of KES {hit['target']:,.2f} — "
            f"current price KES {hit['current']:,.2f}.",
            icon="🔔",
        )

    available = [c for c in df["Company"] if c not in wl["watching"]]
    if available:
        add_col1, add_col2 = st.columns([3, 1])
        new_company = add_col1.selectbox(t("Add company to watchlist"), available, key="watch_add")
        add_col2.write("")
        add_col2.write("")
        if add_col2.button(t("➕ Add")):
            add_company(wl, new_company)
            st.rerun()
    else:
        st.caption(t("All companies are already on your watchlist."))

    st.divider()

    if not wl["watching"]:
        st.info(t("Your watchlist is empty — add a company above to get started."))
        return

    alerts_by_company = {a["company"]: a for a in wl["alerts"]}
    for company in wl["watching"]:
        price = price_by_company.get(company)
        label = f"{company} — KES {price:,.2f}" if price is not None else company
        with st.expander(label):
            existing = alerts_by_company.get(company)
            c1, c2, c3 = st.columns([2, 2, 1])
            direction = c1.selectbox(t("Alert when price is"),
                ["above", "below"],
                index=0 if not existing or existing["direction"] == "above" else 1,
                format_func=t, key=f"dir_{company}_{st.session_state.get('lang')}",
            )
            target = c2.number_input(t("Target price (KES)"),
                min_value=0.0,
                value=float(existing["target"]) if existing else float(price or 0.0),
                step=0.5,
                key=f"target_{company}",
            )
            with c3:
                st.write("")
                st.write("")
                if st.button(t("💾 Save alert"), key=f"save_{company}"):
                    set_alert(wl, company, target, direction)
                    st.rerun()
            b1, b2 = st.columns(2)
            if existing and b1.button(t("🗑️ Remove alert"), key=f"rm_alert_{company}"):
                remove_alert(wl, company)
                st.rerun()
            if b2.button(t("❌ Remove from watchlist"), key=f"rm_watch_{company}"):
                remove_company(wl, company)
                st.rerun()


def load_news() -> list[dict]:
    path = Path(__file__).parent / "data" / "news.json"
    return json.loads(path.read_text())["items"]


AI_NOTES_PATH = Path(__file__).parent / "data" / "ai_notes.json"


def load_ai_notes() -> list[dict]:
    return json.loads(AI_NOTES_PATH.read_text()) if AI_NOTES_PATH.exists() else []


def generate_ai_note(df: pd.DataFrame, subject: str) -> str:
    """Ask Groq for a market brief ("Daily market brief") or a single-company research note."""
    from groq import Groq

    if subject == "Daily market brief":
        prev = df[PREV_PRICE_COL].where(df[PREV_PRICE_COL] > 0)
        data = df.assign(**{"Change %": ((df[PRICE_COL] / prev - 1) * 100).round(2)})
        context = data[["Company", "Sector", PRICE_COL, "Change %", "Avg Return %"]].to_csv(index=False)
        task = ("Write a concise daily NSE market brief (about 200 words): overall tone, top gainers and losers, "
                "sector themes, and 2-3 things to watch. Use markdown headings and bullets.")
    else:
        row = df[df["Company"] == subject]
        context = row.to_csv(index=False)
        task = (f"Write a short equity research note on {subject} (about 250 words) with sections: Snapshot, "
                "Performance (use the FY returns), Valuation context, Key risks, Bottom line. Use markdown.")
    client = Groq(api_key=GROQ_API_KEY)
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.4,
        max_tokens=900,
        messages=[
            {"role": "system", "content": "You are a sell-side equity analyst covering the Nairobi Securities "
             "Exchange. Base figures only on the data provided; it is illustrative demo data. End with one line: "
             "'Illustrative AI-generated note — not financial advice.'"},
            {"role": "user", "content": f"{task}\n\nData (CSV):\n{context}"},
        ],
    )
    return completion.choices[0].message.content


def render_news(df: pd.DataFrame) -> None:
    st.subheader(t("📰 Research & market news"))

    # --- AI research desk ---------------------------------------------------------------
    st.markdown(t("#### 🤖 AI research desk"))
    notes = load_ai_notes()
    if not GROQ_API_KEY:
        st.info(t("Set `GROQ_API_KEY` in `.env` to generate AI market briefs and company research notes."), icon="🔑")
    else:
        g1, g2 = st.columns([3, 1])
        subject = g1.selectbox(t("Generate"), ["Daily market brief", *sorted(df["Company"])], format_func=t, key=f"ai_subject_{st.session_state.get('lang')}",
                               label_visibility="collapsed")
        if g2.button(t("✨ Generate note"), type="primary", use_container_width=True):
            with st.spinner(t("Drafting the note…")):
                try:
                    body = generate_ai_note(df, subject)
                    notes.insert(0, {"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "subject": subject, "body": body})
                    AI_NOTES_PATH.write_text(json.dumps(notes[:30], indent=2))
                except Exception as exc:  # noqa: BLE001 - surface API/config errors
                    st.error(f"The AI research desk hit an error: `{exc}`")
    for i, note in enumerate(notes[:10]):
        with st.expander(f"{note['subject']} · {note['date']}", expanded=i == 0):
            st.markdown(note["body"])
    if notes and st.button(t("Clear AI notes"), key="ai_notes_clear"):
        AI_NOTES_PATH.unlink(missing_ok=True)
        st.rerun()

    # --- Curated sample feed -----------------------------------------------------------------
    st.markdown(t("#### 🗞️ Market news"))
    st.caption(t("Curated sample research notes and market commentary for this demo — not a live news "
        "feed, and not financial advice.")
    )
    items = load_news()
    sectors = sorted({item["sector"] for item in items})
    selected = st.multiselect(t("Filter by sector"), sectors, default=sectors, key="news_sector_filter")
    filtered = sorted(
        (item for item in items if item["sector"] in selected),
        key=lambda x: x["date"],
        reverse=True,
    )
    if not filtered:
        st.info(t("No news items match the selected sectors."))
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
    st.subheader(t("🏛️ Bonds, T-Bills & Funds"))
    st.caption(t("Illustrative sample rates for fixed-income and pooled-fund instruments available in "
        "Kenya — not live rates, not financial advice.")
    )

    st.markdown(t("**Government securities (Treasury Bills & Bonds)**"))
    st.dataframe(pd.DataFrame(data["governmentSecurities"]), use_container_width=True, hide_index=True)

    st.markdown(t("**Corporate bonds**"))
    st.dataframe(pd.DataFrame(data["corporateBonds"]), use_container_width=True, hide_index=True)

    st.markdown(t("**Money market funds**"))
    st.dataframe(pd.DataFrame(data["moneyMarketFunds"]), use_container_width=True, hide_index=True)

    st.markdown(t("**Unit trusts / collective investment schemes**"))
    st.dataframe(pd.DataFrame(data["unitTrusts"]), use_container_width=True, hide_index=True)


def load_ipo_calendar() -> list[dict]:
    path = Path(__file__).parent / "data" / "ipo_calendar.json"
    return json.loads(path.read_text())["events"]


def load_corporate_actions() -> list[dict]:
    path = Path(__file__).parent / "data" / "corporate_actions.json"
    return json.loads(path.read_text())["events"]


def render_calendar(df: pd.DataFrame) -> None:
    from data.portfolio import load_portfolio

    st.subheader(t("🗓️ Dividends & corporate actions"))
    st.caption(t("Illustrative sample calendar — not official announcements. Hold shares at **book closure** to qualify; "
        "cash dividends are paid into your paper portfolio on the **payment date**, less 5% withholding tax.")
    )
    events = load_corporate_actions()
    held = {h["company"] for h in load_portfolio()["holdings"]}
    today_iso = date.today().isoformat()
    price_by_company = dict(zip(df["Company"], df[PRICE_COL]))

    upcoming = sorted((e for e in events if e["bookClosure"] >= today_iso), key=lambda e: e["bookClosure"])
    cards = ""
    for e in upcoming[:4]:
        px = price_by_company.get(e["company"])
        yield_txt = f"{e['dps'] / px * 100:.1f}% yield" if e.get("dps") and px else e["notes"]
        amount = f"KES {e['dps']:,.2f}/share" if e.get("dps") else e["type"]
        cards += (
            f'<div class="mk-card"><div class="hd"><span>{html.escape(e["type"])}</span>'
            f'<span style="color:var(--ki-muted)">closes {e["bookClosure"]}</span></div>'
            f'<div style="font-weight:600">{html.escape(e["company"])}{" ⭐" if e["company"] in held else ""}</div>'
            f'<div class="up" style="font-size:20px">{html.escape(amount)}</div>'
            f'<div style="color:var(--ki-muted);font-size:12px">{html.escape(yield_txt)} · pays {e["paymentDate"]}</div></div>'
        )
    if cards:
        st.markdown(f'<div class="mk-cards">{cards}</div>', unsafe_allow_html=True)

    types = sorted({e["type"] for e in events})
    f1, f2 = st.columns([3, 1])
    chosen = f1.multiselect(t("Event types"), types, default=types, key="ca_types", label_visibility="collapsed")
    mine = f2.toggle(t("Only my holdings"), key="ca_mine")
    rows = []
    for e in sorted(events, key=lambda e: e["bookClosure"], reverse=True):
        if e["type"] not in chosen or (mine and e["company"] not in held):
            continue
        px = price_by_company.get(e["company"])
        status = "Paid" if e["paymentDate"] <= today_iso else "Book closed" if e["bookClosure"] < today_iso else "Upcoming"
        rows.append({
            "Status": status,
            "Company": e["company"],
            "Event": e["type"],
            "DPS (KES)": e.get("dps"),
            "Yield %": round(e["dps"] / px * 100, 2) if e.get("dps") and px else None,
            "Announced": e["announced"],
            "Book closure": e["bookClosure"],
            "Payment / effective": e["paymentDate"],
            "You hold": "⭐" if e["company"] in held else "",
            "Notes": e["notes"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader(t("🚀 IPO & Rights Issue calendar"))
    st.caption(t("Illustrative sample calendar of primary-market events — not a live feed, not financial advice."))
    ipo = load_ipo_calendar()
    status_order = {"Ongoing": 0, "Upcoming": 1, "Closed": 2}
    ipo = sorted(ipo, key=lambda e: (status_order.get(e["Status"], 9), e["openDate"]))
    st.dataframe(pd.DataFrame(ipo), use_container_width=True, hide_index=True)


def render_price_updates(df: pd.DataFrame) -> None:
    st.subheader(t("💹 Update current market price per share"))
    st.caption(t("Edit the **Market Price (KES/share)** column below to reflect a new price, then click "
        "**Save price updates**. Changes are written to `data/companies.json` and the previous "
        "price + update date are recorded automatically.")
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

    if st.button(t("💾 Save price updates"), type="primary"):
        changed = save_price_updates(edited)
        if changed:
            st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1
            st.success(t("Prices updated and saved."))
            st.rerun()
        else:
            st.info(t("No price changes detected."))


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

    st.subheader(t("👤 My investor profile"))
    st.caption(t("Keep your details and investment preferences up to date — you can edit and save this "
        "any time. Stored locally in `data/profile.json`.")
    )

    if profile["lastUpdated"]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(t("Name"), profile["fullName"] or "—")
        c2.metric(t("Risk tolerance"), profile["riskTolerance"])
        c3.metric(t("Monthly budget (KES)"), f"{profile['monthlyBudget']:,.0f}")
        c4.metric(t("Last updated"), profile["lastUpdated"].replace("T", " "))
    else:
        st.info(t("You haven't set up a profile yet — fill in the form below and save."))

    sectors = sorted(df["Sector"].unique())
    with st.form("profile_form"):
        st.markdown(t("**Personal details**"))
        p1, p2 = st.columns(2)
        full_name = p1.text_input(t("Full name *"), value=profile["fullName"])
        email = p2.text_input(t("Email"), value=profile["email"])
        phone = p1.text_input(t("Phone"), value=profile["phone"], placeholder="+254 7xx xxx xxx")
        location = p2.text_input(t("County / town"), value=profile["location"])

        st.markdown(t("**Investment preferences**"))
        q1, q2, q3 = st.columns(3)
        experience = q1.selectbox(t("Experience level"), EXPERIENCE_LEVELS, index=EXPERIENCE_LEVELS.index(profile["experience"]), format_func=t
        )
        risk = q2.selectbox(t("Risk tolerance"), RISK_LEVELS, index=RISK_LEVELS.index(profile["riskTolerance"]), format_func=t)
        horizon = q3.selectbox(t("Investment horizon"), HORIZONS, index=HORIZONS.index(profile["horizon"]), format_func=t)
        goals = st.multiselect(t("Investment goals"), GOALS, default=[g for g in profile["goals"] if g in GOALS], format_func=t)
        preferred = st.multiselect(t("Preferred sectors"),
            sectors,
            default=[s for s in profile["preferredSectors"] if s in sectors],
        )
        budget = st.number_input(t("Monthly investment budget (KES)"),
            min_value=0.0,
            value=float(profile["monthlyBudget"]),
            step=1000.0,
        )
        bio = st.text_area(t("About me / notes"), value=profile["bio"], max_chars=500)
        email_alerts = st.checkbox(t("📧 Email me price alerts"), value=profile["emailAlerts"],
                                   help="Sends watchlist alerts to the email above when they trigger.")

        if st.form_submit_button(t("💾 Save profile"), type="primary"):
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
                    "emailAlerts": email_alerts,
                },
            )
            (st.success if result.ok else st.error)(result.message)
            if result.ok:
                st.rerun()

    if profile["preferredSectors"]:
        st.markdown(t("**Companies in your preferred sectors**"))
        matches = df[df["Sector"].isin(profile["preferredSectors"])]
        st.dataframe(
            matches.set_index("Company")[["Sector", PRICE_COL, "Avg Return %"]],
            use_container_width=True,
        )

    with st.expander(t("⚠️ Clear profile")):
        st.caption(t("Deletes all saved profile details. Cannot be undone."))
        if st.button(t("Clear my profile"), key="reset_profile_btn"):
            reset_profile()
            st.success(t("Profile cleared."))
            st.rerun()


def render_asset_classes() -> None:
    st.subheader(t("🎓 Asset classes explained"))
    st.caption(t("A quick primer on the main asset classes available to investors in Kenya, and how each one works."))
    for a in ASSET_CLASSES:
        with st.expander(f"{a['icon']} {a['name']}"):
            st.markdown(a["how"])


CHAT_PATH = Path(__file__).parent / "data" / "chat_history.json"
CHAT_CONTEXT_MESSAGES = 20  # how many past messages are sent to the AI with each question


def load_chat_history() -> list[dict]:
    return json.loads(CHAT_PATH.read_text()) if CHAT_PATH.exists() else []


def save_chat_history(history: list[dict]) -> None:
    CHAT_PATH.write_text(json.dumps(history, indent=2))


def render_chat(df: pd.DataFrame) -> None:
    st.subheader(t("🤖 Ask Groq AI about these opportunities"))

    # Chat history is saved to data/chat_history.json so it survives reloads and restarts.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    history = st.session_state.chat_history

    if history:
        h1, h2, h3 = st.columns([4, 1, 1])
        h1.caption(t("Your conversation is saved automatically.") + f" · {len(history)} " + t("messages"))
        h2.download_button(
            t("⬇️ Download chat"),
            data="\n\n".join(f"[{m.get('time', '')}] {m['role'].upper()}: {m['content']}" for m in history),
            file_name="ai_chat_history.txt",
            mime="text/plain",
            use_container_width=True,
        )
        if h3.button(t("🗑️ Clear chat"), key="chat_clear", use_container_width=True):
            st.session_state.chat_history = []
            CHAT_PATH.unlink(missing_ok=True)
            st.rerun()

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("time"):
                st.caption(msg["time"])

    if not GROQ_API_KEY:
        st.info(t("Set `GROQ_API_KEY` in a `.env` file (see `.env.example`) to enable the AI assistant. "
            "The rest of the app works without it."),
            icon="🔑",
        )
        return

    question = st.chat_input(t("e.g. Which sectors had the best average returns?"))
    if question:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        history.append({"role": "user", "content": question, "time": now})
        save_chat_history(history)
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                past = [{"role": m["role"], "content": m["content"]} for m in history[:-1][-CHAT_CONTEXT_MESSAGES:]]
                answer = ask_groq(question, build_ai_context(df), past)
            except Exception as exc:  # noqa: BLE001 - surface any API/config error to the user
                answer = f"Sorry, the AI assistant hit an error: `{exc}`"
            st.markdown(answer)

        history.append({"role": "assistant", "content": answer, "time": datetime.now().strftime("%Y-%m-%d %H:%M")})
        save_chat_history(history)


def main() -> None:
    st.set_page_config(page_title="Kenya Investment Explorer", page_icon="📈", layout="wide")
    hide_default_chrome()

    from data.profile import load_profile as _load_profile

    if "lang" not in st.session_state:
        st.session_state["lang"] = _load_profile().get("language", "en")
    global THEME
    theme_type = getattr(st.context.theme, "type", None) or "dark"
    THEME = THEMES["light" if theme_type == "light" else "dark"]
    st.markdown("<style>:root {" + ";".join(f"{k}:{v}" for k, v in THEME.items()) + "}</style>",
                unsafe_allow_html=True)

    version = st.session_state.get("data_version", 0)
    df = load_data(version)
    raw = load_raw()

    from data.portfolio import load_portfolio, process_open_orders

    from data.portfolio import credit_dividends, run_auto_invest

    prices = dict(zip(df["Company"], df[PRICE_COL]))
    portfolio = load_portfolio()
    jobs = (
        process_open_orders(portfolio, prices)
        + run_auto_invest(portfolio, prices)
        + credit_dividends(portfolio, load_corporate_actions())
    )
    for result in jobs:
        st.toast(result.message, icon="✅" if result.ok else "⚠️")

    # Email any newly triggered watchlist alerts (once per trigger).
    from data.notify import email_alerts, email_configured
    from data.profile import load_profile
    from data.watchlist import check_alerts, load_watchlist, save_watchlist

    prof = load_profile()
    if prof["emailAlerts"] and prof["email"] and email_configured():
        wl = load_watchlist()
        for message in email_alerts(wl, check_alerts(wl, prices), prof["email"]):
            st.toast(message, icon="📧")
        save_watchlist(wl)

    render_hero(raw)
    render_sidebar(df)

    (
        tab_overview,
        tab_trade_desk,
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
            t("🏠 Overview"),
            t(TRADE_TAB),
            t("📊 Companies & Sectors"),
            t("⭐ Watchlist"),
            t("💼 Portfolio"),
            t("🏛️ Bonds & Funds"),
            t("🗓️ Calendar"),
            t("📰 Research & News"),
            t("💹 Update Market Prices"),
            t("🎓 Asset Classes"),
            t("🤖 Ask AI"),
            t("👤 Profile"),
        ],
        default=t(TRADE_TAB) if "trade" in st.query_params else None,
    )

    with tab_overview:
        render_overview(df, raw)

    with tab_trade_desk:
        render_trade_desk(df, raw)

    with tab_companies:
        filtered = render_companies(df)

    with tab_watchlist:
        render_watchlist(df)

    with tab_trade:
        render_trading(df, raw)

    with tab_bonds:
        render_bonds_funds()

    with tab_ipo:
        render_calendar(df)

    with tab_news:
        render_news(df)

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
