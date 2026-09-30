# Kenya Investment Explorer

A Streamlit app for exploring investment opportunities across every major sector of the Kenyan
economy — listed companies, current market price per share, and returns per financial year —
with a [Groq](https://groq.com) AI chatbot you can ask natural-language questions about the data,
plus a primer on how each major asset class works.

> ⚠️ **Sample data disclaimer**: The dataset bundled with this app (NSE-listed Kenyan companies,
> market prices, and per-financial-year returns) is illustrative/placeholder data for
> demonstration purposes. It is **not** live market data and nothing in this app is financial
> advice. Swap in a real data source before using it to make investment decisions.

## Features

- **Overview** — market snapshot (companies tracked, sectors covered, total market cap), plus
  charts of companies per sector and average return per sector.
- **Companies & Sectors** — table of ~30 companies spanning Banking, Telecommunication &
  Technology, Manufacturing & Allied, Construction & Allied, Commercial & Services, Insurance,
  Energy & Petroleum, Investment, Real Estate (REIT), Agricultural, Automobiles & Accessories,
  and Media, with sidebar filters (sector, max price, minimum average return) and a return chart.
- **Update Market Prices** — an editable table to change any company's current market price per
  share; saving persists the change (plus the previous price and update date) to
  `data/companies.json` so the new price sticks across restarts.
- **Asset Classes** — an expandable primer on how each major asset class available to Kenyan
  investors works (equities, government/corporate bonds, money market funds, unit trusts, REITs,
  direct real estate, fixed deposits, SACCOs, pension funds, commodities, derivatives).
- **Ask AI** — a Groq-powered chatbot that answers questions about the (filtered) dataset.
- **Trade** — an exchange-style trading view for every sector: pick a sector and company, see a
  candlestick or line chart (1M–5Y) and place market or limit buy/sell orders with 25–100% quick
  amounts, plus open orders, trade history and holdings. Simulated with virtual cash — the chart
  history is illustrative, reconstructed from yearly returns (`data/price_history.py`).
- **Broker-style extras** (inspired by AIB-AXYS DigiTrader and Binance):
  - NSE trading fees (brokerage, NSE/CMA levies, CDSC, VAT) on every trade, with a breakdown
  - Stop-loss, take-profit, stop-limit and limit orders, day or good-till-cancelled, with order status history
  - Chart indicators (MA, EMA, Bollinger, RSI, MACD, volume), a simulated order book and recent-trades tape
  - Market statistics (volume, turnover, deals, advancers/decliners) and company search
  - Portfolio analysis (sector allocation, value over time, P&L by sector)
  - Auto-invest plans (weekly/monthly) and a dividend & corporate-actions calendar that pays dividends into
    the paper portfolio (`data/corporate_actions.json`)
  - PDF contract notes and portfolio statements, email price alerts (SMTP settings in `.env`) and
    AI-drafted research notes (Groq)
- **Profile** — a personal investor profile (contact details, risk tolerance, horizon, goals,
  preferred sectors, monthly budget) that can be updated any time; saved locally to
  `data/profile.json` (git-ignored).

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your Groq API key (get one at
   [console.groq.com](https://console.groq.com/keys)):

   ```bash
   cp .env.example .env
   ```

   ```
   GROQ_API_KEY=your-key-here
   GROQ_MODEL=openai/gpt-oss-120b
   ```

   The app runs fine without a key — everything except the AI chat tab works without it.

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically http://localhost:8501).

## Project structure

```
app.py               Streamlit app (overview, companies/sectors, price editor, asset classes, Groq chat)
data/companies.json  Company dataset (sector, market price, market cap, FY returns)
requirements.txt     Python dependencies
.env.example         Template for local environment variables (copy to .env)
.gitignore           Excludes .env and local/build artifacts from git
```
