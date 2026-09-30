"""PDF contract notes and portfolio statements for the paper-trading account."""

from __future__ import annotations

from datetime import datetime

from fpdf import FPDF

from data.fees import fee_breakdown

GOLD = (240, 185, 11)
DARK = (30, 35, 41)
MUTED = (110, 118, 132)


def _t(text: object) -> str:
    """Core PDF fonts are Latin-1 only — swap common Unicode punctuation and drop the rest."""
    s = str(text).replace("—", "-").replace("–", "-").replace("·", "|").replace("’", "'")
    return s.encode("latin-1", "replace").decode("latin-1")


class _Doc(FPDF):
    def __init__(self, title: str):
        super().__init__()
        self.doc_title = title
        self.set_auto_page_break(True, margin=18)
        self.add_page()

    def header(self):
        self.set_fill_color(*DARK)
        self.rect(0, 0, 210, 22, "F")
        self.set_xy(12, 7)
        self.set_text_color(*GOLD)
        self.set_font("Helvetica", "B", 15)
        self.cell(0, 8, "KENYA INVEST")
        self.set_xy(-110, 7)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "", 11)
        self.cell(98, 8, _t(self.doc_title), align="R")
        self.set_y(28)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 6, _t("Simulated paper-trading account - not a real brokerage document. "
                           f"Generated {datetime.now():%Y-%m-%d %H:%M}. Page {self.page_no()}"), align="C")

    def kv(self, key: str, value: str, bold: bool = False):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*MUTED)
        self.cell(60, 7, _t(key))
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "B" if bold else "", 10)
        self.cell(0, 7, _t(value), new_x="LMARGIN", new_y="NEXT")

    def section(self, title: str):
        self.ln(3)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, _t(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*GOLD)
        self.line(self.l_margin, self.get_y(), 210 - self.r_margin, self.get_y())
        self.ln(2)

    def table(self, headers: list[str], rows: list[list[str]], widths: list[int]):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(240, 240, 240)
        for h, w in zip(headers, widths):
            self.cell(w, 7, _t(h), border="B", fill=True, align="L" if h == headers[0] else "R")
        self.ln()
        self.set_font("Helvetica", "", 9)
        for row in rows:
            for i, (v, w) in enumerate(zip(row, widths)):
                self.cell(w, 6.5, _t(v), border="B", align="L" if i == 0 else "R")
            self.ln()


def contract_note_pdf(trade: dict, client: str, note_no: int) -> bytes:
    doc = _Doc(f"Contract note #{note_no:05d}")
    doc.kv("Client", client or "Guest investor")
    doc.kv("Trade date/time (UTC)", trade["timestamp"].replace("T", " ")[:19])
    doc.kv("Security", trade["company"])
    doc.kv("Transaction", "Purchase" if trade["action"] == "BUY" else "Sale", bold=True)
    doc.kv("Order", trade.get("note", "Market order"))
    doc.kv("Quantity", f"{trade['shares']:,.0f} shares")
    doc.kv("Price", f"KES {trade['price']:,.2f}")

    gross = trade.get("gross", trade["shares"] * trade["price"])
    doc.section("Charges")
    rows = [[label, f"{amount:,.2f}"] for label, amount in fee_breakdown(gross)]
    doc.table(["Charge", "KES"], [["Consideration", f"{gross:,.2f}"], *rows], [130, 56])
    doc.ln(3)
    total = trade["total"]
    doc.kv("Net amount " + ("payable" if trade["action"] == "BUY" else "receivable"), f"KES {total:,.2f}", bold=True)
    doc.kv("Cash balance after", f"KES {trade['cashAfter']:,.2f}")
    return bytes(doc.output())


def statement_pdf(portfolio: dict, client: str, price_by_company: dict[str, float], month: str | None) -> bytes:
    period = month or "All time"
    doc = _Doc(f"Portfolio statement - {period}")
    holdings_value = sum(h["shares"] * price_by_company.get(h["company"], h["avgCost"]) for h in portfolio["holdings"])
    total = portfolio["cash"] + holdings_value
    baseline = portfolio["startingCash"] + portfolio.get("netDeposits", 0.0)

    doc.kv("Client", client or "Guest investor")
    doc.kv("Statement period", period)
    doc.kv("Cash balance", f"KES {portfolio['cash']:,.2f}")
    doc.kv("Holdings value", f"KES {holdings_value:,.2f}")
    doc.kv("Total portfolio value", f"KES {total:,.2f}", bold=True)
    doc.kv("Total profit / loss", f"KES {total - baseline:,.2f} ({(total / baseline - 1) * 100 if baseline else 0:+.2f}%)")

    doc.section("Holdings")
    rows = []
    for h in portfolio["holdings"]:
        px = price_by_company.get(h["company"], h["avgCost"])
        rows.append([h["company"], f"{h['shares']:,.0f}", f"{h['avgCost']:,.2f}", f"{px:,.2f}",
                     f"{h['shares'] * px:,.2f}", f"{(px / h['avgCost'] - 1) * 100:+.2f}%"])
    if rows:
        doc.table(["Company", "Shares", "Avg cost", "Price", "Value (KES)", "P&L"], rows, [62, 22, 24, 22, 32, 24])
    else:
        doc.kv("No holdings", "")

    doc.section("Transactions")
    trades = [t for t in portfolio["trades"] if not month or t["timestamp"][:7] == month]
    rows = [[f"{t['timestamp'][:10]}  {t['action']}  {t['company']}"[:52], f"{t['shares']:,.0f}",
             f"{t['price']:,.2f}", f"{t.get('fees', 0):,.2f}", f"{t['total']:,.2f}"] for t in trades]
    if rows:
        doc.table(["Date / type / security", "Shares", "Price", "Fees/tax", "Net (KES)"], rows, [96, 20, 22, 22, 26])
    else:
        doc.kv("No transactions in this period", "")
    return bytes(doc.output())
