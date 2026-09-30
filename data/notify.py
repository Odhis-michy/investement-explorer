"""Email notifications (price alerts) over SMTP.

Configure in .env (or Streamlit secrets):
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=you@gmail.com
    SMTP_PASSWORD=your-app-password   # for Gmail, create an App Password
    ALERT_FROM=you@gmail.com          # optional, defaults to SMTP_USER
"""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage


def email_configured() -> bool:
    return all(os.environ.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"))


def send_email(to: str, subject: str, body: str) -> tuple[bool, str]:
    if not email_configured():
        return False, "Email isn't set up — add SMTP settings to .env."
    msg = EmailMessage()
    msg["From"] = os.environ.get("ALERT_FROM") or os.environ["SMTP_USER"]
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", 587)), timeout=15) as smtp:
            smtp.starttls()
            smtp.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
            smtp.send_message(msg)
    except Exception as exc:  # noqa: BLE001 - report any SMTP/network failure to the UI
        return False, f"Couldn't send email: {exc}"
    return True, f"Email sent to {to}."


def email_alerts(wl: dict, triggered: list[dict], to: str) -> list[str]:
    """Email each newly triggered alert once; re-arm alerts that are no longer triggered.

    Mutates wl["alerts"][*]["notified"]; the caller saves the watchlist. Returns status messages.
    """
    hit = {t["company"] for t in triggered}
    messages = []
    for alert in wl["alerts"]:
        if alert["company"] not in hit:
            alert["notified"] = False
            continue
        if alert.get("notified"):
            continue
        t = next(x for x in triggered if x["company"] == alert["company"])
        verb = "risen above" if t["direction"] == "above" else "fallen below"
        ok, msg = send_email(
            to,
            f"Price alert: {t['company']} {t['direction']} KES {t['target']:,.2f}",
            f"{t['company']} has {verb} your target of KES {t['target']:,.2f}.\n"
            f"Current price: KES {t['current']:,.2f}.\n\n— Kenya Invest (simulated trading app, not financial advice)",
        )
        if ok:
            alert["notified"] = True
        messages.append(msg)
    return messages
