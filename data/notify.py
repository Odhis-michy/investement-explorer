"""Price-alert notifications by email (SMTP) and SMS (Africa's Talking or Twilio).

Configure in .env locally, or in the app's Secrets on Streamlit Community Cloud:

    # Email
    SMTP_HOST=smtp.gmail.com
    SMTP_PORT=587
    SMTP_USER=you@gmail.com
    SMTP_PASSWORD=your-app-password   # for Gmail, create an App Password
    ALERT_FROM=you@gmail.com          # optional, defaults to SMTP_USER

    # SMS via Africa's Talking (preferred for Kenyan numbers)
    AT_USERNAME=sandbox               # your AT app username; "sandbox" for testing
    AT_API_KEY=...
    AT_SENDER_ID=                     # optional registered sender ID / shortcode

    # ...or SMS via Twilio
    TWILIO_ACCOUNT_SID=...
    TWILIO_AUTH_TOKEN=...
    TWILIO_FROM=+1...
"""

from __future__ import annotations

import os
import re
import smtplib
from email.message import EmailMessage

import requests

DISCLAIMER = "Kenya Invest (simulated trading app, not financial advice)"


# --- Email --------------------------------------------------------------------------------


def email_configured() -> bool:
    return all(os.environ.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"))


def send_email(to: str, subject: str, body: str) -> tuple[bool, str]:
    if not email_configured():
        return False, "Email isn't set up — add SMTP settings to .env or the app's Secrets."
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


# --- SMS -----------------------------------------------------------------------------------


def normalize_phone(phone: str) -> str | None:
    """Return an E.164 number (+2547XXXXXXXX) for Kenyan formats, or pass through other +country numbers."""
    digits = re.sub(r"[^\d+]", "", phone or "")
    if re.fullmatch(r"0[17]\d{8}", digits):
        return "+254" + digits[1:]
    if re.fullmatch(r"254[17]\d{8}", digits):
        return "+" + digits
    if re.fullmatch(r"\+\d{9,15}", digits):
        return digits
    return None


def sms_provider() -> str | None:
    if os.environ.get("AT_USERNAME") and os.environ.get("AT_API_KEY"):
        return "africastalking"
    if all(os.environ.get(k) for k in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM")):
        return "twilio"
    return None


def sms_configured() -> bool:
    return sms_provider() is not None


def send_sms(to: str, body: str) -> tuple[bool, str]:
    number = normalize_phone(to)
    if not number:
        return False, f"'{to}' isn't a valid phone number — use 07XX XXX XXX or +2547XX XXX XXX."
    provider = sms_provider()
    try:
        if provider == "africastalking":
            username = os.environ["AT_USERNAME"]
            host = "api.sandbox.africastalking.com" if username == "sandbox" else "api.africastalking.com"
            data = {"username": username, "to": number, "message": body}
            if os.environ.get("AT_SENDER_ID"):
                data["from"] = os.environ["AT_SENDER_ID"]
            resp = requests.post(
                f"https://{host}/version1/messaging",
                headers={"apiKey": os.environ["AT_API_KEY"], "Accept": "application/json"},
                data=data,
                timeout=15,
            )
            resp.raise_for_status()
            recipients = resp.json().get("SMSMessageData", {}).get("Recipients", [])
            if not recipients or recipients[0].get("status") != "Success":
                detail = recipients[0].get("status") if recipients else resp.json().get("SMSMessageData", {}).get("Message")
                return False, f"SMS not sent: {detail}"
        elif provider == "twilio":
            sid = os.environ["TWILIO_ACCOUNT_SID"]
            resp = requests.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
                auth=(sid, os.environ["TWILIO_AUTH_TOKEN"]),
                data={"To": number, "From": os.environ["TWILIO_FROM"], "Body": body},
                timeout=15,
            )
            resp.raise_for_status()
        else:
            return False, "SMS isn't set up — add Africa's Talking or Twilio keys to .env or the app's Secrets."
    except Exception as exc:  # noqa: BLE001 - report any provider/network failure to the UI
        return False, f"Couldn't send SMS: {exc}"
    return True, f"SMS sent to {number}."


# --- Price alerts ------------------------------------------------------------------------------


def alert_targets(prof: dict) -> tuple[str | None, str | None]:
    """(email, phone) to notify, only for channels the user turned on and the app has credentials for."""
    email_to = prof["email"] if prof.get("emailAlerts") and prof.get("email") and email_configured() else None
    phone = normalize_phone(prof.get("phone", "")) if prof.get("smsAlerts") else None
    sms_to = phone if phone and sms_configured() else None
    return email_to, sms_to



def _alert_texts(t: dict) -> tuple[str, str, str]:
    verb = "risen above" if t["direction"] == "above" else "fallen below"
    subject = f"Price alert: {t['company']} {t['direction']} KES {t['target']:,.2f}"
    body = (f"{t['company']} has {verb} your target of KES {t['target']:,.2f}.\n"
            f"Current price: KES {t['current']:,.2f}.\n\n— {DISCLAIMER}")
    sms = f"Kenya Invest alert: {t['company']} is KES {t['current']:,.2f}, {verb} your KES {t['target']:,.2f} target."
    return subject, body, sms


def send_alerts(wl: dict, triggered: list[dict], email_to: str | None, sms_to: str | None) -> list[str]:
    """Notify each newly triggered alert once per channel; re-arm alerts that are no longer triggered.

    Mutates wl["alerts"][*]["notified"] (email) and ["smsNotified"]; the caller saves the watchlist.
    Returns status messages.
    """
    hit = {t["company"]: t for t in triggered}
    messages = []
    for alert in wl["alerts"]:
        t = hit.get(alert["company"])
        if t is None:
            alert["notified"] = False
            alert["smsNotified"] = False
            continue
        subject, body, sms = _alert_texts(t)
        if email_to and not alert.get("notified"):
            ok, msg = send_email(email_to, subject, body)
            alert["notified"] = ok
            messages.append(msg)
        if sms_to and not alert.get("smsNotified"):
            ok, msg = send_sms(sms_to, sms)
            alert["smsNotified"] = ok
            messages.append(msg)
    return messages


def send_test_alert(email_to: str | None, sms_to: str | None) -> list[tuple[bool, str]]:
    sample = {"company": "Safaricom PLC", "direction": "above", "target": 35.0, "current": 37.85}
    subject, body, sms = _alert_texts(sample)
    results = []
    if email_to:
        results.append(send_email(email_to, "[Test] " + subject, "This is a test alert.\n\n" + body))
    if sms_to:
        results.append(send_sms(sms_to, "[Test] " + sms))
    return results
