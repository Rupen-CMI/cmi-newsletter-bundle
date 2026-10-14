"""Send a rendered CMI brief as an email via SMTP.

    python mailer.py output/cmi_signalbrief_immunoact_india_20260810.html you@example.com

Two providers. SendGrid is used automatically when SENDGRID_WEB_API is set, because
raw SMTP from an unauthenticated domain was being rejected as bulk mail (see
send_html); SMTP remains the fallback. Force one with CMI_MAIL_PROVIDER=sendgrid|smtp.

Credentials come from the environment / local .env so they never live in the SQLite
DB or the repo:
    SENDGRID_WEB_API     a SendGrid API key. When present, sending goes over the
                          SendGrid v3 HTTP API and the SMTP_* vars are not needed.
    SMTP_HOST            required, e.g. smtp.gmail.com / smtp.office365.com
    SMTP_PORT            optional, default 587 (STARTTLS)
    SMTP_USER            required — the mailbox that authenticates
    SMTP_PASSWORD        required — an app password, not the account password, for
                          providers that support 2FA app-passwords (Gmail, Outlook)
    SENDER_EMAIL          optional — From address, defaults to SMTP_USER
    SENDER_NAME           optional — From display name, default "Coherent Market Insights"

Non-secret settings (from-name override, reply-to list) can also come from the
SQLite settings table (db.py) so they're editable from the dashboard without
touching .env; env vars still win if set.
"""
from __future__ import annotations

import os
import smtplib
import ssl
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import dotenv_values

from emailify import emailify
import db as dbmod

_here = os.path.dirname(os.path.abspath(__file__))
_local_env = os.path.join(_here, ".env")
_vals = dotenv_values(_local_env) if os.path.exists(_local_env) else {}


def _env(name: str, default: str = "") -> str:
    return os.getenv(name) or _vals.get(name) or default


DEFAULT_REPLY_TO = ["ujjwal@coherentmarketinsights.com", "aditya@coherentmarketinsights.com"]


# SendGrid's max `limit` for these endpoints is 500 -- a value above that is a 400
# Bad Request. Ported from the sibling newsletter-cc-cl project's campaign.py.
_SUPPRESSION_PAGE_LIMIT = 500


def fetch_sendgrid_suppressed(api_key: str) -> set[str]:
    """Every email SendGrid itself considers unsubscribed OR hard-bounced, across
    the account.

    db.mark_unsubscribed() exists but nothing in this project ever called it -- the
    template's unsubscribe link points at a generic contact page, not a real
    unsubscribe endpoint, so clicking it did nothing. This is the actual, working
    signal: SendGrid's own suppression lists, populated automatically whenever a
    recipient unsubscribes via any SendGrid-sent mail, hard-bounces, or lands in the
    global suppression group.

    Three endpoints, not two: `suppression/bounces` turned out NOT to be a subset of
    `asm/suppressions/global` -- checked directly against this account's real data
    (2026-10-07) and found 500+ bounce records with zero overlap against the other
    two endpoints. Skipping it would silently keep re-sending to addresses SendGrid
    has already confirmed are undeliverable.

    Requires an API key with suppressions.read / asm.suppressions.global.read scope
    -- the mail.send-only key already in use here cannot call these endpoints and
    will get a 403.  Degrades to an empty set on any failure (missing scope, no key,
    network error) rather than raising, so a key without this scope just means no
    local pre-check -- SendGrid's own send-time enforcement still blocks a
    suppressed address regardless (see stage_send).
    """
    import json
    import urllib.error
    import urllib.request

    suppressed: set[str] = set()
    if not api_key:
        return suppressed
    for path in ("suppression/unsubscribes", "asm/suppressions/global", "suppression/bounces"):
        offset = 0
        while True:
            req = urllib.request.Request(
                f"https://api.sendgrid.com/v3/{path}?limit={_SUPPRESSION_PAGE_LIMIT}&offset={offset}",
                headers={"Authorization": f"Bearer {api_key}"})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = json.loads(r.read())
            except (urllib.error.HTTPError, urllib.error.URLError, ValueError, TimeoutError,
                    ConnectionError) as e:
                print(f"  [suppression] fetch failed ({path}, offset={offset}): "
                      f"{type(e).__name__}: {e} -- continuing with whatever was fetched so far",
                      file=sys.stderr)
                break

            if not isinstance(data, list) or not data:
                break
            suppressed.update((item.get("email") or "").lower() for item in data if item.get("email"))
            if len(data) < _SUPPRESSION_PAGE_LIMIT:
                break  # last page
            offset += _SUPPRESSION_PAGE_LIMIT
    return suppressed


def _reply_to_list(conn) -> list[str]:
    stored = dbmod.get_setting(conn, "reply_to", "")
    if stored:
        return [a.strip() for a in stored.split(",") if a.strip()]
    return DEFAULT_REPLY_TO


def _smtp_config() -> dict:
    host = _env("SMTP_HOST")
    user = _env("SMTP_USER")
    password = _env("SMTP_PASSWORD")
    if not (host and user and password):
        raise RuntimeError(
            "SMTP is not configured: set SMTP_HOST, SMTP_USER, SMTP_PASSWORD in .env."
        )
    return {
        "host": host,
        "port": int(_env("SMTP_PORT", "587")),
        "user": user,
        "password": password,
        "from_email": _env("SENDER_EMAIL", user),
        "from_name": _env("SENDER_NAME", "Coherent Market Insights"),
    }


def _provider() -> str:
    """Which transport to use. SendGrid wins when its key is present: it is the
    authenticated path, and the SMTP path is what was bouncing."""
    forced = _env("CMI_MAIL_PROVIDER").strip().lower()
    if forced:
        return forced
    return "sendgrid" if _env("SENDGRID_WEB_API") else "smtp"


def _send_via_sendgrid(html: str, to: str, subject: str, reply_to: list[str]) -> str:
    """POST one message to the SendGrid v3 API. Returns the X-Message-Id header
    (empty string if SendGrid did not send one back).

    Uses urllib rather than the sendgrid SDK to avoid another dependency -- the
    payload is small and the call is a single POST.

    The message ID is the only way to later match a reply or a bounce/complaint
    webhook back to a specific send -- the sibling newsletter-cc-cl project's
    send_brief.py learned this the hard way: several early runs' IDs only ever
    existed in whatever log file happened to capture that run's console output,
    and were unrecoverable once the terminal scrolled past. Returning it here
    (not just printing it) is what lets the caller persist it into `sends`.
    """
    import json
    import urllib.error
    import urllib.request

    key = _env("SENDGRID_WEB_API")
    if not key:
        raise RuntimeError("SENDGRID_WEB_API is not set in .env")
    from_email = _env("SENDER_EMAIL") or _env("SMTP_USER")
    if not from_email:
        raise RuntimeError("set SENDER_EMAIL in .env -- SendGrid needs a verified sender")

    payload = {
        "personalizations": [{"to": [{"email": to}]}],
        "from": {"email": from_email,
                 "name": _env("SENDER_NAME", "Coherent Market Insights")},
        "subject": subject,
        "content": [
            {"type": "text/plain",
             "value": "This brief is formatted as HTML. Please view it in an "
                      "HTML-capable client."},
            {"type": "text/html", "value": html},
        ],
    }
    if reply_to:
        payload["reply_to"] = {"email": reply_to[0]}

    req = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            if r.status not in (200, 202):
                raise RuntimeError(f"SendGrid returned {r.status}")
            return r.headers.get("X-Message-Id", "")
    except urllib.error.HTTPError as e:
        # SendGrid puts the actual reason in the body; the status alone is not
        # actionable (403 is usually an unverified sender, not a bad key).
        detail = e.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"SendGrid {e.code}: {detail}") from None


def send_html(source_html: str, to: str, subject: str, *, filename_hint: str = "brief",
              conn=None) -> str:
    """Core sender: takes already-rendered browser HTML (not a path), converts it via
    emailify(), and sends it to one recipient over SMTP.

    Returns the SendGrid X-Message-Id (empty string over SMTP, which has no
    equivalent) so a caller can persist it for later reply/bounce matching.
    """
    # The 2026-09-16 bounce wave (~40% rejected 550 5.4.1 by Outlook/M365) was raw
    # Gmail SMTP from an unauthenticated domain. SendGrid is the fix that block asked
    # for, so it is not gated by the same switch -- but SMTP still is.
    if _provider() != "sendgrid" and os.getenv("CMI_SENDING_DISABLED", "1") != "0":
        raise RuntimeError(
            "SMTP sending is disabled (CMI_SENDING_DISABLED != '0'). ~40% of sends on "
            "2026-09-16 bounced with 550 5.4.1 -- raw Gmail SMTP from an "
            "unauthenticated domain is rejected as bulk mail. Either set "
            "SENDGRID_WEB_API to send over SendGrid instead, or fix SPF/DKIM/DMARC "
            "for the sending domain before re-enabling."
        )
    provider = _provider()
    html = emailify(source_html)

    owns_conn = conn is None
    _c = conn or dbmod.get_conn()
    try:
        dbmod.init_db(_c)
        reply_to = _reply_to_list(_c)
    finally:
        if owns_conn:
            _c.close()

    if provider == "sendgrid":
        msg_id = _send_via_sendgrid(html, to, subject, reply_to)
        print(f"sent {filename_hint} -> {to} (sendgrid)"
              + (f"  [X-Message-Id: {msg_id}]" if msg_id else ""))
        return msg_id

    cfg = _smtp_config()

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{cfg['from_name']} <{cfg['from_email']}>"
    msg["To"] = to

    if reply_to:
        msg["Reply-To"] = ", ".join(reply_to)

    msg.attach(MIMEText("This brief is formatted as HTML. Please view it in an HTML-capable client.", "plain"))
    msg.attach(MIMEText(html, "html"))

    context = ssl.create_default_context()
    with smtplib.SMTP(cfg["host"], cfg["port"], timeout=30) as server:
        server.starttls(context=context)
        server.login(cfg["user"], cfg["password"])
        server.sendmail(cfg["from_email"], [to], msg.as_string())

    print(f"sent {filename_hint} -> {to} (smtp)")
    return ""


def send(path: str | Path, to: str, subject: str | None = None) -> None:
    """CLI entry point: send an already-rendered .html FILE to one recipient."""
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    send_html(source, to, subject or path.stem.replace("_", " "), filename_hint=path.name)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python mailer.py <html_path> <to_email> [subject]")
        raise SystemExit(1)
    send(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
