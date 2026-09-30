"""Continuous scheduler: sends each queued lead's newsletter during THEIR country's
local business hours, so a lead in Tokyo gets it at 9am JST and a lead in New York
gets it at 9am ET, not both at the same UTC instant.

Meant to run as a long-lived process on a VM (systemd service, Windows Task
Scheduler entry, or `nohup python scheduler.py &`):

    python scheduler.py                       # poll every 5 minutes, forever
    python scheduler.py --once                # one pass then exit (for cron)
    python scheduler.py --poll-seconds 60      # tighter polling

Leads live in cmi_newsletter.db (db.py), populated by import_leads.py or the
dashboard's xlsx upload. Each lead has a `schedule_status`: 'queued' means still
waiting to be sent. On every pass, the scheduler:
  1. loads all 'queued' leads
  2. for each, checks whether it's currently business hours in that lead's country
     (timezones.in_business_hours) — 9am-6pm local on a business day by default,
     adjusted per-country in timezones.COUNTRY_TZ (e.g. Gulf countries use Sun-Thu)
  3. if in-window, generates and sends that lead's brief (send_one.py's pipeline),
     records the result in `sends` + updates `leads.schedule_status`
  4. if not in-window, leaves it queued and tries again on a later pass

This spreads sends across the day/week following each recipient's local clock,
rather than blasting the whole list at once regardless of when they'll actually
read it. A per-lead --delay still applies between individual sends within the
same pass to stay within SMTP rate limits.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import db as dbmod
import send_one
import timezones


def _safe_log(msg: str = "") -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(str(msg).encode(enc, errors="replace").decode(enc, errors="replace"))


def run_once(delay_seconds: float = 12.0, campaign: str | None = None) -> dict:
    """One scheduler pass: send every queued lead currently in their business-hours
    window, skip the rest until a later pass. Returns a summary dict."""
    conn = dbmod.get_conn()
    dbmod.init_db(conn)

    leads = dbmod.queued_leads(conn, campaign=campaign)
    summary = {"checked": len(leads), "sent": 0, "failed": 0, "not_in_window": 0}

    now = datetime.now(timezone.utc)
    for lead in leads:
        if dbmod.is_unsubscribed(conn, lead["email"]):
            dbmod.set_lead_status(conn, lead["email"], "unsubscribed")
            continue

        if not timezones.in_business_hours(lead["market"], now):
            summary["not_in_window"] += 1
            continue

        _safe_log(f"[{now.isoformat()}] sending to {lead['email']} "
                   f"({lead['market']} — local business hours)")
        try:
            send_one.send_one(
                email=lead["email"], company=lead["company"] or "", recipient=lead["name"] or "there",
                market=lead["market"] or "Global", kind=lead["kind"] or "signal",
                report_name=lead["report_name"] or "", campaign_label=lead["campaign"] or "default",
            )
            dbmod.set_lead_status(conn, lead["email"], "sent")
            summary["sent"] += 1
        except Exception as e:  # noqa: BLE001 — one bad lead must not kill the scheduler
            _safe_log(f"  failed: {type(e).__name__}: {e}")
            dbmod.set_lead_status(conn, lead["email"], "failed")
            summary["failed"] += 1

        time.sleep(delay_seconds)

    conn.close()
    return summary


def run_forever(poll_seconds: float = 300.0, delay_seconds: float = 12.0,
                 campaign: str | None = None) -> None:
    _safe_log(f"scheduler starting — polling every {poll_seconds:.0f}s, "
               f"{delay_seconds:.0f}s between sends within a pass")
    while True:
        try:
            result = run_once(delay_seconds=delay_seconds, campaign=campaign)
            _safe_log(f"pass done: {result}")
        except Exception as e:  # noqa: BLE001 — a bad pass must not kill the loop
            _safe_log(f"pass errored: {type(e).__name__}: {e}")
        time.sleep(poll_seconds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Send queued leads' newsletters during their local business hours."
    )
    ap.add_argument("--once", action="store_true", help="Run a single pass and exit (for cron)")
    ap.add_argument("--poll-seconds", type=float, default=300.0,
                     help="Seconds between passes when running continuously (default: 300)")
    ap.add_argument("--delay", type=float, default=12.0,
                     help="Seconds between individual sends within a pass (default: 12)")
    ap.add_argument("--campaign", default=None, help="Only process leads in this campaign label")
    args = ap.parse_args()

    if args.once:
        print(run_once(delay_seconds=args.delay, campaign=args.campaign))
    else:
        run_forever(poll_seconds=args.poll_seconds, delay_seconds=args.delay, campaign=args.campaign)
