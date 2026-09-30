"""One-off importer: load an .xlsx lead list into cmi_newsletter.db so the leads
live in the versionable SQLite DB instead of a raw spreadsheet.

    python import_leads.py "sample leads.xlsx" --campaign launch1

Uses campaign.read_contacts() (the same header-detecting / column-sniffing parser
used by the batch sender) so a lead list with no header row, like CMI's own
tracking exports, imports the same way it would send. Every imported lead starts
with schedule_status='queued' so scheduler.py picks it up on its next pass.
"""
from __future__ import annotations

import argparse

import campaign
import db as dbmod


def import_xlsx(xlsx_path: str, sheet: str | None = None, campaign_label: str = "default",
                 kind: str = "signal") -> dict:
    conn = dbmod.get_conn()
    dbmod.init_db(conn)

    contacts = campaign.read_contacts(xlsx_path, sheet=sheet)
    imported = 0
    skipped_unsub = 0
    for c in contacts:
        email_l = c.email.lower()
        if dbmod.is_unsubscribed(conn, email_l):
            skipped_unsub += 1
            continue
        dbmod.upsert_lead(
            conn, email=email_l, name=c.name, company=c.company, job_title=c.job_title,
            market=c.market, report_name=c.report_name, source="xlsx_import",
            campaign=campaign_label, kind=kind, schedule_status="queued",
        )
        imported += 1

    conn.close()
    return {"total_rows": len(contacts), "imported": imported, "skipped_unsubscribed": skipped_unsub}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Import an xlsx lead list into cmi_newsletter.db.")
    ap.add_argument("xlsx", help="Path to the .xlsx lead list")
    ap.add_argument("--sheet", default=None, help="Sheet name (default: first sheet)")
    ap.add_argument("--campaign", default="default", help="Campaign label for these leads")
    ap.add_argument("--kind", choices=["signal", "report"], default="signal")
    args = ap.parse_args()

    result = import_xlsx(args.xlsx, sheet=args.sheet, campaign_label=args.campaign, kind=args.kind)
    print(f"Imported {result['imported']}/{result['total_rows']} leads into cmi_newsletter.db "
          f"(campaign='{args.campaign}', {result['skipped_unsubscribed']} already-unsubscribed skipped).")
