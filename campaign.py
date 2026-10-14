"""Batch-generate and send CMI briefs to a recipient list loaded from Excel or the
leads DB.

    python campaign.py contacts.xlsx --sheet Sheet1 --limit 5
    python campaign.py contacts.xlsx --kind signal --market India

Reads any .xlsx with columns identifying at least a name and an email address
(header names matched case/whitespace-insensitively — see `_HEADER_ALIASES`).
One brief is generated and sent per row, using the Signal Brief pipeline
(tenant profile -> collect news -> score -> LLM synthesis -> compose -> render)
unless `--kind report` is given, in which case every row needs its own report
topic in a "market"/"report" column.

State is tracked in cmi_newsletter.db (db.py) so a batch can be safely re-run:
rows already sent (for this --campaign label) are skipped, and rows marked
unsubscribed are always skipped.

Rate limiting: --delay seconds between sends (default 12) to stay within SMTP
provider limits and avoid spam-pattern flags on a new sending domain.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import openpyxl

import compose
import config
import geo
import mailer
import render
import db as dbmod
from brief_builder import build_brief
from collect import collect
from score import score_all, eligible
from tenant import Profile
from validate_brief import check_html

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = "cmi_brief.html"

_HEADER_ALIASES = {
    "email": "email", "email 1": "email", "email address": "email", "e-mail": "email",
    "name": "name", "first name": "name", "firstname": "name", "client name": "name",
    "company": "company", "organisation": "company", "organization": "company",
    "job title": "job_title", "jobtitle": "job_title", "title": "job_title",
    "market": "market", "country": "market",
    "report": "report_name", "report name": "report_name", "report topic": "report_name",
}


@dataclass
class Contact:
    email: str
    name: str
    company: str = ""
    job_title: str = ""
    market: str = "Global"
    report_name: str = ""
    row_num: int = 0


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Known country/region names, used to sniff a "market" column by content when there's
# no header to name it. Deliberately short — just enough to recognize common values
# in real-world lead lists (see config.COUNTRIES for the full locale table).
_COUNTRY_HINTS = {
    "india", "united states", "usa", "us", "uk", "united kingdom", "germany", "japan",
    "china", "brazil", "australia", "canada", "france", "italy", "spain", "global",
    "asia pacific", "europe", "north america", "middle east", "africa", "apac", "emea",
}

_TITLE_HINTS = (
    "ceo", "cfo", "coo", "cto", "founder", "president", "director", "manager",
    "head of", "vp", "vice president", "chief", "team lead", "officer",
)
_TITLE_RE = re.compile(r"\b(" + "|".join(re.escape(h) for h in _TITLE_HINTS) + r")\b")

# A person's name: 2-4 capitalized words/initials, no digits, no company-ish suffixes
# (Inc/Ltd/Co/...) or punctuation that would mark it as an organization or free text.
_NAME_RE = re.compile(r"^([A-Z][a-zA-Z.'-]*\s+){1,3}[A-Z][a-zA-Z.'-]*$")
_COMPANY_SUFFIX_RE = re.compile(
    r"\b(inc|ltd|llc|llp|co|corp|corporation|company|group|holdings|pvt|plc|gmbh)\b\.?",
    re.IGNORECASE,
)


def _looks_like_header(row: tuple) -> bool:
    """True if this row's cells look like column names (matches _HEADER_ALIASES)
    rather than a first row of real data."""
    for h in row:
        key = str(h or "").strip().lower()
        if key in _HEADER_ALIASES:
            return True
    return False


def _column_values(sample_rows: list[tuple], i: int) -> list[str]:
    return [str(r[i]).strip() for r in sample_rows if i < len(r) and r[i] is not None and str(r[i]).strip()]


def _sniff_columns(sample_rows: list[tuple]) -> dict[int, str]:
    """No usable header row — guess column roles from cell content across a sample
    of data rows. Always finds 'email' (the only column with @-addresses); name/
    company/job_title/market are best-effort based on value shape and keyword hints.
    Real-world lead exports often have no header row at all, or use columns this app
    doesn't otherwise know how to name (e.g. CMI's own internal-tracking sheets) —
    this keeps those usable instead of rejecting the whole sheet.
    """
    if not sample_rows:
        return {}
    width = max(len(r) for r in sample_rows)
    col_map: dict[int, str] = {}

    email_col = None
    for i in range(width):
        vals = _column_values(sample_rows, i)
        if vals and sum(1 for v in vals if _EMAIL_RE.match(v)) >= max(1, len(vals) // 2):
            email_col = i
            break
    if email_col is None:
        return {}
    col_map[email_col] = "email"

    market_col = None
    title_col = None
    name_col = None
    company_col = None
    for i in range(width):
        if i == email_col:
            continue
        vals = _column_values(sample_rows, i)
        if not vals:
            continue
        lower_vals = [v.lower() for v in vals]

        if market_col is None and sum(1 for v in lower_vals if v in _COUNTRY_HINTS) >= max(1, len(vals) // 2):
            market_col = i
            col_map[i] = "market"
        elif title_col is None and sum(1 for v in lower_vals if _TITLE_RE.search(v)) >= max(1, len(vals) // 2):
            title_col = i
            col_map[i] = "job_title"
        elif name_col is None and sum(1 for v in vals if _NAME_RE.match(v)) >= max(1, len(vals) // 2):
            name_col = i
            col_map[i] = "name"
        elif (company_col is None
              and sum(1 for v in lower_vals if _COMPANY_SUFFIX_RE.search(v)) >= 1
              and not any(v.isdigit() for v in vals)):
            company_col = i
            col_map[i] = "company"

    # Fallback: if no column matched the strict name pattern, use the first unclaimed
    # text column (non-numeric, more than one distinct value) immediately left of the
    # email column — "Name, ..., Email" is the most common real-world layout.
    if name_col is None:
        candidates = [i for i in range(width)
                      if i not in col_map and _column_values(sample_rows, i)
                      and not all(v.isdigit() for v in _column_values(sample_rows, i))]
        left = email_col - 1
        pick = left if left in candidates else (candidates[0] if candidates else None)
        if pick is not None:
            col_map[pick] = "name"

    # Fallback: first remaining unclaimed non-numeric text column becomes company.
    if company_col is None:
        remaining = [i for i in range(width)
                     if i not in col_map and _column_values(sample_rows, i)
                     and not all(v.isdigit() for v in _column_values(sample_rows, i))]
        if remaining:
            col_map[remaining[0]] = "company"

    return col_map


def _load_rows(path: str, sheet: str | None = None) -> tuple[list[tuple], str]:
    """Load every row as a tuple of cell values from either an .xlsx sheet or a
    .csv file, returning (rows, label) where label is the sheet name / file name
    for error messages. CSV support exists because CMI's own lead-tracking export
    (leads_export_*.csv) is CSV, not Excel."""
    if str(path).lower().endswith(".csv"):
        import csv
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = [tuple(r) for r in csv.reader(f)]
        return rows, os.path.basename(path)

    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet] if sheet else wb[wb.sheetnames[0]]
    return list(ws.iter_rows(values_only=True)), ws.title


def read_contacts(xlsx_path: str, sheet: str | None = None) -> list[Contact]:
    all_rows, label = _load_rows(xlsx_path, sheet)
    if not all_rows:
        raise ValueError(f"'{label}' is empty.")

    first_row = all_rows[0]
    if _looks_like_header(first_row):
        header_row = first_row
        data_rows = all_rows[1:]
        col_map: dict[int, str] = {}
        for i, h in enumerate(header_row):
            key = str(h or "").strip().lower()
            if key in _HEADER_ALIASES:
                col_map[i] = _HEADER_ALIASES[key]
        start_row_num = 2
    else:
        # No recognizable header — every row is data. Sniff column roles by content
        # instead of rejecting the sheet outright (real lead exports, e.g. CMI's own
        # tracking sheets, often have no header row or use columns this app doesn't
        # otherwise know how to name).
        data_rows = all_rows
        col_map = _sniff_columns(data_rows[:50])
        start_row_num = 1

    if "email" not in col_map.values():
        raise ValueError(
            f"No recognizable email column in '{label}'. "
            f"First row: {[c for c in first_row]}"
        )

    contacts: list[Contact] = []
    for row_num, row in enumerate(data_rows, start=start_row_num):
        fields = {"email": "", "name": "", "company": "", "job_title": "", "market": "", "report_name": ""}
        for i, field in col_map.items():
            v = row[i] if i < len(row) else None
            if v is not None:
                fields[field] = str(v).strip()
        email = fields["email"]
        if not email or not _EMAIL_RE.match(email):
            continue
        name = fields["name"] or email.split("@")[0]
        contacts.append(Contact(
            email=email, name=name.split()[0] if name else "there",
            company=fields["company"], job_title=fields["job_title"],
            market=fields["market"] or "Global", report_name=fields["report_name"],
            row_num=row_num,
        ))
    return contacts


def _generate_brief_html(contact: Contact, kind: str, log) -> tuple[str, object]:
    country = config.get_country(contact.market or "Global")
    market_name = country["name"]

    # Raises ValueError if contact.company is blank -- see Profile.from_company's
    # docstring for why there is no CMI-self fallback here.
    profile = Profile.from_company(contact.company, "", contact.name, market_name)
    profile.recipient = contact.name

    articles = collect(profile, country)
    if not geo.is_global(country):
        kept, _ = geo.filter_country(articles, country)
        articles = kept

    scored = score_all(articles, profile)
    keep = eligible(scored)

    brief = build_brief(keep or scored, profile, log=log)
    brief = brief[0] if isinstance(brief, tuple) else brief

    ctx = compose.build_context(profile, brief, kind, market_topic=contact.report_name, log=log)
    return render.render(ctx, TEMPLATE), profile


def run_campaign(
    xlsx_path: str,
    sheet: str | None = None,
    limit: int | None = None,
    delay_seconds: float = 12.0,
    subject: str = "Coherent Market Insights · Your Brief",
    kind: str = "signal",
    campaign_label: str = "",
    on_progress=None,
    should_stop=None,
) -> dict:
    conn = dbmod.get_conn()
    dbmod.init_db(conn)

    contacts = read_contacts(xlsx_path, sheet)
    if limit:
        contacts = contacts[:limit]

    campaign_label = campaign_label or os.path.basename(xlsx_path)
    summary = {"sent": 0, "failed": 0, "skipped": 0, "unsubscribed": 0, "stopped": False}

    for c in contacts:
        if should_stop and should_stop():
            summary["stopped"] = True
            if on_progress:
                on_progress({"email": "", "status": "stopped", "detail": "stopped by user"})
            break

        email_l = c.email.lower()
        dbmod.upsert_lead(conn, email=email_l, name=c.name, company=c.company,
                           job_title=c.job_title, market=c.market,
                           report_name=c.report_name, source="xlsx_import")

        if dbmod.is_unsubscribed(conn, email_l):
            dbmod.record_send(conn, email=email_l, status="unsubscribed",
                               detail="local suppression list", campaign=campaign_label)
            summary["unsubscribed"] += 1
            if on_progress:
                on_progress({"email": c.email, "status": "unsubscribed"})
            continue

        if dbmod.send_status(conn, email_l, campaign_label) == "sent":
            summary["skipped"] += 1
            if on_progress:
                on_progress({"email": c.email, "status": "skipped", "detail": "already sent"})
            continue

        try:
            html, profile = _generate_brief_html(c, kind, log=lambda m: None)
            # Was `if kind == "signal"` -- report-kind briefs sent through this path
            # got ZERO validation, silently. Every brief sent here must be checked
            # regardless of kind. email/topic also now passed through so the
            # DOMAIN/TOPIC checks (independent of `profile`, which can itself be
            # wrong) actually run here instead of silently no-op'ing on blank args.
            expected_subject = c.report_name if kind == "report" else c.company
            fails, _warns = check_html(html, profile, email=c.email, kind=kind,
                                       topic=expected_subject or "")
            if fails:
                raise RuntimeError(f"brief failed validation: {'; '.join(fails)}")
            mailer.send_html(html, c.email, subject, filename_hint=f"row{c.row_num}", conn=conn)
            dbmod.record_send(conn, email=email_l, status="sent", subject=subject,
                               campaign=campaign_label)
            summary["sent"] += 1
            if on_progress:
                on_progress({"email": c.email, "status": "sent"})
        except Exception as e:  # noqa: BLE001 — one bad row must not kill the whole batch
            dbmod.record_send(conn, email=email_l, status="failed",
                               detail=f"{type(e).__name__}: {e}", campaign=campaign_label)
            summary["failed"] += 1
            if on_progress:
                on_progress({"email": c.email, "status": "failed", "detail": str(e)})

        time.sleep(delay_seconds)

    conn.close()
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Batch-send CMI briefs from an xlsx contact list.")
    ap.add_argument("xlsx", help="Path to the .xlsx contact list")
    ap.add_argument("--sheet", default=None, help="Sheet name (default: first sheet)")
    ap.add_argument("--limit", type=int, default=None, help="Only send to the first N contacts")
    ap.add_argument("--delay", type=float, default=12.0, help="Seconds between sends (default: 12)")
    ap.add_argument("--subject", default="Coherent Market Insights · Your Brief")
    ap.add_argument("--kind", choices=["signal", "report"], default="signal")
    ap.add_argument("--campaign", default="", help="Label for this batch (default: xlsx filename)")
    args = ap.parse_args()

    result = run_campaign(args.xlsx, sheet=args.sheet, limit=args.limit,
                           delay_seconds=args.delay, subject=args.subject, kind=args.kind,
                           campaign_label=args.campaign,
                           on_progress=lambda e: print(f"  {e['status']:12} {e['email']}"))
    print(f"\nDone: {result}")
