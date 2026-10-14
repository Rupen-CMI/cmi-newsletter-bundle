"""Generate and send a single CMI brief to one lead.

    python send_one.py --company ImmunoACT --recipient Satyen --email satyen@immunoact.com --market India
    python send_one.py --email you@example.com --kind report --report "Bone Marrow Transplant Market"

Runs the same pipeline as run.py (tenant profile -> collect -> score -> synthesize
-> compose -> render), then emails the result via mailer.py and records the send
in cmi_newsletter.db.
"""
from __future__ import annotations

import argparse
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import compose
import config
import db as dbmod
import geo
import mailer
import render
from brief_builder import build_brief
from collect import collect
from score import score_all, eligible
from tenant import Profile
from validate_brief import check_html

TEMPLATE = "cmi_brief.html"


def _safe_log(msg: str = "") -> None:
    """print() that never raises on a console codec that can't encode the message
    (e.g. Windows cp1252 choking on an em dash or arrow in generated brief text)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(str(msg).encode(enc, errors="replace").decode(enc, errors="replace"))


def _validate_or_raise(html: str, profile, *, email: str = "", kind: str = "",
                       topic: str = "") -> None:
    """Gate a send on validate_brief.py's structure/roles/fit checks -- a brief with
    a missing section or a rival in the customer slot must never reach an inbox
    silently. Warnings are logged but don't block; only FAILs stop the send.

    `email`/`kind`/`topic` enable the DOMAIN and TOPIC checks (independent of
    `profile`, which can itself be wrong -- see tenant.Profile.from_company's
    docstring). Omitting them, as every call site here used to, silently skipped
    both checks for every brief sent through this path."""
    fails, warns = check_html(html, profile, email=email, kind=kind, topic=topic)
    for w in warns:
        _safe_log(f"  [validate] WARN: {w}")
    if fails:
        for f in fails:
            _safe_log(f"  [validate] FAIL: {f}")
        raise RuntimeError(f"brief failed validation for {profile.company}: {'; '.join(fails)}")


def send_one(*, email: str, company: str = "", recipient: str = "there", market: str = "Global",
             kind: str = "signal", report_name: str = "", subject: str = "",
             campaign_label: str = "manual", website: str = "") -> str:
    country = config.get_country(market)
    market_name = country["name"]

    # Opened before from_company (rather than at the DB-write point below) so it can
    # reuse enrich.py's per-domain company_intel cache instead of re-fetching the
    # homepage live -- same reasoning as pipeline.stage_generate.
    conn = dbmod.get_conn()
    dbmod.init_db(conn)

    # Pass the website through: Profile.from_company fetches it and profiles from the
    # real homepage copy instead of guessing from the company name. Raises ValueError
    # if company is blank -- see that function's docstring for why there is no
    # fallback here.
    profile = Profile.from_company(company, website, recipient or "there", market_name,
                                   conn=conn)
    profile.recipient = recipient or "there"

    _safe_log(f"generating brief for {email} ({profile.company}) [{market_name}]")
    articles = collect(profile, country)
    if not geo.is_global(country):
        kept, _ = geo.filter_country(articles, country)
        articles = kept

    scored = score_all(articles, profile)
    keep = eligible(scored)

    brief = build_brief(keep or scored, profile, log=_safe_log)
    brief = brief[0] if isinstance(brief, tuple) else brief

    ctx = compose.build_context(profile, brief, kind, market_topic=report_name, log=_safe_log)
    html = render.render(ctx, TEMPLATE)

    # Was `if kind == "signal"` -- report-kind briefs sent through this path got
    # ZERO validation (not even STRUCTURE/ROLES/FIT), silently, since the day this
    # function grew a `kind` parameter. Every brief this function sends must be
    # checked regardless of kind.
    expected_subject = report_name if kind == "report" else company
    _validate_or_raise(html, profile, email=email, kind=kind, topic=expected_subject)

    dbmod.upsert_lead(conn, email=email, name=recipient, company=company,
                       market=market_name, report_name=report_name, source="manual")

    subject = subject or f"Coherent Market Insights · {profile.company}'s Brief"
    try:
        msg_id = mailer.send_html(html, email, subject, filename_hint=email, conn=conn)
        dbmod.record_send(conn, email=email, status="sent", subject=subject, campaign=campaign_label,
                          message_id=msg_id)
        _safe_log(f"sent -> {email}")
    except Exception as e:  # noqa: BLE001
        dbmod.record_send(conn, email=email, status="failed", detail=f"{type(e).__name__}: {e}",
                           campaign=campaign_label)
        conn.close()
        raise
    conn.close()
    return html


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate and send one CMI brief.")
    ap.add_argument("--email", required=True, help="recipient email address")
    ap.add_argument("--company", default="", help="lead's company (blank = CMI's own default profile)")
    ap.add_argument("--website", default="", help="lead's website; strongly recommended -- the profile is "
                                                   "inferred from the homepage copy rather than the name alone")
    ap.add_argument("--recipient", default="there", help="recipient first name")
    ap.add_argument("--market", default="Global")
    ap.add_argument("--kind", choices=["signal", "report"], default="signal")
    ap.add_argument("--report", default="", help="report/market topic (required for --kind report)")
    ap.add_argument("--subject", default="")
    args = ap.parse_args()

    if args.kind == "report" and not args.report:
        print("error: --kind report requires --report <topic>", file=sys.stderr)
        raise SystemExit(1)

    send_one(email=args.email, company=args.company, recipient=args.recipient,
              market=args.market, kind=args.kind, report_name=args.report, subject=args.subject,
              website=args.website)
