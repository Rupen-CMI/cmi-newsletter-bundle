"""Generate real briefs for fine-tuning training data, WITHOUT sending anything.

Runs the exact same pipeline as send_one.py (tenant profile -> collect -> score ->
LLM synthesis -> compose -> render) but stops short of mailer.send_html — nothing
is emailed. Every LLM call is captured by llm.py's training-data logger (enabled
via CMI_LOG_TRAINING_DATA=1 in .env) into training_data/live_calls.jsonl, and each
brief's full rendered HTML is saved to training_data/fresh_briefs/ for reference.

    python generate_training_briefs.py --market India --limit 30
    python generate_training_briefs.py --market India --limit 30 --kind report

Leads are read from cmi_newsletter.db (the same leads used for the real campaign),
filtered to ones not yet used for training data this run, so re-running with a
higher --limit only generates the DELTA rather than repeating work.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import compose
import config
import db as dbmod
import geo
import llm
import render
from brief_builder import build_brief
from collect import collect
from score import score_all, eligible
from tenant import Profile

TEMPLATE = "cmi_brief.html"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data", "fresh_briefs")


def _safe_log(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(str(msg).encode(enc, errors="replace").decode(enc, errors="replace"), flush=True)


def _safe_filename(email: str) -> str:
    return email.replace("@", "_at_").replace(".", "_").replace("/", "_")


def generate_one(lead, kind: str) -> tuple[str, bool]:
    """Runs the pipeline through render() only — no send. Returns (html, ok)."""
    market = lead["market"] or "Global"
    country = config.get_country(market)
    market_name = country["name"]

    # Raises ValueError if lead["company"] is blank -- see Profile.from_company's
    # docstring for why there is no CMI-self fallback here.
    profile = Profile.from_company(lead["company"], "", lead["name"] or "there", market_name)
    profile.recipient = lead["name"] or "there"

    articles = collect(profile, country)
    if not geo.is_global(country):
        kept, _ = geo.filter_country(articles, country)
        articles = kept

    scored = score_all(articles, profile)
    keep = eligible(scored)

    brief = build_brief(keep or scored, profile, log=_safe_log)
    brief = brief[0] if isinstance(brief, tuple) else brief

    ctx = compose.build_context(profile, brief, kind, market_topic=lead["report_name"] or "", log=_safe_log)
    html = render.render(ctx, TEMPLATE)

    # A degraded brief (LLM outage, etc.) has an empty lede — don't save it as
    # training data, and flag it so the caller can retry the lead later.
    ok = bool(ctx.get("lede", "").strip())
    return html, ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate briefs for training data only — never sends.")
    ap.add_argument("--market", default="India", help="Country/market to pull leads from")
    ap.add_argument("--limit", type=int, default=30, help="How many leads to generate for")
    ap.add_argument("--kind", choices=["signal", "report"], default="signal")
    ap.add_argument("--campaign", default=None, help="Only pull leads from this campaign label")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    already = {f.split(".html")[0] for f in os.listdir(OUT_DIR) if f.endswith(".html")}

    conn = dbmod.get_conn()
    dbmod.init_db(conn)

    query = "SELECT * FROM leads WHERE market = ?"
    params: list = [args.market]
    if args.campaign:
        query += " AND campaign = ?"
        params.append(args.campaign)
    query += " ORDER BY created_at"
    all_leads = conn.execute(query, params).fetchall()

    candidates = [l for l in all_leads if _safe_filename(l["email"]) not in already]
    batch = candidates[: args.limit]

    print(f"{len(all_leads)} {args.market} leads total, {len(already)} already have training "
          f"briefs, generating {len(batch)} more (NO SENDING — training data only)\n")

    ok_count = degraded_count = failed_count = 0
    for idx, lead in enumerate(batch, start=1):
        _safe_log(f"[{idx}/{len(batch)}] generating for {lead['email']} ({lead['company'] or lead['name']})")
        try:
            html, ok = generate_one(lead, args.kind)
            fname = f"{_safe_filename(lead['email'])}.html"
            with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
                f.write(html)
            if ok:
                ok_count += 1
                _safe_log(f"  saved (healthy) -> {fname}")
            else:
                degraded_count += 1
                _safe_log(f"  saved (DEGRADED — empty lede, LLM likely failed) -> {fname}")
        except Exception as e:  # noqa: BLE001 — one bad lead must not kill the batch
            failed_count += 1
            _safe_log(f"  FAILED: {type(e).__name__}: {e}")
        time.sleep(2)  # be polite to news/LLM APIs; no SMTP involved so no 12s send-delay needed

    conn.close()
    _safe_log(f"\nDone: healthy={ok_count} degraded={degraded_count} failed={failed_count} "
              f"of {len(batch)} attempted. Nothing was sent.")
    _safe_log(f"LLM stats: {llm.STATS}")


if __name__ == "__main__":
    main()
