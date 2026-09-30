"""Generate briefs for a list of leads through the local fine-tune, then validate each.

    python batch_generate.py --xlsx Book1.xlsx --limit 5
    python batch_generate.py --xlsx Book1.xlsx --base-url https://<host>/v1

Writes HTML to output/ and a summary table at the end. Does NOT email anything --
this is the dry run that comes before send_one.py.

The website matters more than anything else here. Profile inference from a company
NAME alone is where this pipeline fails worst (ImmunoACT, a CAR-T maker, was profiled
as an "immunoassay CDMO", which poisoned every downstream query). So the email domain
is used as the website when the sheet has no explicit one -- a free-mail domain
(gmail/yahoo/...) is not a company site and is skipped rather than guessed at.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time

FREE_MAIL = {"gmail.com", "yahoo.com", "yahoo.co.in", "hotmail.com", "outlook.com",
             "rediffmail.com", "live.com", "icloud.com", "protonmail.com", "aol.com"}


def read_leads(path: str, limit: int) -> list[dict]:
    """Columns are positional: this export has no header row."""
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.worksheets[0]
    out = []
    for r in ws.iter_rows(values_only=True):
        if not any(c is not None and str(c).strip() for c in r):
            continue
        name = (str(r[3]) if r[3] else "there").strip()
        company = (str(r[7]) if r[7] else "").strip()
        email = (str(r[8]) if r[8] else "").strip()
        country = (str(r[11]) if r[11] else "India").strip()
        # Column 5 is the market the lead actually enquired about ("Wood Veneers
        # Market"). Every row in this export has BOTH a company and a topic -- they are
        # two facts about one lead, not alternatives.
        topic = (str(r[5]) if r[5] else "").strip()
        domain = email.split("@")[-1].lower() if "@" in email else ""

        website = "" if (not domain or domain in FREE_MAIL) else f"https://{domain}"
        if (not company or company.lower() == "none") and website:
            # Fall back to the domain's second-level label as the company name.
            company = domain.split(".")[0].title()
        # "VVF LTd." -> "VVF" reads better in a salutation and searches better.
        company = company.rstrip(". ").strip()

        if not company:
            continue
        out.append({"name": name.split()[0] if name else "there", "company": company,
                    "email": email, "country": country, "website": website,
                    "topic": topic})
        if len(out) >= limit:
            break
    return out


def dedupe_by_company(leads: list[dict]) -> list[dict]:
    """One brief per company. Two contacts at the same firm get the same signals, so
    generating twice burns GPU time to produce near-identical copy."""
    seen, out = set(), []
    for l in leads:
        key = l["company"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(l)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-generate briefs for leads in a spreadsheet.")
    ap.add_argument("--xlsx", default="Book1.xlsx")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--base-url", default=os.environ.get("LOCAL_LLM_BASE_URL", ""))
    ap.add_argument("--model", default="cmi-7b")
    ap.add_argument("--per-company", action="store_true",
                    help="One brief per company (default). Use --no-dedupe for one per lead.")
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--kind", choices=["signal", "report", "auto"], default="auto",
                    help="'report' builds section 1 from the lead's enquired market "
                         "(sheet column 5); 'auto' uses report when a topic is present.")
    ap.add_argument("--outdir", default="output",
                    help="where the HTML lands; use separate dirs to compare runs.")
    ap.add_argument("--workers", type=int, default=8,
                    help="companies generated concurrently. vLLM batches these into one "
                         "GPU pass; a plain transformers server gains nothing above 1.")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    os.environ["LLM_PROVIDER"] = "local"
    if args.base_url:
        os.environ["LOCAL_LLM_BASE_URL"] = args.base_url
    os.environ["LOCAL_LLM_MODEL"] = args.model

    import compose
    import config
    import geo
    import llm
    import render
    import validate_brief as vb
    from brief_builder import build_brief
    from collect import collect
    from score import eligible, score_all
    from tenant import Profile

    # Two brief types, and they are not interchangeable:
    #   signal  -- ACCOUNT specific: prospect the lead's own company (default)
    #   report  -- the market the lead enquired about, from sheet column 5
    # 'auto' keeps signal, because a leads sheet carrying a topic does not by itself
    # mean the lead wants a market report rather than account intelligence.
    if args.kind == "auto":
        args.kind = "signal"
    os.makedirs(args.outdir, exist_ok=True)
    leads = read_leads(args.xlsx, args.limit if args.no_dedupe else args.limit * 3)
    if not args.no_dedupe:
        leads = dedupe_by_company(leads)[: args.limit]

    print(f"endpoint: {llm.LOCAL_BASE_URL}")
    print(f"{len(leads)} companies to generate\n")
    for l in leads:
        print(f"  {l['company']:<22} {l['website'] or '(no website -- name-only inference)'}")
    print()

    # One brief at a time leaves the GPU idle between tokens: generating a token
    # requires reading all 7B weights, and the compute units wait on that read. vLLM's
    # continuous batching computes the next token for many requests off ONE weight
    # read, so throughput multiplies -- measured 19 tok/s sequential against 380 tok/s
    # at 32 concurrent on an L4, which is Rs.1.77 -> Rs.0.09 per brief.
    #
    # The calls WITHIN one brief stay sequential (call 2 needs call 1's output), so the
    # concurrency has to come from running companies in parallel. Each worker owns its
    # own Profile and article list, and the only shared state is llm.STATS (counter
    # increments, harmless) and the output directory (distinct filenames per company).
    def _one(idx_lead):
        i, lead = idx_lead
        t0 = time.time()
        row = {"company": lead["company"], "website": bool(lead["website"])}
        try:
            country = config.get_country(lead["country"])
            profile = Profile.from_company(lead["company"], lead["website"],
                                           lead["name"], country["name"])
            arts = collect(profile, country)
            if not geo.is_global(country):
                arts, _ = geo.filter_country(arts, country)
            scored = score_all(arts, profile)
            keep = eligible(scored)
            _topic = lead.get("topic", "") if args.kind == "report" else ""
            brief = build_brief(keep or scored, profile, log=lambda *_: None,
                                market_topic=_topic)
            brief = brief[0] if isinstance(brief, tuple) else brief
            # A lead who asked about "Wear Parts Market" wants that market, not a
            # prospect list. Signal mode sent the pipeline hunting for buyers of and
            # rivals to a tiny carbide shop, found neither, and dropped sections; report
            # mode builds section 1 as "Your Market" around the topic instead.
            kind = "report" if (args.kind == "report" and lead.get("topic")) else "signal"
            ctx = compose.build_context(profile, brief, kind,
                                        market_topic=lead.get("topic", "") if kind == "report" else "",
                                        log=lambda *_: None)
            html = render.render(ctx, "cmi_brief.html")

            # Include the recipient: two contacts at the same company produced the
            # same filename, so the second brief silently overwrote the first and a
            # 5-brief batch left 4 files on disk.
            slug = "".join(c for c in lead["company"].lower() if c.isalnum())[:20]
            who = "".join(c for c in lead["name"].lower() if c.isalnum())[:12] or "there"
            path = os.path.join(args.outdir, f"cmi_batch_{slug}_{who}.html")
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)

            fails, warns = vb.check(path, profile)
            row.update(path=path, fails=fails, warns=warns,
                       status="FAIL" if fails else ("WARN" if warns else "PASS"),
                       secs=round(time.time() - t0))
        except Exception as e:  # noqa: BLE001 -- one bad lead must not lose the batch
            row.update(status="ERROR", error=f"{type(e).__name__}: {e}",
                       secs=round(time.time() - t0), fails=[], warns=[])
        return row

    results = []
    t_batch = time.time()
    if args.workers == 1:
        for pair in enumerate(leads, 1):
            r = _one(pair)
            results.append(r)
            print(f"[{pair[0]}/{len(leads)}] {r['company']}: {r['status']} {r['secs']}s", flush=True)
    else:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_one, pair): pair for pair in enumerate(leads, 1)}
            done = 0
            for fut in cf.as_completed(futs):
                r = fut.result()
                results.append(r)
                done += 1
                print(f"[{done}/{len(leads)}] {r['company']}: {r['status']} {r['secs']}s", flush=True)
                for f_ in r.get("fails", []):
                    print(f"        x {f_}")
                for w_ in r.get("warns", []):
                    print(f"        ! {w_}")
    batch_secs = time.time() - t_batch

    print("\n" + "=" * 66)
    print(f"{'company':<22}{'status':<8}{'secs':>6}  notes")
    for r in results:
        note = r.get("error") or (r["fails"][0][:34] if r.get("fails")
                                  else (r["warns"][0][:34] if r.get("warns") else ""))
        print(f"{r['company'][:21]:<22}{r['status']:<8}{r.get('secs', 0):>6}  {note}")
    ok = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n{ok}/{len(results)} passed validation")
    print(f"LLM calls: {llm.STATS}")


if __name__ == "__main__":
    main()
