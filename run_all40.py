"""Generate both brief sets for every lead in ONE concurrency window.

    python run_all40.py --workers 40 --base-url https://<host>/v1

`batch_generate.py --kind signal` then `--kind report` runs two batches back to back, so
the wall clock is the sum of two batches. But the calls INSIDE one brief are sequential
(call N+1 needs call N's output) while different briefs are fully independent -- so the
batch wall clock is bounded below by the slowest single brief, not by the total work.
Once workers >= briefs, adding briefs is nearly free.

That is the whole idea here: submit all 40 jobs (20 account + 20 report) to one pool.
If the KV cache holds them, 40 briefs cost about the same wall clock as 20, and the
per-brief cost halves.

Prints a throughput and cost line so runs can be compared directly.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="Book1.xlsx")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--base-url", default=os.environ.get("LOCAL_LLM_BASE_URL", ""))
    ap.add_argument("--model", default="cmi-7b")
    ap.add_argument("--account-dir", default="output_all40_account")
    ap.add_argument("--report-dir", default="output_all40_report")
    ap.add_argument("--rate", type=float, default=41.31,
                    help="GPU Rs/hr for the cost line (on-demand L4 default)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    os.environ["LLM_PROVIDER"] = "local"
    if args.base_url:
        os.environ["LOCAL_LLM_BASE_URL"] = args.base_url
    os.environ["LOCAL_LLM_MODEL"] = args.model
    os.makedirs(args.account_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    sys.argv = ["x"]
    import batch_generate as bg
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

    leads = bg.read_leads(args.xlsx, args.limit)
    jobs = ([(l, "signal", args.account_dir) for l in leads]
            + [(l, "report", args.report_dir) for l in leads])

    print(f"endpoint: {llm.LOCAL_BASE_URL}")
    print(f"{len(jobs)} briefs ({len(leads)} account + {len(leads)} report), "
          f"{args.workers} workers\n", flush=True)

    def one(job):
        lead, kind, outdir = job
        t0 = time.time()
        row = {"company": lead["company"], "kind": kind}
        try:
            country = config.get_country(lead["country"])
            profile = Profile.from_company(lead["company"], lead["website"],
                                           lead["name"], country["name"])
            arts = collect(profile, country)
            if not geo.is_global(country):
                arts, _ = geo.filter_country(arts, country)
            scored = score_all(arts, profile)
            keep = eligible(scored)
            topic = lead.get("topic", "") if kind == "report" else ""
            brief = build_brief(keep or scored, profile, log=lambda *_: None,
                                market_topic=topic)
            brief = brief[0] if isinstance(brief, tuple) else brief
            ctx = compose.build_context(profile, brief, kind, market_topic=topic,
                                        log=lambda *_: None)
            html = render.render(ctx, "cmi_brief.html")

            slug = "".join(c for c in lead["company"].lower() if c.isalnum())[:20]
            who = "".join(c for c in lead["name"].lower() if c.isalnum())[:12] or "there"
            path = os.path.join(outdir, f"cmi_{slug}_{who}.html")
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)

            fails, warns = vb.check(path, profile)
            row.update(status="FAIL" if fails else ("WARN" if warns else "PASS"),
                       secs=round(time.time() - t0), fails=fails, warns=warns)
        except Exception as e:  # noqa: BLE001 -- one bad lead must not lose the batch
            row.update(status="ERROR", secs=round(time.time() - t0),
                       fails=[f"{type(e).__name__}: {e}"], warns=[])
        return row

    t0 = time.time()
    results = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            print(f"[{i}/{len(jobs)}] {r['company'][:20]:<22}{r['kind']:<8}"
                  f"{r['status']:<6}{r['secs']:>4}s", flush=True)

    wall = time.time() - t0
    n = len(results)
    per = {k: sum(1 for r in results if r["status"] == k)
           for k in ("PASS", "WARN", "FAIL", "ERROR")}
    slowest = max(r["secs"] for r in results)

    print("\n" + "=" * 68)
    print(f"  briefs        : {n}  ({per['PASS']} PASS / {per['WARN']} WARN / "
          f"{per['FAIL']} FAIL / {per['ERROR']} ERROR)")
    print(f"  wall clock    : {wall:.0f}s = {wall / 60:.1f} min")
    print(f"  slowest brief : {slowest}s   <- the floor; batch wall cannot beat this")
    print(f"  per brief     : {wall / n:.1f}s")
    print(f"  throughput    : {n / (wall / 60):.1f} briefs/min")
    print(f"  COST          : Rs.{wall / 3600 * args.rate / n:.4f}/brief "
          f"(Rs.{args.rate}/hr)")
    print(f"  LLM calls     : {llm.STATS}")


if __name__ == "__main__":
    main()
