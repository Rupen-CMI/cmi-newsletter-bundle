"""One command from a lead sheet to sent newsletters.

    python pipeline.py --xlsx Book1.xlsx --campaign oct1 --kind report \
        --workers 40 --base-url https://<host>/v1
    python pipeline.py --campaign oct1 --send          # send what was generated
    python pipeline.py --xlsx Book1.xlsx --stage enrich   # just fill the cache

Five stages, each resumable because every result lands in `cmi_newsletter.db` rather
than in memory:

    import    xlsx -> leads             (skips rows already present)
    enrich    domains -> company_intel  (once per DOMAIN, 90-day cache)
    generate  leads -> briefs           (all briefs in ONE concurrency window)
    validate  briefs.validation         (keyword rules + optional LLM judge)
    send      briefs -> sends           (only briefs with no hard failure)

Sending is deliberately NOT part of the default run. It is the one irreversible stage,
so it needs `--send` said out loud; everything before it can be re-run freely.

Generation puts every brief in one pool rather than one batch per kind. The ~11 LLM
calls inside a brief are sequential (call N+1 needs call N's output) but briefs are
independent, so the wall clock is bounded by the slowest single brief, not by the total
work -- measured Rs.0.286 -> Rs.0.102 per brief from that change alone, with 40 workers
the sweet spot (64 is worse: requests queue instead of batching).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

STAGES = ("import", "enrich", "generate", "validate", "send")


def _log(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(str(msg).encode(enc, "replace").decode(enc, "replace"), flush=True)


# ── stage 1: import ───────────────────────────────────────────────────────────
def stage_import(conn, xlsx: str, campaign: str, kind: str, limit: int) -> list[dict]:
    """xlsx -> leads table. Returns the lead dicts the later stages work on."""
    import db

    sys.argv = ["x"]
    import batch_generate as bg

    leads = bg.read_leads(xlsx, limit)

    # One query for every already-known email rather than one SELECT per row (this
    # used to be N SELECTs + N individually-committed upserts for N leads -- for a
    # 400-row sheet, 400 disk-flushing commits where one does the same job). The
    # existing set is only needed for the "added" count in the log line below, not for
    # any actual import decision, so a single upfront query covers it.
    existing = {row["email"] for row in
               conn.execute("SELECT email FROM leads").fetchall()}
    added = 0
    for l in leads:
        db.upsert_lead(conn, email=l["email"], name=l["name"], company=l["company"],
                       market=l["country"], report_name=l.get("topic", ""),
                       source="xlsx_import", campaign=campaign, kind=kind, commit=False)
        if l["email"].strip().lower() not in existing:
            added += 1
    conn.commit()
    _log(f"  {len(leads)} rows read, {added} new leads, {len(leads) - added} already known")
    return leads


# ── stage 2: enrich ───────────────────────────────────────────────────────────
def stage_enrich(conn, leads: list[dict], workers: int, force: bool) -> dict:
    import enrich

    summary = enrich.enrich_leads(conn, leads, workers=min(workers, 12),
                                  force=force, log=_log)
    if summary.get("thin"):
        _log(f"  {summary['thin']} domains gave too little — see: python enrich.py --thin")
    return summary


# ── stage 3: generate ─────────────────────────────────────────────────────────
def stage_generate(conn, leads: list[dict], *, kinds: list[str], workers: int,
                   run_id: int, outdir: str) -> list[dict]:
    """Every (lead, kind) pair in one pool. Each brief is stored as it finishes, so a
    crash halfway leaves the completed ones in the DB rather than losing the run."""
    import compose
    import config
    import db
    import geo
    import llm
    import render
    from brief_builder import build_brief
    from collect import collect
    from score import eligible, score_all
    from tenant import Profile

    os.makedirs(outdir, exist_ok=True)
    jobs = [(l, k) for l in leads for k in kinds]
    _log(f"  {len(jobs)} briefs ({len(leads)} leads x {len(kinds)} kind(s)), "
         f"{workers} workers")

    # A dead LLM endpoint used to degrade silently into hollow-but-valid-looking
    # briefs (see llm.chat_json). Now every LLM call failure raises immediately, so
    # an outage shows up here as many workers hitting "LLM provider ... failed" at
    # once. That must stop the run rather than grind through the rest of the batch
    # repeating the same failure -- track a consecutive-failure streak across
    # completions and abort once it is unambiguously an outage, not a one-off bad lead.
    _LLM_ABORT_THRESHOLD = 5
    llm_failure_streak = 0
    aborted_for_llm_outage = False

    def one(job):
        lead, kind = job
        t0 = time.time()
        row = {"email": lead["email"], "company": lead["company"], "kind": kind}
        try:
            country = config.get_country(lead["country"])
            # get_conn() is safe to call from a worker thread -- it hands back a
            # connection cached per-thread (see db.py), never the main thread's
            # connection object. Passed through so from_company can reuse the
            # company_intel homepage cache stage_enrich already populated, instead of
            # re-fetching the same homepage live on every brief.
            worker_conn = db.get_conn()
            profile = Profile.from_company(lead["company"], lead["website"],
                                           lead["name"], country["name"], conn=worker_conn)
            arts = collect(profile, country)
            # A report brief is about the market the lead enquired about, not just the
            # seller's own news. Without this, `market_topic` only re-LABELS whatever
            # the company queries returned: a Bharat Forge brief on the global drone
            # market opened with electric-truck news and said outright that the signal
            # named no drone company. Collect for the topic itself as well.
            topic_q = lead.get("topic", "") if kind == "report" else ""
            if topic_q:
                from collect import collect_queries

                base = topic_q.replace(" Market", "").replace(" market", "").strip()
                qs = [topic_q, f"{base} market growth", f"{base} companies",
                      f"{base} industry India"]
                try:
                    arts = arts + collect_queries(qs, country, bucket="industry")
                except Exception:  # noqa: BLE001 -- topic news is a bonus, not required
                    pass
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
            path = os.path.join(outdir, f"cmi_{slug}_{who}_{kind}.html")
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)

            row.update(html=html, topic=topic, path=path, profile=profile,
                       secs=round(time.time() - t0, 1), error="")
        except Exception as e:  # noqa: BLE001 -- one bad lead must not lose the batch
            row.update(html="", topic="", path="", profile=None,
                       secs=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}")
        return row

    results = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            # Store immediately: a crash later should not cost the briefs already made.
            r["brief_id"] = db.save_brief(
                conn, email=r["email"], company=r["company"], kind=r["kind"],
                topic=r["topic"], html=r["html"],
                status="failed" if r["error"] else "generated",
                gen_seconds=r["secs"], run_id=run_id)
            results.append(r)
            flag = "ERROR" if r["error"] else "ok"
            _log(f"  [{i}/{len(jobs)}] {r['company'][:20]:<22}{r['kind']:<8}"
                 f"{flag:<6}{r['secs']:>6}s")
            if r["error"]:
                _log(f"        {r['error'][:90]}")

            is_llm_error = r["error"].startswith("RuntimeError") and "LLM provider" in r["error"]
            if is_llm_error:
                llm_failure_streak += 1
            else:
                llm_failure_streak = 0

            if llm_failure_streak >= _LLM_ABORT_THRESHOLD and not aborted_for_llm_outage:
                aborted_for_llm_outage = True
                _log(f"\n{'!' * 64}")
                _log(f"  ABORTING: {llm_failure_streak} consecutive LLM failures -- "
                     f"the endpoint is down, not a handful of bad leads. Cancelling "
                     f"remaining unstarted jobs; briefs already in flight will still "
                     f"finish (or fail) and be recorded.")
                _log(f"{'!' * 64}")
                for pending_fut in futs:
                    pending_fut.cancel()

    if aborted_for_llm_outage:
        raise RuntimeError(
            f"generation aborted: {_LLM_ABORT_THRESHOLD}+ consecutive LLM failures "
            f"(endpoint down). {len(results)}/{len(jobs)} jobs completed before abort.")
    return results


# ── stage 4: validate ─────────────────────────────────────────────────────────
def stage_validate(conn, results: list[dict]) -> dict:
    """Run the gate over each generated brief and write the verdict onto its row."""
    import db
    import validate_brief as vb

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "ERROR": 0}
    flagged = 0  # briefs the LLM judge added a finding to (rule-clean otherwise)
    for r in results:
        if r.get("error") or not r.get("path"):
            counts["ERROR"] += 1
            continue
        try:
            fails, warns, llm_fails, llm_warns = vb.check_detailed(r["path"], r["profile"])
        except Exception as e:  # noqa: BLE001 -- a broken check must not fail the brief
            fails, warns, llm_fails, llm_warns = [], [f"validator error: {type(e).__name__}: {e}"], [], []
        # Text-rule status first -- this is what actually gates sending, unchanged from
        # before. The LLM judge only ever runs on what would already PASS/WARN (see
        # validate_brief.check_detailed), so its findings layer on top as `llm_flagged`
        # rather than changing FAIL/WARN/PASS themselves: a brief a human should look at
        # is not the same thing as a brief that is blocked from sending.
        status = "FAIL" if fails else ("WARN" if warns else "PASS")
        counts[status] += 1
        llm_flagged = bool(llm_fails or llm_warns)
        if llm_flagged:
            flagged += 1
        conn.execute("UPDATE briefs SET validation = ?, status = ? WHERE id = ?",
                     (json.dumps({"fails": fails, "warns": warns,
                                  "llm_fails": llm_fails, "llm_warns": llm_warns,
                                  "llm_flagged": llm_flagged}, ensure_ascii=False),
                      "failed" if fails else "validated", r["brief_id"]))
        tag = f"{status}+LLM" if llm_flagged else status
        _log(f"  [{tag}] {r['company'][:22]:<24}{r['kind']}")
        for f in fails:
            _log(f"        x {f[:100]}")
        for w in warns[:2]:
            _log(f"        ! {w[:100]}")
        for f in llm_fails:
            _log(f"        ?x [LLM] {r['email']}: {f[:100]}")
        for w in llm_warns:
            _log(f"        ?! [LLM] {r['email']}: {w[:100]}")
    conn.commit()
    if flagged:
        _log(f"\n  {flagged} brief(s) flagged by the LLM judge for human review "
             f"(see: python flagged_briefs.py)")
    return counts


# ── stage 5: send ─────────────────────────────────────────────────────────────
def stage_send(conn, run_id: int, campaign: str, *, dry_run: bool, limit: int,
               brief_ids: list[int] | None = None) -> dict:
    """Email the briefs that carry no hard failure.

    Reads the HTML back out of `briefs` rather than regenerating: what gets sent is
    exactly the document that was validated.

    `brief_ids` sends exactly those briefs and ignores `limit`. Callers that show the
    user a list beforehand must use it: `limit` slices the sendable set BEFORE the
    unsubscribe check, so an unsubscribed brief inside the slice silently consumes a
    slot -- a UI promising "send 2" then delivering 1.

    PACING AND TRIPWIRES -- ported from the sibling newsletter-cc-cl project's
    campaign.py, which already sends real batches of email and has been tuned
    against real bounce/throttle incidents there. See sendpacing.py for the full
    reasoning and every tunable. In short:

      - sends go out in small BATCHES (default 10), with a short random delay
        between individual sends inside a batch and a longer random pause between
        batches, so hundreds of sends do not fire in one uninterrupted burst.
      - a provider fault that will repeat for every remaining recipient -- a 429
        rate limit, an expired/invalid key, insufficient balance -- ABORTS THE
        WHOLE RUN immediately rather than grinding through the rest at full speed.
        Retrying or continuing at the same pace after a 429 specifically risks
        turning a temporary throttle into a longer block.
      - an ordinary consecutive-failure streak (bad addresses, a dead relay) that
        does not match a known fatal pattern still aborts after N in a row
        (default 3) rather than burning through hundreds of doomed sends one at a
        time -- the counter resets on any success.

    Returns `sent`/`failed`/`skipped` as before, plus `aborted` (the reason string,
    or "" if the run completed normally) so a caller can tell "finished" apart from
    "stopped early and the rest still need sending".
    """
    import db
    import mailer
    import sendpacing as sp

    rows = db.sendable_briefs(conn, run_id=run_id)
    if brief_ids is not None:
        wanted = set(brief_ids)
        rows = [r for r in rows if r["id"] in wanted]
    else:
        rows = rows[:limit]
    _log(f"  {len(rows)} briefs clear to send")

    size = sp.batch_size()
    short_range = sp.short_delay_range()
    long_range = sp.long_pause_range()
    max_consec = sp.max_consecutive_failures()
    batches = sp.chunk(rows, size)
    _log(f"  {len(batches)} batch(es) of up to {size}, "
         f"{short_range[0]:.0f}-{short_range[1]:.0f}s between sends, "
         f"{long_range[0]:.0f}-{long_range[1]:.0f}s between batches")

    sent = failed = skipped = 0
    consecutive_failures = 0
    aborted = ""
    total = len(rows)

    def _one(r, position: int) -> str:
        """Send one brief. Returns 'sent' | 'skipped' | 'failed'."""
        nonlocal sent, failed, skipped
        email = r["email"]
        progress = f"[{position}/{total}]"
        if db.is_unsubscribed(conn, email):
            _log(f"  {progress} skip  {email} (unsubscribed)")
            skipped += 1
            return "skipped"
        full = db.get_brief(conn, r["id"])
        subject = db.brief_subject(r["company"], r["kind"], r["topic"])
        if dry_run:
            _log(f"  {progress} DRY   {email:<34}{subject[:44]}")
            skipped += 1
            return "skipped"
        try:
            mailer.send_html(full["html"], email, subject, filename_hint=email, conn=conn)
            db.record_send(conn, email=email, status="sent", subject=subject,
                           campaign=campaign)
            db.set_brief_status(conn, r["id"], "sent")
            db.set_lead_status(conn, email, "sent")
            _log(f"  {progress} sent  {email}")
            sent += 1
            return "sent"
        except Exception as e:  # noqa: BLE001 -- one bad address must not stop the batch
            detail = f"{type(e).__name__}: {e}"
            db.record_send(conn, email=email, status="failed", detail=detail, campaign=campaign)
            db.set_lead_status(conn, email, "failed")
            _log(f"  {progress} FAIL  {email}: {detail[:70]}")
            failed += 1
            raise RuntimeError(detail) from e

    try:
        position = 0
        for batch_idx, batch in enumerate(batches, start=1):
            batch_did_work = False
            _log(f"  -- batch {batch_idx}/{len(batches)} --")
            for i, r in enumerate(batch, start=1):
                position += 1
                try:
                    outcome = _one(r, position)
                    if outcome == "sent":
                        batch_did_work = True
                        consecutive_failures = 0
                    # "skipped" (unsubscribed / dry-run) makes no network call, so it
                    # neither counts as a failure nor as work needing to be paced.
                except RuntimeError as e:
                    batch_did_work = True  # a real send attempt was made and rejected
                    detail = str(e)
                    # Two checks: by TYPE first (e.__cause__ is the original mailer
                    # exception, preserved via `raise ... from e` in _one() above --
                    # this catches an SMTP fault whose wording doesn't match any
                    # string below, e.g. SMTPAuthenticationError regardless of what
                    # the server's text says), then by TEXT as a fallback (covers
                    # SendGrid, which embeds its status as a literal string).
                    if sp.is_fatal_exception(e.__cause__) or sp.is_fatal_provider_error(detail):
                        raise sp.SendAborted(
                            f"provider error that will not clear during this run: "
                            f"{detail[:300]}") from e
                    consecutive_failures += 1
                    if consecutive_failures >= max_consec:
                        raise sp.SendAborted(
                            f"{consecutive_failures} consecutive send failures -- "
                            f"stopping rather than burning through the remaining "
                            f"{sum(len(b) for b in batches[batch_idx - 1:]) - i} recipient(s). "
                            f"Last error: {detail[:250]}") from e

                # Short randomized delay between sends WITHIN a batch, skipped after
                # the last one and while pacing is disabled (tests / dry-run checks
                # that don't want to wait). Applied even in --dry-run-send so pacing
                # itself can be verified without a real send.
                if i < len(batch) and not sp.pacing_disabled():
                    import random
                    time.sleep(random.uniform(*short_range))

            # Long pause between batches -- skipped when this batch made no real send
            # attempts (everyone in it was unsubscribed or dry-run), since there is
            # nothing to space out, and skipped on the last batch (nothing follows it).
            if batch_idx < len(batches) and batch_did_work and not sp.pacing_disabled():
                import random
                p = random.uniform(*long_range)
                _log(f"  ...pausing {p:.0f}s before batch {batch_idx + 1}/{len(batches)}")
                time.sleep(p)
    except sp.SendAborted as e:
        aborted = str(e)
        _log("\n" + "!" * 64)
        _log(f"  SEND STOPPED: {aborted}")
        _log(f"  {sent} sent, {failed} failed, {skipped} skipped before stopping.")
        _log("  Every attempted send is recorded; re-run stage_send once the cause "
             "is fixed to pick up where this left off.")
        _log("!" * 64)

    return {"sent": sent, "failed": failed, "skipped": skipped, "aborted": aborted}


def main() -> None:
    ap = argparse.ArgumentParser(description="Lead sheet -> newsletters, end to end.")
    ap.add_argument("--xlsx", help="lead sheet; omit to work on leads already in the DB")
    ap.add_argument("--campaign", default="default")
    ap.add_argument("--kind", choices=["signal", "report", "both"], default="report")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--workers", type=int, default=40,
                    help="measured sweet spot is 40; 64 is worse (requests queue)")
    ap.add_argument("--base-url", default=os.environ.get("LOCAL_LLM_BASE_URL", ""))
    ap.add_argument("--no-autoconnect", action="store_true",
                    help="do not resume/heal the VM; fail if the endpoint is down")
    ap.add_argument("--spot", action="store_true",
                    help="resume onto spot pricing when autoconnecting")
    ap.add_argument("--model", default="cmi-7b")
    ap.add_argument("--outdir", default="output_pipeline")
    ap.add_argument("--stage", choices=STAGES, help="run only this stage")
    ap.add_argument("--send", action="store_true",
                    help="actually email; without it the run stops after validate")
    ap.add_argument("--dry-run-send", action="store_true",
                    help="with --send: print what would go out, send nothing")
    ap.add_argument("--force-enrich", action="store_true")
    ap.add_argument("--rate", type=float, default=41.31, help="GPU Rs/hr for the cost line")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    os.environ["LLM_PROVIDER"] = "local"
    if args.base_url:
        os.environ["LOCAL_LLM_BASE_URL"] = args.base_url
    os.environ["LOCAL_LLM_MODEL"] = args.model

    # Must be set before the autoconnect check below, which reads it -- this was
    # previously assigned after that check ran, raising UnboundLocalError on every
    # invocation regardless of --stage (including the plain documented command with
    # no --stage at all, since `only` is a local variable the moment ANY assignment
    # to it exists later in the function).
    only = args.stage

    # Confirm the endpoint actually completes BEFORE generating, and heal it if not.
    # The machine id changes on every resume and a spot box can be reclaimed, so a
    # URL in .env is routinely stale. Skipping this check once cost a full run: four
    # briefs generated against a dead endpoint, every LLM call 404ing, and the batch
    # still reported PASS because empty sections are structurally valid.
    if not args.no_autoconnect and (only in (None, "generate")):
        import vm_connect

        if not vm_connect.is_serving(os.environ.get("LOCAL_LLM_BASE_URL", "")
                                     or vm_connect.current_base_url()):
            _log("[0/5] endpoint not answering — connecting")
            st = vm_connect.ensure(spot=args.spot, log=lambda m: _log("  " + m))
            if not st.get("serving"):
                raise SystemExit(f"no model endpoint: {st.get('error') or 'unknown'}")
            os.environ["LOCAL_LLM_BASE_URL"] = st["endpoint"]

    import db
    import llm

    conn = db.get_conn()
    db.init_db(conn)
    kinds = ["signal", "report"] if args.kind == "both" else [args.kind]

    _log(f"campaign={args.campaign} kind={args.kind} workers={args.workers}")
    _log(f"endpoint={llm.LOCAL_BASE_URL}\n")

    # ---- import
    leads: list[dict] = []
    if args.xlsx and (only in (None, "import") or only in ("enrich", "generate")):
        _log("[1/5] import")
        leads = stage_import(conn, args.xlsx, args.campaign, args.kind, args.limit)
    if only == "import":
        return

    if not leads and only in (None, "enrich", "generate"):
        _log("no --xlsx given; nothing to enrich or generate")
        return

    # ---- enrich
    if only in (None, "enrich"):
        _log("\n[2/5] enrich")
        stage_enrich(conn, leads, args.workers, args.force_enrich)
    if only == "enrich":
        return

    # `--stage send` means "send what a PRIOR run already generated and validated" --
    # there is nothing to generate here, and creating a fresh run_id below would leave
    # stage_send looking at an empty run (run_id filters sendable_briefs to just that
    # run). Go straight to send, across every prior run's sendable briefs, and skip
    # generate/validate entirely.
    if only == "send":
        if not args.send:
            raise SystemExit("--stage send requires --send (that is the point of this "
                             "stage: send whatever already passed validation)")
        _log("\n[5/5] send (all prior runs)")
        s = stage_send(conn, run_id=None, campaign=args.campaign,
                       dry_run=args.dry_run_send, limit=args.limit * len(kinds))
        _log(f"\n  sent {s['sent']} | failed {s['failed']} | skipped {s['skipped']}")
        if s.get("aborted"):
            raise SystemExit(f"send stopped early: {s['aborted']}")
        return

    # ---- generate
    run_id = db.start_run(conn, workers=args.workers, kind=args.kind,
                          endpoint=llm.LOCAL_BASE_URL, notes=args.campaign)
    t0 = time.time()
    _log(f"\n[3/5] generate  (run {run_id})")
    results = stage_generate(conn, leads, kinds=kinds, workers=args.workers,
                             run_id=run_id, outdir=args.outdir)
    wall = time.time() - t0

    # ---- validate
    _log("\n[4/5] validate")
    counts = stage_validate(conn, results)

    n = max(1, len(results))
    cost = wall / 3600 * args.rate / n
    db.finish_run(conn, run_id, n_briefs=len(results), n_pass=counts["PASS"],
                  n_warn=counts["WARN"], n_fail=counts["FAIL"] + counts["ERROR"],
                  wall_secs=wall, cost_inr=cost)

    _log("\n" + "=" * 64)
    _log(f"  run {run_id}: {len(results)} briefs  "
         f"{counts['PASS']} PASS / {counts['WARN']} WARN / "
         f"{counts['FAIL']} FAIL / {counts['ERROR']} ERROR")
    _log(f"  wall {wall:.0f}s  ({wall / n:.1f}s each)   Rs.{cost:.4f}/brief")
    _log(f"  LLM  {llm.STATS}")

    if only == "validate":
        return

    # ---- send
    if not args.send:
        _log(f"\n  not sending (no --send). Briefs are in the DB and {args.outdir}/")
        return
    _log("\n[5/5] send")
    s = stage_send(conn, run_id, args.campaign, dry_run=args.dry_run_send,
                   limit=args.limit * len(kinds))
    _log(f"\n  sent {s['sent']} | failed {s['failed']} | skipped {s['skipped']}")
    if s.get("aborted"):
        raise SystemExit(f"send stopped early: {s['aborted']}")


if __name__ == "__main__":
    main()
