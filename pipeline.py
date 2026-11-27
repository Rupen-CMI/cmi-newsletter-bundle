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
import dataclasses
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

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
                       source="xlsx_import", campaign=campaign,
                       kind=kind if l["company"] else "report", commit=False)
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
def kinds_for_lead(lead: dict, kinds: list[str]) -> list[str]:
    """A lead with no company information (personal email, no company in the sheet)
    only gets a report brief on the market it enquired about -- and none at all
    without one."""
    if lead.get("company"):
        return list(kinds)
    return ["report"] if lead.get("topic") else []


_EVIDENCE_ART_FIELDS = ("title", "url", "source", "summary", "query", "bucket", "composite",
                        "relevancy", "freshness", "importance", "trigger_type",
                        "event_country", "ev_relevancy", "ev_recency", "ev_age_days",
                        "ev_reason", "topic_rel", "geo_scope", "text_source",
                        "seller_rel", "topic_sim", "news_margin", "l2_score")


def _evidence_profile(profile) -> dict:
    return {k: getattr(profile, k, None) for k in
            ("value_prop", "competitors", "industries", "trigger_queries", "industry_queries")}


def _evidence_pool(scored, keep, fallback) -> list[dict]:
    keep_ids = {id(a) for a in keep}
    fb_ids = {id(a) for a in fallback}
    out = []
    for a in scored:
        d = {k: getattr(a, k, None) for k in _EVIDENCE_ART_FIELDS}
        pub = getattr(a, "published", None)
        d["published"] = pub.isoformat() if hasattr(pub, "isoformat") else pub
        d["eligible"] = id(a) in keep_ids
        d["in_fallback"] = id(a) in fb_ids
        out.append(d)
    return out


def _evidence_used(brief: dict, debug: dict) -> dict:
    """Per section, the source article. Customer/competition come from build_brief's debug
    block; industry/market cards carry their own source_url/source_name/date."""
    used = {}
    for key, name in (("customer", "Potential Customer"), ("competition", "Competition Outlook")):
        d = (debug or {}).get(key)
        if d:
            used[name] = {"company": d.get("company"), "title": d.get("article"),
                          "url": d.get("url"), "source": d.get("source"),
                          "published": d.get("published"),
                          "free_research": d.get("free_research", "")}
    for card in (brief or {}).get("industry_outlook") or []:
        used.setdefault("Industry Outlook", []).append(
            {"name": card.get("headline"), "title": card.get("headline"),
             "url": card.get("source_url"), "source": card.get("source_name"),
             "published": card.get("date")})
    return used


def _country_check(lead: dict, stated_name: str, conn=None, log=_log) -> dict | None:
    """Compare a lead's STATED country with the HQ country its own website shows.
    Returns {"stated","website","company"} on a confident mismatch, else None. Never
    changes the country; never raises (evidence only)."""
    try:
        import enrich
        import regions
        site = lead.get("website") or ""
        domain = enrich.domain_of(website=site) if site else ""
        text = ""
        if conn is not None and domain:
            try:
                import db as _db
                cached = _db.get_intel(conn, domain)
                if cached and cached.get("raw_json"):
                    text = str(json.loads(cached["raw_json"]).get("homepage_text", ""))
            except Exception:  # noqa: BLE001
                text = ""
        hq = enrich.hq_country_from_site(text, domain)
        p = regions.get_profile(stated_name)
        stated = (p or {}).get("iso2", "")
        if hq and stated and hq != stated:
            log(f"  country check: {lead.get('company', '')} stated {stated}, website says {hq}")
            return {"stated": stated, "website": hq}
    except Exception:  # noqa: BLE001
        pass
    return None


def _write_evidence(path: str, ev: dict) -> str:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ev, f, ensure_ascii=False, indent=1, default=str)
        return path
    except Exception:  # noqa: BLE001 -- evidence must never fail a brief
        return ""


def _load_frozen_pool(fname: str):
    """Articles rebuilt from a saved evidence file in CMI_EVIDENCE_DIR, or None."""
    d = os.environ.get("CMI_EVIDENCE_DIR")
    if not d:
        return None
    p = os.path.join(d, fname)
    if not os.path.exists(p):
        return None
    from collect import Article
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for r in data.get("pool", []):
        try:
            pub = datetime.fromisoformat(r["published"])
        except Exception:  # noqa: BLE001
            pub = datetime.now(timezone.utc)
        out.append(Article(title=r.get("title") or "", url=r.get("url") or "",
                           source=r.get("source") or "", published=pub,
                           summary=r.get("summary") or "", query=r.get("query") or "",
                           bucket=r.get("bucket") or "",
                           topic_rel=r.get("topic_rel") or 0.0,
                           geo_scope=r.get("geo_scope") or "",
                           text_source=r.get("text_source") or ""))
    return out


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
    from score import eligible, fallback_pool, is_unsafe_to_cite, score_all
    from tenant import ProfileCache

    profile_cache = ProfileCache()
    os.makedirs(outdir, exist_ok=True)
    if os.environ.get("CMI_EVIDENCE_DIR"):
        _log(f"  frozen-evidence mode: reading pools from {os.environ['CMI_EVIDENCE_DIR']}")
    jobs = [(l, k) for l in leads for k in kinds_for_lead(l, kinds)]
    _log(f"  {len(jobs)} briefs ({len(leads)} leads x {len(kinds)} kind(s)), "
         f"{workers} workers")
    n_no_co = sum(1 for l in leads if not l["company"])
    if n_no_co:
        _log(f"  {n_no_co} lead(s) with no company information: report brief only")

    # The leads table is authoritative for each lead's country (country-first
    # evidence filtering). One upfront query keyed by lowercased email, read here
    # in the main thread so workers never touch the shared connection. A leads
    # table without the country columns (not yet migrated) keeps the old
    # behaviour: fall back to the sheet's own country value.
    lead_country_map: dict[str, tuple[str, str]] = {}
    try:
        for r in conn.execute(
                "SELECT email, country, country_source FROM leads").fetchall():
            lead_country_map[(r["email"] or "").strip().lower()] = (
                (r["country"] or "").strip(), (r["country_source"] or "").strip())
    except sqlite3.OperationalError:
        pass

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
        row = {"email": lead["email"], "company": lead["company"], "kind": kind,
               "country": ""}
        slug = "".join(c for c in (lead["company"] or lead.get("topic", "")).lower()
                       if c.isalnum())[:20]
        who = "".join(c for c in lead["name"].lower() if c.isalnum())[:12] or "there"
        path = os.path.join(outdir, f"cmi_{slug}_{who}_{kind}.html")
        ev_path = path[:-len(".html")] + ".evidence.json"
        ev = {"lead": {"email": lead["email"], "company": lead["company"],
                       "name": lead["name"], "kind": kind,
                       "topic": lead.get("topic", "")},
              "build_log": [], "compose_log": []}
        try:
            # DB country wins over the sheet's own value; an unresolved DB country
            # means there is no defensible place for this brief — skip it rather
            # than generate a Global (unfiltered) brief.
            db_country, db_source = lead_country_map.get(
                lead["email"].strip().lower(), ("", ""))
            if db_source == "unresolved":
                row.update(html="", topic="", path="", profile=None,
                           secs=round(time.time() - t0, 1),
                           error="country unresolved")
                return row
            country_name = db_country if (db_country and db_source != "unresolved") \
                else lead["country"]
            country = config.get_country(country_name)
            row["country"] = country["name"]
            ev["lead"]["country"] = country["name"]
            if db_source == "stated" and db_country:
                _cc = _country_check(lead, db_country, conn=db.get_conn())
                if _cc:
                    ev["lead"]["country_check"] = _cc
            # get_conn() is safe to call from a worker thread -- it hands back a
            # connection cached per-thread (see db.py), never the main thread's
            # connection object. Passed through so from_company can reuse the
            # company_intel homepage cache stage_enrich already populated, instead of
            # re-fetching the same homepage live on every brief.
            worker_conn = db.get_conn()
            # One Profile per lead per run (and persisted in profile_cache): the signal
            # and report briefs of a lead must describe the same business.
            profile = profile_cache.get(lead["company"], lead["website"],
                                        lead["name"], country["name"], conn=worker_conn,
                                        topic=lead.get("topic", ""))
            frozen = _load_frozen_pool(os.path.basename(ev_path))
            ev["profile"] = _evidence_profile(profile)
            arts = frozen if frozen is not None else collect(profile, country, conn=worker_conn)
            try:
                import collect as _collect_mod
                # LAST_HARVESTED is module-global: read right after this lead's collect()
                # (another worker can still overwrite it in between -- diagnostics only).
                ev["searches"] = {
                    "planned": _collect_mod.plan_searches(profile, country),
                    "harvested": list(getattr(_collect_mod, "LAST_HARVESTED", []))}
            except Exception:  # noqa: BLE001 -- evidence only
                pass
            # A report brief is about the market the lead enquired about, not just the
            # seller's own news. Without this, `market_topic` only re-LABELS whatever
            # the company queries returned: a Bharat Forge brief on the global drone
            # market opened with electric-truck news and said outright that the signal
            # named no drone company. Collect for the topic itself as well.
            topic_q = lead.get("topic", "") if kind == "report" else ""
            topic_arts = []
            if topic_q and frozen is None:
                import collect as _collect_t
                from collect import collect_topic

                try:
                    # Local-language search of the topic: English news misses most of a
                    # niche market's coverage in non-English countries.
                    local_topic = ""
                    if profile.local_language:
                        local_topic = _collect_t.translate_topic(
                            _collect_t.topic_subject(topic_q, _collect_t._lead_iso2(country)),
                            profile.local_language)
                    topic_arts = collect_topic(topic_q, country, local_topic=local_topic)
                except Exception:  # noqa: BLE001 -- topic news is a bonus, not required
                    topic_arts = []
            # Country-first evidence filter, company and topic articles SEPARATELY: keep
            # the lead's own country, drop pure other-country evidence (explicit-link
            # rule), keep no-geo industry articles, and only fall back to same-REGION
            # countries when trigger+competitor evidence runs dry. Topic news widens
            # country -> region -> global (on-topic only), see geo.filter_topic_for_lead.
            # A Global lead is unfiltered, as before.
            # Topic fit (keyword + embedding) must exist before the topic geo filter reads it.
            if topic_arts:
                import score as _score_t
                _score_t.annotate_topic(topic_arts, topic_q, profile)
            if not geo.is_global(country) and frozen is None:
                arts, _geo_scope = geo.filter_for_lead(arts, country, profile=profile,
                                                       conn=worker_conn)
                if topic_arts:
                    import score as _score
                    topic_arts, ev["topic_scope"] = geo.filter_topic_for_lead(
                        topic_arts, country,
                        _score.topic_tokens(topic_q, _score.topic_drop_words(profile.market)),
                        profile=profile, conn=worker_conn)
            arts = arts + topic_arts
            # Layer 1: drop what is always junk (source/headline rules), before scoring.
            import filters
            arts, ev["l1_dropped"] = filters.apply_l1(arts)
            scored = score_all(arts, profile, topic=topic_q)
            # Real article text for the top-ranked items (feed summaries are headlines).
            # Skipped in frozen replay so replay stays prompt-only.
            if config.FETCH_ARTICLE_TEXT and frozen is None:
                try:
                    from collect import enrich_texts
                    ev["texts_fetched"] = enrich_texts(scored)
                except Exception:  # noqa: BLE001 -- never fail the brief
                    ev["texts_fetched"] = 0
            keep = eligible(scored)
            topic = lead.get("topic", "") if kind == "report" else ""
            if topic:
                # The seller-relevance gate must not drop the on-topic evidence the topic
                # search was run to find.
                import score as _score
                _kept = {id(a) for a in keep}
                keep = keep + [a for a in scored if _score.topical_evidence(a)
                               and id(a) not in _kept]
            # `keep or scored`: a thin-coverage lead can score everything below
            # min_composite, and falling back to the whole unfiltered pool rather than
            # an empty brief is the right call for a low composite score -- but
            # is_research_pr/ticker-noise are not "eligible, just ranked low", they are
            # never safe to put in front of a reader (a rival firm's own report PR, or
            # stock-ticker churn). Filter those out of the fallback pool too, so a thin
            # lead can't silently reintroduce exactly what eligible() just removed.
            # It must also clear the relevance gate's absolute floor (score.fallback_pool),
            # or unrelated news the gate removed comes straight back through here.
            fallback = fallback_pool(scored)
            ev["pool"] = _evidence_pool(scored, keep, fallback)
            ev["passed_to_builder"] = "keep" if keep else "fallback"
            brief = build_brief(keep or fallback, profile,
                                log=lambda *a: ev["build_log"].append(" ".join(map(str, a))),
                                market_topic=topic)
            debug = brief[1] if isinstance(brief, tuple) and len(brief) > 1 else {}
            brief = brief[0] if isinstance(brief, tuple) else brief
            ev["used"] = _evidence_used(brief, debug)
            ctx = compose.build_context(profile, brief, kind, market_topic=topic,
                                        log=lambda *a: ev["compose_log"].append(" ".join(map(str, a))))
            # Nothing on-topic and no buyer: an empty brief ("covers 0 developments") must
            # not be rendered or sent -- record it as an error so the lead is skipped.
            if not ctx.get("sections"):
                raise ValueError("no usable content: no section passed the evidence gates")
            html = render.render(ctx, "cmi_brief.html")

            with open(path, "w", encoding="utf-8") as f:
                f.write(html)

            row.update(html=html, topic=topic, path=path, profile=profile,
                       secs=round(time.time() - t0, 1), error="")
        except Exception as e:  # noqa: BLE001 -- one bad lead must not lose the batch
            row.update(html="", topic="", path="", profile=None,
                       secs=round(time.time() - t0, 1), error=f"{type(e).__name__}: {e}")
        finally:
            if "pool" in ev:
                ev["error"] = row.get("error", "")
                row["evidence_path"] = _write_evidence(ev_path, ev)
        return row

    results = []
    ex = cf.ThreadPoolExecutor(max_workers=workers)
    try:
        futs = {ex.submit(one, j): j for j in jobs}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            # Persist the profile that produced this brief so a later validation pass
            # (even in a separate process/run) can reload it with zero extra LLM
            # calls -- re-deriving competitors/value_prop via from_company() again is
            # the exact duplicate LLM spend this is meant to avoid.
            profile_json = json.dumps(dataclasses.asdict(r["profile"]), ensure_ascii=False) \
                if r["profile"] is not None else ""
            # Store immediately: a crash later should not cost the briefs already made.
            r["brief_id"] = db.save_brief(
                conn, email=r["email"], company=r["company"], kind=r["kind"],
                topic=r["topic"], html=r["html"],
                status="failed" if r["error"] else "generated",
                validation={"fails": [r["error"]]} if r["error"] else None,
                gen_seconds=r["secs"], run_id=run_id, profile_json=profile_json)
            results.append(r)
            flag = "ERROR" if r["error"] else "ok"
            _log(f"  [{i}/{len(jobs)}] {r['company'][:20]:<22}{r['kind']:<8}"
                 f"{flag:<6}{r['secs']:>6}s {r.get('country', '') or '-'}")
            if r["error"]:
                _log(f"        {r['error']}")

            is_llm_error = r["error"].startswith("RuntimeError") and "LLM provider" in r["error"]
            if is_llm_error:
                llm_failure_streak += 1
            else:
                llm_failure_streak = 0

            if llm_failure_streak >= _LLM_ABORT_THRESHOLD and not aborted_for_llm_outage:
                aborted_for_llm_outage = True
                _log(f"\n{'!' * 64}")
                _log(f"  ABORTING: {llm_failure_streak} consecutive LLM failures -- "
                     f"the endpoint is down, not a handful of bad leads. Stopping "
                     f"immediately: cancelling every unstarted job and NOT waiting on "
                     f"jobs already in flight (their eventual result, if any, is "
                     f"discarded rather than delaying the abort).")
                _log(f"{'!' * 64}")
                break
    finally:
        # cancel_futures drops every not-yet-started job; wait=False returns immediately
        # instead of blocking on threads already mid-call to a dead/overloaded endpoint --
        # those threads keep running in the background (Python cannot force-kill a
        # thread) but the function returns now rather than after every one times out.
        ex.shutdown(wait=False, cancel_futures=True)

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
        # expected_subject independent of r["profile"]: a signal brief's subject is the
        # lead's own company (r["company"], straight from the leads table); a report
        # brief's is the market/report name the lead enquired about (r["topic"]). Never
        # profile.company -- that is exactly the field a bad profile gets wrong (see
        # tenant.Profile.from_company's docstring).
        expected_subject = r.get("topic") if r.get("kind") == "report" else r.get("company")
        try:
            fails, warns, llm_fails, llm_warns = vb.check_detailed(
                r["path"], r["profile"], email=r.get("email", ""),
                kind=r.get("kind", ""), topic=expected_subject or "",
                lead_country=r.get("country") or None)
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

    # SendGrid's own suppression lists are the only unsubscribe signal that actually
    # works today -- the template's unsubscribe link is cosmetic (points at a generic
    # contact page, not a real endpoint), so nothing ever populates `unsubscribes`
    # locally any other way. Fetched once up front; db.is_unsubscribed() (already
    # checked per-send below) then sees it like any other local suppression.
    api_key = mailer._env("SENDGRID_WEB_API")
    remote_suppressed = mailer.fetch_sendgrid_suppressed(api_key)
    if remote_suppressed:
        _log(f"  {len(remote_suppressed)} address(es) on SendGrid's suppression list")
        for email in remote_suppressed:
            db.mark_unsubscribed(conn, email, reason="SendGrid suppression list")

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
            msg_id = mailer.send_html(full["html"], email, subject, filename_hint=email, conn=conn)
            db.record_send(conn, email=email, status="sent", subject=subject,
                           campaign=campaign, message_id=msg_id)
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
            # Fetched ONCE up front (see above), not per batch: with 30k+ records
            # across 3 paginated endpoints, a per-batch re-fetch was roughly
            # doubling the real wall-clock time of a large run for marginal
            # protection SendGrid already provides for free -- a genuinely
            # suppressed address still gets blocked at SendGrid's own send-time
            # enforcement regardless of what our local cache knows (see mailer.
            # send_html's underlying API call). The only cost of NOT re-fetching
            # mid-run is a `sends` row that says "sent" when the real outcome was
            # an immediate suppression bounce -- acceptable, since reconciling
            # delivery outcomes after the fact (not pre-empting them) is the
            # correct job of a separate post-send report, not this loop.
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
    ap.add_argument("--model", default="cmi9b")
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
