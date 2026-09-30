"""Operations dashboard: is the VM up, and what has the pipeline produced?

    streamlit run dashboard.py

Two jobs, in priority order:

  1. **Is the model reachable?** Shown at the top, always, because every other number
     on the page is history and this is the only live one. "Connected" here means a
     real chat completion came back -- not that the process is up, and not that
     /v1/models answered, which it does while the engine is still loading.

  2. **What has the pipeline done?** Briefs, sends, validation verdicts, throughput.

Reads the database directly rather than through api.py, so it works with nothing else
running. The Reconnect button is the one action here, and it is safe to press twice:
`vm_connect.ensure` does nothing when the endpoint already answers.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="CMI Operations", page_icon="📊", layout="wide")

NAVY, GREEN, AMBER, RED = "#002060", "#70AD47", "#E0A800", "#C0392B"


def _conn():
    """NOT cached with st.cache_resource.

    That caches one object for the whole process, but Streamlit runs each rerun on a
    different thread and a sqlite3 connection may only be used on the thread that
    created it -- so the second rerun raised
    `ProgrammingError: SQLite objects created in a thread can only be used in that
    same thread`. db.get_conn already caches per thread, which is the correct scope.
    """
    import db

    c = db.get_conn()
    db.init_db(c)
    return c


def _q(sql: str, args: tuple = ()) -> list[dict]:
    return [dict(r) for r in _conn().execute(sql, args).fetchall()]


def _one(sql: str, args: tuple = ()):
    row = _conn().execute(sql, args).fetchone()
    return row[0] if row else 0


# ── VM health ─────────────────────────────────────────────────────────────────
def vm_panel() -> None:
    st.subheader("Model endpoint")

    # Not cached: the whole point of this panel is that it is live. A cached health
    # check is worse than none -- it reports "connected" after the box is reclaimed.
    with st.spinner("checking..."):
        try:
            import vm_connect

            s = vm_connect.status()
        except Exception as e:  # noqa: BLE001
            st.error(f"could not check: {type(e).__name__}: {e}")
            return

    serving = s.get("serving")
    c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
    with c1:
        if serving:
            st.markdown(f"### :green[● CONNECTED]")
        else:
            st.markdown(f"### :red[● DISCONNECTED]")
    c2.metric("VM status", s.get("vm_status") or "unknown")
    c3.metric("Machine id", s.get("machine_id") or "—")
    gpu = s.get("gpu") or "—"
    if s.get("is_spot"):
        gpu += " (spot)"
    c4.metric("GPU", gpu)

    ep = s.get("endpoint") or s.get("env_url") or ""
    if ep:
        st.caption(f"endpoint: `{ep}`")
    if s.get("error"):
        st.warning(s["error"])

    # The specific trap: .env still holds the URL from a previous machine id, so
    # generation quietly runs against a dead host and every section comes back empty.
    if ep and s.get("endpoint") and not s.get("env_matches"):
        st.warning("`.env` points at a different endpoint than the running VM. "
                   "Press **Reconnect** to record the live one.")

    b1, b2, b4, b3 = st.columns([1, 1, 1, 3])
    if b1.button("🔄 Refresh", width='stretch'):
        st.rerun()
    # Pausing belongs next to connecting: an idle L4 is Rs.27-41/hr for nothing, and
    # forgetting this is the easiest way to burn the balance.
    if b4.button("⏸ Pause VM", width='stretch',
                 disabled=s.get("vm_status") != "Running",
                 help="Stops billing. /home/merged survives; the server does not."):
        with st.spinner("pausing..."):
            import vm_connect

            r = vm_connect.pause(log=lambda m: None)
        if r.get("paused"):
            st.success(f"paused {r.get('machine_id')} — billing stopped")
            st.rerun()
        else:
            st.error(r.get("error") or "could not pause")
    if b2.button("🔌 Reconnect", type="primary", width='stretch',
                 disabled=bool(serving),
                 help="Resume the VM if needed, start the server, and record the "
                      "endpoint in .env. Safe to press twice."):
        box = st.empty()
        lines: list[str] = []

        def log(m: str) -> None:
            lines.append(m)
            box.code("\n".join(lines[-12:]))

        with st.spinner("connecting — a cold resume takes ~6 minutes"):
            import vm_connect

            out = vm_connect.ensure(spot=st.session_state.get("use_spot", False), log=log)
        if out.get("serving"):
            st.success(f"connected — {out.get('endpoint')}")
            st.rerun()
        else:
            st.error(out.get("error") or "could not connect")
    b3.checkbox("resume on spot (cheaper, can be reclaimed)", key="use_spot")


# ── analytics ─────────────────────────────────────────────────────────────────
def totals_panel() -> None:
    import db

    st.subheader("Totals")
    c = _conn()
    cols = st.columns(6)
    cols[0].metric("Leads", f"{_one('SELECT COUNT(*) FROM leads'):,}")
    cols[1].metric("Briefs", f"{_one('SELECT COUNT(*) FROM briefs'):,}")
    cols[2].metric("Sent", f"{_one('SELECT COUNT(*) FROM briefs WHERE status = ?', ('sent',)):,}")
    cols[3].metric("Clear to send", f"{len(db.sendable_briefs(c)):,}",
                   help="generated or validated, carrying no hard failure")
    cols[4].metric("Sends logged", f"{_one('SELECT COUNT(*) FROM sends'):,}")
    cols[5].metric("Unsubscribed", f"{_one('SELECT COUNT(*) FROM unsubscribes'):,}")


def quality_panel() -> None:
    import json

    st.subheader("Validation")
    verdicts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    reasons: dict[str, int] = {}
    for r in _conn().execute("SELECT validation FROM briefs WHERE validation IS NOT NULL"):
        try:
            v = json.loads(r[0] or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(v, dict):
            continue
        fails, warns = v.get("fails") or [], v.get("warns") or []
        verdicts["FAIL" if fails else ("WARN" if warns else "PASS")] += 1
        for f in fails:
            k = str(f).split("--")[0].split(":")[0][:60].strip()
            reasons[k] = reasons.get(k, 0) + 1

    total = sum(verdicts.values()) or 1
    c1, c2, c3 = st.columns(3)
    c1.metric("Pass", verdicts["PASS"], f"{verdicts['PASS']/total:.0%}")
    c2.metric("Warn", verdicts["WARN"], f"{verdicts['WARN']/total:.0%}",
              help="a judgement a human can overrule — these are still sendable")
    c3.metric("Fail", verdicts["FAIL"], f"{verdicts['FAIL']/total:.0%}",
              help="blocked from sending")

    if reasons:
        st.caption("Most common hard failures")
        st.dataframe([{"reason": k, "count": v} for k, v in
                      sorted(reasons.items(), key=lambda kv: -kv[1])[:8]],
                     width='stretch', hide_index=True)


def activity_panel() -> None:
    st.subheader("Activity")
    briefs = _q("SELECT date(created_at) AS day, COUNT(*) AS briefs FROM briefs "
                "GROUP BY day ORDER BY day")
    # NB: sends uses `sent_at`, not `created_at` like briefs.
    sends = _q("SELECT date(sent_at) AS day, COUNT(*) AS sends FROM sends "
               "GROUP BY day ORDER BY day")
    if briefs or sends:
        merged: dict[str, dict] = {}
        for r in briefs:
            merged.setdefault(r["day"], {"day": r["day"], "briefs": 0, "sends": 0})["briefs"] = r["briefs"]
        for r in sends:
            merged.setdefault(r["day"], {"day": r["day"], "briefs": 0, "sends": 0})["sends"] = r["sends"]
        data = sorted(merged.values(), key=lambda d: d["day"])
        try:
            import pandas as pd

            st.line_chart(pd.DataFrame(data).set_index("day"))
        except ImportError:
            st.dataframe(data, width='stretch', hide_index=True)
    else:
        st.info("nothing generated yet")


def runs_panel() -> None:
    st.subheader("Recent runs")
    runs = _q("SELECT id, started_at, workers, kind, n_briefs, n_pass, n_warn, n_fail, "
              "wall_secs, cost_inr FROM runs ORDER BY id DESC LIMIT 10")
    if not runs:
        st.info("no runs recorded")
        return
    for r in runs:
        # Cost is only stored when the caller measured it; derive a per-brief figure
        # from the documented Rs.0.10 rate otherwise so the column is never blank.
        if not r.get("cost_inr") and r.get("n_briefs"):
            r["cost_inr"] = round(r["n_briefs"] * 0.10, 2)
    st.dataframe(runs, width='stretch', hide_index=True)


def briefs_panel() -> None:
    st.subheader("Latest briefs")
    rows = _q("SELECT id, email, company, kind, topic, status, gen_seconds, created_at "
              "FROM briefs ORDER BY id DESC LIMIT 25")
    if not rows:
        st.info("no briefs yet")
        return
    st.dataframe(rows, width='stretch', hide_index=True)

    ids = [r["id"] for r in rows]
    pick = st.selectbox("Preview a brief", ids, format_func=lambda i: f"#{i}")
    if pick:
        import db

        row = db.get_brief(_conn(), pick)
        if row and row["html"]:
            st.components.v1.html(row["html"], height=650, scrolling=True)
        else:
            st.warning("this brief has no stored HTML")



# ── generation (batch) ────────────────────────────────────────────────────────
def _lead_rows(conn, campaign: str = "", only_unbriefed: bool = True) -> list[dict]:
    """Leads eligible for generation, newest first.

    `only_unbriefed` skips anyone who already has a brief from today -- regenerating
    the same lead on the same day is nearly always a mis-click, and it costs a GPU
    minute each time.
    """
    q = "SELECT * FROM leads"
    args: list = []
    if campaign:
        q += " WHERE campaign = ?"
        args.append(campaign)
    q += " ORDER BY id DESC LIMIT 2000"
    rows = [dict(r) for r in conn.execute(q, args).fetchall()]
    if not only_unbriefed:
        return rows
    done = {r[0] for r in conn.execute(
        "SELECT DISTINCT email FROM briefs WHERE date(created_at) = date('now')")}
    return [r for r in rows if r["email"] not in done]


def generate_panel() -> None:
    """Batch generation: choose how many newsletters to make, then make them.

    Kept separate from Send on purpose. Generating is cheap and repeatable; sending is
    neither. Bundling them behind one button is how a batch of unreviewed briefs ends
    up in real inboxes.
    """
    import db

    st.subheader("Generate")
    conn = _conn()

    campaigns = [r[0] for r in conn.execute(
        "SELECT DISTINCT campaign FROM leads WHERE campaign IS NOT NULL "
        "AND campaign != '' ORDER BY campaign")]

    c1, c2, c3 = st.columns([2, 1, 1])
    campaign = c1.selectbox("Campaign", ["(all)"] + campaigns, key="gen_campaign")
    campaign = "" if campaign == "(all)" else campaign
    kind = c2.selectbox("Kind", ["report", "signal", "both"], key="gen_kind")
    skip_done = c3.checkbox("Skip leads briefed today", value=True, key="gen_skip")

    leads = _lead_rows(conn, campaign, skip_done)
    if not leads:
        st.info("No eligible leads. Import a sheet, or untick 'skip leads briefed "
                "today' to regenerate.")
        return

    per_lead = 2 if kind == "both" else 1
    max_leads = len(leads)
    n_leads = st.slider("How many leads to process", 1, max_leads,
                        min(5, max_leads), key="gen_n")
    n_briefs = n_leads * per_lead

    # State the real unit of work: "10 leads, both kinds" is 20 briefs and 20x the cost.
    st.caption(f"**{n_briefs} newsletter(s)** — {n_leads} lead(s) x {per_lead} kind(s) "
               f"· approx Rs.{n_briefs * 0.10:.2f} at the measured Rs.0.10/brief")

    workers = st.slider("Workers", 1, 64, min(40, n_briefs), key="gen_workers",
                        help="40 is the measured sweet spot; 64 is worse because "
                             "requests queue instead of batching.")

    picked = leads[:n_leads]
    with st.expander(f"Preview the {n_leads} lead(s)"):
        st.dataframe([{"email": l["email"], "company": l["company"],
                       "topic": l["report_name"] or "—",
                       "campaign": l["campaign"] or "—"} for l in picked],
                     width='stretch', hide_index=True)

    vm_ok = st.session_state.get("_vm_serving")
    if vm_ok is False:
        st.warning("The model endpoint is not connected. Generating now would produce "
                   "empty briefs — press **Reconnect** above first.")

    if st.button(f"⚙ Generate {n_briefs} newsletter(s)", type="primary",
                 width='content'):
        box = st.empty()
        lines: list[str] = []

        def log(m: str = "") -> None:
            lines.append(str(m))
            box.code("\n".join(lines[-18:]))

        import pipeline
        import vm_connect

        # Never generate against a dead endpoint: every call 404s, the sections come
        # back empty, and the briefs still pass structural validation.
        if not vm_connect.is_serving(vm_connect.current_base_url()):
            log("endpoint not answering - connecting...")
            stt = vm_connect.ensure(log=log)
            if not stt.get("serving"):
                st.error(stt.get("error") or "could not reach the model")
                return

        jobs = [{"email": l["email"], "name": l["name"] or "there",
                 "company": l["company"] or "", "country": l["market"] or "Global",
                 "website": "", "topic": l["report_name"] or ""} for l in picked]
        kinds = ["report", "signal"] if kind == "both" else [kind]

        rid = db.start_run(conn, workers=workers, kind=kind,
                           endpoint=vm_connect.current_base_url(), notes="dashboard")
        t0 = time.time()
        prev, pipeline._log = pipeline._log, log
        try:
            with st.spinner(f"generating {n_briefs} newsletter(s) — "
                            f"roughly {max(1, n_briefs // max(1, workers)) * 4} min"):
                res = pipeline.stage_generate(conn, jobs, kinds=kinds, workers=workers,
                                              run_id=rid, outdir="output_dashboard")
                counts = pipeline.stage_validate(conn, res)
        finally:
            pipeline._log = prev

        db.finish_run(conn, rid, n_briefs=len(res), n_pass=counts["PASS"],
                      n_warn=counts["WARN"], n_fail=counts["FAIL"],
                      wall_secs=time.time() - t0, cost_inr=round(len(res) * 0.10, 2))
        ok = sum(1 for r in res if not r.get("error"))
        st.success(f"generated {ok}/{len(res)} · PASS {counts['PASS']} "
                   f"WARN {counts['WARN']} FAIL {counts['FAIL']} "
                   f"· {time.time() - t0:.0f}s")
        st.rerun()



# ── sending ───────────────────────────────────────────────────────────────────
def send_panel() -> None:
    """The one irreversible control on this page.

    Deliberately awkward to fire by accident: it shows exactly who would receive what,
    requires typing SEND to arm, and defaults to a dry run. Email cannot be recalled,
    so a single mis-click here is worse than any other mistake the dashboard allows.
    """
    import db

    st.subheader("Send")
    conn = _conn()
    rows = db.sendable_briefs(conn)

    if not rows:
        st.info("Nothing clear to send. Briefs qualify once they are generated or "
                "validated and carry no hard validation failure.")
        return

    # Resolve recipients up front: an unsubscribed address is skipped by stage_send,
    # so showing it as "will receive" would be a lie.
    unsub = [r for r in rows if db.is_unsubscribed(conn, r["email"])]
    live = [r for r in rows if not db.is_unsubscribed(conn, r["email"])]

    if not live:
        st.info(f"All {len(unsub)} sendable brief(s) are for unsubscribed addresses — "
                "nothing would go out.")
        return

    # Choose how many to send. The count on the button is this number, not the number
    # of sendable briefs: those differ whenever an unsubscribed address is in the set,
    # and a button that overstates what it will do is how surprises happen.
    n_send = st.slider("How many to send", 1, len(live), len(live), key="send_n")         if len(live) > 1 else 1
    picked = live[:n_send]

    st.caption(f"**{len(picked)} email(s)** will be sent"
               + (f" · {len(unsub)} skipped (unsubscribed)" if unsub else "")
               + (f" · {len(live) - len(picked)} held back" if n_send < len(live) else ""))
    st.dataframe(
        [{"id": r["id"], "to": r["email"], "company": r["company"],
          "kind": r["kind"],
          "subject": db.brief_subject(r["company"], r["kind"], r["topic"])}
         for r in picked],
        width='stretch', hide_index=True)

    provider = "unknown"
    try:
        import mailer

        provider = mailer._provider()
    except Exception:  # noqa: BLE001
        pass
    st.caption(f"transport: **{provider}**")

    dry = st.checkbox("Dry run — show what would be sent, send nothing", value=True,
                      key="send_dry")
    campaign = st.text_input("Campaign label", value="dashboard", key="send_campaign",
                             help="Recorded on each send; also what makes a re-send "
                                  "idempotent for the same address.")

    armed = True
    if not dry:
        typed = st.text_input(
            "This sends real email. Type SEND to confirm.", key="send_confirm",
            placeholder="SEND")
        armed = typed.strip() == "SEND"
        if not armed:
            st.warning(f"{len(picked)} real email(s) will go out once you type SEND.")

    label = (f"Dry run ({len(picked)})" if dry
             else f"✉ Send {len(picked)} email(s)")
    if st.button(label, type="primary", disabled=not armed, width='content'):
        box = st.empty()
        lines: list[str] = []

        def log(m: str = "") -> None:
            lines.append(m)
            box.code("\n".join(lines[-18:]))

        import pipeline

        # Reuse stage_send rather than reimplementing: it already reads the stored
        # HTML (so what ships is what was validated), records every outcome, and
        # keeps going when one address fails.
        prev, pipeline._log = pipeline._log, log
        try:
            with st.spinner("sending..." if not dry else "checking..."):
                # Pass explicit ids, not a count: `limit` slices before the
                # unsubscribe check, so a count can overstate what actually goes out.
                out = pipeline.stage_send(conn, run_id=None, campaign=campaign,
                                          dry_run=dry, limit=len(picked),
                                          brief_ids=[r["id"] for r in picked])
        finally:
            pipeline._log = prev

        if dry:
            st.info(f"Dry run complete — {out['skipped']} would be sent, nothing emailed.")
        elif out.get("aborted"):
            st.error(f"STOPPED EARLY: {out['aborted']}\n\n"
                     f"sent {out['sent']} · failed {out['failed']} · "
                     f"skipped {out['skipped']} before stopping. Fix the cause, "
                     f"then send again to pick up the rest.")
        else:
            st.success(f"sent {out['sent']} · failed {out['failed']} · "
                       f"skipped {out['skipped']}")
        st.rerun()


def main() -> None:
    st.title("CMI Operations")
    st.caption(f"cmi_newsletter.db · {datetime.now():%Y-%m-%d %H:%M}")

    vm_panel()
    st.divider()
    totals_panel()
    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        quality_panel()
    with c2:
        activity_panel()
    st.divider()
    runs_panel()
    st.divider()
    generate_panel()
    st.divider()
    send_panel()
    st.divider()
    briefs_panel()


if __name__ == "__main__":
    main()
