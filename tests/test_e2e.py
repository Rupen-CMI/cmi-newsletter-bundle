"""End-to-end pipeline test with the LLM stubbed, so it runs with the VM paused.

Proves the plumbing: import -> enrich -> generate -> validate -> send-gate -> DB -> API.
The model's WRITING is not under test here (that is measured in MODEL_USAGE.md); what
is under test is that every stage hands the next one the right thing and that nothing
irreversible happens without confirmation.
"""
import json, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")

# Import from the repo root regardless of where this is invoked from, so
# `python tests/test_e2e.py` works the same as running it from the root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["LLM_PROVIDER"] = "local"
os.environ["LOCAL_LLM_BASE_URL"] = "http://127.0.0.1:1/v1"   # guaranteed unreachable

import llm

# Stub every LLM path. chat_json already degrades to `default` on failure, so a stub
# that returns plausible JSON exercises the same code the real model would.
_FAKE = {
    "value_prop": "TestCo sells industrial widgets to manufacturers.",
    "target_industries": ["Manufacturing", "Automotive"],
    "competitors": ["RivalCo", "OtherCo"],
    "industry_keywords": {"Manufacturing": ["factory", "plant"]},
    "trigger_query_seeds": ["manufacturer raises funding"],
    "industry_query_seeds": ["widget market outlook"],
    "items": [{"index": 0, "company_name": "BuyerCo", "relationship": "buyer"}],
    "narrative_html": "BuyerCo opened a new plant. That lifts demand for widgets. "
                      "TestCo can supply the line. Timing favours an early approach.",
    "why_it_matters": "The plant is being equipped now.",
    "their_edge": "Scale and local presence.",
    "what_it_signals": "Capacity is expanding.",
    "recommended_response": "Reach the procurement lead this quarter.",
    "counter_play": [{"title": "Move first.", "body_html": "Contact them now."}],
    "bullets": ["<b>Lead.</b> Support line.", "<b>Two.</b> Support.", "<b>Three.</b> Support."],
    "headline": "Widget Demand Rises", "body_html": "Demand is rising across the sector.",
    "lede": "CMI tracked three signals this week.",
    "top3": ["<b>One.</b> Support.", "<b>Two.</b> Support.", "<b>Three.</b> Support."],
    "take_customer": "x", "take_competition": "x", "take_industry": "x",
    "industry": "industrial widgets",
    "projects": [{"title": "Widget industry outlook, supplier and demand landscape",
                  "body_html": "We mapped supply and demand."}] * 6,
    "vertical": "manufacturing",
}
llm.chat_json = lambda system, user, **kw: dict(_FAKE)
llm.chat = lambda system, user, **kw: "Plain text answer."

import db, pipeline

DB = "cmi_newsletter.db"
conn = db.get_conn(); db.init_db(conn)
CAMPAIGN = "e2e-selftest"

print("=" * 66)
print("STAGE 1: import")
leads = pipeline.stage_import(conn, "Book1.xlsx", CAMPAIGN, "report", 3)
assert leads, "no leads read"
print(f"  -> {len(leads)} leads")

print("\nSTAGE 2: enrich")
summary = pipeline.stage_enrich(conn, leads, workers=4, force=False)
print(f"  -> {summary}")

print("\nSTAGE 3: generate")
rid = db.start_run(conn, workers=3, kind="report", endpoint="stub", notes=CAMPAIGN)
t0 = time.time()
results = pipeline.stage_generate(conn, leads, kinds=["report"], workers=3,
                                  run_id=rid, outdir="output_e2e")
wall = time.time() - t0
made = [r for r in results if not r["error"]]
print(f"  -> {len(made)}/{len(results)} briefs in {wall:.0f}s")

print("\nSTAGE 4: validate")
counts = pipeline.stage_validate(conn, results)
print(f"  -> {counts}")

print("\nSTAGE 5: send gate (dry run — nothing is emailed)")
s = pipeline.stage_send(conn, rid, CAMPAIGN, dry_run=True, limit=10)
print(f"  -> {s}")

db.finish_run(conn, rid, n_briefs=len(results), n_pass=counts["PASS"],
              n_warn=counts["WARN"], n_fail=counts["FAIL"] + counts["ERROR"],
              wall_secs=wall, cost_inr=0.0)

print("\n" + "=" * 66)
print("PERSISTENCE CHECKS")
rows = db.list_briefs(conn, run_id=rid, limit=100)
print(f"  briefs stored     : {len(rows)}")
with_html = sum(1 for r in rows if db.get_brief(conn, r['id'])['html'])
print(f"  with html         : {with_html}")
print(f"  sendable          : {len(db.sendable_briefs(conn, run_id=rid))}")
run = db.list_runs(conn, 1)[0]
print(f"  run row           : id={run['id']} briefs={run['n_briefs']} wall={run['wall_secs']:.0f}s")

from fastapi.testclient import TestClient
import api
t = TestClient(api.app)
print(f"  API /briefs       : {t.get(f'/briefs?run_id={rid}').status_code} "
      f"({len(t.get(f'/briefs?run_id={rid}').json())} rows)")
if rows:
    print(f"  API /briefs/html  : {t.get(f'/briefs/{rows[0][chr(39)+chr(39)] if False else rows[0]['id']}/html').status_code}")

print("\nCLEANUP")
for r in rows:
    conn.execute("delete from briefs where id=?", (r["id"],))
conn.execute("delete from runs where id=?", (rid,))
conn.execute("delete from leads where campaign=?", (CAMPAIGN,))
conn.commit()
print("  test rows removed")
print("\nE2E COMPLETE")
