"""List briefs the LLM judge flagged for human review, with its reasoning.

    python flagged_briefs.py                  # every flagged brief still in the DB
    python flagged_briefs.py --run 15         # just one run
    python flagged_briefs.py --emails-only     # just the email addresses, one per line
                                                # (pipe straight into a re-run xlsx/filter)

These are NOT the same as FAIL. A FAIL already failed the text rules and is already
held back from sending -- see validate_brief.check_detailed. A "flagged" brief passed
every keyword rule (PASS or WARN) but the LLM judge, reading the same text, thinks a
named company is in the wrong role (a customer that reads like a rival, a competitor
that is actually a customer, etc.). It still sends today under the advisory default
(CMI_LLM_STRICT=0) -- this script exists so those specific contacts can be set aside,
looked at, corrected in company_overrides.json if needed, and regenerated later,
without reading all the other briefs that were never in question.
"""
from __future__ import annotations

import argparse
import json

import db


def flagged(conn, run_id: int | None = None) -> list[dict]:
    """Every brief whose stored validation carries llm_flagged=True, newest first."""
    rows = db.list_briefs(conn, run_id=run_id, limit=10000)
    out = []
    for r in rows:
        try:
            v = json.loads(r["validation"] or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(v, dict) or not v.get("llm_flagged"):
            continue
        out.append({
            "id": r["id"], "email": r["email"], "company": r["company"],
            "kind": r["kind"], "status": r["status"],
            "llm_fails": v.get("llm_fails", []), "llm_warns": v.get("llm_warns", []),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="List briefs the LLM judge flagged, with its reasoning.")
    ap.add_argument("--run", type=int, default=None, help="only this run_id")
    ap.add_argument("--emails-only", action="store_true",
                    help="print just the email addresses, one per line")
    args = ap.parse_args()

    conn = db.get_conn()
    db.init_db(conn)
    rows = flagged(conn, run_id=args.run)

    if args.emails_only:
        for r in rows:
            print(r["email"])
        return

    if not rows:
        print("nothing flagged.")
        return

    print(f"{len(rows)} brief(s) flagged by the LLM judge (not blocked from sending -- "
          f"advisory by default; review before sending if in doubt):\n")
    for r in rows:
        print(f"[{r['status']}] id={r['id']}  {r['email']}  {r['company']} ({r['kind']})")
        for f in r["llm_fails"]:
            print(f"    x {f}")
        for w in r["llm_warns"]:
            print(f"    ! {w}")
        print()


if __name__ == "__main__":
    main()
