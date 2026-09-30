"""Send exactly the 438 text-validated (WARN) signal briefs from run_id=20. Uses
explicit brief_ids so stage_send's own sendable_briefs() query -- which has no
kind filter -- cannot pull in anything outside this exact, pre-approved set.
"""
import json
import os
import sys

os.chdir(r'C:\Users\rupen\Documents\cmi-newsletter-bundle-20260924T080359Z-1-001\cmi-newsletter-bundle')

import db
import pipeline

IDS_PATH = 'send_ids_run20_warn438.json'

with open(IDS_PATH) as f:
    brief_ids = json.load(f)

print(f'about to send {len(brief_ids)} signal briefs (run_id=20)', flush=True)

conn = db.get_conn()
db.init_db(conn)

result = pipeline.stage_send(
    conn, run_id=20, campaign='signal_run20_warn438',
    dry_run=False, limit=len(brief_ids), brief_ids=brief_ids,
)

print(f"\nFINAL: sent={result['sent']} failed={result['failed']} "
      f"skipped={result['skipped']} aborted={result.get('aborted') or '(none)'}", flush=True)
