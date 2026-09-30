"""Send exactly the 364 validated signal briefs from run_id=18 (PASS+WARN, company
non-blank). Uses explicit brief_ids so pipeline.stage_send's own sendable_briefs()
query -- which has no kind filter -- cannot pull in the 14 report-kind briefs from
the same run.
"""
import json
import os
import sys

sys.path.insert(0, r'C:\Users\rupen\Documents\cmi-newsletter-bundle-20260924T080359Z-1-001\cmi-newsletter-bundle')
os.chdir(r'C:\Users\rupen\Documents\cmi-newsletter-bundle-20260924T080359Z-1-001\cmi-newsletter-bundle')

import db
import pipeline

IDS_PATH = r'C:\Users\rupen\Documents\cmi-newsletter-bundle-20260924T080359Z-1-001\cmi-newsletter-bundle\send_ids_run18_signal.json'

with open(IDS_PATH) as f:
    brief_ids = json.load(f)

print(f'about to send {len(brief_ids)} signal briefs (run_id=18)', flush=True)

conn = db.get_conn()
db.init_db(conn)

result = pipeline.stage_send(
    conn, run_id=18, campaign='signal_full400_batch1',
    dry_run=False, limit=len(brief_ids), brief_ids=brief_ids,
)

print(f"\nFINAL: sent={result['sent']} failed={result['failed']} "
      f"skipped={result['skipped']} aborted={result.get('aborted') or '(none)'}", flush=True)
