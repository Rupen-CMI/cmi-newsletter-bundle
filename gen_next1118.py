"""Generate signal-kind (company) briefs only for the next 1118-lead batch.
Mirrors the first 400-lead run: same stage_generate call, same kinds=['signal'],
same 40 workers. Explicitly scoped to signal only -- no report/industry briefs.
"""
import json
import os

os.environ['LLM_PROVIDER'] = 'local'
os.environ['LOCAL_LLM_BASE_URL'] = 'https://9002aa5206321.notebooksn.jarvislabs.net/v1'
os.environ['LOCAL_LLM_MODEL'] = 'cmi-7b'

os.chdir(r'C:\Users\rupen\Documents\cmi-newsletter-bundle-20260924T080359Z-1-001\cmi-newsletter-bundle')

import db
import pipeline

with open('next1118_leads.json', encoding='utf-8') as f:
    leads = json.load(f)

print(f'{len(leads)} leads loaded, generating SIGNAL briefs only', flush=True)

conn = db.get_conn()
db.init_db(conn)

run_id = db.start_run(conn, workers=40, kind='signal', endpoint=os.environ['LOCAL_LLM_BASE_URL'],
                       notes='next1118_run1')
print(f'run_id={run_id}', flush=True)

import time
t0 = time.time()
results = pipeline.stage_generate(conn, leads, kinds=['signal'], workers=40,
                                   run_id=run_id, outdir='output_next1118')
wall = time.time() - t0

print(f'\nDONE: {len(results)} results in {wall:.0f}s ({wall/60:.1f} min)', flush=True)
