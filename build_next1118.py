"""Build the lead-dict list for the next 1118-lead signal-brief generation batch.

Same shape as the earlier run400_leads.json: one dict per lead with the fields
stage_generate expects (email, company, name, website, country, topic). Reads
directly from the leads table -- ids already selected and saved to
next1118_lead_ids.json (queued, no existing signal brief, non-blank company).
"""
import json
import os
import sqlite3

os.chdir(r'C:\Users\rupen\Documents\cmi-newsletter-bundle-20260924T080359Z-1-001\cmi-newsletter-bundle')

import enrich

with open('next1118_lead_ids.json') as f:
    ids = json.load(f)

conn = sqlite3.connect('cmi_newsletter.db')
conn.row_factory = sqlite3.Row
placeholders = ','.join('?' for _ in ids)
rows = conn.execute(
    f"SELECT id, email, name, company FROM leads WHERE id IN ({placeholders})", ids
).fetchall()
by_id = {r['id']: r for r in rows}
ordered = [by_id[i] for i in ids]

leads = []
for r in ordered:
    domain = enrich.domain_of(email=r['email'])
    leads.append({
        "email": r['email'],
        "company": r['company'] or '',
        "name": r['name'] or '',
        "website": domain,
        "country": "",
        "topic": "",
    })

with open('next1118_leads.json', 'w', encoding='utf-8') as f:
    json.dump(leads, f, ensure_ascii=False)

print(f"wrote {len(leads)} leads to next1118_leads.json")
print("sample:", leads[0])
print("sample:", leads[-1])
