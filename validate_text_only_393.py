"""Text-rule validation ONLY (no LLM judge, no GPU needed) for the 393 clean
first-pass signal briefs pending validation. CMI_LLM_VALIDATE is deliberately
left unset so check_detailed's LLM-judge branch never fires here -- this is
pure keyword/structure checks, run independently of (and in parallel with)
the LLM-judge pass.
"""
import json
import os
import tempfile
import time
import concurrent.futures as cf

os.chdir(r'C:\Users\rupen\Documents\cmi-newsletter-bundle-20260924T080359Z-1-001\cmi-newsletter-bundle')

import db
import validate_brief as vb
import tenant

IDS_PATH = 'validate_ids_393.json'
with open(IDS_PATH) as f:
    ids = json.load(f)

conn = db.get_conn()
q = ','.join('?' for _ in ids)
rows = conn.execute(f"SELECT id, email, company, html FROM briefs WHERE id IN ({q})", ids).fetchall()
print(f'{len(rows)} briefs to text-validate (no LLM)', flush=True)

companies = sorted(set(r['company'] for r in rows if r['company']))
print(f'{len(companies)} distinct companies to profile', flush=True)


def _safe_print(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(__import__('sys').stdout, 'encoding', None) or 'ascii'
        print(str(msg).encode(enc, errors='replace').decode(enc, errors='replace'), flush=True)


def make_profile(company):
    worker_conn = db.get_conn()
    try:
        return company, tenant.Profile.from_company(company, '', 'there', '', conn=worker_conn)
    except Exception as e:
        _safe_print(f'  profile failed for {company}: {type(e).__name__}: {e}')
        return company, None


profiles = {}
t0 = time.time()
with cf.ThreadPoolExecutor(max_workers=20) as ex:
    futs = [ex.submit(make_profile, c) for c in companies]
    for i, fut in enumerate(cf.as_completed(futs), 1):
        c, p = fut.result()
        profiles[c] = p
        if i % 50 == 0 or i == len(companies):
            print(f'  profiles: {i}/{len(companies)} ({time.time()-t0:.0f}s)', flush=True)
print(f'all profiles done in {time.time()-t0:.0f}s', flush=True)


def validate_one(row):
    company = row['company']
    profile = profiles.get(company) or tenant.Profile.cmi_default('')
    fd, path = tempfile.mkstemp(suffix='.html')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(row['html'])
        fails, warns = vb.check(path, profile)
    except Exception as e:
        fails, warns = [], [f'validator error: {type(e).__name__}: {e}']
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return row['id'], row['email'], company, fails, warns


counts = {'PASS': 0, 'WARN': 0, 'FAIL': 0}
t1 = time.time()
with cf.ThreadPoolExecutor(max_workers=20) as ex:
    futs = [ex.submit(validate_one, r) for r in rows]
    for i, fut in enumerate(cf.as_completed(futs), 1):
        brief_id, email, company, fails, warns = fut.result()
        status = 'FAIL' if fails else ('WARN' if warns else 'PASS')
        counts[status] += 1
        conn.execute(
            "UPDATE briefs SET validation = ?, status = ? WHERE id = ?",
            (json.dumps({'fails': fails, 'warns': warns, 'llm_fails': [], 'llm_warns': []},
                        ensure_ascii=False),
             'failed' if fails else 'validated', brief_id))
        _safe_print(f'  [{i}/{len(rows)}] [{status}] {email:<32} {company[:22]:<24}')
        if i % 20 == 0:
            conn.commit()
conn.commit()

print(f'\nTEXT VALIDATION DONE: {counts}, {time.time()-t1:.0f}s elapsed', flush=True)
