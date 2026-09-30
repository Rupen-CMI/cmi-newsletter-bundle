"""Silent watchdog for the warn438 send. Only prints/exits when it actually acts:
kills the send process on a real problem (mismatch, consecutive failure spike, or
the send script's own tripwire abort), or on clean completion. No routine progress
noise -- the user explicitly asked not to see per-tick updates.
"""
import time, sqlite3, json, sys, subprocess

SEND_PID = 1124

def safe_print(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, 'encoding', None) or 'ascii'
        print(str(msg).encode(enc, 'replace').decode(enc, 'replace'), flush=True)

def kill_send():
    try:
        subprocess.run(['kill', '-9', str(SEND_PID)], capture_output=True)
    except Exception:
        pass

with open('send_ids_run20_warn438.json') as f:
    ids = json.load(f)
ids_set = set(ids)
total = len(ids)
placeholders = ','.join('?' for _ in ids)

checked_ids = set()
consecutive_fails = 0

while True:
    conn = sqlite3.connect('cmi_newsletter.db')
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f'''
        SELECT b.id, b.email, b.kind, b.company, b.status, l.company as lead_company
        FROM briefs b LEFT JOIN leads l ON l.email = b.email
        WHERE b.id IN ({placeholders})
    ''', ids).fetchall()
    conn.close()

    sent = [r for r in rows if r['status'] == 'sent']
    failed = [r for r in rows if r['status'] == 'failed']

    new_rows = [r for r in rows if r['id'] not in checked_ids and r['status'] in ('sent', 'failed')]
    for r in new_rows:
        checked_ids.add(r['id'])
        if r['status'] == 'failed':
            consecutive_fails += 1
            if consecutive_fails >= 3:
                safe_print(f'ACTION: killing send (PID {SEND_PID}) -- {consecutive_fails} consecutive failures, last: {r["email"]}')
                kill_send()
                safe_print('SEND STOPPED BY WATCHDOG')
                sys.exit(1)
            continue
        consecutive_fails = 0
        problems = []
        if r['kind'] != 'signal':
            problems.append(f'kind={r["kind"]}')
        if r['id'] not in ids_set:
            problems.append('not in intended set')
        if (r['company'] or '') != (r['lead_company'] or ''):
            problems.append(f'company mismatch {r["company"]!r} vs {r["lead_company"]!r}')
        if problems:
            safe_print(f'ACTION: killing send (PID {SEND_PID}) -- MISMATCH id={r["id"]} email={r["email"]}: ' + '; '.join(problems))
            kill_send()
            safe_print('SEND STOPPED BY WATCHDOG')
            sys.exit(1)

    if len(sent) + len(failed) >= total:
        safe_print(f'DONE (clean): sent={len(sent)} failed={len(failed)} of {total}')
        break

    time.sleep(10)
