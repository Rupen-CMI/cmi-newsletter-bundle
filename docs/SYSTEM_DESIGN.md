# End-to-end system

**Built and tested.** This was the design note; the sections below now describe working
code. See "Status" at the end for what was verified and what is still open.

The important finding from the audit: **six of the ten steps already exist and work.**
`cmi_newsletter.db` holds 2,606 leads and 328 logged sends. The real gap is an
orchestrator that runs the steps as one job, plus per-lead scraped intel and persisted
output.

---

## What already exists

| Step | Module | State |
|---|---|---|
| 1. Excel -> SQLite | `import_leads.py`, `campaign.read_contacts` | **Works.** `campaign._sniff_columns` already infers columns by content, so a new sheet layout does not need code changes |
| 2. Batch scheduler | `scheduler.py` | **Works.** Polls `leads.schedule_status='queued'`, sends in each lead's local window |
| 3-5. Workers + data | `run_all40.py` | **Works.** 40 briefs in one concurrency window, Rs.0.10 each |
| 4. Scraping | `tenant.fetch_site_text`, `assets/smartcrawl.py` | **Partly.** Homepage fetch is wired in; smartcrawl runs but its output is not yet persisted per lead |
| 6. LLM | `llm.py` (`LLM_PROVIDER=local`) | **Works.** 100% on the fine-tune |
| 7. Validation | `validate_brief.py` + `llm_validate.py` | **Works.** Keyword rules plus the LLM judge |
| 8. Save output | — | **Missing.** Briefs are written to `output/*.html` only; nothing lands in the DB |
| 9. SMTP | `mailer.py` | **Works.** Reads `SMTP_*` from `.env`, sends and records |
| 10. Logging | `db.record_send` | **Partly.** Sends are logged; generation and validation are not |

---

## What is actually missing

### A. Schema additions

`db.init_db` gains three tables. `leads` and `sends` are unchanged, so existing rows
survive.

```sql
-- per-lead scraped intel, so smartcrawl runs ONCE per company, not once per brief
CREATE TABLE IF NOT EXISTS company_intel (
    domain        TEXT PRIMARY KEY,
    company       TEXT,
    value_prop    TEXT,
    industries    TEXT,      -- json array
    competitors   TEXT,      -- json array
    product_terms TEXT,      -- json array
    raw_json      TEXT,      -- the full smartcrawl dump, for re-derivation
    source        TEXT,      -- 'smartcrawl' | 'homepage' | 'curated'
    fetched_at    TEXT,
    expires_at    TEXT       -- refresh after N days rather than never
);

-- every generated brief, whether or not it was sent
CREATE TABLE IF NOT EXISTS briefs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT NOT NULL,
    company       TEXT,
    kind          TEXT,      -- 'signal' | 'report'
    topic         TEXT,      -- the enquired market, for report briefs
    html          TEXT,      -- the brief itself
    status        TEXT,      -- 'generated' | 'validated' | 'sent' | 'failed'
    validation    TEXT,      -- json: {"fails": [...], "warns": [...]}
    llm_calls     INTEGER,
    gen_seconds   REAL,
    created_at    TEXT,
    UNIQUE(email, kind, date(created_at))   -- one brief per lead per kind per day
);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT,
    finished_at TEXT,
    workers     INTEGER,
    n_briefs    INTEGER,
    n_pass      INTEGER,
    n_warn      INTEGER,
    n_fail      INTEGER,
    wall_secs   REAL,
    cost_inr    REAL,
    endpoint    TEXT,
    notes       TEXT
);
```

`briefs.html` holds the full document. At ~33 KB each that is ~33 MB per thousand
briefs -- acceptable, and it means a re-send never regenerates. If it becomes a problem,
move the HTML to disk and keep a path.

### B. `pipeline.py` — the orchestrator

One command replacing the current manual sequence:

```bash
python pipeline.py --xlsx leads.xlsx --campaign oct1 --workers 40 \
    --kind report --send
```

Stages, each resumable because state lives in the DB:

```
  import   xlsx -> leads table (skips rows already present)
  enrich   for each distinct domain without fresh company_intel:
             homepage fetch, then smartcrawl if that yields under ~400 chars
  generate run_all40's pool over queued leads -> briefs table
  validate validate_brief + llm_validate -> briefs.validation
  send     mailer.send_html for briefs that pass the gate -> sends table
```

Run with `--dry-run` to stop before `send`. A crash mid-run resumes from the DB rather
than starting over.

### C. Enrichment cache

The measured failure this fixes: profile inference from a company NAME alone produced
"immunoassay CDMO" for a CAR-T maker, and VVF's homepage yields 301 characters of pure
navigation. smartcrawl solves it but takes ~20 minutes per company on the local model.

So enrich **once per domain**, not once per brief:

```
  company_intel hit and not expired  -> use it
  miss                               -> homepage fetch (fast)
  homepage under ~400 chars          -> queue for smartcrawl (slow, offline)
  curated company_overrides.json     -> always wins over both
```

Refresh on a 90-day expiry. A lead sheet of 2,606 rows has far fewer distinct domains,
so this is the difference between hours and minutes.

### D. Mail wiring

**SendGrid is wired and is now the default** whenever `SENDGRID_WEB_API` is set in
`.env`. `mailer.send_html` picks the transport in `_provider()`; `CMI_MAIL_PROVIDER`
forces one either way. The SendGrid path posts to the v3 API with urllib rather than
the SDK -- one small POST does not justify a dependency.

This matters because of the 2026-09-16 bounce wave: ~40% of sends were rejected
`550 5.4.1` by Outlook/M365, since raw Gmail SMTP from an unauthenticated domain reads
as bulk mail. `CMI_SENDING_DISABLED` was the blanket stop for that. It now gates only
the SMTP path -- SendGrid is the fix that block asked for.

**The open risk is the sender address, not the code.** `SENDER_EMAIL` is
`aditya@coherentmarketinsights.info` while Reply-To is `...@coherentmarketinsights.com`
-- different domains. SendGrid rejects a send with **403** if the From address is not a
verified sender or on an authenticated domain. The API key is scoped to mail-send only,
so this could not be checked by querying `/v3/verified_senders` (403 access forbidden);
it will only be confirmed by a real send.

### D2. Per-campaign SMTP (not wired)

`mailer.py` already reads `SMTP_HOST/PORT/USER/PASSWORD/SENDER_EMAIL` from `.env` and
raises a clear error when unset. What is missing is **per-campaign** credentials, so one
install can send as different senders:

```sql
-- reuse the existing settings table rather than adding another
INSERT INTO settings VALUES ('smtp.oct1.host', 'smtp.gmail.com');
INSERT INTO settings VALUES ('smtp.oct1.user', 'aditya@...');
```

`_smtp_config()` takes an optional campaign and falls back to `.env`. **Never put a
password in a shell script or a committed file** -- an earlier debug script in this repo
held a plaintext API key and had to be deleted.

### E. API

FastAPI over the DB, for the dashboard and for logging:

```
POST /runs                start a run          -> {"run_id": n}
GET  /runs/{id}           progress + counts
GET  /briefs?status=...   list, filter, paginate
GET  /briefs/{id}/html    render one brief
POST /briefs/{id}/send    send a single brief
GET  /leads?campaign=...  lead list with status
POST /leads/import        upload an xlsx
```

Read endpoints are straightforward. **`POST /briefs/{id}/send` is the one to guard** --
it sends real email to a real person, so it needs an explicit confirm flag and must
refuse a brief whose validation has hard failures.

---

## Ordering

1. **Schema + `briefs` persistence.** Everything else depends on generated briefs being
   somewhere other than a folder of HTML.
2. **`pipeline.py` without `--send`.** Import -> generate -> validate -> DB, end to end.
3. **Enrichment cache.** The largest quality win: it is what stops the model guessing.
4. **Sending.** Last, because it is the only irreversible step.
5. **API.** Once the data model has settled.

---

## Constraints carried over from the measurements

- **Rs.0.10/brief on-demand, Rs.0.068 spot** at 40 workers in one concurrency window.
  Two sequential batches cost 2.8x that.
- **40 workers is the sweet spot.** 64 is worse -- requests queue instead of batching.
- **~2.5% of LLM calls fail** with a 400 context-length error. They degrade a section
  rather than losing a brief, but they are why the 40-worker run lands fewer briefs in
  PASS than the 20-worker run.
- **Validation is advisory by default.** Only structural problems block; role calls
  warn. `CMI_LLM_STRICT=1` for a curated send.
- **One missing section warns, two fail.** Some markets genuinely have no rival news.
- **Never leave the GPU idle.** Spin up, run the batch, destroy. An idle box is a 20x
  cost swing, and pause/resume wipes pip installs -- vLLM needs reinstalling (~4 min)
  after every resume, though `/home/merged` survives.

---

## VM connection

`VM_RUNBOOK.md` carries every step from a cold laptop to a working endpoint — CLI path,
SSH key name and fingerprint, resume, serve, generate, tear down — plus the nine traps
that have already cost time here (pause wipes pip installs, Git Bash mangles remote
paths, resume returns a new machine id, and so on). It also holds the full system
architecture diagram.

Read that first when picking this up in a new session.

---

# Status — built 2026-09-23

## What was built

| component | file | state |
|---|---|---|
| Three new tables + helpers | `db.py` | **Built.** `company_intel`, `briefs`, `runs`. Existing 2,603 leads and 328 sends untouched |
| Enrichment cache | `enrich.py` | **Built.** Per-domain, 90-day expiry, curated > homepage > thin |
| Orchestrator | `pipeline.py` | **Built.** import → enrich → generate → validate → send, each stage resumable |
| HTTP API | `api.py` | **Built.** FastAPI over the DB, 10 endpoints |
| End-to-end test | `tests/test_e2e.py` | **Built.** Runs with the GPU paused by stubbing the LLM |

## Verified

```
STAGE 1 import    3 rows read, deduplicated against existing leads
STAGE 2 enrich    20 domains cached; second run 0.9s instead of ~60s
STAGE 3 generate  3/3 briefs with HTML, stored as each finished
STAGE 4 validate  3 WARN, 0 FAIL, verdicts written onto the rows
STAGE 5 send      gate cleared 3, dry-run emailed nothing
PERSIST           3 briefs with html, run row, API serving all of it
```

Send-gate behaviour, tested directly:

| brief | result |
|---|---|
| clean | **sendable** |
| warn-only | **sendable** — a warning is a judgement call, not a defect |
| hard failure | **blocked** |
| unsubscribed | **blocked** |

API guard on `POST /briefs/{id}/send`:

| request | response |
|---|---|
| no `confirm` | 400 — "this sends a real email" |
| `confirm` + hard failure | 409 with the failure list |
| missing brief | 404 |

## Two bugs found and fixed while testing

**SQLite across threads.** The first enrichment run failed all 20 domains with
`ProgrammingError: SQLite objects created in a thread can only be used in that thread`.
A connection had been shared with a `ThreadPoolExecutor`. Fixed by splitting
`_fetch_facts` (network, parallel, no DB) from `_store` (one thread) — which is the
right shape anyway, since the network is the slow part and the writes are tiny.

This is worth recording because it was a **usage** error, not a SQLite limitation: the
database was never the constraint.

**A test-data flaw that looked like a pipeline bug.** Generation failed with
`AttributeError: 'dict' object has no attribute 'strip'`. `compose.editorial` expects
`top3` as a list of strings and `_clean()`s each one; the stub returned dicts. The
pipeline was correct and the fixture was wrong.

## Still open

- **Sending has not been exercised against a real SMTP server.** Every test ran with
  `--dry-run-send`. The path is the same `mailer.send_html` the existing `scheduler.py`
  already uses in production, but the first real send should be a single address you
  control.
- **7 of 20 domains need smartcrawl** (`source='thin'`): `abahomecare.org`,
  `grcap.ru`, `mindthecustomer.dk`, `s8.dion.ne.jp`, `salient.us`,
  `scorevolunteer.org`, `synthsinc.com`. `python enrich.py --thin` lists them.
- **Per-campaign SMTP credentials** are designed (section D) but not wired: `mailer.py`
  still reads only `.env`.
- **Generation quality was not re-measured here.** The end-to-end test stubs the model
  deliberately — it proves the plumbing, not the writing. `MODEL_USAGE.md` §17–19 holds
  the quality and cost numbers.

## Links in the brief — fixed 2026-09-23

Three defects, all user-visible:

| defect | cause | fix |
|---|---|---|
| Cited sources were not clickable | `Article.url` was collected but never propagated into the cards; `_prospect` dropped it | cards carry `source_url`/`source_name`; template renders a `Source: <publisher>` link in the provenance row |
| Unsubscribe was `href="#"` | `unsub_url` hardcoded to `"#"` in `compose.py` | `UNSUB_URL` -> `/contact-us` (verified 200), overridable with `CMI_UNSUB_URL` |
| Link would have been dropped in the EMAIL only | `emailify._analyst` REBUILDS the provenance row from `.src`/`.verified` rather than copying it | it now re-emits `a.srclink` explicitly |

That third one is the trap worth remembering: the link renders correctly in the browser
file and vanishes in the sent email, so checking the generated HTML alone proves nothing.

**The CTA URL is `/talkwithanalyst`, not `/talktoanalyst`.** Checked against the live
site: `/talkwithanalyst` -> 200, `/talktoanalyst` -> 404. A test pins this so the
plausible-looking "fix" cannot silently break every CTA in every brief.

Verified on four freshly generated briefs: 2-3 source links each, all resolving 200,
4 analyst CTAs each, real unsubscribe URL, links present in both browser and email HTML.

## Tests

`tests/test_units.py` — 55 unit tests, offline (no GPU, network or SMTP), each on its
own temp database. Weighted toward what has actually broken here: thread-safety, the
send gate, free-mail domains, the mail provider switch, link integrity, JSON repair,
geo fallback, and validator role logic.

`tests/test_e2e.py` — full pipeline with the LLM stubbed.

```bash
python -m pytest tests/test_units.py -q
python tests/test_e2e.py
```

## Commands

```bash
# full run, stopping before send
python pipeline.py --xlsx Book1.xlsx --campaign oct1 --kind report \
    --workers 40 --base-url https://<host>/v1

# one stage at a time
python pipeline.py --xlsx Book1.xlsx --stage enrich
python enrich.py --thin                      # domains needing smartcrawl
python enrich.py --domain vvfltd.com --show  # inspect one

# send what a run produced (dry first)
python pipeline.py --campaign oct1 --send --dry-run-send
python pipeline.py --campaign oct1 --send

# API
python api.py                                # http://127.0.0.1:8080/docs

# the end-to-end test, no GPU needed
python tests/test_e2e.py
```
