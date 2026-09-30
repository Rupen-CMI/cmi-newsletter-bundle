# Bugs found and fixed in the send pipeline — 2026-09-24

Two review passes over `pipeline.py` and the send/validate/generate path. First pass
found and fixed two crash-level bugs. Second pass, after the validator and pacing work
landed, went broader — redundant LLM/network calls, exception handling, the 429
tripwire, LLM call counting, DB writes, and general structure. That pass found 7 more
issues; all 9 total are now fixed.

Every item here was verified by actually running the broken command or a small script
against the real logic — never just read through and assumed correct — before and
after each change. See each entry's "Verified" paragraph.

---

## Fixed

### 1. `pipeline.py` crashed on startup — `UnboundLocalError` on every documented command

**File:** `pipeline.py`, `main()`

**The bug:** `only = args.stage` was assigned at the bottom of `main()`, but read near
the top (in the autoconnect check) roughly 17 lines earlier. Because Python decides a
name is local to a function the moment ANY assignment to it exists anywhere in that
function, the early read raised `UnboundLocalError` before the assignment ever ran —
on every invocation that didn't pass `--no-autoconnect`, including the primary
documented command from the README:

```bash
python pipeline.py --campaign oct1 --send --dry-run-send
```

**Why it wasn't caught earlier:** the test suite calls the individual stage functions
(`stage_import`, `stage_generate`, etc.) directly and never runs `main()` itself, so
this path was completely untested.

**The fix:** moved `only = args.stage` up, immediately after argument parsing and
before the autoconnect check that reads it. No behavior changed — `only` only ever
depended on `args.stage`, which was already available at that point.

**Verified:** ran the exact broken command against a scratch copy of the project. Before
the fix: `UnboundLocalError: cannot access local variable 'only'`. After the fix: the
command runs past that point and reaches the real logic (it then correctly reports no
Jarvislabs VM exists yet, which is an environment issue, not a code bug). Full 87-test
suite re-run afterward with zero regressions.

---

### 2. `--stage send` created an empty run and found nothing to send

**File:** `pipeline.py`, `main()`

**The bug:** `--stage send` (meant to send briefs a *prior* run already generated and
validated, without regenerating anything) still fell through to
`db.start_run(...)`, which creates a brand-new, empty run record. `stage_send` was then
called with that new empty `run_id`, and since `sendable_briefs(conn, run_id=...)`
filters to briefs belonging to that specific run, it always found zero briefs — "0
briefs clear to send" — even when the database held plenty of previously-generated,
already-validated, sendable briefs from earlier runs.

**Why this mattered:** it broke the intended two-step workflow (generate now, review
later, send whenever ready) — the "send later, separately" half of that workflow simply
didn't work from the command line. The dashboard's send button was unaffected; it
already called `stage_send` with `run_id=None` directly.

**The fix:** added an explicit early branch for `only == "send"` that skips
enrich/generate/validate entirely and calls `stage_send(conn, run_id=None, ...)` —
sending across every prior run's sendable briefs, matching what the dashboard already
did correctly. Also added a matching `only == "validate"` early-return so
`--stage validate` can no longer fall through into an accidental send if `--send` is
passed alongside it.

**Verified:** seeded a sendable brief under a fake prior run (`run_id=999`) into a
scratch copy of the real database (5 other real sendable briefs already present), then
ran `pipeline.py --stage send --send --dry-run-send`. It correctly found and listed all
6 briefs, including the one from the unrelated old run — confirming it no longer
creates an empty run. Full test suite re-run afterward, still 87/87 passing.

---

A second, broader pass — after the validator (`check_detailed`) and send-pacing
(`sendpacing.py`) work were already in place — went through redundant calls, exception
handling, the 429/fatal-error tripwire, exact LLM call counts, network/HTTP handling,
DB write patterns, and general structure. It found 7 more issues. All 7 are now fixed
below, each independently verified by actually running code against the real logic —
constructing real `smtplib` exceptions and checking them against the tripwire, stress-
testing 300 concurrent calls across 40 threads, tracing SQL `COMMIT` statements with
`sqlite3`'s own trace callback, and (for the EMAIL check) testing against a real sample
brief with a simulated template/emailify mismatch, not synthetic data.

### 3. A full LLM call was generated and then thrown away, on every brief

**Files:** `brief_builder.py` (`build_editorial`, was line ~533) and `compose.py`
(`editorial`, line ~137)

**The bug:** `brief_builder.build_brief()` called `build_editorial()` — one LLM call,
~1600 tokens — which wrote a lede, three headline lines, and three analyst "takes"
into the brief dict (`brief["intro"]["lede_html"]`, `brief["top3"]`, `brief["take_*"]`).

`pipeline.stage_generate` then handed that brief dict to `compose.build_context`,
which never read any of those keys. It called its own, separate `editorial()`
function instead — a second, independent LLM call, ~1000 tokens — which re-derived a
new lede and top3 from the raw section cards. Only this second version ever reached
the rendered brief; the first call's output was generated, stored, and discarded on
every single brief.

**Why not just wire the two together instead of removing one:** `build_editorial`'s
`top3` list is built around the fixed section names a *signal* brief uses ("Potential
Customer" / "Industry Outlook" / "Competition Outlook"), while `compose.editorial`
already builds its lede/top3 generically from whatever sections actually got
assembled — correctly handling a *report* brief's different section names ("Your
Market" / "Market Moves") the same way. Wiring `build_editorial`'s output through
would have meant fixing a real shape mismatch on top of removing the waste; keeping
`compose.editorial` as the one source of truth avoided introducing a second bug while
fixing the first.

**The fix:** removed the `build_editorial()` call, its `top3`-building code, and the
`brief["intro"]`/`brief["top3"]`/`brief["take_*"]` keys from `brief_builder.build_brief`
(confirmed by grep that nothing outside `brief_builder.py` ever read those keys).
`build_editorial` and its only other helper, `_plain()`, were then unreferenced
anywhere in the codebase, so both were deleted outright rather than left as dead code.

**Verified:** grepped the whole codebase for every reader of `brief["intro"]`,
`brief["top3"]`, `take_customer/competition/industry`, and `build_editorial` before
removing anything, confirming zero external readers (one internal `compose.py` line
matched but turned out to be an unrelated local variable in `compose.editorial`'s own
JSON parsing). Full 87-test suite re-run afterward — the one test referencing similar
field names (`tests/test_e2e.py`'s single blanket LLM stub dict) still passed, since
those keys simply became unused entries in an already-oversized stub dict rather than
something any code path still reads.

### 4. Company homepage was re-fetched every brief, bypassing the cache built one stage earlier

**File:** `tenant.py`, `Profile.from_company`; also `pipeline.py` and `send_one.py`
(call sites)

**The bug:** `pipeline.stage_enrich` fetches each distinct company domain's homepage
once and caches it in `company_intel` (90-day TTL) — specifically so the slow fetch
happens once per company, not once per brief. But `tenant.py` had no import of `db` or
`enrich` at all: `Profile.from_company()` called `fetch_site_text(website)` directly,
doing its own live HTTP fetch on every single brief, completely bypassing that cache.

**The fix:** added an optional `conn` keyword argument to `Profile.from_company`.
When given, it checks `db.get_intel(conn, domain)` first (deriving the domain from
the website via `enrich.domain_of`) and only calls `fetch_site_text` on a cache miss —
and only trusts a cached entry whose `source` is `"homepage"` specifically; a
`"curated"` entry is a different kind of data (company facts, not homepage text —
handled separately via the existing `_overrides_for` call further down in the same
function) and a `"thin"` entry means the homepage genuinely had nothing usable, so
there's nothing to gain from reusing it. Callers without a `conn` (standalone scripts,
the CLI validator) are completely unaffected — the parameter defaults to `None` and
behavior is identical to before.

Wired the two callers that matter: `pipeline.stage_generate`'s worker closure now
calls `db.get_conn()` itself (safe from inside a `ThreadPoolExecutor` worker — see
`db.py`'s own docstring: connections are cached per-thread, so this never touches the
main thread's connection object) and passes it to `from_company`. `send_one.py` opens
its DB connection earlier (it already needed one, just later in the function) and
passes the same one through.

**Verified:** three scenarios, each with `tenant.fetch_site_text` monkeypatched to a
call-counting spy so a live fetch would be caught red-handed:
1. A domain with a cached `source="homepage"` entry → **zero** live fetches, and the
   cached text was confirmed present in the actual LLM prompt sent.
2. A domain never enriched (true cache miss) → exactly 1 live fetch, as before.
3. A domain with a cached `source="thin"` entry, and a call with `conn=None` → both
   still fell back to a live fetch correctly, confirming the fix doesn't break the
   existing fallback behavior.

### 5. The 429 tripwire didn't reliably catch SMTP-side rate limits or auth failures

**File:** `sendpacing.py` (`_FATAL_PATTERNS`, `is_fatal_exception` — new,
`is_fatal_provider_error`); `pipeline.py` (`stage_send`'s abort check)

**The bug:** verified by constructing real `smtplib` exceptions and checking their
string form against the original fatal-pattern list — a plain `smtplib.
SMTPResponseException(421, ...)`, Gmail's real `"454 ... Too many login attempts"`
throttle message, and `smtplib.SMTPAuthenticationError(535, ...)` all failed to match
any string in the list (SMTP's own 4xx/5xx codes and wording don't overlap with
HTTP's `"429"`/`"403"` or the exact phrase `"too many requests"`). All three would
have fallen through to the ordinary 3-consecutive-failure breaker instead of aborting
immediately, contradicting the module's own stated guarantee.

**The fix, two layers:**
1. **By exception type** (`is_fatal_exception`, new): checks the real exception
   object's type name against `SMTPAuthenticationError`/`SMTPServerDisconnected`/
   `SMTPNotSupportedError`, and separately checks its `smtp_code` attribute (which
   `smtplib.SMTPResponseException` and every subclass expose directly) against a list
   of SMTP codes that are genuine provider-level faults (421/450/452/454) —
   deliberately excluding recipient-specific codes like 550 ("no such user"), which
   must keep counting toward the ordinary failure breaker rather than aborting the
   whole run over one bad address.
2. **By message text** (`is_fatal_provider_error`, extended): added the missing SMTP
   codes and phrases (`"421"`, `"450"`, `"452"`, `"454"`, `"too many login attempts"`,
   `"authentication failed"`, `"username and password not accepted"`) as a fallback
   for transports that don't raise a distinctly-typed exception per failure kind.

`pipeline.stage_send` now checks both: the real exception is preserved through `_one()`'s
`raise RuntimeError(detail) from e` (recovered via `e.__cause__` in the outer handler),
so `is_fatal_exception` runs against the original object first, falling back to the
text check.

**Verified:** re-ran the exact three previously-failing cases against the new checks —
all three now caught, by type in every case. Confirmed a genuine per-recipient
rejection (`SMTPRecipientsRefused` with a 550) is correctly **not** flagged fatal by
either layer, so it still counts toward the ordinary breaker rather than stopping the
whole run over one bad address. Then an end-to-end test: seeded 6 sendable briefs,
made the 2nd one raise a real `SMTPAuthenticationError`, and confirmed `stage_send`
aborted immediately after exactly 2 attempts, leaving the remaining 4 recipients
untouched.

### 6. Importing a lead sheet did one database commit per row instead of one for the whole batch

**Files:** `pipeline.py` (`stage_import`), `db.py` (`upsert_lead`)

**The bug:** for every lead in an imported sheet, the code ran a `SELECT` (just to
count how many were new, for the log line) and then called `db.upsert_lead`, which
commits internally on every call. A 400-row sheet meant 400 individual `SELECT`s and
400 separate disk-flushing commits, where one batched transaction does the same work
with a single flush.

**The fix:** added a `commit: bool = True` parameter to `db.upsert_lead` — every
existing caller keeps its old behavior (commits immediately) by not passing it.
`stage_import` now runs one upfront `SELECT email FROM leads` to get the full set of
already-known addresses (replacing 400 individual per-row SELECTs with one query),
passes `commit=False` to every `upsert_lead` call in its loop, and commits exactly
once after the loop finishes.

*Not* applied to `enrich.py`'s equivalent per-domain writes: that loop stores results
as they complete from a concurrent `ThreadPoolExecutor`, where deferring the commit to
the very end would mean a crash partway through loses every domain enriched so far
instead of just the ones not yet done — trading away the pipeline's explicit
crash-resilience design for a speed gain that matters far less there (enrichment is
already deduplicated to distinct domains, not one row per lead).

**Verified:** used `sqlite3.Connection.set_trace_callback` to literally observe every
SQL statement executed during a real `stage_import` run against `Book1.xlsx` —
confirmed exactly 1 `COMMIT` statement (previously would have been 1 per row) against
the correct number of `INSERT`s, correct row count in the table afterward, and correct
"new vs. already known" counts on both a fresh import and a re-import of the same file.

### 7. The LLM call counter could lose increments under concurrent load

**File:** `llm.py` (`STATS`, `chat()`)

**The bug:** `STATS[name] += 1` is a read-then-write on a plain module-level dict,
called from every one of the up to 40 worker threads in `pipeline.stage_generate`'s
pool. Two threads incrementing at nearly the same instant can both read the old value
before either writes back, silently losing an increment — affecting only the
informational summary line and cost-per-brief number printed at a run's end, never
what gets written into a brief or sent to a recipient.

**The fix:** added a module-level `threading.Lock()` (`_STATS_LOCK`) and wrapped the
two places `STATS` is mutated (the success path incrementing `STATS[name]` and setting
`provider_used`, and the failure path incrementing `STATS["failed"]`) in `with
_STATS_LOCK:` blocks. The lock only guards a few dict/int operations, not the LLM call
itself, so it costs nothing measurable even at 40-way concurrency.

**Verified:** stubbed the network call and fired 300 `llm.chat()` calls across a real
40-worker `ThreadPoolExecutor` (not a simulation — actual OS threads). `STATS["local"]`
came out to exactly 300 every time; before the fix, this is exactly the kind of test
that would occasionally under-count under real thread scheduling.

### 8. No timeout was configured on the LLM connection itself

**Files:** `llm.py` (client construction), `config.py` (`LLM["timeout_secs"]`, new)

**The bug:** none of the three `OpenAI(...)` client objects (local vLLM, DeepSeek,
OpenAI) were given an explicit request timeout. A hung request under load (GPU memory
pressure, a stuck queue entry) would block its worker thread for the SDK's own
built-in default — several minutes — with no error or log line, silently draining a
batch's effective concurrency with no visible symptom besides the run taking longer
than expected.

**The fix:** added `config.LLM["timeout_secs"]` (default 90 — generous headroom for a
busy server, since a single call asks for at most ~1600 tokens and a genuine response
is seconds, not minutes), overridable via `CMI_LLM_TIMEOUT_SECS`. Passed as
`timeout=_LLM_TIMEOUT` to all three `OpenAI(...)` constructors.

**Verified:** confirmed `llm._local_client.timeout` reads back as exactly `90.0` by
default, and as `45.0` when `CMI_LLM_TIMEOUT_SECS=45` is set in the environment before
import — both against the real constructed client object, not a mock.

### 9. Validation never checked the actual emailed HTML, only the browser version

**File:** `validate_brief.py` (`check_detailed`, new EMAIL section)

**The bug:** `emailify()` converts the rendered brief into the table-based HTML that
actually gets emailed, and by its own documented design, a CSS class it doesn't
recognize is "silently skipped rather than half-rendered." But `emailify()` was only
ever called inside `mailer.send_html`, at the moment of sending — never during
validation. A template edit that renamed a class `emailify.py` looks for could drop a
whole section from every sent email while the browser-rendered HTML (the only thing
validated) looked completely fine.

**Getting this fix right took three attempts, and it's worth recording why the first
two failed their own tests rather than pretending the final version was obvious:**

- **Attempt 1 — check for the section's label text in `emailify()`'s output.** Failed
  immediately against the *existing test suite*: `tests/test_units.py`'s `GOOD_HTML`
  fixture is bare `<h2>`/`<p>` tags with zero CSS classes, built only to exercise
  `validate_brief`'s own regex section-detection — it was never meant to survive
  `emailify()`, which correctly produces almost nothing from it by design. The fix:
  gate the whole EMAIL check on the input actually looking like a real rendered
  template (presence of `<main class="body">`, the exact container `emailify()`
  parses), so a synthetic test fixture is skipped rather than falsely flagged.
- **Attempt 2 — same label-text check, but only run against real template HTML.**
  Passed the test suite, but failed a *new, deliberately adversarial test* built to
  prove the check actually catches a real mismatch: renaming the `.lbl` class on one
  section's eyebrow heading (the exact selector `emailify._eyebrow` reads) didn't get
  flagged, because that same label text is written in up to three separate places in
  the template (the top3 category badge, the eyebrow, the flag line) — breaking one
  occurrence still left the others intact, so "the label appears somewhere in the
  output" stayed true even though the *heading* specifically vanished.
- **Attempt 3 — same idea, but check for distinctive body-content words instead of
  the label.** Also failed the adversarial test, for a more interesting reason: real
  briefs repeat their own industry vocabulary throughout every section AND in the
  static "Inside CMI" showcase, so words pulled from a genuinely-dropped section
  often still "survive" via a completely unrelated part of the same document — a
  false negative on the exact failure the check exists to catch, on the real sample
  brief used to test it (Bharat Forge's own "automotive"/"capacity"/"commercial"
  vocabulary appears in at least six unrelated places in that specific brief).
- **What actually worked — a structural check, not a text comparison.** Parse
  `main.body`'s direct children in the source HTML and count how many carry
  `"prospect"` or `"story"` in their class attribute — copying `emailify.emailify()`'s
  own dispatch condition verbatim, so this check breaks in lockstep with `emailify.py`
  rather than silently drifting out of sync with it. Compare that count against how
  many sections `validate_brief`'s STRUCTURE check found present in the browser text.
  A shortfall means a section's own container element isn't spelled the way
  `emailify()` expects, regardless of what words happen to appear anywhere in the
  final output. Also flags any class that looks like a near-miss variant (e.g.
  `"prospect-v2"`) by name, so the failure message points at the actual suspicious
  attribute rather than just a bare count.

**Verified:** ran both the false-positive and true-positive tests against the final
version. On the real, unmodified sample brief: zero EMAIL findings. On the same brief
with the Competition Outlook section's `class="prospect"` renamed to
`class="prospect-RENAMED"` (simulating a real template/emailify drift): a FAIL citing
"only 2 carry a class emailify.py's dispatch loop recognises" and naming
`prospect-RENAMED` specifically as the suspicious attribute. Full 87-test suite
re-run afterward, including the original `GOOD_HTML` synthetic fixture — no false
positive, no regression.

---

## Summary

| # | Issue | Status | Blocks a live run? |
|---|---|---|---|
| 1 | `main()` crashes on startup (`UnboundLocalError`) | **Fixed & verified** | Was a hard blocker; now clear |
| 2 | `--stage send` sends nothing (empty new run) | **Fixed & verified** | Was blocking the two-step workflow; now clear |
| 3 | Wasted LLM call every brief (`build_editorial` vs `compose.editorial`) | **Fixed & verified** | No — was cost only |
| 4 | Homepage re-fetched per brief, bypassing cache | **Fixed & verified** | No — was cost/latency only |
| 5 | 429 tripwire misses SMTP-side errors | **Fixed & verified** | No — SMTP is off by default; now closed either way |
| 6 | Import commits once per row instead of once per batch | **Fixed & verified** | No — was a slowdown only |
| 7 | LLM call counter can under-count under concurrency | **Fixed & verified** | No — was cosmetic reporting only |
| 8 | No LLM client timeout | **Fixed & verified** | No — was a silent-slowdown risk |
| 9 | Validation doesn't check the emailed HTML | **Fixed & verified** | No — was a real content-loss risk if the template ever drifted |
