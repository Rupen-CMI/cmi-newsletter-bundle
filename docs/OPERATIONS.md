# Operations — run this system on any machine, under any owner

Written for someone who has **never touched this before** and may be on a new laptop,
a new Jarvislabs account, and a VM that no longer exists. Follow it top to bottom and
you get the same output quality that is documented in `MODEL_USAGE.md`.

The one thing you cannot regenerate is `training_data/adapter/` — 323 MB, md5
`5918b9989c07ed75ba21fc898132ea94`. That is the fine-tune. Everything else in this
repo is code, cache, or reconstructible from it.

---

## 0. The 60-second version

```bash
pip install -r requirements.txt
cp .env.example .env                  # then fill it in — see §2

python -m pytest tests/test_units.py -q   # 65 tests, no GPU needed
python vm_connect.py --ensure             # resume/boot the model, ~6 min cold
streamlit run dashboard.py                # watch it: green = connected

python pipeline.py --xlsx "sample leads.xlsx" --campaign oct1 --kind report --workers 40
python pipeline.py --campaign oct1 --send --dry-run-send
python pipeline.py --campaign oct1 --send
```

`python vm_connect.py --ensure` is the command that replaces almost all of the old
manual VM procedure. It finds the instance **by name**, resumes it if paused, installs
vLLM if a pause wiped it, starts the server in tmux, waits for a real completion, and
writes the endpoint into `.env`. Run it as often as you like; it is idempotent.

---

## 1. Why quality drifts, and what protects it

Four things have each silently degraded output here. Every one now has a guard, and
knowing them is most of what keeps quality stable across a VM or owner change.

| what went wrong | what it looked like | the guard now |
|---|---|---|
| Generated against a **dead endpoint** | 4 briefs, every LLM call 404, still reported `PASS` — empty sections are structurally valid | `pipeline.py` verifies a real completion before generating, and heals the endpoint |
| The **machine id changed** on resume | `.env` pointed at a dead host; nothing looked wrong | nothing is keyed on the id; the instance is found by **name** every time |
| **Profile inferred from a company name** | ImmunoACT (CAR-T) profiled as an "immunoassay CDMO"; wrong competitors everywhere downstream | `enrich.py` scrapes per domain; `company_overrides.json` beats inference and always wins |
| The **topic never reached collection** | a "global drone market" brief opened on electric trucks and admitted it found no drone company | `pipeline.py` collects topic-derived queries for report briefs |

There is a fifth that is not automatable: **the v1 adapter is the good one.** A later
filtered retrain (v2) had a better eval loss and *worse* briefs — 27.3 vs 21.7
violations per 1,000 words. Judge a model on generated briefs, never on eval loss.
`training_data/adapter/` is v1. Keep it.

---

## 2. Credentials

Copy `.env.example` to `.env` and fill it in. **`.env` is never committed** — it holds
live credentials, and `.gitignore` blocks it.

```ini
LLM_PROVIDER=local                    # every call goes to our fine-tune
LOCAL_LLM_BASE_URL=                   # leave blank; vm_connect writes it
LOCAL_LLM_MODEL=cmi-7b
LOCAL_LLM_API_KEY=local

SENDGRID_WEB_API=SG.xxxx              # SendGrid wins whenever this is set
SENDER_EMAIL=aditya@coherentmarketinsights.info
SENDER_NAME=Coherent Market Insights
```

Read secrets at run time; never paste one into a script or a markdown file:

```python
from dotenv import dotenv_values
key = dotenv_values(".env")["SENDGRID_WEB_API"]
```

**On ownership change, rotate everything**: the SendGrid key, SMTP app password, and
the Jarvislabs API token. A SendGrid key is a bearer credential — whoever holds it can
send as your domain.

### Jarvislabs, on a new machine or account

```powershell
pip install jarvislabs
$env:Path += ";$env:LOCALAPPDATA\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts"
jl setup            # paste the API key
jl status
```

Run `jl` from **PowerShell, not Git Bash** — Git Bash rewrites remote paths
(`/home/x` → `C:/.../Git/home/x`) and `jl list` crashes on the ₹ symbol under cp1252.
`vm_connect.py` locates the CLI itself, so Python callers do not need the PATH line.

---

## 3. Standing up the model from nothing

If the VM was deleted, or this is a new account, create one and rebuild the model:

```powershell
jl create --gpu L4 --spot --template pytorch --storage 60 --name cmi-serve-v1 --yes
```

**Use the `pytorch` template, never `axolotl`.** vLLM failed three times on the axolotl
image because installing it upgrades torch under a prebuilt environment.

Then rebuild `/home/merged` from the adapter (`MODEL_USAGE.md` §13) and:

```bash
python vm_connect.py --ensure
```

`CMI_VM_NAME` overrides the instance name if you call it something else.

### GPU choice

| GPU | VRAM | on-demand | spot | when |
|---|---|---|---|---|
| **L4** | 24 GB | ₹41.31 | ₹27.54 | the tuned target |
| A30 | 24 GB | ₹38.88 | ₹27.54 | straight substitute when L4 is out |
| A100 | 40 GB | ₹84.24 | ₹74.52 | only when nothing else is free |

`vm_connect` walks that ladder automatically. **Spot gets reclaimed** — three times in
one session on 2026-09-23, each costing a ~5 min vLLM reinstall. Spot for long
restartable batches; on-demand when a short run needs to actually finish.

---

## 4. Running a batch

```bash
python pipeline.py --xlsx leads.xlsx --campaign oct1 --kind report --workers 40
```

Five stages, each resumable because state lives in `cmi_newsletter.db`, not memory:

```
import    xlsx -> leads            (skips rows already present)
enrich    domains -> company_intel (once per DOMAIN, 90-day cache)
generate  leads -> briefs          (all briefs in ONE concurrency window)
validate  briefs.validation        (keyword rules + the fine-tune as judge)
send      briefs -> sends          (only briefs with no hard failure)
```

Useful flags:

| flag | effect |
|---|---|
| `--stage enrich` | run one stage only |
| `--send` | actually email — required, and deliberately not the default |
| `--dry-run-send` | show who would receive what, send nothing |
| `--spot` | resume onto spot when autoconnecting |
| `--no-autoconnect` | fail instead of healing the VM (use in CI) |

**40 workers is the measured sweet spot.** 64 is worse: requests queue instead of
batching. At 40, in one window, a brief costs **₹0.10** on-demand / **₹0.068** on spot.
Two sequential batches cost 2.8× one combined run.

Free-mail leads (gmail/yahoo/outlook) get **no scrape** — a personal address is not a
company website. Add those companies to `company_overrides.json` instead; curated
facts always win.

---

## 5. Watching it

```bash
streamlit run dashboard.py     # http://localhost:8501
python api.py                  # http://127.0.0.1:8080/docs
```

The dashboard leads with **CONNECTED / DISCONNECTED**, because it is the only live
number on the page — everything else is history. Connected means *a real chat
completion came back*, not that the process is up. It also warns when `.env` points at
a different endpoint than the running VM, which is the exact condition that produced a
batch of empty briefs.

The **Reconnect** button runs the same healing as `vm_connect.py --ensure`, streaming
progress. Safe to press twice. **⏸ Pause VM** stops billing when you are done.

### Generating from the dashboard

The **Generate** panel is the batch method: pick a campaign, a kind, and **how many
leads to process**. It states the real unit of work before you commit — "10 leads x 2
kinds" is 20 newsletters and 20x the cost, which a lead count alone hides.

It skips leads already briefed today by default (regenerating the same lead is almost
always a mis-click and costs a GPU minute), previews exactly who is in the batch, and
verifies the endpoint answers before starting — generating against a dead endpoint
produces empty briefs that still pass structural validation.

Generation and sending are deliberately **two separate buttons**. Generating is cheap
and repeatable; sending is neither. One button doing both is how a batch of unreviewed
briefs reaches real inboxes.

### Sending from the dashboard

The **Send** panel lists exactly who would receive what, then sends. It is deliberately
awkward to fire by accident, because email cannot be recalled:

- **The number on the button is exactly how many emails go out.** A slider picks how
  many of the sendable briefs to send, and the button carries that count.
- **Dry run is checked by default** — it shows the recipients and sends nothing.
- Unchecking it reveals a confirmation box: you must **type `SEND`** before the button
  arms.
- Unsubscribed addresses are listed as skipped, not as recipients, because that is
  what `stage_send` actually does with them.
- The transport in use (sendgrid / smtp) is shown before you commit.

It calls the same `pipeline.stage_send` as the CLI rather than reimplementing sending,
so the HTML that ships is the HTML that was validated, every outcome is recorded, and
one bad address cannot stop the batch.

| endpoint | what |
|---|---|
| `GET /vm/status` | live health: serving, machine id, GPU, whether `.env` matches |
| `POST /vm/reconnect` | heal and record the endpoint |
| `GET /analytics` | totals, verdicts, top failures, daily activity, throughput |
| `GET /briefs`, `/briefs/{id}/html` | list and preview |
| `POST /briefs/{id}/send` | send one — needs `{"confirm": true}` |

The API binds to `127.0.0.1` on purpose: nothing here authenticates a caller, so it
must not be reachable from another machine without a proxy that does.

To keep the endpoint alive unattended:

```bash
python vm_connect.py --watch --interval 120
```

---

## 6. Sending

Sending is the only irreversible stage, so it is gated at three levels: `--send` on the
CLI, `{"confirm": true}` on the API, and a send gate that refuses any brief with a hard
validation failure or an unsubscribed recipient.

SendGrid is used automatically whenever `SENDGRID_WEB_API` is set; SMTP is the
fallback. Force one with `CMI_MAIL_PROVIDER=sendgrid|smtp`.

**The SMTP path stays gated behind `CMI_SENDING_DISABLED=1`.** On 2026-09-16 roughly
40% of sends bounced `550 5.4.1` from Outlook/M365: raw Gmail SMTP from an
unauthenticated domain reads as bulk mail. Fix SPF/DKIM/DMARC before re-enabling it.
SendGrid is not gated by that switch, because it is the fix that block asked for.

If SendGrid returns **403**, the From address is not a verified sender on that account
— not a bad key. A mail-send-scoped key cannot query `/v3/verified_senders`, so this
only shows up on a real send. Verify the sender in the SendGrid dashboard after any
ownership change.

---

## 7. Tear down — every time

```bash
python vm_connect.py --status      # confirm what is running
python vm_connect.py --pause       # stop billing; keeps /home/merged
jl destroy <id>                    # everything gone; the adapter is safe locally
```

The dashboard has a **⏸ Pause VM** button that does the same thing.

An idle GPU is a 20× cost swing and the easiest way to burn the balance. The GPU holds
no state worth keeping: `/home/merged` is rebuildable from the local adapter.

---

## 8. Traps

| symptom | cause | fix |
|---|---|---|
| Briefs "PASS" but read as empty | every LLM call failed; empty sections are structurally valid | verify a real completion — `vm_connect.py --status` |
| `/v1/models` answers but generation 404s | that endpoint responds while the engine is still loading | poll `POST /v1/chat/completions`, which is what `is_serving` does |
| Endpoint dies partway through a run | server not started under tmux, or a spot reclaim | `vm_connect.py --ensure`; `ops/boot_serve.sh` uses tmux |
| `ModuleNotFoundError: vllm` after resume | a pause wipes site-packages; `/home` survives | `vm_connect.py --ensure` reinstalls |
| Resume "lost" the instance | resume returns a **new machine id** | never key on the id; find by name |
| `L4 not available` | transient capacity | retry, or let the GPU ladder fall through |
| `jl exec` exits 255 immediately | `pkill -f vllm` matched its own shell | do not pkill from inside `jl exec` |
| `jl list` UnicodeEncodeError | ₹ under cp1252 | use PowerShell, not Git Bash |
| Uploaded shell script misbehaves | CRLF line endings | `sed -i 's/\r$//' <file>` |
| Link works in browser, missing in email | `emailify` rebuilds the provenance row rather than copying it | re-emit the anchor; check emailified output |
| CTA goes nowhere | `/talktoanalyst` **404s** | it is `/talkwithanalyst` — a test pins this |

---

## 9. Verify the whole thing works

```bash
python -m pytest tests/test_units.py -q   # 65 unit tests, offline
python tests/test_e2e.py                  # whole pipeline, LLM stubbed
python vm_connect.py --status             # exits 0 only when serving
```

The first two need no GPU, no network and no SMTP, and each test uses its own temp
database. If they pass, the plumbing is sound and the only remaining variable is the
model endpoint.

For a real end-to-end check, generate two briefs and confirm the sections are grounded
and the links resolve:

```bash
python pipeline.py --xlsx "sample leads.xlsx" --campaign smoke --kind report --limit 2
```

Then read them. A brief that names the wrong competitors, or admits it found no
on-topic news, is telling you the enrichment or collection step needs attention — not
the model.
