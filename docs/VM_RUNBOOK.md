# VM runbook — connect, serve, generate, tear down

Everything needed to get from a cold laptop to a working endpoint, with no
reconfiguration. Follow it top to bottom.

**On credentials:** this file records paths, key names and fingerprints — never secrets.
The Jarvislabs API token lives in the `jl` CLI config, and `DEEPSEEK_API_KEY`,
`SMTP_PASSWORD` and `SENDGRID_WEB_API` live in `.env`. A markdown file is the wrong place for a password: an
earlier debug script here (`_key.sh`) carried a plaintext API key and had to be deleted.
Section 2 shows how to read them at run time instead.

---

## 1. Fixed facts

| | |
|---|---|
| Provider | Jarvislabs.ai — account `coherentdrive@gmail.com` |
| CLI | `jl` v0.2.17+ (`pip install jarvislabs`) |
| CLI binary | `%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts\jl.exe` |
| SSH key | `~/.ssh/id_ed25519` — registered as **`aditya-win`**, id `aZa1sScp` |
| Key fingerprint | `SHA256:5ejO55OTwUTax8BcI8P08ry2JqG/SfnphzoMgffZAwU` |
| Instance name | `cmi-serve-v1` |
| GPU / template | L4 24 GB / **pytorch** (never axolotl — see §8) |
| Region | IN2 |
| Model on disk | `/home/merged` (15 GB, survives pause) |
| Adapter backup | `training_data/adapter/out/` locally, md5 `5918b9989c07ed75ba21fc898132ea94` |

**The machine ID changes on every resume.** Never hardcode it; always read it from
`jl list` first.

---

## 2. Laptop setup (once)

```powershell
pip install jarvislabs
$env:Path += ";$env:LOCALAPPDATA\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts"
jl status        # already authenticated; if not: jl setup
```

Two things that will waste time if you forget them:

- **Run `jl` from PowerShell, not Git Bash.** Git Bash mangles remote paths
  (`/home/x` becomes `C:/Users/.../Git/home/x`) and `jl list` crashes on the ₹ symbol
  under cp1252.
- **Each PowerShell call is a fresh process**, so prepend the `$env:Path` line every
  time, or add the Scripts directory to your user PATH permanently.

Reading secrets at run time, rather than pasting them anywhere:

```python
from dotenv import dotenv_values
key = dotenv_values(".env")["DEEPSEEK_API_KEY"]
```

---

## 3. Bring the VM up

```powershell
jl list                          # find the CURRENT id and status
jl resume <id> --gpu L4 --yes    # if Paused -> returns a NEW id
# or, if no instance exists:
jl create --gpu L4 --spot --template pytorch --storage 60 --name cmi-serve-v1 --yes
```

`--spot` is ~Rs.27.5/hr against ~Rs.41.3 on-demand, and can be reclaimed without warning.
Fine for batch work — everything important is already on your laptop.

Then read the endpoint (it is `null` while paused, which is the usual source of
confusion):

```powershell
jl get <id> --json | Select-String -Pattern '"endpoints"' -Context 0,2
# -> https://<hash>.notebooksn.jarvislabs.net     (port 6006, publicly exposed)
```

Direct SSH, if you want a shell:

```powershell
jl get <id> --json | Select-String '"ssh_command"'
# -> ssh -o StrictHostKeyChecking=no root@<ip>
```

---

## 4. Serve the model

**A pause wipes every pip install.** `/home` survives, site-packages does not — so vLLM
must be reinstalled after each resume (~4 min). `/home/merged` is still there.

```powershell
jl exec <id> -- bash -lc "nohup pip install vllm > /home/vllm_reinstall.log 2>&1 & echo started"
# wait ~4 min, then:
jl upload <id> "<repo>\ops\serve_vllm.sh" /home/serve_vllm.sh
jl exec <id> -- bash -lc "cd /home && sed -i 's/\r$//' serve_vllm.sh && bash serve_vllm.sh"
```

`ops/serve_vllm.sh` runs:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /home/merged --served-model-name cmi-7b \
  --max-model-len 4096 --gpu-memory-utilization 0.94 \
  --max-num-seqs 64 --enable-prefix-caching --enable-chunked-prefill --port 6006
```

Ready when `/v1/models` answers with `cmi-7b`. Expect `GPU KV cache size: ~96,048 tokens`
in `/home/vllm.log`.

If `/home/merged` is ever gone, rebuild it from the local adapter — see
`MODEL_USAGE.md` §13.

---

## 5. Generate

```powershell
cd <repo>
python run_all40.py --workers 40 `
    --account-dir output_account --report-dir output_report `
    --base-url https://<hash>.notebooksn.jarvislabs.net/v1
```

40 briefs in ~6 min at **Rs.0.10 each on-demand, Rs.0.068 on spot**. Use `--workers 30`
when the batch ships without human review (§19 of `MODEL_USAGE.md` has the numbers).

With the LLM judge:

```powershell
$env:CMI_LLM_VALIDATE = "1"    # advisory by default; CMI_LLM_STRICT=1 to block on role calls
```

---

## 6. Tear down

```powershell
jl pause <id>      # keeps /home/merged, small storage charge, KILLS the server
jl destroy <id>    # everything gone; adapter is safe locally
```

**Always do one of these.** An idle GPU is a 20x cost swing, and it is the single
easiest way to burn the balance.

---

## 7. First-run smoke test

```powershell
python -c @"
from openai import OpenAI
c = OpenAI(api_key='local', base_url='https://<hash>.notebooksn.jarvislabs.net/v1')
print([m.id for m in c.models.list().data])
"@
```

Expect `['cmi-7b']`. A 502 means the server is still loading or the box is paused.

---

## 8. Traps, each of which has already cost time here

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: vllm` after resume | pause wipes site-packages | reinstall (§4) |
| 404 / 502 from the endpoint | box paused, or server not started | `jl list`, then §4 |
| vLLM fails 3x on install | **axolotl template** upgrades torch under a prebuilt image | use the pytorch template |
| `CUDA out of memory` | a crashed server still holds VRAM | `pkill -9 -f vllm.entrypoints` |
| Remote path becomes `C:/Users/.../Git/home/...` | Git Bash path translation | use PowerShell |
| `jl list` UnicodeEncodeError | ₹ symbol under cp1252 | use PowerShell |
| Uploaded shell script fails oddly | CRLF line endings | `sed -i 's/\r$//' <file>` after upload |
| `jl exec` fails intermittently | flaky transport, **not** preemption | check `jl list --json` before assuming |
| Resume "lost" the instance | resume returns a **new machine id** | use the id `jl resume` printed |
| `jl exec` exits 255 immediately | `pkill -f vllm` matched its own shell | don't pkill from inside `jl exec`; a fresh resume has nothing running |
| Endpoint 404s partway through a run | server was not started under tmux, or the **spot** box was reclaimed | relaunch under tmux (§4); check `jl list` |
| `L4 not available` on resume | transient, not an account limit | retry; A30 is the same 24GB at the same spot price, A100 is ~3x |
| Briefs "PASS" but read as empty | every LLM call failed; empty sections still pass structure checks | verify a real completion before generating |

---

# System architecture

```
  ┌────────────────────────────── LAPTOP (Windows) ──────────────────────────────┐
  │                                                                              │
  │  leads.xlsx                                                                  │
  │      │                                                                       │
  │      ▼                                                                       │
  │  import_leads.py ──► cmi_newsletter.db                                       │
  │   (campaign.py         ├── leads          2,606 rows                         │
  │    sniffs columns      ├── company_intel  cached scrape, 90-day expiry  [NEW] │
  │    by content)         ├── briefs         html + validation             [NEW] │
  │                        ├── sends          328 rows                           │
  │                        ├── runs           throughput + cost            [NEW] │
  │                        └── settings       per-campaign SMTP                  │
  │                             │                                                │
  │  ┌──────────────────────────┼────────────────────────────────────────────┐   │
  │  │ pipeline.py  (orchestrator, resumable — state lives in the DB)  [NEW] │   │
  │  │                          │                                            │   │
  │  │  1 ENRICH   tenant.fetch_site_text ──► under 400 chars? ──► smartcrawl│   │
  │  │             once per DOMAIN, never per brief                          │   │
  │  │                          │                                            │   │
  │  │  2 COLLECT  collect.py ── Google News RSS ──┐                         │   │
  │  │             trigger 182d │ industry 96h │ competitor 182d             │   │
  │  │                          ▼                                            │   │
  │  │  3 SCORE    score.py — relevancy x freshness x importance             │   │
  │  │             separate decay curves per bucket                          │   │
  │  │                          │                                            │   │
  │  │  4 BUILD    brief_builder.py — 40 briefs in ONE pool, 40 workers      │   │
  │  │             ~11 sequential LLM calls per brief, briefs independent    │   │
  │  └──────────────────────────┼────────────────────────────────────────────┘   │
  │                             │  OpenAI-compatible HTTP                        │
  └─────────────────────────────┼────────────────────────────────────────────────┘
                                │
  ┌─────────────────────────────▼──────── JARVISLABS L4 (spot) ──────────────────┐
  │  vLLM 0.30  ·  port 6006  ·  KV cache ~96k tokens                            │
  │  /home/merged = Qwen2.5-7B-Instruct + our QLoRA adapter (v1)                 │
  │  continuous batching · prefix caching · chunked prefill                      │
  │  NO DeepSeek, NO OpenAI — every call is ours                                 │
  └─────────────────────────────┬────────────────────────────────────────────────┘
                                │
  ┌─────────────────────────────▼────────────────────────────────────────────────┐
  │  5 VALIDATE   validate_brief.py   structure · roles · fit   (keyword rules)   │
  │               llm_validate.py     the fine-tune judging its own output        │
  │                 method 1 passage-only  ──► catches a CUSTOMER filed as rival  │
  │                 method 3 describe+yes/no ─► catches an out-of-industry name   │
  │               advisory by default: only structure blocks                      │
  │                             │                                                 │
  │  6 PERSIST    briefs table — html, validation json, timings           [NEW]   │
  │                             │                                                 │
  │  7 SEND       mailer.py — SMTP from .env or per-campaign settings             │
  │               scheduler.py — each lead's local window                         │
  │                             │                                                 │
  │  8 LOG        sends + runs ──► FastAPI ──► dashboard                  [NEW]   │
  └──────────────────────────────────────────────────────────────────────────────┘
```

## Why it is shaped this way

**Scraping is cached per domain, not per brief.** smartcrawl takes ~20 min per company
on the local model. 2,606 leads share far fewer domains, so this is the difference
between hours and minutes — and it is what stops the model guessing, which produced
"immunoassay CDMO" for a CAR-T maker.

**Briefs run in one concurrency window.** The ~11 calls inside a brief are sequential
(call N+1 needs call N), but briefs are independent — so wall clock is bounded by the
*slowest single brief*, not total work. Once workers ≥ briefs, adding briefs is nearly
free: Rs.0.286 → Rs.0.102 with no server change. Measured sweet spot is **40 workers**;
64 is worse because requests queue instead of batching.

**Validation is two-layer and advisory.** Keyword rules catch structure; the LLM judge
catches meaning a rule cannot — Reckitt filed as VVF's rival when VVF manufactures for
them. Advisory by default because a role call is a judgement a human can overrule, and
nothing is hidden: every finding still prints.

**The GPU holds no state worth keeping.** `/home/merged` is reconstructible from the
323 MB adapter on your laptop. Destroy freely.
