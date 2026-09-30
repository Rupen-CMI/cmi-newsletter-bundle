# CMI Newsletter System

Generates and sends personalised market-intelligence briefs for Coherent Market Insights.
Every word is written by our own fine-tuned Qwen2.5-7B — no DeepSeek, no OpenAI.

Self-contained: code, the model adapter, docs, tests, the live database and sample
output. Clone it to a new machine, fill in `.env`, and run.

---

## 5-minute start

```bash
pip install -r requirements.txt
cp .env.example .env          # then fill in LOCAL_LLM_BASE_URL and SENDGRID_WEB_API

python -m pytest tests/test_units.py -q   # 65 tests, offline, no GPU needed
python tests/test_e2e.py                  # whole pipeline, LLM stubbed
```

Both suites run with the GPU switched off. If they pass, the plumbing is sound and
the only remaining variable is the model endpoint.

To generate for real you need the model served. One command finds the VM by name,
resumes it, installs vLLM if a pause wiped it, starts the server, and records the
endpoint in `.env`:

```bash
python vm_connect.py --ensure     # ~6 min cold, instant if already serving
streamlit run dashboard.py        # green CONNECTED banner + analytics
```

Then:

```bash
python pipeline.py --xlsx "sample leads.xlsx" --campaign oct1 --kind report \
    --workers 40 --base-url https://<vllm-host>/v1
```

Sending is deliberately NOT part of that run. It is the one irreversible stage, so it
needs `--send` said out loud:

```bash
python pipeline.py --campaign oct1 --send --dry-run-send   # prove it first
python pipeline.py --campaign oct1 --send
```

---

## What is in here

| path | what |
|---|---|
| `pipeline.py` | the orchestrator: import → enrich → generate → validate → send |
| `db.py` | SQLite schema and every query. `leads`, `briefs`, `runs`, `sends`, `company_intel` |
| `enrich.py` | per-domain company facts, cached 90 days |
| `collect.py` `score.py` | Google News RSS collection, then relevancy × freshness × importance |
| `brief_builder.py` `compose.py` `render.py` | the three sections, then template context, then HTML |
| `validate_brief.py` `llm_validate.py` | keyword rules, then the fine-tune judging its own output |
| `mailer.py` `emailify.py` | SendGrid/SMTP, and browser-HTML → email-safe HTML |
| `api.py` | FastAPI over the database (`python api.py` → :8080/docs) |
| `vm_connect.py` | finds the VM by name, heals a dead endpoint, records it in `.env` |
| `dashboard.py` | ops frontend: live VM health + analytics (`streamlit run dashboard.py`) |
| `tenant.py` `config.py` `geo.py` | company profiles, markets, country filtering |
| `templates/` | `cmi_brief.html` is the live one |
| `training_data/adapter/` | **the fine-tuned adapter** — 323 MB, md5 `5918b9989c07ed75ba21fc898132ea94` |
| `training_data/*.jsonl` | the 559 house-voice training pairs |
| `ops/` | VM boot, serve, and adapter-merge scripts |
| `docs/` | see below |
| `sample_output/` | four real briefs, links verified |
| `cmi_newsletter.db` | the live database: 2,600+ leads, logged sends, cached intel |

**The adapter is the one irreplaceable file here.** Everything else is code or is
reconstructible; the adapter is the output of a training run. `/home/merged` on the VM
can be rebuilt from it (`docs/MODEL_USAGE.md` §13).

---

## Docs, in reading order

| file | read it when |
|---|---|
| `docs/OPERATIONS.md` | **start here.** End to end on a new machine or under new ownership: credentials, standing the model up from nothing, running a batch, watching it, sending, tear-down, and what protects output quality |
| `docs/VM_RUNBOOK.md` | the raw VM procedure `vm_connect.py` automates — Cold laptop → working endpoint. CLI paths, SSH key, resume, serve, tear down, and every trap that has already cost time |
| `docs/SYSTEM_DESIGN.md` | what each stage does and why it is shaped that way |
| `docs/MODEL_USAGE.md` | the model itself: training, evaluation, quality and cost numbers |
| `docs/HANDOFF.md` | picking up mid-stream in a new session |
| `docs/PROJECT_LAYOUT.md` | file-by-file map |

---

## Things that will bite you

**`/talkwithanalyst`, not `/talktoanalyst`.** Checked against the live site:
`/talkwithanalyst` → 200, `/talktoanalyst` → 404. A test pins this, because the wrong
one looks more natural and would break every CTA in every brief.

**`emailify` rebuilds the provenance row rather than copying it.** A link added to the
template renders fine in the browser file and silently vanishes from the sent email.
Check the emailified output, not just the generated HTML.

**`/v1/models` answering does not mean the model is serving.** It can respond while
the engine is not ready. Poll `POST /v1/chat/completions`. A whole run was once
generated against a dead endpoint and still reported `PASS`, because empty sections
are structurally valid.

**A pause wipes pip installs.** `/home` survives, site-packages does not, so vLLM
needs reinstalling (~4 min) after every resume. `ops/boot_serve.sh` handles it.

**Spot instances get reclaimed.** Three times in one session on 2026-09-23, each
costing a reinstall. Spot for long restartable batches; on-demand when you need a
short run to actually finish.

**Free-mail leads get no scrape.** `enrich.domain_of` refuses gmail/yahoo/outlook —
a personal address is not a company website, and guessing from one is how a CAR-T
maker got profiled as an "immunoassay CDMO". Curate those in
`company_overrides.json` instead.

**Sending is real.** `pipeline.py` needs `--send`; the API needs `{"confirm": true}`
and refuses any brief carrying a hard validation failure or an unsubscribed address.

---

## Cost

**₹0.10 per brief** on-demand, **₹0.068** on spot, at 40 workers in one concurrency
window. 40 is the measured sweet spot — 64 is worse, because requests queue instead
of batching. Two sequential batches cost 2.8× one combined run.

Never leave the GPU idle: it is a 20× cost swing and the easiest way to burn the
balance.

---

## Security

`.env` is NOT in this bundle — only `.env.example`. Fill it in locally and keep it
out of version control. It holds a live SendGrid key and SMTP credentials.

If a key is ever printed to a terminal or pasted anywhere, rotate it rather than
hoping. A SendGrid key is a bearer credential: whoever holds it can send as your
domain.
