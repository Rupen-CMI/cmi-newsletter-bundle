# Project layout

What each file is for, after the cleanup. Start with `VM_RUNBOOK.md` if you are picking
this up fresh.

## Documentation

| file | what it covers |
|---|---|
| `VM_RUNBOOK.md` | **Read first.** Cold laptop -> working endpoint. CLI paths, SSH key, resume, serve, tear down, nine traps, full architecture diagram |
| `MODEL_USAGE.md` | Operational manual. Serving, generating, validating, costs, every measurement taken (19 sections) |
| `HANDOFF.md` | Why each decision was made, with the evidence. For a new session |
| `SYSTEM_DESIGN.md` | The ten-step production system: what exists, what is missing, build order |
| `PLAN.md` | The original fine-tuning plan |

## Pipeline (the newsletter itself)

| file | role |
|---|---|
| `config.py` | Tenant profile, trigger taxonomy, scoring weights, lookback windows |
| `collect.py` | Google News RSS across trigger / industry / competitor queries |
| `score.py` | relevancy x freshness x importance, separate decay per bucket |
| `tenant.py` | Company profile: website fetch, curated overrides, query seeds |
| `brief_builder.py` | Section selection + LLM synthesis (customer / market moves / competition) |
| `compose.py` | Sections -> template context, both brief types |
| `voice.py` | The house voice, and `active_style()` which trims it for the fine-tune |
| `render.py` | Context -> HTML |
| `llm.py` | Provider routing (`local` / deepseek / openai) + voice-filtered training log |
| `geo.py`, `cluster.py`, `free_research.py`, `social_proof.py` | Geo filter, TF-IDF theming, enrichment, logo wall |

## Generation and validation

| file | role |
|---|---|
| `run_all40.py` | **Main runner.** All briefs in ONE concurrency window. 40 workers, Rs.0.10/brief |
| `batch_generate.py` | Per-kind batch runner, `--kind signal\|report`, `--outdir` |
| `validate_brief.py` | Structure / roles / fit. Exit 0 only when nothing FAILs |
| `llm_validate.py` | The fine-tune judging its own output. Advisory by default |
| `company_overrides.json` | Hand-curated facts per company. **Wins over anything inferred** |
| `company_intel.py` | smartcrawl JSON -> overrides, with a customer/competitor cross-check |

## Sending

| file | role |
|---|---|
| `db.py` | SQLite: leads, sends, settings, unsubscribes |
| `import_leads.py` | xlsx -> leads table |
| `campaign.py` | Column sniffing + campaign runner |
| `scheduler.py` | Sends in each lead's local window |
| `mailer.py` | SMTP from `.env` |
| `send_one.py` | Single brief, end to end |
| `app.py`, `campaign_dashboard.py` | Streamlit UIs |

## Ops and assets

| path | contents |
|---|---|
| `ops/serve_vllm.sh` | The vLLM launch line, tuned (max-num-seqs 64, util 0.94, chunked prefill) |
| `ops/merge_adapter.py` | Base + adapter -> `/home/merged`, fp16 on CPU |
| `ops/wait_vllm.sh` | Blocks until `/v1/models` answers |
| `assets/smartcrawl.py` | The crawler (unmodified upstream, two schema edits) |
| `assets/backend_shim.py` | Supplies the `backend.*` modules smartcrawl imports |
| `assets/run_crawl.py` | Crawl driver for the GPU box |

## Model artifacts

| path | what |
|---|---|
| `training_data/adapter/out/` | **v1 adapter — use this.** md5 `5918b998...` |
| `training_data/adapter_v2/` | v2, kept for reference only. Worse on generated briefs |
| `training_data/live_calls.jsonl` | 1,535 logged calls, voice-filtered going forward |
| `training_data/train*.jsonl` | Training sets, filtered and unfiltered |
| `training_data/remote/` | Training configs and `run_train.sh` |
| `training_data/serve_adapter.py` | Serve base+adapter without merging (4-bit) |

`training_data/adapter/out/checkpoint-{180,200,207}` are 1.4 GB of mid-training
snapshots. The finished adapter is already in `adapter_model.bin`, so they are only
useful for resuming that specific run. Safe to delete if you want the space back.

## Output

| dir | what |
|---|---|
| `output/` | Ad-hoc single briefs |
| `output_A40_*` | The winning config (40 workers, one window) |
| `output_account*`, `output_report*` | Validation-gate comparison, cited in `MODEL_USAGE.md` §18 |

## Removed in cleanup

Throwaway debug scripts (`_dbg*.py`, `_test*.py`, `_w*.sh`), the `_scratch/` folder, and
the losing AB-test output folders (B20, C64, D30 — their numbers live in
`MODEL_USAGE.md` §19). A `_key.sh` holding a plaintext API key was deleted earlier;
do not recreate that pattern — read credentials from `.env` at run time.
