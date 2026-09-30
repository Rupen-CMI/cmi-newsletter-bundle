# Handoff — self-hosted newsletter model

Read this first if you are a new session picking up this work. It records what was
decided, what was measured, and what is still open. `MODEL_USAGE.md` is the operational
manual; this is the reasoning behind it.

Last updated: 2026-09-21.

---

## The one-line state

The newsletter generates end to end on our own fine-tuned 7B — no DeepSeek, no OpenAI —
and beats DeepSeek on the house voice rules. What still needs human input is the
**competitor list per company**.

---

## Decisions, and the evidence behind each

### 1. Use adapter v1. Do not use v2.

`training_data/adapter/out/` — md5 `5918b9989c07ed75ba21fc898132ea94`.

v2 (`training_data/adapter_v2/`) was trained on a voice-filtered corpus (mean slop 20.2
→ 3.1) on the theory that the model had inherited DeepSeek's rule-breaking. Its eval loss
was better at every step (0.715 vs 0.836 at step 20; 0.400 final). **Its generated briefs
were worse**: 27.3 violations per 1000 words against v1's 21.7.

Why the eval number lied: v2's eval set was filtered too, so it scored against an easier
target. And filtering to slop ≤ 8 removed the long, complex examples that teach
brief-writing — short JSON classification outputs score 0 for having no prose to get
wrong, so they survived and came to dominate.

**Generalisable lesson: judge on generated briefs, not eval loss.** The eval set and the
training set share a bias; the detector on real output does not.

### 2. Serve in 4-bit, on a clean pytorch image.

4-bit is how the adapter was **trained** (QLoRA), so it is native, not a compromise. It is
also load-bearing: bf16 weights (~15 GB) left too little of a 24 GB L4 for activations,
and a 9.4k-token prompt died with `CUBLAS_STATUS_ALLOC_FAILED` at 22.5/23.0 GB. 4-bit
uses ~6 GB.

Never use the axolotl template for serving. Installing vLLM there upgrades torch
2.3.1 → 2.13.0 and breaks flash-attn, then cudnn, then flashinfer, in that order. Three
separate failures, all from one upgrade.

`training_data/serve_adapter.py` loads base + adapter at runtime rather than a pre-merged
15 GB copy — same weights, no merge step.

### 3. Always pass `--website`.

Profile inference from a company NAME alone is the worst failure in this pipeline.
ImmunoACT (CAR-T cell therapy) was profiled as an "immunoassay CDMO" — the model guessed
from the letters — which poisoned its competitors, keywords and every query until the
Potential Customer slot surfaced a **pet-food delivery company**.

`tenant.fetch_site_text()` fetches the homepage and marks it authoritative. That fixed
ImmunoACT completely. It does nothing when the homepage is empty (section 5).

### 4. Competition Outlook needed three fixes, not one.

The section kept printing a real rival doing something irrelevant (Novartis buying
blood-pressure brands, in a CAR-T maker's brief).

- **Lookback**: `COMPETITOR_LOOKBACK_HOURS = 24 * 182` (6 months) vs 96 hours for
  triggers. These are different kinds of fact — a buyer's funding round expires, a
  rival's product move does not. This was the root cause: **CAR-T news for India inside
  96 hours returns nothing at all** (freshest genuine story was ~28 days old), so the
  section could only ever fall back to noise. `score.py` has matching
  `_COMPETITOR_BANDS` so those articles are not scored as stale.
- **Attribution by headline, not query**: Google News answers "Bristol Myers Squibb
  India" with a Dr Reddy's headline. Matching on the query wrote that up as if BMS were
  the subject.
- **Rank on category, then newsworthiness**: sorting on composite alone let a rival's
  high-volume unrelated news win. Also scan the **whole bucket** — the on-category
  signals sat at rank 20+ and were never considered.

Result: **Immuneel Therapeutics raising ₹100 crore to scale CAR-T** instead of Novartis
buying Minipress. Verified to generalise: Paytm correctly gets **PhonePe**.

### 5. Competitor naming needs curation. This is not fixable by prompting.

Measured, not assumed:

- Same prompt, two calls: real CAR-T rivals once, a list of CDMOs the next
- Asked to audit its own list, it kept all five CDMOs and **invented "Cellgene Biotech"**
- For VVF (oleochemicals; contract-manufactures soap for J&J, Henkel, Reckitt) it named
  **Dentsu India**, an advertising agency
- In the smartcrawl extraction it listed **P&G and Reckitt as competitors** — they are
  VVF's customers — while listing "Procter & Gamble" as a client in the same response

It is reliable for companies it knows (Paytm → PhonePe). So: `company_overrides.json`
for the accounts that matter, inference for the rest.

### 6. Everything runs on our model, including crawl extraction.

This was tested, not assumed. I had claimed the 7B was unreliable at structured output
and argued for DeepSeek; that was an unfounded extrapolation from its *recall* weakness.
Head to head on VVF:

| | DeepSeek | cmi-7b |
|---|---|---|
| Completeness | 0.617 | **0.883** |
| Customers | generic categories | **Unilever, P&G, L'Oréal, J&J, Colgate** |
| Competitors | `[]` | 6 names, mostly wrong |
| Time | 16s | ~20 min |

Our model extracted 43% more fields. The cost is speed — fine for one-time enrichment,
not for anything interactive.

---

## What is built

| File | Purpose |
|---|---|
| `MODEL_USAGE.md` | Operational manual — read for commands |
| `training_data/serve_adapter.py` | OpenAI-compatible server, base + LoRA, 4-bit |
| `validate_brief.py` | Structure / roles / fit gate; exit 0 only on no FAILs |
| `company_overrides.json` | Hand-curated facts; wins over inference |
| `company_intel.py` | smartcrawl JSON → overrides, with a customer/competitor cross-check |
| `assets/backend_shim.py` | Lets smartcrawl import unmodified (no fork) |
| `assets/run_crawl.py` | Crawl driver for the GPU box |
| `batch_generate.py` | Leads spreadsheet → briefs + validation table |
| `training_data/filter_dataset.py` | Corpus filter (used for v2; keep for reference) |

Pipeline changes live in `llm.py` (local provider, voice-filtered logging),
`tenant.py` (website fetch, overrides), `brief_builder.py` (competition + buyer guards),
`collect.py` / `score.py` (competitor lookback), `voice.py` (`active_style` token trim),
`compose.py` (project-card repair).

---

## Measured numbers worth keeping

- **Brief**: ~19,000 tokens, ~11 LLM calls, ~4 min on an L4
- **Cost**: ~₹0.92/brief sequential on spot; ~₹0.12 with vLLM batching. **Idle GPU is a
  20× swing** — always destroy after a batch
- **Voice vs DeepSeek** (same ImmunoACT brief): triad 9 vs 21, em-dash 1 vs 10,
  passive 0 vs 10
- **Held-out eval**: mean slop 12.6 (ours) vs 12.8 (DeepSeek), 15/20 wins
- **Batch of 5**: 63 calls, zero failures, 1/5 passed validation cleanly

---

## Running the model: adapter vs merged

Default is **adapter at runtime** (`serve_adapter.py`) -- 323 MB, no merge step, and what
every result in these notes used. Merge only for vLLM, which cannot apply a LoRA cheaply
on the fly. `MODEL_USAGE.md` section 13 has all three routes and the traps:

- merge needs the base in **fp16, not 4-bit**, or the adapter's precision is lost
- axolotl's `merge_lora` reads `load_in_4bit` from the YAML; passing it again on the CLI
  raises a duplicate-kwarg error (this broke both training runs)
- a merged 15 GB model is **reconstructible** from the adapter -- never archive it

---

## Open items

1. **Curate overrides for VVF, Entrust, Sable Sanjivani.** VVF's real rivals are
   contract manufacturers and oleochemical producers — Godrej Industries, KLK Oleo,
   Wilmar, IOI Oleochemicals, Galaxy Surfactants, Nirma — not the FMCG brands it
   manufactures *for*.
2. **Re-run the VVF crawl** with the improved `INTEL_SCHEMA` description (the
   `competitors` field was a bare `_arr()` with no definition; it now has one, including
   permission to return an empty array). Untested.
3. **vLLM batching** for the ₹0.12/brief figure. Needs a clean image.
4. **`validate_brief.py` role check is inconsistent** — it passed one VVF brief and
   flagged an identical one, because the heuristic depends on which words appear in the
   section text.
5. **Run-to-run variance**: the same company can lose a section between runs depending on
   what the feed returns. The validator catches it; regeneration is the fix.

---

## Traps that cost time

- **CRLF** breaks every uploaded shell script. `sed -i 's/\r$//'` after upload.
- **`jl exec` fails intermittently** on a healthy box — do not read that as preemption.
  Check `jl list --json`. (A monitor did exactly this and cried wolf.)
- **`jl list` crashes in Git Bash** on the ₹ symbol. Use PowerShell.
- **Resume changes the machine ID.** Use the one `jl resume` returns.
- **Pause kills the server** and only `/home` survives.
- **smartcrawl's `EXTRACTION_TIMEOUT_S` is gated behind `DEEPSEEK_BASE_URL` being set** —
  point it at the *local* endpoint to unlock the longer timeout. Without it the timeout
  is hardcoded to 30s and a local 7B never finishes.
- **A crashed server does not release GPU memory.** `pkill -9 -f serve_adapter` first.

---

## Scratch

`_scratch/` holds throwaway debug scripts from the build. Safe to delete. A `_key.sh`
containing a plaintext API key was removed — do not recreate that pattern; read keys from
`.env` via `dotenv_values` at run time.
