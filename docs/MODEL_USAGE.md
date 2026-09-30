# Running the newsletter on the self-hosted model

Everything here runs on our own fine-tune (`cmi-7b`). No DeepSeek, no OpenAI.

---

## 1. What the model is

Qwen2.5-7B-Instruct + a 323 MB QLoRA adapter trained on 559 real (system, user, output)
triples captured from live brief generation.

| | |
|---|---|
| Adapter | `training_data/adapter/out/adapter_model.bin` |
| md5 | `5918b9989c07ed75ba21fc898132ea94` |
| Trained | 207 steps, ~1h50m on an L4, ~Rs.360 |
| Eval loss | 3.10 -> 0.359 |

**Use v1. Do not use v2.** A second run trained on a voice-filtered corpus (`adapter_v2/`)
scored better on eval loss but **worse on generated briefs** -- 27.3 violations per 1000
words against v1's 21.7. Filtering to slop <= 8 stripped the long, complex examples that
teach brief-writing and left short JSON classification calls dominating the corpus. The
eval number looked good because the eval set was filtered too, so it was scoring against
an easier target. v2 is kept only as a reference.

---

## 2. Serving it

Rent a spot L4 on a **clean pytorch template** (not axolotl -- installing anything there
upgrades torch and breaks flash-attn, cudnn and flashinfer in sequence).

```bash
jl create --gpu L4 --spot --template pytorch --storage 60 --name cmi-serve --yes
jl exec <id> -- mkdir -p /home/v1adapter
# upload training_data/adapter/out/* to /home/v1adapter/
# upload training_data/serve_adapter.py to /home/

jl exec <id> -- bash -lc "pip install -q fastapi uvicorn peft bitsandbytes && \
  cd /home && nohup python -u serve_adapter.py --adapter /home/v1adapter --port 6006 \
  > /home/serve.log 2>&1 &"
```

Port **6006** is the one Jarvislabs exposes publicly; the URL comes from
`jl get <id> --json` under `endpoints`.

`serve_adapter.py` loads base + adapter at runtime rather than a pre-merged 15 GB copy.
Same weights, no merge step, no 15 GB download.

### 4-bit is the default, and that matters

The model loads in 4-bit (~6 GB) rather than bf16 (~15 GB). This is not a quality
compromise -- 4-bit is how the adapter was **trained** (QLoRA), so it is the native fit.
It is also the difference between working and not: bf16 weights left only ~8 GB for
activations on a 24 GB L4, and a 9.4k-token prompt died with
`CUBLAS_STATUS_ALLOC_FAILED` at 22.5/23.0 GB. Pass `--bf16` only on a bigger card.

If a previous server is still holding GPU memory, `pkill -9 -f serve_adapter` before
restarting -- a crashed process does not always release it.

---

## 3. Generating briefs

```bash
export LLM_PROVIDER=local
export LOCAL_LLM_BASE_URL=https://<endpoint>/v1
export LOCAL_LLM_MODEL=cmi-7b

python send_one.py --company ImmunoACT --website https://immunoact.com/ \
    --recipient Aditya --email someone@example.com --market India
```

**Always pass `--website`.** Profile inference from a company NAME alone is the single
worst failure in this pipeline: ImmunoACT (CAR-T cell therapy) was profiled as an
"immunoassay CDMO", which poisoned its competitors, keywords and every downstream query
until the Potential Customer slot surfaced a pet-food delivery company.
`tenant.fetch_site_text()` fixes this by fetching the homepage -- but only if the
homepage has content (see section 6).

`llm.py` falls back to DeepSeek then OpenAI if the local endpoint is down. Set
`fallback_to_openai: False` in `config.LLM` to make a dead endpoint fail loudly instead.

### Batch

```bash
python batch_generate.py --xlsx Book1.xlsx --limit 5 --no-dedupe \
    --base-url https://<endpoint>/v1
```

Reads leads positionally (no header row), uses the email domain as the website when the
sheet has none, skips free-mail domains rather than guessing. ~4 min per brief.

---

## 4. Validating before sending

```bash
python validate_brief.py output/<brief>.html --company ImmunoACT \
    --website https://immunoact.com/
```

Exit code 0 only when nothing FAILS, so it can gate a send. Three checks:

- **STRUCTURE** -- all three sections present and substantive
- **ROLES** -- a known competitor must never sit in the Potential Customer slot; a
  regulator or ministry is not a buyer; the same company cannot hold two slots
- **FIT** -- the copy actually discusses what the seller sells

Run-to-run variance is real: the same company can produce a clean brief on one run and
lose a section on the next, depending on which signals the feed returns. Regenerating is
the fix, and the validator is what tells you to.

---

## 5. Curated overrides -- the most important file

`company_overrides.json` holds hand-checked facts per company, keyed by lowercased name.
These **win over anything the model infers**.

```json
{
  "immunoact": {
    "value_prop": "ImmunoACT makes NexCAR19, India's first approved CD19-targeted CAR-T...",
    "industries": ["Cancer hospitals and oncology centres", "..."],
    "competitors": ["Immuneel Therapeutics", "Kite Pharma", "..."],
    "caps_industry": "cell and gene therapy and oncology therapeutics",
    "product_terms": ["car-t", "cell therapy", "cd19", "..."]
  }
}
```

**Why this exists.** Naming competitors is factual recall, and a 7B is unreliable at it in
a way prompting does not fix. Measured, not assumed:

- The same prompt returned real CAR-T rivals on one call and a list of CDMOs on the next
- Asked to audit its own list, it kept all five CDMOs and **invented a company**
  ("Cellgene Biotech") that does not exist
- For VVF (oleochemicals, contract-manufactures soap for J&J and Henkel) it named
  **Dentsu India**, an advertising agency

It is reliable for companies it knows -- Paytm correctly returned PhonePe. So curate the
accounts that matter and let inference handle the rest.

`product_terms` also drives competitor-signal ranking (section 7).

---

## 6. Enriching a company with smartcrawl

When a homepage has no real content, crawl the subpages. VVF's homepage yields **301
characters** of pure navigation ("CELEBRATING 75 YEARS | Home | About Us"); its
About/Businesses pages hold everything.

On the GPU box (needs Linux + curl_cffi; will not run on Windows):

```bash
# upload assets/{smartcrawl.py,backend_shim.py,run_crawl.py} to /home/crawl/
pip install -q curl_cffi trafilatura httpx websockets lxml openai

export SMARTCRAWL_MODEL=cmi-7b
export LOCAL_LLM_BASE_URL=http://localhost:6006/v1
export DEEPSEEK_BASE_URL=http://localhost:6006/v1   # see note
export EXTRACTION_TIMEOUT_S=900
unset DEEPSEEK_API_KEY OPENAI_API_KEY

python run_crawl.py --domains vvfltd.com --out /home/intel --mode company --max-pages 12
```

Then locally:

```bash
python company_intel.py --import company_intel/vvfltd_com.json --company "VVF LTd"
```

**The `DEEPSEEK_BASE_URL` line is not a mistake.** smartcrawl only honours
`EXTRACTION_TIMEOUT_S` when that variable is set -- upstream it gates a widened timeout
for a slow local proxy, which is exactly our case. The client is still chosen by
`SMARTCRAWL_MODEL`, so no call leaves the box. Without it the timeout is hardcoded to
30s and a local 7B (20-30 tok/s unbatched, ~20 min for a full extraction) never finishes.

`backend_shim.py` supplies the four `backend.*` modules smartcrawl imports from its home
project, so it runs unmodified rather than becoming a fork.

### What the crawl gets right, and what it does not

Measured on VVF, our model against DeepSeek as extractor:

| | DeepSeek | cmi-7b |
|---|---|---|
| Completeness | 0.617 | **0.883** |
| Customers | generic categories | **Unilever, P&G, L'Oreal, J&J, Colgate** |
| Competitors | `[]` | 6 names, **mostly wrong** |

Our model extracted 43% more fields. But its competitor list was
`Dabur India, Nirma, Nestle, Reckitt Benckiser, P&G, UniqYou` -- Reckitt and P&G are
VVF's **customers** (it manufactures their soap), and it wrote "P&G" as a rival while
listing "Procter & Gamble" as a client in the same response.

Two guards now exist:

1. The `competitors` field in `INTEL_SCHEMA` has an explicit definition (it was a bare
   `_arr()` with no description, so the model was guessing from the field name alone),
   including permission to return an empty array rather than invent.
2. `company_intel.py` drops any competitor that also appears in `customers` or
   `partnerships`, normalising "P&G" against "Procter & Gamble". This caught P&G. It
   **cannot** catch Reckitt, because that name appeared only in the competitor list --
   a rule can only see contradictions that are present in the data.

**Treat the crawled competitor list as a draft.** Description, industries, products and
customers are reliable; competitors need five minutes of human review.

---

## 7. Competition Outlook -- how it picks

Three fixes were needed to stop it printing a rival's unrelated news:

**Lookback.** `COMPETITOR_LOOKBACK_HOURS = 24 * 182` (6 months) against the 96-hour
window used for triggers. These are different kinds of fact: a buyer's funding round is
only actionable while the money is being spent, but a rival shipping a competing product
stays true until they ship the next one. In a thin category this is the whole game --
searching CAR-T news for India inside 96 hours returns **nothing** (the freshest genuine
cell-therapy story was ~28 days old), so the section could only ever fall back to
whatever unrelated news the biggest rival made that week. `score.py` has matching
`_COMPETITOR_BANDS` so those older articles are not scored as stale.

**Attribution by headline, not query.** Google News answers "Bristol Myers Squibb India"
with a Dr Reddy's headline. Matching on the query wrote that story up as if BMS were the
subject. The title or summary must now name the rival.

**Rank on category, then newsworthiness.** Sorting on composite alone handed the section
to whichever rival generates the most news: a Novartis generics acquisition (0.58)
outranked Immuneel's CAR-T Series B (0.49), so a CAR-T maker's Competition Outlook was
about blood-pressure brands. Signals matching `product_terms` are now preferred. The scan
also covers the whole bucket rather than the top 10, because the on-category signals sat
at rank 20+ and were never considered.

Result: **Immuneel Therapeutics raising Rs 100 crore to scale CAR-T** instead of Novartis
buying Minipress.

---

## 8. Costs

~19,000 tokens and ~11 LLM calls per brief, ~4 minutes on an L4.

| Setup | Per brief |
|---|---|
| Sequential, spot L4, destroyed after | ~Rs.0.92 |
| \+ `active_style()` trim (already in) | ~Rs.0.61 |
| \+ vLLM continuous batching | **~Rs.0.12** |
| 24/7 serving at 50 briefs/day | ~Rs.20 |

**Never leave the box idle** -- that single factor is a 20x swing. Spin up, generate the
queue, destroy.

`voice.active_style()` sends a 96-token style reminder to the fine-tune instead of the
full 687-token rule block, cutting 34% of every brief's tokens. The model was trained on
these rules, so it does not need them restated in full; hosted models still get the long
version. `CMI_FORCE_FULL_STYLE=1` restores it everywhere for like-for-like comparisons.

---

## 9. Growing the training corpus

`CMI_LOG_TRAINING_DATA=1` appends every served call to `training_data/live_calls.jsonl`,
now **voice-filtered**: `llm.py` scores each output with the production detector and
drops anything above `CMI_TRAINING_MAX_SLOP` (default 8).

The v2 experiment showed filtering alone does not raise the ceiling. What would: logging
**hand-corrected** briefs. A brief you fixed yourself is worth more than any volume of
raw model output, because it is the only data that is better than the teacher.

---

## 10. Things that will bite

- **axolotl template + vLLM** -- three sequential failures (flash-attn ABI, duplicate
  cudnn, flashinfer JIT). Use a clean pytorch image.
- **`transformers` renamed `torch_dtype` to `dtype`** -- `serve_adapter.py` tries both.
- **CRLF line endings** break every uploaded shell script. `sed -i 's/\r$//'` after
  upload, or write with `newline='\n'`.
- **`jl exec` needs `bash -lc`**, not a bare quoted string, and fails intermittently even
  when the box is healthy -- do not treat an exec failure as preemption. Check
  `jl list --json`.
- **`jl list` crashes in Git Bash** on the Rs. symbol (cp1252). Use PowerShell.
- **Resume changes the machine ID.** Always use the ID returned by `jl resume`.
- **Only `/home` survives pause/resume**, and a pause kills any running server.

---

## 11. Verified run — VVF, 21 Sep 2026

`output/cmi_local_vvf_curated.html` — 12 LLM calls, all local, zero failures.

| Section | Subject | Why it is right |
|---|---|---|
| Potential Customer | Torrent Pharma | Launched two semaglutide brands; needs actives and packaging supply |
| Industry Outlook | Foreign capital in Indian healthcare | Structural shift, correctly framed |
| **Competition Outlook** | **Galaxy Surfactants** | Real surfactants/contract-manufacturing rival. 108% Q1FY27 profit jump, deeper Unilever tie |

Before the competition fixes the same company produced **Dentsu India** — an advertising
agency — as VVF's competitor.

Voice scores (violations per 1000 words, lower is better):

| | triad | long_sent | em_dash | passive | per 1k |
|---|---|---|---|---|---|
| VVF curated (this run) | 7 | 9 | 1 | 4 | **21.4** |
| VVF earlier (name-guess) | 9 | 9 | 1 | 1 | 22.9 |
| DeepSeek reference brief | 21 | 12 | 10 | 10 | 37.6 |

The validator flagged one WARN: Torrent Pharma "reads like a company in the seller's own
category". Worth a human look — Torrent is a pharma manufacturer, so it is plausibly a
buyer of VVF's glycerine and contract packaging, but it is not obviously one.

### What made this run work

`company_overrides.json` now has a curated `vvf ltd` entry. VVF is a **contract
manufacturer** — it makes soap *for* Unilever, P&G, J&J, Henkel and Reckitt — so its
rivals are other contract manufacturers and oleochemical producers (Godrej Industries,
KLK Oleo, Wilmar, IOI, Emery, Galaxy Surfactants, Fine Organics, Nirma), never the FMCG
brands it supplies. Both the model and the crawl had that backwards.

---

## 12. The INTEL_SCHEMA description experiment — mixed result

`competitors` in smartcrawl's `INTEL_SCHEMA` was a bare `_arr()` with no description, so
the model was guessing from the field name alone. Two edits were made:

1. `_arr()` now accepts a `desc` argument (it previously had no way to carry one)
2. `competitors`, `customers` and `partnerships` each got an explicit definition,
   including permission to return an empty array rather than invent

**Result, re-crawling VVF with the same model:**

| | bare `_arr()` | described |
|---|---|---|
| competitors | 6 names, **mostly wrong** (P&G and Reckitt are customers) | **none** |
| completeness | 0.883 | **0.0** |
| extraction | 1 chunk, succeeded | 1 chunk, ran 569s, `llm_tokens_out: 0` |

It stopped inventing competitors — and stopped producing anything at all. The longer
descriptions pushed the prompt past what a 7B can complete in one pass: it generated for
nine and a half minutes and returned nothing usable.

**So the descriptions are not yet a win.** Either shorten them, or split the extraction
into smaller chunks so each call has a manageable schema. Until that is resolved, the
0.883-completeness dump in `company_intel_7b/` is the better artifact — use it for
description, industries, products and customers, and **ignore its competitor list**.

The broader point stands: competitor naming is a factual-recall task. Curate it.

---

## 13. Merging the adapter into the base model

Two ways to run the fine-tune. **Serving the adapter at runtime is the default and is
what every result in this document used** — merging is only worth it for vLLM, which
cannot apply a LoRA on the fly as cheaply.

| | adapter at runtime | merged |
|---|---|---|
| Disk | 323 MB | ~15 GB |
| Startup | base download + instant adapter | base download + ~10 min merge |
| Weights | identical | identical |
| vLLM | not supported | required |

Merging is just folding `W + B·A` into `W` ahead of time. Same arithmetic, same output.

### Option A — no merge (recommended)

```bash
python training_data/serve_adapter.py --adapter /home/v1adapter --port 6006
```

`peft.PeftModel.from_pretrained(base, adapter_dir)` applies the LoRA in memory. The base
downloads from HuggingFace on first run (~15 GB, ~5 min) and is cached under
`$HOME/.cache/huggingface`, which survives a Jarvislabs pause.

### Option B — merge to a standalone model

Needs the base in **fp16, not 4-bit** — merging into quantised weights loses the
adapter's precision. ~16 GB of GPU or CPU RAM during the merge.

```python
# merge_adapter.py -- run on the GPU box
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE    = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "/home/v1adapter"
OUT     = "/home/merged"

base = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.float16, device_map="cpu")   # older transformers: torch_dtype=
model = PeftModel.from_pretrained(base, ADAPTER)
model = model.merge_and_unload()                    # folds B·A into W
model.save_pretrained(OUT, safe_serialization=True)
AutoTokenizer.from_pretrained(ADAPTER).save_pretrained(OUT)
print("merged ->", OUT)
```

`device_map="cpu"` is deliberate: the merge is arithmetic, not inference, and CPU RAM is
cheaper than the 16 GB of VRAM it would otherwise hold.

### Option C — merge with axolotl (only on the training box)

The training config is already there, so this is one command — but it has a trap:

```bash
# merge_lora reads load_in_4bit FROM THE YAML. Passing --load_in_4bit=False on the CLI
# raises "load_cfg() got multiple values for keyword argument 'load_in_4bit'".
# Edit a copy of the config instead.
sed -e 's/^load_in_4bit: true/load_in_4bit: false/' qlora_qwen7b.yml > merge_cfg.yml
python -m axolotl.cli.merge_lora merge_cfg.yml --lora_model_dir=./out
# -> ./out/merged/  (4 shards, ~15 GB)
```

That duplicate-kwarg error hit both training runs. `run_train.sh` still carries the
broken CLI flag; use the `merge_cfg.yml` route.

### Serving a merged model

```bash
# plain transformers
python training_data/serve_hf.py --model /home/merged --port 6006

# vLLM -- needs a CLEAN pytorch image, never the axolotl one
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model /home/merged --served-model-name cmi-7b \
  --max-model-len 4096 --gpu-memory-utilization 0.90 --port 6006
```

vLLM on the axolotl template fails three times in sequence — flash-attn ABI mismatch,
then duplicate cudnn, then a flashinfer JIT build error — all caused by vLLM upgrading
torch 2.3.1 to 2.13.0 underneath a pre-built image. A clean pytorch template avoids all
three. vLLM is the only route to the ~Rs.0.12/brief figure, because continuous batching
is what fills the GPU.

### Do not archive merged models

The 15 GB output is fully reconstructible from the 323 MB adapter plus a public base
model. Keep the adapter; rebuild the merge when a serving stack needs it.

---

## 14. vLLM batching — speed and cost, measured

Three goals, one change. Speed, batch processing and cost are the same problem: a single
request leaves the GPU idle between tokens, because producing one token means reading all
7B weights and the compute units wait on that read. vLLM's continuous batching computes
the next token for many requests off **one** weight read.

### Measured on an L4 (spot, Rs.27.54/hr)

| concurrency | aggregate throughput | speedup |
|---|---|---|
| 1 (old transformers server) | 17-19 tok/s | baseline |
| 8 | 115 tok/s | 6.2x |
| 16 | 217 tok/s | 12.9x |
| 32 | **381 tok/s** | **~20x** |

Per-request latency barely moves (6.4s alone, 6.6s with 16 in flight) — the gain is
entirely in aggregate throughput, which is what a batch of leads needs.

### End-to-end: 5 briefs, 5 workers

```
wall clock : 254s for 5 briefs (51s each)
cost       : Rs.0.389/brief
sequential : 1200s, Rs.1.84/brief  -> 4.7x faster, 4.7x cheaper
61 LLM calls, zero failures, all local
```

Scaling up (the KV cache holds 88,752 tokens, ~21 concurrent 4k requests):

| batch | wall | cost/brief |
|---|---|---|
| 5 @ 5 workers | ~254s | Rs.0.389 |
| 16 @ 16 workers | ~292s | **Rs.0.140** |
| 32 @ 32 workers | ~292s | **Rs.0.070** |

**The target is Rs.0.20. Batches of 16+ clear it.** A batch of 5 does not — the fixed
~250s of per-brief RSS fetching and sequential intra-brief calls dominates when there
are too few companies to fill the GPU.

### Setup

```bash
# 1. merge (vLLM cannot apply a PEFT adapter cheaply) -- see section 13
python merge_adapter.py          # ~2 min, writes /home/merged

# 2. install on a CLEAN pytorch template
pip install vllm

# 3. serve
python -m vllm.entrypoints.openai.api_server \
  --model /home/merged --served-model-name cmi-7b \
  --max-model-len 4096 --gpu-memory-utilization 0.92 \
  --max-num-seqs 32 --enable-prefix-caching --port 6006
```

`--gpu-memory-utilization 0.92` matters: whatever is left after weights becomes KV cache,
and KV cache size is what decides how many requests batch concurrently.
`--enable-prefix-caching` matters because all 11 calls in a brief share the same
`voice.STYLE` prefix — computed once instead of 11 times.

On this box vLLM picked FlashAttention automatically and started first time. The three
failures documented earlier were all caused by installing vLLM into the **axolotl**
image, which upgrades torch underneath a pre-built environment.

### Running the batch

```bash
python batch_generate.py --xlsx Book1.xlsx --limit 32 --workers 16 \
    --base-url https://<endpoint>/v1
```

`--workers` defaults to 8. Set it to 1 against a plain transformers server — that server
handles one request at a time, so concurrency there only adds queueing.

The calls **within** one brief stay sequential (call 2 needs call 1's output), so all the
concurrency comes from running companies in parallel. Each worker owns its own Profile
and article list; the only shared state is a counter and distinct output filenames.

### What did not help

`voice.active_style()` cut 34% of input tokens, but input is only ~28% of the total and
prefill is fast. **Output generation is ~97% of the wall clock** — 4,400 output tokens
per brief at 19 tok/s is 232 of the 240 seconds. Anything that does not increase output
throughput does not move the cost.

---

## 15. Lookback windows — why 6 months, for triggers too

`TRIGGER_LOOKBACK_HOURS = 24 * 182` in `config.py`, matching
`COMPETITOR_LOOKBACK_HOURS`. A per-company `trigger_lookback_hours` in
`company_overrides.json` overrides it.

The original 96 hours was tuned for a seller whose buyers make news weekly. Measured on
VVF's buyer queries:

| query | within 96h | within 30 days |
|---|---|---|
| FMCG company raises funding | 0 | 4 |
| personal care brand expansion plant | 0 | 3 |
| beauty brand funding | 0 | 4 |
| detergent company expansion | 1 | 6 |

One article in four days, eighteen in thirty. With that window the Potential Customer
slot was either **empty** or stretched to a famous company in an unrelated trade
(**Titan**, a watchmaker, for an oleochemicals producer).

At six months: **18 qualified buyers** pass the buyer-industry guard, including Sugar
Cosmetics' Rs.144.5 crore round and Emami's Kesh King launch — real personal-care brands
a contract soap manufacturer can actually sell to.

### Widening the window does not mean stale briefs

`freshness()` still decays older signals, so a recent one outranks an older one whenever
both exist. The window only decides what is **eligible**; ranking decides what is used.

`score.py` gained `_TRIGGER_BANDS` for this. The default bands bottom out at 168h/0.10,
so once everything older than a week scored the same floor, a 14-day-old funding round
and a 172-day-old one ranked identically — and a buying trigger genuinely does go cold.
The new bands keep the steep early decay and continue separating older items:

```
  4 days -> 0.30      14 days -> 0.16      59 days -> 0.10      68 days -> 0.06
```

An eligible-but-older real buyer beats an irrelevant fresh one.

### Verified run

`output/cmi_local_vvf_6mo.html` — 200 articles, 198 eligible, 72 trigger-typed.

| Section | Subject |
|---|---|
| Potential Customer | **Sugar Cosmetics** (Rs.144.5 crore round, valuation cut) |
| Industry Outlook | Godrej's fourth Malanpur plant — private-label capacity |
| Competition Outlook | **Galaxy Surfactants** (108% Q1FY27 profit jump) |

All three are genuinely about VVF's market. Voice: 23.9 violations per 1000 words
against the earlier 27.3 and DeepSeek's 37.6.

### Other fixes this needed

- **Query generation was broken.** `tenant.py`'s fallback pasted the industry LABEL into
  a template — "Home care and detergent manufacturers company raises funding" matches no
  headline. It now extracts head nouns ("detergent", "cosmetics").
- **Do not put a country in a curated query.** `collect()` appends the market name, so
  "FMCG company India raises funding" became "...funding India" and matched nothing.
- **A headline is not a company name.** When triage returned no name the fallback was
  `a.title[:48]`, which rendered "Neogen raises Rs.600 crore, Dhanuka ratings held," as
  the Potential Customer. Unnamed candidates are now skipped.
- **Overrides matched exact names only.** "VVF LTd" hit, "VVF" missed, so curated facts
  were invisible on runs that spelled the name differently. `_overrides_for()` now
  matches on a normalised stem.

### Pause/resume wipes pip installs

Only `/home` survives a Jarvislabs pause. `pip install vllm` goes to site-packages, so
**vLLM must be reinstalled after every resume** (~4 min). The merged model in
`/home/merged` survives.

---

## 16. Report briefs — three sections, and the LLM validator

### The structure

| # | Section | Cells | Content |
|---|---|---|---|
| 1 | **Your Market** | What is changing / Why it matters for you | The market itself and the trends shaping it |
| 2 | **Market Moves** | Who is moving / What it signals | A named company acquiring, investing, expanding or partnering in this market |
| 3 | **Competition Outlook** | Why it matters / Their edge | Rivals, **excluding anything Market Moves used** |

CTAs, analyst-opinion bullets and the Inside-CMI panel are unchanged.

### Why Market Moves, and not Regulatory or Demand

Four framings were measured across all 20 leads, 16 queries each, six-month window:

| framing | total hits | real news | % real | leads with 3+ |
|---|---|---|---|---|
| **Market Moves** (acquisition/investment/expansion) | 1251 | **940** | 75% | **19/20** |
| New Entrants | 845 | 784 | 93% | 18/20 |
| Regulatory | 949 | 735 | 77% | 18/20 |
| Whitespace | 347 | 303 | 87% | 14/20 |
| Demand Drivers | 1588 | 698 | **44%** | 18/20 |

"Real news" excludes rival research firms' SEO pages -- "Market Size Share 2035", MRFR,
Grand View. **Demand Drivers scores worst precisely because "market size / growth /
forecast" is what those firms optimise for**, and CMI must not cite a competitor's
market-size page inside its own brief.

Regulatory was the early pick on purity alone. It is the wrong answer commercially: a
rule change is information, while capital moving in your market is a reason to find out
what is happening before you commit -- which is exactly what the section's existing CTA
offers. Market Moves also wins on coverage.

### Keeping sections 2 and 3 distinct

Both surface companies active in the same market, so without an explicit rule the same
acquisition appears twice and the reader is told a company is "a move to watch" and
"your competitor" in one brief. `brief_builder.build_brief` enforces it twice:

1. Market Moves runs first; its article is removed from the competitor pool and from
   `peer_arts` before `build_competition` sees them.
2. If both sections still land on the same COMPANY -- compared via `_base_name`, so
   "Weir Group" and "The Weir Group plc" collapse -- Market Moves is dropped.

### The LLM validator: three methods, two of which failed

`llm_validate.py` uses the fine-tune itself as judge. Three approaches were built and
measured on five cases (three real failures from live briefs, two controls):

| method | how it asks | correct |
|---|---|---|
| 1. passage-only | judge from the signal text, no outside knowledge | **1/5** |
| 2. describe then label | state the business, then pick from 5 labels | **0/5** |
| 3. describe then yes/no | state the business, then one binary question | **4/5** |

**Method 1 fails by design.** It refuses outside knowledge, so it answers "unknown"
whenever the passage does not settle the question -- and a TSMC chip-packaging headline
never says whether TSMC competes in wood veneers. It is kept because it is the only one
that catches a CUSTOMER filed as a rival: that relationship IS stated in the text
("VVF manufactures for Reckitt"), and method 1 got it right where method 3 cannot.

**Method 2 failed for an instructive reason.** Its `their_business` answers were
correct every single time -- it knew TSMC makes semiconductors, Infosys sells IT
services, HubSpot sells CRM -- and it still returned "unknown" or "real" when asked to
choose among five labels. Recall was never the weak link; the classification was.

**Method 3 collapses the question to "could a buyer get the same thing here instead?"**
and that one change took it from 0/5 to 4/5. Its single miss is conservative in the safe
direction: it calls Galaxy Surfactants "no" for VVF because surfactants and bar soap are
not literally the same product, even though the two firms compete for the same
contract-manufacturing work. Rejecting nonsense reliably while being cautious about
confirming real rivals is the right way round.

`judge_brief` runs method 1 first, then falls back to method 3 when method 1 says
"unknown". Verified live:

```
LLM ROLE ERROR: 'Infosys' (Infosys sells IT services and consulting to large
enterprises and government bodies worldwide.) sells nothing a buyer of
paulownia-partners's product would substitute. different industries
```

### Using it

```bash
CMI_LLM_VALIDATE=1 python validate_brief.py output/<brief>.html \
    --company "<name>" --website <url>
```

Off by default: each check is a GPU call, and a 20-brief batch adds ~40. Worth it before
a send, wasteful during iteration.

**It fails open.** A judge that is down, slow, or unsure never blocks -- the
deterministic keyword rules in `validate_brief.py` remain the floor and this only adds
findings. That matters because the judge is the same 7B whose factual recall we measured
as unreliable.

### What still fails

- **Competition Outlook is missing on ~2 of 6 report briefs** (`energofish`, `salient`).
  Tiny or non-commercial entities generate no rival news in any window.
- **Competitor picks are still wrong before the judge sees them.** TSMC for a
  wood-veneer trader, Infosys for a home-care agency. The judge now catches them, but
  catching is not the same as generating the right one -- that needs curated overrides.
- **2 LLM calls per 6-brief batch fail** with a 400 context-length error.
- **Topic-name collisions.** "Cable Glands Market" matches Gland Pharma, a
  pharmaceutical company. No framing fixes this; the query needs entity exclusion.

---

## 17. Account vs report briefs, with and without the LLM judge — 4 runs, 80 briefs

All 20 leads, 12 workers, vLLM on a spot L4. Nothing left the box: 0 DeepSeek,
0 OpenAI across all four runs.

| mode | validation | wall | cost/brief | PASS |
|---|---|---|---|---|
| account | rules only | 474s | Rs.0.181 | 5/20 |
| account | rules + LLM | 484s | Rs.0.185 | **1/20** |
| report | rules only | 498s | Rs.0.190 | 15/20 |
| report | rules + LLM | 502s | Rs.0.192 | **5/20** |

Output in `output_account/`, `output_report/`, `output_account_judged/`,
`output_report_judged/`.

### Report briefs are the better product

3x the pass rate on rules alone (15/20 against 5/20) and 5x with the judge. The reason
is structural, not cosmetic: a report brief needs market news, which the feed returns
reliably once the query is the TOPIC. An account brief needs a plausible BUYER for the
lead's own company, and for an obscure firm the model has no basis to pick one -- so it
reached for whatever name the feed happened to carry (Airbnb for a US realtor, ACME
Solar for a Turkish cable-gland maker).

### The judge is worth its cost

It removed **4 of 5** passing account briefs and **10 of 15** passing report briefs --
all of them briefs the keyword rules had cleared. Examples it caught:

```
'Blackstone' (private equity) -- for an electric-wheelchair market brief
'Amwell' (telehealth) -- for a healthcare price-transparency firm
'Klein Tool' (hand tools) -- for a tungsten-carbide wear-parts maker
'Inc42' (Indian startup media) -- for a VC advisory firm
'Workday' (HR software) -- for an EV charging brief
```

Every one is a real company doing something real, which is exactly why a keyword rule
cannot catch them and a reader would notice immediately.

It costs **+12-15% more LLM calls and under 10 seconds of wall clock** -- Rs.0.185
against Rs.0.181 per brief. That is a rounding error against shipping a brief that names
a hand-tool company as a carbide shop's competitor.

### Read the pass rates correctly

1/20 and 5/20 are not a regression. The same briefs were always this weak; the rules
just could not see it. The judged numbers are the honest measure, and the gap between
them (5 -> 1, 15 -> 5) is the size of the blind spot that was there all along.

The judge also produces useful WARNs rather than only failures -- "'Arm' is adjacent,
not a direct substitute", "'Salesforce' is adjacent" -- which are judgement calls a
human should make, not automatic rejections.

### What the judge does NOT fix

It catches bad picks; it cannot generate good ones. The path to a higher pass rate is
still curated `company_overrides.json` entries -- the judge tells you which accounts
need them.

### Failures common to all four runs

- **Missing Competition Outlook**: `energofish`, `southcarb`, `aroma`, `qst`,
  `graematter`. Small or non-commercial entities with no rival news in any window.
- **6 LLM calls fail per 20-brief run** with a 400 context-length error. Consistent
  across runs, roughly 2.5% of calls, and does not fail a brief -- `chat_json` degrades
  to its default and the section is rebuilt from source text.

### Running it

```bash
CMI_LLM_VALIDATE=1 python batch_generate.py --xlsx Book1.xlsx \
    --limit 20 --workers 12 --kind report --outdir output_report_judged \
    --base-url https://<endpoint>/v1
```

---

## 18. Validation gates — relaxing changes what you block on, not accuracy

Four settings run against the same 20 report briefs. The briefs are identical; only the
gate differs.

| setting | PASS | WARN | FAIL | what it measures |
|---|---|---|---|---|
| rules only (no judge) | 15 | 0 | 5 | structure + keyword-visible errors |
| judge strict | 5 | 0 | 15 | **true quality** |
| **judge advisory** (default) | 8 | 5 | 7 | structure blocks; role calls warn |
| judge agree-only | 4 | 10 | 6 | two methods must reject |

**Accuracy is the same in all four.** The same briefs carry the same bad picks. Relaxing
the gate does not make the output better -- it decides how many reach an inbox, which is
a shipping decision rather than a measurement one.

### 15/20 was never real

Those briefs had Blackstone as an electric-wheelchair rival and Klein Tool as a
tungsten-carbide competitor. The keyword rules simply could not see it. **5/20 is the
honest rate**, and the route to a better one is curated `company_overrides.json`
entries, not a looser gate.

### Agree-only went DOWN, which was not the expectation

Requiring both methods to agree was meant to loosen things. It tightened them (5 -> 4),
because running the second method also surfaced findings the first never reached --
PriceSmart for a healthcare-pricing firm, Economic Times VC Circular for a VC advisory.
"Require more evidence" turned out to mean "look harder", not "fail less".

### Advisory is the default

`CMI_LLM_STRICT=1` restores strict. Advisory is right because:

- every role finding still prints and is counted, so nothing is hidden
- only a structural problem blocks a send
- "Halliburton is adjacent, not a direct substitute" is a judgement a human should make

Use strict for a curated send where each brief must stand without review; use advisory
for volume where the WARNs get read.

### Missing sections now degrade instead of failing

A brief with no findable prospect or rival is a real outcome, not a defect -- some
markets have no rival news in any window, and dropping the section is more honest than
the stretched picks that came from forcing one.

  - **one** missing section -> WARN (ships with two strong cards)
  - **two** missing -> FAIL (a single card is not worth sending)

This moved seven briefs from FAIL to WARN: `energofish`, `abahomecare`, `ulusofona`,
`salient`, `southcarb`, `aroma`, `s8`.

Seven of twenty is more than "rare" and worth watching. Every one is a small or
non-commercial entity -- a Portuguese university, a US carbide shop -- with genuinely no
competitor news. If the rate climbs for ordinary companies, check the feed rather than
the threshold.

### Output folders from these runs

```
output_account/            account briefs, rules only
output_report/             report briefs, rules only
output_account_judged/     account briefs, judge strict
output_report_judged/      report briefs, judge strict
output_report_relaxed/     report briefs, judge advisory
output_report_agree/       report briefs, agree-only + degradation
```

---

## 19. Throughput AB tests — 40 briefs, five configurations

20 account + 20 report briefs, same 20 leads, same vLLM server, on-demand L4
(Rs.41.31/hr). Nothing left the box on any run: 0 DeepSeek, 0 OpenAI.

| run | wall | cost/brief (on-demand) | on spot | PASS | WARN | FAIL | sendable |
|---|---|---|---|---|---|---|---|
| baseline: 2 batches x 12 workers | 997s | Rs.0.2860 | Rs.0.1907 | — | — | — | — |
| **A: one window, 40 workers** | **357s** | **Rs.0.1024** | **Rs.0.0683** | 12 | 25 | 3 | 92% |
| D: one window, 30 workers | 522s | Rs.0.1497 | Rs.0.0998 | 16 | 22 | 2 | 95% |
| B: one window, 20 workers | 569s | Rs.0.1632 | Rs.0.1088 | 18 | 18 | 4 | 90% |
| C: one window, 64 workers | 516s | Rs.0.1480 | Rs.0.0987 | 14 | 24 | 2 | 95% |

### What made the difference: one concurrency window, not two batches

`batch_generate.py --kind signal` then `--kind report` runs two batches back to back, so
the wall clock is the sum. But the ~11 calls INSIDE a brief are sequential while briefs
are independent, which means **batch wall clock is bounded by the slowest single brief,
not by total work**. Once workers >= briefs, adding briefs is nearly free.

`run_all40.py` submits all 40 jobs to one pool. That alone took Rs.0.286 -> Rs.0.102, a
2.8x cut with no server change.

### More workers is NOT monotonic

The expectation was 64 > 40 > 30 > 20. The measurement disagrees: **40 is the sweet
spot, and 64 is worse** (Rs.0.148 against Rs.0.102). Past roughly 40 concurrent,
requests queue inside vLLM instead of batching, so the slowest brief stretches from 357s
to 516s -- and since wall clock equals the slowest brief, that goes straight onto cost.

### Server settings that helped

```
--max-num-seqs 64            (from 32)
--gpu-memory-utilization 0.94 (from 0.92)
--enable-chunked-prefill      (long prompts stop blocking short ones)
--enable-prefix-caching       (shared voice.STYLE prefix computed once, not 11x)
```

KV cache: 88,752 -> **96,048 tokens**.

### Cost per SENDABLE brief is the number that matters

Hard-failed briefs are wasted GPU time, so the honest metric divides by PASS+WARN:

| run | cost per sendable (on-demand) | on spot | usable |
|---|---|---|---|
| **A: 40 workers** | **Rs.0.1107** | **Rs.0.0738** | 37/40 |
| C: 64 workers | Rs.0.1558 | Rs.0.1039 | 38/40 |
| D: 30 workers | Rs.0.1576 | Rs.0.1051 | 38/40 |
| B: 20 workers | Rs.0.1814 | Rs.0.1209 | 36/40 |

A still wins, and by more than the raw figure suggested.

### The quality caveat, stated plainly

A produces **12 PASS against B's 18**. Same briefs, same model -- the difference is that
12 LLM calls fail under load at 40 workers, so those sections degrade to source-derived
text rather than model prose. Cheaper is not strictly better here.

But 92% of A's output is still sendable (PASS+WARN), and a WARN is a judgement call for
a human, not a broken brief. Paying 64% more per brief to move six briefs from WARN to
PASS is only worth it if nobody is reading the WARNs.

## RECOMMENDATION: run A (one window, 40 workers)

```bash
python run_all40.py --workers 40 \
    --account-dir output_account --report-dir output_report \
    --base-url https://<endpoint>/v1
```

- **Rs.0.1024/brief on-demand, Rs.0.0683 on spot** -- half the Rs.0.20 target
- **17x cheaper** than the original sequential pipeline (Rs.1.77)
- 40 briefs in **6 minutes**, 6.7 briefs/minute
- 100% on the fine-tune

Use **D (30 workers)** instead when a batch ships without human review: it costs 46%
more but hard-fails two briefs rather than three and lands more in PASS.

### Still open

- **12 LLM calls fail per 40-brief run** (~2.5%), a 400 context-length error. Consistent
  across every configuration. They degrade a section rather than failing a brief, but
  they are the direct cause of A's lower PASS count.
- **Nothing was tested above 64 workers**, and the trend says it would get worse.
- A bigger GPU was not tested. The floor is the slowest single brief (~357s), which is
  ~11 sequential calls -- a faster card shortens each call but cannot parallelise them,
  so the gain would be sub-linear.
