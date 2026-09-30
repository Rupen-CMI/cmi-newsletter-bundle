# Coherent Market Insights — Lead-Nurture Brief (standalone system)

A refactored, self-contained system for the **Coherent Market Insights (CMI)** template —
the inbound lead-nurture brief. This is the CMI counterpart to the `coherent-newsletter`
folder (which is the **CoherentConnect** outbound system). Same free engine, CMI-only framing.

## What it does
Input is a **lead** who requested a report/sample. CMI nurtures them toward purchase with a
personalized 3-section brief, each followed by a *Coherent Take*:

1. **What's Happening in Your Market** — live news in the market they requested
2. **Other Markets That Might Interest You** — adjacent markets to cross-sell (LLM-derived)
3. **What Your Competitors Are Up To** — moves by the lead *company's* competitors

Every figure is framed as **CMI's own** intelligence — rival research firms and their numbers
are filtered out (`config.is_research_pr`), never cited.

## Pipeline
```
brief_builder.derive_context()   LLM plans the market / adjacent markets / competitors
collect.collect_queries()        Google News RSS for market + competitor news (geo-aware)
brief_builder._synth_section()   DeepSeek→OpenAI synthesis, CMI house voice
brief_builder._editorial()       lede + 3 section Takes + top-3 lines
render.py                        Jinja → templates/daily_brief_cmi.html → output/*.html
```

## Lead schema (`leads.json`)
```json
{
  "name": "Ravi Kumar",
  "company": "ImmunoACT",
  "job_title": "Head of Strategy",
  "report_name": "Bone Marrow Transplant Market",
  "report_link": "https://www.coherentmarketinsights.com/market-insight/...",
  "price": "US $3500",
  "objectives": "Evaluating white-space in cell therapy ...",
  "market": "India"
}
```

## Run
```bash
pip install -r requirements.txt
# keys auto-read from ../coherent-newsletter/.env  (or drop a local .env)
python run.py                       # every lead in leads.json
python run.py --lead "Ravi"         # one lead (substring match)
python run.py --market Global       # override target country
# → output/cmi_leadbrief_<name>_<company>_<YYYYMMDD>.html
```

## Files
| File | Role |
|---|---|
| `config.py` | market lexicon, feeds, country profiles, **research-firm PR filter**, LLM settings |
| `collect.py` | Google News RSS collection + fingerprint dedup |
| `geo.py` | country geo-matching / fallback |
| `llm.py` | DeepSeek→OpenAI chat wrapper (`chat`, `chat_json`) |
| `brief_builder.py` | derive plan + synthesize the 3 sections + editorial + assemble `brief` |
| `render.py` | Jinja render → the CMI template |
| `templates/daily_brief_cmi.html` | the Coherent Market Insights template |
| `leads.json` | the leads to nurture |
| `run.py` | entrypoint / orchestration |

## Relationship to the other folders
| Folder | System | Template |
|---|---|---|
| `coherent-newsletter/` | CoherentConnect (outbound prospecting) | `daily_brief.html` |
| **`coherent-market-insights/`** | **CMI (inbound lead nurture)** | **`daily_brief_cmi.html`** |
| `pitch-agent/` | Expo pitch emails (DeepSeek) | plain-text |
