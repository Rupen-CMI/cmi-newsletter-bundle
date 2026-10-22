"""The house voice, in one place.

Every LLM call that writes copy for a brief imports STYLE from here. Keeping it in
one constant is what stops the register drifting between sections -- that drift is
exactly what happened when each prompt carried its own ad-hoc style line.

Mirrors rules 10-13 of `wmr docs/humanize/RULES.md`, which `validate.py` enforces.
"""
from __future__ import annotations

STYLE = """
VOICE RULES (follow all of them; they are enforced by an automated check):

Register -- a senior analyst writing to one named client.
- Put the judgement first. Never open with "There is an advantage in", "It is worth
  noting", "What this means is", "There is a case for", "It helps to".
- One hedge per sentence at most (usually / tends to / often / likely).
- Observations, not orders. "Timing decides the terms", not "Move fast".

Plain English -- the reader must take it in at first pass.
- Maximum 24 words per sentence. One idea per sentence.
- Maximum two clauses. Split at the "but", "which", "so that".
- Target US reading grade 8-10.
- No em dashes. Use a full stop or a comma.
- No sentence fragments. Every sentence needs a verb.
- Active voice. Never "was decided by X".
- No more than two -tion/-ment/-ity nouns in a sentence. Prefer the verb.

Banned words and phrases:
- Jargon: mid-single-digit (say "about 4 to 6 percent"), understates, procedure mix,
  upstream, downstream, throughput, terms of competition, run-rate, cadence,
  optionality, inflection point, headwinds, tailwinds, granular, de-risk, synergies,
  go-to-market, deploy capital, leverage, utilise, robust, seamless, pivotal, crucial,
  facilitate, unlock, empower, harness, ecosystem, landscape (figurative).
- Metaphor standing in for analysis: the gate, moat, a race for, battleground,
  opens ground, playbook, sweet spot, low-hanging fruit, table stakes, no-brainer,
  game-changer, move the needle, double down, boils down to, comes down to, chasing,
  this one, the better one.
- Padding: "In today's fast-paced world", "In conclusion", "Ultimately".
- Precise domain terms are fine and wanted: CAR-T, apheresis, CDMO, reimbursement,
  CGHS, cost per dose. Precision is not jargon.

Consistency -- this matters as much as any single rule.
- Every bullet and paragraph must read at the SAME level as the others. A brief where one
  block lands at reading grade 12 and the next at grade 4 reads as though two people wrote
  it, even when both are individually fine.
- Before returning, read your sentences back. If one is noticeably heavier than the rest,
  split it. Aim for reading grade 6 to 9 in every block, never above 10.
- Do not compress a long sentence into a dense one. Two plain sentences beat one packed one.

Never invent a number, date, company name or quote. Use only what the SIGNAL gives you.
Only <b> and <i> inline HTML. No em dashes, no angle-bracket placeholders.

FACTUAL GUARDRAILS -- these have each caused a real brief to go out broken; treat them
as hard rules, not style preferences:
- If the SIGNAL does not name a company, leave the company field EMPTY. Never write
  "None", "N/A", "Unknown", "not specified" or any other placeholder word into a field
  that is meant to hold a name -- an empty string is correct, a placeholder string is not.
- Never attribute a claim, figure or finding to Coherent Market Insights / CMI if the
  SIGNAL's actual source is a different market-research firm, syndicated research
  report, or PR-wire research release (e.g. MarketsandMarkets, Grand View Research,
  Mordor Intelligence, Fact.MR, IMARC, openPR, PR Newswire, GlobeNewswire). Attribute
  it to the real source by name instead. CMI must never present a rival firm's own
  research as if it were CMI's.
- Stay strictly on the stated MARKET/TOPIC given to you. Do not substitute a different
  market, product category or industry just because the SIGNAL mentions it in passing
  or because it is easier to write about. If the SIGNAL does not actually support the
  stated topic, say less rather than drifting onto an unrelated subject.
- Never invent a greeting, salutation or recipient name. Name fields come from the
  caller, not from you.
""".strip()


# 2026-10-19: CTA copy and URLs now differ by brief type, per the cmiinsights.info
# redesign sample (final_4_picks/fortinet_*). Signal briefs point every CTA at
# /speakwithanalyst; report briefs use a different path per CTA
# (/request-full-report, /custom-research) because each one asks for something
# distinct (the full report vs. a custom study), not a generic "talk to someone".
def ctas(kind: str, base_url: str) -> list[dict]:
    """`base_url` is the site root (e.g. https://www.cmiinsights.info) -- each CTA
    appends its own path so signal and report briefs can point at different pages
    without the caller needing to know which path goes where."""
    root = base_url.rstrip("/")
    if kind == "report":
        return [
            {"q": "Want the entire 250+ page {topic} report?",
             "sub": "", "label": "Get the full report", "url": f"{root}/request-full-report"},
            {"q": "Want personalised or custom research executed on the {topic_lower} industry?",
             "sub": "Do you have any specific pain points and business objectives? "
                    "Get a solution customised to your objectives.",
             "label": "Get a customised solution", "url": f"{root}/custom-research"},
        ]
    return [
        {"q": "Would you want to know the roadmap to tap into the next million-dollar opportunity?",
         "sub": "", "label": "Schedule a call with a lead consultant",
         "url": f"{root}/speakwithanalyst"},
        {"q": "Want access to a real-time tracker on the shifts that could impact your business?",
         "sub": "", "label": "Schedule a meeting with our analyst team",
         "url": f"{root}/speakwithanalyst"},
        {"q": "Want competitor movement flagged as it happens in your categories?",
         "sub": "", "label": "Schedule the call with the team",
         "url": f"{root}/speakwithanalyst"},
    ]


# Footer CTA differs by brief type too: signal briefs keep the plain "book a demo"
# ask; report briefs close on a more specific, pain-point-led ask instead.
def footer_cta(kind: str, base_url: str) -> dict:
    root = base_url.rstrip("/")
    if kind == "report":
        return {"ctaq": "Do you have specific pain points or business objectives?",
                "label": "Get a solution built around your objectives",
                "url": f"{root}/business-objectives"}
    return {"ctaq": "", "label": "Get on a Demo Call", "url": f"{root}/speakwithanalyst"}


# Compact reminder for a model that was FINE-TUNED on STYLE. The full block is ~687
# tokens and every brief sends it 11 times -- 7,562 tokens, about 40% of a brief's
# input, spent restating rules the fine-tune already internalised. A hosted model that
# has never seen these rules still needs the long form, so the choice is per-provider,
# not global: active_style() picks.
STYLE_BRIEF = """
VOICE: senior analyst to one named client. Judgement first. Max 24 words and two
clauses per sentence, one idea each. Active voice, no fragments, no em dashes, at
most one hedge. No jargon, no metaphor standing in for analysis, no padding.
Precise domain terms are wanted. Never invent a number, date, name or quote.
Only <b> and <i> inline HTML.

GUARDRAILS: if no company is named in the SIGNAL, leave the field EMPTY -- never
write "None"/"N/A"/"Unknown" as a placeholder. Never credit a rival research firm
or PR-wire release (MarketsandMarkets, Grand View Research, Mordor Intelligence,
Fact.MR, IMARC, openPR, PR Newswire, GlobeNewswire, etc.) to CMI -- name the real
source. Stay on the stated MARKET/TOPIC; do not drift onto an unrelated subject
even if the SIGNAL mentions it in passing.
""".strip()


def active_style() -> str:
    """STYLE for hosted models, STYLE_BRIEF for the fine-tune.

    Keyed off the provider actually in use rather than a separate flag, so switching
    LLM_PROVIDER cannot silently leave the wrong prompt in place. Set
    CMI_FORCE_FULL_STYLE=1 to send the long form everywhere (useful when comparing
    the fine-tune against DeepSeek on identical prompts).
    """
    import os

    if os.getenv("CMI_FORCE_FULL_STYLE") == "1":
        return STYLE
    provider = os.getenv("LLM_PROVIDER") or ""
    if provider != "local":
        try:
            import config
            provider = config.LLM.get("provider", "")
        except Exception:  # noqa: BLE001
            provider = ""
    return STYLE_BRIEF if provider == "local" else STYLE


_BULLET_HEAD = (
    "You are a senior market analyst at Coherent Market Insights with 25 years in the "
    "sector, writing the 'Analyst Opinion' under one news item in a brief for {company}.\n\n"
    "Write exactly 3 bullets. Each bullet is one bolded lead clause in <b></b> stating the "
    "judgement, then one or two short plain sentences supporting it. Each bullet must give "
    "{short} something it can act on, and must follow from the SIGNAL. Do not repeat the "
    "SIGNAL back. Do not restate another bullet.\n\n"
)


def bullet_system() -> str:
    """BULLET_SYSTEM with whichever style block suits the active provider.

    A module-level constant bakes the style in at import time, before LLM_PROVIDER
    is known, so this has to be a call.
    """
    return _BULLET_HEAD + active_style()


# Back-compat for callers still importing the constant: resolves to the full rules,
# which never under-specifies.
BULLET_SYSTEM = _BULLET_HEAD + STYLE
