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
""".strip()


# CTA copy, shared by both brief types so they cannot drift apart.
def ctas(demo_url: str, consult_url: str) -> list[dict]:
    return [
        {"q": "Would you want to know the roadmap to tap into the next million-dollar opportunity?",
         "sub": "Get access to our lead consultant. They will guide you on positioning "
                "and market entry strategy to capitalise on this opportunity.",
         "label": "Schedule a call with a lead consultant", "url": consult_url},
        {"q": "Want access to a real-time tracker on the shifts that could impact your business?",
         "sub": "CMI Intelligence covers changing trends, shifts in market dynamics, "
                "whitespaces and the roadmap to the next million-dollar opportunity in "
                "the industry.",
         "label": "Schedule a meeting with our analyst team", "url": demo_url},
        {"q": "Want competitor movement flagged as it happens in your categories?",
         "sub": "A daily tracker mapped to your competitors, tracking their key market "
                "moments. It includes detailed intelligence on financials, "
                "strategy roadmap, key developments, operational moves and more.",
         "label": "Schedule the call with the team", "url": demo_url},
    ]


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
