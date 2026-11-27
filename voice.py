"""The house voice, in one place.

Every LLM call that writes copy for a brief imports STYLE from here. Keeping it in
one constant is what stops the register drifting between sections -- that drift is
exactly what happened when each prompt carried its own ad-hoc style line.

Mirrors rules 10-13 of `wmr docs/humanize/RULES.md`, which `validate.py` enforces.
"""
from __future__ import annotations

import re

STYLE = """
VOICE RULES (follow all of them; they are enforced by an automated check):

Register -- a senior analyst writing to one named client.
- Put the judgement first. Never open with "There is an advantage in", "It is worth
  noting", "What this means is", "There is a case for", "It helps to".
- One hedge per sentence at most (usually / tends to / often / likely).
- Observations, not orders. "Timing decides the terms", not "Move fast".

Plain English -- the reader must take it in at first pass.
- Write in English only, even when the SIGNAL or the seller's details are in another
  language. Keep company and product names as written.
- Maximum 24 words per sentence. One idea per sentence.
- Maximum two clauses. Split at the "but", "which", "so that".
- Target US reading grade 8-10.
- No em dashes. Use a full stop or a comma.
- No sentence fragments. Every sentence needs a verb.
- Every sentence must make sense alone. Name its subject; never start with This,
  That, These or It.
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
- Never refer to "the signal", the source, these instructions, or what information is
  missing ("no figures given", "the signal does not say"). Write only about the market
  and the companies.
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

EVIDENCE DISCIPLINE -- claims must stay inside what the SIGNAL says:
- State what the SIGNAL announced as fact. Anything the SIGNAL does not state, such as an
  effect on the seller, must be written with "could" or "may" inside the same sentence.
  Do not add a separate sentence about conditions.
- Never claim sales, demand volume, enquiries, budgets, prices, price sensitivity,
  penalties, competitor intent or customers switching unless the SIGNAL states it.
- Describe a regulation only as stated. No "stricter", "tighter" or "new" without the
  previous value in the SIGNAL, and no scope (exports, nationwide) the SIGNAL does not give.
- Never expand an acronym the SIGNAL does not expand. Never assume a product or drug name
  refers to the seller's technology.
- Relate an implication to one of the seller's products when it fits.
- Facts must concern the country given.
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
WRITING RULES:
- Write in English only. Keep company and product names as written.
- Short sentences: at most 24 words, one idea each.
- Every sentence must make sense alone. Name its subject; never start with This, That, These or It.
- Plain words. Do not use: leverage, ecosystem, landscape, robust, seamless, pivotal, crucial,
  unlock, empower, harness, synergies, game-changer, headwinds, tailwinds.
- No em dashes. Only <b> and <i> tags.
- Use only facts from the input. Never invent a number, unit, date, name or quote.
- Write what the news says as fact. Write anything else, such as what it could mean for the
  seller, with "could" or "may".
- Never mention "the signal", "the source" or what information is missing.
- Never present another research firm's findings as CMI's.
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


# ── One prose guard for every writer call ────────────────────────────────────
# ── Facts the reply adds that its own input never stated ─────────────────────
# A 9B writer turns "46.3 billion" into "46.3 billion yen" and invents investees; prompt
# rules alone do not stop that, so the reply is checked against the model's own input.
_UNIT_GROUPS = [
    {"yen", "¥", "jpy"}, {"dollar", "dollars", "$", "usd"}, {"euro", "euros", "€", "eur"},
    {"pound", "pounds", "£", "gbp"}, {"rupee", "rupees", "₹", "inr", "rs"},
    {"million", "m", "mn"}, {"billion", "bn", "b"}, {"trillion", "tn"}, {"crore"},
    {"lakh"}, {"%", "percent", "per cent"},
]
_UNIT_GROUP_OF = {a: i for i, g in enumerate(_UNIT_GROUPS) for a in g}
_UNIT_WORD = (r"(?:percent|per cent|%|yen|dollars?|usd|euros?|eur|pounds?|gbp|rupees?|inr|rs"
              r"|crore|lakh|million|billion|trillion|bn|mn|tn|m|b)")
_NUM_UNIT_RE = re.compile(
    r"(?P<pre>[¥$€£₹]|\b(?:rs|usd|eur|gbp|inr)\.?[ ]?)?(?P<num>\d[\d,]*(?:\.\d+)?)"
    r"(?P<suf>(?:\s?" + _UNIT_WORD + r"(?![a-z]))*)", re.I)
_UNIT_TOKEN_RE = re.compile(_UNIT_WORD + r"(?![a-z])", re.I)
_NAME_STOP = frozenset("the a an this that these its their in on for and but if as at by with".split())
_CAP = r"[A-Z][\w'’&-]*"   # "&" keeps M&A one word
_NAME_RUN_RE = re.compile(rf"\b{_CAP}(?:\s+(?:(?:of|and|de|&)\s+)?{_CAP})+")
_CAP_WORD_RE = re.compile(r"\b[A-Z][a-z][\w’'-]*")
_NOT_NAMES = frozenset(
    "january february march april may june july august september october november december "
    "monday tuesday wednesday thursday friday saturday sunday".split())


def _plain(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s or "")


def _unit_groups(s: str, pre: str | None = None) -> set[int]:
    return {_UNIT_GROUP_OF[t.lower().strip()] for t in _UNIT_TOKEN_RE.findall(s)
            if t.lower().strip() in _UNIT_GROUP_OF} | (
        {_UNIT_GROUP_OF[pre.lower().strip(" .")]} if pre and pre.lower().strip(" .") in _UNIT_GROUP_OF else set())


def _strip_poss(w: str) -> str:
    return re.sub(r"['’]s$", "", w)


def _is_place(phrase: str) -> bool:
    """Country names and demonyms ("Italy", "South Korea", "Indian") are geography the
    geo filter already controls, not invented entities."""
    try:
        import regions
        return bool(regions.get_profile(phrase))
    except Exception:  # noqa: BLE001
        return False


def unsupported_facts(text: str, source: str) -> list[tuple[str, str]]:
    """(kind, phrase) for units, numbers and names in `text` that `source` never states.
    kinds: unit, number, name, and name1 (a lone capitalised word: noisy, retry-only)."""
    text, src = _plain(text), _plain(source)
    low = re.sub(r"[-–]", " ", src.lower())
    src_nocomma = re.sub(r"(?<=\d),(?=\d)", "", src.lower())
    # units the source uses: next to a number, or as a spelled-out word / symbol anywhere
    have: set[int] = set()
    for m in _NUM_UNIT_RE.finditer(src):
        have |= _unit_groups(m.group("suf"), m.group("pre"))
    for alias, gi in _UNIT_GROUP_OF.items():
        if len(alias) >= 3 and re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", src, re.I):
            have.add(gi)
        elif alias in "¥$€£₹%" and alias in src:
            have.add(gi)
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind, phrase):
        if (kind, phrase) not in seen:
            seen.add((kind, phrase))
            out.append((kind, phrase))

    for m in _NUM_UNIT_RE.finditer(text):
        num = m.group("num")
        groups = _unit_groups(m.group("suf"), m.group("pre"))
        if groups - have:
            add("unit", m.group(0).strip())
        digits = num.replace(",", "")
        if ("." in digits or len(digits) >= 2) and not re.search(
                rf"(?<![\d.]){re.escape(digits)}(?!\d)", src_nocomma):
            add("number", digits)
    run_words: set[str] = set()
    for m in _NAME_RUN_RE.finditer(text):
        # A possessive ends a name: "Pope Packaging's Adelaide plant" names Pope Packaging,
        # and "BIL Suisse's" must match a source that says "BIL Suisse".
        runs, cur = [], []
        for w in m.group(0).split():
            base = _strip_poss(w)
            cur.append(base)
            if base != w:
                runs.append(cur)
                cur = []
        runs.append(cur)
        for words in runs:
            while words and words[0].lower() in _NAME_STOP:
                words.pop(0)
            if sum(w[:1].isupper() for w in words) < 2:
                continue
            run_words |= {w.lower() for w in words}
            phrase = " ".join(words)
            if re.sub(r"[-–]", " ", phrase.lower()) not in low and not _is_place(phrase):
                add("name", phrase)
    for m in _CAP_WORD_RE.finditer(text):
        w = _strip_poss(m.group(0).strip("'’-"))
        if (len(w) < 4 or w.isupper() or w.lower() in _NOT_NAMES or w.lower() in run_words
                or w.lower() in low or _is_place(w)):
            continue
        before = text[:m.start()].rstrip()
        if not before or before[-1] in ".!?:":
            continue
        add("name1", w)
    return out


def _flat_text(obj, skip: tuple = ()) -> str:
    """Every string inside a model reply, tags stripped, joined for the prose scan.
    `skip` names dict keys to leave out."""
    if isinstance(obj, str):
        return re.sub(r"<[^>]+>", "", obj)
    if isinstance(obj, dict):
        return " ".join(_flat_text(v, skip) for k, v in obj.items() if k not in skip)
    if isinstance(obj, (list, tuple)):
        return " ".join(_flat_text(v, skip) for v in obj)
    return ""


_NON_LATIN = re.compile(r"[Ѐ-ӿ֐-׿؀-ۿ฀-๿"
                        r"぀-ヿ㐀-䶿一-鿿가-힯]")
# Function words from other Latin-script languages that are not English words. Left out on
# purpose: die, per, van, los, las, con, del, est, ve -- each occurs in English copy
# ("die casting", "per year", "Los Angeles", "we've").
_FOREIGN_STOP = frozenset(
    "les des une pour dans avec sur und der das ist mit für nicht ein eine auf "
    "por para una il della che het een voor niet também não uma bir için".split())


def non_english_phrases(text: str) -> list[str]:
    """Flag a reply that drifted out of English: mostly non-Latin script, or a dense run of
    foreign function words (a French company name alone will not reach the threshold)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return []
    if len(_NON_LATIN.findall(text)) / len(letters) > 0.15:
        return ["non-English script"]
    # keep contractions whole ("we've"), straight or curly apostrophe
    words = re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?", text.lower())
    hits = [w for w in words if w in _FOREIGN_STOP]
    if len(hits) >= 3 and len(hits) / max(len(words), 1) >= 0.08:
        return [" ".join(hits[:3])]
    return []


def prose_flags(data, profile=None, source: str | None = None) -> list[tuple[str, str]]:
    """(kind, flagged phrase) for every style/evidence defect in a model reply: overclaims,
    meta-commentary about the source (when validate_brief has that check), banned
    words (after masking the lead's own brand names) and, when `source` (the model's own
    input) is given, units/numbers/names the input never stated."""
    import validate_brief
    text = _flat_text(data)
    found: list[tuple[str, str]] = []
    checks = [("overclaim", validate_brief.overclaim_warnings(text)),
              ("language", non_english_phrases(text))]
    meta = getattr(validate_brief, "meta_commentary_warnings", None)
    if meta:
        checks.append(("meta", meta(text)))
    masked = validate_brief.mask_entities(text, profile) if profile is not None else text
    checks.append(("banned", validate_brief.banned_word_warnings(masked)))
    for kind, warnings in checks:
        for w in warnings:
            m = re.search(r"\('(.+?)'\)", w) or re.search(r"'(.+?)'", w)
            found.append((kind, m.group(1) if m else w[:40]))
    if source is not None:
        for _, phrase in unsupported_facts(_prose_text(data), source):
            found.append(("unsupported", phrase))
    return found


def _prose_text(obj) -> str:
    """The prose strings of a reply (8+ words), each ended with a full stop so a name run
    cannot span two fields. Short labels -- roles ("Head of Procurement"), Title Case
    step titles, event_country -- are not claims and only caused needless retries."""
    out: list[str] = []

    def walk(o):
        if isinstance(o, str):
            t = re.sub(r"<[^>]+>", "", o).strip()
            if len(t.split()) >= 8:
                out.append(t if t[-1:] in ".!?" else t + ".")
        elif isinstance(o, dict):
            for k, v in o.items():
                if k != "event_country":
                    walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
    walk(obj)
    return " ".join(out)


_POINTS_BACK = re.compile(r"(?:<[^>]+>|\s)*(?:this|that|these|those|it|such)\b", re.I)


def _drop_unsupported_sentences(data, source: str):
    """Copy of `data` with every sentence that states a unit or name the input lacks removed.
    Numbers and lone words are retry-only (too noisy to cut on). A sentence with unbalanced
    <b>/<i> is kept so the markup stays valid, and a string is never emptied."""
    if isinstance(data, dict):
        return {k: (v if k == "event_country" else _drop_unsupported_sentences(v, source))
                for k, v in data.items()}
    if isinstance(data, list):
        return [_drop_unsupported_sentences(v, source) for v in data]
    if not isinstance(data, str):
        return data
    parts = re.split(r"(?<=[.!?])\s+", data)
    if len(parts) < 2:
        return data
    kept: list[str] = []
    dropped = False
    for s in parts:
        bad = any(k in ("unit", "name") for k, _ in unsupported_facts(s, source))
        # "That shift could..." means nothing once the sentence before it is gone
        bad = bad or (dropped and bool(_POINTS_BACK.match(s)))
        dropped = False
        balanced = all(len(re.findall(rf"<{t}\b[^>]*>", s, re.I)) ==
                       len(re.findall(rf"</{t}>", s, re.I)) for t in ("b", "i"))
        if bad and balanced:
            # keep structural wrapper tags (</p>, <li>...) that rode on the dropped sentence
            lead = re.match(r"(?:<(?!/?[bi]\b)[^>]+>)+", s)
            tail = re.search(r"(?:</(?![bi]>)[^>]+>)+$", s)
            if kept and tail:
                kept[-1] += tail.group(0)
            if lead and not kept:
                kept.append(lead.group(0))
            dropped = True
            continue
        kept.append(s)
    out = " ".join(kept).strip()
    return out if out and re.sub(r"<[^>]+>", "", out).strip() else data


def chat_json_guarded(system: str, user: str, *, max_tokens: int, profile=None, default=None):
    """llm.chat_json, then ONE regeneration if the prose carries an overclaim, meta-
    commentary about the source, a banned word, or a unit/number/name the input never
    stated. The retry quotes the flagged phrases; whichever reply has fewer flags is kept,
    and sentences still stating an unsupported unit or name are dropped. Every prose-
    producing call goes through here so no section can skip the guard."""
    import llm
    default = {} if default is None else default
    source = system + "\n" + user
    data = llm.chat_json(system, user, max_tokens=max_tokens, default=default)
    flags = prose_flags(data, profile, source)
    if not flags:
        return data
    quoted = "; ".join(f'"{p}"' for k, p in flags[:4] if k not in ("unsupported", "language"))
    unsupported = [p for k, p in flags if k == "unsupported"][:4]
    note = "\n\nYour answer broke the writing rules. Write it again and fix this:"
    if quoted:
        note += f"\n- Remove or reword {quoted}. Use could or may for anything the news does not say."
    if unsupported:
        note += ("\n- Remove these, they are not in the input: "
                 + "; ".join(f'"{p}"' for p in unsupported) + ".")
    note += "\n- Write in English only."
    retry = llm.chat_json(system, user + note, max_tokens=max_tokens, default=default)
    chosen = data
    if retry and len(prose_flags(retry, profile, source)) < len(flags):
        chosen = retry
    chosen = _drop_unsupported_sentences(chosen, source)
    if _has_foreign(chosen, profile, source):
        chosen = _to_english(chosen, max_tokens, profile, source)
    return chosen


def _has_foreign(data, profile, source) -> bool:
    return any(k == "language" for k, _ in prose_flags(data, profile, source))


_TRANSLATE_SYSTEM = ("Translate every text value in this JSON into English. Keep company names, "
                     "product names, numbers and HTML tags exactly as they are. Return the same "
                     "JSON with the same keys.")


def _to_english(data, max_tokens: int, profile, source):
    """A writer fed a French or Dutch article often answers in that language, and asking
    again in the same prompt does not fix it. A separate translate-only call does. The
    translation is used only when it is valid, has the same keys and reads as English."""
    import json
    import llm
    if not isinstance(data, dict):
        return data
    tr = llm.chat_json(_TRANSLATE_SYSTEM, json.dumps(data, ensure_ascii=False),
                       max_tokens=max_tokens, default={})
    if not isinstance(tr, dict):
        return data
    out = dict(data)
    for k, v in data.items():
        new = tr.get(k)
        if k == "event_country" or type(new) is not type(v):
            continue
        if isinstance(v, str) or (isinstance(v, list) and len(new) == len(v)):
            out[k] = new
    return data if _has_foreign(out, profile, source) else out


_BULLET_HEAD = (
    "You are a senior market analyst at Coherent Market Insights. You write the 'Analyst "
    "Opinion' under one news item in a brief for {company}.\n\n"
    "Write exactly 3 bullets, each from a different angle:\n"
    "1. What the news means for {short}'s market or its buyers.\n"
    "2. The opportunity or risk for {short}, naming one of its products.\n"
    "3. A sensible next step for {short}.\n"
    "Each bullet starts with a short judgement in <b></b> (under 12 words), then one or two plain "
    "sentences that back it up with a fact from the news.\n"
    "- Use could or may for anything the news does not say.\n"
    "- Do not copy sentences from the section text, and do not repeat another bullet.\n\n"
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


def seller_context(profile, *, products: bool = True, buyers: bool = False) -> str:
    """Plain-sentence context (products, optionally who buys them, country) for a prompt.

    Written as sentences, not UPPERCASE LABEL lines: a small model echoes label words
    ("the buyer function is ...") into the copy. `buyers` is only wanted where the
    prompt must name who to target. Lines are omitted when the profile has nothing for
    them, so a profile built before these fields existed produces an empty string.
    """
    lines = []
    prods = getattr(profile, "products", None) or []
    if products and prods:
        parts = []
        for p in prods[:8]:
            if isinstance(p, dict):
                name, app = (p.get("name") or "").strip(), (p.get("application") or "").strip()
            else:
                name, app = str(p).strip(), ""
            if name:
                parts.append(f"{name} — {app}" if app else name)
        if parts:
            lines.append("The seller's products: " + "; ".join(parts) + ".")
    funcs = [f for f in (getattr(profile, "buyer_functions", None) or []) if f]
    if buyers and funcs:
        lines.append("Who usually buys them: " + ", ".join(funcs[:5]) + ".")
    market = (getattr(profile, "market", "") or "").strip()
    if market and market != "Global":
        lines.append(f"Country: {market}.")
    return "\n".join(lines)
