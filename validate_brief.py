"""Check a generated brief has all three sections and that they fit the company.

    python validate_brief.py output/cmi_localmodel_immunoact_aditya_v6.html --company ImmunoACT
    python validate_brief.py output/*.html --company ImmunoACT --website https://immunoact.com/

Exists because the failures worth catching are not writing failures. The model writes
fluently about whatever it is handed, so a brief reads fine while a competitor sits in
the Potential Customer slot, or a section is silently absent because nothing matched.
Both shipped during development and neither is visible without reading the whole thing.

Four checks, in the order they matter:

  STRUCTURE  all three sections present and non-empty
  ROLES      the company in each slot holds the right relationship to the seller --
             a named competitor must never appear as a Potential Customer
  FIT        the copy talks about what the seller actually sells
  EMAIL      every section that survived STRUCTURE above is STILL present after
             running through emailify() -- the actual conversion applied at send
             time. emailify.py's own docstring says a CSS class it doesn't recognise
             is "silently skipped rather than half-rendered", so a brief can pass
             every check above and still ship with a whole section missing
             specifically in the version that reaches an inbox, if a template edit
             ever drifts out of sync with emailify's selectors. This check is what
             catches that before a send, not after a human notices a live email.

Exit code is 0 only when nothing FAILS; warnings alone still pass so this can gate a
send without blocking on judgement calls.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import config as _config

SECTIONS = ("Potential Customer", "Industry Outlook", "Competition Outlook")

# 2026-10-19: report briefs are capped at 2 sections (Your Market + Competition
# Outlook) per the cmiinsights.info redesign -- the middle slot (previously "Market
# Moves", aliased to "Industry Outlook" below) is no longer generated at all, not
# just occasionally thin. Demanding it would make every correctly-built report brief
# warn for a section it was never supposed to have, the same false-failure shape the
# "Your Market" alias below was added to fix for section 1.
REPORT_SECTIONS = ("Potential Customer", "Competition Outlook")

# A report brief replaces section 1 with a market topic card, built around the topic
# the lead enquired about. Demanding "Potential Customer" there failed every report
# brief for a section it is not supposed to have. "Your Market" was the fixed label
# compose.py rendered for this section before the 2026-10-22 fix; it's kept here only
# as a fallback for briefs built before that fix -- the STRUCTURE check's own
# report-brief section-1 handling now finds it structurally (first "prospect"-kind
# div) rather than relying on fixed label text, since the label is the dynamic topic
# name per brief and has no fixed string to alias anymore.
SECTION_ALIASES = {
    "Potential Customer": ("Potential Customer", "Potential Deal",
                          "Potential Partnership", "Your Market"),
}

# The three headings a lead-type-aware brief may render for section 1's prospect
# card (compose._prospect picks by lead_type: a company gets "Potential Customer",
# an investor gets "Potential Deal", a non-profit/university/public body gets
# "Potential Partnership"). Kept separate from SECTION_ALIASES because the role
# checks below must NOT treat a report brief's old "Your Market" label as a company.
CUSTOMER_LABELS = ("Potential Customer", "Potential Deal", "Potential Partnership")


def visible_text(html: str) -> str:
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    t = re.sub(r"<br\s*/?>", "\n", t)
    t = re.sub(r"</(p|div|td|tr|h1|h2|h3|li)>", "\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&#39;", "'"), ("&middot;", "·"),
                 ("&rsquo;", "'"), ("&mdash;", "-"), ("&deg;", "°")):
        t = t.replace(a, b)
    t = re.sub(r"&[a-z]+;|&#\d+;", " ", t)
    return re.sub(r"[ \t]+", " ", t)


def section_company(text: str, label) -> str:
    """The company name rendered under a section's flag line.

    The template puts the flag ('● Potential Customer · <industry>') immediately above
    the company heading, so the first substantial line after it is the subject.
    `label` may be one string or several accepted headings for the same slot.
    """
    labels = (label,) if isinstance(label, str) else tuple(label)
    for one in labels:
        m = re.search(rf"●?\s*{re.escape(one)}\s*·[^\n]*\n(.*)", text)
        if not m:
            continue
        for line in m.group(1).split("\n")[:4]:
            line = line.strip()
            if 2 < len(line) < 70 and not line.lower().startswith(("why ", "who ", "their ")):
                return line
    return ""


_GENERIC_BUYER_WORDS = {"companies", "company", "brands", "buyers", "makers",
                        "manufacturers", "providers", "institutional", "industrial"}


def buyer_context_terms(profile) -> set[str]:
    """Words that mark a customer slot as a BUYER vertical for this seller.

    Replaces the old pharma-only list ("hospital", "clinic", ...) that flagged every
    non-healthcare lead. Built from the profile's own buyer verticals and buyer
    functions, so it works for any tenant, and returns an empty set when the profile
    predates those fields (getattr default) rather than raising.
    """
    terms: set[str] = set()
    for src in (getattr(profile, "industries", None),
                getattr(profile, "buyer_functions", None)):
        if not src:
            continue
        if isinstance(src, str):
            src = [src]
        for item in src:
            for w in re.findall(r"[a-z][a-z0-9-]{3,}", str(item).lower()):
                if len(w) > 4 and w not in _GENERIC_BUYER_WORDS:
                    terms.add(w)
    return terms


def norm(s: str) -> str:
    # NFKD + drop combining marks BEFORE the ASCII-only strip below, so an accented
    # Latin letter degrades to its plain form (í -> i) instead of vanishing outright
    # (the old behaviour turned "Vítkovice" into "v tkovice" -- a gap, not a letter --
    # which then failed every substring/acronym match against vitkovicesteel.com even
    # though the names plainly match). Non-Latin scripts (Cyrillic, CJK, Arabic) have
    # no Latin decomposition and still strip to nothing here -- that gap is real and
    # not fixed by this change; DOMAIN comparisons for those companies are silently
    # unable to confirm a match either way, not incorrectly confirmed.
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s*[(\[][^)\]]*[)\]]", " ", s.lower())
    s = re.sub(r"\b(ltd|limited|inc|corp|plc|pvt|private|laboratories|labs|group|holdings)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def name_overlap(a: str, b: str) -> bool:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    ha, hb = " ".join(na.split()[:2]), " ".join(nb.split()[:2])
    return len(ha) > 5 and ha == hb


def check(path: str, profile, *, email: str = "", kind: str = "",
          topic: str = "", lead_country: str | None = None) -> tuple[list[str], list[str]]:
    fails, warns, _llm_fails, _llm_warns = check_detailed(
        path, profile, email=email, kind=kind, topic=topic,
        lead_country=lead_country)
    return fails + _llm_fails, warns + _llm_warns


def geo_scrub(text: str, profile) -> str:
    """Strip salutation/header/footer boilerplate and the lead's own names before
    the geography check counts countries (2026-11 live run: the recipient's first
    name "Sanaa" was read as Yemen's capital Sanaa and flagged OFF-COUNTRY).

    Removed: "Hello, <name>." salutation lines, "Prepared for ..." lines,
    "Market Signal Brief · ..." lines, and word-boundary occurrences of the
    recipient's first name and the lead company's name — none of that is
    evidence about where the NEWS happened.
    """
    t = text
    t = re.sub(r"(?im)^\s*hello,?\s+[a-z .'-]{1,40}[.!?]?\s*$", " ", t)
    t = re.sub(r"(?im)^.*\bprepared for\b.*$", " ", t)
    t = re.sub(r"(?im)^.*market signal brief\s*·.*$", " ", t)
    # the recipient's first name must never count as a city/country token
    r = (getattr(profile, "recipient", "") or "").strip()
    if r and r.lower() not in ("there", "there,", "hi", "hello"):
        first = re.split(r"\s+", r)[0].strip(".,")
        if first and len(first) >= 4:
            t = re.sub(rf"(?<![0-9a-z]){re.escape(first.lower())}(?![0-9a-z])",
                       " ", t, flags=re.I)
    # the lead company's name likewise ("Georgia-Pacific" must not read as the
    # country or the US state)
    comp = (getattr(profile, "company", "") or "").strip()
    if comp:
        comp_norm = re.sub(r"\s+", " ", re.sub(r"[^0-9a-z ]", " ", comp.lower())).strip()
        if comp_norm:
            t = re.sub(rf"(?<![0-9a-z]){re.escape(comp_norm)}(?![0-9a-z])",
                       " ", t, flags=re.I)
    return t


def main_content_html(html: str) -> str:
    """The model-written part of a brief, with the fixed template blocks removed.

    Section cards, analyst bullets and the lede are written from news; testimonials,
    the "Inside Coherent Market Insights" caps/projects showcase, certification
    tiles, per-section CTAs, the fixed banner line and the footer are static template
    copy that is identical (or semi-random marketing text) in every brief. Running
    style/jargon checks over the static copy reports defects nobody can fix. The cut
    is made at the "Inside Coherent Market Insights" eyebrow, which precedes the
    whole static tail of <main>.
    """
    h = html or ""
    cuts = [h.find(m) for m in ("Inside Coherent Market Insights",
                                '<footer class="foot"', "<footer class='foot'")
            if h.find(m) != -1]
    if cuts:
        h = h[:min(cuts)]
    h = re.sub(r"<header\b[^>]*>.*?</header>", " ", h, flags=re.S | re.I)
    h = re.sub(r'<p class="agentline">.*?</p>', " ", h, flags=re.S | re.I)
    h = re.sub(r'<div class="seccta">.*?</div>', " ", h, flags=re.S | re.I)
    # Fixed labels inside the dynamic blocks: the "Analyst Opinion" tag and the
    # "CMI Intelligence · date · ✓ Analyst-reviewed" badge. Left in, they glue onto
    # the first bullet and turn repeated bullets into a duplicate-sentence hit.
    h = re.sub(r'<div class="tag">\s*Analyst Opinion\s*</div>', " ", h, flags=re.I)
    h = re.sub(r'<div class="prov">.*?</div>', " ", h, flags=re.S | re.I)
    return h


def main_content_text(html: str) -> str:
    """Visible text of the model-written content only (see main_content_html)."""
    return visible_text(main_content_html(html))


LEDE_LEAK_PHRASES = ("bolded label phrase", "optionally followed", "generic words of color",
                     "as indicated by the trigger", "json", "return a ", "<b></b>")
_LEDE_BLANK_SLOT = re.compile(r"\b(?:the|a|an)\s+(?:in|of|for|by|where|on|and)\b", re.I)
_LEDE_EMPTY_SLOT = re.compile(r"\b(?:by|where|from)\s+(?:in|is|are)\b", re.I)


def lede_leaks(lede_text: str, lede_html: str = "") -> list[str]:
    """Prompt instructions or unfilled slots that leaked into the lede."""
    low = (lede_text or "").lower()
    raw_low = (lede_html or "").lower()
    out: list[str] = []
    for phrase in LEDE_LEAK_PHRASES:
        hit = phrase in raw_low if phrase == "<b></b>" else phrase in low
        if hit:
            out.append(f"LEDE LEAK ERROR: the lede contains '{phrase}' -- a writing "
                       f"instruction or empty tag leaked into the copy instead of a "
                       f"finished sentence.")
    for m in _LEDE_BLANK_SLOT.finditer(low):
        out.append(f"LEDE LEAK ERROR: the lede has a blank slot ('{m.group(0)}') -- a "
                   f"name, place or label the model was supposed to fill was left out.")
        break
    m = _LEDE_EMPTY_SLOT.search(low)
    if m:
        out.append(f"LEDE LEAK ERROR: the lede has a broken slot ('{m.group(0)}') -- a "
                   f"clause was cut after its preposition.")
    return out


_OVERCLAIM_PATTERNS = (
    ("will raise/increase demand", re.compile(
        r"\bwill\s+(?:raise|increase|boost|drive|lift)\b.{0,40}"
        r"\b(?:demand|sales|orders|adoption)\b", re.I)),
    ('"lift sales"', re.compile(r"\blift\s+sales\b", re.I)),
    ('"more inquiries"', re.compile(r"\bmore\s+inquir(?:y|ies)\b", re.I)),
    ('"more enquiries"', re.compile(r"\bmore\s+enquir(?:y|ies)\b", re.I)),
    ('"price sensitivity"', re.compile(r"\bprice\s+sensitivity\b", re.I)),
    ("a penalty claim", re.compile(r"\bpenalt", re.I)),
    ('"stricter"', re.compile(r"\bstricter\b", re.I)),
)


_HEDGE_WORDS = re.compile(
    r"\b(?:could|may|might|can|possibly|potentially|likely)\b", re.I)


def _sentences(text: str) -> list[str]:
    """Split on . ! ? followed by whitespace (or end of text).

    Pure and deliberately simple: a sentence is the unit the over-claim and
    meta-commentary checks reason about, because a hedge anywhere in the same
    sentence qualifies the whole claim it carries.
    """
    out: list[str] = []
    for chunk in re.split(r"(?<=[.!?])\s+", text or ""):
        s = chunk.strip()
        if s:
            out.append(s)
    return out


def overclaim_warnings(text: str) -> list[str]:
    """Commercial effects the signal does not state (demand, sales, penalties...).

    Judged per SENTENCE: a sentence is flagged only when it matches a pattern
    AND carries no hedge word ('could', 'may', 'might', 'can', 'possibly',
    'potentially', 'likely'). "Waters may see more inquiries from food labs" is
    a qualified inference and passes; "Price sensitivity will rise as inflation
    bites" states it as fact and is flagged. Each pattern is still reported once
    (its first unhedged hit), so the message format callers quote is unchanged.
    """
    out: list[str] = []
    seen: set[str] = set()
    for sentence in _sentences(text):
        if _HEDGE_WORDS.search(sentence):
            continue
        for name, pat in _OVERCLAIM_PATTERNS:
            if name in seen:
                continue
            m = pat.search(sentence)
            if m:
                seen.add(name)
                out.append(f"OVERCLAIM: the copy uses {name} ('{m.group(0).strip()}') -- "
                           f"the signal does not state that commercial effect. Write what "
                           f"was announced, or qualify any implication with 'could'/'may' "
                           f"and its condition.")
    return out


# Talk about the source/instructions rather than the market. Kept as patterns
# (not bare words) so ordinary market copy -- "the signal is a direct fit for its
# products" is flagged deliberately, while the fixed plastic opener "This week's
# brief covers N signals" (plural, no "the"/"this") never matches.
_META_COMMENTARY_PATTERNS = (
    re.compile(r"\bthe signal\b", re.I),
    re.compile(r"\bthis signal\b", re.I),
    re.compile(r"\bno figures\b", re.I),
    re.compile(r"\bnot (?:stated|given|disclosed) in the (?:signal|source|article)\b", re.I),
    re.compile(r"\btreat (?:it|this|the risk) as a scenario\b", re.I),
    re.compile(r"\bas indicated by\b", re.I),
    re.compile(r"\bthe (?:source|article) (?:says|does not)\b", re.I),
)


def meta_commentary_warnings(text: str) -> list[str]:
    """Sentences that talk about the source or the instructions, not the market.

    Pure text -> list[str]; one warning per distinct offending sentence, so a
    sentence matching several patterns (e.g. "No figures in the signal, so treat
    the risk as a scenario") is reported once.
    """
    out: list[str] = []
    seen: set[str] = set()
    for sentence in _sentences(text):
        normed = " ".join(sentence.lower().split())
        if normed in seen:
            continue
        for pat in _META_COMMENTARY_PATTERNS:
            m = pat.search(sentence)
            if m:
                seen.add(normed)
                out.append(f"META COMMENTARY: the copy talks about the source or the "
                           f"instructions ('{m.group(0).strip()}') instead of the market "
                           f"-- write what the announcement means for the reader, not "
                           f"what the signal does or does not give.")
                break
    return out


BANNED_WORDS = ("leverage", "ecosystem", "landscape", "throughput", "go-to-market",
                "synergies", "robust", "seamless", "pivotal", "crucial", "unlock",
                "empower", "harness", "game-changer", "double down", "table stakes")


def banned_word_warnings(text: str) -> list[str]:
    """Marketing jargon that voice.py's style rules ban from main content."""
    out: list[str] = []
    for w in BANNED_WORDS:
        if re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", text or "", re.I):
            out.append(f"BANNED WORD: '{w}' is on the style blacklist -- say the "
                       f"concrete thing instead.")
    return out


def _rival_names(profile) -> list[str]:
    """Competitor names plus their written forms (Profile.competitor_aliases and local
    names), for the role checks. Never used for the seller's own name."""
    import tenant as _tenant
    out: list[str] = []
    for c in (getattr(profile, "competitors", None) or []):
        out += _tenant.rival_forms(profile, c) if isinstance(c, str) else [c]
    return out


def _entity_names(profile) -> list[str]:
    """The lead's own proper nouns: company, product names and competitors."""
    names: list[str] = [getattr(profile, "company", "") or ""]
    for p in (getattr(profile, "products", None) or []):
        names.append(p.get("name") if isinstance(p, dict) else str(p or ""))
    for c in (getattr(profile, "competitors", None) or []):
        names.append(c.get("name") if isinstance(c, dict) else str(c or ""))
    return [n.strip() for n in names if isinstance(n, str) and n.strip()]


def mask_entities(text: str, profile, *, token: str = "[entity]") -> str:
    """Replace the lead's own proper nouns with a neutral token before the banned-word
    check, so a BRAND word that is also a banned generic term is not flagged while
    the generic word still is.

    "Empower 3 software" (a Waters product named Empower) was a BANNED WORD hit on
    every brief that mentioned the product; "we empower lab teams" is a genuine
    style defect. A capitalised entity name is therefore matched only in its
    proper-noun casing; an all-lowercase entity name (some brands are) is matched
    case-insensitively since it carries no casing signal to distinguish.

    Pure: no I/O, no imports beyond `re`; longest names first so "Empower
    chromatography software" is consumed before the bare "Empower".
    """
    out = text or ""
    for name in sorted(set(_entity_names(profile)), key=len, reverse=True):
        flags = re.I if name == name.lower() else 0
        out = re.sub(rf"(?<![0-9a-z]){re.escape(name)}(?![0-9a-z])",
                     token, out, flags=flags)
    return out


def duplicate_sentence_warnings(text: str) -> list[str]:
    """The same full sentence appearing twice in the main content."""
    seen: dict[str, int] = {}
    for sentence in re.split(r"(?<=[.!?])\s+", text or ""):
        normed = re.sub(r"\s+", " ", sentence.strip().lower())
        if len(normed.split()) < 8:
            continue
        seen[normed] = seen.get(normed, 0) + 1
    return [f"DUPLICATE SENTENCE: this sentence appears {n} times -- '{s[:140]}'"
            for s, n in seen.items() if n > 1]


def company_case_warnings(text: str, company: str) -> list[str]:
    """The seller's mixed/title-cased name rendered in all lowercase in the copy."""
    c = (company or "").strip()
    if not c or c.lower() == c or c == c.upper():
        return []
    if c in (text or ""):
        return []
    if re.search(rf"(?<![0-9a-z]){re.escape(c.lower())}(?![0-9a-z])", (text or "").lower()):
        return [f"COMPANY CASE: the seller is '{c}' but the copy writes "
                f"'{c.lower()}' -- a product word, not the company name."]
    return []


def check_detailed(path: str, profile, *, email: str = "", kind: str = "",
                   topic: str = "", lead_country: str | None = None
                   ) -> tuple[list[str], list[str], list[str], list[str]]:
    """Same checks as `check()`, but keeps the LLM judge's findings separate from the
    text-rule findings so a caller can tell "a keyword rule caught this" apart from
    "the AI judge flagged this" -- the latter needs a human to read the reasoning and
    decide, the former is usually a clear-cut structural problem.

    Returns (fails, warns, llm_fails, llm_warns). `fails`/`warns` are text-rule findings
    only; `llm_fails`/`llm_warns` are what the LLM judge added, each string carrying its
    own one-line reasoning already baked in (see llm_validate.judge_brief).

    The judge only runs when the text rules alone would PASS or WARN -- not on FAIL.
    A brief the rules already reject is already held back from sending; spending an
    extra GPU call asking the judge to grade something that will not ship anyway is
    wasted cost for no benefit.

    `email`/`kind`/`topic` are optional and enable the topic-alignment judge check:
    does the brief's actual content match what the recipient was promised, checked
    against `topic` (the lead's real company for kind='signal', or the market/report
    name the lead enquired about for kind='report') -- independent of `profile`, which
    a caller cannot fully trust (see tenant.Profile.from_company's docstring on the
    2026-10-07 incident this exists to catch). Omitting them just skips that check,
    same as before this parameter existed.
    """
    html = open(path, encoding="utf-8").read()
    text = visible_text(html)
    fails: list[str] = []
    warns: list[str] = []

    # ---- STRUCTURE ---------------------------------------------------------
    # A brief with no findable prospect or rival is a real outcome, not a defect: some
    # markets simply have no rival news in the window. Dropping the section is the
    # honest response -- better than the stretched picks that came out of forcing one.
    # But it must stay RARE, so losing one section is a warning and losing two is a
    # failure: at that point the brief is a single card and not worth sending.
    #
    # 2026-10-19: report briefs are built and rendered with only 2 sections by
    # design (REPORT_SECTIONS) -- checking them against the signal brief's 3-section
    # SECTIONS list would warn every correctly-built report brief for a section
    # (the old "Market Moves" slot) that compose.py/brief_builder.py no longer
    # generate at all, not just occasionally.
    expected_sections = REPORT_SECTIONS if kind == "report" else SECTIONS
    missing: list[str] = []
    present = {}
    for s in expected_sections:
        body = ""
        # 2026-10-22: a report brief's section 1 pill/flag is the bare market-topic
        # name (e.g. "Wireless Access Point Market"), not the fixed "Your Market"
        # string -- confirmed against the Fortinet industry-brief sample, which has
        # neither "Your Market" nor "Potential Customer" anywhere. There is no fixed
        # alias text left to search for, since the topic differs per brief. A report
        # brief has exactly two "prospect"-kind sections (no "story"-kind section at
        # all, per REPORT_SECTIONS/compose.py), so for "Potential Customer" in a
        # report brief, take the body of the FIRST <div class="prospect"> block
        # structurally -- the one before "Competition Outlook" -- rather than
        # matching on any particular label text.
        if kind == "report" and s == "Potential Customer":
            comp_idx = text.find("Competition Outlook")
            first_prospect_m = re.search(r'<div class="prospect">(.*?)</div>\s*</div>',
                                         html, re.S)
            if first_prospect_m:
                candidate = re.sub(r"<[^>]+>", " ", first_prospect_m.group(1))
                # Guard against the body match actually belonging to Competition
                # Outlook (e.g. if a future redesign reorders sections) by checking
                # it does not itself contain that label.
                if "Competition Outlook" not in candidate:
                    body = candidate
        if not body:
            for alias in SECTION_ALIASES.get(s, (s,)):
                m = re.search(rf"{re.escape(alias)}\s*·(.{{0,1500}})", text, re.S)
                if m:
                    body = m.group(1)
                    break
        if not body:
            # Story-kind sections (Industry Outlook) render their flag as just
            # "<bullet> <label>" with no "· <subcategory>" suffix -- compose._story
            # always sets flag="" and the template falls back to the plain label
            # (see templates/cmi_brief.html's {% else %} branch) -- so the
            # "label ·" pattern above never matches one. Without this fallback a
            # perfectly complete Industry Outlook section reads as entirely
            # missing, because the only thing distinguishing it from plain running
            # prose is the bare label text, not a flag-line separator.
            for alias in SECTION_ALIASES.get(s, (s,)):
                m2 = re.search(rf"\b{re.escape(alias)}\b", text)
                if m2:
                    body = text[m2.end():m2.end() + 1500]
                    break
        present[s] = bool(body and len(body.split()) > 25)
        if not present[s]:
            missing.append(s)

    if len(missing) >= 2:
        fails.append("MISSING SECTIONS: " + ", ".join(missing) +
                     " — too little left to send.")
    elif missing:
        warns.append(f"MISSING SECTION: {missing[0]} — no usable signal found. "
                     f"Acceptable occasionally; check the feed if it repeats.")

    # ---- EMAIL --------------------------------------------------------------
    # Every section-rendering element in the source (main.body's direct children
    # carrying "prospect" or "story" in their class -- the two classes
    # emailify.emailify()'s own dispatch loop looks for to render a section body)
    # must actually be recognised by that same dispatch logic. emailify.py's own
    # docstring warns that a class it does not recognise is "silently skipped
    # rather than half-rendered" -- so a template edit renaming one of these two
    # classes drops that section's body from every SENT email while the browser
    # HTML (the only thing validated until now) looks completely fine.
    #
    # This is a STRUCTURAL check (does the class spelling match), not a text
    # comparison. An earlier version tried matching section label text or
    # distinctive body words in emailify's rendered output, but a real brief
    # repeats its own industry vocabulary throughout every section AND in the
    # static "Inside CMI" showcase -- so words from a dropped section often still
    # appear to "survive" via a completely unrelated part of the same document,
    # producing a false negative on the exact failure this check exists to catch.
    # Comparing element counts by class, independent of wording, does not have
    # that blind spot: emailify.py's dispatch condition is copied here verbatim
    # (`"prospect" in cls or "story" in cls`) so this check breaks in lockstep
    # with emailify.py itself rather than silently drifting out of sync with it.
    if 'class="body"' in html or "class='body'" in html:
        try:
            from bs4 import BeautifulSoup as _Soup

            soup = _Soup(html, "html.parser")
            main = soup.select_one("main.body")
            recognized = 0
            unrecognized_classes: list[str] = []
            if main is not None:
                for el in main.find_all(recursive=False):
                    cls = el.get("class") or []
                    if "prospect" in cls or "story" in cls:
                        recognized += 1
                    elif any(c in ("prospect", "story") for c in
                            [c2.split("-")[0].split("_")[0] for c2 in cls]):
                        # Looks like a renamed/typo'd variant of a class emailify
                        # expects exactly (e.g. "prospect-v2", "story_alt") --
                        # emailify's dispatch does an exact membership check, so
                        # this element would be silently skipped even though it is
                        # clearly meant to be one of the two section-body kinds.
                        unrecognized_classes.append(" ".join(cls))
            n_sections_expected = sum(1 for s in expected_sections if present.get(s))
            if main is not None and recognized < n_sections_expected:
                fails.append(
                    f"EMAIL ERROR: {n_sections_expected} section(s) found in the "
                    f"browser HTML but only {recognized} carry a class "
                    f"emailify.py's dispatch loop recognises ('prospect' or "
                    f"'story') -- the rest would be silently dropped from every "
                    f"SENT email."
                    + (f" Suspicious class attribute(s) found: "
                       f"{', '.join(unrecognized_classes)}." if unrecognized_classes else "")
                    + " Check templates/cmi_brief.html's CSS classes against "
                      "emailify.py's selectors.")
        except Exception as e:  # noqa: BLE001 -- a broken emailify check must not block
                                # a send the browser-HTML checks already cleared; this
                                # adds scrutiny, it must not become a new failure point
            warns.append(f"EMAIL check could not run: {type(e).__name__}: {e}")

    # ---- ROLES -------------------------------------------------------------
    cust = section_company(text, CUSTOMER_LABELS)
    if not cust and kind == "report":
        # A report brief's section 1 is a market topic (e.g. "Wireless Access Point
        # Market"), not a company -- the buyer-role checks below ("is this a known
        # competitor") have nothing to judge there and must not fire. Previously
        # detected via the fixed "Your Market ·" string that rendered in the HTML;
        # as of the 2026-10-22 fix that string is no longer rendered at all (section
        # 1's pill/flag is the bare topic name), so gate on `kind` directly instead
        # -- more robust than matching rendered text that can vary per brief anyway.
        cust = ""
    comp = section_company(text, "Competition Outlook")

    for rival in _rival_names(profile):
        if cust and name_overlap(cust, rival):
            fails.append(f"ROLE ERROR: Potential Customer is '{cust}', a known competitor "
                         f"(matches '{rival}'). A rival must appear in Competition Outlook.")
            break

    # The competitor list only names the rivals someone thought to write down. A supplier
    # in the seller's OWN category is just as wrong in the customer slot even when it is
    # absent from that list -- Dr Reddy's launching an immuno-oncology biosimilar passed
    # every name check while being plainly not a buyer of CAR-T. Catch it by category:
    # the customer should read like a BUYER vertical, not like the seller's own trade.
    #
    # 2026-11: the old buyerish list was pharma-only ("hospital", "clinic", ...), so
    # every non-healthcare lead looked like a seller-category company. Buyerish is now
    # profile-driven: any word from the profile's own buyer verticals (industries) or
    # buyer_functions, which works for any tenant and cannot drift between runs.
    if cust:
        seller_trade = {w for w in re.findall(r"[a-z][a-z0-9-]{4,}",
                                              (profile.value_prop or "").lower())} - {
            "india", "indian", "supplies", "approved", "first", "targeted", "relapsed",
            "refractory", "patients", "treating"}
        cust_ctx = ""
        _cust_labels_re = "|".join(re.escape(l) for l in CUSTOMER_LABELS)
        m = re.search(rf"(?:{_cust_labels_re})\s*·(.{{0,900}})", text, re.S)
        if m:
            cust_ctx = m.group(1).lower()
        # Whole-word match: a substring test let "payment" match "payments" in an
        # unrelated heading and silently suppress this warning.
        cust_words = set(re.findall(r"[a-z][a-z0-9-]{3,}", cust_ctx))
        buyerish = bool(buyer_context_terms(profile) & cust_words)
        sellerish = sum(1 for w in seller_trade if len(w) > 5 and w in cust_ctx)
        if cust_ctx and not buyerish and sellerish >= 2:
            warns.append(f"Potential Customer '{cust}' reads like a company in the seller's own "
                         f"category, not a buyer. Check it is not a rival or supplier.")
        # A regulator, ministry or trade body is a real signal but has no purchase order
        # to give. Left unchecked the model invents a reason it might buy. The list is
        # shared with the generator's own candidate filter (config.NON_BUYER_TERMS) so
        # a name rejected at generation time cannot sail through validation.
        _nb = _config.NON_BUYER_TERMS
        if _config.government_buyers_ok(profile):   # this lead sells to government bodies
            _nb = [w for w in _nb if not _config.is_government(w)]
        if any(w in cust.lower() for w in _nb):
            fails.append(f"ROLE ERROR: Potential Customer is '{cust}', which regulates or "
                         f"governs the market rather than buying in it.")

    if comp:
        if not any(name_overlap(comp, r) for r in _rival_names(profile)):
            warns.append(f"Competition Outlook features '{comp}', which is not on the "
                         f"competitor list. Verify it actually sells a substitutable product.")
        # Being a real rival is not enough -- the SIGNAL has to be about the shared
        # product category. A diversified major's generics acquisition is a true
        # competitor doing something irrelevant to this seller, which reads as padding.
        try:
            import tenant as _tenant
            terms = [t.lower() for t in
                     ((_tenant._overrides_for(profile.company) or {}).get("product_terms") or [])]
        except Exception:  # noqa: BLE001
            terms = []
        if not terms:
            # Most sellers have no curated product_terms override, which used to
            # mean this check silently never ran for them at all -- confirmed
            # 2026-10-13: sgdp.it/Mailchimp and Offeri/PriceSmart, two Competition
            # Outlook entries with zero real relevance to the seller's actual
            # business, both passed clean because neither seller has an override.
            # Fall back to the same value_prop-derived nouns brief_builder.
            # build_competition() already uses for its own on-category ranking, so
            # this check has real teeth for every seller, not just curated ones.
            # (A local stop-word set, not the one defined later in the FIT section
            # below -- this code runs first, before that name exists.)
            _roles_stop = {"india", "indian", "sells", "makes", "first", "approved",
                          "their", "which", "supplies", "company", "companies",
                          "market", "product", "products", "global"}
            terms = [w for w in re.findall(r"[a-z][a-z0-9-]{4,}",
                                           (profile.value_prop or "").lower())
                     if w not in _roles_stop][:12]
        if terms:
            m = re.search(r"Competition Outlook\s*·(.{0,1200})", text, re.S)
            ctx = (m.group(1) if m else "").lower()
            if ctx and not any(t in ctx for t in terms):
                warns.append(f"Competition Outlook covers '{comp}' but the signal never "
                             f"touches the seller's product category. A rival doing something "
                             f"unrelated is padding, not competitive intelligence.")
    if cust and comp and name_overlap(cust, comp):
        fails.append(f"ROLE ERROR: '{cust}' appears as both customer and competitor.")

    # ---- FIT ---------------------------------------------------------------
    # Does the brief discuss what the seller sells? Built from the value_prop's own
    # distinctive nouns rather than a hardcoded list, so this works for any tenant.
    stop = {"india", "indian", "sells", "makes", "first", "approved", "their", "which",
            "supplies", "company", "companies", "market", "product", "products", "global"}
    vp_terms = {w for w in re.findall(r"[a-z][a-z0-9-]{3,}", (profile.value_prop or "").lower())
                if w not in stop}
    low = text.lower()
    hits = sorted(w for w in vp_terms if w in low)
    if vp_terms and len(hits) < 2:
        fails.append(f"FIT ERROR: the brief barely mentions what {profile.company} sells. "
                     f"Matched only {hits or 'nothing'} from the value proposition.")
    elif vp_terms and len(hits) < 4:
        warns.append(f"Weak topical fit: only {hits} from the value proposition appear.")

    if profile.company.lower() not in low:
        fails.append(f"FIT ERROR: '{profile.company}' is never named in the brief.")

    # ---- TOPIC (report briefs only) -----------------------------------------
    # A report brief promises coverage of a SPECIFIC market the lead enquired about
    # (`topic`, e.g. "Ship Breaking Market") -- unlike FIT above, which checks the
    # brief against its own `profile`, this checks it against the one fact a caller
    # can independently verify from the leads table. This ONLY catches the topic
    # being completely absent -- deliberately NOT extended to a body-density/mention
    # count, because that was tried and measured against the real 2026-10-07 incident
    # brief (profiled as CMI itself, content entirely about hydraulic hammers /
    # Salesforce / GPU-as-a-Service for a lead who asked about ship breaking) and
    # FAILED: the model had mentioned "ship breaking" 18 times, tacked onto sentences
    # about the unrelated real content ("may point to...", "for ship breaking firms
    # that need..."). Counting mentions cannot distinguish "genuinely about X" from
    # "X referenced as a label while discussing Y" -- that needs actual reading
    # comprehension, which is what llm_validate.topic_alignment_verdict is for. Do not
    # re-add a density/count version of this check without re-testing it against that
    # exact brief first.
    if topic:
        topic_terms = {w for w in re.findall(r"[a-z][a-z0-9-]{3,}", topic.lower())
                       if w not in stop}
        # A signal brief's topic is the lead's raw company field, often squashed from the
        # domain ("Vdlhapro", "Thetravelportfolio") while the brief writes "VDL Hapro".
        # Compare with spaces removed, and accept the profile's name when the raw field
        # starts with its first word ("Bluematterconsulting" vs "Blue Matter", "Pgigrup" vs
        # "PGI Engineering") and the brief names the profile's company.
        flat = re.sub(r"[^a-z0-9]", "", low)
        t_flat = re.sub(r"[^a-z0-9]", "", topic.lower())
        c_words = re.findall(r"[a-z0-9]+", (profile.company or "").lower())
        c_flat = "".join(c_words)
        named = (len(t_flat) >= 4 and t_flat in flat) or (
            kind == "signal" and c_words and len(c_words[0]) >= 3
            and t_flat.startswith(c_words[0]) and c_flat in flat)
        if topic_terms and not named and not any(w in low for w in topic_terms):
            fails.append(
                f"TOPIC ERROR: none of the expected subject '{topic}' appears anywhere "
                f"in the brief -- this looks like the wrong report entirely.")

    # ---- DOMAIN (independent of profile) ------------------------------------
    # profile.company is exactly the field a bad profile gets wrong (see
    # tenant.Profile.from_company's 2026-10-07 incident docstring) -- the FIT check
    # above only verifies the brief matches `profile`, which proves nothing when
    # `profile` itself is wrong. profile.company is set verbatim from the lead's own
    # `company` field (tenant.py:318, `company=company` -- never LLM-rewritten), so
    # this check is really "did two different reads of the SAME lead's company agree",
    # not an inference -- it fails only when something upstream substituted an
    # entirely different company (e.g. the cmi_default() fallback this incident
    # exposed), not when the match is merely imperfect.
    #
    # Two match strategies, either one is enough: substring (bharatpetroleum ~
    # "Bharat Petroleum Corporation Limited") and acronym (jrtt ~ "Japan Railway
    # Construction, Transport and Technology Agency", hp ~ "Hewlett-Packard", ibm ~
    # "International Business Machines"). Substring alone was tested against real
    # lead data and produces false FAILs on exactly this acronym pattern, which is
    # common in real B2B/government lead lists -- a validator that cries wolf on
    # legitimate briefs gets ignored or disabled, which is worse than the gap it
    # closes. A domain neither strategy explains is still only surfaced as a FAIL
    # when it is unambiguous (no overlap by either method); genuinely uncertain
    # cases are not silently passed, they are impossible to construct here because
    # company_norm/dom_root are deterministic strings, not probabilities.
    if email and "@" in email:
        import enrich as _enrich

        dom = _enrich.domain_of(email=email)
        if dom:
            dom_root = re.sub(r"\.(com|net|org|co|io|in|biz|info)$", "", dom.split(".")[0])
            dom_root = re.sub(r"[^a-z0-9]", "", dom_root.lower())

            # norm() strips parentheticals entirely (built for a different job: fuzzy
            # "are these roughly the same company name" matching). That is wrong here --
            # a parenthetical is very often the exact alias/acronym that explains the
            # domain ('Kirey (Insirio)' <-> insirio.it, 'Nanjing Iron & Steel (NISCO)'
            # <-> njsteel.com.cn). Tested against the 2026-10-08 regen batch: plain
            # norm() produced 103 FAILs on this check and ~100 of them were false
            # positives of exactly this shape. Build TWO candidates -- the primary name
            # and, if present, the parenthetical -- and accept either one matching.
            paren_m = re.search(r"[(\[]([^)\]]+)[)\]]", profile.company or "")
            candidates = [profile.company or ""]
            if paren_m:
                candidates.append(paren_m.group(1))

            # Acronym stop-words: ONLY words that are never part of a real company
            # initialism. "of"/"the" qualify (Indian Institute of Technology Bombay ->
            # IITB not IIOTB; University of Arkansas -> UARK not UOA). "for"/"and"/"to"
            # do NOT belong here even though they look like the same kind of connector
            # word -- "Connecting for Results" really is branded CFR (cfrincorporated.com),
            # "Tires to Green Recycling" really is branded T2GR. Whether a given
            # company's real acronym keeps or drops a connector is a branding choice
            # with no textual rule to predict it from the name alone -- stop-wording
            # all of them trades IITB/UARK false FAILs for CFR/T2GR false FAILs, it does
            # not remove the ambiguity. "of"/"the" are kept because they are the two
            # connectors observed to NEVER survive into a real acronym across every
            # case checked so far; if a counter-example turns up, move it here too.
            _ACRONYM_STOP = {"of", "the", "de", "la", "le", "der", "die",
                             "das", "und", "a", "an", "at", "in", "on"}

            # Common legal-suffix abbreviations a domain often contracts to, independent
            # of the acronym logic below (Syngene International -> syngeneintl.com: the
            # literal word "international" never appears in the domain, only its
            # standard business abbreviation does). Checked as whole-word substitutions
            # on the word list, not a blind regex replace, so it cannot corrupt an
            # unrelated word that happens to contain one of these as a substring.
            _ABBREV = {"international": "intl", "incorporated": "inc", "corporation": "corp",
                      "limited": "ltd", "company": "co", "technologies": "tech",
                      "technology": "tech", "industries": "ind", "solutions": "sol"}

            def _candidate_match(name: str) -> bool:
                words_all = norm(name).split()
                company_norm = "".join(words_all)
                abbrev_norm = "".join(_ABBREV.get(w, w) for w in words_all)
                words = [w for w in words_all if w not in _ACRONYM_STOP]
                acronym = "".join(w[0] for w in words if w)
                # Substring form that keeps "group"/"inc"/"ltd"/etc -- norm() strips
                # those as corporate-suffix noise (right call for fuzzy name_overlap,
                # the other consumer of norm()), but a domain can legitimately spell
                # one out literally (itagroup.bg for "ITA Group Bulgaria" -- norm()
                # drops "group" and "bulgaria" survives alone, losing the match).
                import unicodedata as _ud
                raw = _ud.normalize("NFKD", name or "")
                raw = "".join(c for c in raw if not _ud.combining(c)).lower()
                raw_norm = re.sub(r"[^a-z0-9]", "", raw)
                substring_match = bool(dom_root and company_norm and
                                       (dom_root in company_norm or company_norm in dom_root
                                        or dom_root in abbrev_norm or abbrev_norm in dom_root
                                        or (raw_norm and (dom_root in raw_norm or raw_norm in dom_root))))
                # Acronym as a PREFIX of dom_root, not just the whole root: domains
                # routinely append a suffix to an acronym (cfrincorporated.com = CFR +
                # "incorporated" spelled out; t2gr.com has no suffix at all, already
                # covered by the equality branch). Requiring it anchor at position 0
                # (not merely `in`) keeps this from matching an acronym that is coincidentally
                # buried in the middle of an unrelated domain string.
                acronym_match = bool(dom_root and acronym and len(words) >= 2 and
                                     (dom_root == acronym or dom_root.startswith(acronym)
                                      or acronym.startswith(dom_root)))
                # First word as the head of the domain: "PGI Engineering" <-> pgigrup.com
                first = words_all[0] if words_all else ""
                head_match = bool(dom_root and len(first) >= 3 and dom_root.startswith(first))
                return substring_match or acronym_match or head_match

            # norm() has no transliteration for non-Latin scripts (Cyrillic, CJK,
            # Arabic, ...) -- a company name in one of those normalizes to "" and
            # would otherwise "fail to match" against every domain, which is not the
            # same claim as "these are different companies". Skip the check entirely
            # rather than report a verdict this logic has no actual ability to make.
            any_checkable = any(norm(c) for c in candidates if c)

            name_vs_domain_mismatch = (dom_root and len(dom_root) > 2 and any_checkable
                                       and not any(_candidate_match(c) for c in candidates if c))

            # A domain string that shares no text with the company name is NOT proof
            # of a wrong company -- confirmed 2026-10-16 against 104 real Campaign 3
            # DOMAIN fails: lhcommerce.com really is "Capline" (a product-line brand
            # name, nothing to do with the registered domain), gtallp.com really is
            # "Goldenhill", 3ds.com really is "Dassault Systemes" -- every one of these
            # was confirmed against the company's OWN live homepage <title>, which
            # names the profiled company outright even though the domain string does
            # not. Boutique firms, rebrands, and product-line brands routinely run on
            # a domain with no textual relationship to the public-facing name; that is
            # normal, not a data error. enrich.py already fetches and caches this exact
            # homepage text into company_intel once per domain (stage_enrich), so this
            # is a free second opinion, not a new network call: if the profiled name's
            # own significant words show up in that cached homepage text, the domain
            # mismatch is explained and should NOT fail the brief. Only surface the
            # FAIL when the mismatch is uncorroborated by the one source that would
            # settle it -- the company's own site talking about itself.
            if name_vs_domain_mismatch:
                homepage_confirms = False
                try:
                    import db as _db

                    _conn = _db.get_conn()
                    _cur = _conn.cursor()
                    _cur.execute(
                        "SELECT value_prop, source FROM company_intel WHERE domain=?", (dom,))
                    _row = _cur.fetchone()
                    _homepage = (_row["value_prop"] if _row and _row["source"] == "homepage" else "") or ""
                    if _homepage:
                        _homepage_norm = re.sub(r"[^a-z0-9]", "", _homepage.lower())
                        _name_words = [w for w in norm(profile.company or "").split()
                                      if len(w) > 2]
                        if _name_words and all(w in _homepage_norm for w in _name_words):
                            homepage_confirms = True
                except Exception:  # noqa: BLE001 -- best-effort corroboration only;
                    pass          # a DB/lookup problem must not block on this check

                if not homepage_confirms:
                    fails.append(
                        f"DOMAIN ERROR: brief is profiled as '{profile.company}' but the "
                        f"recipient's email domain is '{dom}' -- these do not match by "
                        f"name or acronym, and the company's own cached homepage text "
                        f"does not confirm the name either. This brief was very likely "
                        f"built for the wrong company.")

    # ---- RIVAL_FIRM (independent of profile) --------------------------------
    # CMI must never cite a competitor's own market-research report or a PR wire as if
    # it were CMI's own tracked intelligence -- confirmed as a real, frequent defect by
    # manual review of the 2026-10-08 regen batch (276 of 1,156 briefs cited
    # MarketsandMarkets/Grand View/openPR/etc. as a "CMI tracked" signal). config.py
    # already carries the curated RESEARCH_FIRM_SOURCES blocklist for this reason
    # (wired into score.eligible() at generation time so new briefs should not have the
    # source article in their pool at all) -- this is the deterministic backstop that
    # catches it in the rendered output regardless of how it got there (a stale
    # company_intel cache entry, a direct quote inside a "Source:" link, etc).
    low_html = html.lower()
    for firm in _config.RESEARCH_FIRM_SOURCES:
        if f"source: {firm}" in low_html or f">{firm}<" in low_html:
            fails.append(
                f"RIVAL FIRM ERROR: brief cites '{firm}' (a rival market-research firm "
                f"or PR wire) as a tracked signal/source. CMI must never present a "
                f"competitor's own report as its own research.")
            break

    # ---- GREETING (independent of profile) ----------------------------------
    # The salutation name should be a real parsed first name, not a raw email/username
    # artifact. Confirmed widespread in the 2026-10-08 regen batch (145 of 1,156
    # briefs): "Hello, scott.altmann", "Hello, ANJALI", "Hello, j.", "Hello, zeguo.zhang"
    # -- all symptoms of the lead-name parser falling back to the email local-part
    # instead of a real first name. Deterministic: a real first name is a single
    # alphabetic word with no '.', '@', or '_', and is not ALL-CAPS/all-lowercase-only
    # when it is more than 2 characters (a 1-2 letter name/initial is ambiguous either
    # way and is not flagged to avoid false positives on real short names like "Jo").
    greeting_m = re.search(r"Hello,\s*([^\n<]{1,40}?)\.?\s*(?:\n|$)", text)
    if greeting_m:
        greeting_name = greeting_m.group(1).strip().rstrip(".,")
        if "@" in greeting_name or "." in greeting_name or "_" in greeting_name:
            fails.append(
                f"GREETING ERROR: salutation uses a raw email/username artifact "
                f"('Hello, {greeting_name}') instead of a parsed first name.")
        elif len(greeting_name) > 2 and greeting_name.isalpha() and (
                greeting_name.isupper() or greeting_name.islower()):
            warns.append(
                f"Salutation name '{greeting_name}' is all-caps or all-lowercase -- "
                f"likely an unparsed raw field rather than a properly cased first name.")

    # ---- EMPTY_SECTION (independent of profile) -----------------------------
    # A "prospect" section can render with a real headline (compose.py forces
    # section 1's title to market_topic on report briefs, independent of body
    # content) while its body paragraph and stat-grid cells are completely empty --
    # confirmed in a manual audit of the 2026-10-08 regen batch (160 of 922 briefs)
    # as a previously-undetected bug: brief_builder.synthesize_news() returns []
    # when its industry-signal pool is empty, so compose.build_context's
    # `news = (brief.get("industry_outlook") or [{}])[0] or {}` silently falls
    # back to an empty dict -- body_html and cells both come back empty, but the
    # section still renders with its flag/title intact, so it reads as a genuine
    # but blank market card rather than a missing one. Purely structural: a
    # prospect section's own <h2> is non-empty but its <p> is not.
    for empty_m in re.finditer(
            r'<div class="prospect">\s*<span class="flag">([^<]*)</span>\s*'
            r'<h2>([^<]*)</h2>\s*<p>\s*</p>', html):
        flag, section_title = empty_m.group(1).strip(), empty_m.group(2).strip()
        fails.append(
            f"EMPTY SECTION ERROR: the '{section_title}' prospect section "
            f"('{flag}') has a headline but no body text and no stat cells -- "
            f"the underlying signal pool was empty when this brief was built.")

    # ---- LEDE_AUTHORSHIP (independent of profile) ---------------------------
    # The lede must never name the client as the one doing the tracking ("Masdar
    # tracks...", "thyssenkrupp tracked..."). compose.py's fixed opening (originally
    # "CMI tracked N signals this week.", now "This week's brief covers N signals:"
    # as of the 2026-10-21 cmiinsights.info redesign) keeps CMI/the newsletter itself
    # as the implicit author rather than leaving that framing to the model, but this
    # check is the backstop for every brief generated before either fix, and for any
    # other code path that builds a lede independently. Confirmed as a real, live
    # defect 2026-10-12: a sent Masdar brief read "Masdar tracks three signals this
    # week" -- the reader was made the author of their own market intelligence.
    lede_m = re.search(r"Hello,\s*[^\n<]{1,40}?\.?\s*(?:\n|$)\s*([^\n<]{10,400})", text)
    if lede_m:
        lede_text = lede_m.group(1).strip()
        company_bits = [b for b in re.split(r"[\s,]+", profile.company or "") if len(b) > 2]
        short_bit = getattr(profile, "short", "") or ""
        for name in ([short_bit] + company_bits):
            if not name:
                continue
            # Checked both anchored at the very start of the lede (the original,
            # confirmed Masdar incident) AND anywhere later in the lede -- confirmed
            # live 2026-10-13: "CMI tracked 3 signals this week. Peak Strategic
            # tracks 3 signals this week." restates the client as author in a
            # SECOND sentence right after the correct CMI opener, which the
            # start-anchored pattern alone does not see.
            if (re.match(rf"^{re.escape(name)}\s+track", lede_text, re.I)
                    or re.search(rf"[.!?]\s*{re.escape(name)}\s+tracks?\b", lede_text, re.I)):
                fails.append(
                    f"LEDE AUTHORSHIP ERROR: the brief opens with '{name} track...', "
                    f"naming the client as the one doing the tracking. CMI is the "
                    f"research firm; the opening must read 'CMI tracked...'.")
                break
        else:
            # profile.company/short do not always hold the human-readable name the
            # brief's own text uses -- confirmed 2026-10-13: a brief profiled as the
            # raw domain 'peakstrategic.net' still wrote "Peak Strategic tracks 3
            # signals this week.", which cannot match against either stored field.
            # This phrase template ("<Capitalized Name> track(s) N signals this
            # week") only ever appears as an echo of the hardcoded opener, so catch
            # it independent of what the profile says the client is called.
            if re.search(r"[.!?]\s*[A-Z][\w&.'-]*(?:\s+[A-Z][\w&.'-]*){0,3}\s+"
                        r"tracks?\s+\d+\s+signals?\s+this week\.", lede_text):
                fails.append(
                    "LEDE AUTHORSHIP ERROR: the brief restates a named entity as "
                    "the one doing the tracking ('<Name> track(s) N signals this "
                    "week') instead of CMI.")

    # ---- LEDE_DUPLICATION (independent of profile) --------------------------
    # The fixed opener is meant to appear exactly once per brief. Before the
    # 2026-10-21 cmiinsights.info redesign this was "CMI tracked N signals this
    # week."; compose.editorial() now opens with "This week's brief covers N
    # signals:" instead, but the same risk applies -- the model is asked to write
    # one sentence starting with that fixed phrase, not to repeat it, and that is an
    # instruction, not a guarantee (confirmed live 2026-10-13 via manual sample
    # review under the old opener: 9 of 30 briefs had the sentence twice, because
    # the model's own continuation independently opened with the same phrase it was
    # trained on seeing). Check both the current and legacy opener so this stays a
    # backstop for briefs built under either design.
    dup_count = (len(re.findall(r"This week.?s brief covers \w+ signals?:", text, re.I))
                 + len(re.findall(r"CMI tracked \d+ signals? this week\.", text, re.I)))
    if dup_count >= 2:
        fails.append(
            f"LEDE DUPLICATION ERROR: the brief's fixed opening sentence appears "
            f"{dup_count} times -- the model echoed the opener back into its own "
            f"continuation.")

    # ---- LEDE_NAME_LEAK (independent of profile) -----------------------------
    # 2026-10-29: confirmed by the user that naming the specific entity in the
    # lede (e.g. "a potential customer Candid Health raises $120 million...",
    # "Apple is expanding its retail...") is WRONG -- it hands the reader the
    # whole headline before they have any reason to keep reading. The lede may
    # add generic color (geography/sector/theme) but must never use the entity's
    # own proper name; that belongs only in the numbered card's own <h3> title
    # below. Pull each card's real title out of its own <h3> (the template
    # renders `{{ sec.title }}` there for both report and signal briefs) and
    # check it does not also appear in the lede paragraph.
    lede_html_m = re.search(r'<p class="lede">(.*?)</p>', html, re.S)
    if lede_html_m:
        lede_only_text = re.sub(r"<[^>]+>", " ", lede_html_m.group(1))
        lede_only_text = re.sub(r"\s{2,}", " ", lede_only_text).strip()
        for title_m in re.finditer(r"<h3>([^<]{2,80})</h3>", html):
            card_title = title_m.group(1).strip()
            if not card_title or card_title == "—":
                continue
            # Skip the fixed "Inside CMI" marketing h3 (not a signal card title).
            if "50+ countries" in card_title or "revenue-driven opportunities" in card_title:
                continue
            if re.search(re.escape(card_title), lede_only_text, re.I):
                fails.append(
                    f"LEDE NAME LEAK ERROR: the lede names the specific entity "
                    f"'{card_title}' -- the lede must only use the generic label "
                    f"(e.g. 'a potential customer'), never the entity's own name. "
                    f"Specifics belong in the numbered card only.")

    # ---- MAIN-CONTENT QUALITY (independent of profile) ----------------------
    # 2026-11: the reviews of the 12 Nov 17 briefs and the analyst's Waters notes
    # named the same defects repeatedly -- a lede carrying the writer prompt's own
    # instructions ("each a short bolded label phrase..."), inferred commercial
    # effects the signal never stated ("stricter limits will raise testing demand"),
    # voice.py's banned jargon, sentences duplicated between a card and its bullets,
    # and the seller's name lower-cased into a product word ("For waters,").
    #
    # All five read the MODEL-WRITTEN text only (section cards, lede, analyst
    # bullets, the numbered top3 rows) and never the fixed template blocks --
    # testimonials, the "Inside Coherent Market Insights" caps/projects showcase,
    # CTAs, creds and the footer are identical in every brief, so a jargon hit there
    # is not a defect anyone can fix. See main_content_html().
    if lede_html_m:
        lede_visible = visible_text(lede_html_m.group(1)).strip()
        fails.extend(lede_leaks(lede_visible, lede_html_m.group(1)))

    main_text = main_content_text(html)
    warns.extend(overclaim_warnings(main_text))
    warns.extend(meta_commentary_warnings(main_text))
    # Mask the lead's own proper nouns first: a product/brand that doubles as a banned
    # generic word ("Empower 3 software") must not read as jargon; the lowercase
    # generic use ("we empower teams") is still flagged.
    warns.extend(banned_word_warnings(mask_entities(main_text, profile)))
    warns.extend(duplicate_sentence_warnings(main_text))
    warns.extend(company_case_warnings(main_text, getattr(profile, "company", "")))

    # ---- GEOGRAPHY (optional; only when the lead's country is known) ---------
    # Country-first rule, checked on the RENDERED brief: every section is about
    # the lead's own country, or explicitly names it alongside other countries
    # (the explicit-link rule). Detection is the same word-boundary country
    # matcher generation uses (regions.countries_in_text), run per section.
    #   WARN  OFF-COUNTRY      one section names other countries but never the
    #                          lead's own
    #   FAIL  MIXED GEOGRAPHY  two sections each name only DIFFERENT non-lead
    #                          countries -- the brief is not about one place
    # `lead_country=None` (the default) skips the check entirely, so every
    # existing caller and the checks above are unchanged.
    if lead_country:
        try:
            import regions as _regions

            _prof = _regions.get_profile(lead_country)
        except Exception:  # noqa: BLE001 -- a broken geo check must not fail the brief
            _prof = None
        if _prof and _prof.get("iso2"):
            _lead_iso = _prof["iso2"]
            _lead_name = _prof["name"]
            # Boilerplate first: salutation ("Hello, Sanaa."), "Prepared for ...",
            # "Market Signal Brief · ..." and the recipient/company names carry
            # place-like words that are not evidence about the news.
            try:
                _geo_text = geo_scrub(text, profile)
            except Exception:  # noqa: BLE001 -- a broken scrub must not fail the brief
                _geo_text = text
            # Section labels mark each section's end, so a section's body cannot
            # bleed into the next one (the plain 1500-char window above the
            # STRUCTURE check would count the NEXT section's geography too).
            _all_labels = {a for s in expected_sections
                           for a in SECTION_ALIASES.get(s, (s,))}
            _label_starts = sorted(m.start() for alias in _all_labels
                                   for m in re.finditer(
                                       rf"\b{re.escape(alias)}\b", _geo_text))
            _sec_geo: dict[str, set[str]] = {}
            for s in expected_sections:
                # Split sections the same way the STRUCTURE check above does.
                _start = _end = -1
                for alias in SECTION_ALIASES.get(s, (s,)):
                    m = re.search(rf"{re.escape(alias)}\s*·(.{{0,1500}})", _geo_text, re.S)
                    if m:
                        _start, _end = m.start(), m.end()
                        break
                if _start < 0:
                    for alias in SECTION_ALIASES.get(s, (s,)):
                        m2 = re.search(rf"\b{re.escape(alias)}\b", _geo_text)
                        if m2:
                            _start, _end = m2.end(), m2.end() + 1500
                            break
                if _start < 0:
                    continue
                _next = [p for p in _label_starts if p > _start + 2]
                if _next:
                    _end = min(_end, _next[0])
                _cset = _regions.countries_in_text(_geo_text[_start:_end])
                if _cset:
                    _sec_geo[s] = _cset
            _off: list[tuple[str, set[str]]] = []
            for _label, _cset in _sec_geo.items():
                _others = _cset - {_lead_iso}
                if _others and _lead_iso not in _cset:
                    _names = ", ".join(_regions.name_of(i) or i
                                       for i in sorted(_others))
                    warns.append(f"OFF-COUNTRY: section '{_label}' mentions "
                                 f"{_names} but not {_lead_name}.")
                    _off.append((_label, _others))
            for _i in range(len(_off)):
                for _j in range(_i + 1, len(_off)):
                    if not (_off[_i][1] & _off[_j][1]):
                        fails.append(
                            f"MIXED GEOGRAPHY: sections '{_off[_i][0]}' and "
                            f"'{_off[_j][0]}' each name only different non-lead "
                            f"countries ({', '.join(sorted(_off[_i][1]))} vs "
                            f"{', '.join(sorted(_off[_j][1]))}) -- the brief is "
                            f"not about one place.")
                        break
                else:
                    continue
                break

    # Rules above are keyword checks. They cannot tell that Reckitt is VVF's CUSTOMER
    # rather than its rival -- that needs judgement, so ask the model that wrote the
    # brief. Off unless CMI_LLM_VALIDATE=1, and it only ever ADDS findings: a judge
    # that is down or unsure leaves the deterministic verdict untouched.
    #
    # Only worth asking when the text rules did not already reject the brief -- a FAIL
    # is already held back from sending, so judging it teaches nothing new and costs a
    # GPU call for no benefit. See check_detailed's docstring.
    llm_fails: list[str] = []
    llm_warns: list[str] = []
    if not fails:
        try:
            import llm_validate

            llm_fails, llm_warns = llm_validate.judge_brief(
                text, profile, market=getattr(profile, "market", ""),
                email=email, kind=kind, expected_subject=topic)
        except Exception:  # noqa: BLE001 -- the optional judge must never break validation
            pass

    return fails, warns, llm_fails, llm_warns


def check_html(html: str, profile, *, email: str = "", kind: str = "",
              topic: str = "", lead_country: str | None = None
              ) -> tuple[list[str], list[str]]:
    """Same as `check()`, but takes HTML text directly instead of a file path --
    for gating a send before the brief is written to disk."""
    import os
    import tempfile

    fd, tmp_path = tempfile.mkstemp(suffix=".html")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(html)
        return check(tmp_path, profile, email=email, kind=kind, topic=topic,
                     lead_country=lead_country)
    finally:
        os.unlink(tmp_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a generated brief's structure, roles and fit.")
    ap.add_argument("paths", nargs="+", help="brief HTML file(s); globs allowed")
    ap.add_argument("--company", required=True)
    ap.add_argument("--website", default="")
    ap.add_argument("--market", default="India")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from tenant import Profile

    profile = Profile.from_company(args.company, args.website, "there", args.market)
    print(f"seller:      {profile.company}")
    print(f"value_prop:  {profile.value_prop[:100]}")
    print(f"competitors: {', '.join(profile.competitors[:6])}\n")

    files = [p for pat in args.paths for p in (glob.glob(pat) or [pat])]
    worst = 0
    for path in files:
        if not os.path.isfile(path):
            print(f"{path}: not found")
            worst = 1
            continue
        fails, warns = check(path, profile)
        status = "FAIL" if fails else ("WARN" if warns else "PASS")
        print(f"[{status}] {os.path.basename(path)}")
        for f in fails:
            print(f"    x {f}")
        for w in warns:
            print(f"    ! {w}")
        if fails:
            worst = 1
    raise SystemExit(worst)


if __name__ == "__main__":
    main()
