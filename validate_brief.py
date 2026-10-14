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

SECTIONS = ("Potential Customer", "Industry Outlook", "Competition Outlook")

# A report brief replaces section 1 with "Your Market" built around the topic the lead
# enquired about. Demanding "Potential Customer" there failed every report brief for a
# section it is not supposed to have.
SECTION_ALIASES = {
    "Potential Customer": ("Potential Customer", "Your Market"),
    # Report briefs put Market Moves where a signal brief has Industry Outlook.
    "Industry Outlook": ("Industry Outlook", "Market Moves"),
}


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


def section_company(text: str, label: str) -> str:
    """The company name rendered under a section's flag line.

    The template puts the flag ('● Potential Customer · <industry>') immediately above
    the company heading, so the first substantial line after it is the subject.
    """
    m = re.search(rf"●?\s*{re.escape(label)}\s*·[^\n]*\n(.*)", text)
    if not m:
        return ""
    for line in m.group(1).split("\n")[:4]:
        line = line.strip()
        if 2 < len(line) < 70 and not line.lower().startswith(("why ", "who ", "their ")):
            return line
    return ""


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
         topic: str = "") -> tuple[list[str], list[str]]:
    fails, warns, _llm_fails, _llm_warns = check_detailed(
        path, profile, email=email, kind=kind, topic=topic)
    return fails + _llm_fails, warns + _llm_warns


def check_detailed(path: str, profile, *, email: str = "", kind: str = "",
                   topic: str = "") -> tuple[list[str], list[str], list[str], list[str]]:
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
    missing: list[str] = []
    present = {}
    for s in SECTIONS:
        body = ""
        for alias in SECTION_ALIASES.get(s, (s,)):
            m = re.search(rf"{re.escape(alias)}\s*·(.{{0,1500}})", text, re.S)
            if m:
                body = m.group(1)
                break
        if not body and s == "Industry Outlook" and re.search(r"\bIndustry Outlook\b", text):
            # Industry Outlook has no company flag line -- it is a themed section.
            i = text.find("Industry Outlook")
            body = text[i:i + 1500]
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
            n_sections_expected = sum(1 for s in SECTIONS if present.get(s))
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
    cust = section_company(text, "Potential Customer")
    if not cust and re.search(r"Your Market\s*·", text):
        cust = ""   # report brief: section 1 is a market, so the buyer role checks below
                    # have nothing to judge and must not fire
    comp = section_company(text, "Competition Outlook")

    for rival in (profile.competitors or []):
        if cust and name_overlap(cust, rival):
            fails.append(f"ROLE ERROR: Potential Customer is '{cust}', a known competitor "
                         f"(matches '{rival}'). A rival must appear in Competition Outlook.")
            break

    # The competitor list only names the rivals someone thought to write down. A supplier
    # in the seller's OWN category is just as wrong in the customer slot even when it is
    # absent from that list -- Dr Reddy's launching an immuno-oncology biosimilar passed
    # every name check while being plainly not a buyer of CAR-T. Catch it by category:
    # the customer should read like a BUYER vertical, not like the seller's own trade.
    if cust:
        seller_trade = {w for w in re.findall(r"[a-z][a-z0-9-]{4,}",
                                              (profile.value_prop or "").lower())} - {
            "india", "indian", "supplies", "approved", "first", "targeted", "relapsed",
            "refractory", "patients", "treating"}
        cust_ctx = ""
        m = re.search(rf"Potential Customer\s*·(.{{0,900}})", text, re.S)
        if m:
            cust_ctx = m.group(1).lower()
        buyerish = any(w in cust_ctx for w in
                       ("hospital", "clinic", "centre", "center", "network", "institute",
                        "provider", "insurer", "payer", "medical college"))
        sellerish = sum(1 for w in seller_trade if len(w) > 5 and w in cust_ctx)
        if cust_ctx and not buyerish and sellerish >= 2:
            warns.append(f"Potential Customer '{cust}' reads like a company in the seller's own "
                         f"category, not a buyer. Check it is not a rival or supplier.")
        # A regulator, ministry or trade body is a real signal but has no purchase order
        # to give. Left unchecked the model invents a reason it might buy.
        non_buyers = ("regulator", "ministry", "cdsco", "dcgi", "government", "court",
                      "tribunal", "association", "council", "committee", "commission",
                      "department of", "task force")
        if any(w in cust.lower() for w in non_buyers):
            fails.append(f"ROLE ERROR: Potential Customer is '{cust}', which regulates or "
                         f"governs the market rather than buying in it.")

    if comp:
        if not any(name_overlap(comp, r) for r in (profile.competitors or [])):
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
        if topic_terms and not any(w in low for w in topic_terms):
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
                return substring_match or acronym_match

            # norm() has no transliteration for non-Latin scripts (Cyrillic, CJK,
            # Arabic, ...) -- a company name in one of those normalizes to "" and
            # would otherwise "fail to match" against every domain, which is not the
            # same claim as "these are different companies". Skip the check entirely
            # rather than report a verdict this logic has no actual ability to make.
            any_checkable = any(norm(c) for c in candidates if c)

            if (dom_root and len(dom_root) > 2 and any_checkable
                    and not any(_candidate_match(c) for c in candidates if c)):
                fails.append(
                    f"DOMAIN ERROR: brief is profiled as '{profile.company}' but the "
                    f"recipient's email domain is '{dom}' -- these do not match by name "
                    f"or acronym. This brief was very likely built for the wrong company.")

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
    import config as _config

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
    # The lede must open with CMI as the one doing the tracking ("CMI tracked N
    # signals..."), never the client ("Masdar tracks...", "thyssenkrupp tracked...").
    # compose.py now hardcodes this opening rather than leaving it to the model, but
    # this check is the backstop for every brief generated before that fix, and for
    # any other code path that builds a lede independently. Confirmed as a real,
    # live defect 2026-10-12: a sent Masdar brief read "Masdar tracks three signals
    # this week" -- the reader was made the author of their own market intelligence.
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
    # The hardcoded opener ("CMI tracked N signals this week.") is meant to appear
    # exactly once. compose.editorial() asks the model to continue after it without
    # repeating it, but that is an instruction, not a guarantee -- confirmed live
    # 2026-10-13 via manual sample review: 9 of 30 briefs had the sentence twice
    # (model's own lede independently opened with the same phrase it was trained on
    # seeing). compose.py now strips a leading/trailing echo before concatenating,
    # but this check is the backstop for every brief built before that fix and for
    # any other path that assembles a lede independently.
    dup_count = len(re.findall(r"CMI tracked \d+ signals? this week\.", text, re.I))
    if dup_count >= 2:
        fails.append(
            f"LEDE DUPLICATION ERROR: the opening sentence 'CMI tracked N signals "
            f"this week.' appears {dup_count} times -- the model echoed the "
            f"hardcoded opener back into its own continuation.")

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
              topic: str = "") -> tuple[list[str], list[str]]:
    """Same as `check()`, but takes HTML text directly instead of a file path --
    for gating a send before the brief is written to disk."""
    import os
    import tempfile

    fd, tmp_path = tempfile.mkstemp(suffix=".html")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(html)
        return check(tmp_path, profile, email=email, kind=kind, topic=topic)
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
