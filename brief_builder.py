"""Selection + LLM synthesis + assembly into the `brief` dict the templates expect.

Newsletter structure (after the salutation):
  Top-3 signals — one per section, categorized, in the same order as the sections:
     1) Potential Customer   2) Industry Outlook   3) Competition Outlook
  Then three parallel OUTLOOK blocks, EACH followed by its own Coherent Take:
     1) POTENTIAL CUSTOMER  — a company captured on a trigger signal   → Coherent Take
     2) INDUSTRY OUTLOOK    — the industry trend we track              → Coherent Take
     3) COMPETITION OUTLOOK — a competitor's move + the edge to take   → Coherent Take
  Then: Inside CoherentConnect — a static capability showcase baked into the template.

Every LLM step degrades to source-derived text so the newsletter always renders. All
tenant-specific facts come from a `tenant.Profile`, so this works for ANY seller.
"""
from __future__ import annotations
import re
import concurrent.futures as cf
from dataclasses import replace
from datetime import datetime, timezone

import config
import llm
import voice
import cluster
import regions
import tenant
import free_research
from collect import Article

_INLINE_OPEN = re.compile(r"^<(b|i|em|strong|span|a)\b", re.I)
_INLINE_CLOSE = re.compile(r"</(b|i|em|strong|span|a)>$", re.I)

# Every "Book a demo" link in the brief (3 section CTAs + the footer button) resolves
# from this one value via footer.cta_url. No ?month= param — Calendly opens on the
# current month by default, and a pinned month goes stale the moment it passes.
DEMO_URL = "https://calendly.com/aditya-jha-coherentmarketinsights/30min"
SITE_URL = "https://coherentconnect.ai/"


# The model sometimes answers a "name the company" prompt with the literal word
# "None" (or a close variant) instead of leaving the field empty, when the signal
# genuinely names no company. A bare truthiness check treats that string as a real
# answer -- `data.get("company_name") or fallback` never falls back, because "None"
# is non-empty -- so it rendered verbatim as a section headline ("None in the
# signal", 2026-10-08). Anything matching this is treated as no answer at all.
_NON_ANSWERS = {"none", "n/a", "na", "unknown", "unnamed", "not specified",
                "not applicable", "no company", "null", "nil", "-", "—"}


_SIG_LEADIN = re.compile(r"^((?:<[^>]+>)*)\s*(?:the\s+signal\s+states(?:\s+that)?|per\s+the\s+signal\s*,?)\s+",
                         re.I)
_SIG_ONLY = re.compile(
    r"^(?:<[^>]+>)*\s*the\s+(?:signal|free\s+research)\s+(?:does\s+not|doesn't|do\s+not|gives?|gave|"
    r"provides?|offers?|contains?|mentions?|shows?|names?|lacks?|has\s+no|is\s+silent|only|says?|"
    r"notes?|reports?|describes?|indicates?|suggests?|refers?|is|was)\b", re.I)


def _scrub_signal_refs(s: str) -> str:
    """Prompt-leak scrub: strip "The SIGNAL states (that)" / "Per the SIGNAL," lead-ins
    (capitalising the remainder) and drop sentences that are only about the signal or
    the free research themselves."""
    if not s or not re.search(r"signal|free research", s, re.I):
        return s
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", s):
        m = _SIG_LEADIN.match(sent)
        if m:
            rest = sent[m.end():]
            sent = m.group(1) + (rest[:1].upper() + rest[1:] if not m.group(1) else rest)
            if m.group(1):
                sent = re.sub(r"^((?:<[^>]+>)*)(\w)", lambda x: x.group(1) + x.group(2).upper(), sent)
        elif _SIG_ONLY.match(sent):
            continue
        out.append(sent)
    return " ".join(out)


def _clean_frag(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = _scrub_signal_refs(s.strip()).strip()
    if s.strip(".").strip().lower() in _NON_ANSWERS:
        return ""
    # Block tags the template supplies itself. The model sometimes wraps its answer in
    # <p>...</p>; stripping only the leading '<' left a literal "p>" rendering as text
    # at the top of the card, so drop the whole tag.
    s = re.sub(r"</?(p|div|span|br|ul|ol|li|h[1-6])\b[^>]*>", " ", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if s.startswith("<") and s.endswith(">") and not _INLINE_OPEN.match(s):
        s = s[1:-1].strip()
    if s.startswith("<") and not _INLINE_OPEN.match(s):
        s = s[1:].lstrip()
    if s.endswith(">") and not _INLINE_CLOSE.search(s):
        s = s[:-1].rstrip()
    return s.strip()


def _evq(a: Article) -> float:
    """Evidence quality used to order shortlists: relevancy x recency."""
    return (getattr(a, "ev_relevancy", 0.0) or 0.0) * (getattr(a, "ev_recency", 0.0) or 0.0)


def _above_floor(arts: list[Article]) -> list[Article]:
    """Drop items under config.MIN_EV_RELEVANCY; if that leaves nothing keep them all."""
    floor = getattr(config, "MIN_EV_RELEVANCY", 0.15)
    kept = [a for a in arts if (getattr(a, "ev_relevancy", 0.0) or 0.0) >= floor]
    return kept or list(arts)


_SCOPE_RANK = {"region": 1, "global": 2}   # topic news geo tier, "" / "country" = 0


def _rank_by_evidence(arts: list[Article]) -> list[Article]:
    """Floor, then order by relevancy x recency (desc); composite breaks ties, so an
    unscored list keeps its current order."""
    return sorted(_above_floor(arts), key=lambda a: (_evq(a), a.composite), reverse=True)


def _date_label(dt: datetime) -> str:
    return f"{dt.strftime('%b')} {dt.day}"


_SCORE_NOTE = "\nFIT (0 to 1) is how well an item matches the seller. Prefer items with high FIT."


def _scores_line(a: Article) -> str:
    return f"FIT: {a.ev_relevancy:.2f} | AGE: {a.ev_age_days:.0f} days"


def _cands_block(arts: list[Article]) -> str:
    lines = []
    for i, a in enumerate(arts):
        # Topic news admitted from outside the lead's country: the writer must say where.
        loc = (" | OUTSIDE THE READER'S COUNTRY: say where it happened"
               if getattr(a, "geo_scope", "") in ("region", "global") else "")
        lines.append(f"[{i}] TITLE: {a.title}\n    SOURCE: {a.source} | DATE: {_date_label(a.published)}"
                     f" | {_scores_line(a)}{loc}\n"
                     f"    SUMMARY: {a.summary[:280]}")
    return "\n".join(lines)


def _fallback_plan(profile, kind: str) -> list[dict]:
    if kind == "competitor":
        return [
            {"title": "Name the gap.", "body_html": "Identify exactly where this competitor is thin and you are strong."},
            {"title": "Reframe the conversation.", "body_html": "Lead prospects to the criteria where you win, before price."},
            {"title": "Move first.", "body_html": "Reach the accounts this competitor is chasing while the moment is live."},
        ]
    return [
        {"title": "Open with fit, not introductions.",
         "body_html": f"Lead with {profile.short}'s coverage of this space as proof you understand it."},
        {"title": "Name the moment.", "body_html": "Reference the specific event in the subject line so it doesn't read as mass-mail."},
        {"title": "Offer something usable today.", "body_html": "Position a low-friction first deliverable they can use immediately."},
    ]


def _base_name(s: str) -> str:
    """Company name reduced to a comparable stem, so "Weir Group" and "The Weir Group
    plc" collapse. Used to stop one company holding two sections of the same brief."""
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    drop = {"the", "ltd", "limited", "inc", "corp", "corporation", "plc", "pvt",
            "private", "company", "co", "group", "holdings", "international", "global"}
    words = [w for w in s.split() if w and w not in drop]
    return " ".join(words[:2])


def _industry_of(a: Article, profile) -> str:
    return a.matched_industries[0] if a.matched_industries else (profile.industries[0] if profile.industries else "Technology")


_SHORT_TERM = 4   # terms this short (MS, LC, AI) are ambiguous on their own


def _product_terms(profile) -> list[str]:
    """Lower-cased product terms, curated first. Uses tenant.product_terms_of when it
    exists, else products' names plus the curated override."""
    import tenant as _tenant
    fn = getattr(_tenant, "product_terms_of", None)
    if fn:
        try:
            terms = fn(profile) or []
        except Exception:  # noqa: BLE001
            terms = []
    else:
        ov = _tenant._overrides_for(getattr(profile, "company", "")) or {}
        terms = list(ov.get("product_terms") or [])
        terms += [p.get("name", "") for p in (getattr(profile, "products", None) or [])
                  if isinstance(p, dict)]
    out, seen = [], set()
    for t in terms:
        t = (t or "").strip().lower()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _term_hits(terms: list[str], hay: str) -> int:
    """How many distinct terms appear as whole words/phrases in hay. A short acronym
    (MS, LC, AI) only counts when another distinct term matched too: 'Lucid-MS' is a
    multiple sclerosis drug, not mass spectrometry."""
    long_hits, short_hits = 0, 0
    for t in terms:
        if not re.search(r"(?<![a-z0-9])" + re.escape(t) + r"s?(?![a-z0-9])", hay):
            continue
        if len(re.sub(r"[^a-z0-9]", "", t)) <= _SHORT_TERM:
            short_hits += 1
        else:
            long_hits += 1
    return long_hits + (short_hits if (long_hits or short_hits >= 2) else 0)


_LEGAL_SUFFIX = {"inc", "ltd", "limited", "llc", "plc", "corp", "corporation", "co", "pvt",
                 "private", "the"}


def _self_norm(s: str) -> str:
    s = re.sub(r"['’]s\b", "", (s or "").lower())      # "OHS's" -> "ohs"
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(w for w in s.split() if w not in _LEGAL_SUFFIX)


def is_self(name: str, profile) -> bool:
    """True when `name` is the seller itself: its full name, short name, acronym or an
    abbreviated form, alone or inside a longer phrase ("OHS's", "Ottawa Humane Society's
    new clinic"). The seller must never be its own customer, peer or competitor."""
    n = _self_norm(name)
    if not n:
        return False
    full = _self_norm(getattr(profile, "company", ""))
    short = _self_norm(getattr(profile, "short", ""))
    forms = {f for f in (full, short) if len(f) >= 3}
    core = full.split()
    if len(core) >= 3:
        forms.add("".join(w[0] for w in core))              # acronym: "ohs"
    padded = f" {n} "
    if any(f" {f} " in padded for f in forms):
        return True
    # an abbreviated head of the full name ("Ottawa Humane") counts too
    return len(n.split()) >= 2 and full.startswith(n + " ")


_HEADLINE_VERB = re.compile(r"\b(signs?|raises?|opens?|launch(?:es)?|unveils?|acquires?|bags?|"
                            r"secures?|announces?|appoints?)\b", re.I)
_HEADLINE_MONEY = re.compile(r"[$₹€£]|\b(million|billion|crore|lakh)\b", re.I)


def _looks_like_headline(name: str) -> bool:
    """A truncated headline is not a company name."""
    n = (name or "").strip()
    return (len(n.split()) > 6 or n.endswith((",", ":"))
            or bool(_HEADLINE_MONEY.search(n)) or bool(_HEADLINE_VERB.search(n)))


def _heading(profile) -> str:
    import compose
    return compose.prospect_heading(profile)


def _foreign_event(a: Article, profile) -> bool:
    """True when the triage model placed the article's event in a KNOWN country other than
    the lead's. "unknown"/unrecognised/empty -> False (never skip on a guess)."""
    ec = (getattr(a, "event_country", "") or "").strip()
    if not ec or ec.lower() in ("unknown", "none", "n/a", "global", "worldwide"):
        return False
    ev = regions.get_profile(ec)
    lead = regions.get_profile(getattr(profile, "market", "") or "")
    return bool(ev and lead and ev["iso2"] != lead["iso2"])


def _in_lead_country(a: Article, profile) -> bool:
    """True when the article is tied to the lead's country. A Global/unknown market passes.
    "unknown" event country alone is not enough: a Singapore startup reached a US lead."""
    lead = regions.get_profile(getattr(profile, "market", "") or "")
    if not lead:
        return True
    if lead["iso2"] in regions.article_countries(a):
        return True
    ec = (getattr(a, "event_country", "") or "").strip()
    ev = regions.get_profile(ec) if ec else None
    return bool(ev and ev["iso2"] == lead["iso2"])


# The v4 model was fine-tuned on this exact triage prompt (training_data/v4 kind "triage_role",
# 304 examples; prompt = triage-lab variant E_balanced). Keep it word-for-word, like _CHECK_*.
# Measured 2026-11-26 on the 906 held-out labels in training_data/v4/test_triage.jsonl
# (session_scratch/triage_bench.py): this prompt 80.6% accuracy, buyer precision 0.49; adding
# event_country/would_buy fields 77.7% / 0.44; a short rewrite 71.1% / 0.36. Country and the
# product a buyer would buy come from the separate pick call instead (_pick_buyer).
def _triage_block(arts: list[Article]) -> str:
    """The SIGNALS block in the format the triage training data used."""
    lines = []
    for i, a in enumerate(arts):
        lines.append(f"[{i}] TITLE: {a.title}\n    SOURCE: {a.source} | DATE: {_date_label(a.published)}"
                     f" | TRIGGER: {a.trigger_type or 'none'} | QUERY: {a.query} | SCORE: {a.composite:.4f}\n"
                     f"    SUMMARY: {a.summary[:280]}")
    return "\n".join(lines)


def _triage_prompt(profile, block: str) -> tuple[str, str]:
    """(system, user) for triage_triggers; `block` is the SIGNALS text."""
    system = (
        "You are a B2B sales analyst. For each news item you decide which company the item is about and how "
        "that company relates to the SELLER.\n\n"
        f"SELLER: {profile.company} ({profile.short}). It sells: {profile.value_prop}\n"
        f"THE SELLER'S BUYER INDUSTRIES: {', '.join(profile.industries)}.\n\n"
        "STEP 1 - WHO IS THE ITEM ABOUT?\n"
        "Name the one company the item is mainly about. The publisher is never the subject (Inc42, Reuters, "
        "Economic Times, Business Standard are outlets, not companies in the story). If the item is a roundup, "
        "tracker, list, opinion piece, market report or event with no single company as its subject, there is no "
        "company: leave company_name empty and the relationship is 'other'.\n\n"
        "STEP 2 - LABEL THAT COMPANY (check in this order):\n"
        "1. 'peer' - the company SELLS the same kind of product or service as the seller, to the same kind of "
        "buyer. Any company on the Known competitors list is a peer. A peer is never a buyer, however large or "
        "newsworthy; its funding round or launch is competition.\n"
        "2. 'buyer' - the company's OWN business sits in one of the buyer industries AND the news shows a "
        "reason it would plausibly buy what the seller sells (expansion, a new plant or office, hiring, "
        "funding, a launch, a problem the product solves). Being large or famous is not enough: a company in an "
        "unrelated trade is not a buyer.\n"
        "3. 'other' - everything else: regulators, ministries and courts; industry bodies and associations; "
        "universities and research institutes; media, consultancies and analyst firms; investors; the seller's "
        "own suppliers; companies in unrelated trades; and items with no clear company.\n\n"
        "Decide on the evidence in the item. Do not default to 'other': if the company clearly sells what the "
        "seller sells, it is a 'peer'; if its own business is in a buyer industry and the news shows a need, it is "
        "a 'buyer'. Use 'other' only when neither is true."
    )
    user = (f"SELLER sells: {profile.value_prop}\nKnown competitors: {', '.join(profile.competitors) or '—'}\n\n"
            f"SIGNALS:\n{block}\n\n"
            "Return JSON {\"items\":[{\"index\":<int>,\"company_name\":\"...\",\"relationship\":\"buyer|peer|other\"}]}"
            " — one entry per signal index above.")
    return system, user


# ── Triage: classify each trigger company as BUYER vs PEER (competitor) vs OTHER ─
def triage_triggers(triggers: list[Article], profile):
    """One LLM call. Returns {index: (company_name, relationship)} over the first 10 candidates.
    'buyer' = would purchase the seller's product; 'peer' = sells a competing product
    (a rival); 'other' = neither. This is what keeps a rival's funding round OUT of the
    Potential Customer slot and INTO Competition Outlook."""
    cands = triggers[:10]
    if not cands:
        return {}, cands
    system, user = _triage_prompt(profile, _triage_block(cands))
    data = llm.chat_json(system, user, max_tokens=800, default={})
    rel: dict[int, tuple[str, str]] = {}
    for it in (data.get("items") or []):
        if not isinstance(it, dict):
            continue  # model occasionally returns a bare string instead of {"index":...}
        i = it.get("index")
        if isinstance(i, int) and 0 <= i < len(cands):
            r = (it.get("relationship") or "other").strip().lower()
            r = r if r in ("buyer", "peer", "other") else "other"
            name = _clean_frag(it.get("company_name") or "") or cands[i].title[:48]
            if is_self(name, profile):
                r = "self"      # the seller's own news is never a customer, peer or rival
            rel[i] = (name, r)
    return rel, cands


def _pick_buyer(buyers: list[tuple[Article, str]], profile) -> tuple[int, str, str]:
    """One LLM call: the model picks the best of the labelled buyers. Returns
    (index into `buyers` or -1 for none, event_country, would_buy)."""
    market = (getattr(profile, "market", "") or "").strip() or "the reader's country"
    system = (
        f"You are a B2B sales analyst for {profile.company} ({profile.short}). {profile.short} sells: "
        f"{profile.value_prop}\n"
        f"BUYER INDUSTRIES: {', '.join(profile.industries)}\n\n"
        f"Each CANDIDATE is a company in the news that may buy from {profile.short}. Pick the ONE most "
        f"likely to buy from {profile.short} soon, because of its news.\n"
        "A good pick:\n"
        "- has its own business in one of the buyer industries;\n"
        "- does the action itself (it is not being bought or sold);\n"
        f"- acts in {market};\n"
        f"- has news that creates a clear need for one {profile.short} product.\n"
        "If no candidate fits, answer -1."
    )
    lines = "\n".join(f"[{k}] {name} | {a.title} | {(a.summary or '')[:160]}"
                      for k, (a, name) in enumerate(buyers))
    user = (f"CANDIDATES:\n{lines}\n\n"
            "Return JSON {\"best\": <index or -1>, \"event_country\": \"\", \"would_buy\": \"\"}\n"
            "- event_country: the country where the picked company's news happens, in English, or \"unknown\".\n"
            f"- would_buy: the {profile.short} product it could buy and why, in one short phrase.")
    data = llm.chat_json(system, user, max_tokens=200, default={})
    k = data.get("best") if isinstance(data, dict) else None
    if k == -1:
        return -1, "", ""
    if not isinstance(k, int) or not 0 <= k < len(buyers):
        k = 0     # a failed or unparseable reply falls back to the top-ranked buyer
    ec, wb = data.get("event_country"), data.get("would_buy")
    return k, (ec.strip() if isinstance(ec, str) else ""), (wb.strip() if isinstance(wb, str) else "")


# ── 1. POTENTIAL CUSTOMER — best BUYER (never a peer/rival), free-enrich, write ─
# Buying-event verbs in the local-language editions the pipeline searches. score's trigger
# lexicon is English only, so "MGTA investit dans le broyage des déblais" had no trigger_type.
_LOCAL_EVENT_RE = re.compile(
    r"\b(investit|investissement|lance|ouvre|inaugure|rach[eè]te|acquiert|s'allie|signe|remporte|"
    r"invierte|inversi[oó]n|lanza|abre|adquiere|inaugura|firma|"
    r"investiert|er[oö]ffnet|[uü]bernimmt|startet|baut|"
    r"investeert|opent|neemt over|lanceert|"
    r"investe|lancia|apre|acquisisce|"
    r"yat[ıi]r[ıi]m|fabrika|a[cç][ıi]l[ıi]ş)\b", re.I)


_APPOINT_RE = re.compile(
    r"\b(appoints?|appointed|appointment|names? (?:new )?(?:\w+ ){0,3}(?:ceo|cfo|cto|coo|chief|president|"
    r"head|director|chair\w*|md)|hires?|hired|joins as|steps down|resigns|promoted|new (?:ceo|cfo|chief)|"
    r"nomme|nommé|nombra|nombramiento|ernennt|ernannt|benoemd|benoemt|nomina|nominato|atadı|atandı)\b",
    re.I)


def _is_appointment(a: Article) -> bool:
    """A staff appointment headline: a person's move, not a market development."""
    return bool(_APPOINT_RE.search(a.title or ""))


def _has_event(a: Article) -> bool:
    return bool(a.trigger_type) or bool(_LOCAL_EVENT_RE.search(a.title or ""))


_TRIAGE_BATCH = 10   # triage_triggers judges this many per call (its training size)
_TRIAGE_POOL = 40    # the scorer's top items (buyer + competitor searches) the model labels


def build_customer(triggers: list[Article], profile, log=print, competitor_arts=None):
    gov_ok = config.government_buyers_ok(profile)
    if not gov_ok:
        kept = [a for a in triggers if not config.is_government(a.title)]
        if len(kept) < len(triggers):
            log(f"    · dropped {len(triggers) - len(kept)} government items (not a buyer for this lead)")
        triggers = kept

    def _looks_like_seller_category(name: str, art: Article) -> bool:
        """True if this candidate is in the seller's OWN trade rather than a buyer.

        The triage LLM answers this inconsistently -- the same Dr Reddy's biosimilar
        launch came back 'other' on one run and 'buyer' on the next, which put a drug
        maker in the Potential Customer slot for a CAR-T seller. A deterministic guard
        on top of the model's judgement costs nothing and cannot drift between runs.
        """
        hay = f"{name} {art.title} {art.summary[:200]}".lower()
        # A company already named as a rival is never a customer, whatever triage said.
        for c0 in (profile.competitors or []):
            for c in tenant.rival_forms(profile, c0):
                cb = re.sub(r"[^a-z0-9 ]", " ", (c or "").lower()).strip()
                head = " ".join(cb.split()[:2])
                if len(head) > 5 and head in hay:
                    return True
        # Nor is a body that regulates or reports on the market rather than buying in
        # it. "India's drug regulator has formed an AI expert committee" landed in the
        # Potential Customer slot: a real signal, but a regulator has no purchase order
        # to give, and the model dutifully invented a reason it might buy.
        non_buyers = getattr(config, "NON_BUYER_TERMS", (
            "regulator", "regulatory authority", "ministry", "government", "city of",
            "municipal", "city council",
            "cdsco", "dcgi", "fda", "court", "tribunal", "parliament",
            "association", "council", "committee", "commission", "board of",
            "department of", "task force", "research institute", "journal"))
        if gov_ok:      # this lead sells to government: a municipality is a real buyer
            non_buyers = [w for w in non_buyers if not config.is_government(w)]
        if any(w in (name or "").lower() for w in non_buyers):
            return True
        # Nor is a company whose news is about selling the seller's own product type.
        seller_acts = ("launches", "launch of", "unveils", "rolls out", "approval for",
                       "gets approval", "receives approval")
        return any(s in hay for s in seller_acts) and _term_hits(_product_terms(profile), hay) > 0

    # A curated buyer-industry list is evidence the triage should not be able to ignore.
    # It did: VVF's overrides name FMCG brands and detergent makers as its buyers, and
    # Titan -- a watchmaker -- still landed in the Potential Customer slot, because the
    # model reaches for the most newsworthy name rather than the most relevant one.
    import tenant as _tenant
    _ov = _tenant._overrides_for(getattr(profile, "company", ""))
    _curated_inds = _ov.get("industries") or []
    _ind_words = {w for i in _curated_inds
                  for w in re.split(r"[\s,&/]+", i.lower())
                  if len(w) > 4 and w not in {"companies", "company", "brands", "buyers",
                                              "makers", "manufacturers", "providers",
                                              "institutional", "industrial"}}

    def _matches_buyer_industry(name: str, art: Article) -> bool:
        """True when nothing in the signal ties the candidate to a curated buyer vertical.

        Only applied when industries were curated by hand -- an LLM-inferred list is
        not trustworthy enough to reject candidates on.
        """
        if not _ind_words:
            return True
        hay = f"{name} {art.title} {art.summary[:300]}".lower()
        return any(w in hay for w in _ind_words)

    # The scorer only filters: the top _TRIAGE_POOL items (event items first) go to the model,
    # labelled 10 per call -- the size it was trained on; one 40-item call drifted (0 buyers
    # for one lead, 20 of 40 for another, session_scratch/triage_one_call.py) -- with the 4
    # calls in parallel. Then the model picks the best buyer itself. Measured on the 15 test
    # leads: the scorer's own top 10 held 0 real buyers for 3 leads, and the model's best picks
    # sat at ranks 16, 21 and 40 (session_scratch/triage_bench.py).
    # One pool, one classification: half from the buyer searches, half from the competitor
    # searches, so every item is labelled buyer / peer / other once and both sections draw on
    # the same labels. A short side's spare places go to the other side.
    # A staff appointment is a person's move, not a buying event or a competitive move
    # ("Banque Richelieu Appoints New CEO" was pitched to BIL Suisse as a customer).
    ranked = sorted(_rank_by_evidence([a for a in triggers if not _is_appointment(a)]),
                    key=lambda a: 0 if _has_event(a) else 1)
    comp_ranked = _rank_by_evidence([a for a in (competitor_arts or [])
                                     if not config.is_government(a.title)
                                     and not _is_appointment(a)])
    half = _TRIAGE_POOL // 2 if comp_ranked else _TRIAGE_POOL
    n_comp = min(len(comp_ranked), _TRIAGE_POOL - min(len(ranked), half))
    pool = ranked[:_TRIAGE_POOL - n_comp] + comp_ranked[:n_comp]
    batches = [pool[s:s + _TRIAGE_BATCH] for s in range(0, len(pool), _TRIAGE_BATCH)]
    with cf.ThreadPoolExecutor(max_workers=max(1, len(batches))) as ex:
        results = list(ex.map(lambda b: triage_triggers(b, profile), batches))
    peers, buyers = [], []
    for bi, (rel, cands) in enumerate(results):
        for i, a in enumerate(cands):     # every label in the build log, so a miss can be traced
            n, r = rel.get(i, ("", "-"))
            log(f"    · triage [{bi * _TRIAGE_BATCH + i}] {r}: {n[:30] or '—'} | {a.title[:60]}")
            if r == "peer":
                peers.append((a, n))
            elif r == "buyer":
                # A truncated headline is not a company name ("Neogen raises Rs.600 crore,
                # Dhanuka ratings held," once rendered as the Potential Customer).
                if not n or _looks_like_headline(n):
                    continue
                if _looks_like_seller_category(n, a):
                    log(f"    · skipping '{n[:40]}' — reads as a rival/supplier, not a buyer")
                    continue
                if not _matches_buyer_industry(n, a):
                    log(f"    · skipping '{n[:40]}' — no tie to a curated buyer vertical")
                    continue
                buyers.append((a, n))
    chosen = None
    for _ in range(3):     # a pick that fails the country check is removed and the model re-picks
        if not buyers:
            break
        k, ec, wb = _pick_buyer(buyers, profile)
        if k < 0:
            log("    · pick: no buyer fits")
            break
        a, name = buyers.pop(k)
        a.event_country, a.would_buy = ec, wb
        if not _in_lead_country(a, profile) or _foreign_event(a, profile):
            log(f"    · pick '{name[:40]}' skipped — event is in {ec or 'another country'}")
            continue
        log(f"    · pick: {name} | would buy: {wb[:60]}")
        chosen = (a, name)
        break
    if not chosen:
        return None, None, peers
    art, company = chosen
    log(f"    · free-research: {company}")
    enr = free_research.enrich_company(company, config.get_country(getattr(profile, "market", "Global")))
    card = _write_card(art, company, "customer", enr, profile)
    card["_sources"] = enr.get("sources_used", [])
    card["_free_research"] = free_research.context_block(enr)
    return card, art, peers


# ── Fit triage: does this candidate genuinely map to the seller's specific value_prop,
# or only to its broad vertical keywords? ────────────────────────────────────────────
def triage_industry_fit(cands: list[Article], profile) -> list[Article]:
    """One LLM call. Vertical-keyword matching (score.py) is deliberately broad -- it
    floors relevancy for anything already in the trigger/industry bucket so a niche
    seller doesn't lose every candidate to a narrow lexicon. That breadth lets through
    industry-generic hits (any hospital story for a CAR-T-only seller, say) that read
    as on-topic but never connect to what the seller specifically sells. Re-rank the
    shortlist by genuine fit before the writer step turns a weak pick into fluent
    prose that validate_brief.py's FIT check would only catch after the fact."""
    top = cands[:8]
    if len(top) <= 1:
        return cands
    system = (
        f"You are a market editor for {profile.company} ({profile.short}). {profile.short} sells: "
        f"{profile.value_prop}\n"
        "Rate how closely each SIGNAL connects to what the seller sells.\n"
        "- high: it is about the seller's exact product area.\n"
        "- low: it only shares a broad industry with the seller."
    )
    user = (
        f"SIGNALS:\n{_cands_block(top)}\n\n"
        "Return JSON {\"items\":[{\"index\":0,\"fit\":\"high|medium|low\"}]} with one entry per SIGNAL."
        + _SCORE_NOTE
    )
    data = llm.chat_json(system, user, max_tokens=400, default={})
    fit: dict[int, str] = {}
    for it in (data.get("items") or []):
        if not isinstance(it, dict):
            continue  # model occasionally returns a bare string instead of {"index":...}
        i = it.get("index")
        if isinstance(i, int) and 0 <= i < len(top):
            f = (it.get("fit") or "medium").strip().lower()
            fit[i] = f if f in ("high", "medium", "low") else "medium"
    if not fit:
        return cands  # triage failed/degraded -- fall back to composite order, don't drop candidates
    order = {"high": 0, "medium": 1, "low": 2}
    ranked = sorted(range(len(top)), key=lambda i: order.get(fit.get(i, "medium"), 1))
    return [top[i] for i in ranked] + cands[len(top):]


# The v4 model was fine-tuned on this exact check (training_data/v4/check_prompt.json).
# The templates are reused verbatim; do not reword them.
_CHECK_SYSTEM = (
    "You are a competitive-intelligence analyst for <company> (<short>), which sells <value_prop>\n\n"
    "You will see one news SIGNAL about <competitor>, a company on the seller's competitor list. "
    "Decide whether the SIGNAL is a real competitive move that matters to <short>'s own market.\n\n"
    "A REAL competitive move: the company does something in or near the seller's own product category "
    "-- a product or service launch, a capacity or market expansion, an acquisition, a funding round, a "
    "price move, a major contract, or a partnership with product substance.\n\n"
    "NOT a real competitive move: stock or valuation notes, analyst ratings, fund or insider holding "
    "filings; HR, legal or leadership news; sports, sponsorship, awards or other PR with no product "
    "content; an unrelated business line of a diversified company; a regulator, university or namesake "
    "mistaken for the rival; a generic mention.\n\n"
    "Strip every category label and buzzword from the SIGNAL and ask: is there still a real move in the "
    "seller's specific product category? If not, the answer is false.")
_CHECK_USER = ("SIGNAL:\n TITLE: <title>\n SOURCE: <source> | DATE: <date>\n SUMMARY: <summary>\n\n"
               'Return JSON {"real_competitive_move": true|false}')


def _competitive_move_check(a: Article, competitor: str, profile) -> bool:
    """False when the model says the SIGNAL is not a real competitive move. A failed or
    unparseable call also returns False, so the candidate is skipped."""
    fill = {"<company>": profile.company, "<short>": profile.short,
            "<value_prop>": profile.value_prop, "<competitor>": competitor,
            "<title>": a.title, "<source>": a.source, "<date>": _date_label(a.published),
            "<summary>": a.summary[:400]}
    system, user = _CHECK_SYSTEM, _CHECK_USER
    for k, v in fill.items():
        system, user = system.replace(k, str(v)), user.replace(k, str(v))
    data = llm.chat_json(system, user, max_tokens=40, default={"real_competitive_move": False})
    return bool(isinstance(data, dict) and data.get("real_competitive_move") is True)


_COMPETITION_JUDGE_TOP = 24  # candidates sent (in parallel) to the competitive-move check


def _fill_event_countries(arts: list[Article]) -> None:
    """One small batched call that sets event_country on the articles that lack it (named-rival
    articles never went through triage). A failed call leaves it empty -- never skip on a guess."""
    todo = [a for a in arts if not (getattr(a, "event_country", "") or "").strip()]
    if not todo:
        return
    system = "You are a news analyst. For each SIGNAL, say in which country the event happens."
    user = (f"SIGNALS:\n{_cands_block(todo)}\n\n"
            "Return JSON {\"items\":[{\"i\":0,\"event_country\":\"\"}]} with one entry per SIGNAL. "
            "event_country is the country name in English, or \"unknown\" if the SIGNAL does not say.")
    data = llm.chat_json(system, user, max_tokens=300, default={})
    for it in (data.get("items") or []) if isinstance(data, dict) else []:
        if not isinstance(it, dict):
            continue
        i, ec = it.get("i"), it.get("event_country")
        if isinstance(i, int) and 0 <= i < len(todo) and isinstance(ec, str):
            todo[i].event_country = ec.strip()


# ── 2. COMPETITION OUTLOOK — best move among PEERS (from triggers) + competitor bucket ─
def build_competition(competitor_arts: list[Article], peer_arts, profile, log=print):
    # combine peers surfaced in the trigger pool (e.g. a rival's raise) with the
    # dedicated competitor-name bucket, then take the most newsworthy (highest score).
    named = [c for c in (profile.competitors or []) if c.strip()]

    def _matches(name: str, text: str) -> bool:
        """True if the rival's distinctive first words -- or any of its aliases (English
        form, local form, acronym; Profile.competitor_aliases) -- appear in the text."""
        return tenant.rival_named_in(profile, name, text)

    non_buyers = getattr(config, "NON_BUYER_TERMS", ())

    def _usable_peer(n: str) -> bool:
        """Triage-identified peers are accepted even when the LLM-guessed competitor
        list missed them (the trained competitive-move check judges them); only
        regulators/bodies and headline fragments are excluded."""
        n = (n or "").strip()
        return bool(n) and not _looks_like_headline(n) \
            and not is_self(n, profile) \
            and not any(w in n.lower() for w in non_buyers)

    named = [c for c in named if not is_self(c, profile)]
    combined: list[tuple[Article, str]] = [(a, n) for (a, n) in peer_arts if _usable_peer(n)]
    # Scan the whole bucket, not the top 10 by score. The list is sorted on composite,
    # and a diversified rival's high-volume unrelated news fills those slots -- the
    # on-category signals (a pure-play's CAR-T raise) sat at rank 20+ and were never
    # even considered. Attribution is cheap; do it for everything and rank afterwards.
    for a in competitor_arts:
        name = ""
        # The TITLE must name the rival, never the query alone. Google News answers
        # "Bristol Myers Squibb India" with a Dr Reddy's headline, and matching on the
        # query wrote that story up as if BMS were the subject -- a rival's section
        # about a company that is not that rival.
        for c in named:
            if _matches(c, a.title) or _matches(c, a.summary or ""):
                name = c; break
        if name:
            combined.append((a, name))
    # Category-search articles that name no listed rival were labelled once, with everything
    # else, in build_customer's pool: the ones labelled peer arrive in peer_arts. No second
    # triage call here.
    if not combined:
        return None, None

    # Product-category match is a RANKING signal only. A keyword filter in front of the
    # trained judge dropped most leads' Competition Outlook (a relevant move whose words
    # don't echo a product name never reached the model); the competitive-move check
    # below makes that judgement. On-category candidates are simply tried first.
    product_terms = _product_terms(profile)

    def _on_category(a: Article) -> int:
        hay = f"{a.title} {a.summary[:300]}".lower()
        return 1 if product_terms and _term_hits(product_terms, hay) else 0

    pool, seen_titles = [], set()
    for art, comp in sorted(combined, key=lambda x: (_on_category(x[0]), _evq(x[0]), x[0].composite),
                            reverse=True):
        key = re.sub(r"[^a-z0-9]+", " ", (art.title or "").lower()).strip()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        pool.append((art, comp))
    _floor = getattr(config, "MIN_EV_RELEVANCY", 0.15)
    _strong = [p for p in pool if (getattr(p[0], "ev_relevancy", 0.0) or 0.0) >= _floor]
    pool = (_strong or pool)[:_COMPETITION_JUDGE_TOP]
    log(f"    · judging {len(pool)} of {len(combined)} competitor signals")
    _fill_event_countries([a for a, _ in pool])      # before the check, so foreign events cost no call
    local = []
    for art, comp in pool:
        if _foreign_event(art, profile):
            log(f"    · skipping '{comp[:40]}' — event is in {art.event_country}, not the lead's country")
        else:
            local.append((art, comp))
    # Every candidate gets the trained check at once (it used to stop at the first pass, so a
    # staff hire that came first beat a real deal further down). Then the model picks.
    with cf.ThreadPoolExecutor(max_workers=max(1, min(8, len(local)))) as ex:
        verdicts = list(ex.map(lambda p: _competitive_move_check(p[0], p[1], profile), local))
    passed = []
    for (art, comp), ok in zip(local, verdicts):
        log(f"    · {comp}: '{art.title[:60]}' "
            + ("passed the competitive-move check" if ok else "failed the competitive-move check"))
        if ok:
            passed.append((art, comp))
    for _ in range(3):     # a pick whose write-up fails is removed and the model re-picks
        if not passed:
            break
        k = _pick_competitor(passed, profile) if len(passed) > 1 else 0
        if k < 0:
            log("    · pick: no competitor move matters enough")
            break
        art, comp = passed.pop(k)
        log(f"    · pick: {comp} | {art.title[:60]}")
        log(f"    · free-research: {comp}")
        enr = free_research.enrich_company(comp, config.get_country(getattr(profile, "market", "Global")))
        card = _write_card(art, comp, "competitor", enr, profile)
        if card is None:
            log(f"    · {comp}'s signal is not a real competitive move -- trying next candidate")
            continue
        # The model judges its own write-up with the same trained check: a card whose text
        # says the move is not relevant must not ship (the writer rarely sets no_real_signal).
        own = re.sub(r"<[^>]+>", "", card.get("narrative_html") or "").strip()
        if own and not _competitive_move_check(replace(art, summary=own), comp, profile):
            log(f"    · {comp}'s write-up failed the competitive-move check -- trying next candidate")
            continue
        card["_sources"] = enr.get("sources_used", [])
        card["_free_research"] = free_research.context_block(enr)
        return card, art
    return None, None


def _pick_competitor(moves: list[tuple[Article, str]], profile) -> int:
    """One LLM call: the move that matters most to the seller, or -1 when none does."""
    market = (getattr(profile, "market", "") or "").strip() or "the reader's country"
    system = (
        f"You are a competitive-intelligence analyst for {profile.company} ({profile.short}). "
        f"{profile.short} sells: {profile.value_prop}\n\n"
        f"Each CANDIDATE is news about a competitor of {profile.short}. Pick the ONE move that matters "
        f"most to {profile.short}.\n"
        "A strong move:\n"
        f"- is a launch, expansion, acquisition, large contract, funding round or price move in "
        f"{profile.short}'s own product area;\n"
        f"- happens in {market}.\n"
        "Weak news: staff hires, share price or results notes, awards, and moves in a different business.\n"
        "If every candidate is weak, answer -1."
    )
    lines = "\n".join(f"[{k}] {name} | {a.title} | {(a.summary or '')[:160]}"
                      for k, (a, name) in enumerate(moves))
    data = llm.chat_json(system, f"CANDIDATES:\n{lines}\n\nReturn JSON {{\"best\": <index or -1>}}",
                         max_tokens=60, default={})
    k = data.get("best") if isinstance(data, dict) else None
    # an explicit -1 means none fits; a failed or unparseable reply falls back to the top-ranked
    return k if isinstance(k, int) and -1 <= k < len(moves) else 0


# ── Prose guard: one shared helper (voice.chat_json_guarded) for every writer call ──
def _chat_guarded(system: str, user: str, *, max_tokens: int, profile=None) -> dict:
    return voice.chat_json_guarded(system, user, max_tokens=max_tokens, profile=profile)


# ── Shared card writer (customer or competitor) ──────────────────────────────
def _write_card(article: Article, name: str, kind: str, enr: dict, profile) -> dict:
    a = article
    ind = _industry_of(a, profile)
    ctx = free_research.background_block(enr)
    fr = f"\nFREE RESEARCH (background on who the company is; not news):\n{ctx}\n" if ctx else ""
    src = (f"SIGNAL:\n TITLE: {a.title}\n SOURCE: {a.source} | DATE: {_date_label(a.published)}\n"
           f" SUMMARY: {a.summary[:400]}\n")
    _sc = voice.seller_context(profile, buyers=(kind != "competitor"))
    if _sc:
        src += _sc + "\n"
    facts = ("- Facts about the event come only from the SIGNAL. Use FREE RESEARCH only to say who "
             f"{name} is.\n"
             f"- {name} and {profile.short} are different companies. Never describe {name} with "
             f"{profile.short}'s products or business.\n"
             "- Do not add numbers, units, places or events that are not in the SIGNAL.\n"
             "- Skip parts of the SIGNAL that are not about the event, such as quotes or stories about "
             "people.\n"
             "- If the SIGNAL is not in English, state its facts in English.\n")

    if kind == "competitor":
        system = (
            f"You are a competitive-intelligence analyst for {profile.company} ({profile.short}). "
            f"{profile.short} sells: {profile.value_prop}\n"
            f"You write a short card for {profile.short}'s team about one move by the competitor {name}, "
            "based on one news item (the SIGNAL).\n"
            + facts +
            f"- If the SIGNAL is not a real move by {name} in {profile.short}'s product area, return "
            "{\"no_real_signal\": true} and nothing else.\n\n"
            + voice.active_style()
        )
        user = (
            f"{src}{fr}\nReturn JSON with these keys:\n"
            "- narrative_html: 3 to 5 short sentences in this order: what "
            f"{name} did, with the key fact from the SIGNAL; who {name} is; how the move touches "
            f"{profile.short}'s market, naming one {profile.short} product. <b> allowed.\n"
            f"- why_it_matters: one sentence on what this move could change for {profile.short}.\n"
            f"- their_edge: a short phrase: the strength {name} relies on in this move, from the SIGNAL.\n"
            f"- recommended_response: 1 to 2 sentences: a concrete step {profile.short} could take, naming "
            "the product to lead with.\n"
            "- counter_play: exactly 3 steps, each {\"title\": a short instruction, \"body_html\": one "
            "sentence on how to do it}. Each step must be different."
        )
        data = _chat_guarded(system, user, max_tokens=1000, profile=profile)
        if data.get("no_real_signal"):
            return None
        steps = _steps(data.get("counter_play"), profile, "competitor")
        return {
            "flag": f"Competition Outlook · {ind}",
            "company_name": name,
            "source_url": a.url,
            "source_name": a.source,
            "narrative_html": _clean_frag(data.get("narrative_html") or "") or a.summary or a.title,
            "left": {"h": "Why it matters", "v": _clean_frag(data.get("why_it_matters") or "") or "A competitor is moving on your turf right now."},
            "right": {"h": "Their edge", "v": _clean_frag(data.get("their_edge") or "") or "Scale and cadence."},
            "action": {"h": "Recommended response", "v": _clean_frag(data.get("recommended_response") or "")},
            "plan": {"h": "Counter-play", "steps": steps},
            "_source_text": f"{a.title}. {a.summary[:400]}",
        }

    # customer -- for an investor the card is a potential deal, for a non-profit,
    # university or public body a potential partnership (see compose.prospect_heading)
    _lt = getattr(profile, "lead_type", "seller") or "seller"
    if _lt == "investor":
        _kind_txt, _need_txt, _roles_txt = (
            "potential-deal brief", f"why this could interest {profile.short} as a deal", "contact roles")
    elif _lt in ("nonprofit", "education", "government"):
        _kind_txt, _need_txt, _roles_txt = (
            "potential-partnership brief", f"how this fits {profile.short}'s mission", "contact roles")
    else:
        _kind_txt, _need_txt, _roles_txt = (
            "prospect brief", f"which {profile.short} product could fit this, and why",
            "job titles")
    system = (
        f"You are a sales analyst for {profile.company} ({profile.short}). {profile.short} sells: "
        f"{profile.value_prop}\n"
        f"You write a {_kind_txt} for {profile.short}'s sales team about {name}, based on one news "
        "item (the SIGNAL).\n"
        + facts + "\n"
        + voice.active_style()
    )
    user = (
        f"{src}{fr}\nReturn JSON with these keys:\n"
        "- narrative_html: 3 to 5 short sentences in this order: what "
        f"{name} did, with the key fact from the SIGNAL; who {name} is; {_need_txt}. <b> allowed.\n"
        f"- why_now: one sentence: what in the SIGNAL makes now a good time to contact {name}, using "
        "only the SIGNAL.\n"
        f"- who_to_target: 2 to 3 {_roles_txt} at {name}, comma separated.\n"
        f"- recommended_action: 1 to 2 sentences: the first step {profile.short}'s team could take, "
        "naming the product to lead with.\n"
        "- approach_plan: exactly 3 steps, each {\"title\": a short instruction, \"body_html\": one "
        "sentence on how to do it}. Each step must be different."
    )
    data = _chat_guarded(system, user, max_tokens=1000, profile=profile)
    steps = _steps(data.get("approach_plan"), profile, "customer")
    return {
        "flag": f"{_heading(profile)} · {ind}",
        "company_name": name,
        "source_url": a.url,
        "source_name": a.source,
        "narrative_html": _clean_frag(data.get("narrative_html") or "") or a.summary or a.title,
        "left": {"h": "Why now", "v": _clean_frag(data.get("why_now") or "") or "The buying window is open now."},
        "right": {"h": "Who to target", "v": _clean_frag(data.get("who_to_target") or "") or _who_fallback(profile)},
        "action": {"h": "Recommended action", "v": _clean_frag(data.get("recommended_action") or "")},
        "plan": {"h": "Approach plan", "steps": steps},
        "_source_text": f"{a.title}. {a.summary[:400]}",
    }


def _who_fallback(profile) -> str:
    funcs = [f for f in (getattr(profile, "buyer_functions", None) or []) if f]
    return ", ".join(funcs[:3]) if funcs else "Head of Corporate Development, VP of Strategy."


def _steps(raw, profile, kind: str) -> list[dict]:
    out = []
    for step in (raw or [])[:3]:
        if isinstance(step, dict) and step.get("title"):
            out.append({"title": _clean_frag(step["title"]), "body_html": _clean_frag(step.get("body_html", ""))})
    return out or _fallback_plan(profile, kind)


def _ctx_prefix(profile) -> str:
    sc = voice.seller_context(profile)
    return sc + "\n\n" if sc else ""


# ── 3. INDUSTRY OUTLOOK — synthesize the industry news we track ───────────────
def _topic_line(market_topic: str) -> str:
    """Report briefs promise one market; keep the writer on it."""
    if not market_topic:
        return ""
    return (f"The reader asked about the {market_topic} market. Write only about {market_topic}, "
            "using only what the SIGNAL says about it.\n")


def _outlook_order(short: str) -> str:
    """What the Industry Outlook / Your Market paragraph says, sentence by sentence."""
    return ("Write 3 to 5 short sentences in this order:\n"
            "1. What happened, with the key fact from the SIGNAL (a number, a date, a place or a name).\n"
            "2. Who this affects: which buyers or which part of the market.\n"
            f"3. How it could matter to {short}, naming one {short} product.\n"
            "If the SIGNAL happened outside the reader's country, say where.\n"
            "If the SIGNAL is not in English, state its facts in English.\n")


def synthesize_news(arts: list[Article], profile, market_topic: str = "") -> list[dict]:
    """`market_topic`, when given, is the EXACT subject the recipient was promised (a
    report brief's section 1, "Your Market", hard-codes its title to this string in
    compose.py regardless of what this function writes) -- so on a report brief this is
    the only writer step that can keep the body content honest about what the headline
    claims. Added 2026-10-08 after a report brief headlined "Metal Roofings Market" over
    a body about UAE renewable energy: the selected signal was off-topic and nothing in
    this prompt ever told the model what topic it was supposed to be on."""
    if not arts:
        return []
    system = (
        f"You are a market editor for {profile.company} ({profile.short}). {profile.short} sells: "
        f"{profile.value_prop}\n"
        "For each SIGNAL, write a headline and a short paragraph for the Industry Outlook section of "
        f"a brief sent to {profile.short}.\n"
        + _outlook_order(profile.short) +
        f"{_topic_line(market_topic)}\n"
        + voice.active_style()
    )
    user = (
        f"{_ctx_prefix(profile)}SIGNALS:\n{_cands_block(arts)}\n\n"
        "Return JSON {\"items\":[{\"index\":0,\"headline\":\"\",\"body_html\":\"\"}]} with one entry per SIGNAL.\n"
        "- headline: short and true to the SIGNAL, plain text.\n"
        "- body_html: your sentences. <b> and <i> allowed." + _SCORE_NOTE
    )
    data = _chat_guarded(system, user, max_tokens=2000, profile=profile)
    items = data.get("items") if isinstance(data, dict) else None
    out: list[dict] = []
    if items:
        for it in items:
            if not isinstance(it, dict):
                continue  # model occasionally returns a bare string instead of {"index":...}
            i = it.get("index")
            if not isinstance(i, int) or i < 0 or i >= len(arts):
                continue
            a = arts[i]
            out.append({"headline": _clean_frag(it.get("headline") or "") or a.title,
                        "body_html": _clean_frag(it.get("body_html") or "") or a.summary or a.title,
                        "source": a.source, "source_url": a.url, "source_name": a.source,
                        "date": _date_label(a.published), "verified": True,
                        "_source_text": f"{a.title}. {a.summary[:400]}"})
    if not out:
        for a in arts:
            out.append({"headline": a.title, "body_html": a.summary or a.title,
                        "source": a.source, "source_url": a.url, "source_name": a.source,
                        "date": _date_label(a.published), "verified": True,
                        "_source_text": f"{a.title}. {a.summary[:400]}"})
    return out[:1]   # Industry Outlook carries a single signal


def triage_and_synthesize_industry(cands: list[Article], profile,
                                   market_topic: str = "", _depth: int = 0) -> list[dict]:
    """Merges triage_industry_fit + synthesize_news into one call: build_brief only
    ever uses the #1-ranked candidate out of triage_industry_fit's output (fit_pool[:1]
    right before calling synthesize_news on it), so ranking 8 candidates and writing
    prose for exactly one of them were two separate LLM round-trips for what is really
    one decision -- "which of these is the best fit, and what does it say". Combined
    2026-10-14 to cut one call per brief; same fallback behavior as each original
    function (degraded ranking falls back to composite order, no candidates returns [],
    a parse failure still falls back to the raw top candidate's own title/summary
    rather than producing nothing)."""
    if not cands:
        return []
    top = cands[:8]
    if len(top) == 1:
        return synthesize_news(top, profile, market_topic=market_topic)
    pick = (f"the {market_topic} market" if market_topic
            else f"what {profile.short} sells (its exact product area, not just its broad industry)")
    system = (
        f"You are a market editor for {profile.company} ({profile.short}). {profile.short} sells: "
        f"{profile.value_prop}\n"
        "STEP 1. Pick one SIGNAL.\n"
        "- It must report a real event or change: a new rule, a deal, a launch, an investment, a "
        "price or demand shift.\n"
        "- Skip guides, lists, product reviews, opinion pieces and stories about one person.\n"
        f"- Among the rest, pick the one closest to {pick}.\n"
        "STEP 2. For that SIGNAL only, write a headline and a short paragraph.\n"
        + _outlook_order(profile.short) +
        f"{_topic_line(market_topic)}\n"
        + voice.active_style()
    )
    user = (
        f"{_ctx_prefix(profile)}SIGNALS:\n{_cands_block(top)}\n\n"
        'Return JSON {"best_index":0,"headline":"","body_html":"","event_country":""}\n'
        "- best_index: the index of the SIGNAL you picked.\n"
        "- headline: short and true to that SIGNAL, plain text.\n"
        "- body_html: your sentences. <b> and <i> allowed.\n"
        "- event_country: the country where the event happens, in English, or \"unknown\"." + _SCORE_NOTE
    )
    data = _chat_guarded(system, user, max_tokens=900, profile=profile)
    i = data.get("best_index") if isinstance(data, dict) else None
    if not isinstance(i, int) or i < 0 or i >= len(top):
        # Triage/write failed or degraded -- fall back to composite order and let
        # synthesize_news make its own independent attempt at the top candidate,
        # same as the pre-merge code path did on a failed triage.
        return synthesize_news(cands, profile, market_topic=market_topic)
    a = top[i]
    ec = data.get("event_country")
    a.event_country = ec.strip() if isinstance(ec, str) else ""
    if (_foreign_event(a, profile) and _depth < 3
            and getattr(a, "geo_scope", "") not in ("region", "global")):
        # The chosen signal happened in another known country: pick again from the rest.
        # (Topic news admitted from abroad on purpose, geo_scope region/global, is kept.)
        # (Extra call only in this rare case; the normal path stays one call.)
        rest = [c for c in cands if c is not a]
        if rest:
            return triage_and_synthesize_industry(rest, profile, market_topic, _depth + 1)
    headline = _clean_frag(data.get("headline") or "") or a.title
    body_html = _clean_frag(data.get("body_html") or "") or a.summary or a.title
    return [{"headline": headline, "body_html": body_html,
             "source": a.source, "source_url": a.url, "source_name": a.source,
             "date": _date_label(a.published), "verified": True,
             "_source_text": f"{a.title}. {a.summary[:400]}"}]


# ── Assemble ─────────────────────────────────────────────────────────────────
def build_brief(scored: list[Article], profile, log=print, *,
                market_topic: str = "") -> tuple[dict, dict]:
    now = datetime.now(timezone.utc)

    # Every trigger-bucket item is a candidate: it came from a buyer-event query, and triage
    # judges it. trigger_type is an English keyword match, so gating on it dropped nearly
    # all local-language news (Terra Innova: 8 of 142, VDL Hapro: 1 of 77).
    triggers = [a for a in scored if a.bucket == "trigger"]
    industry = [a for a in scored if a.bucket == "industry"] or [a for a in scored if a.bucket == "trigger" and not a.trigger_type]
    competitor = [a for a in scored if a.bucket == "competitor"]
    log(f"  selecting: {len(triggers)} trigger / {len(industry)} industry / {len(competitor)} competitor eligible")

    log("  → [1] Potential Customer: triage buyers/peers → free-enrich → write")
    customer, cust_art, peer_arts = build_customer(triggers, profile, log=log,
                                                   competitor_arts=competitor)
    # peers (rivals surfaced in the trigger pool) feed Competition Outlook, minus the customer pick
    peer_arts = [(a, n) for (a, n) in peer_arts if cust_art is None or a.title != cust_art.title]

    # 2026-10-19: Market Moves is no longer generated for report briefs -- compose.py
    # now caps report briefs at 2 sections (Your Market + Competition Outlook) per the
    # cmiinsights.info redesign, so a Market Moves card was being written and then
    # unconditionally discarded, wasting one full LLM call per report brief for
    # content nothing ever rendered. Removed rather than left dead: see build_brief's
    # old comment history if this section needs reviving.
    if getattr(profile, "profile_confidence", "normal") == "low":
        # a low-confidence profile's rival list is a guess; writing it up misleads
        competition, comp_art = None, None
        log("  → [2] Competition Outlook: skipped — low-confidence profile (rival list is a guess)")
    else:
        log("  → [2] Competition Outlook: peers + competitor bucket → free-enrich → write")
        competition, comp_art = build_competition(competitor, peer_arts, profile, log=log)

    used = {a.title for a in (cust_art, comp_art) if a}
    # A staff appointment is not a market development ("BD Appoints Matthew Sassone as
    # President of Pharmacy Automation" became Kirey's Your Market).
    industry_pool = [a for a in industry if a.title not in used and not _is_appointment(a)]
    theme, themed = cluster.dominant_theme(industry_pool)
    if theme:
        log(f"  → [3] Industry Outlook: TF-IDF theme = '{theme}' ({len(themed)} signals)")
    # Report briefs promise a specific market, and that card ("Your Market") is written
    # from this pool. The TF-IDF theme above is the biggest cluster of the SELLER's own
    # industry-query stories, which are rarely about the enquired market (measured: 0-1%
    # of the eligible pool mentioned the topic; a "Space Ground Station" brief was
    # written from UK banking cyber-loss news). So when a topic was given, rank articles
    # that are actually about it -- from any bucket except competitor. Nothing on-topic means
    # no Your Market section, never the seller's own unrelated news under the topic heading.
    topical: list[Article] = []
    if market_topic:
        import score as _score
        topical = sorted(
            (a for a in scored if a.title not in used and a.bucket != "competitor"
             and getattr(a, "topic_rel", 0.0) >= _score.TOPICAL and not _is_appointment(a)),
            key=lambda a: (_SCOPE_RANK.get(getattr(a, "geo_scope", ""), 0),
                           -a.topic_rel, -a.composite))
        # In-country first, then region, then global: the writer only sees the nearest tier
        # that has anything, so a foreign story is used only when the lead's own country
        # has nothing on-topic (the sort alone did not stop the model picking India).
        if topical:
            nearest = _SCOPE_RANK.get(getattr(topical[0], "geo_scope", ""), 0)
            topical = [a for a in topical if _SCOPE_RANK.get(getattr(a, "geo_scope", ""), 0) == nearest]
        log(f"  → [3] Your Market: {len(topical)} on-topic signals for '{market_topic}'")
    # One industry signal only: the strongest of the dominant theme. More than one
    # dilutes the section and repeats the same analyst read. triage (rank by fit) and
    # synthesize (write prose for the #1 pick) are one combined call -- see
    # triage_and_synthesize_industry's docstring for why.
    log("  → [3] Industry Outlook: triage + synthesize news")
    if market_topic:
        # Nothing on-topic -> no section: the seller's own unrelated news must not be
        # written up under the topic heading (compose drops the empty section).
        shortlist = _above_floor(topical) if topical else []
        if not topical:
            log("  → [3] Your Market: no on-topic news — section skipped")
    else:
        # The TF-IDF theme is often junk and excluded the best-scored items, so the
        # signal brief shortlists the whole pool by evidence score (theme: log only).
        shortlist = _rank_by_evidence(industry_pool)
    industry_news = triage_and_synthesize_industry(shortlist, profile,
                                                    market_topic=market_topic)

    # NOTE: this used to also call build_editorial() here to produce a lede, three
    # top3 headline lines, and three analyst "takes" -- a full extra LLM call
    # (~1600 tokens). Nothing ever read that output: compose.build_context() computes
    # its OWN lede/top3 from the section cards via compose.editorial() (a second,
    # different LLM call), and that second version is the only one that reaches the
    # rendered template. The first call's output sat in brief["intro"]/brief["top3"]/
    # brief["take_*"] and was discarded unread every single time. Removed rather than
    # wired through, because build_editorial's top3 shape is keyed to the SIGNAL brief's
    # fixed section names ("Potential Customer"/"Industry Outlook"/"Competition
    # Outlook") and does not line up with a report brief's actual sections ("Your
    # Market"/"Market Moves"/"Competition Outlook") the way compose.editorial's
    # sections-driven version already does correctly for both kinds -- so keeping
    # compose's version and dropping this one avoids a mismatch on top of the waste.

    _market = getattr(profile, "market", "Global") or "Global"
    brief = {
        "brand": {"wordmark": "CoherentConnect",
                  "issue_label": "Daily Signal Brief" + (f" · {_market}" if _market != "Global" else ""),
                  "issue_date": now.strftime("%d %b %Y"), "tagline": profile.tagline},
        "recipient": {"first_name": profile.recipient},
        "meta": {"date_pill": now.strftime("%A, %B ") + str(now.day) + now.strftime(", %Y"), "seller": profile.company},
        # 1. Potential Customer
        "potential_customer": customer,
        # 2. Competition Outlook
        "competition_outlook": competition,
        # 3. Industry Outlook (signal briefs only; report briefs cap at 2 sections)
        "industry_outlook": industry_news,
        # closing — the "Inside CoherentConnect" showcase is static in the template
        "footer": {"assembled_note": "Assembled & verified by CoherentConnect", "cta_text": "Book a Demo →",
                   "cta_url": DEMO_URL, "site_text": "Visit the Site →", "site_url": SITE_URL,
                   "tagline": profile.tagline, "subline": "Your autonomous sales intelligence agent",
                   "unsubscribe_url": "#"},
    }
    debug = {
        "customer": {"company": customer["company_name"], "article": cust_art.title, "score": cust_art.composite,
                     "url": cust_art.url, "source": cust_art.source, "published": cust_art.published,
                     "enriched_via": customer.get("_sources", []),
                     "free_research": customer.get("_free_research", "")} if customer else None,
        "competition": {"company": competition["company_name"], "article": comp_art.title, "score": comp_art.composite,
                        "url": comp_art.url, "source": comp_art.source, "published": comp_art.published,
                        "enriched_via": competition.get("_sources", []),
                        "free_research": competition.get("_free_research", "")} if competition else None,
        "industry_theme": theme,
        "counts": {"trigger": len(triggers), "industry": len(industry_news), "competitor": len(competitor)},
    }
    return brief, debug
