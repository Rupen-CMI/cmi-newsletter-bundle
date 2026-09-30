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
from datetime import datetime, timezone

import config
import llm
import voice
import cluster
import free_research
from collect import Article

_INLINE_OPEN = re.compile(r"^<(b|i|em|strong|span|a)\b", re.I)
_INLINE_CLOSE = re.compile(r"</(b|i|em|strong|span|a)>$", re.I)

# Every "Book a demo" link in the brief (3 section CTAs + the footer button) resolves
# from this one value via footer.cta_url. No ?month= param — Calendly opens on the
# current month by default, and a pinned month goes stale the moment it passes.
DEMO_URL = "https://calendly.com/aditya-jha-coherentmarketinsights/30min"
SITE_URL = "https://coherentconnect.ai/"


def _clean_frag(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
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


def _date_label(dt: datetime) -> str:
    return f"{dt.strftime('%b')} {dt.day}"


def _cands_block(arts: list[Article]) -> str:
    lines = []
    for i, a in enumerate(arts):
        lines.append(f"[{i}] TITLE: {a.title}\n    SOURCE: {a.source} | DATE: {_date_label(a.published)}"
                     f" | TRIGGER: {a.trigger_type or 'none'} | QUERY: {a.query} | SCORE: {a.composite}\n"
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


# ── Triage: classify each trigger company as BUYER vs PEER (competitor) vs OTHER ─
def triage_triggers(triggers: list[Article], profile):
    """One LLM call. Returns {index: (company_name, relationship)} over the top candidates.
    'buyer' = would purchase the seller's product; 'peer' = sells a competing product
    (a rival); 'other' = neither. This is what keeps a rival's funding round OUT of the
    Potential Customer slot and INTO Competition Outlook."""
    cands = triggers[:10]
    if not cands:
        return {}, cands
    system = (
        f"You are a B2B sales analyst for {profile.company} ({profile.short}), which sells {profile.value_prop} "
        f"Its CUSTOMERS are companies in these buyer industries: {', '.join(profile.industries)}. "
        "For each news signal, classify the company's relationship to the seller:\n"
        "  'buyer' = a company that would BUY the seller's product (a customer in the buyer industries with a "
        "need the product solves)\n"
        "  'peer'  = a company that SELLS a similar/competing product — a RIVAL in the seller's own space "
        "(even if well-funded). A rival's funding/launch is competition, NOT a customer.\n"
        "  'other' = neither.\n"
        "Be strict: if the company does what the seller does, it is 'peer', never 'buyer'.\n"
        "Be equally strict the other way: 'buyer' requires a company whose OWN business "
        "sits in one of the buyer industries listed above and that would plausibly place "
        "a purchase order for what the seller makes. A large, famous company in an "
        "unrelated trade is 'other', not 'buyer' -- a watchmaker does not buy "
        "oleochemicals, a fashion label does not buy ayurvedic bulk actives. If no "
        "candidate is a genuine buyer, mark them all 'other' rather than stretching for "
        "the biggest name."
    )
    user = (f"SELLER sells: {profile.value_prop}\nKnown competitors: {', '.join(profile.competitors) or '—'}\n\n"
            f"SIGNALS:\n{_cands_block(cands)}\n\n"
            "Return JSON {\"items\":[{\"index\":<int>,\"company_name\":\"...\",\"relationship\":\"buyer|peer|other\"}]}"
            " — one entry per signal index above.")
    data = llm.chat_json(system, user, max_tokens=600, default={})
    rel: dict[int, tuple[str, str]] = {}
    for it in (data.get("items") or []):
        i = it.get("index")
        if isinstance(i, int) and 0 <= i < len(cands):
            r = (it.get("relationship") or "other").strip().lower()
            r = r if r in ("buyer", "peer", "other") else "other"
            rel[i] = (_clean_frag(it.get("company_name") or "") or cands[i].title[:48], r)
    return rel, cands


# ── MARKET MOVES (report briefs) — capital moving in the reader's market ───────
def build_market_moves(pool: list[Article], profile, topic: str, log=print):
    """One card on who is acquiring, investing, expanding or partnering in `topic`.

    Section 2 of a report brief. Chosen over a regulatory or demand framing on two
    measurements across the 20-lead sample: it finds the most usable signal (940 real
    articles, 19/20 leads covered), and "market size / growth / forecast" queries turn
    out to be dominated by rival research firms' SEO pages -- 44% real news against 75%
    here. CMI must not cite a competitor's market-size page inside its own brief.

    It is also the framing that sells. A rule change is information; capital moving in
    your market is a reason to find out what is happening before you commit, which is
    exactly what this section's existing CTA offers.
    """
    if not pool:
        return None, None
    art = pool[0]
    log(f"    · market moves: {art.title[:52]}")
    system = (
        "You are a senior market analyst at Coherent Market Insights writing the "
        "'Market Moves' section of a brief for " + profile.company +
        " about the " + topic + ".\n\n"
        "Report what a named company did in this market -- an acquisition, investment, "
        "expansion, new plant, or partnership -- and what it signals about where the "
        "market is heading. Ground every claim in the SIGNAL; invent no figures.\n\n"
        + voice.active_style()
    )
    user = (
        "MARKET: " + topic + "\n"
        "SIGNAL:\n TITLE: " + art.title + "\n"
        " SOURCE: " + art.source + " | DATE: " + _date_label(art.published) + "\n"
        " SUMMARY: " + art.summary[:400] + "\n\n"
        "Return a JSON object with keys:\n"
        "  company_name — the company making the move\n"
        "  narrative_html — 3 to 5 SHORT sentences (max 24 words each) on what they did "
        "and what it signals for the market. Inline <b> allowed.\n"
        "  why_it_matters — one sentence on why this move matters now\n"
        "  what_it_signals — one phrase on the direction it points"
    )
    data = llm.chat_json(system, user, max_tokens=900, default={})
    name = _clean_frag(data.get("company_name") or "") or art.source
    return {
        "flag": "Market Moves · " + topic.lower(),
        "company_name": name,
        # Carry the article through to the template. The brief names its sources in
        # prose ("according to a Livemint report"), so without this the reader has a
        # citation they cannot click and the brief looks unsourced.
        "source_url": art.url,
        "source_name": art.source,
        "narrative_html": _clean_frag(data.get("narrative_html") or "") or art.summary or art.title,
        "left": {"h": "Who is moving",
                 "v": _clean_frag(data.get("why_it_matters") or "")
                      or "Capital is moving in this market right now."},
        "right": {"h": "What it signals",
                  "v": _clean_frag(data.get("what_it_signals") or "")
                       or "Consolidation and fresh capacity."},
    }, art


# ── 1. POTENTIAL CUSTOMER — best BUYER (never a peer/rival), free-enrich, write ─
def build_customer(triggers: list[Article], profile, log=print):
    rel, cands = triage_triggers(triggers, profile)

    def _looks_like_seller_category(name: str, art: Article) -> bool:
        """True if this candidate is in the seller's OWN trade rather than a buyer.

        The triage LLM answers this inconsistently -- the same Dr Reddy's biosimilar
        launch came back 'other' on one run and 'buyer' on the next, which put a drug
        maker in the Potential Customer slot for a CAR-T seller. A deterministic guard
        on top of the model's judgement costs nothing and cannot drift between runs.
        """
        hay = f"{name} {art.title} {art.summary[:200]}".lower()
        # A company already named as a rival is never a customer, whatever triage said.
        for c in (profile.competitors or []):
            cb = re.sub(r"[^a-z0-9 ]", " ", (c or "").lower()).strip()
            head = " ".join(cb.split()[:2])
            if len(head) > 5 and head in hay:
                return True
        # Nor is a body that regulates or reports on the market rather than buying in
        # it. "India's drug regulator has formed an AI expert committee" landed in the
        # Potential Customer slot: a real signal, but a regulator has no purchase order
        # to give, and the model dutifully invented a reason it might buy.
        non_buyers = ("regulator", "regulatory authority", "ministry", "government of",
                      "cdsco", "dcgi", "fda", "court", "tribunal", "parliament",
                      "association", "council", "committee", "commission", "board of",
                      "department of", "task force", "research institute", "journal")
        if any(w in (name or "").lower() for w in non_buyers):
            return True
        # Nor is a company whose news is about selling the seller's own product type.
        seller_acts = ("launches", "launch of", "unveils", "rolls out", "approval for",
                       "gets approval", "receives approval")
        seller_goods = ("biosimilar", "car-t", "cell therapy", "gene therapy", "drug",
                        "therapy", "oncology portfolio", "molecule")
        return any(s in hay for s in seller_acts) and any(g in hay for g in seller_goods)

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

    # prefer buyers (in composite order); if none, fall back to 'other'; never a 'peer'
    chosen = None
    for want in ("buyer", "other"):
        for i, a in enumerate(cands):
            name, r = rel.get(i, ("", "other"))
            if r != want:
                continue
            # A truncated headline is not a company name. When triage returned nothing
            # for this index the old fallback was a.title[:48], which rendered
            # "Neogen raises Rs.600 crore, Dhanuka ratings held," as the Potential
            # Customer. Skip instead: an unnamed candidate cannot be written up.
            if not name or len(name.split()) > 6 or name.rstrip().endswith((",", ":")):
                continue
            if _looks_like_seller_category(name, a):
                log(f"    · skipping '{name[:40]}' — reads as a rival/supplier, not a buyer")
                continue
            if want == "buyer" and not _matches_buyer_industry(name, a):
                log(f"    · skipping '{name[:40]}' — no tie to a curated buyer vertical")
                continue
            chosen = (a, name); break
        if chosen:
            break
    peers = [(cands[i], name) for i, (name, r) in rel.items() if r == "peer"]
    if not chosen:
        return None, None, peers
    art, company = chosen
    log(f"    · free-research: {company}")
    enr = free_research.enrich_company(company, config.get_country(getattr(profile, "market", "Global")))
    card = _write_card(art, company, "customer", enr, profile)
    card["_sources"] = enr.get("sources_used", [])
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
        f"You are a sales-intelligence editor for {profile.company} ({profile.short}), which sells "
        f"{profile.value_prop} Score how well each news signal actually connects to what the seller "
        "specifically sells -- not just its broad industry. A story about the seller's exact niche is a 'high' "
        "fit; a story that only shares a broad vertical label (e.g. any hospital news for a therapy-specific "
        "seller) with no real connection to the specific product/segment is 'low', even if it looks on-topic."
    )
    user = (
        f"SIGNALS:\n{_cands_block(top)}\n\n"
        "Return JSON {\"items\":[{\"index\":<int>,\"fit\":\"high|medium|low\"}]} — one entry per signal index above."
    )
    data = llm.chat_json(system, user, max_tokens=400, default={})
    fit: dict[int, str] = {}
    for it in (data.get("items") or []):
        i = it.get("index")
        if isinstance(i, int) and 0 <= i < len(top):
            f = (it.get("fit") or "medium").strip().lower()
            fit[i] = f if f in ("high", "medium", "low") else "medium"
    if not fit:
        return cands  # triage failed/degraded -- fall back to composite order, don't drop candidates
    order = {"high": 0, "medium": 1, "low": 2}
    ranked = sorted(range(len(top)), key=lambda i: order.get(fit.get(i, "medium"), 1))
    return [top[i] for i in ranked] + cands[len(top):]


# ── 2. COMPETITION OUTLOOK — best move among PEERS (from triggers) + competitor bucket ─
def build_competition(competitor_arts: list[Article], peer_arts, profile, log=print):
    # combine peers surfaced in the trigger pool (e.g. a rival's raise) with the
    # dedicated competitor-name bucket, then take the most newsworthy (highest score).
    named = [c for c in (profile.competitors or []) if c.strip()]

    def _base(s: str) -> str:
        """'Dr. Reddy's Laboratories (CAR-T)' -> 'dr reddys laboratories'."""
        s = re.sub(r"\s*[(\[][^)\]]*[)\]]", " ", s or "").lower()
        s = re.sub(r"[^a-z0-9 ]", "", s)
        return re.sub(r"\s+", " ", s).strip()

    def _matches(name: str, text: str) -> bool:
        """True if the rival's distinctive first words appear in the text."""
        b, tx = _base(name), _base(text)
        if not b or not tx:
            return False
        if b in tx:
            return True
        head = " ".join(b.split()[:2])          # "dr reddys", "intas pharmaceuticals"
        return len(head) > 5 and head in tx

    def _is_named_rival(n: str) -> bool:
        """Only a company on the profile's competitor list counts.

        The Competition Outlook must never feature a company that could be a
        potential customer. Peers surfaced in the trigger pool are drawn from the
        BUYER verticals, so an unvetted name there is usually a prospect (a large
        pharma group next to a cell-therapy developer, say) rather than a rival.
        """
        return any(_matches(c, n) or _matches(n, c) for c in named)

    combined: list[tuple[Article, str]] = [(a, n) for (a, n) in peer_arts if _is_named_rival(n)]
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
        if not name:
            continue  # unattributable -> could be a buyer; drop rather than guess
        combined.append((a, name))
    if not combined:
        return None, None

    # Rank on RELEVANCE to the seller's product category first, newsworthiness second.
    # Sorting on composite alone hands the section to whichever rival generates the most
    # news, not the one making the most relevant move: a Novartis generics acquisition
    # outranked every genuine cell-therapy story, so the Competition Outlook for a CAR-T
    # maker was about blood-pressure brands. A diversified major is a real competitor
    # (Novartis does sell Kymriah) but only when the SIGNAL is about the shared category.
    import tenant as _tenant
    _ov = _tenant._overrides_for(getattr(profile, "company", ""))
    product_terms = [t.lower() for t in (_ov.get("product_terms") or [])]
    if not product_terms:
        # Fall back to the distinctive nouns in the seller's own value_prop.
        stop = {"india", "indian", "sells", "makes", "first", "approved", "supplies",
                "company", "their", "which", "through", "these", "other"}
        product_terms = [w for w in re.findall(r"[a-z][a-z0-9-]{4,}",
                                               (profile.value_prop or "").lower())
                         if w not in stop][:12]

    def _on_category(a: Article) -> int:
        hay = f"{a.title} {a.summary[:300]}".lower()
        return 1 if any(t in hay for t in product_terms) else 0

    on_cat = [(a, n) for (a, n) in combined if _on_category(a)]
    pool = on_cat or combined
    if on_cat:
        log(f"    · {len(on_cat)}/{len(combined)} competitor signals are on-category")
    else:
        log("    · no competitor signal mentions the seller's product category; "
            "using the most newsworthy rival move instead")
    pool.sort(key=lambda x: x[0].composite, reverse=True)
    art, comp = pool[0]

    log(f"    · free-research: {comp}")
    enr = free_research.enrich_company(comp, config.get_country(getattr(profile, "market", "Global")))
    card = _write_card(art, comp, "competitor", enr, profile)
    card["_sources"] = enr.get("sources_used", [])
    return card, art


# ── Shared card writer (customer or competitor) ──────────────────────────────
def _write_card(article: Article, name: str, kind: str, enr: dict, profile) -> dict:
    a = article
    ind = _industry_of(a, profile)
    ctx = free_research.context_block(enr)
    fr = f"\nFREE RESEARCH (Wikipedia / recent company news):\n{ctx}\n" if ctx else ""
    src = (f"SIGNAL:\n TITLE: {a.title}\n SOURCE: {a.source} | DATE: {_date_label(a.published)}\n"
           f" SUMMARY: {a.summary[:400]}\n")

    if kind == "competitor":
        system = (
            f"You are a competitive-intelligence analyst for {profile.company} ({profile.short}), which sells "
            f"{profile.value_prop} Write a competitor brief on {name}. Ground every claim in the SIGNAL and FREE "
            "RESEARCH; do NOT invent numbers. No angle-bracket placeholders.\n\n" + voice.active_style()
        )
        user = (
            f"{src}{fr}\nReturn a JSON object with keys:\n"
            f"  narrative_html — 3 to 5 SHORT sentences (max 24 words each): what {name} did, the figures if stated, and what it means "
            f"competitively for {profile.short}. Inline <b> allowed.\n"
            "  why_it_matters — one sentence on why this competitor move matters now\n"
            "  their_edge — one phrase: where this competitor is strong / what they're leaning on\n"
            f"  recommended_response — 1 to 2 sentences: how {profile.short} should respond\n"
            "  counter_play — array of exactly 3 objects with 'title' (imperative) and 'body_html' (one sentence)"
        )
        data = llm.chat_json(system, user, max_tokens=1000, default={})
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
        }

    # customer
    system = (
        f"You are CoherentConnect, a sales-intelligence analyst for {profile.company} ({profile.short}), which "
        f"sells {profile.value_prop} Write a prospect brief for {name}. Ground every claim in the SIGNAL and FREE "
        "RESEARCH; do NOT invent numbers. No angle-bracket placeholders.\n\n" + voice.active_style()
    )
    user = (
        f"{src}{fr}\nReturn a JSON object with keys:\n"
        f"  narrative_html — 3 to 5 SHORT sentences (max 24 words each): what happened, figures if stated, company context from the free "
        f"research, and why it creates a buying need for {profile.short}. Inline <b> allowed.\n"
        "  why_now — one sentence on why the buying window is open now\n"
        "  who_to_target — 2 to 3 buyer roles\n"
        f"  recommended_action — 1 to 2 sentences: the concrete first outreach move for {profile.short}\n"
        "  approach_plan — array of exactly 3 objects with 'title' (imperative) and 'body_html' (one sentence)"
    )
    data = llm.chat_json(system, user, max_tokens=1000, default={})
    steps = _steps(data.get("approach_plan"), profile, "customer")
    return {
        "flag": f"Potential Customer · {ind}",
        "company_name": name,
        "source_url": a.url,
        "source_name": a.source,
        "narrative_html": _clean_frag(data.get("narrative_html") or "") or a.summary or a.title,
        "left": {"h": "Why now", "v": _clean_frag(data.get("why_now") or "") or "The buying window is open now."},
        "right": {"h": "Who to target", "v": _clean_frag(data.get("who_to_target") or "") or "Head of Corporate Development, VP of Strategy."},
        "action": {"h": "Recommended action", "v": _clean_frag(data.get("recommended_action") or "")},
        "plan": {"h": "Approach plan", "steps": steps},
    }


def _steps(raw, profile, kind: str) -> list[dict]:
    out = []
    for step in (raw or [])[:3]:
        if isinstance(step, dict) and step.get("title"):
            out.append({"title": _clean_frag(step["title"]), "body_html": _clean_frag(step.get("body_html", ""))})
    return out or _fallback_plan(profile, kind)


# ── 3. INDUSTRY OUTLOOK — synthesize the industry news we track ───────────────
def synthesize_news(arts: list[Article], profile) -> list[dict]:
    if not arts:
        return []
    system = (
        f"You are a sales-intelligence editor for {profile.company} ({profile.short}). For each signal, write "
        "3 to 5 SHORT sentences (max 24 words each) about the industry shift and what it means for the "
        f"seller. End each with the 'so what' for {profile.short}. Ground in the signal; no invented figures. "
        "\n\n" + voice.active_style()
    )
    user = (
        f"SIGNALS:\n{_cands_block(arts)}\n\n"
        "Return a JSON object {\"items\":[ ... ]} with one entry per signal index above. Each entry has keys: "
        "'index' (integer), 'headline' (punchy, faithful, plain text no angle brackets), and 'body_html' "
        "(the paragraph, inline <b>/<i> allowed). Finished prose, never placeholders."
    )
    data = llm.chat_json(system, user, max_tokens=1300, default={})
    items = data.get("items") if isinstance(data, dict) else None
    out: list[dict] = []
    if items:
        for it in items:
            i = it.get("index")
            if not isinstance(i, int) or i < 0 or i >= len(arts):
                continue
            a = arts[i]
            out.append({"headline": _clean_frag(it.get("headline") or "") or a.title,
                        "body_html": _clean_frag(it.get("body_html") or "") or a.summary or a.title,
                        "source": a.source, "source_url": a.url, "source_name": a.source,
                        "date": _date_label(a.published), "verified": True})
    if not out:
        for a in arts:
            out.append({"headline": a.title, "body_html": a.summary or a.title,
                        "source": a.source, "source_url": a.url, "source_name": a.source,
                        "date": _date_label(a.published), "verified": True})
    return out[:1]   # Industry Outlook carries a single signal


# ── Assemble ─────────────────────────────────────────────────────────────────
def build_brief(scored: list[Article], profile, log=print, *,
                market_topic: str = "") -> tuple[dict, dict]:
    now = datetime.now(timezone.utc)

    triggers = [a for a in scored if a.bucket == "trigger" and a.trigger_type]
    industry = [a for a in scored if a.bucket == "industry"] or [a for a in scored if a.bucket == "trigger" and not a.trigger_type]
    competitor = [a for a in scored if a.bucket == "competitor"]
    log(f"  selecting: {len(triggers)} trigger / {len(industry)} industry / {len(competitor)} competitor eligible")

    log("  → [1] Potential Customer: triage buyers/peers → free-enrich → write")
    customer, cust_art, peer_arts = build_customer(triggers, profile, log=log)
    # peers (rivals surfaced in the trigger pool) feed Competition Outlook, minus the customer pick
    peer_arts = [(a, n) for (a, n) in peer_arts if cust_art is None or a.title != cust_art.title]

    # Market Moves runs FIRST on report briefs so Competition Outlook can be told what
    # it already used. The two sections both surface companies active in this market, so
    # without the exclusion the same acquisition appears twice under different headings
    # -- and the reader is told a company is "a move to watch" and "your competitor" in
    # the same brief.
    moves = moves_art = None
    if market_topic:
        log("  → [2] Market Moves: capital moving in the reader's market")
        moves_pool = [a for a in (industry + competitor)
                      if cust_art is None or a.title != cust_art.title]
        moves, moves_art = build_market_moves(moves_pool, profile, market_topic, log=log)

    log("  → [3] Competition Outlook: peers + competitor bucket → free-enrich → write")
    comp_pool = competitor
    if moves_art is not None:
        comp_pool = [a for a in competitor if a.title != moves_art.title]
        peer_arts = [(a, n) for (a, n) in peer_arts if a.title != moves_art.title]
    competition, comp_art = build_competition(comp_pool, peer_arts, profile, log=log)
    # The same COMPANY must not hold both slots either, not just the same article.
    if moves and competition and _base_name(moves.get("company_name")) and             _base_name(moves["company_name"]) == _base_name(competition.get("company_name")):
        log("    · dropping Market Moves — same company as Competition Outlook")
        moves, moves_art = None, None

    used = {a.title for a in (cust_art, comp_art, moves_art) if a}
    industry_pool = [a for a in industry if a.title not in used]
    theme, themed = cluster.dominant_theme(industry_pool)
    # One industry signal only: the strongest of the dominant theme. More than one
    # dilutes the section and repeats the same analyst read.
    fit_pool = triage_industry_fit(themed or industry_pool, profile)
    industry_sel = fit_pool[:1]
    if theme:
        log(f"  → [3] Industry Outlook: TF-IDF theme = '{theme}' ({len(themed)} signals)")
    log("  → [3] Industry Outlook: synthesize news")
    industry_news = synthesize_news(industry_sel, profile)

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
        # 2. Market Moves (report briefs only; None on signal briefs)
        "market_moves": moves,
        # 3. Competition Outlook
        "competition_outlook": competition,
        # 3. Industry Outlook
        "industry_outlook": industry_news,
        # closing — the "Inside CoherentConnect" showcase is static in the template
        "footer": {"assembled_note": "Assembled & verified by CoherentConnect", "cta_text": "Book a Demo →",
                   "cta_url": DEMO_URL, "site_text": "Visit the Site →", "site_url": SITE_URL,
                   "tagline": profile.tagline, "subline": "Your autonomous sales intelligence agent",
                   "unsubscribe_url": "#"},
    }
    debug = {
        "customer": {"company": customer["company_name"], "article": cust_art.title, "score": cust_art.composite,
                     "enriched_via": customer.get("_sources", [])} if customer else None,
        "competition": {"company": competition["company_name"], "article": comp_art.title, "score": comp_art.composite,
                        "enriched_via": competition.get("_sources", [])} if competition else None,
        "industry_theme": theme,
        "counts": {"trigger": len(triggers), "industry": len(industry_news), "competitor": len(competitor)},
    }
    return brief, debug
