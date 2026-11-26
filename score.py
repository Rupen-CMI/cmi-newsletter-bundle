"""The scoring matrix: composite = relevancy x freshness x importance (weighted sum).

Each candidate signal is scored on three independent axes, then combined. This is the
gate that decides which trigger signals earn a place in the newsletter — the explicit
ask. All three sub-scores are in [0,1]; composite is a weighted sum in [0,1].
"""
from __future__ import annotations
import re
from datetime import datetime, timezone

import config
import relevance
from collect import Article

_TRIG = config.TRIGGER_TYPES
_S = config.SCORING

# Stock-ticker churn ("Novartis India Share Price Today: 6.19% Fall") is not a
# competitive move -- it is market noise that happens to share a competitor's
# name, and it slips into the eligible pool purely on the competitor-name query
# match with no real trigger type. Screen it out regardless of composite score.
_TICKER_NOISE = re.compile(
    r"\bshare price\b|\bstock price\b|\b\d+(\.\d+)?%\s*(fall|surge|gain|jump|drop|rise|slip)\b"
    r"|\b(surges?|falls?|jumps?|slips?|gains?|rises?|drops?|tumbles?|rallies?)\s+\d+(\.\d+)?%"
    r"|\bday'?s high\b|\bday'?s low\b|\b52-week\b|\binvestor call\b",
    re.I,
)


def _is_ticker_noise(a: Article) -> bool:
    return bool(_TICKER_NOISE.search(f"{a.title} {a.summary}"))


# Paid "Top 10 <X> Manufacturers in 2026" listicles syndicated through local-newspaper
# sites, and video/social posts, are not reporting and make poor evidence.
_SEO_LISTICLE = re.compile(
    r"\b(top\s+\d+|\d+|five|ten|seven|three)\s+(reputable|leading|best|top|trusted|premier\s+)?"
    r"[\w\- ]{0,40}?\b(manufacturers?|suppliers?|producers?|companies|brands)\b.{0,30}\b(in|for)\s+(\w+\s+){0,2}20\d\d\b"
    r"|\b(reputable|premium|leading)\b.{0,40}\bmanufacturers?\b.{0,25}\b(in|of)\s+(china|india)\b",
    re.I)
_LOW_VALUE_SOURCES = ("youtube", "facebook", "reddit", "pinterest", "tiktok", "instagram", "quora")


def is_unsafe_to_cite(a: Article) -> bool:
    """Ticker noise or a rival research firm's own report PR -- never safe to show a
    reader regardless of composite score, unlike an ordinary low-relevancy article.
    Exposed so a caller falling back to the unfiltered pool (pipeline.stage_generate's
    `keep or scored`, for a thin-coverage lead) can still exclude these two."""
    src = (a.source or "").lower()
    return (_is_ticker_noise(a) or config.is_research_pr(a.source, a.title)
            or bool(_SEO_LISTICLE.search(a.title or "")) or any(w in src for w in _LOW_VALUE_SOURCES))


def _text(a: Article) -> str:
    return f"{a.title} {a.summary}".lower()


# ── 1. RELEVANCY — does this map onto the seller's coverage verticals? ───────
def relevancy(a: Article, industry_keywords: dict) -> tuple[float, list[str]]:
    t = _text(a)
    matched: list[str] = []
    hits = 0
    for industry, kws in industry_keywords.items():
        ind_hit = sum(1 for kw in kws if kw in t)
        if ind_hit:
            matched.append(industry)
            hits += ind_hit
    # saturating: 0 hits → 0.0, 1 → ~0.45, 3 → ~0.75, 6+ → ~0.95
    score = 1.0 - 0.6 ** hits if hits else 0.0
    # a signal touching 2+ distinct verticals is broadly relevant → small boost
    if len(matched) >= 2:
        score = min(1.0, score + 0.10)
    return round(score, 3), matched


# Competitor decay: roughly a month at full weight, then a slow taper over six
# months. Deliberately gentler than the trigger bands -- see freshness().
_COMPETITOR_BANDS = [(24 * 30, 1.00), (24 * 60, 0.85), (24 * 90, 0.70),
                     (24 * 120, 0.55), (24 * 182, 0.40)]

# Trigger decay, now that the eligibility window is six months rather than four days.
# config.SCORING's bands bottom out at 168h/0.10, so once everything older than a week
# scored the same floor a 14-day-old funding round and a 172-day-old one ranked
# identically -- and a buying trigger genuinely does go cold. These keep the steep
# early decay (a fresh signal still wins) but continue separating older items instead
# of collapsing them.
_TRIGGER_BANDS = [(24, 1.00), (48, 0.85), (72, 0.65), (96, 0.45), (168, 0.30),
                  (24 * 14, 0.22), (24 * 30, 0.16), (24 * 60, 0.10),
                  (24 * 120, 0.06)]


# ── 2. FRESHNESS — recency decay (piecewise, from config bands) ──────────────
def freshness(a: Article) -> float:
    hours = (datetime.now(timezone.utc) - a.published).total_seconds() / 3600.0

    # Competitor signals decay on their own, far slower schedule. The default bands
    # bottom out at a week, which is right for a buying trigger -- a funding round
    # stops being actionable quickly. A rival's product move does not: it stays true
    # until they make the next one. Scoring the two on the same curve pushed every
    # genuine competitor signal in a thin category below min_composite, so the section
    # could only ever show whatever unrelated news a large rival made this week.
    if a.bucket == "competitor":
        for band_h, factor in _COMPETITOR_BANDS:
            if hours <= band_h:
                return factor
        return 0.30

    if a.bucket == "trigger":
        for band_h, factor in _TRIGGER_BANDS:
            if hours <= band_h:
                return factor
        return 0.03

    for band_h, factor in _S["freshness_bands"]:
        if hours <= band_h:
            return factor
    return _S["freshness_floor"]


# ── 3. IMPORTANCE — strength of the trigger event (+ money-scale boost) ──────
def _kw_regex(kw: str) -> "re.Pattern":
    """Whole-word match (plus plain suffixes). The old test was `kw in text`, so the
    European Medicines Agency keyword "ema" matched inside "d-ema-nd": 350 of its 351
    hits in 12,862 real articles were accidents, each one scored as a regulatory
    approval (importance 0.85). That alone floated "...Demand Drive Growth" press
    releases to the top of unrelated leads' signal lists."""
    pre = r"(?<![a-z0-9])"
    post = r"(?:s|es|ed|d)?(?![a-z0-9])" if kw[-1].isalnum() else ""
    return re.compile(pre + re.escape(kw) + post)


_TRIG_RE = {t: [_kw_regex(k) for k in spec["keywords"]] for t, spec in _TRIG.items()}


# "raises" / "raised" / "backed by" / "led by" only mean FUNDING when money is in the
# sentence. "Bank of England raises rates" and "Sequoia closes London office" are not
# buying triggers, yet scored 0.95 because the bare verb matched.
_WEAK_FUNDING = {"raises", "raised", "backed by", "led by", "closes $", "secures $"}
_MONEY = re.compile(r"[$€£₹]|\d[\d.,]*\s?(?:million|billion|bn|crore|lakh|m|k)|series [a-f]|"
                    r"seed|funding|financing|valuation|funding round|venture")


def _type_matches(ttype: str, spec: dict, t: str) -> bool:
    hits = [kw for kw, rx in zip(spec["keywords"], _TRIG_RE[ttype]) if rx.search(t)]
    if ttype == "funding_round" and hits and all(h in _WEAK_FUNDING for h in hits):
        return bool(_MONEY.search(t))
    return bool(hits)


def importance(a: Article) -> tuple[float, str]:
    t = _text(a)
    best_type, best_imp = "", 0.0
    for ttype, spec in _TRIG.items():
        if _type_matches(ttype, spec, t):
            if spec["importance"] > best_imp:
                best_imp, best_type = spec["importance"], ttype
    if not best_type:
        # structural/industry item with no company trigger → low but non-zero
        return 0.30, ""
    # money-scale amplifier
    if any(h in t for h in config.BIG_MONEY_HINTS):
        best_imp = min(1.0, best_imp + 0.05)
    return round(best_imp, 3), best_type


# ── 4. TOPIC RELEVANCE — is it about the market the lead enquired about? ─────
# Report briefs are promised a specific market ("Stem Cells Market"). Nothing above
# measures that: relevancy() scores against the seller's industry lexicon, and the
# bucket floors make every query hit "relevant". Measured on real Campaign 6 leads, 0-1%
# of the eligible pool mentioned the topic at all.
_TOPIC_STOP = {"market", "markets", "global", "industry", "report", "analysis", "size", "share",
               "forecast", "tracker", "pricing", "price", "database", "strategy", "partnership",
               "regional", "hoc", "adhoc", "india", "study", "outlook", "trends", "trend",
               "overview", "research", "insights", "sector", "the", "and", "for", "with", "from",
               # generic nouns that match almost any industrial headline
               "manufacturing", "manufacturer", "associated", "equipment", "service", "services",
               "product", "products", "system", "systems", "technology", "technologies", "solution",
               "solutions", "company", "companies", "business", "application", "applications"}


def _stem(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
        return w[:-1]
    return w


def topic_drop_words(market: str) -> set[str]:
    """The lead country's own words ("spain", "spanish"): a topic like "Spain Data Center"
    must not demand them in the headline -- geo already handles the country."""
    import regions
    import tenant
    prof = regions.get_profile(market or "") or {}
    iso2 = prof.get("iso2") or ""
    if not iso2:
        return set()
    return {w for ph in tenant._country_words(iso2) for w in ph.lower().split()}


def topic_tokens(topic: str, drop: set[str] | None = None) -> list[str]:
    seen: list[str] = []
    for w in re.findall(r"[a-z0-9]+", (topic or "").lower()):
        if w in _TOPIC_STOP or len(w) < 3 or w.isdigit():
            continue
        s = _stem(w)
        if drop and (w in drop or s in drop):
            continue
        if s not in seen:
            seen.append(s)
    return seen


def topic_relevance(a: Article, tokens: list[str]) -> float:
    """0..1. A topic word in the headline counts fully, in the summary only half; an
    exact adjacent-words phrase in the headline adds 0.2. >=0.6 means 'about it'."""
    if not tokens:
        return 0.0
    tw = {_stem(w) for w in re.findall(r"[a-z0-9]+", (a.title or "").lower())}
    sw = {_stem(w) for w in re.findall(r"[a-z0-9]+", (a.summary or "").lower())} - tw
    hits = sum(1.0 for t in tokens if t in tw) + sum(0.5 for t in tokens if t in sw)
    rel = min(1.0, hits / len(tokens))
    if len(tokens) >= 2 and " ".join(tokens) in " ".join(_stem(w) for w in re.findall(r"[a-z0-9]+", (a.title or "").lower())):
        rel = min(1.0, rel + 0.2)
    # Two consecutive topic words adjacent in the headline ("monoclonal antibodies") are
    # on-topic on their own: 60% of a 3-4 word topic is too strict for a headline.
    hw = [_stem(w) for w in re.findall(r"[a-z0-9]+", (a.title or "").lower())]
    adj = {(hw[i], hw[i + 1]) for i in range(len(hw) - 1)}
    if any((tokens[i], tokens[i + 1]) in adj for i in range(len(tokens) - 1)):
        rel = max(rel, 0.7)
    return round(rel, 3)


TOPICAL = 0.6     # topic_rel at or above this counts as being about the topic


# ── 5. SELLER RELEVANCE (embedding) ──────────────────────────────────────────
# Cosine between the article and the seller text (relevance.py, multilingual MiniLM).
# Cosine is ~0.0-0.1 for unrelated text, ~0.3+ for same-subject text, so it is rescaled
# to 0..1 between SELLER_REL_LO and SELLER_REL_HI before it feeds the relevancy axis.
SELLER_REL_LO = 0.0
SELLER_REL_HI = 0.30
# eligible() drops an article only when it is BOTH in the bottom SELLER_REL_PERCENTILE of
# its pool AND below SELLER_REL_FLOOR in absolute terms: the percentile alone would always
# cut 30% of even a pool of good items, the floor alone would gut a thin-coverage pool.
SELLER_REL_PERCENTILE = 30
SELLER_REL_FLOOR = 0.10


def _scaled_seller_rel(cos: float) -> float:
    return max(0.0, min(1.0, (cos - SELLER_REL_LO) / (SELLER_REL_HI - SELLER_REL_LO)))


# ── Composite ────────────────────────────────────────────────────────────────
def score_all(articles: list[Article], profile, topic: str = "") -> list[Article]:
    """`topic` (report briefs) only ANNOTATES each article with topic_rel; it never
    changes composite or eligibility, so every other section behaves exactly as before.
    brief_builder uses topic_rel to choose the 'Your Market' evidence."""
    toks = topic_tokens(topic, topic_drop_words(getattr(profile, "market", "")))
    # Embedding similarity of every article to the seller; None per article (or overall)
    # when the local model is unavailable -> bucket floors below apply as before.
    try:
        sims = relevance.seller_relevance(articles, profile)
    except Exception:  # noqa: BLE001 -- never let the optional model break scoring
        sims = None
    for i, a in enumerate(articles):
        a.seller_rel = sims[i] if sims is not None else None
        a.topic_rel = topic_relevance(a, toks) if toks else 0.0
        rel, inds = relevancy(a, profile.industry_keywords)
        # Bucket-based relevancy floor: a trigger/industry article already matched a
        # seller-SCOPED query, so it is topically relevant even if the keyword lexicon
        # (LLM-inferred, sometimes narrow) missed it. Prevents broad-market sellers from
        # losing every prospect to relevancy=0.
        # With the embedding score the floor is REPLACED by how related the article really
        # is to the seller; the fixed floors stay as the fallback when there is no model.
        if a.seller_rel is not None:
            rel = max(rel, _scaled_seller_rel(a.seller_rel))
        elif a.bucket == "trigger":
            rel = max(rel, 0.35)
        elif a.bucket == "industry":
            rel = max(rel, 0.25)
        fr = freshness(a)
        imp, ttype = importance(a)
        a.relevancy, a.freshness, a.importance = rel, fr, imp
        a.matched_industries, a.trigger_type = inds, ttype
        a.composite = round(
            _S["w_relevancy"] * rel + _S["w_freshness"] * fr + _S["w_importance"] * imp, 4
        )
    articles.sort(key=lambda x: x.composite, reverse=True)
    try:
        import evidence_score
        evidence_score.score_evidence(articles, profile)   # annotation only
    except Exception:  # noqa: BLE001
        pass
    return articles


def passes_relevance_floor(a: Article) -> bool:
    """The relevance gate's ABSOLUTE floor, without its pool-relative percentile: True when
    the article has no embedding score (model unavailable) or its seller_rel is at or
    above SELLER_REL_FLOOR. Below it the article is unrelated to the seller."""
    sr = getattr(a, "seller_rel", None)
    return sr is None or sr >= SELLER_REL_FLOOR


def fallback_pool(scored: list[Article]) -> list[Article]:
    """What a brief is built from when eligible() returns nothing: items that are safe to
    cite AND clear the relevance floor, so unrelated news cannot return via the fallback."""
    return [a for a in scored if not is_unsafe_to_cite(a) and passes_relevance_floor(a)]


def eligible(articles: list[Article]) -> list[Article]:
    # config.is_research_pr exists specifically to keep a rival market-research firm's
    # own report PR (MarketsandMarkets, Grand View, Mordor Intelligence, ...) out of a
    # CMI newsletter -- CMI must never cite a competitor's numbers to its own client,
    # let alone build a whole "Market Moves" section around one and link to it as the
    # source. The function was written and documented but never actually called from
    # anywhere, so this kept happening in real sends (caught 2026-10-08: an American
    # Roller brief whose Market Moves section was MarketsandMarkets' own press
    # release, "Source: MarketsandMarkets ->" link included). Filtering here, in the
    # one function every section's candidate pool passes through, closes it for all
    # of them at once rather than patching each section separately.
    out = [a for a in articles if a.composite >= _S["min_composite"] and not is_unsafe_to_cite(a)]
    # Seller-relevance gate (embedding score; skipped when the model was unavailable):
    # drop what is clearly unrelated to the seller -- national headlines that merely rode
    # in on a seller-scoped query. Threshold is relative to the pool AND absolute.
    sims = [a.seller_rel for a in out if getattr(a, "seller_rel", None) is not None]
    if len(sims) >= 10:
        sims_sorted = sorted(sims)
        cut = sims_sorted[int(len(sims_sorted) * SELLER_REL_PERCENTILE / 100)]
        thr = min(cut, SELLER_REL_FLOOR)
        out = [a for a in out
               if getattr(a, "seller_rel", None) is None or a.seller_rel >= thr]
    return out
