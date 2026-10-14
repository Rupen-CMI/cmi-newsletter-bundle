"""The scoring matrix: composite = relevancy x freshness x importance (weighted sum).

Each candidate signal is scored on three independent axes, then combined. This is the
gate that decides which trigger signals earn a place in the newsletter — the explicit
ask. All three sub-scores are in [0,1]; composite is a weighted sum in [0,1].
"""
from __future__ import annotations
import re
from datetime import datetime, timezone

import config
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


def is_unsafe_to_cite(a: Article) -> bool:
    """Ticker noise or a rival research firm's own report PR -- never safe to show a
    reader regardless of composite score, unlike an ordinary low-relevancy article.
    Exposed so a caller falling back to the unfiltered pool (pipeline.stage_generate's
    `keep or scored`, for a thin-coverage lead) can still exclude these two."""
    return _is_ticker_noise(a) or config.is_research_pr(a.source, a.title)


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
def importance(a: Article) -> tuple[float, str]:
    t = _text(a)
    best_type, best_imp = "", 0.0
    for ttype, spec in _TRIG.items():
        if any(kw in t for kw in spec["keywords"]):
            if spec["importance"] > best_imp:
                best_imp, best_type = spec["importance"], ttype
    if not best_type:
        # structural/industry item with no company trigger → low but non-zero
        return 0.30, ""
    # money-scale amplifier
    if any(h in t for h in config.BIG_MONEY_HINTS):
        best_imp = min(1.0, best_imp + 0.05)
    return round(best_imp, 3), best_type


# ── Composite ────────────────────────────────────────────────────────────────
def score_all(articles: list[Article], profile) -> list[Article]:
    for a in articles:
        rel, inds = relevancy(a, profile.industry_keywords)
        # Bucket-based relevancy floor: a trigger/industry article already matched a
        # seller-SCOPED query, so it is topically relevant even if the keyword lexicon
        # (LLM-inferred, sometimes narrow) missed it. Prevents broad-market sellers from
        # losing every prospect to relevancy=0.
        if a.bucket == "trigger":
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
    return articles


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
    return [a for a in articles if a.composite >= _S["min_composite"] and not is_unsafe_to_cite(a)]
