"""Country geo-matching — hard-filter collected articles to the target country.

Compact port of the mother repo's CountryProfile geo logic: an article belongs to the
country if its text/source/URL carries a country signal (name, nationality, a major city,
a country-specific currency token, a known local outlet, or a country TLD).
"""
from __future__ import annotations
import re
from collect import Article

import regions

MIN_KEEP = 6   # below this, fall back to unfiltered so the brief isn't empty


def is_global(country: dict) -> bool:
    return not country or country.get("name") == "Global"


def country_signals(country: dict) -> list[str]:
    toks = []
    toks += [a.lower() for a in country.get("aliases", [])]
    if country.get("nationality"):
        toks.append(country["nationality"].lower())
    toks += [c.lower() for c in country.get("cities", [])]
    toks += [c.lower() for c in country.get("currency", [])]
    toks += [o.lower() for o in country.get("outlets", [])]
    return [t for t in toks if t]


def matches(article: Article, country: dict) -> bool:
    if is_global(country):
        return True
    hay = f"{article.title} {article.summary} {article.source}".lower()
    for tok in country_signals(country):
        if tok in hay:
            return True
    low_url = (article.url or "").lower()
    low_src = (article.source or "").lower()
    for tld in country.get("tld", []):
        if tld in low_url or low_src.endswith(tld):
            return True
    return False


def filter_country(articles: list[Article], country: dict) -> tuple[list[Article], bool]:
    """Return (kept, used_fallback). Strict country match; if too few match, fall back
    to the full set (a thin real newsletter is worse than an off-country dilution here)."""
    if is_global(country):
        return articles, False
    kept = [a for a in articles if matches(a, country)]
    if len(kept) < MIN_KEEP:
        return articles, True   # not enough country-specific signal today → don't blank the brief
    return kept, False


def filter_for_lead(articles: list[Article], country: dict, *,
                    region_fallback_min: int = 4) -> tuple[list[Article], str]:
    """Country-first evidence filter for one lead's brief. Returns (kept, scope)
    where scope is "country" or "region".

    Rules, replacing filter_country's fail-open behaviour for the lead pipeline:

      - IN-COUNTRY: the article mentions the lead's country (regions.article_countries:
        title/summary/source text or the URL's ccTLD), or its source is one of the
        country's curated outlets. Keep.
      - OTHER-COUNTRY: mentions other countries but not the lead's. DROP — the
        explicit-link rule: a foreign event is allowed only when the article itself
        names the lead's country.
      - NO-GEO: no country detected anywhere. Keep ONLY for the 'industry' bucket
        AND only when the article still points at the lead's country some other
        way: the search `query` that found it carries the country's name or an
        alias/demonym (collect() appends the country name to every query;
        collect_topic's anchored queries carry the name or the demonym), or its
        source is one of the country's curated outlets, or its URL has the
        country's ccTLD. An un-anchored no-geo industry article — the no-geo
        'industry'/topic leak the 2026-11 live run showed (Korea's IGIS stem-cell
        JV reached a Morocco lead; a Korea/US autonomous-vessel story reached an
        India lead) — is dropped.
      - REGION FALLBACK: if fewer than region_fallback_min trigger+competitor
        articles survive, the dropped other-country articles whose countries ALL
        sit inside the lead's REGION may be added back (scope="region"). Never
        crosses into another region and NEVER falls back to the unfiltered pool.

    A Global lead (country == "Global") is unfiltered, as before. Countries outside
    the regions table (should not happen for real leads) keep the legacy fail-open
    behaviour rather than blanking the brief.
    """
    if is_global(country):
        return list(articles), "country"
    prof = regions.get_profile(country.get("iso2") or country.get("name") or "")
    if prof is None:
        return filter_country(articles, country)
    lead_iso = prof["iso2"]
    region_isos = set(regions.countries_in(prof["region"]))
    outlets = [o.lower() for o in (country.get("outlets") or [])]
    # Anchor tokens for the no-geo rule: the query string that found the article
    # must itself name the lead country. Aliases carry the demonym for the
    # curated countries ("indian"); the regions profile adds its own. Matched
    # with word boundaries — a plain substring check would let "desi" match
    # "design" or "uk" match "ukraine".
    anchor_toks: list[str] = []
    for t in ([country.get("name", "")] + list(country.get("aliases", []))
              + [country.get("nationality", ""), prof["name"]]
              + prof["aliases"] + [prof["nationality"]]):
        t = (t or "").strip().lower()
        if t and t not in anchor_toks:
            anchor_toks.append(t)
    anchor_rx = re.compile(
        "(?<![0-9a-z])(?:" + "|".join(re.escape(t) for t in sorted(anchor_toks, key=len, reverse=True))
        + ")(?![0-9a-z])") if anchor_toks else None
    tlds = [t.lower() for t in (country.get("tld") or [])]

    kept: list[Article] = []
    foreign: list[Article] = []
    for a in articles:
        cset = regions.article_countries(a)
        src = (a.source or "").lower()
        if lead_iso in cset or (src and any(o in src for o in outlets)):
            kept.append(a)                      # in-country evidence
        elif cset:
            foreign.append(a)                   # other-country evidence: dropped
        elif getattr(a, "bucket", "") == "industry":
            q = (getattr(a, "query", "") or "").lower()
            url = (a.url or "").lower()
            if ((anchor_rx and anchor_rx.search(q))
                    or (src and any(o in src for o in outlets))
                    or any(t in url for t in tlds)):
                kept.append(a)                  # no-geo but country-anchored
            # else: un-anchored no-geo industry/topic article — dropped

    def _tc(items: list[Article]) -> int:
        return sum(1 for a in items
                   if getattr(a, "bucket", "") in ("trigger", "competitor"))

    scope = "country"
    if _tc(kept) < region_fallback_min and region_isos:
        regional = [a for a in foreign
                    if regions.article_countries(a) <= region_isos]
        if regional:
            kept.extend(regional)
            scope = "region"
    return kept, scope
