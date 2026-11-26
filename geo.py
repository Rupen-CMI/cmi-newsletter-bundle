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
                    region_fallback_min: int = 4, profile=None, conn=None,
                    presence_fetch=None) -> tuple[list[Article], str]:
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
        collect_topic's anchored queries carry the name or the demonym) AND names
        no other country (a "food safety standards India South Korea" query is
        about India; collect()'s suffix is not lead evidence), or its source is
        one of the country's curated outlets, or its URL has the country's ccTLD.
        An un-anchored no-geo industry article — the no-geo 'industry'/topic leak
        the 2026-11 live run showed (Korea's IGIS stem-cell JV reached a Morocco
        lead; a Korea/US autonomous-vessel story reached an India lead) — is
        dropped.
      - SOURCE COUNTRY: an article with no detected country whose outlet maps to a
        country (source_country.py, learned from collected articles) is treated as
        about that country: in-country when it is the lead's, otherwise other-country.
      - NO-GEO COMPETITOR (needs `profile`): a competitor-bucket article naming no
        country is KEPT when the rival it names (any written form, see
        tenant.rival_forms) is active in the lead's country (rivals.rival_active:
        >= 2 articles in 12 months naming both). An article naming ANOTHER country
        stays dropped. `conn` / `presence_fetch` feed that check's cache and fetcher.
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

    import source_country
    import rivals
    src_map = source_country.learned_map(conn)

    kept: list[Article] = []
    foreign: list[Article] = []
    foreign_isos: dict[int, set[str]] = {}   # id(article) -> effective country set
    for a in articles:
        cset = regions.article_countries(a)
        src = (a.source or "").lower()
        if not cset and src in src_map:
            cset = {src_map[src]}               # outlet belongs to a known country
        if lead_iso in cset or (src and any(o in src for o in outlets)):
            kept.append(a)                      # in-country evidence
        elif cset:
            foreign.append(a)                   # other-country evidence: dropped
            foreign_isos[id(a)] = cset
        elif getattr(a, "bucket", "") == "competitor":
            # No country anywhere: keep only if the rival it names operates in the lead's
            # country. No profile (or no rival named) -> dropped, as before.
            rival = rivals.attributed_rival(a, profile) if profile is not None else ""
            if rival and rivals.rival_active(profile, rival, country, lead_iso,
                                             conn=conn, fetch=presence_fetch):
                kept.append(a)
        elif getattr(a, "bucket", "") == "industry":
            q = (getattr(a, "query", "") or "").lower()
            url = (a.url or "").lower()
            # The query anchors the article to the lead only if it names the lead
            # country AND names no OTHER country. collect() appends the lead's country
            # to every query, so "food safety standards India" searched for a South
            # Korean lead becomes "... India South Korea" and DOES carry the Korean
            # anchor -- but the query is really about India, and the Korean evidence is
            # only the suffix collect() added. Treating that as anchored is how an
            # FSSAI/Bisleri prosecution (no country in its headline) was kept for a
            # Korean lead. A query naming another country is not lead evidence.
            q_isos = regions.countries_in_text(q) if q else set()
            names_lead = lead_iso in q_isos or bool(anchor_rx and anchor_rx.search(q))
            names_other = bool(q_isos - {lead_iso})
            if ((names_lead and not names_other)
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
                    if foreign_isos.get(id(a), set()) <= region_isos]
        if regional:
            kept.extend(regional)
            scope = "region"
    return kept, scope


def filter_topic_for_lead(topic_arts: list[Article], country: dict, tokens: list[str], *,
                          min_on_topic: int = 2, profile=None, conn=None
                          ) -> tuple[list[Article], str]:
    """Geo filter for TOPIC news (report briefs), with a widening fallback so a niche
    market does not end with zero on-topic evidence just because nothing names the
    lead's country. Returns (kept, scope); scope is the widest tier used.

      country: filter_for_lead as usual.  region: + on-topic articles whose countries all
      sit in the lead's region.  global: + every other on-topic article.
    Tiers 2-3 only ever add ON-TOPIC articles; each added article carries geo_scope."""
    import score
    if is_global(country):
        return list(topic_arts), "country"
    # region_fallback_min=0: filter_for_lead's own region add-back is topic-blind
    kept, _ = filter_for_lead(topic_arts, country, profile=profile, conn=conn,
                              region_fallback_min=0)
    for a in kept:
        a.geo_scope = "country"

    def _on(a: Article) -> bool:
        return score.topic_relevance(a, tokens) >= score.TOPICAL

    if sum(1 for a in kept if _on(a)) >= min_on_topic:
        return kept, "country"
    kept_ids = {id(a) for a in kept}
    rest = [a for a in topic_arts if id(a) not in kept_ids and _on(a)]
    prof = regions.get_profile(country.get("iso2") or country.get("name") or "")
    region_isos = set(regions.countries_in(prof["region"])) if prof else set()
    added: set[int] = set()
    for a in rest:
        cset = regions.article_countries(a)
        if cset and cset <= region_isos:
            a.geo_scope = "region"
            kept.append(a)
            added.add(id(a))
    if sum(1 for a in kept if _on(a)) >= min_on_topic:
        return kept, "region"
    far = [a for a in rest if id(a) not in added]
    for a in far:
        a.geo_scope = "global"
        kept.append(a)
    # scope = widest tier that actually contributed an article
    return kept, "global" if far else ("region" if added else "country")
