"""Country geo-matching — hard-filter collected articles to the target country.

Compact port of the mother repo's CountryProfile geo logic: an article belongs to the
country if its text/source/URL carries a country signal (name, nationality, a major city,
a country-specific currency token, a known local outlet, or a country TLD).
"""
from __future__ import annotations
from collect import Article

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
