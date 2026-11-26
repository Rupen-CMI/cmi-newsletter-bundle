"""Rival presence check: is a competitor active in the lead's country?

geo.filter_for_lead drops an article that names no country; for the competitor bucket
that also dropped global rival news (a major maker's results) that matters to a lead
whose market the rival sells in. A no-country competitor article is therefore KEPT when
the rival is active in the lead's country.

Activity = a Google News search "<rival> <country>" over the last 12 months (English
edition of the lead's country, plus the local-language edition with the rival's local
name) finds >= MIN_HITS distinct articles that name the rival (any written form, see
tenant.rival_forms) AND the country (regions.article_countries, or being returned by the
local edition). Results are cached in the rival_presence table for 90 days
(db.get_rival_presence / put_rival_presence); without a DB connection they are cached in
memory for the process.
"""
from __future__ import annotations
import threading
from datetime import datetime, timedelta, timezone

import config
import regions
import tenant

MIN_HITS = 2
LOOKBACK_DAYS = 365

_LOCK = threading.Lock()
_MEM: dict[tuple[str, str], tuple[bool, int]] = {}     # (rival_lower, ISO2) -> (active, hits)


def attributed_rival(article, profile) -> str:
    """The stored competitor name whose any written form appears in the article's title
    or summary ('' when none does -- e.g. a category-search article)."""
    for c in (getattr(profile, "competitors", None) or []):
        if not (c or "").strip():
            continue
        if (tenant.rival_named_in(profile, c, article.title)
                or tenant.rival_named_in(profile, c, article.summary or "")):
            return c
    return ""


def _fetch(query: str, locale: tuple):
    import collect
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    return collect._fetch_one(query, "competitor", cutoff, tuple(locale))


def count_presence(profile, rival: str, country: dict, lead_iso: str, *, fetch=None) -> int:
    """Distinct articles (by URL) that name the rival and the country. `fetch(query,
    locale) -> list[Article]` is injectable for tests."""
    fetch = fetch or _fetch
    forms = tenant.rival_forms(profile, rival)
    cname = (country.get("name") or regions.name_of(lead_iso) or "").strip()
    seen: set[str] = set()

    def names_rival(a) -> bool:
        text = f"{a.title} {a.summary or ''}"
        return any(tenant.name_matches(f, text, strict_short=(i > 0))
                   for i, f in enumerate(forms))

    # English edition of the lead's country: the article itself must name the country.
    locale = country.get("gn") or config.GN_DEFAULT_LOCALE
    try:
        for a in fetch(f"{rival} {cname}".strip(), locale) or []:
            if names_rival(a) and lead_iso in regions.article_countries(a):
                seen.add(a.url or a.title)
    except Exception:  # noqa: BLE001 -- a failed fetch is just no evidence
        pass
    # Local-language edition: returning from the country's own edition counts as the
    # country; the rival's local name (or the stored name where it is unchanged) is searched.
    local_edition = (getattr(config, "LOCAL_NEWS_EDITIONS", None) or {}).get(cname)
    if local_edition:
        local_name = ((getattr(profile, "local_competitor_names", None) or {}).get(rival)
                      or rival)
        try:
            for a in fetch(str(local_name).strip(), local_edition) or []:
                if names_rival(a):
                    seen.add(a.url or a.title)
        except Exception:  # noqa: BLE001
            pass
    return len(seen)


def rival_active(profile, rival: str, country: dict, lead_iso: str, *,
                 conn=None, fetch=None) -> bool:
    """Cached activity check for (rival, lead country). See module docstring."""
    key = ((rival or "").strip().lower(), (lead_iso or "").upper())
    if not key[0] or not key[1]:
        return False
    with _LOCK:
        hit = _MEM.get(key)
    if hit is not None:
        return hit[0]
    if conn is not None:
        try:
            import db as _db
            row = _db.get_rival_presence(conn, rival, lead_iso)
        except Exception:  # noqa: BLE001
            row = None
        if row is not None:
            with _LOCK:
                _MEM[key] = (row["active"], row["hits"])
            return row["active"]
    hits = count_presence(profile, rival, country, lead_iso, fetch=fetch)
    active = hits >= MIN_HITS
    with _LOCK:
        _MEM[key] = (active, hits)
    if conn is not None:
        try:
            import db as _db
            _db.put_rival_presence(conn, rival, lead_iso, active, hits)
        except Exception:  # noqa: BLE001
            pass
    return active


def reset_memory() -> None:
    with _LOCK:
        _MEM.clear()
