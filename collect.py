"""News collection — Google News RSS across trigger / industry / competitor queries.

Distilled from the mother repo's `trends/collector.py`: parallel feed fetch, SHA-256
fingerprint dedup, date-window filtering. Free sources only (no paid news APIs).
"""
from __future__ import annotations
import concurrent.futures as cf
import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from urllib.parse import quote_plus

import feedparser

import config


@dataclass
class Article:
    title: str
    url: str
    source: str
    published: datetime
    summary: str = ""
    query: str = ""
    bucket: str = ""            # trigger | industry | competitor  (set later)
    fingerprint: str = field(default="", repr=False)
    # scoring fields filled by score.py
    relevancy: float = 0.0
    freshness: float = 0.0
    importance: float = 0.0
    composite: float = 0.0
    trigger_type: str = ""
    matched_industries: list = field(default_factory=list)


def _fingerprint(title: str, url: str) -> str:
    norm = re.sub(r"\s+", " ", (title or "").lower()).strip()
    return hashlib.sha256(f"{norm}|{url}".encode()).hexdigest()


_STOPWORDS = {
    "the", "a", "an", "to", "for", "in", "on", "with", "and", "of", "at", "by",
    "its", "as", "over", "amid", "after", "from", "into", "up", "down", "new",
    "news", "says", "said", "will", "has", "have", "is", "are", "be",
}


def _story_key(title: str) -> str:
    """Normalized key so the SAME event covered by different outlets (different
    headline wording, different URL) collapses to one candidate instead of
    flooding the eligible pool with near-identical copies -- e.g. 8 outlets all
    reporting the same Dr Reddy's/Takeda dengue-vaccine deal each got a distinct
    exact-title fingerprint and all 8 counted as separate signals."""
    words = re.findall(r"[a-z0-9']+", (title or "").lower())
    sig = sorted({w for w in words if w not in _STOPWORDS and len(w) > 2})
    return " ".join(sig)


def _collapse_near_duplicates(articles: list["Article"]) -> list["Article"]:
    """Keep one article per normalized story key -- the best bucket-ranked, then
    highest-composite (falls back to most recent pre-scoring) -- so a widely
    syndicated story occupies one eligibility slot, not one per outlet."""
    bucket_rank = {"trigger": 0, "competitor": 1, "industry": 2}
    best: dict[str, "Article"] = {}
    for a in articles:
        key = _story_key(a.title)
        if not key:
            continue
        cur = best.get(key)
        if cur is None:
            best[key] = a
            continue
        a_rank, cur_rank = bucket_rank.get(a.bucket, 3), bucket_rank.get(cur.bucket, 3)
        if a_rank < cur_rank or (a_rank == cur_rank and a.published > cur.published):
            best[key] = a
    return list(best.values())


def _clean(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html or "").strip()


def _parse_dt(entry) -> datetime | None:
    for attr in ("published_parsed", "updated_parsed"):
        t = getattr(entry, attr, None)
        if t:
            try:
                return datetime.fromtimestamp(time.mktime(t), tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                # Some feeds carry a nonsense year (seen: a "forecast to 2035" SEO page
                # dated past mktime's range). One bad entry must not abort the fetch.
                continue
    return None


def _source_of(entry, fallback: str) -> str:
    src = getattr(entry, "source", None)
    if src and getattr(src, "title", None):
        return src.title
    # Google News titles are usually "Headline - Publisher"
    return fallback


def _fetch_one(query: str, bucket: str, cutoff: datetime, locale: tuple) -> list[Article]:
    hl, gl, ceid = locale
    url = config.GOOGLE_NEWS.format(q=quote_plus(query), hl=hl, gl=gl, ceid=ceid)
    out: list[Article] = []
    try:
        feed = feedparser.parse(url)
    except Exception:
        return out
    for e in feed.entries:
        dt = _parse_dt(e)
        if dt is None or dt < cutoff:
            continue
        title = _clean(getattr(e, "title", ""))
        link = getattr(e, "link", "")
        if not title or not link:
            continue
        # split "Headline - Publisher"
        publisher = ""
        if " - " in title:
            head, publisher = title.rsplit(" - ", 1)
            title = head.strip()
        out.append(Article(
            title=title, url=link,
            source=_source_of(e, publisher or "Google News"),
            published=dt, summary=_clean(getattr(e, "summary", ""))[:600],
            query=query, bucket=bucket, fingerprint=_fingerprint(title, link),
        ))
    return out


def collect(profile, country: dict | None = None) -> list[Article]:
    """Fetch all queries in parallel, dedupe by fingerprint, return within window.

    `profile` carries trigger_queries / industry_queries / competitors.
    `country` (a config.COUNTRIES entry) sets the Google-News locale and appends the
    country name to each query so results are geo-biased. None/Global = worldwide English.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.LOOKBACK_HOURS)
    locale = (country or config.COUNTRIES["Global"]).get("gn", config.GN_DEFAULT_LOCALE)
    cname = (country or {}).get("name", "Global")
    suffix = "" if cname in ("", "Global") else f" {cname}"   # e.g. "biotech funding India"

    # Competitor queries pair each rival with the seller's product category where one is
    # curated. Searching a bare company name returns whatever that company is biggest in:
    # "Novartis India" surfaced a generics acquisition, so the Competition Outlook for a
    # CAR-T maker was about blood-pressure brands. Anchoring the query to the shared
    # category is what makes the returned signal comparable.
    comp_jobs = []
    try:
        import tenant as _tenant
        _terms = (_tenant._overrides_for(getattr(profile, "company", "")) or {}).get("product_terms") or []
    except Exception:  # noqa: BLE001 -- enrichment is optional, never fatal
        _terms = []
    anchor = _terms[0] if _terms else ""
    for c in profile.competitors:
        if anchor:
            comp_jobs.append((f"{c} {anchor}{suffix}", "competitor"))
        comp_jobs.append((c + suffix, "competitor"))       # plain name too, as a fallback

    jobs = ([(q + suffix, "trigger") for q in profile.trigger_queries]
            + [(q + suffix, "industry") for q in profile.industry_queries]
            + comp_jobs)

    # Competitor rows use their own, much longer cutoff (see config.COMPETITOR_LOOKBACK_HOURS).
    comp_cutoff = datetime.now(timezone.utc) - timedelta(
        hours=getattr(config, "COMPETITOR_LOOKBACK_HOURS", config.LOOKBACK_HOURS))

    # A per-company trigger window, when curated. 96 hours suits a seller whose buyers
    # make news weekly; it starves one whose buyers do not. Measured for VVF: its buyer
    # queries return 1 article in 96h and 18 in 30 days, so the Potential Customer slot
    # was either empty or stretched to an irrelevant big name. Widening the window for
    # that account beats loosening the relevance guards for every account.
    try:
        import tenant as _tenant
        _ov = _tenant._overrides_for(getattr(profile, "company", "")) or {}
        _tl = int(_ov.get("trigger_lookback_hours") or 0)
    except Exception:  # noqa: BLE001 -- enrichment is optional
        _tl = 0
    _tl = _tl or getattr(config, "TRIGGER_LOOKBACK_HOURS", config.LOOKBACK_HOURS)
    trig_cutoff = datetime.now(timezone.utc) - timedelta(hours=_tl)

    articles: list[Article] = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        def _cut(b):
            return comp_cutoff if b == "competitor" else (trig_cutoff if b == "trigger" else cutoff)

        futs = {ex.submit(_fetch_one, q, b, _cut(b), locale): (q, b) for q, b in jobs}
        for fut in cf.as_completed(futs):
            try:
                articles.extend(fut.result())
            except Exception:
                pass

    # dedupe (first seen wins, but prefer a trigger bucket over industry/competitor)
    seen: dict[str, Article] = {}
    bucket_rank = {"trigger": 0, "competitor": 1, "industry": 2}
    for a in articles:
        cur = seen.get(a.fingerprint)
        if cur is None or bucket_rank[a.bucket] < bucket_rank[cur.bucket]:
            seen[a.fingerprint] = a
    # second pass: collapse the same event covered by different outlets (see
    # _collapse_near_duplicates) -- exact-fingerprint dedup only catches identical
    # title+url pairs, which republished wire stories rarely are.
    return _collapse_near_duplicates(list(seen.values()))


def collect_queries(queries: list[str], country: dict | None = None, bucket: str = "market") -> list[Article]:
    """Fetch an explicit list of Google-News queries (used by the CMI lead-nurture pipeline).

    Uses the country's GN locale for sourcing but does NOT hard-filter by country — market
    intelligence is inherently global. Dedupes by fingerprint, returns within the lookback window.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.LOOKBACK_HOURS)
    locale = (country or config.COUNTRIES["Global"]).get("gn", config.GN_DEFAULT_LOCALE)
    articles: list[Article] = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_fetch_one, q, bucket, cutoff, locale): q for q in queries if q and q.strip()}
        for fut in cf.as_completed(futs):
            try:
                articles.extend(fut.result())
            except Exception:
                pass
    seen: dict[str, Article] = {}
    for a in articles:
        seen.setdefault(a.fingerprint, a)
    out = list(seen.values())
    out.sort(key=lambda x: x.published, reverse=True)   # freshest first
    return out
