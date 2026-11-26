"""News collection — Google News RSS across trigger / industry / competitor queries.

Distilled from the mother repo's `trends/collector.py`: parallel feed fetch, SHA-256
fingerprint dedup, date-window filtering. Free sources only (no paid news APIs).
"""
from __future__ import annotations
import concurrent.futures as cf
import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from urllib.parse import quote_plus

import feedparser

import config
import regions


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
    # How directly this article is about the enquired market topic (report briefs only);
    # set by score.score_all(topic=...). 0.0 when no topic was given.
    topic_rel: float = 0.0
    # Cosine similarity to the seller text (relevance.py); None = no embedding model.
    seller_rel: float | None = None
    # ISO2 of the market whose LOCAL-language Google News edition returned this article
    # ("" for English-edition hits). Feeds source_country.learn.
    edition_iso2: str = ""
    # Country the event happened in, as named by the triage model ("" = not asked/unknown).
    event_country: str = ""
    # Evidence scores (evidence_score.py): fit to the use case searched for, freshness, age.
    ev_relevancy: float = 0.0
    ev_recency: float = 0.0
    ev_age_days: float = 0.0
    ev_reason: str = ""
    # How a TOPIC article passed the geo rule: "country" / "region" / "global" ("" = not a topic article).
    geo_scope: str = ""
    # Where summary came from after enrich_texts(): "page" / "bing" / "" (feed summary).
    text_source: str = ""


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


def _blocked(a: "Article") -> bool:
    """True when the article's source name or URL matches config.SOURCE_BLOCKLIST
    (research-firm SEO, press-release farms, stock-ticker sites): lowercase substrings."""
    bl = getattr(config, "SOURCE_BLOCKLIST", None) or ()
    if not bl:
        return False
    hay = f"{a.source or ''} {a.url or ''}".lower()
    return any(b and b in hay for b in bl)


def _drop_blocked(articles: list["Article"]) -> list["Article"]:
    return [a for a in articles if not _blocked(a)]


def _prefer(a: "Article", cur: "Article") -> bool:
    """True if `a` should replace `cur` as the representative of one event: better bucket
    first (trigger > competitor > industry), then the longer summary (more text to cite),
    then the more recent copy."""
    bucket_rank = {"trigger": 0, "competitor": 1, "industry": 2}
    ra, rc = bucket_rank.get(a.bucket, 3), bucket_rank.get(cur.bucket, 3)
    if ra != rc:
        return ra < rc
    if abs(len(a.summary) - len(cur.summary)) > 40:
        return len(a.summary) > len(cur.summary)
    return a.published > cur.published


# Cosine on headline embeddings at/above which two articles are one event, across
# outlets AND languages ("Lidl opens store in Krakow" == "Lidl otwiera sklep w Krakowie").
EVENT_SIM_THRESHOLD = 0.85


def _collapse_events(articles: list["Article"]) -> list["Article"]:
    """One representative per EVENT. Token-overlap collapse first (_collapse_near_duplicates);
    then, when the local embedding model is available, cluster the survivors by headline
    cosine >= EVENT_SIM_THRESHOLD so the same story in two languages / worded differently
    also collapses. Without the model the token collapse alone applies (old behaviour)."""
    arts = _collapse_near_duplicates(articles)
    if len(arts) < 2:
        return arts
    try:
        import relevance
        groups = relevance.title_similarity_groups([a.title for a in arts], EVENT_SIM_THRESHOLD)
    except Exception:  # noqa: BLE001 -- embeddings are optional
        groups = None
    if groups is None:
        return arts
    best: dict[int, "Article"] = {}
    for a, g in zip(arts, groups):
        cur = best.get(g)
        if cur is None or _prefer(a, cur):
            best[g] = a
    kept = {id(a) for a in best.values()}
    return [a for a in arts if id(a) in kept]


def _clean(html: str) -> str:
    import html as _html
    s = _html.unescape(re.sub(r"<[^>]+>", "", html or ""))
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


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


def _fetch_one(query: str, bucket: str, cutoff: datetime, locale: tuple,
               when_days: int = 0) -> list[Article]:
    hl, gl, ceid = locale
    # "when:Nd" goes into the URL only: Article.query stays the clean query (evidence_score reads it).
    q_url = f"{query} when:{when_days}d" if when_days > 0 else query
    url = config.GOOGLE_NEWS.format(q=quote_plus(q_url), hl=hl, gl=gl, ceid=ceid)
    out: list[Article] = []
    # Google News answers a burst of requests with an error status or a non-feed page,
    # which feedparser returns as an empty feed -- indistinguishable from "no news"
    # unless checked. Seen while scoring ~150 rivals: Autodesk, Medtronic and SNC-Lavalin
    # came back empty, then returned 25-100 articles each when asked again.
    for attempt in range(3):
        try:
            feed = feedparser.parse(url)
        except Exception:
            return out
        status = getattr(feed, "status", 200)
        blocked = (status in (429, 500, 502, 503, 504)
                   or (getattr(feed, "bozo", False) and not getattr(feed, "entries", None)))
        if not blocked:
            break
        time.sleep(2 * (attempt + 1))
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


def _lead_iso2(country: dict | None) -> str:
    """ISO2 of the lead's country, or '' for a Global/unresolvable lead.

    '' means "do not filter": a Global lead is worldwide by definition, and an
    unknown country keeps the pre-existing behaviour rather than dropping queries.
    """
    if not country or (country.get("name") or "") == "Global":
        return ""
    prof = regions.get_profile(country.get("iso2") or country.get("name") or "")
    return ((prof or {}).get("iso2") or "")


def _foreign_query_countries(query: str, lead_iso2: str) -> set[str]:
    """ISO2 codes a profile-written query names OTHER than the lead's own.

    A query the LLM wrote for one market leaks another market's news into the
    pool: "food safety standards India" (written for an Indian lead, reused for a
    South Korean one) found an FSSAI/Bisleri prosecution story, and collect()
    appending " South Korea" made geo.filter_for_lead's no-geo anchor rule read
    the article as Korean. Such a query must never be searched for that lead.
    """
    if not lead_iso2 or not (query or "").strip():
        return set()
    return regions.countries_in_text(query) - {lead_iso2}


def _keep_lead_queries(queries, lead_iso2: str, dropped: list) -> list[str]:
    """Drop profile-written queries that name a country other than the lead's.
    Appends (query, foreign_isos) to `dropped` for the caller's log/count."""
    kept: list[str] = []
    for q in (queries or []):
        foreign = _foreign_query_countries(q, lead_iso2)
        if foreign:
            dropped.append((q, sorted(foreign)))
        else:
            kept.append(q)
    return kept


# Queries the last collect() call dropped as naming a foreign country, as
# [(query, [other_iso2, ...]), ...]. Diagnostics/tests; reset per call.
LAST_DROPPED_QUERIES: list[tuple[str, list[str]]] = []
LAST_HARVESTED: list[str] = []   # buyer names the last collect() searched for


def plan_searches(profile, country: dict | None = None
                  ) -> tuple[list[tuple[str, str, tuple, str]], list[tuple[str, list[str]]]]:
    """Every Google-News search collect() runs for this profile and lead, without
    running any: ([(query, bucket, locale, market_tag), ...], dropped_queries).

    `country` (a config.COUNTRIES entry) sets the Google-News locale and appends the
    country name to each English query so results are geo-biased -- unless the query
    already names that country ("German telecom merger" is not suffixed with "Germany"
    again). None/Global = worldwide English, no suffix.

    Profile-written queries that name a country OTHER than the lead's are dropped
    before any job is built (see _foreign_query_countries): the LLM writes queries
    for the market it is asked about, but a profile reused for another lead carries
    those market names into a search that gets country-suffixed, which makes the
    no-geo anchor rule in geo.filter_for_lead trust the wrong country.
    """
    locale = (country or config.COUNTRIES["Global"]).get("gn", config.GN_DEFAULT_LOCALE)
    cname = (country or {}).get("name", "Global")
    suffix = "" if cname in ("", "Global") else f" {cname}"   # e.g. "biotech funding India"

    lead_iso2 = _lead_iso2(country)

    def _sfx(q: str) -> str:
        if suffix and lead_iso2 and lead_iso2 in regions.countries_in_text(q):
            return q
        return q + suffix

    dropped: list[tuple[str, list[str]]] = []
    trig_qs = _keep_lead_queries(profile.trigger_queries, lead_iso2, dropped)
    ind_qs = _keep_lead_queries(profile.industry_queries, lead_iso2, dropped)
    seed_qs = _keep_lead_queries(getattr(profile, "competitor_query_seeds", None),
                                lead_iso2, dropped)

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
    if not anchor:
        # No curated terms: anchor on the first product the profile names -- never one
        # carrying the seller's own brand ("Tutanota GMX Cloud" searches for GMX).
        import competitor_score as _cs
        for _p in (getattr(profile, "products", None) or []):
            _n = ((_p.get("name") if isinstance(_p, dict) else str(_p or "")) or "").strip()
            if _n and not _cs.is_seller(_n, profile):
                anchor = _n
                break
    for c in profile.competitors:
        if anchor:
            comp_jobs.append((_sfx(f"{c} {anchor}"), "competitor"))
        comp_jobs.append((_sfx(c), "competitor"))       # plain name too, as a fallback
        # Other written forms of the same rival (English form of a local official name,
        # short form): press in English often uses a form the profile did not store.
        for alias in (getattr(profile, "competitor_aliases", None) or {}).get(c, [])[:2]:
            alias = (alias or "").strip()
            if alias and alias.lower() != c.lower():
                comp_jobs.append((_sfx(alias), "competitor"))

    # Profile-written generic category queries (competitor_query_seeds): "surface ANY
    # company's new X" searches. They fill the competitor bucket without naming a
    # rival, so the articles they return have no named competitor until downstream
    # attribution runs -- that is expected, not a defect. Same country suffix and
    # same long competitor cutoff as the named-rival queries.
    for q in seed_qs:
        comp_jobs.append((_sfx(q), "competitor"))

    jobs = ([(_sfx(q), "trigger") for q in trig_qs]
            + [(_sfx(q), "industry") for q in ind_qs]
            + comp_jobs)
    jobs = [(q, b, locale, "") for q, b in jobs]

    # Local-language pass: every non-English market also gets its queries run on the
    # country's own-language Google News edition (config.LOCAL_NEWS_EDITIONS), which the
    # regional-source experiments showed surfaces far more relevant evidence than the
    # English edition. No country-name suffix (the query is already local-language and the
    # edition is country-scoped). Bucket choice: profile.local_queries are written buyers/
    # triggers FIRST, then category/industry/market, so the first half goes to the trigger
    # bucket and the rest to the industry bucket; local competitor names go to competitor.
    # Same cutoffs and same foreign-country query filter as the English pass.
    local_edition = (getattr(config, "LOCAL_NEWS_EDITIONS", None) or {}).get(cname)
    if local_edition:
        lq = _keep_lead_queries(getattr(profile, "local_queries", None), lead_iso2, dropped)
        n_trig = (len(lq) + 1) // 2
        for i, q in enumerate(lq):
            jobs.append((q, "trigger" if i < n_trig else "industry", tuple(local_edition), cname))
        for loc in (getattr(profile, "local_competitor_names", None) or {}).values():
            if (loc or "").strip():
                jobs.append((loc.strip(), "competitor", tuple(local_edition), cname))
    return jobs, dropped


def _when(hours: float) -> int:
    """Days for Google's when: operator, 0 when the operator is switched off."""
    return math.ceil(hours / 24) if getattr(config, "SEARCH_WHEN_OPERATOR", False) else 0


# Headline: <Company> <buying-event phrase>. The company is the run of capitalised tokens
# just before the phrase, so "Home appliances maker Novamax to invest" gives Novamax.
_BUY_EVENT_RE = re.compile(
    r"\b(?:to invest|invests?|invested|to set up|sets up|plans(?: to)?|to build|builds|opens|"
    r"inaugurates|commissions|(?:to )?acquires?|(?:to )?raises?|bags|wins|secures|signs|"
    r"launches|(?:to )?expands?|gets (?:nod|approval)|board approves|commences|"
    r"starts production|(?:to )?pumps?(?= (?:[$€£₹]|\d|us|rs|inr)))\b", re.I)
_NAME_TOKEN_RE = re.compile(r"^(?:[A-Z0-9][\w&.\-']*|&|\(India\))$")
_NOT_COMPANY = {
    "centre", "center", "cabinet", "government", "govt", "ministry", "minister", "pm", "cm",
    "state", "niti", "aayog", "board", "council", "court", "rbi", "sebi", "commission",
    "department", "parliament", "authority", "bank", "union", "committee", "tribunal",
    "agency", "president", "prime", "sensex", "nifty", "exclusive", "breaking", "report",
}
_GENERIC_NAME = {"indian", "global", "top", "new", "leading", "major", "big", "local", "world",
                 "firm", "company", "group", "india", "asia", "europe", "us", "uk", "eu"}


def harvest_buyers(articles, profile, country: dict | None = None, limit: int = 0) -> list[str]:
    """Names of companies that are the subject of a buying-event headline in the
    trigger/industry articles (no LLM), most-named then most recent first, at most `limit`."""
    if limit <= 0:
        return []
    import competitor_score
    import tenant
    found: dict[str, dict] = {}
    for a in articles:
        if a.bucket not in ("trigger", "industry"):
            continue
        title = (a.title or "").strip()
        m = _BUY_EVENT_RE.search(title)
        if not m:
            continue
        run: list[str] = []
        for tok in reversed(title[:m.start()].split()):
            if not _NAME_TOKEN_RE.match(tok) or tok.endswith(("'s", "’s")) or len(run) >= 5:
                break
            run.insert(0, tok)
        name = re.sub(r"\s*\(India\)", "", " ".join(run)).strip(" .,-&")
        toks = name.split()
        if len(name) < 3 or not toks or any(t.lower().strip(".") in _NOT_COMPANY for t in toks):
            continue
        if len(toks) == 1 and toks[0].lower() in _GENERIC_NAME:
            continue
        if regions.countries_in_text(name):
            left = [t for t in toks if not regions.countries_in_text(t)]
            if not left or len(left) == len(toks):
                continue
        if competitor_score.is_seller(name, profile):
            continue
        if any(c.lower() == name.lower() or tenant.rival_named_in(profile, c, name)
               for c in (profile.competitors or [])):
            continue
        d = found.setdefault(name.lower(), {"name": name, "n": 0, "t": a.published})
        d["n"] += 1
        d["t"] = max(d["t"], a.published)
    searched = " ".join((profile.trigger_queries or []) + (profile.industry_queries or [])).lower()
    ranked = sorted(found.values(), key=lambda d: (-d["n"], -d["t"].timestamp()))
    return [d["name"] for d in ranked if d["name"].lower() not in searched][:limit]


def collect(profile, country: dict | None = None, *, log=None, conn=None) -> list[Article]:
    """Fetch all queries in parallel, dedupe by fingerprint, return within window.

    `profile` carries trigger_queries / industry_queries / competitor_query_seeds /
    competitors; the searches themselves are built by plan_searches()."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.LOOKBACK_HOURS)
    lead_iso2 = _lead_iso2(country)
    jobs, dropped = plan_searches(profile, country)
    LAST_DROPPED_QUERIES[:] = dropped
    if dropped and log:
        log(f"  collect: dropped {len(dropped)} query/queries naming another country: "
            + "; ".join(f"'{q}' -> {','.join(isos)}" for q, isos in dropped))

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

        _hrs = {"competitor": getattr(config, "COMPETITOR_LOOKBACK_HOURS", config.LOOKBACK_HOURS),
                "trigger": _tl}
        futs = {ex.submit(_fetch_one, q, b, _cut(b), loc,
                          _when(_hrs.get(b, config.LOOKBACK_HOURS))): tag
                for q, b, loc, tag in jobs}
        for fut in cf.as_completed(futs):
            try:
                got = fut.result()
            except Exception:
                continue
            if futs[fut]:
                # Record the market on a local-edition hit's query: the edition IS the
                # country (gl=), and geo.filter_for_lead trusts a country word in `query`
                # for articles whose text names no place (local text rarely says "Poland").
                for a in got:
                    a.query = f"{a.query} {futs[fut]}"
                    a.edition_iso2 = lead_iso2
            articles.extend(got)
    articles = _drop_blocked(articles)

    # Second pass: search each buyer company named in a first-pass headline by name.
    names = harvest_buyers(articles, profile, country, getattr(config, "HARVEST_BUYERS", 0))
    if names:
        locale = (country or config.COUNTRIES["Global"]).get("gn", config.GN_DEFAULT_LOCALE)
        cname = (country or {}).get("name", "Global")
        sfx = "" if cname in ("", "Global") else f" {cname}"
        qs = [f'"{n}"' + ("" if sfx and lead_iso2 and lead_iso2 in regions.countries_in_text(n)
                          else sfx) for n in names]
        if log:
            log(f"  collect: harvested buyers {names}")
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            for got in ex.map(lambda q: _fetch_one(q, "trigger", trig_cutoff, locale, _when(_tl)), qs):
                articles.extend(got)
        articles = _drop_blocked(articles)
    LAST_HARVESTED[:] = names

    # Learn which country each outlet belongs to (source_country.py); `conn` is optional
    # -- without it the map is in-memory only.
    try:
        import source_country
        source_country.learn(articles, conn)
    except Exception:  # noqa: BLE001 -- learning is a bonus, never fatal
        pass

    # dedupe (first seen wins, but prefer a trigger bucket over industry/competitor)
    seen: dict[str, Article] = {}
    bucket_rank = {"trigger": 0, "competitor": 1, "industry": 2}
    for a in articles:
        cur = seen.get(a.fingerprint)
        if cur is None or bucket_rank[a.bucket] < bucket_rank[cur.bucket]:
            seen[a.fingerprint] = a
    # second pass: collapse the same event covered by different outlets / languages (see
    # _collapse_events) -- exact-fingerprint dedup only catches identical title+url
    # pairs, which republished wire stories rarely are.
    return _collapse_events(list(seen.values()))


# ── Topic collection (report briefs) ─────────────────────────────────────────
# Words that describe the REPORT, not the subject. Leaving "market" in a query is the
# single worst thing to do for a news search: it pulls in rival research firms' own
# "X Market Size, CAGR" press releases (which eligible() must then drop by policy) and
# crowds out actual reporting. Stripped from the search string; the topic itself stays.
_TOPIC_NOISE = {"market", "markets", "global", "report", "analysis", "size", "share", "forecast",
                "tracker", "pricing", "database", "ad-hoc", "adhoc", "study", "outlook", "trends",
                "overview", "insights", "regional", "strategy", "partnership"}


def topic_subject(topic: str, iso2: str = "") -> str:
    """'Global Mesoporous Silica Market' -> 'Mesoporous Silica'. Falls back to the raw
    topic if stripping would leave nothing. Years ("2025-2032") are dropped; with `iso2`
    the lead country's own words too ("Spain Data Center Market" -> "Data Center") --
    collect_topic anchors queries to the country itself."""
    raw = topic or ""
    if iso2:
        import tenant   # local: tenant imports collect
        raw = tenant.strip_country_words(raw, iso2) or raw
    words = [w for w in re.split(r"[\s\-–—:|/]+", raw) if w]
    kept = [w for w in words if w.lower().strip(".,") not in _TOPIC_NOISE
            and not re.fullmatch(r"[\d.,]+", w)]
    return " ".join(kept).strip() or (topic or "").strip()


def translate_topic(subject: str, language: str) -> str:
    """The market subject as a short news-search phrase in `language` ("" on any failure)."""
    if not (subject or "").strip() or not (language or "").strip():
        return ""
    try:
        import llm
        out = llm.chat_json(
            "You translate short market names for news search.",
            f'Translate "{subject}" into {language} as a 1-4 word phrase a news reporter would '
            f'search for, in the native script. Reply as JSON: {{"translation": "..."}}',
            max_tokens=60, default={})
        t = (out or {}).get("translation", "")
        return t.strip() if isinstance(t, str) else ""
    except Exception:  # noqa: BLE001
        return ""


def _bing_title_split(title: str) -> tuple[str, str]:
    """Bing's RSS title is sometimes 'Headline | Publisher'."""
    if " | " in title:
        head, pub = title.rsplit(" | ", 1)
        if 0 < len(pub) <= 40:
            return head.strip(), pub.strip()
    return title.strip(), ""


def _fetch_bing(query: str, bucket: str, cutoff: datetime) -> list[Article]:
    """Bing News RSS: a second free engine. Unlike Google News RSS (whose 'summary' is
    the headline again, 99% of the time), Bing returns the article's opening lines, and
    its links carry the publisher's real URL in a `url=` parameter."""
    from urllib.parse import parse_qs, urlparse

    url = config.BING_NEWS.format(q=quote_plus(query))
    out: list[Article] = []
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": config.HTTP_UA})
    except Exception:  # noqa: BLE001
        return out
    for e in feed.entries:
        dt = _parse_dt(e)
        if dt is None or dt < cutoff:
            continue
        raw_title = _clean(getattr(e, "title", ""))
        title, pub = _bing_title_split(raw_title)
        link = getattr(e, "link", "")
        real = parse_qs(urlparse(link).query).get("url", [""])[0]
        link = real or link
        if not title or not link:
            continue
        src = getattr(e, "news_source", "") or pub or "Bing News"
        out.append(Article(title=title, url=link, source=str(src), published=dt,
                          summary=_clean(getattr(e, "summary", ""))[:600], query=query,
                          bucket=bucket, fingerprint=_fingerprint(title, link)))
    return out


_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_TEXT_MAX = 800


def _is_gnews(url: str) -> bool:
    return "news.google.com" in (url or "").lower()


def _same_as_title(text: str, title: str) -> bool:
    import difflib
    return difflib.SequenceMatcher(None, text.lower()[:len(title) + 40], title.lower()).ratio() >= 0.8


def _page_text(url: str, timeout: int) -> str:
    """Description meta + first paragraphs of a publisher page. Never raises."""
    import html as _html
    import requests
    try:
        r = requests.get(url, timeout=timeout, headers=_BROWSER_HEADERS,
                         allow_redirects=True, verify=False)
        if r.status_code >= 400:
            return ""
        page = r.text or ""
    except Exception:  # noqa: BLE001 -- best-effort
        return ""

    def clean(s: str) -> str:
        s = re.sub(r"(?is)<(script|style|noscript)\b.*?</\1>", " ", s)
        s = re.sub(r"(?s)<[^>]+>", " ", s)
        return re.sub(r"\s+", " ", _html.unescape(s)).strip()

    parts: list[str] = []
    for m in re.finditer(r"(?is)<meta\b[^>]*>", page):
        tag = m.group(0)
        if re.search(r"""(?i)(property|name)\s*=\s*["'](og:description|description)["']""", tag):
            c = re.search(r"""(?is)content\s*=\s*("([^"]*)"|'([^']*)')""", tag)
            if c:
                t = clean(c.group(2) or c.group(3) or "")
                if t:
                    parts.append(t)
                    break
    total = sum(len(p) for p in parts)
    if total < _TEXT_MAX:
        for m in re.finditer(r"(?is)<p\b[^>]*>(.*?)</p>", page):
            t = clean(m.group(1))
            if len(t) < 40:  # nav / caption crumbs
                continue
            parts.append(t)
            total += len(t)
            if total >= _TEXT_MAX:
                break
    return " ".join(parts)[:_TEXT_MAX].strip()


def _title_key(t: str) -> str:
    """Headline for same-story matching: no " - Publisher" tail, plain quotes, lowercase."""
    t = re.sub(r"\s+[-|–]\s+[^-|–]{2,40}$", "", t or "")
    return t.replace("’", "'").replace("‘", "'").lower().strip()


def _article_text_src(a: Article, timeout: int) -> tuple[str, str]:
    """(text, "page"|"bing"|"") -- see fetch_article_text."""
    import difflib
    text = ""
    if a.url and not _is_gnews(a.url):
        text = _page_text(a.url, timeout)
    if _same_as_title(text, a.title):
        text = ""
    if len(text) >= 120:
        return text, "page"
    cutoff = (a.published or datetime.now(timezone.utc)) - timedelta(days=30)
    best, best_r = None, 0.0
    for b in _fetch_bing(a.title, a.bucket, cutoff):
        r = difflib.SequenceMatcher(None, _title_key(b.title), _title_key(a.title)).ratio()
        if r > best_r:
            best, best_r = b, r
    # Same story only: at 0.6 "Tesla raises Wall Connector prices" matched a Supercharger
    # story, which would paste another event's facts into this article.
    if best is None or best_r < 0.85:
        return text, ("page" if text else "")
    btext, src = (best.summary or "").strip(), "bing"
    if best.url and "bing.com" not in best.url.lower() and not _is_gnews(best.url):
        ptext = _page_text(best.url, timeout)
        if len(ptext) > len(btext):
            btext, src = ptext, "page"
    btext = btext[:_TEXT_MAX].strip()
    if _same_as_title(btext, a.title):
        btext = ""
    if len(btext) > len(text):
        return btext, src
    return text, ("page" if text else "")


def fetch_article_text(a: Article, *, timeout: int = 8) -> str:
    """Real opening text for an article ("" on any failure, max 800 chars).

    Google News summaries are the headline again, so writers saw a headline only and
    invented the rest. Publisher pages are read directly; Google News redirect links
    cannot be resolved by plain requests, so those go through Bing News, which returns
    the article's opening lines and the publisher's real URL. Only the headline is sent
    out, never lead data."""
    try:
        return _article_text_src(a, timeout)[0]
    except Exception:  # noqa: BLE001 -- enrichment never fails a brief
        return ""


def enrich_texts(articles: list[Article], per_bucket: int = 8, max_workers: int = 8) -> int:
    """Replace headline-only summaries with fetched text for the articles that matter:
    top `per_bucket` per bucket by evidence score, plus the top 8 by topic relevance
    (report briefs). Returns how many summaries were replaced."""
    picked: list[Article] = []
    seen: set[int] = set()

    def take(group: list[Article], n: int, key) -> None:
        for a in sorted(group, key=key, reverse=True)[:n]:
            if id(a) not in seen:
                seen.add(id(a))
                picked.append(a)

    for bucket in ("trigger", "industry", "competitor"):
        take([a for a in articles if a.bucket == bucket], per_bucket,
             lambda a: (getattr(a, "ev_relevancy", 0) * getattr(a, "ev_recency", 0),
                        getattr(a, "composite", 0)))
    take([a for a in articles if getattr(a, "topic_rel", 0) > 0], 8,
         lambda a: getattr(a, "topic_rel", 0))
    if not picked:
        return 0

    def work(a: Article) -> int:
        try:
            text, src = _article_text_src(a, 8)
            if text and len(text) >= len(a.summary or "") + 40:
                a.summary = text
                a.text_source = src or "bing"
                return 1
        except Exception:  # noqa: BLE001
            pass
        return 0

    with cf.ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
        return sum(ex.map(work, picked))


def topic_queries(topic: str, iso2: str = "") -> list[str]:
    """Search strings about the SUBJECT of a market, worded the way reporters write."""
    s = topic_subject(topic, iso2)
    return [s, f"{s} industry", f"{s} manufacturers", f"{s} demand supply",
            f"{s} expansion investment", f"{s} regulation policy"]


def anchored_topic_queries(topic: str, country: dict | None, iso2: str = "") -> list[str]:
    """Country-anchored variants of topic_queries(topic): every topic query with
    the country's name appended ("stem cells Morocco") plus demonym forms for the
    first two ("Indian autonomous vessels"), fetched with the country's Google
    News locale.

    Why: the 2026-11 live run showed REPORT briefs leaking through no-geo
    'industry'/topic articles — "HD Hyundai and Anduril launch autonomous vessel"
    reached an India lead because the bare "autonomous vessels" query found it and
    no country word appeared in the text. With an anchored query in the pool, the
    article's `query` field carries the country anchor and geo.filter_for_lead's
    tightened no-geo rule can tell anchored results from un-anchored ones.
    """
    cname = ((country or {}).get("name") or "").strip()
    if not cname or cname == "Global":
        return []
    qs = topic_queries(topic, iso2)
    out = [f"{q} {cname}" for q in qs]
    demonym = ((country or {}).get("nationality") or "").strip().lower()
    if demonym:
        out += [f"{q} {demonym}" for q in qs[:2]]
    return out


def collect_topic(topic: str, country: dict | None = None, local_topic: str = "") -> list[Article]:
    """News ABOUT the market a report lead enquired about.

    Why this exists: the old path called collect_queries() with
    ["<topic> Market", "<x> market growth", "<x> companies", "<x> industry India"], which
    (a) inherited collect_queries' 96-hour cutoff -- the company queries get six months --
    so niche topics returned 0-6 articles; (b) put the word "market" in every query,
    which draws rival research firms' own report PR; and (c) hard-coded 'India'.
    Measured on 8 real Campaign 6 topics: 0-6 on-topic articles before, 5-62 after.

    Google News (country locale) plus Bing News (snippets). Results stay in the
    'industry' bucket; research-firm PR is still excluded later by score.eligible().

    `local_topic` (optional) is the topic translated into the market's main local
    language; when given and the market has a local edition (config.LOCAL_NEWS_EDITIONS)
    it is also searched there. Absent -> the local pass is skipped.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.TOPIC_LOOKBACK_HOURS)
    locale = (country or config.COUNTRIES["Global"]).get("gn", config.GN_DEFAULT_LOCALE)
    iso2 = _lead_iso2(country)
    qs = topic_queries(topic, iso2)
    aq = anchored_topic_queries(topic, country, iso2)
    articles: list[Article] = []
    cname = ((country or {}).get("name") or "").strip()
    local_edition = (getattr(config, "LOCAL_NEWS_EDITIONS", None) or {}).get(cname)
    local_q = (local_topic or "").strip()
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_fetch_one, q, "industry", cutoff, locale, _when(config.TOPIC_LOOKBACK_HOURS))
                for q in qs + aq]
        futs += [ex.submit(_fetch_bing, q, "industry", cutoff) for q in qs[:4]]
        local_futs = []
        if local_edition and local_q:
            local_futs = [ex.submit(_fetch_one, local_q, "industry", cutoff, tuple(local_edition),
                                    _when(config.TOPIC_LOOKBACK_HOURS))]
        for fut in cf.as_completed(futs + local_futs):
            try:
                got = fut.result()
            except Exception:  # noqa: BLE001
                continue
            if fut in local_futs:
                for a in got:
                    a.query = f"{a.query} {cname}"      # see collect(): edition == country
            articles.extend(got)
    articles = _drop_blocked(articles)
    # Same story from both engines / both query forms: keep the copy that carries
    # real text, and on a tie prefer the country-ANCHORED copy — its `query` field
    # then names the country, which is exactly the evidence geo.filter_for_lead's
    # no-geo rule looks for.
    _anchor_toks = [t for t in (((country or {}).get("name") or "").lower().strip(),
                                ((country or {}).get("nationality") or "").lower().strip()) if t]

    def _anchored(a: Article) -> bool:
        ql = (a.query or "").lower()
        return any(t in ql for t in _anchor_toks)

    best: dict[str, Article] = {}
    for a in articles:
        key = _story_key(a.title)
        if not key:
            continue
        cur = best.get(key)
        if cur is None or (_anchored(a) and not _anchored(cur)):
            best[key] = a
        elif not _anchored(a) and _anchored(cur):
            continue
        elif len(a.summary) > len(cur.summary) + 40:
            best[key] = a
    out = _collapse_events(list(best.values()))
    out.sort(key=lambda x: x.published, reverse=True)
    return out


def collect_queries(queries: list[str], country: dict | None = None, bucket: str = "market") -> list[Article]:
    """Fetch an explicit list of Google-News queries (used by the CMI lead-nurture pipeline).

    Uses the country's GN locale for sourcing but does NOT hard-filter by country — market
    intelligence is inherently global. Dedupes by fingerprint, returns within the lookback window.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.LOOKBACK_HOURS)
    locale = (country or config.COUNTRIES["Global"]).get("gn", config.GN_DEFAULT_LOCALE)
    articles: list[Article] = []
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_fetch_one, q, bucket, cutoff, locale, _when(config.LOOKBACK_HOURS)): q for q in queries if q and q.strip()}
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
