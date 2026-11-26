"""Evidence scores for every collected article, run after the whole search is done.

  relevancy 0..1  how well the article fits the use case it was searched for
                    trigger     a company in a buyer industry with a buying event
                    industry    a structural trend / regulation / demand shift for the buyer industries
                    competitor  a move by a named rival in the seller's own product category
                  = 0.55 embedding + 0.25 query-term coverage + 0.20 bucket check
                    (0.6 coverage + 0.4 bucket check when the embedding model is unavailable)
  recency   0..1  0.5 ** (age_days / half-life), half-life per bucket

Annotation only: score.score_all calls score_evidence, which sets ev_* fields on each
article and never touches composite, order or eligibility. Nothing here raises.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

import config
import relevance
import tenant

HALF_LIFE_DAYS = {"trigger": 30, "industry": 45, "competitor": 60}
DEFAULT_HALF_LIFE = 30
# MiniLM cosine of (article, use-case text): ~0.20 off-topic, ~0.65+ on the use case.
SIM_LO, SIM_HI = 0.20, 0.65
W_EMB, W_COV, W_BUCKET = 0.55, 0.25, 0.20
W_COV_NOEMB, W_BUCKET_NOEMB = 0.6, 0.4

_INTENT = {
    "trigger": "A company in the buyer industries with a buying event: funding, expansion, new plant, "
               "launch, acquisition, hiring or a compliance obligation.",
    "industry": "A structural trend, regulation or demand shift affecting the buyer industries.",
    "competitor": "A move by a competitor in the seller's own product category: launch, expansion, "
                  "contract, funding, partnership or results.",
}
_STOP = {"with", "from", "that", "this", "their", "into", "over", "amid", "after", "about", "news",
         "says", "will", "have", "been", "market", "markets", "company", "companies", "industry"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9'-]*", (text or "").lower())


def _iso2(profile) -> str:
    try:
        import regions
        return (regions.get_profile(getattr(profile, "market", "") or "") or {}).get("iso2", "")
    except Exception:  # noqa: BLE001
        return ""


def _clean_query(query: str, iso2: str) -> str:
    """Query without country words: the profile's market plus any country the query names
    (a stored profile's market is often 'Global', and collect() appends the lead's country)."""
    q = query or ""
    try:
        import regions
        isos = set(regions.countries_in_text(q))
    except Exception:  # noqa: BLE001
        isos = set()
    if iso2:
        isos.add(iso2)
    for iso in isos:
        q = tenant.strip_country_words(q, iso)
    return q


def _query_terms(query: str) -> list[str]:
    seen: list[str] = []
    for w in _words(query):
        if len(w) >= 4 and w not in _STOP and w not in seen:
            seen.append(w)
    return seen


def age_days(a, now: datetime) -> float:
    return max(0.0, (now - a.published).total_seconds() / 86400.0)


def recency(bucket: str, days: float) -> float:
    return 0.5 ** (days / HALF_LIFE_DAYS.get(bucket, DEFAULT_HALF_LIFE))


def _use_case_text(bucket: str, query: str, profile, iso2: str) -> str:
    intent = _INTENT.get(bucket) or _INTENT["industry"]
    q = _clean_query(query, iso2)
    vp = (getattr(profile, "value_prop", "") or "").strip()[:160]
    inds = ", ".join(str(i) for i in (getattr(profile, "industries", None) or [])[:6])
    ctx = f"Seller: {vp}" if vp else ""
    if inds:
        ctx += f" Buyers: {inds}."
    return re.sub(r"\s+", " ", f"{q}. {intent} {ctx}").strip()


def _industry_words(profile) -> set[str]:
    out: set[str] = set()
    for kws in (getattr(profile, "industry_keywords", None) or {}).values():
        out.update(str(k).lower() for k in kws if str(k).strip())
    for i in (getattr(profile, "industries", None) or []):
        out.update(w for w in _words(str(i)) if len(w) >= 4 and w not in _STOP)
    return out


# Buying events / competitive moves the trigger lexicon in score.importance misses.
_EVENT_RE = re.compile(
    r"(?<![a-z0-9])(invest(?:s|ed|ing|ment|ments)?|capex|ramps? up|capacity|new (?:plant|facility|factory|"
    r"campus|centre|center|office)|inaugurat\w*|opens?|opened|commission\w*|expan\w*|acqui\w*|merger|"
    r"joint venture|jv|partner\w*|tie-?ups?|ties up|launch\w*|unveil\w*|introduc\w*|funding|raises|raised|"
    r"secures?|secured|orders?|contracts?|deal wins?|wins?|hires?|appoints?|"
    r"set(?:s|ting)? up|plans? (?:to )?(?:build|set up|invest)|"
    r"plans? (?:[\w.$₹-]+ ){0,4}(?:plant|factory|facility|unit|cluster)s?|"
    r"(?:build\w*|open\w*|new|launch\w*|inaugurat\w*) (?:[\w.$₹-]+ ){0,5}(?:plant|factory|facility|unit|cluster)s?|"
    r"pump(?:s|ed)?(?= (?:[$€£₹]|\d|us|rs|inr))|pours?|commits?|approv\w*|clears?|cleared|nod|sanction\w*|green-?light\w*|"
    r"foundation stone|groundbreaking|commences?|starts? production|fundrais\w*|raise)(?![a-z0-9])")
_WEAK_EVENT = {"hire", "hires", "appoint", "appoints", "order", "orders", "win", "wins", "open", "opens"}


def _event(a) -> str:
    """Event word for a buying event / move: score.importance's type, else the local regex."""
    import score
    ttype = score.importance(a)[1]
    if ttype:
        return ttype
    m = _EVENT_RE.search(f"{a.title} {a.summary or ''}".lower())
    return m.group(1) if m else ""


# Junk the safe-to-cite helpers miss: research-report pages and stock-chatter lists.
_REPORT_RE = re.compile(r"market size|market share|growth analysis|analysis report|market report|"
                        r"\bforecast\b.{0,20}20\d\d|\b20\d\d\s*(?:-|to)\s*20\d\d\b|trends?,?\s*20[2-4]\d\b", re.I)
_STOCK_RE = re.compile(r"stocks? to (?:buy|watch|sell)|buy or hold|buy,? sell|death cross|golden cross|"
                       r"target price|price target|technical (?:analysis|rating)|stock (?:picks?|recommendation)|"
                       r"\bvs\.?\b.{0,60}\bstocks?\b", re.I)
# "Why investors are looking beyond solar modules", "How PLI is helping brands go global":
# explainers about a sector, not a company's buying event, even when an event word appears.
_EXPLAINER_RE = re.compile(r"^\W*(?:why|how|what)\b", re.I)


def _junk(a) -> str:
    import score
    t = a.title or ""
    if score._is_ticker_noise(a) or _STOCK_RE.search(t):
        return "stock noise"
    # is_unsafe_to_cite also rejects by SOURCE (Instagram posts, research firms such as
    # Spherical Insights that issue real "India Approves ..." items). score.py stays as is
    # (it decides brief eligibility); here a title with a buying event is judged by its
    # own text only, so report/listicle headlines are still capped.
    if _EVENT_RE.search(t.lower()):
        if score._SEO_LISTICLE.search(t) or _REPORT_RE.search(t) or config._REPORT_PR_RE.search(t):
            return "SEO/report page"
        return ""
    if score.is_unsafe_to_cite(a) or score._SEO_LISTICLE.search(t) or _REPORT_RE.search(t):
        return "SEO/report page"
    return ""


_GENERIC_FIRST = {"general", "national", "united", "international", "global", "american", "indian",
                  "british", "first", "new", "royal", "standard", "china", "japan", "india", "asia",
                  "euro", "digital", "smart", "tech", "group", "industries"}
_LEGAL = {"inc", "ltd", "limited", "llc", "gmbh", "ag", "corp", "corporation", "plc", "co", "sa", "pvt",
          "private", "company", "se", "nv", "bv", "lp"}


def _core_forms(name: str) -> set[str]:
    out: set[str] = set()
    n = (name or "").strip()
    try:
        import regions
        for iso in regions.countries_in_text(n):
            n = tenant.strip_country_words(n, iso)
    except Exception:  # noqa: BLE001
        pass
    toks = _words(n.replace(".", " ").replace(",", " "))
    while toks and toks[-1] in _LEGAL:
        toks.pop()
    if toks:
        out.add(" ".join(toks))
        if len(toks[0]) >= 4 and toks[0] not in _GENERIC_FIRST and toks[0] not in _LEGAL:
            out.add(toks[0])
    return out


def _named_rival(profile, text: str) -> str:
    tl = (text or "").lower()
    for c in (getattr(profile, "competitors", None) or []):
        try:
            if tenant.rival_named_in(profile, c, text):
                return c
            forms: set[str] = set()
            for f in tenant.rival_forms(profile, c):
                forms |= _core_forms(f)
            if any(re.search(r"(?<![a-z0-9])" + re.escape(f) + r"(?![a-z0-9])", tl) for f in forms if f):
                return c
        except Exception:  # noqa: BLE001
            continue
    return ""


def _term_rx(t: str):
    return re.compile(r"(?<![a-z0-9])" + re.escape(t) + r"(?:s|es)?(?![a-z0-9])")


_SUFFIXES = ("ments", "ment", "ers", "er", "ion", "ing", "ed", "es", "s")


def _stem_rx(t: str):
    """Query term as a stem (>= 4 chars) matching any word that starts with it, so
    'raises' finds 'raise' and 'funding' finds 'fundraising'."""
    for suf in _SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            t = t[:-len(suf)]
            break
    return re.compile(r"(?<![a-z0-9])" + re.escape(t) + r"[a-z]*")


def _bucket_check(a, profile, ind_words: set[str], text_l: str) -> tuple[float, str]:
    import score
    if a.bucket == "trigger":
        if _EXPLAINER_RE.search(a.title or ""):
            return 0.0, "explainer headline"
        _imp, ttype = score.importance(a)
        if ttype:
            return 1.0, f"trigger: {ttype}"
        ev = _event(a)
        if ev:
            return (0.7 if ev in _WEAK_EVENT else 1.0), f"event: {ev}"
        return 0.0, "no buying event"
    if a.bucket == "competitor":
        rival = _named_rival(profile, f"{a.title} {a.summary or ''}")
        return (1.0, f"names rival {rival}") if rival else (0.0, "no rival named")
    hit = next((w for w in sorted(ind_words) if _term_rx(w).search(text_l)), "")
    return (1.0, f"buyer-industry term '{hit}'") if hit else (0.0, "no buyer-industry term")


def score_evidence(articles: list, profile) -> list:
    """Annotate ev_relevancy / ev_recency / ev_age_days / ev_reason on each article, in
    place, and return the list. Order and every other field are left alone."""
    if not articles:
        return articles
    try:
        _score(articles, profile)
    except Exception:  # noqa: BLE001 -- annotation must never break scoring
        pass
    return articles


def _score(articles: list, profile) -> None:
    now = datetime.now(timezone.utc)
    iso2 = _iso2(profile)
    ind_words = _industry_words(profile)
    rival_words = {w for c in (getattr(profile, "competitors", None) or [])
                   for f in tenant.rival_forms(profile, c) for w in _words(f)}

    uc_texts = [_use_case_text(a.bucket, a.query, profile, iso2) for a in articles]
    uniq = list(dict.fromkeys(uc_texts))
    # The long use-case text gives a short headline a near-zero cosine, so the bare query is
    # embedded too and the better of the two counts.
    q_texts = [_clean_query(a.query, iso2).strip() for a in articles]
    q_uniq = list(dict.fromkeys(t for t in q_texts if t))
    cos: list[float] | None = None
    try:
        mat = relevance.embed(uniq + q_uniq + [relevance.article_text(a) for a in articles])
        if mat is not None:
            idx = {t: i for i, t in enumerate(uniq)}
            qidx = {t: len(uniq) + i for i, t in enumerate(q_uniq)}
            arts_m = mat[len(uniq) + len(q_uniq):]
            cos = [max(float(arts_m[i] @ mat[idx[uc_texts[i]]]),
                       float(arts_m[i] @ mat[qidx[q_texts[i]]]) if q_texts[i] else -1.0)
                   for i in range(len(articles))]
    except Exception:  # noqa: BLE001
        cos = None

    for i, a in enumerate(articles):
        text_l = f"{a.title} {a.summary or ''}".lower()
        terms = _query_terms(_clean_query(a.query, iso2))
        move = ""
        if a.bucket == "competitor":
            # A rival's name in the query is already the bucket check; what the query adds is a move.
            terms = [t for t in terms if t not in rival_words]
            if not terms:
                move = _event(a) or "none"
        found = sum(1 for t in terms if _stem_rx(t).search(text_l))
        cov = found / len(terms) if terms else (
            (0.6 if move in _WEAK_EVENT else 1.0) if move != "none" and move else 0.0)
        chk, chk_reason = _bucket_check(a, profile, ind_words, text_l)
        if cos is not None:
            emb = max(0.0, min(1.0, (cos[i] - SIM_LO) / (SIM_HI - SIM_LO)))
            rel = W_EMB * emb + W_COV * cov + W_BUCKET * chk
        else:
            rel = W_COV_NOEMB * cov + W_BUCKET_NOEMB * chk
        days = age_days(a, now)
        junk = _junk(a)
        if junk:
            rel = min(rel, 0.15)
        a.ev_relevancy = round(max(0.0, min(1.0, rel)), 3)
        a.ev_age_days = round(days, 1)
        a.ev_recency = round(recency(a.bucket, days), 3)
        reason = [chk_reason]
        if terms:
            reason.append(f"query match {found}/{len(terms)}")
        elif move:
            reason.append(f"move: {move}" if move != "none" else "no move")
        if cos is not None:
            reason.append(f"topic fit {emb:.2f}")
        if junk:
            reason.append(f"capped: {junk}")
        a.ev_reason = "; ".join(reason)
