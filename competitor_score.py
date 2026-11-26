"""Competitor cleanup and scoring for a seller profile.

Cleanup (deterministic, applied to every profile through tenant.tidy_profile):
  - the seller is never its own competitor: brief_builder.is_self plus the seller's
    website label ("Gmx.net" for gmx, "GMX GmbH" for GMX)
  - placeholder names the model emits ("Company A", "Competitor 2") are dropped
  - a trailing country word that is NOT the lead's country is stripped: the model
    "localises" global rivals for the wrong market ("Stryker India" for an Irish lead,
    "Grant Thornton Bharat" for an Irish lead) -> "Stryker", "Grant Thornton"
  - local-language queries and competitor query seeds that name the seller are dropped
    (they search the seller's own news: "GMX Lotto Online")

Scoring (one Google News search per competitor, the lead country's English edition,
last 12 months; articles must name the rival in any written form):
  relevance  cosine(rival headlines, what the seller sells) -- relevance.py embeddings,
             keyword overlap with the seller's category words when no model is present
  recency    0.5 ** (days since the newest such headline / 90)
  presence   share of those headlines that name the lead's country
  volume     min(1, headlines / 10)
  score      0.45 relevance + 0.25 recency + 0.20 presence + 0.10 volume

Verdicts: keep (score, relevance and >= 2 headlines naming the lead's country), out of
market (relevant and recent, but not seen in the lead's country -- Tanishq for a Thai
jeweller), weak, drop, no news, no data (the search returned nothing at all, which can
be a temporary Google News block). For review only: nothing in the pipeline acts on them.

    python competitor_score.py --brief 20294 --score
    python competitor_score.py --ids sample_ids.json --score --out report.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
from datetime import datetime, timedelta, timezone

import regions

HALF_LIFE_DAYS = 90
LOOKBACK_DAYS = 365
W_REL, W_REC, W_PRES, W_VOL = 0.45, 0.25, 0.20, 0.10
KEEP_SCORE, KEEP_REL, WEAK_SCORE = 0.45, 0.30, 0.25
IN_MARKET_HITS = 2       # headlines naming the rival AND the lead's country (as rivals.MIN_HITS)
# MiniLM cosine between a headline and a seller description: ~0.15 unrelated,
# ~0.55+ same product category. Mapped linearly onto 0..1.
SIM_LO, SIM_HI = 0.15, 0.55

_PLACEHOLDER = re.compile(
    r"^(?:company|competitor|rival|player|vendor|brand|firm|manufacturer)\s*[a-z0-9]{0,2}$"
    r"|^(?:n/?a|none|unknown|tbd|other|others|various)$", re.I)
_KEEP_BEFORE = {"of", "de", "do", "da", "del", "di", "du", "des", "for", "and", "&",
                "the", "von", "van", "la", "le"}
_STOP = {"and", "the", "for", "with", "from", "into", "that", "this", "their", "its",
         "sells", "makes", "provides", "offers", "products", "services", "solutions",
         "company", "companies", "customers", "businesses", "global", "leading"}


# ── seller identity ──────────────────────────────────────────────────────────
def _site_label(website: str) -> str:
    host = (website or "").split("://")[-1].split("/")[0].lower()
    host = host[4:] if host.startswith("www.") else host
    return host.split(".")[0] if host else ""


def is_seller(name: str, profile, website: str = "") -> bool:
    """True when `name` (a competitor, alias or query) names the seller itself."""
    import brief_builder

    if brief_builder.is_self(name, profile):
        return True
    label = _site_label(website)
    if len(label) < 3:
        return False
    norm = " " + re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip() + " "
    return f" {label} " in norm


# ── foreign country words in a rival's name ──────────────────────────────────
def _country_phrase(phrase: str) -> str:
    """ISO2 when the whole phrase is a country name/alias/nationality, else ''."""
    p = regions.get_profile(phrase)
    if not p or phrase.strip().lower() == p["iso2"].lower():
        return ""      # bare ISO codes ("SA", "AG") are legal suffixes, not countries
    return p["iso2"]


def strip_foreign_suffix(name: str, lead_iso: str) -> str:
    """Drop trailing country words that are not the lead's ("Stryker India" -> "Stryker"
    for an Irish lead). Names built on a country ("Bank of America", "India Post",
    "Emirates") are left alone."""
    if not lead_iso:
        return name
    words = (name or "").split()
    changed = True
    while changed and len(words) >= 2:
        changed = False
        for k in (3, 2, 1):
            if len(words) <= k:
                continue
            iso = _country_phrase(" ".join(words[-k:]).strip(" ,.-"))
            if iso and iso != lead_iso and words[-k - 1].lower() not in _KEEP_BEFORE:
                words = words[:-k]
                changed = True
                break
    out = " ".join(words).strip(" ,-")
    return out or name


def foreign_countries(name: str, lead_iso: str) -> set[str]:
    return regions.countries_in_text(name) - {lead_iso} if lead_iso else set()


# ── cleanup ──────────────────────────────────────────────────────────────────
def clean_profile(profile, website: str = "", lead_iso: str = "") -> dict:
    """Apply the cleanup rules in place. Returns what changed, by rule."""
    removed: dict[str, list] = {"seller": [], "placeholder": [], "renamed": [],
                                "local_queries": [], "competitor_query_seeds": []}
    kept: list[str] = []
    rename: dict[str, str] = {}
    for c in list(getattr(profile, "competitors", None) or []):
        c = (c or "").strip()
        if not c:
            continue
        if _PLACEHOLDER.match(c):
            removed["placeholder"].append(c)
            continue
        if is_seller(c, profile, website):
            removed["seller"].append(c)
            continue
        new = strip_foreign_suffix(c, lead_iso)
        if new != c:
            removed["renamed"].append([c, new])
        if new.lower() not in {k.lower() for k in kept}:
            kept.append(new)
            rename[c] = new
    profile.competitors = kept

    for field_ in ("local_competitor_names", "competitor_aliases"):
        old = getattr(profile, field_, None) or {}
        new_map = {}
        for k, v in old.items():
            if k not in rename:
                continue
            if isinstance(v, list):
                v = [f for f in v if not is_seller(f, profile, website)]
                if v:
                    new_map[rename[k]] = v
            elif v and not is_seller(v, profile, website):
                new_map[rename[k]] = v
        setattr(profile, field_, new_map)

    for field_ in ("local_queries", "competitor_query_seeds"):
        qs = list(getattr(profile, field_, None) or [])
        keep = [q for q in qs if not is_seller(q, profile, website)]
        removed[field_] = [q for q in qs if q not in keep]
        setattr(profile, field_, keep)
    return {k: v for k, v in removed.items() if v}


# ── scoring ──────────────────────────────────────────────────────────────────
def _category_words(profile) -> set[str]:
    text = " ".join([getattr(profile, "value_prop", "") or ""]
                    + [(p.get("name", "") if isinstance(p, dict) else str(p))
                       for p in (getattr(profile, "products", None) or [])]
                    + list(getattr(profile, "competitor_query_seeds", None) or []))
    return {w for w in re.findall(r"[a-z][a-z-]{3,}", text.lower()) if w not in _STOP}


def _fetch(query: str, locale: tuple) -> list:
    import collect
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    return collect._fetch_one(query, "competitor", cutoff, tuple(locale))


def _relevance(arts: list, profile) -> tuple[float, str]:
    import relevance
    sims = relevance.seller_relevance(arts, profile) if arts else []
    if sims:
        top = sorted(sims, reverse=True)[:3]
        sim = sum(top) / len(top)
        return max(0.0, min(1.0, (sim - SIM_LO) / (SIM_HI - SIM_LO))), "embedding"
    words = _category_words(profile)
    if not arts or not words:
        return 0.0, "keywords"
    hits = sum(1 for a in arts
               if words & set(re.findall(r"[a-z][a-z-]{3,}",
                                         f"{a.title} {a.summary or ''}".lower())))
    return hits / len(arts), "keywords"


def score_competitor(profile, rival: str, country: dict, lead_iso: str, *,
                     fetch=None, now: datetime | None = None) -> dict:
    import config
    import tenant

    fetch = fetch or _fetch
    now = now or datetime.now(timezone.utc)
    locale = country.get("gn") or config.GN_DEFAULT_LOCALE
    try:
        got = fetch(rival, locale) or []
    except Exception:  # noqa: BLE001 -- a failed search is just no evidence
        got = []
    arts, seen = [], set()
    for a in got:
        key = a.url or a.title
        if key in seen or not tenant.rival_named_in(profile, rival, f"{a.title} {a.summary or ''}"):
            continue
        seen.add(key)
        arts.append(a)
    # An empty search can be a temporary Google News block, not a quiet rival: "no data"
    # is never evidence against a competitor.
    row = {"rival": rival, "n": len(arts), "newest_days": None, "relevance": 0.0,
           "recency": 0.0, "presence": 0.0, "volume": 0.0, "score": 0.0, "method": "",
           "verdict": "no news" if got else "no data", "sample": [a.title for a in arts[:3]]}
    if not arts:
        return row
    newest = max(a.published for a in arts)
    days = max(0.0, (now - newest).total_seconds() / 86400)
    rel, method = _relevance(arts, profile)
    hits = sum(1 for a in arts if lead_iso in regions.article_countries(a)) if lead_iso else 0
    pres = hits / len(arts) if lead_iso else 1.0
    in_market = not lead_iso or hits >= IN_MARKET_HITS
    if not in_market:
        # Global rivals' headlines rarely name a country ("Autodesk stock rises"), so a
        # plain-name search undercounts them. Ask directly: "<rival> <country>" on the
        # English and local-language editions (rivals.count_presence).
        import rivals
        hits = max(hits, rivals.count_presence(profile, rival, country, lead_iso,
                                               fetch=fetch))
        in_market = hits >= IN_MARKET_HITS
    row.update(newest_days=round(days, 1), relevance=round(rel, 3),
               recency=round(0.5 ** (days / HALF_LIFE_DAYS), 3),
               presence=round(pres, 3), presence_hits=hits, in_market=in_market,
               volume=round(min(1.0, len(arts) / 10), 3), method=method)
    row["score"] = round(W_REL * row["relevance"] + W_REC * row["recency"]
                         + W_PRES * row["presence"] + W_VOL * row["volume"], 3)
    if row["score"] >= KEEP_SCORE and rel >= KEEP_REL:
        row["verdict"] = "keep" if in_market else "out of market"
    else:
        row["verdict"] = "weak" if row["score"] >= WEAK_SCORE else "drop"
    return row


def score_competitors(profile, country: dict, *, fetch=None, workers: int = 3) -> list[dict]:
    lead_iso = (regions.get_profile(country.get("iso2") or country.get("name") or "")
                or {}).get("iso2", "")
    rivals = list(getattr(profile, "competitors", None) or [])
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(lambda r: score_competitor(profile, r, country, lead_iso,
                                                      fetch=fetch), rivals))
    return sorted(rows, key=lambda r: -r["score"])


# ── benchmark ────────────────────────────────────────────────────────────────
def profile_checks(profile, country: dict, website: str = "") -> dict:
    """Counts of known profile defects, on the profile as it stands (no network)."""
    import collect

    lead_iso = (regions.get_profile(country.get("iso2") or country.get("name") or "")
                or {}).get("iso2", "")
    comps = list(profile.competitors or [])
    jobs, _dropped = collect.plan_searches(profile, country)
    english = [q for q, b, loc, tag in jobs if not tag]
    trig = {q.lower() for q in (profile.trigger_queries or []) + (profile.industry_queries or [])}
    return {
        "competitors": len(comps),
        "seller_as_competitor": sum(1 for c in comps if is_seller(c, profile, website)),
        "placeholder_competitors": sum(1 for c in comps if _PLACEHOLDER.match(c)),
        "foreign_named_competitors": sum(1 for c in comps if foreign_countries(c, lead_iso)),
        "local_queries_naming_seller": sum(
            1 for q in (getattr(profile, "local_queries", None) or [])
            if is_seller(q, profile, website)),
        "seeds_duplicating_triggers": sum(
            1 for q in (getattr(profile, "competitor_query_seeds", None) or [])
            if q.lower() in trig),
        "searches": len(jobs),
        "searches_country_twice": sum(
            1 for q in english if lead_iso and _country_mentions(q, lead_iso) >= 2),
        "searches_over_5_words": sum(1 for q in english if len(q.split()) > 5),
        "searches_other_country": sum(
            1 for q, b, loc, tag in jobs
            if lead_iso and b != "competitor" and foreign_countries(q, lead_iso)),
        "competitor_searches_naming_seller": sum(
            1 for q, b, loc, tag in jobs
            if b == "competitor" and is_seller(q, profile, website)),
    }


def _country_mentions(q: str, iso: str) -> int:
    p = regions.get_profile(iso) or {}
    toks = {t for t in [p.get("name", "").lower(), p.get("nationality", "")]
            + list(p.get("aliases", [])) if t}
    rx = re.compile(r"(?<![a-z])(?:" + "|".join(re.escape(t) for t in
                    sorted(toks, key=len, reverse=True)) + r")(?![a-z])", re.I)
    return len(rx.findall(q))


def _load(brief_id: int, conn):
    import config
    import tenant

    b = conn.execute("SELECT * FROM briefs WHERE id=?", (brief_id,)).fetchone()
    lead = conn.execute("SELECT * FROM leads WHERE email=?", (b["email"],)).fetchone()
    country = config.get_country(lead["country"] or lead["market"] or "Global")
    prof = tenant.Profile.from_dict(json.loads(b["profile_json"] or "{}"))
    return lead, country, prof


def main() -> None:
    import db
    import tenant

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--brief", type=int, action="append", default=[])
    ap.add_argument("--ids", help="JSON file with a list of brief ids")
    ap.add_argument("--score", action="store_true", help="search news and score rivals")
    ap.add_argument("--out", help="write the full report as JSON here")
    args = ap.parse_args()
    ids = list(args.brief) + (json.load(open(args.ids)) if args.ids else [])
    conn = db.get_conn()

    report, totals_before, totals_after = [], {}, {}
    for bid in ids:
        lead, country, prof = _load(bid, conn)
        site = lead["website"] or ""
        before = profile_checks(prof, country, site)
        changes = tenant.tidy_profile(prof, website=site, country=country)
        after = profile_checks(prof, country, site)
        row = {"brief_id": bid, "company": lead["company"], "country": country.get("name"),
               "before": before, "after": after, "changes": changes,
               "competitors": prof.competitors}
        if args.score:
            row["scores"] = score_competitors(prof, country)
        report.append(row)
        for k, v in before.items():
            totals_before[k] = totals_before.get(k, 0) + v
        for k, v in after.items():
            totals_after[k] = totals_after.get(k, 0) + v
        print(f"{bid:>6} {country.get('name', '')[:14]:<15} {lead['company'][:22]:<23} "
              f"{', '.join(prof.competitors[:4])[:70]}")
        for s in row.get("scores", []):
            print(f"         {s['verdict']:<8} {s['score']:.2f}  rel {s['relevance']:.2f}  "
                  f"rec {s['recency']:.2f}  pres {s['presence']:.2f}  n={s['n']:<3} {s['rival']}")

    print(f"\n{'check':<36}{'before':>8}{'after':>8}")
    for k in totals_before:
        print(f"{k:<36}{totals_before[k]:>8}{totals_after.get(k, 0):>8}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"briefs": report, "before": totals_before, "after": totals_after},
                      f, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
