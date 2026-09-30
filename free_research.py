"""Free company enrichment — the ZERO-COST tiers of the mother repo's free_deep_research.

Only sources that cost nothing and need no API key:
  • Wikipedia REST summary        (en.wikipedia.org/api/rest_v1)
  • Company-scoped Google News    (feedparser, country-localized)
  • yfinance financials           (best-effort, optional — free, but name→ticker is fuzzy)

No embeddings, no Tavily/Apollo/Hunter, no smart_crawl browser. Returns a compact context
block the prospect-card LLM call can ground on, so narratives are richer and less hallucinated.
"""
from __future__ import annotations
import concurrent.futures as cf
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta

import feedparser

import config

_UA = {"User-Agent": "CoherentConnect-Newsletter/1.0 (research)"}
_TIMEOUT = 8


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


# ── Wikipedia (free) ─────────────────────────────────────────────────────────
def _wikipedia_summary(name: str) -> str:
    if not name:
        return ""
    title = urllib.parse.quote(name.strip().replace(" ", "_"))
    try:
        data = _get_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}")
        if data.get("type") == "standard" and data.get("extract"):
            return data["extract"][:800]
    except Exception:
        pass
    # fallback: opensearch → first hit → summary
    try:
        q = urllib.parse.quote(name)
        hits = _get_json(f"https://en.wikipedia.org/w/api.php?action=opensearch&search={q}&limit=1&format=json")
        if len(hits) >= 2 and hits[1]:
            t2 = urllib.parse.quote(hits[1][0].replace(" ", "_"))
            data = _get_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{t2}")
            if data.get("extract"):
                return data["extract"][:800]
    except Exception:
        pass
    return ""


# ── Company-scoped Google News (free) ────────────────────────────────────────
def _company_news(name: str, country: dict, limit: int = 6) -> list[str]:
    if not name:
        return []
    hl, gl, ceid = (country or config.COUNTRIES["Global"]).get("gn", config.GN_DEFAULT_LOCALE)
    url = config.GOOGLE_NEWS.format(q=urllib.parse.quote_plus(f'"{name}"'), hl=hl, gl=gl, ceid=ceid)
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    out = []
    try:
        feed = feedparser.parse(url)
        for e in feed.entries[: limit * 2]:
            t = getattr(e, "title", "")
            if " - " in t:
                t = t.rsplit(" - ", 1)[0]
            if t:
                out.append(t.strip())
            if len(out) >= limit:
                break
    except Exception:
        pass
    return out


# ── yfinance financials (free, best-effort) ──────────────────────────────────
def _yfinance_facts(name: str) -> str:
    try:
        import yfinance as yf  # optional dep
    except Exception:
        return ""
    # crude name→ticker: try the first word uppercased; skip if it fails fast.
    guess = "".join(ch for ch in name.split()[0] if ch.isalnum()).upper()[:5] if name.split() else ""
    if not guess:
        return ""
    try:
        info = yf.Ticker(guess).get_info()
        if info and info.get("longName") and name.split()[0].lower() in info["longName"].lower():
            bits = []
            if info.get("sector"):
                bits.append(f"sector {info['sector']}")
            if info.get("marketCap"):
                bits.append(f"market cap ~{info['marketCap']:,}")
            if info.get("fullTimeEmployees"):
                bits.append(f"{info['fullTimeEmployees']:,} employees")
            return "; ".join(bits)
    except Exception:
        pass
    return ""


# ── Orchestrator ─────────────────────────────────────────────────────────────
def enrich_company(name: str, country: dict | None = None) -> dict:
    """Gather free facts about a company. Parallel, best-effort, never raises."""
    result = {"wikipedia": "", "recent_news": [], "financials": "", "sources_used": []}
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        f_wiki = ex.submit(_wikipedia_summary, name)
        f_news = ex.submit(_company_news, name, country or config.COUNTRIES["Global"])
        f_fin = ex.submit(_yfinance_facts, name)
        result["wikipedia"] = f_wiki.result()
        result["recent_news"] = f_news.result()
        result["financials"] = f_fin.result()
    if result["wikipedia"]:
        result["sources_used"].append("Wikipedia")
    if result["recent_news"]:
        result["sources_used"].append("Google News")
    if result["financials"]:
        result["sources_used"].append("yfinance")
    return result


def context_block(enr: dict) -> str:
    """Render enrichment into a compact text block for an LLM prompt (empty if nothing)."""
    parts = []
    if enr.get("wikipedia"):
        parts.append(f"WIKIPEDIA: {enr['wikipedia']}")
    if enr.get("financials"):
        parts.append(f"FINANCIALS: {enr['financials']}")
    if enr.get("recent_news"):
        parts.append("RECENT COMPANY HEADLINES:\n- " + "\n- ".join(enr["recent_news"][:6]))
    return "\n".join(parts)
