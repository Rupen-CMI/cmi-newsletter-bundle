"""Outlet -> country map, learned from collected articles.

For each news source, count which country its articles name (regions.article_countries)
plus, for articles that came from a market's LOCAL-language Google News edition, that
edition's country. A source whose articles are >= MIN_SHARE one country over >= MIN_N
articles belongs to that country. geo.filter_for_lead then treats a no-country article
from such a source as being about that country.

Persisted in the source_country table, updated incrementally each collect run; with no
DB connection the map lives in memory for the process.
"""
from __future__ import annotations
import threading

import regions

MIN_N = 5
MIN_SHARE = 0.70

_SKIP = {"", "google news", "bing news"}
_LOCK = threading.Lock()
# source_lower -> (iso2, n, share)
_MEM: dict[str, tuple[str, int, float]] = {}


def _norm(source: str) -> str:
    return (source or "").strip().lower()


def _batch_counts(articles) -> dict[str, tuple[int, dict[str, int]]]:
    out: dict[str, tuple[int, dict[str, int]]] = {}
    for a in articles:
        src = _norm(getattr(a, "source", ""))
        if src in _SKIP:
            continue
        isos = set(regions.article_countries(a))
        ed = (getattr(a, "edition_iso2", "") or "").upper()
        if ed:
            isos.add(ed)
        n, counts = out.setdefault(src, (0, {}))
        for iso in isos:
            counts[iso] = counts.get(iso, 0) + 1
        out[src] = (n + 1, counts)
    return out


def learn(articles, conn=None) -> int:
    """Fold this run's articles into the map (memory, and the DB when `conn` is given).
    Returns the number of sources updated. Never raises on a DB problem."""
    batch = _batch_counts(articles)
    if not batch:
        return 0
    prior: dict[str, tuple[str, int, float]] = {}
    if conn is not None:
        try:
            import db as _db
            prior = {s: (d["iso2"], d["n"], d["share"])
                     for s, d in _db.get_source_countries(conn).items()}
        except Exception:  # noqa: BLE001
            prior = {}
    updated = 0
    with _LOCK:
        for src, (nb, counts) in batch.items():
            old = _MEM.get(src) or prior.get(src)
            if old:
                oiso, on, oshare = old
                counts = dict(counts)
                counts[oiso] = counts.get(oiso, 0) + round(oshare * on)
                total = on + nb
            else:
                total = nb
            if not counts:
                continue
            iso, c = max(counts.items(), key=lambda kv: kv[1])
            share = min(1.0, c / total) if total else 0.0
            _MEM[src] = (iso, total, share)
            updated += 1
            if conn is not None:
                try:
                    import db as _db
                    _db.put_source_country(conn, src, iso, total, share)
                except Exception:  # noqa: BLE001
                    pass
    return updated


def learned_map(conn=None) -> dict[str, str]:
    """{source_lower: iso2} for sources that qualify (>= MIN_N articles, >= MIN_SHARE)."""
    rows: dict[str, tuple[str, int, float]] = {}
    if conn is not None:
        try:
            import db as _db
            rows = {s: (d["iso2"], d["n"], d["share"])
                    for s, d in _db.get_source_countries(conn).items()}
        except Exception:  # noqa: BLE001
            rows = {}
    with _LOCK:
        rows.update(_MEM)
    return {s: iso for s, (iso, n, share) in rows.items() if n >= MIN_N and share >= MIN_SHARE}


def reset_memory() -> None:
    with _LOCK:
        _MEM.clear()
