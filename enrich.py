"""Resolve what a company actually does, once per domain, into `company_intel`.

The worst failure in this pipeline is not a writing failure. It is the model inventing a
business: ImmunoACT, which makes CAR-T cell therapy, was profiled from its NAME alone as
an "immunoassay CDMO", and every downstream step inherited that -- wrong competitors,
wrong queries, and a pet-food delivery company in the Potential Customer slot.

`tenant.fetch_site_text` fixes it when the homepage has content. Often it does not:
VVF's homepage yields 301 characters of pure navigation ("CELEBRATING 75 YEARS | Home |
About Us"), so there was nothing to ground on and the model guessed again.

The ladder, strongest first:

    curated      company_overrides.json        hand-checked, always wins
    smartcrawl   assets/smartcrawl.py dump     About/Products pages, LLM-extracted
    homepage     tenant.fetch_site_text        fast, often enough
    (nothing)    profile falls back to name-only inference, as before

Results are cached per DOMAIN with a 90-day expiry, never per brief: smartcrawl takes
~20 minutes per company on the local model, and a lead list shares far fewer domains
than it has rows.

    python enrich.py --xlsx Book1.xlsx            # fill the cache for a lead list
    python enrich.py --domain vvfltd.com --show   # inspect one
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys

# Personal / free-mail domains. The single list: batch_generate.py imports it.
FREE_MAIL = {
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.in", "yahoo.co.uk", "yahoo.fr",
    "yahoo.de", "ymail.com", "rocketmail.com", "hotmail.com", "hotmail.co.uk", "hotmail.fr",
    "hotmail.de", "hotmail.it", "outlook.com", "outlook.de", "live.com", "live.com.au",
    "live.co.uk", "live.de", "live.fr", "msn.com", "icloud.com", "me.com", "mac.com",
    "aol.com", "protonmail.com", "proton.me", "pm.me", "tutanota.com", "zoho.com",
    "zohomail.com", "mail.com", "gmx.net", "gmx.de", "gmx.com", "gmx.ch", "gmx.at",
    "web.de", "t-online.de", "freenet.de", "yandex.ru", "yandex.com", "mail.ru", "bk.ru",
    "inbox.ru", "list.ru", "rambler.ru", "qq.com", "163.com", "126.com", "sina.com",
    "sohu.com", "aliyun.com", "naver.com", "daum.net", "hanmail.net", "rediffmail.com",
    "libero.it", "virgilio.it", "orange.fr", "free.fr", "laposte.net", "sfr.fr", "wp.pl",
    "o2.pl", "onet.pl", "interia.pl", "seznam.cz", "bigpond.com", "optusnet.com.au",
    "shaw.ca", "rogers.com", "sympatico.ca", "comcast.net", "verizon.net", "att.net",
    "sbcglobal.net", "btinternet.com", "sky.com", "virginmedia.com", "terra.com.br",
    "uol.com.br", "bol.com.br",
}

_NO_COMPANY = {"", "none", "n/a", "na", "-", "null", "nil", "not available"}


_CC_TLD = {"de": "DE", "fr": "FR", "it": "IT", "ch": "CH", "uk": "GB", "ca": "CA", "au": "AU",
           "jp": "JP", "kr": "KR", "cn": "CN", "es": "ES", "nl": "NL", "ae": "AE", "in": "IN",
           "be": "BE", "at": "AT", "se": "SE", "no": "NO", "dk": "DK", "fi": "FI", "pl": "PL",
           "pt": "PT", "ie": "IE", "br": "BR", "mx": "MX", "sg": "SG", "nz": "NZ", "za": "ZA",
           "tr": "TR", "ru": "RU", "sa": "SA", "hk": "HK", "tw": "TW", "il": "IL", "lu": "LU"}
_COUNTRY_NAMES = {
    "korea": "KR", "south korea": "KR", "republic of korea": "KR", "germany": "DE",
    "deutschland": "DE", "france": "FR", "italy": "IT", "italia": "IT", "switzerland": "CH",
    "schweiz": "CH", "suisse": "CH", "united kingdom": "GB", "uk": "GB", "england": "GB",
    "canada": "CA", "australia": "AU", "japan": "JP", "china": "CN", "spain": "ES",
    "netherlands": "NL", "the netherlands": "NL", "united arab emirates": "AE", "uae": "AE",
    "india": "IN", "belgium": "BE", "austria": "AT", "sweden": "SE", "norway": "NO",
    "denmark": "DK", "finland": "FI", "poland": "PL", "portugal": "PT", "ireland": "IE",
    "brazil": "BR", "mexico": "MX", "singapore": "SG", "new zealand": "NZ",
    "south africa": "ZA", "turkey": "TR", "saudi arabia": "SA", "hong kong": "HK",
    "taiwan": "TW", "israel": "IL", "luxembourg": "LU", "united states": "US", "usa": "US",
    "u.s.a.": "US"}


def hq_country_from_site(text: str, domain: str = "") -> str:
    """ISO2 of the company's HQ when the homepage states it: an explicit country name
    closing a postal-address line ("..., Seoul, Korea") wins, else a country-code TLD.
    Generic TLDs and ambiguous text give ''."""
    names = sorted(_COUNTRY_NAMES, key=len, reverse=True)
    pat = re.compile(r",\s*(" + "|".join(re.escape(n) for n in names) + r")\b\.?(?=\s*(?:[,;|.)]|\d|$)|\s+(?:tel|phone|fax|e-?mail|copyright|all|©)\b)",
                     re.I)
    found = {_COUNTRY_NAMES[m.group(1).lower()] for m in pat.finditer(text or "")}
    if len(found) == 1:
        return found.pop()
    if len(found) == 0:
        labels = _host(domain).split(".")
        if len(labels) >= 2 and labels[-1] in _CC_TLD:
            return _CC_TLD[labels[-1]]
    return ""


def _host(url_or_domain: str) -> str:
    d = (url_or_domain or "").split("://")[-1].split("/")[0].lower().strip()
    return d[4:] if d.startswith("www.") else d


def is_personal_domain(domain: str) -> bool:
    return _host(domain) in FREE_MAIL


def company_info(email: str, company: str = "", website: str = "") -> tuple[str, str]:
    """(company, website) for a lead, '' where there is no real company information.

    A personal email domain is never a company: the website is dropped, and a company
    field that only repeats the mail provider ("Gmail", "gmx", "icloud.com") is blanked.
    A real company name from the lead sheet is kept even when the email is personal.
    With a company email domain and no company name, the domain's label is the name.
    """
    email_dom = _host(email.split("@")[-1]) if "@" in (email or "") else ""
    web_dom = _host(website)
    if web_dom and web_dom in FREE_MAIL:
        web_dom = ""
    if not web_dom and email_dom and email_dom not in FREE_MAIL:
        web_dom = email_dom
    site = f"https://{web_dom}" if web_dom else ""

    name = (company or "").strip()
    if name.lower() in _NO_COMPANY:
        name = ""
    squashed = re.sub(r"\s+", "", name.lower())
    if squashed in FREE_MAIL:
        name = ""
    elif email_dom in FREE_MAIL and squashed in {email_dom, email_dom.split(".")[0]}:
        name = ""
    if not name and web_dom:
        name = web_dom.split(".")[0].title()
    return name, site

# Below this, a homepage is navigation chrome rather than a description of the business.
# Measured: VVF's homepage is 301 chars of menu links; ImmunoACT's is 6,000 chars of
# real copy and produced a correct profile.
MIN_USEFUL_CHARS = 400


def domain_of(email: str = "", website: str = "") -> str:
    """The company domain for a lead, or '' when there is none worth using.

    A free-mail address is not a company site -- guessing from one is how a gmail lead
    ended up profiled from its display name.
    """
    if website:
        d = _host(website)
        return "" if d in FREE_MAIL else d
    if "@" in email:
        d = _host(email.split("@")[-1])
        return "" if d in FREE_MAIL else d
    return ""


def _curated_for(company: str, domain: str) -> dict:
    """company_overrides.json entry for this company or domain, if one exists."""
    try:
        import tenant

        hit = tenant._overrides_for(company) if company else {}
        if not hit and domain:
            hit = tenant._overrides_for(domain.split(".")[0])
        return hit or {}
    except Exception:  # noqa: BLE001 -- overrides are optional
        return {}


def _fetch_facts(domain: str, company: str = "") -> dict:
    """Gather facts for one domain WITHOUT touching the database.

    Split from the write deliberately: a sqlite3.Connection may only be used on the
    thread that created it, so the network work fans out across a pool and the single
    write happens back on the caller's thread.
    """
    curated = _curated_for(company, domain)
    if curated:
        return {"domain": domain, "company": company, "source": "curated",
                "value_prop": curated.get("value_prop", ""),
                "industries": curated.get("industries") or [],
                "competitors": curated.get("competitors") or [],
                "product_terms": curated.get("product_terms") or [],
                "raw_json": json.dumps(curated, ensure_ascii=False), "chars": -1}

    try:
        import tenant

        text = tenant.fetch_site_text(f"https://{domain}", max_chars=6000)
    except Exception:  # noqa: BLE001 -- an unreachable site is a normal outcome
        text = ""

    # A thin result is still cached, on a short TTL: without a row, every run re-fetches
    # a site already shown to have nothing. `source='thin'` is the smartcrawl queue.
    usable = len(text) >= MIN_USEFUL_CHARS
    return {"domain": domain, "company": company,
            "source": "homepage" if usable else "thin",
            "value_prop": text[:1200] if usable else text,
            "industries": [], "competitors": [], "product_terms": [],
            "raw_json": json.dumps({"homepage_text": text}, ensure_ascii=False),
            "chars": len(text)}


def _store(conn, facts: dict) -> None:
    import db

    db.put_intel(conn, domain=facts["domain"], company=facts.get("company", ""),
                 value_prop=facts.get("value_prop", ""),
                 industries=facts.get("industries") or [],
                 competitors=facts.get("competitors") or [],
                 product_terms=facts.get("product_terms") or [],
                 raw_json=facts.get("raw_json", ""),
                 source=facts["source"],
                 ttl_days=7 if facts["source"] == "thin" else 90)


def enrich_domain(conn, domain: str, *, company: str = "", force: bool = False,
                  log=print) -> dict:
    """Ensure `company_intel` holds usable facts for `domain`. Returns the row.

    Never raises: enrichment is best-effort, and a failure degrades to whatever the
    profile step can infer on its own -- the behaviour that existed before this module.
    """
    import db

    if not domain:
        return {}
    if not force:
        hit = db.get_intel(conn, domain)
        if hit:
            return hit

    facts = _fetch_facts(domain, company)
    if facts["source"] == "curated":
        log(f"    · {domain}: curated override")
    elif facts["source"] == "homepage":
        log(f"    · {domain}: homepage {facts['chars']} chars")
    else:
        log(f"    · {domain}: homepage thin ({facts['chars']} chars) — needs smartcrawl")
    _store(conn, facts)
    return db.get_intel(conn, domain) or {}


def enrich_leads(conn, leads: list[dict], *, workers: int = 8, force: bool = False,
                 log=print) -> dict:
    """Enrich every distinct domain across a lead list. Returns a small summary.

    Deduplicating by domain first is the whole point: 2,606 leads carry far fewer
    companies, and the slow paths run once each.
    """
    seen: dict[str, str] = {}
    for l in leads:
        d = domain_of(l.get("email", ""), l.get("website", ""))
        if d and d not in seen:
            seen[d] = l.get("company", "")

    todo = list(seen) if force else [d for d in seen if not _has_fresh(conn, d)]
    log(f"  {len(seen)} distinct domains, {len(todo)} need enrichment")
    if not todo:
        return {"domains": len(seen), "enriched": 0, "thin": 0}

    # Fetch in parallel (network-bound, no DB), then write on this thread: a
    # sqlite3.Connection is only valid on the thread that opened it.
    thin = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_facts, d, seen[d]): d for d in todo}
        for fut in cf.as_completed(futs):
            d = futs[fut]
            try:
                facts = fut.result()
            except Exception as e:  # noqa: BLE001
                log(f"    · {d}: {type(e).__name__}: {str(e)[:60]}")
                continue
            _store(conn, facts)
            if facts["source"] == "curated":
                log(f"    · {d}: curated override")
            elif facts["source"] == "homepage":
                log(f"    · {d}: homepage {facts['chars']} chars")
            else:
                thin += 1
                log(f"    · {d}: thin ({facts['chars']} chars) — needs smartcrawl")
    return {"domains": len(seen), "enriched": len(todo), "thin": thin}


def _has_fresh(conn, domain: str) -> bool:
    import db

    return db.get_intel(conn, domain) is not None


def thin_domains(conn) -> list[str]:
    """Domains whose homepage gave too little -- the smartcrawl queue."""
    return [r[0] for r in conn.execute(
        "SELECT domain FROM company_intel WHERE source = 'thin' ORDER BY domain")]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fill company_intel for a lead list.")
    ap.add_argument("--xlsx")
    ap.add_argument("--domain", help="enrich just this one")
    ap.add_argument("--company", default="")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="re-fetch even if cached")
    ap.add_argument("--show", action="store_true", help="print what is cached and exit")
    ap.add_argument("--thin", action="store_true", help="list domains needing smartcrawl")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    import db

    conn = db.get_conn()
    db.init_db(conn)

    if args.thin:
        for d in thin_domains(conn):
            print(" ", d)
        return

    if args.domain:
        if args.show:
            row = db.get_intel(conn, args.domain)
            print(json.dumps(row, indent=2, ensure_ascii=False) if row else "(not cached)")
            return
        row = enrich_domain(conn, args.domain, company=args.company, force=args.force)
        print(json.dumps({k: v for k, v in (row or {}).items() if k != "raw_json"},
                         indent=2, ensure_ascii=False))
        return

    if not args.xlsx:
        raise SystemExit("pass --xlsx or --domain")

    sys.argv = ["x"]
    import batch_generate as bg

    leads = bg.read_leads(args.xlsx, args.limit)
    print(f"{len(leads)} leads")
    summary = enrich_leads(conn, leads, workers=args.workers, force=args.force)
    print(f"\n{summary['domains']} domains | {summary['enriched']} enriched | "
          f"{summary['thin']} thin (need smartcrawl)")
    thin = thin_domains(conn)
    if thin:
        print("\nrun smartcrawl for:")
        for d in thin[:20]:
            print("  ", d)


if __name__ == "__main__":
    main()
