"""Read smartcrawl intel dumps and turn them into tenant.Profile facts.

The pipeline's worst failures all trace to one step: `Profile.from_company` asking a
7B model what a company does. It answers well for companies it knows (Paytm -> PhonePe
is correct) and invents for ones it does not (VVF, India's largest oleochemicals maker
and a contract soap manufacturer for J&J and Henkel, got "Dentsu India" -- an ad agency
-- as its competitor).

`tenant.fetch_site_text` was meant to fix that by grounding the answer in the homepage,
and it does when the homepage has content: ImmunoACT's gave 6000 chars and produced a
correct CAR-T profile. VVF's homepage yields 301 characters of pure navigation
("CELEBRATING 75 YEARS | Home | About Us"), so there was nothing to ground on.

smartcrawl already solves this properly -- it crawls About/Business/Product pages and
LLM-extracts a structured INTEL_SCHEMA including `relationships.competitors` and
`business.products[].target_customers`. It lives in another project with its own
dependency tree (scrapling, a Lightpanda browser, backend.*), so rather than port it,
this module consumes its JSON output.

    # in the smartcrawl project
    python -c "import asyncio,json;from smartcrawl import smart_crawl; \
               print(json.dumps(asyncio.run(smart_crawl('vvfltd.com'))))" > vvfltd.json

    # here
    python company_intel.py --import vvfltd.json --company "VVF"

Writes into company_overrides.json, which tenant.Profile already prefers over model
inference. Nothing else in the pipeline changes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
INTEL_DIR = os.path.join(HERE, "company_intel")
OVERRIDES_PATH = os.path.join(HERE, "company_overrides.json")


def _first_str(*vals) -> str:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _as_list(v) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def profile_facts(intel: dict) -> dict:
    """Map a smartcrawl INTEL_SCHEMA dump onto the keys company_overrides.json uses.

    Only fields the crawler actually found are emitted -- an empty list here is worse
    than no key at all, because an override wins over inference and would replace a
    guess with nothing.
    """
    data = intel.get("data") or intel
    company = data.get("company") or {}
    business = data.get("business") or {}
    rel = data.get("relationships") or {}

    out: dict = {}

    # value_prop: the crawler's own description, else tagline.
    desc = _first_str(company.get("description"), company.get("tagline"))
    if desc:
        name = _first_str(company.get("name"), company.get("legal_name"))
        # Lead with the company name so downstream prompts read naturally.
        out["value_prop"] = desc if (name and name.lower() in desc.lower()[:60]) \
            else (f"{name} {desc}" if name else desc)

    # industries = who BUYS, not what the seller is. products[].target_customers is
    # the crawler's closest field; fall back to key_markets.
    buyers: list[str] = []
    for p in (business.get("products") or []):
        buyers += _as_list(p.get("target_customers"))
    for s in (business.get("services") or []):
        buyers += _as_list(s.get("target_customers"))
    if not buyers:
        buyers = _as_list(business.get("key_markets"))
    seen: set[str] = set()
    buyers = [b for b in buyers if not (b.lower() in seen or seen.add(b.lower()))]
    if buyers:
        out["industries"] = buyers[:7]

    # competitors: the field the model gets most wrong, and the reason this module
    # exists. For VVF it returned "Dabur India, Nirma, Nestle, Reckitt Benckiser, P&G,
    # UniqYou" -- Reckitt and P&G are VVF's CUSTOMERS (it contract-manufactures their
    # soap), and it listed "P&G" as a rival while listing "Procter & Gamble" as a
    # client in the same response. Cross-checking the lists catches that; a name
    # cannot be both, and the customer role is the one the site actually evidences.
    comps = _as_list(rel.get("competitors"))
    cust_names = []
    for c in (rel.get("customers") or []):
        cust_names.append(c if isinstance(c, str) else str(c.get("name", "")))
    for p in (rel.get("partnerships") or []):
        cust_names.append(p if isinstance(p, str) else str(p.get("name", "")))

    def _norm_co(s: str) -> str:
        s = re.sub(r"\s*[(\[][^)\]]*[)\]]", " ", (s or "").lower())
        s = re.sub(r"\b(ltd|limited|inc|corp|plc|pvt|private|international|india|"
                   r"industries|group|holdings|company|co)\b", " ", s)
        s = re.sub(r"[^a-z0-9& ]", " ", s)
        s = s.replace("procter & gamble", "p&g").replace("johnson & johnson", "j&j")
        return re.sub(r"\s+", " ", s).strip()

    blocked = {_norm_co(n) for n in cust_names if n}
    kept, dropped = [], []
    for c in comps:
        n = _norm_co(c)
        # substring match either way: "P&G" vs "Procter & Gamble" normalise together,
        # and "Reckitt" should match "Reckitt Benckiser".
        if n and any(n == b or (len(n) > 3 and (n in b or b in n)) for b in blocked):
            dropped.append(c)
        else:
            kept.append(c)
    if dropped:
        out["_competitors_dropped"] = dropped
        out["_competitors_note"] = ("dropped because the same crawl lists them as customers "
                                    "or partners -- verify before re-adding")
    if kept:
        out["competitors"] = kept[:8]
    if cust_names:
        out["known_customers"] = [c for c in dict.fromkeys(cust_names) if c][:8]

    # caps_industry names the seller's OWN market for the Inside-CMI panel.
    inds = _as_list(business.get("industries"))
    if inds:
        out["caps_industry"] = ", ".join(inds[:2]).lower()

    # product_terms anchor competitor news to the shared category (see
    # brief_builder.build_competition). Built from product and service names.
    terms: list[str] = []
    for p in (business.get("products") or []):
        terms += [t for t in (_first_str(p.get("name")), _first_str(p.get("category"))) if t]
    for s in (business.get("services") or []):
        terms += [t for t in (_first_str(s.get("name")), _first_str(s.get("category"))) if t]
    terms += inds
    seen = set()
    terms = [t.lower() for t in terms if len(t) > 3
             and not (t.lower() in seen or seen.add(t.lower()))]
    if terms:
        out["product_terms"] = terms[:14]

    return out


def merge_into_overrides(company: str, facts: dict, *, replace: bool = False) -> dict:
    key = company.strip().lower()
    try:
        with open(OVERRIDES_PATH, encoding="utf-8") as f:
            store = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        store = {}

    existing = store.get(key) if isinstance(store.get(key), dict) else {}
    # Hand-curated values win unless --replace: someone checked those by hand, and a
    # crawler's guess should not silently overwrite a verified fact.
    merged = dict(facts) if replace else {**facts, **existing}
    merged["_source"] = "smartcrawl" if replace or not existing else "smartcrawl+curated"
    store[key] = merged

    with open(OVERRIDES_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description="Import a smartcrawl intel dump into company_overrides.json.")
    ap.add_argument("--import", dest="path", required=True, help="smartcrawl JSON output")
    ap.add_argument("--company", required=True, help="company name as it appears in the leads sheet")
    ap.add_argument("--replace", action="store_true",
                    help="overwrite hand-curated values instead of preserving them")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    with open(args.path, encoding="utf-8") as f:
        intel = json.load(f)

    facts = profile_facts(intel)
    if not facts:
        raise SystemExit("no usable fields found -- is this a smartcrawl INTEL_SCHEMA dump?")

    print(f"extracted for {args.company!r}:")
    for k, v in facts.items():
        shown = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        print(f"  {k:<16} {shown[:110]}")

    if args.dry_run:
        print("\n(dry run -- nothing written)")
        return
    merge_into_overrides(args.company, facts, replace=args.replace)
    print(f"\nwrote to {os.path.relpath(OVERRIDES_PATH)}")


if __name__ == "__main__":
    main()
