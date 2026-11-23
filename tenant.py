"""Tenant (seller) profile — either the CMI default or inferred from any company name.

This generalizes the hardcoded `config.TENANT` so the newsletter can be produced for ANY
seller: given a company name, the LLM infers what they sell, which customer industries they
target, who their competitors are, and the news queries that surface buying-trigger prospects.
Mirrors the mother repo's onboarding "enrich domain" + niche-query-generation step.
"""
from __future__ import annotations
import json
import warnings
import os
import re
from dataclasses import dataclass, field

import config
import llm


warnings.filterwarnings("ignore", message="Unverified HTTPS request")

_OVERRIDES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "company_overrides.json")
_overrides_cache: dict | None = None


def _overrides_for(company: str) -> dict:
    """Hand-curated facts for a known account, or {}.

    Profile inference is mostly a writing task the fine-tune handles well, but naming
    competitors is factual recall -- and a 7B is unreliable there (it returned CDMOs
    for a cell-therapy developer, then invented a company when asked to check itself).
    Curating the accounts that matter beats re-rolling that dice on every send.
    """
    global _overrides_cache
    if _overrides_cache is None:
        try:
            with open(_OVERRIDES_PATH, encoding="utf-8") as f:
                _overrides_cache = {k.lower(): v for k, v in json.load(f).items()
                                    if isinstance(v, dict)}
        except Exception:  # noqa: BLE001 -- a missing/broken file just means no overrides
            _overrides_cache = {}

    key = (company or "").strip().lower()
    hit = _overrides_cache.get(key)
    if hit:
        return hit

    # Exact-match only silently dropped curated facts: a leads sheet says "VVF LTd",
    # a manual run says "VVF", and the second one got no override at all -- so the
    # careful curation was invisible on exactly the runs it was written for. Compare
    # on a normalised stem instead.
    def _stem(s: str) -> str:
        s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
        s = re.sub(r"\b(ltd|limited|inc|corp|corporation|plc|pvt|private|company|co|"
                   r"group|holdings|india|technologies|labs|laboratories)\b", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    want = _stem(key)
    if not want:
        return {}
    for k, v in _overrides_cache.items():
        have = _stem(k)
        if have and (have == want or (len(want) > 2 and (want in have or have in want))):
            return v
    return {}


def fetch_site_text(url: str, *, max_chars: int = 6000, timeout: int = 12) -> str:
    """Fetch a company's homepage and return its visible text.

    Passing only a URL string into the prompt taught the model nothing -- it cannot
    browse, so it fell back to guessing from the company NAME. That is how ImmunoACT
    (CAR-T cell therapy) was profiled as an "immunoassay services CDMO", which then
    poisoned competitors, keywords and every downstream query. Giving the model the
    actual homepage copy replaces recall with evidence.

    Never raises: an unreachable site degrades to name-only inference, same as before.
    """
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    import requests

    # A real browser User-Agent, not an honest self-identifying one. corcoran.com,
    # qst.go.jp and abahomecare.org all answer 200 to Chrome and stonewall
    # "CMI-brief/1.0" -- three of nine "unreachable" sites in a 20-lead batch were
    # only unreachable because of the header.
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    # Try the bare host, then www. -- a surprising number of company sites serve only
    # one of the two. verify=False because marketing sites routinely mis-serve their
    # intermediate certs, and this only ever reads public HTML.
    host = url.split("://", 1)[1]
    candidates = [url] if host.startswith("www.") else [url, url.replace("://", "://www.", 1)]
    html = ""
    for cand in candidates:
        try:
            r = requests.get(cand, timeout=timeout, headers=headers,
                             allow_redirects=True, verify=False)
            if r.status_code < 400 and len(r.text) > 200:
                html = r.text
                break
        except Exception:  # noqa: BLE001 -- enrichment is best-effort, never fatal
            continue
    if not html:
        return ""

    html = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;?|&#160;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&[a-z]+;|&#\d+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


@dataclass
class Profile:
    company: str
    short: str
    recipient: str
    tagline: str
    value_prop: str
    industries: list[str]                       # customer verticals the seller targets
    competitors: list[str]
    industry_keywords: dict[str, list[str]]     # vertical -> relevancy keywords
    trigger_queries: list[str]                  # Google-News queries for prospect triggers
    industry_queries: list[str]                 # structural / industry-trend queries
    market: str = "Global"                      # target country name, or "Global"

    def __post_init__(self):
        # Lead names arrive raw from the CRM export ("oyin", "HARSHIKA"); fix only
        # single-case names so mixed-case ones ("McKenzie", "van der Berg") are kept.
        r = (self.recipient or "").strip()
        if r and r != "there" and (r.islower() or r.isupper()):
            self.recipient = re.sub(r"[A-Za-z]+", lambda m: m.group(0).capitalize(), r)

    # ---- constructors -------------------------------------------------------
    @classmethod
    def from_company(cls, company: str, website: str = "", recipient: str = "there",
                     country_name: str = "", conn=None) -> "Profile":
        """Infer a full tenant profile from just a company name (LLM), with safe fallbacks.
        If country_name is given, competitors and queries are biased to that market.

        `conn` is optional and enables reusing enrich.py's per-domain company_intel
        cache: when given, a homepage already fetched and cached by stage_enrich is
        read from the DB instead of being fetched again here. Every caller that has a
        DB connection open already (pipeline.py, send_one.py, run_all40.py,
        batch_generate.py) should pass it -- callers that don't (standalone scripts,
        the CLI validator) keep working exactly as before, just without the reuse.

        Raises ValueError if `company` is blank -- there used to be a silent fallback
        here (`cmi_default()`) that profiled the brief as CMI's OWN company instead of
        the lead's. That meant a lead with a blank company field got sent a brief
        written as if Coherent Market Insights were the client -- wrong industries,
        wrong competitors, wrong everything, with no connection to the market/report
        the lead actually asked about. Found 2026-10-07: 45 real sends had already gone
        out this way. A missing company is a data problem to fix upstream (the lead
        import step, not this function), not something to paper over with a fake
        profile -- raising here is what makes that failure visible instead of silently
        mailing the wrong content to a real person.
        """
        company = (company or "").strip()
        market = (country_name or "Global").strip() or "Global"
        if not company:
            raise ValueError(
                "Profile.from_company: company is blank -- refusing to fall back to a "
                "generic CMI-self profile (that fallback used to exist and caused 45 "
                "real sends with wrong industries/competitors/content, 2026-10-07). "
                "Fix the lead's company upstream, or exclude it from this run.")

        geo_line = (f" The target MARKET is {market}: prefer competitors, keywords and news queries "
                    f"relevant to {market}." if market != "Global" else "")
        system = (
            "You are CoherentConnect, a B2B sales-intelligence engine. Given a SELLER company, infer its "
            "sales-prospecting profile. CRITICAL DISTINCTION: a POTENTIAL CUSTOMER is a company that would BUY "
            "the seller's product — it operates in the seller's CUSTOMER industries (its buyers), NOT in the "
            "seller's OWN product category. A company that sells a similar/competing product is a COMPETITOR, "
            "not a customer. So `target_industries` must be the industries of the seller's BUYERS, never the "
            "seller's own category. Example: an AI waste-management/recycling SaaS sells TO consumer-goods "
            "brands, manufacturers, retailers, packaging firms and municipalities (those are target_industries) "
            "— other 'waste management' or 'recycling platform' companies are its COMPETITORS, never "
            "target_industries. A 'buying trigger' is an event (funding, IPO, M&A, expansion, new plant, product "
            "launch, regulatory/compliance obligation, big hire) at a BUYER that creates demand for what the "
            "seller sells." + geo_line + " Return finished values — never placeholders, never angle brackets."
        )
        # Ground the inference in the seller's own homepage copy where we can reach it.
        # Without this the model profiles from the NAME alone and confidently invents a
        # business (see fetch_site_text's docstring).
        #
        # stage_enrich already fetched and cached this exact homepage per DOMAIN (90-day
        # TTL) specifically so the slow fetch happens once per company, not once per
        # brief -- but from_company used to call fetch_site_text() directly every time,
        # ignoring that cache entirely. Check it first; only fetch live on a miss (no
        # conn passed, this domain was never enriched, the entry expired, or the
        # cached source was 'thin'/'curated' rather than a real homepage fetch --
        # curated facts are applied separately below via _overrides_for, and a 'thin'
        # result means the homepage genuinely had nothing usable, so there is nothing
        # to gain from re-fetching it here either).
        site_text = ""
        if conn is not None and website:
            try:
                import json as _json

                import db as _db
                import enrich as _enrich

                domain = _enrich.domain_of(website=website)
                cached = _db.get_intel(conn, domain) if domain else None
                if cached and cached.get("source") == "homepage" and cached.get("raw_json"):
                    site_text = str(_json.loads(cached["raw_json"]).get("homepage_text", ""))[:6000]
            except Exception:  # noqa: BLE001 -- a cache miss/error just falls through to a live fetch
                site_text = ""
        if not site_text and website:
            site_text = fetch_site_text(website)
        # The model is served at --max-model-len 4096. 6000 chars of site_text (~1,700
        # tokens) plus the system/instruction text (~850 tokens) left too little room
        # for the 1500 output tokens requested below -- this 400-error'd on 118/1170
        # briefs in the 2026-10-07 run and again on the first briefs of the 2026-10-08
        # regen, verified against a long homepage scrape (2,597 input tokens on a
        # 4,097-token request). Capped to what fits with real headroom to spare.
        site_text = site_text[:2200]

        user = (
            f"SELLER: {company}\n" + (f"WEBSITE: {website}\n" if website else "") +
            (f"\nWHAT THEIR WEBSITE SAYS (authoritative -- prefer this over anything you "
             f"think you know about the company; if it contradicts your assumption, the "
             f"website is right):\n\"\"\"\n{site_text}\n\"\"\"\n\n" if site_text else "") +
            (f"TARGET MARKET: {market}\n" if market != "Global" else "") +
            "\nReturn a JSON object with keys:\n"
            "  value_prop — one sentence: what the seller sells and to whom\n"
            "  target_industries — 4 to 7 CUSTOMER verticals that BUY from the seller (its buyers). Do NOT list "
            "the seller's own product category here.\n"
            f"  competitors — 4 to 8 objects, each {{\"name\":\"...\", \"their_product\":\"...\"}}, for real "
            f"companies in the seller's OWN product category that sell a directly substitutable "
            f"product{' active in ' + market if market != 'Global' else ''}. \"their_product\" is a few words "
            "stating what THIS company itself makes or sells, in its own right -- not the industry it serves, "
            "not who buys from it. Same vertical only: a buyer chooses the seller OR this company, not both. "
            "Name companies that sell the SAME PRODUCT TYPE stated in value_prop — if the seller makes a cell "
            "therapy, list other cell-therapy developers, not contract manufacturers, not services firms, not "
            "generics makers, however large or nearby they are. Before listing a name, check \"their_product\" "
            "against value_prop yourself: if \"their_product\" is not substantially the same product type as "
            "what the seller sells, do not include that company -- a pharmaceutical API/CDMO maker is not a "
            "competitor to a fragrance/aromatics supplier just because both sell into pharmaceutical "
            "manufacturing, a contract manufacturer is not a competitor to the brand it manufactures for, and a "
            "company whose own product happens to be bought by one of the seller's buyer industries is a "
            "potential CUSTOMER, not a rival, however similar the two industries sound. NEVER list a company "
            "that belongs to any of the target_industries above. A large diversified group that merely operates "
            "nearby (a generics major next to a cell-therapy developer, say) is NOT a competitor unless it "
            "sells the same product category today.\n"
            "  industry_keywords — object mapping each target_industry to a list of 6 to 10 lowercase "
            "keywords/phrases that identify news about companies IN that buyer vertical\n"
            "  trigger_query_seeds — 10 to 14 Google-News queries that surface BUYERS (companies in the "
            "target_industries) with a live buying trigger — combine each buyer vertical with trigger words "
            "(funding, raises, expansion, new plant, launches, compliance, hiring). Do NOT write queries that "
            "surface the seller's competitors.\n"
            "  industry_query_seeds — 5 to 7 Google-News queries for structural/industry-trend stories that "
            "change what the seller's buyers expect"
        )
        data = llm.chat_json(system, user, max_tokens=2500, default={}) or {}

        industries = _as_str_list(data.get("target_industries"))[:7]
        # competitors is now a list of {"name", "their_product"} objects, not bare
        # strings -- added 2026-10-15 so the model has to state what each named
        # company actually sells, not just its name, which lets a deterministic check
        # catch a category mismatch the model's own judgement missed. Confirmed real
        # 2026-10-14: Aroma (fragrance/aromatics ingredients) had "Laurus Labs" (a
        # pharma API/CDMO maker) in its competitor list -- the existing "NEVER list a
        # company that belongs to a target_industry" instruction only catches a
        # competitor whose NAME literally contains an industry word, which a real
        # company name almost never does, so the actual category confusion (a
        # pharma-services company, not a fragrance rival) sailed through untouched.
        raw_competitors = data.get("competitors")
        # Checked against target_industries (the seller's own BUYER verticals), not
        # value_prop -- value_prop's own sentence often names those same verticals
        # ("supplies X to personal care, cosmetics, ... and pharmaceutical
        # manufacturers"), so a naive value_prop/their_product term overlap let every
        # wrongly-named pharma company back in simply because "pharmaceutical" is a
        # buyer vertical mentioned in the seller's own sentence, not because the
        # company's actual product matched. target_industries is the clean, separate
        # list of what the competitor must NOT belong to.
        _stop_terms = {"india", "indian", "sells", "makes", "first", "approved", "supplies",
                      "company", "their", "which", "through", "these", "other", "products",
                      "product", "services", "service", "market", "global", "manufacturers",
                      "manufacturing", "and", "in"}
        _ind_terms_for_check = {w for i in industries
                                for w in re.findall(r"[a-z][a-z0-9-]{3,}", i.lower())
                                if w not in _stop_terms}
        competitors: list[str] = []
        if isinstance(raw_competitors, list):
            for item in raw_competitors:
                if isinstance(item, dict):
                    name = str(item.get("name") or "").strip()
                    their_product = str(item.get("their_product") or "").strip()
                elif isinstance(item, str):
                    # Tolerate the old bare-string shape too (a smaller/older model may
                    # still answer this way despite the schema) -- no their_product to
                    # check against means this entry skips the category check below
                    # rather than being dropped outright.
                    name, their_product = item, ""
                else:
                    continue
                if not name:
                    continue
                if their_product and _ind_terms_for_check:
                    _tp_terms = {w for w in re.findall(r"[a-z][a-z0-9-]{3,}", their_product.lower())
                                if w not in _stop_terms}
                    if _tp_terms & _ind_terms_for_check:
                        continue  # their_product describes a business in one of the
                                  # seller's own BUYER verticals -- a real company,
                                  # but a customer, not a rival; drop it here rather
                                  # than let it reach Competition Outlook.
                competitors.append(name)
        else:
            competitors = _as_str_list(raw_competitors)
        competitors = competitors[:8]

        ov = _overrides_for(company)
        # The model likes to gloss names: "Intas Pharmaceuticals (CAR-T via Actis)".
        # Keep the company only, or nothing downstream can match it to an article.
        competitors = [re.sub(r"\s*[(\[][^)\]]*[)\]]", "", c).strip(" -–,") for c in competitors]
        competitors = [c for c in competitors if len(c) > 2]
        # The model occasionally names a material/chemical/generic noun instead of an
        # actual company ("Fluorocarbon" for a refrigerants seller) -- confirmed
        # 2026-10-14 as one contributor (a minor one; most of the Competition Outlook
        # relevance problem is the selection-fallback fixed in build_competition()
        # above, not this). Tried a company-name-shape heuristic to filter these out
        # automatically, but it also rejected real single-capitalized-word companies
        # (Shopify, Accenture, Novartis, Pfizer, Infosys, Mailchimp all failed a
        # "looks like a brand" check) -- a plain English word being a company's actual
        # name is far too common for a shape-based rule to separate from a genuine
        # material noun. Left unfiltered rather than risk silently dropping real
        # competitors; needs a curated denylist or an LLM check if this recurs often.
        ind_kw = data.get("industry_keywords")
        industry_keywords: dict[str, list[str]] = {}
        if isinstance(ind_kw, dict):
            for k, v in ind_kw.items():
                kws = _as_str_list(v)
                if k and kws:
                    industry_keywords[str(k)] = [w.lower() for w in kws]
        # Enforce the customer/competitor boundary the prompt asks for: a company whose
        # name reads as one of the buyer verticals is a prospect, not a rival.
        _ind_words = {w for i in industries for w in i.lower().replace("&", " ").split()
                      if len(w) > 4 and w not in {"companies", "company", "services", "providers"}}
        competitors = [c for c in competitors
                       if not any(w in c.lower() for w in _ind_words)] or competitors

        # Curated overrides win, applied last so the heuristic filters above cannot
        # strip a hand-checked name. Competitor naming is factual recall, which a 7B
        # gets wrong inconsistently -- see company_overrides.json.
        if ov.get("competitors"):
            competitors = _as_str_list(ov["competitors"])[:8]
        if ov.get("industries"):
            industries = _as_str_list(ov["industries"])[:7]

        trig = _as_str_list(data.get("trigger_query_seeds"))[:14]
        indq = _as_str_list(data.get("industry_query_seeds"))[:7]

        # Fallbacks so the pipeline always has something to search/score.
        if not industries:
            industries = ["Technology", "Healthcare", "Financial Services", "Manufacturing"]
        if not industry_keywords:
            industry_keywords = {i: [i.lower()] for i in industries}
        # Curated query seeds win: they are written as things a headline would actually
        # say. The generated fallback below pastes the industry LABEL into a template,
        # which produces queries like "Home care and detergent manufacturers company
        # raises funding" -- no news story is worded that way, so the feed returns
        # nothing and the Potential Customer slot ends up empty or stretched.
        if ov.get("trigger_queries"):
            trig = _as_str_list(ov["trigger_queries"])[:14]
        if ov.get("industry_queries"):
            indq = _as_str_list(ov["industry_queries"])[:7]

        if not trig:
            # Search the head noun of each vertical, not the whole label. "Cosmetics and
            # toiletries companies" -> "cosmetics", which does appear in headlines.
            heads = []
            for i in industries:
                w = re.split(r"[\s,&/]+", i.strip())
                heads += [x.lower() for x in w[:2] if len(x) > 3 and
                          x.lower() not in {"and", "companies", "manufacturers", "brands",
                                            "buyers", "makers", "providers", "institutional"}]
            heads = list(dict.fromkeys(heads))[:6] or ["industry"]
            mkt = market if market != "Global" else ""
            for h in heads:
                trig.append(f"{h} company {mkt} raises funding".strip())
                trig.append(f"{h} brand {mkt} expansion new plant".strip())
            trig = trig[:14]
        if not indq:
            base = industries[0].split(" and ")[0] if industries else "industry"
            indq = [f"{base} {market} industry outlook".strip(),
                    f"{base} {market} demand trends".strip()]

        short = _short_name(company)
        vp = (ov.get("value_prop") or "").strip() or \
             (data.get("value_prop") or "").strip() or f"{company}'s products and services."
        return cls(
            company=company, short=short, recipient=recipient or "there",
            tagline="Sales intelligence, made coherent.", value_prop=vp,
            industries=industries, competitors=competitors,
            industry_keywords=industry_keywords, trigger_queries=trig, industry_queries=indq,
            market=market,
        )


def _as_str_list(v) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def _short_name(company: str) -> str:
    words = [w for w in company.replace(",", " ").split() if w]
    drop = {"inc", "inc.", "llc", "ltd", "ltd.", "corp", "corp.", "co", "co.", "plc", "the"}
    core = [w for w in words if w.lower().strip(".") not in drop]
    if len(core) >= 2 and all(len(w) > 1 for w in core[:3]):
        acr = "".join(w[0] for w in core[:3]).upper()
        if len(acr) >= 2:
            return acr
    return (core[0] if core else company)[:16]
