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
import threading
import dataclasses
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


def _decode_response(r) -> str:
    """Decode a response honouring <meta charset>; requests falls back to ISO-8859-1 when
    the header has no charset, which turns UTF-8 Japanese pages into mojibake."""
    raw = getattr(r, "content", b"") or b""
    enc = (getattr(r, "encoding", None) or "").lower()
    if raw and (not enc or enc in ("iso-8859-1", "latin-1", "latin1")):
        m = re.search(rb"<meta[^>]+charset=[\"']?\s*([A-Za-z0-9_\-]+)", raw[:4096], re.I)
        cands = [m.group(1).decode("ascii", "ignore")] if m else []
        cands += [getattr(r, "apparent_encoding", None) or "", "utf-8"]
        for c in cands:
            if not c:
                continue
            try:
                return raw.decode(c)
            except (LookupError, UnicodeDecodeError):
                continue
        return raw.decode("utf-8", "replace")
    try:
        return r.text
    except Exception:  # noqa: BLE001
        return raw.decode("utf-8", "replace")


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
            if r.status_code < 400:
                cand_html = _decode_response(r)
                if len(cand_html) > 200:
                    html = cand_html
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
    # Added after profiles were already stored as profile_json: every field below has a
    # default so an old stored dict (without them) still loads via Profile(**d).
    lead_type: str = "seller"                   # seller|investor|nonprofit|education|government
    products: list = field(default_factory=list)         # [{"name", "application"}], 3-8
    buyer_functions: list = field(default_factory=list)  # who physically buys/uses the product
    profile_confidence: str = "normal"          # "low" when homepage text was missing/thin
    competitor_query_seeds: list = field(default_factory=list)  # generic category queries
    product_category: str = ""                  # 1-3 words news uses: "EV charging", "IT services"
    # Local-language search (non-English markets only; "" / [] / {} for English markets).
    local_language: str = ""                    # e.g. "Polish"; "" when the market's language is English
    local_queries: list = field(default_factory=list)    # 10-12 short queries in that language
    local_competitor_names: dict = field(default_factory=dict)  # english name -> name as written in local press
    # competitor name (as stored in `competitors`) -> other common written forms (English
    # form, local form, short form / acronym). Old stored profiles load with {}.
    competitor_aliases: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        """Build from a stored dict, ignoring unknown keys and defaulting missing ones."""
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def __post_init__(self):
        # Lead names arrive raw from the CRM export ("oyin", "HARSHIKA"); fix only
        # single-case names so mixed-case ones ("McKenzie", "van der Berg") are kept.
        r = (self.recipient or "").strip()
        if r and r != "there" and (r.islower() or r.isupper()):
            self.recipient = re.sub(r"[A-Za-z]+", lambda m: m.group(0).capitalize(), r)

    # ---- constructors -------------------------------------------------------
    @classmethod
    def from_company(cls, company: str, website: str = "", recipient: str = "there",
                     country_name: str = "", conn=None, *,
                     subject_is_market: bool = False, topic_hint: str = "") -> "Profile":
        """Infer a full tenant profile from just a company name (LLM), with safe fallbacks.
        If country_name is given, competitors and queries are biased to that market.

        subject_is_market=True: `company` is the MARKET a lead enquired about (a lead with
        no company information, e.g. a personal email address) -- the profile describes a
        typical company selling in that market, and its competitors are the market's
        leading companies.

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

        local_lang = _local_language_for(market)
        in_market = f" in {market}" if market != "Global" else ""
        # Rewritten 2026-11-26 as short plain rules for the 9B model; the long version with
        # worked examples (AI waste SaaS, Laurus Labs vs Aroma) is in git history.
        system = (
            "You are a B2B sales researcher. Given a SELLER company, describe who it sells to, who it "
            "competes with, and which news would show a buyer is ready to buy.\n"
            "- A buyer is a company that would BUY the seller's product.\n"
            "- A competitor sells the SAME kind of product as the seller. A competitor is never a buyer.\n"
            "- A buying trigger is news at a buyer that creates demand for the seller's product "
            "(expansion, new plant, funding, launch, new rules).\n"
            "- Write finished values. No placeholders."
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
        # The model used to be served at --max-model-len 4096, where 6000 chars of
        # site_text overflowed (400-errors on 118/1170 briefs, 2026-10-07), so this was
        # capped at 2200. It is now served at 8192. Budget at ~3.5 chars/token: 5000
        # chars of site_text (~1,430 tokens) + system/instruction text (~1,100 tokens
        # incl. the products/buyer_functions keys) + 2500 output tokens requested below
        # = ~5,000 tokens, leaving ~3,000 of headroom under 8192.
        site_text = site_text[:5000]
        low_confidence = len(site_text.strip()) < 200
        company = _fix_company_case(company, site_text)

        subject_line = (
            f"MARKET: {company}\n(This is a market the reader asked about, not a company. Describe a "
            "typical company that sells in this market; its competitors are the market's leading "
            "companies.)\n" if subject_is_market else f"SELLER: {company}\n")
        user = (
            subject_line + (f"WEBSITE: {website}\n" if website else "") +
            (f"\nWEBSITE TEXT (trust this over what you think you know):\n\"\"\"\n{site_text}\n\"\"\"\n\n"
             if site_text else "") +
            (f"TARGET MARKET: {market}\n" if market != "Global" else "") +
            (f"ENQUIRY: the lead asked for a report on \"{topic_hint.strip()}\". Use it as a hint only "
             "where the website agrees.\n" if (topic_hint or "").strip() and not subject_is_market else "") +
            "\nReturn JSON with these keys. Write every value in English"
            + (f", except local_competitor_names, which is in {local_lang}"
               if local_lang else "")
            + ".\n"
            "- value_prop: one sentence: what the seller sells and to whom.\n"
            "- target_industries: 4 to 7 industries that BUY from the seller. Not the seller's own industry.\n"
            f"- competitors: 4 to 8 real companies{in_market} that sell the same kind of product as the "
            "seller, each {\"name\": \"...\", \"their_product\": \"what this company sells\"}. Leave out "
            "any company whose product is a different type, any buyer, the seller itself and its brands. "
            "Plain names, no country added.\n"
            "- industry_keywords: for each target industry, 6 to 10 lowercase keywords for news about "
            "companies in it.\n"
            "- lead_type: one of seller, investor, nonprofit, education, government.\n"
            "- products: 3 to 8 of the seller's own product lines from the website, each "
            "{\"name\": \"...\", \"application\": \"what it is used for\"}. [] if the website names none.\n"
            "- buyer_functions: 2 to 6 teams or roles inside a buyer that order the product.\n"
            + (
                f"\n- local_competitor_names: each competitor name as {local_lang} news writes it (native "
                "script where the language uses one)."
                if local_lang else "")
            + "\n- competitor_aliases: for each competitor, 1 to 3 other names news uses for it (English "
              "form, local form, short form or acronym), or [] if none. Real forms only."
        )
        data = llm.chat_json(system, user, temperature=0,
                             max_tokens=2500 if local_lang else 2000, default={}) or {}

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
        competitors = _clean_competitors(competitors)
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

        # Search queries come from their own small call (write_search_queries), made after
        # the profile call; an empty/failed result falls through to the fallbacks below.
        try:
            q = write_search_queries(
                (ov.get("value_prop") or "").strip() or (data.get("value_prop") or "").strip(),
                industries, competitors, market, local_lang)
        except Exception:  # noqa: BLE001 -- queries are optional; never fail the profile
            q = {}
        trig = list(q.get("trigger_queries") or [])[:12]
        indq = list(q.get("industry_queries") or [])[:7]

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

        short = company if subject_is_market else _short_name(company)
        vp =(ov.get("value_prop") or "").strip() or \
             (data.get("value_prop") or "").strip() or f"{company}'s products and services."
        lead_type = str(data.get("lead_type") or "").strip().lower()
        if lead_type not in LEAD_TYPES:
            lead_type = "seller"
        products: list[dict] = []
        raw_products = data.get("products")
        if isinstance(raw_products, list):
            for p in raw_products:
                if isinstance(p, dict):
                    pname = str(p.get("name") or "").strip()
                    papp = str(p.get("application") or "").strip()
                elif isinstance(p, str):
                    pname, papp = p.strip(), ""
                else:
                    continue
                if pname and pname.lower() not in {x["name"].lower() for x in products}:
                    products.append({"name": pname, "application": papp})
        if site_text.strip():
            # A 7B invents product names; keep only those the homepage actually shows.
            products = [p for p in products if _product_grounded(p["name"], site_text)]
        products = products[:8]
        buyer_functions = _as_str_list(data.get("buyer_functions"))[:6]
        competitor_query_seeds = list(q.get("competitor_query_seeds") or [])[:5]
        product_category = str(q.get("product_category") or "").strip()
        local_queries: list[str] = []
        local_competitor_names: dict[str, str] = {}
        if local_lang:
            local_queries = list(q.get("local_queries") or [])[:12]
            raw_ln = data.get("local_competitor_names")
            if isinstance(raw_ln, dict):
                # Keep only entries for competitors actually in the final list (the list
                # can be curated/overridden above), matched case-insensitively.
                want = {c.lower(): c for c in competitors}
                for k, v in raw_ln.items():
                    en, loc = str(k).strip(), str(v or "").strip()
                    if loc and en.lower() in want:
                        local_competitor_names[want[en.lower()]] = loc
        competitor_aliases = _parse_competitor_aliases(
            data.get("competitor_aliases"), competitors, local_competitor_names)
        profile = cls(
            company=company, short=short, recipient=recipient or "there",
            tagline="Sales intelligence, made coherent.", value_prop=vp,
            industries=industries, competitors=competitors,
            industry_keywords=industry_keywords, trigger_queries=trig, industry_queries=indq,
            market=market, lead_type=lead_type, products=products,
            buyer_functions=buyer_functions,
            competitor_query_seeds=competitor_query_seeds,
            product_category=product_category,
            local_language=local_lang, local_queries=local_queries,
            local_competitor_names=local_competitor_names,
            competitor_aliases=competitor_aliases,
            profile_confidence="low" if low_confidence else "normal",
        )
        tidy_profile(profile, website=website)
        _drop_inactive_rivals(profile, conn)
        return profile


# ---- search-query writer ----------------------------------------------------
_VAGUE_QUERY_WORDS = frozenset({
    "spending", "trading", "programme", "program", "sector", "market", "markets", "trends",
    "trend", "outlook", "policy", "industry", "growth"})


def _query_problem(q: str) -> str:
    """"" if an English query passes the deterministic check, else why it fails."""
    words = q.split()
    if len(words) > 4:
        return "too long"
    if {w.strip(".,;:!?\"'()").lower() for w in words} & _VAGUE_QUERY_WORDS:
        return "vague"
    return ""


def _check_queries(data, local_lang: str = "") -> tuple[dict, set]:
    """Deterministic clean-up of a write_search_queries result. Returns (clean dict,
    set of problems seen: "too long" / "vague")."""
    data = data if isinstance(data, dict) else {}
    problems: set = set()
    seen: set = set()
    out: dict = {}
    for key, cap in (("trigger_queries", 12), ("industry_queries", 7),
                     ("competitor_query_seeds", 5)):
        keep = []
        for q in _as_str_list(data.get(key)):
            why = _query_problem(q)
            if why:
                problems.add(why)
                continue
            if q.lower() in seen:
                continue
            seen.add(q.lower())
            keep.append(q)
        out[key] = keep[:cap]
    out["product_category"] = " ".join(str(data.get("product_category") or "").split()[:3])
    local, lseen = [], set()
    if local_lang:
        for q in _as_str_list(data.get("local_queries")):
            if len(q.split()) > 4:
                problems.add("too long")
            elif q.lower() not in lseen:
                lseen.add(q.lower())
                local.append(q)
    out["local_queries"] = local[:12]
    return out, problems


def write_search_queries(value_prop: str, industries: list, competitors: list,
                         market: str = "Global", local_lang: str = "") -> dict:
    """Write the news-search queries in their own small LLM call (the big profile call gave
    them too little attention). Returns {trigger_queries, industry_queries,
    competitor_query_seeds, product_category, local_queries}, already run through the
    deterministic check. If fewer than 6 buyer searches survive, asks once more with a
    line naming the problem and keeps whichever result has more valid queries."""
    system = (
        "You write news-search queries for a B2B sales researcher. Follow the rules exactly. "
        "Write every value in English"
        + (f", except local_queries, which are in {local_lang}" if local_lang else "") + ".")
    names = ", ".join(c for c in (competitors or []) if isinstance(c, str))[:300]
    user = (
        f"SELLER SELLS: {value_prop}\n"
        f"BUYER INDUSTRIES: {'; '.join(industries or [])}\n"
        f"COMPETITORS: {names or 'none'}\n"
        + (f"COUNTRY: {market}\n" if market and market != "Global" else "")
        + "\nReturn JSON with these keys.\n"
        "- trigger_queries: 10 to 12 searches. Each is a buyer from the buyer industries, in that "
        "trade's own noun, plus one event that buyer does (opens, builds, buys, wins, launches, raises).\n"
        "- industry_queries: 5 to 7 searches. Each is a concrete change in the buyers' trade, in that "
        "trade's own words.\n"
        "- competitor_query_seeds: 3 to 5 searches about moves in the seller's own product category, in "
        "generic words. Never a company name.\n"
        "- product_category: 1 to 3 words naming what the seller and its competitors sell.\n"
        "Every English search is 2 to 4 words and uses words with one meaning. Do not use these words: "
        "spending, trading, programme, sector, market, trends, outlook, policy, industry, growth. "
        "No abbreviations. No country name."
        + (f"\n- local_queries: 10 to 12 searches of 2 to 3 words in {local_lang}, in its native script. "
           "Mostly a buyer plus an event." if local_lang else ""))

    def _count(d):
        return sum(len(d[k]) for k in ("trigger_queries", "industry_queries",
                                       "competitor_query_seeds", "local_queries"))

    def _ask(u):
        raw = llm.chat_json(system, u, temperature=0,
                            max_tokens=1800 if local_lang else 1200, default={}) or {}
        return _check_queries(raw, local_lang)

    best, problems = _ask(user)
    if len(best["trigger_queries"]) < 6:
        notes = []
        if "too long" in problems:
            notes.append("some searches were too long")
        if "vague" in problems:
            notes.append("some searches used the banned words")
        if not notes:
            notes.append("there were too few buyer searches")
        second, _ = _ask(user + "\nLast time " + " and ".join(notes) + ". Fix that.")
        if _count(second) > _count(best):
            best = second
    return best


def _drop_inactive_rivals(profile, conn) -> None:
    """Invented rivals have no press presence in the lead's country: drop those
    rivals.rival_active says are absent. Skipped without a DB or for a Global/unknown
    market; a failed check keeps the rival (a network error must not wipe the list)."""
    if conn is None or not profile.competitors:
        return
    try:
        import config
        import regions
        import rivals
        # config's country dict, as geo.py passes it: rivals reads its "gn" locale
        country = config.get_country(profile.market)
        iso = (regions.get_profile(profile.market) or {}).get("iso2", "")
    except Exception:  # noqa: BLE001
        return
    if profile.market == "Global" or not iso:
        return
    keep = []
    for c in profile.competitors:
        try:
            ok = rivals.rival_active(profile, c, country, iso, conn=conn)
        except Exception:  # noqa: BLE001
            ok = True
        if ok:
            keep.append(c)
    _prune_rivals(profile, keep)


def _prune_rivals(profile, keep: list[str]) -> None:
    """Restrict competitors and the per-rival alias dicts to `keep`."""
    k = {c.lower() for c in keep}
    profile.competitors = list(keep)
    for f in ("competitor_aliases", "local_competitor_names"):
        d = getattr(profile, f, None) or {}
        setattr(profile, f, {a: b for a, b in d.items() if str(a).strip().lower() in k})


_GENERIC_RIVAL_WORDS = {
    "technology", "technologies", "solutions", "services", "systems", "international",
    "global", "research", "institute", "science", "sciences", "company", "group",
    "enhancement", "industries", "corporation", "partners", "consulting", "association",
}
_LEGAL_SUFFIXES = {"inc", "ltd", "llc", "gmbh", "sa", "ag", "co", "plc", "group", "holdings",
                   "corp", "limited", "the", "and", "of"}


def _generic_rival(name: str, profile) -> bool:
    """True when every word of a rival name is a category/country/generic word the profile
    itself uses ("Cloud Seeding Australia" for a cloud-seeding seller): an invented
    description, not a company."""
    words = [w for w in re.sub(r"[^\w\s]", " ", (name or "").lower()).split()
             if w not in _LEGAL_SUFFIXES]
    if not words:
        return False
    vocab = set(_GENERIC_RIVAL_WORDS)
    texts = [getattr(profile, "value_prop", "") or ""]
    for f in ("industries", "competitor_query_seeds", "trigger_queries", "industry_queries"):
        texts += list(getattr(profile, f, None) or [])
    for t in texts:
        vocab.update(re.sub(r"[^\w\s]", " ", str(t).lower()).split())
    try:
        import regions
        iso = (regions.get_profile(profile.market) or {}).get("iso2", "")
        for w in (_country_words(iso) if iso else []):
            vocab.update(w.split())
    except Exception:  # noqa: BLE001
        pass
    return all(w in vocab for w in words)


def _country_words(iso2: str) -> list[str]:
    """The lead country's name, aliases and nationality, longest first."""
    import regions
    p = regions.get_profile(iso2) or {}
    words = {p.get("name", "").lower(), p.get("nationality", "")} | set(p.get("aliases", []))
    return sorted((w for w in words if w and len(w) > 2), key=len, reverse=True)


def strip_country_words(query: str, iso2: str) -> str:
    """'German telecom merger' -> 'telecom merger' for a German lead: collect() adds the
    country to every search, so a model-written country word only lengthens the query."""
    words = _country_words(iso2)
    if not words or not query:
        return query
    rx = re.compile(r"(?<![\w.])(?:" + "|".join(re.escape(w) for w in words)
                    + r")(?:['’]s)?(?![\w])", re.I)
    return re.sub(r"\s+", " ", rx.sub(" ", query)).strip(" ,-")


_STAFFING_RE = re.compile(
    r"\b(?:hires?|hiring|appoints?|appointments?|executives?|ceo|cxo|leadership|board member)\b", re.I)


def tidy_profile(profile, *, website: str = "", country: dict | None = None) -> dict:
    """Deterministic clean-up of a profile, idempotent, applied to new and cached profiles.

      - competitor_score.clean_profile: the seller as its own competitor, placeholder
        names, foreign country words on rival names, local queries naming the seller
      - English queries lose the lead country's own name/nationality (collect() adds it)
      - competitor_query_seeds that repeat a trigger/industry query are dropped: they
        put buyer news into the competitor bucket

    Returns what changed, by rule ({} when nothing did)."""
    import competitor_score
    import regions

    name = (country or {}).get("iso2") or (country or {}).get("name") or profile.market or ""
    iso2 = "" if name in ("", "Global") else ((regions.get_profile(name) or {}).get("iso2", ""))
    changes = competitor_score.clean_profile(profile, website, iso2)

    if iso2:
        shortened = []
        for field_ in ("trigger_queries", "industry_queries", "competitor_query_seeds"):
            out = []
            for q in getattr(profile, field_, None) or []:
                s = strip_country_words(q, iso2)
                if s != q:
                    shortened.append([q, s])
                if s and s.lower() not in {x.lower() for x in out}:
                    out.append(s)
            setattr(profile, field_, out)
        if shortened:
            changes["country_words"] = shortened

    # Staffing-only trigger queries ("textile company hires new executives") pull HR news,
    # not demand for the seller's product; dropped while at least 4 trigger queries remain.
    trig = list(profile.trigger_queries or [])
    kept = [q for q in trig if not _STAFFING_RE.search(q)]
    if len(kept) < len(trig) and len(kept) >= 4:
        profile.trigger_queries = kept
        changes["staffing_triggers"] = [q for q in trig if q not in kept]

    taken = {q.lower() for q in (profile.trigger_queries or []) + (profile.industry_queries or [])}
    seeds = list(getattr(profile, "competitor_query_seeds", None) or [])
    dup = [q for q in seeds if q.lower() in taken]
    if dup:
        profile.competitor_query_seeds = [q for q in seeds if q.lower() not in taken]
        changes["seed_duplicates"] = dup
    generic = [c for c in (profile.competitors or []) if _generic_rival(c, profile)]
    if generic:
        _prune_rivals(profile, [c for c in profile.competitors if c not in generic])
        changes["generic_rivals"] = generic
    return changes


def _parse_competitor_aliases(raw, competitors: list[str],
                              local_names: dict | None = None) -> dict[str, list[str]]:
    """{competitor: [other written forms]} for the competitors actually kept, merged with
    local_competitor_names (the local form is just another alias). Entries for names not
    in `competitors` are ignored; forms equal to the name itself are dropped."""
    want = {c.lower(): c for c in competitors}
    out: dict[str, list[str]] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            name = want.get(str(k).strip().lower())
            if name:
                out[name] = _as_str_list(v)[:4]
    for k, v in (local_names or {}).items():
        name = want.get(str(k).strip().lower())
        if name and (v or "").strip():
            out.setdefault(name, []).append(str(v).strip())
    clean: dict[str, list[str]] = {}
    for name, forms in out.items():
        seen = {name.lower()}
        keep = []
        for f in forms:
            f = f.strip()
            if f and f.lower() not in seen:
                seen.add(f.lower())
                keep.append(f)
        if keep:
            clean[name] = keep
    return clean


def rival_forms(profile, name: str) -> list[str]:
    """Every written form of a rival: the stored name, its competitor_aliases and its
    local_competitor_names entry. Case-insensitive de-dupe, stored name first. Used for
    RIVAL matching only -- never for the seller's own-name check (brief_builder.is_self)."""
    forms = [name]
    key = (name or "").strip().lower()
    for field_ in ("competitor_aliases", "local_competitor_names"):
        d = getattr(profile, field_, None) or {}
        for k, v in d.items():
            if str(k).strip().lower() == key:
                forms += list(v) if isinstance(v, (list, tuple)) else [v]
    out, seen = [], set()
    for f in forms:
        f = (f or "").strip() if isinstance(f, str) else ""
        if f and f.lower() not in seen:
            seen.add(f.lower())
            out.append(f)
    return out


def _rival_base(s: str) -> str:
    s = re.sub(r"\s*[(\[][^)\]]*[)\]]", " ", s or "").lower()
    # Unicode letters, not just a-z: "엠디엠그룹" / "東京センチュリー" used to reduce to "" and
    # could never match a Korean or Japanese headline (2026-11-26 search audit).
    s = re.sub(r"[^\w ]|_", "", s)
    return re.sub(r"\s+", " ", s).strip()


def name_matches(name: str, text: str, *, strict_short: bool = False) -> bool:
    """True if the rival's name -- or its distinctive first two words -- appears in the
    text. strict_short=True (used for ALIASES, which may be acronyms) requires a word
    boundary for forms of 5 characters or fewer so "abc" does not match "abcdef"."""
    b, tx = _rival_base(name), _rival_base(text)
    if not b or not tx:
        return False
    if strict_short and len(b) <= 5:
        return bool(re.search(r"(?<!\w)" + re.escape(b) + r"(?!\w)", tx))
    if b in tx:
        return True
    head = " ".join(b.split()[:2])
    return len(head) > 5 and head in tx


def rival_named_in(profile, name: str, text: str) -> bool:
    """True if any written form of the rival `name` appears in `text`."""
    forms = rival_forms(profile, name)
    return any(name_matches(f, text, strict_short=(i > 0)) for i, f in enumerate(forms))


def _local_language_for(market: str) -> str:
    """Main local language of `market` in English ("Polish"), or "" when that language is
    English or unknown. Reads config.LOCAL_NEWS_EDITIONS / LOCAL_LANGUAGE_NAMES (getattr
    fallbacks so this runs before those tables exist)."""
    editions = getattr(config, "LOCAL_NEWS_EDITIONS", None) or {}
    if not editions.get(market):
        return ""
    return (getattr(config, "LOCAL_LANGUAGE_NAMES", None) or {}).get(market, "") or ""


LEAD_TYPES = {"seller", "investor", "nonprofit", "education", "government"}

_GENERIC_COMP_WORDS = {
    "services", "service", "solutions", "solution", "company", "companies", "firms", "firm",
    "manufacturers", "manufacturer", "suppliers", "supplier", "providers", "vendors",
    "usługi", "usługa", "servicios", "servicio", "services", "dienstleistungen", "services",
}
_GENERIC_FILLER = {
    "metal", "metals", "fabrication", "machining", "engineering", "industrial", "industry",
    "manufacturing", "testing", "laboratory", "laboratories", "equipment", "products",
    "global", "local", "leading", "other", "various", "and", "of", "for", "the", "in",
    "obróbki", "metalu", "obróbka", "producers", "players", "brands", "makers",
}


def _clean_competitors(names: list[str]) -> list[str]:
    """Drop generic phrases that are not company names ("metal fabrication services",
    "Usługi Obróbki Metalu"), then dedupe case-insensitively, order preserved."""
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        toks = re.findall(r"[^\W\d_][\w'’-]*", n, flags=re.UNICODE)
        low = [t.lower() for t in toks]
        if not toks:
            continue
        has_generic = any(t in _GENERIC_COMP_WORDS for t in low)
        if has_generic:
            if low[0] in _GENERIC_COMP_WORDS:
                continue  # leads with the generic noun: a category, not a company
            distinctive = [t for t in toks
                           if t[0].isupper() and t.lower() not in _GENERIC_COMP_WORDS
                           and t.lower() not in _GENERIC_FILLER]
            if not distinctive:
                continue
        elif len(toks) >= 2 and all(t.islower() for t in toks) and \
                any(t in _GENERIC_FILLER for t in low):
            continue
        k = n.lower()
        if k not in seen:
            seen.add(k)
            out.append(n)
    return out


def _product_grounded(name: str, site_text: str) -> bool:
    """True if the product name is supported by the homepage text, tolerating
    punctuation/hyphen/case differences ("X G2-Si" vs "x g2 si")."""
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (s or "").lower())).strip()
    n, hay = norm(name), norm(site_text)
    if not n:
        return False
    if n in hay:
        return True
    hay_tokens = set(hay.split())
    tokens = [t for t in n.split() if len(t) >= 3]
    return bool(tokens) and all(t in hay_tokens for t in tokens)


def _fix_company_case(company: str, site_text: str = "") -> str:
    """A single-case company name ("waters", "ACME TOOLS") takes the brand's casing from
    the homepage text when it appears there, otherwise is word-capitalised. Mixed-case
    names are left alone."""
    c = (company or "").strip()
    if not c or not (c.islower() or c.isupper()):
        return c
    if site_text:
        m = re.search(r"(?<![\w])" + re.escape(c) + r"(?![\w])", site_text, re.I)
        if m:
            hit = m.group(0)
            if not (hit.islower() or hit.isupper()):
                return hit
            if hit.isupper() and len(hit) <= 4:
                return hit  # a short acronym the site itself writes in capitals
    return re.sub(r"[^\W\d_]+", lambda x: x.group(0).capitalize(), c)


def product_terms_of(profile) -> list[str]:
    """Lowercased product names from the profile plus curated override product_terms,
    deduped, order preserved."""
    terms: list[str] = []
    for p in (getattr(profile, "products", None) or []):
        n = p.get("name") if isinstance(p, dict) else p
        if n and str(n).strip():
            terms.append(str(n).strip().lower())
    ov = _overrides_for(getattr(profile, "company", "") or "")
    terms += [t.strip().lower() for t in _as_str_list(ov.get("product_terms"))]
    return list(dict.fromkeys(t for t in terms if t))


class ProfileCache:
    """Per-run profile cache so one lead's signal and report briefs share ONE Profile.

    Keyed by (domain or company lower, country, recipient). Builds are serialised per
    key (not globally) so two workers on the same lead don't both call the LLM, while
    different leads still build in parallel. With a DB connection it also reads/writes
    the 30-day profile_cache table.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._items: dict[tuple, Profile] = {}
        self._key_locks: dict[tuple, threading.Lock] = {}

    def get(self, company: str, website: str = "", recipient: str = "there",
            country_name: str = "", conn=None, *, topic: str = "") -> Profile:
        """A lead with no company information is profiled from the market it enquired
        about (`topic`); with neither, Profile.from_company raises as before."""
        market = (country_name or "Global").strip() or "Global"
        by_market = not (company or "").strip() and bool((topic or "").strip())
        if by_market:
            company, website = topic.strip(), ""
        domain = ""
        if website:
            try:
                import enrich as _enrich
                domain = _enrich.domain_of(website=website)
            except Exception:  # noqa: BLE001
                domain = ""
        subject = domain or (company or "").strip().lower()
        if by_market:
            subject = "market:" + subject      # never collides with a company of that name
        key = (subject, market.lower(), (recipient or "there").strip().lower())
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                return hit
            klock = self._key_locks.setdefault(key, threading.Lock())
        with klock:
            with self._lock:
                hit = self._items.get(key)
            if hit is not None:
                return hit
            profile = None
            db_key = subject
            if conn is not None and db_key:
                try:
                    import db as _db
                    stored = _db.get_profile_cache(conn, db_key, market)
                    if stored:
                        profile = Profile.from_dict(stored)
                        profile.recipient = recipient or "there"
                        profile.__post_init__()
                        tidy_profile(profile, website=website)
                except Exception:  # noqa: BLE001 -- a cache problem just means rebuild
                    profile = None
            if profile is None:
                profile = Profile.from_company(company, website, recipient, country_name,
                                               conn=conn, subject_is_market=by_market,
                                               **({"topic_hint": topic}
                                                  if topic and not by_market else {}))
                if conn is not None and db_key:
                    try:
                        import db as _db
                        _db.put_profile_cache(conn, db_key, market,
                                              dataclasses.asdict(profile))
                    except Exception:  # noqa: BLE001
                        pass
            with self._lock:
                self._items[key] = profile
            return profile


def _as_str_list(v) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def _short_name(company: str) -> str:
    # "Kirey (Insirio)" -> "Kirey": parenthetical parts are never part of the short name.
    base = re.sub(r"\([^)]*\)?", " ", company or "")
    words = [w for w in base.replace(",", " ").split() if w]
    drop = {"inc", "inc.", "llc", "ltd", "ltd.", "corp", "corp.", "co", "co.", "plc", "the"}
    core = [w for w in words if w.lower().strip(".") not in drop]
    # Acronym only for 2+ capitalised words; a name is never cut mid-word (run-together
    # domain labels like "Bluematterconsulting" stay whole).
    if len(core) >= 2 and all(len(w) > 1 and w[0].isupper() for w in core[:3]):
        acr = "".join(w[0] for w in core[:3]).upper()
        if len(acr) >= 2:
            return acr
    return core[0] if core else (company or "").strip()
