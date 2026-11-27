"""CoherentConnect Newsletter — tenant profile, feeds, trigger taxonomy, scoring weights.

This is a self-contained distillation of the mother codebase's config: the country/
industry keyword bible + trigger-event filters + RSS/Google-News feeds, scoped here to
a single tenant (Coherent Market Insights) so the newsletter can run standalone.
"""
from __future__ import annotations

# ── The tenant (seller) profile ─────────────────────────────────────────────
TENANT = {
    "company": "Coherent Market Insights",
    "short": "CMI",
    "recipient_first_name": "Raj",
    "tagline": "Sales intelligence, made coherent.",
    # What CMI sells — used to frame the pitch / recommended action.
    "value_prop": (
        "analyst-verified syndicated and custom market-research reports: market sizing, "
        "competitive landscaping, patient/segment analysis, and go-to-market intelligence "
        "across healthcare, life sciences, chemicals, ICT and telecom."
    ),
    # CMI's coverage verticals. A prospect is relevant only if its news maps here.
    "industries": ["Healthcare", "Pharmaceuticals & Life Sciences", "Biotechnology",
                   "Chemicals & Materials", "ICT & Technology", "Telecom", "Medical Devices"],
    # Named competitors — used for the "What Competitors Are Up To" section.
    "competitors": [
        "Grand View Research", "MarketsandMarkets", "Precedence Research",
        "Allied Market Research", "Mordor Intelligence", "Fortune Business Insights",
        "Future Market Insights",
    ],
}

# ── Industry relevancy lexicon (keyword → the vertical it signals) ───────────
# Weighted keyword matching drives the "relevancy" score. Lowercased word-ish tokens.
INDUSTRY_KEYWORDS = {
    "Healthcare": ["healthcare", "hospital", "clinical", "patient", "therapy", "therapeutic",
                   "disease", "medical", "health", "care delivery", "payer", "provider"],
    "Pharmaceuticals & Life Sciences": ["pharma", "pharmaceutical", "drug", "molecule", "fda",
                   "phase 1", "phase 2", "phase 3", "trial", "indication", "oncology",
                   "vaccine", "biopharma", "life sciences", "clinical trial"],
    "Biotechnology": ["biotech", "biotechnology", "gene therapy", "mrna", "antibody", "cell therapy",
                   "genomics", "crispr", "fibrosis", "antifibrotic", "rare disease"],
    "Chemicals & Materials": ["chemical", "chemicals", "specialty chemical", "polymer", "coatings",
                   "materials", "petrochemical", "resin", "catalyst", "battery material"],
    "ICT & Technology": ["software", "ai", "artificial intelligence", "cloud", "semiconductor",
                   "chip", "data center", "saas", "enterprise software", "generative ai", "agentic"],
    "Telecom": ["telecom", "5g", "broadband", "spectrum", "fiber", "operator", "network", "wireless"],
    "Medical Devices": ["medical device", "diagnostic", "imaging", "implant", "wearable", "in-vitro"],
}

# ── Non-buyer names ─────────────────────────────────────────────────────────
# Bodies that make news in a market but have no purchase order to give: a regulator
# publishes rules, a ministry issues guidance, a court rules, a trade body lobbies, a
# journal reports. Such a name is a legitimate signal but never a Potential Customer
# (or Deal/Partnership, for investors and public bodies), so both the generator's
# candidate filter (brief_builder._looks_like_seller_category) and the rendered-output
# role check (validate_brief) share this ONE list -- two divergent copies previously
# meant a name caught at generation time still passed validation, and vice versa.
# Evidence-score floor: candidates whose ev_relevancy is below this are dropped from the
# Industry / Customer / Competition shortlists whenever anything above it remains.
MIN_EV_RELEVANCY = 0.15

NON_BUYER_TERMS = (
    "regulator", "regulatory authority", "ministry", "government", "city of",
    "municipal", "municipality", "city council",
    "cdsco", "dcgi", "fda", "court", "tribunal", "parliament",
    "association", "council", "committee", "commission", "board of",
    "department of", "task force", "research institute", "journal",
)

# Government bodies: never a rival, and a customer only for a lead whose profile lists a
# government buyer (state water agencies, municipalities, local authorities...). For
# everyone else they are filtered out of the customer and competitor pools before the
# model sees them: grant schemes and ministry news filled the triage slots (Grow Partners'
# top 10 were all government schemes). Industry Outlook keeps them -- new rules matter.
GOV_TERMS = (
    "government", "govt", "ministry", "minister", "city of", "city council", "municipal",
    "municipality", "department of", "parliament", "senate", "local authority", "council",
    "prefecture", "district of", "secretary", "defra", "european commission", "state agency",
    "federal agency", "public authority",
    "gouvernement", "ministère", "mairie", "ville de", "regierung", "ministerium", "gemeente",
    "ministerie", "gobierno", "ministerio", "ayuntamiento", "governo", "ministero", "comune di",
    "bakanlığı", "belediye",
)
_GOV_RE = None
_GOV_BUYER_RE = None


def is_government(text: str) -> bool:
    """True when `text` (a headline or a name) names a government body."""
    global _GOV_RE
    import re
    if _GOV_RE is None:
        _GOV_RE = re.compile(r"(?<![\w])(?:" + "|".join(re.escape(t) for t in GOV_TERMS) + r")(?![\w])",
                             re.I)
    return bool(_GOV_RE.search(text or ""))


def government_buyers_ok(profile) -> bool:
    """True when the lead's own buyer industries include a government body."""
    global _GOV_BUYER_RE
    import re
    if _GOV_BUYER_RE is None:
        _GOV_BUYER_RE = re.compile(
            r"government|municipal|local authorit|public (?:administration|sector|bod)|"
            r"(?:state|government|public|water|development) agenc|tourism board|ministr|"
            r"council|public utilit", re.I)
    return any(_GOV_BUYER_RE.search(str(i)) for i in (getattr(profile, "industries", None) or []))

# ── Trigger-event taxonomy (importance weight 0..1 + detection keywords) ─────
# A "trigger" is a live event that creates immediate demand for market research
# (funding needs market sizing; a trial needs competitive landscaping; etc.).
TRIGGER_TYPES = {
    "funding_round":       {"importance": 0.95, "keywords": ["raises", "raised", "funding", "series a",
                            "series b", "series c", "seed round", "debut round", "financing", "led by",
                            "million round", "closes $", "secures $", "backed by"]},
    "ipo":                 {"importance": 0.92, "keywords": ["ipo", "goes public", "public listing",
                            "files for ipo", "debut", "lists on", "stock market debut"]},
    "mna":                 {"importance": 0.88, "keywords": ["acquires", "acquisition", "merger",
                            "to acquire", "buys", "takeover", "merges with"]},
    "clinical_trial":      {"importance": 0.90, "keywords": ["phase 3", "phase 2", "pivotal trial",
                            "topline", "trial results", "enrolls", "primary endpoint", "readout"]},
    "regulatory_approval": {"importance": 0.85, "keywords": ["fda approval", "approved", "clearance",
                            "ema", "authorization", "green light", "designation"]},
    "product_launch":      {"importance": 0.70, "keywords": ["launches", "unveils", "rolls out",
                            "introduces", "debuts product", "new platform"]},
    "expansion":           {"importance": 0.72, "keywords": ["expands", "new plant", "new facility",
                            "opens factory", "capacity expansion", "enters market", "expansion into"]},
    "partnership":         {"importance": 0.60, "keywords": ["partners with", "partnership",
                            "collaboration", "joint venture", "teams up", "alliance"]},
    "exec_hire":           {"importance": 0.50, "keywords": ["appoints", "names new", "hires",
                            "joins as", "new ceo", "new cfo", "chief"]},
}

# Money-scale boosters — a big number amplifies importance.
BIG_MONEY_HINTS = ["billion", "$1b", "$2b", "$500m", "$400m", "$300m", "$250m", "$200m",
                   "$180m", "$150m", "$100m", "crore"]

# ── Feeds / queries ─────────────────────────────────────────────────────────
# Locale-parameterized so each target country pulls its own Google-News edition.
GOOGLE_NEWS = "https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"
GN_DEFAULT_LOCALE = ("en-US", "US", "US:en")
# Second free engine. Its RSS returns the article's opening lines as the summary (Google
# News RSS returns the headline again) and a publisher URL inside the `url=` parameter.
BING_NEWS = "https://www.bing.com/news/search?q={q}&format=rss&setlang=en-US"
HTTP_UA = "Mozilla/5.0 (compatible; CoherentConnect-Newsletter/1.0)"
# Fetch real article text for the top-ranked articles (Google News summaries are just the
# headline, so writers otherwise see a title only and invent the rest). 0 = off.
import os as _os
FETCH_ARTICLE_TEXT = _os.getenv("CMI_FETCH_ARTICLE_TEXT", "1") == "1"
# Industry/market news window. 96 hours left thin leads with almost nothing (Dabew:
# 3 eligible industry articles in 96h vs 41 in 3 months, 2026-11-23 window test,
# session_scratch/window_test.py). Widened to 6 months on 2026-11-23 (user decision) so
# every bucket uses the same window; freshness() still ranks recent stories first.
LOOKBACK_HOURS = 24 * 182  # signals older than this are dropped (6 months, same as every other bucket)

# Competitor signals get a much longer window than triggers. The two are not the same
# kind of fact: a buyer's funding round is only actionable while the money is still
# being spent, but a rival shipping a competing product stays true until they ship the
# next one. In a thin category the short window is the difference between real
# competitive intelligence and none -- searching CAR-T news for India inside 96 hours
# returns nothing at all (the freshest genuine cell-therapy story was ~28 days old),
# so the section fell back to whatever unrelated news the biggest rival happened to
# make that week. Six months of a rival's category moves beats four days of noise.
COMPETITOR_LOOKBACK_HOURS = 24 * 182

# Trigger signals get the same six-month window. 96 hours was tuned for a seller whose
# buyers make news weekly; it starves one whose buyers do not. Measured on VVF's buyer
# queries: 1 article within 96h against 18 within 30 days, which left the Potential
# Customer slot empty or stretched to a famous company in an unrelated trade.
#
# Widening the window does NOT mean stale briefs. freshness() still decays older items,
# so a signal from this week outranks one from last month whenever both exist -- the
# window only decides what is ELIGIBLE at all, and an eligible-but-older real buyer
# beats an irrelevant fresh one. A per-company `trigger_lookback_hours` in
# company_overrides.json overrides this.
TRIGGER_LOOKBACK_HOURS = 24 * 182

# Topic news (report briefs) uses the same six months. collect_queries() used LOOKBACK_HOURS
# (96h) for these, which left niche topics with nothing -- see collect.collect_topic().
TOPIC_LOOKBACK_HOURS = 24 * 182

# Append Google's "when:Nd" (N = the bucket's lookback in days) to every news search, so the
# result cap fills with in-window items instead of old ones that get dropped afterwards.
SEARCH_WHEN_OPERATOR = True

# After the first search pass, search up to this many buyer companies named in headlines
# ("<Company> to set up a plant") by name. 0 disables.
HARVEST_BUYERS = 6

# ── Target countries (compact port of the mother repo's config_countries/) ──
# Each: Google-News locale (hl, gl, ceid) + geo-match signals used to HARD-FILTER
# collected articles down to that country. "Global" = no locale bias, no filter.
COUNTRIES = {
    "Global": {"name": "Global", "gn": GN_DEFAULT_LOCALE, "aliases": [], "nationality": "",
               "cities": [], "currency": [], "tld": [], "outlets": []},
    "India": {"name": "India", "gn": ("en-IN", "IN", "IN:en"),
              "aliases": ["india", "indian", "bharat"], "nationality": "indian",
              "cities": ["mumbai", "delhi", "new delhi", "bengaluru", "bangalore", "chennai",
                         "hyderabad", "pune", "kolkata", "gurugram", "gurgaon", "noida", "ahmedabad"],
              "currency": ["₹", "rupee", "rupees", "crore", "lakh", "inr"], "tld": [".in"],
              "outlets": ["economic times", "livemint", "mint", "business standard", "moneycontrol",
                          "the hindu", "times of india", "ndtv", "financial express", "et now", "inc42",
                          "yourstory", "entrackr"]},
    "United States": {"name": "United States", "gn": ("en-US", "US", "US:en"),
              "aliases": ["united states", "u.s.", "us-based", "america", "american", " usa"],
              "nationality": "american",
              "cities": ["new york", "san francisco", "boston", "chicago", "los angeles", "seattle",
                         "austin", "silicon valley", "washington", "houston", "atlanta"],
              "currency": [], "tld": [], "outlets": ["cnbc", "wall street journal", "wsj", "bloomberg",
                          "techcrunch", "the new york times", "forbes", "reuters us", "axios"]},
    "United Kingdom": {"name": "United Kingdom", "gn": ("en-GB", "GB", "GB:en"),
              "aliases": ["uk", "u.k.", "britain", "british", "england", "united kingdom"],
              "nationality": "british",
              "cities": ["london", "manchester", "birmingham", "cambridge", "oxford", "edinburgh",
                         "glasgow", "leeds", "bristol"],
              "currency": ["£", "pound", "gbp", "pence"], "tld": [".co.uk", ".uk"],
              "outlets": ["bbc", "the guardian", "financial times", "sky news", "the telegraph",
                          "city a.m.", "uktech", "sifted"]},
    "Canada": {"name": "Canada", "gn": ("en-CA", "CA", "CA:en"),
              "aliases": ["canada", "canadian"], "nationality": "canadian",
              "cities": ["toronto", "vancouver", "montreal", "ottawa", "calgary", "waterloo", "edmonton"],
              "currency": ["cad", "c$"], "tld": [".ca"],
              "outlets": ["the globe and mail", "cbc", "financial post", "betakit", "the logic"]},
    "Australia": {"name": "Australia", "gn": ("en-AU", "AU", "AU:en"),
              "aliases": ["australia", "australian", "aussie"], "nationality": "australian",
              "cities": ["sydney", "melbourne", "brisbane", "perth", "adelaide", "canberra"],
              "currency": ["a$", "aud"], "tld": [".com.au", ".au"],
              "outlets": ["afr", "australian financial review", "abc news", "smart company", "startupdaily"]},
    "Singapore": {"name": "Singapore", "gn": ("en-SG", "SG", "SG:en"),
              "aliases": ["singapore", "singaporean", "sg"], "nationality": "singaporean",
              "cities": ["singapore"], "currency": ["s$", "sgd"], "tld": [".sg"],
              "outlets": ["straits times", "business times", "cna", "tech in asia", "e27"]},
    "United Arab Emirates": {"name": "United Arab Emirates", "gn": ("en-AE", "AE", "AE:en"),
              "aliases": ["uae", "u.a.e.", "emirati", "emirates"], "nationality": "emirati",
              "cities": ["dubai", "abu dhabi", "sharjah"], "currency": ["aed", "dirham"], "tld": [".ae"],
              "outlets": ["the national", "khaleej times", "gulf news", "arabian business", "wamda"]},
    "Germany": {"name": "Germany", "gn": ("en-DE", "DE", "DE:en"),
              "aliases": ["germany", "german", "deutschland"], "nationality": "german",
              "cities": ["berlin", "munich", "münchen", "frankfurt", "hamburg", "cologne", "stuttgart"],
              "currency": ["€", "euro"], "tld": [".de"],
              "outlets": ["handelsblatt", "der spiegel", "deutsche welle", "gründerszene", "tagesschau"]},
    "Japan": {"name": "Japan", "gn": ("en-JP", "JP", "JP:en"),
              "aliases": ["japan", "japanese"], "nationality": "japanese",
              "cities": ["tokyo", "osaka", "kyoto", "yokohama", "nagoya"], "currency": ["¥", "yen", "jpy"],
              "tld": [".jp"], "outlets": ["nikkei", "japan times", "kyodo", "nhk"]},
    "Brazil": {"name": "Brazil", "gn": ("en-419", "BR", "BR:en"),
              "aliases": ["brazil", "brazilian", "brasil"], "nationality": "brazilian",
              "cities": ["são paulo", "sao paulo", "rio de janeiro", "brasília", "belo horizonte"],
              "currency": ["r$", "real", "reais", "brl"], "tld": [".br", ".com.br"],
              "outlets": ["valor", "exame", "folha", "startse", "neofeed"]},
    "Nigeria": {"name": "Nigeria", "gn": ("en-NG", "NG", "NG:en"),
              "aliases": ["nigeria", "nigerian"], "nationality": "nigerian",
              "cities": ["lagos", "abuja", "ibadan", "port harcourt"], "currency": ["₦", "naira", "ngn"],
              "tld": [".ng", ".com.ng"], "outlets": ["techcabal", "nairametrics", "premium times",
                          "the guardian nigeria", "businessday"]},
}


# ── Market-research firms (CMI's own competitors) + PR wires ────────────────
# In the CMI lead-nurture newsletter we must NOT cite these as sources or reproduce their
# market estimates — CMI would never quote a rival research firm's numbers to its own client.
# Articles from these sources (and generic "market size / forecast" report-PR) are dropped;
# any market figures are presented as Coherent Market Insights' OWN intelligence.
RESEARCH_FIRM_SOURCES = [
    "grand view", "grandview", "fortune business insights", "marketsandmarkets", "markets and markets",
    "precedence research", "mordor intelligence", "allied market", "future market insights",
    "imarc", "market research future", "researchandmarkets", "research and markets", "vantage market",
    "polaris market", "straits research", "data bridge", "databridge", "the business research company",
    "expert market research", "custom market insights", "spherical insights", "fact.mr", "factmr",
    "persistence market", "transparency market", "verified market", "emergen research", "skyquest",
    "insightace", "maximize market", "cognitive market", "astute analytica", "market.us", "market us",
    "meticulous research", "next move strategy", "roots analysis", "delveinsight", "delve insight",
    "coherent market insights", "openpr", "globenewswire", "globe newswire", "prnewswire", "pr newswire",
    "einpresswire", "ein presswire", "digital journal", "benzinga", "yahoo finance market",
    "kenneth research", "zion market", "acumen research", "credence research", "prophecy market",
    "sns insider", "towards healthcare", "biospace market",
    "indexbox", "index box",      # "<X> Market To 2035: <driver> Drives Growth - News and Statistics"
]

# Report-PR headline patterns (a competitor's report title, not a real news event).
import re as _re
_REPORT_PR_RE = _re.compile(
    r"market\s+(size|share|report|forecast|analysis|outlook|trends?|research|revenue|growth\s+analysis)"
    r"|\bcagr\b|forecast\s+(to|till|until|period|by)\s*,?\s*20\d\d|to\s+reach\s+(usd|us\$|\$|inr)"
    r"|\|\s*(report|forecast|industry|20\d\d)|\bmn\b.*\b20\d\d\b|\(7mm\)|worth\s+(usd|us\$|\$)"
    # "<X> Market To 2035: <driver> Drives Growth" and "Market Projected/Expected to ..."
    r"|\bmarkets?\s+to\s+20\d\d\b|\bmarkets?\s+(is\s+)?(projected|expected|set|forecast(ed)?|poised)\s+to\b"
    r"|\bto\s+(hit|touch|surpass|exceed)\s+(usd|us\$|\$|inr)|\bmarkets?\b.{0,80}\bby\s+20[2-5]\d\b",
    _re.I,
)


def is_research_pr(source: str, title: str) -> bool:
    """True if an article is a rival research firm's output or generic market-report PR."""
    s = (source or "").lower()
    if any(firm in s for firm in RESEARCH_FIRM_SOURCES):
        return True
    return bool(_REPORT_PR_RE.search(title or ""))


def get_country(name: str) -> dict:
    """Resolve a country by name/alias/ISO2 -> its profile dict.

    Order: the 12 curated COUNTRIES entries (they carry hand-picked outlet lists)
    first, then the full regions.py table (~180 countries, the country-first
    evidence filter's source of truth), then Global. Unknown/blank -> Global, as
    before."""
    if not name:
        return COUNTRIES["Global"]
    n = name.strip().lower()
    for key, prof in COUNTRIES.items():
        if key.lower() == n or n in [a.strip().lower() for a in prof.get("aliases", [])] or prof["name"].lower() == n:
            return prof
    # Late import: regions.py is the wide table and must not import config back.
    import regions
    rp = regions.get_profile(name)
    if rp:
        # A curated country reached via an ISO2 code or an alias the curated
        # dict doesn't carry ("PL", "South Korea") should still return the
        # curated entry with its outlets.
        for prof in COUNTRIES.values():
            if prof["name"].lower() == rp["name"].lower():
                return prof
        return rp
    return COUNTRIES["Global"]

# Trigger discovery queries (company-level buying triggers in CMI's verticals).
TRIGGER_QUERIES = [
    "biotech funding round", "pharma Series B financing", "drug developer raises million",
    "phase 3 clinical trial start", "FDA approval drug", "gene therapy funding",
    "medical device FDA clearance", "specialty chemicals capacity expansion",
    "semiconductor plant investment", "5G network investment", "healthcare startup raises",
    "life sciences IPO", "biopharma acquisition", "oncology drug financing",
]
# Industry / structural queries ("Industry News" section).
INDUSTRY_QUERIES = [
    "AI agents market research", "market research industry trends 2026",
    "syndicated research market", "B2B data market report", "market intelligence AI disruption",
    "generative AI enterprise adoption research",
]

# ── Scoring matrix weights (relevancy x freshness x importance) ─────────────
SCORING = {
    "w_relevancy": 0.40,
    "w_freshness": 0.25,
    "w_importance": 0.35,
    # Freshness decay (hours → factor). Piecewise, mirrors mother repo's freshness_score.
    "freshness_bands": [(24, 1.00), (48, 0.85), (72, 0.65), (96, 0.45), (168, 0.25),
                        # tail added with the wider windows so a 2-week-old story
                        # still outranks a 2-month-old one instead of both hitting the floor
                        (24 * 14, 0.20), (24 * 30, 0.16), (24 * 60, 0.12),
                        # 6-month window: keep separating 2-4 month items from 4-6 month ones
                        (24 * 120, 0.11)],
    "freshness_floor": 0.10,
    # Minimum composite for a signal to be eligible for the newsletter at all.
    "min_composite": 0.18,
}

# ── LLM settings ────────────────────────────────────────────────────────────
# Provider order: DeepSeek first (as requested); auto-fallback to OpenAI if a call
# fails (e.g. DeepSeek 402 Insufficient Balance). Flip LLM_PROVIDER=deepseek/openai in .env.
LLM = {
    "provider": "deepseek",              # "deepseek" | "openai" | "local"  (env LLM_PROVIDER overrides)
    "deepseek_model": "deepseek-chat",
    "deepseek_base_url": "https://api.deepseek.com",
    "openai_model": "gpt-4o-mini",
    "fallback_to_openai": False,
    "temperature": 0.4,
    # The self-hosted fine-tune (PLAN.md Phase 4). Served behind an OpenAI-compatible
    # endpoint, so it needs no client changes -- only a base_url. Set LOCAL_LLM_BASE_URL
    # to the running server; LLM_PROVIDER=local then routes every call to it.
    "local_model": "cmi-7b",
    "local_base_url": "http://localhost:8000/v1",
    # Per-request timeout (seconds) on every OpenAI SDK client (local/DeepSeek/OpenAI).
    # Without one, a hung request on an overloaded vLLM server just blocks a worker
    # thread indefinitely (the SDK's own default is several minutes) -- under a
    # 40-worker batch that silently drains effective concurrency with no error or log
    # line, only the run taking longer than expected for no visible reason. A brief's
    # own calls run sequentially and each asks for at most ~1600 tokens, so a genuine
    # response is seconds, not minutes -- but under real batch load (40 workers sharing
    # one GPU's request queue), individual calls occasionally sit queued long enough to
    # blow past a tight timeout even though the server is healthy: measured 6/171 calls
    # (3.5%) timing out at 90s during a real 40-worker run with no other sign of trouble
    # (never enough to trip the consecutive-failure abort). 180s gives a queued call
    # room to clear under load while still catching a truly wedged request.
    # Override with CMI_LLM_TIMEOUT_SECS if a slower endpoint needs more.
    "timeout_secs": 180,
}


# ── Local-language news (2026-11-23) ────────────────────────────────────────────
# Every lead gets the English search PLUS a second pass on its country's main
# local-language Google News edition (user decision: English + local for all regions,
# no per-country exceptions). None = the local press language is English, so no
# second pass. Editions verified to return results on 2026-11-23
# (session_scratch/regional_sources3/probe_editions.py); Algeria and Liechtenstein
# had no edition of their own and use the closest one in the same language (FR, CH).
LOCAL_NEWS_EDITIONS = {
    'Afghanistan': ('fa', 'AF', 'AF:fa'),
    'Albania': ('sq', 'AL', 'AL:sq'),
    'Algeria': ('fr', 'FR', 'FR:fr'),
    'American Samoa': None,
    'Argentina': ('es-419', 'AR', 'AR:es-419'),
    'Australia': None,
    'Austria': ('de', 'AT', 'AT:de'),
    'Azerbaijan': ('az', 'AZ', 'AZ:az'),
    'Bahamas': None,
    'Bahrain': ('ar', 'BH', 'BH:ar'),
    'Bangladesh': ('bn', 'BD', 'BD:bn'),
    'Belgium': ('nl', 'BE', 'BE:nl'),
    'Brazil': ('pt-BR', 'BR', 'BR:pt-419'),
    'Bulgaria': ('bg', 'BG', 'BG:bg'),
    'Canada': ('fr', 'CA', 'CA:fr'),
    'Chile': ('es-419', 'CL', 'CL:es-419'),
    'China': ('zh-CN', 'CN', 'CN:zh-Hans'),
    'Colombia': ('es-419', 'CO', 'CO:es-419'),
    'Costa Rica': ('es-419', 'CR', 'CR:es-419'),
    'Croatia': ('hr', 'HR', 'HR:hr'),
    'Cyprus': ('el', 'CY', 'CY:el'),
    'Czechia': ('cs', 'CZ', 'CZ:cs'),
    'Denmark': ('da', 'DK', 'DK:da'),
    'Ecuador': ('es-419', 'EC', 'EC:es-419'),
    'Egypt': ('ar', 'EG', 'EG:ar'),
    'El Salvador': ('es-419', 'SV', 'SV:es-419'),
    'Estonia': ('et', 'EE', 'EE:et'),
    'Eswatini': None,
    'Finland': ('fi', 'FI', 'FI:fi'),
    'France': ('fr', 'FR', 'FR:fr'),
    'Georgia': ('ka', 'GE', 'GE:ka'),
    'Germany': ('de', 'DE', 'DE:de'),
    'Ghana': None,
    'Gibraltar': None,
    'Greece': ('el', 'GR', 'GR:el'),
    'Guernsey': None,
    'Guinea': ('fr', 'GN', 'GN:fr'),
    'Hong Kong': ('zh-HK', 'HK', 'HK:zh-Hant'),
    'Hungary': ('hu', 'HU', 'HU:hu'),
    'India': ('hi', 'IN', 'IN:hi'),
    'Indonesia': ('id', 'ID', 'ID:id'),
    'Iran': ('fa', 'IR', 'IR:fa'),
    'Iraq': ('ar', 'IQ', 'IQ:ar'),
    'Ireland': None,
    'Israel': ('he', 'IL', 'IL:he'),
    'Italy': ('it', 'IT', 'IT:it'),
    'Jamaica': None,
    'Japan': ('ja', 'JP', 'JP:ja'),
    'Jordan': ('ar', 'JO', 'JO:ar'),
    'Kazakhstan': ('ru', 'KZ', 'KZ:ru'),
    'Kenya': None,
    'Kuwait': ('ar', 'KW', 'KW:ar'),
    'Latvia': ('lv', 'LV', 'LV:lv'),
    'Lebanon': ('ar', 'LB', 'LB:ar'),
    'Liechtenstein': ('de', 'CH', 'CH:de'),
    'Lithuania': ('lt', 'LT', 'LT:lt'),
    'Luxembourg': ('fr', 'LU', 'LU:fr'),
    'Madagascar': ('fr', 'MG', 'MG:fr'),
    'Malawi': None,
    'Malaysia': ('ms', 'MY', 'MY:ms'),
    'Maldives': ('dv', 'MV', 'MV:dv'),
    'Malta': None,
    'Mauritius': ('fr', 'MU', 'MU:fr'),
    'Mexico': ('es-419', 'MX', 'MX:es-419'),
    'Moldova': ('ro', 'MD', 'MD:ro'),
    'Morocco': ('fr', 'MA', 'MA:fr'),
    'Namibia': None,
    'Nepal': ('ne', 'NP', 'NP:ne'),
    'Netherlands': ('nl', 'NL', 'NL:nl'),
    'New Zealand': None,
    'Nicaragua': ('es-419', 'NI', 'NI:es-419'),
    'Nigeria': None,
    'North Macedonia': ('mk', 'MK', 'MK:mk'),
    'Norway': ('no', 'NO', 'NO:no'),
    'Oman': ('ar', 'OM', 'OM:ar'),
    'Panama': ('es-419', 'PA', 'PA:es-419'),
    'Peru': ('es-419', 'PE', 'PE:es-419'),
    'Philippines': None,
    'Poland': ('pl', 'PL', 'PL:pl'),
    'Portugal': ('pt-PT', 'PT', 'PT:pt-150'),
    'Puerto Rico': ('es-419', 'PR', 'PR:es-419'),
    'Qatar': ('ar', 'QA', 'QA:ar'),
    'Romania': ('ro', 'RO', 'RO:ro'),
    'Russia': ('ru', 'RU', 'RU:ru'),
    'Rwanda': None,
    'Saudi Arabia': ('ar', 'SA', 'SA:ar'),
    'Serbia': ('sr', 'RS', 'RS:sr'),
    'Singapore': None,
    'Slovakia': ('sk', 'SK', 'SK:sk'),
    'Slovenia': ('sl', 'SI', 'SI:sl'),
    'South Africa': None,
    'South Korea': ('ko', 'KR', 'KR:ko'),
    'Spain': ('es', 'ES', 'ES:es'),
    'Sweden': ('sv', 'SE', 'SE:sv'),
    'Switzerland': ('de', 'CH', 'CH:de'),
    'Taiwan': ('zh-TW', 'TW', 'TW:zh-Hant'),
    'Tanzania': ('sw', 'TZ', 'TZ:sw'),
    'Thailand': ('th', 'TH', 'TH:th'),
    'Trinidad and Tobago': None,
    'Turkey': ('tr', 'TR', 'TR:tr'),
    'Uganda': None,
    'Ukraine': ('uk', 'UA', 'UA:uk'),
    'United Arab Emirates': ('ar', 'AE', 'AE:ar'),
    'United Kingdom': None,
    'United States': None,
    'Uruguay': ('es-419', 'UY', 'UY:es-419'),
    'Uzbekistan': ('uz', 'UZ', 'UZ:uz'),
    'Venezuela': ('es-419', 'VE', 'VE:es-419'),
    'Vietnam': ('vi', 'VN', 'VN:vi'),
    'Yemen': ('ar', 'YE', 'YE:ar'),
    'Zambia': None,
    'Zimbabwe': None,
}

LOCAL_LANGUAGE_NAMES = {
    'Afghanistan': 'Persian',
    'Albania': 'Albanian',
    'Algeria': 'French',
    'American Samoa': 'English',
    'Argentina': 'Spanish',
    'Australia': 'English',
    'Austria': 'German',
    'Azerbaijan': 'Azerbaijani',
    'Bahamas': 'English',
    'Bahrain': 'Arabic',
    'Bangladesh': 'Bengali',
    'Belgium': 'Dutch',
    'Brazil': 'Portuguese',
    'Bulgaria': 'Bulgarian',
    'Canada': 'French',
    'Chile': 'Spanish',
    'China': 'Chinese',
    'Colombia': 'Spanish',
    'Costa Rica': 'Spanish',
    'Croatia': 'Croatian',
    'Cyprus': 'Greek',
    'Czechia': 'Czech',
    'Denmark': 'Danish',
    'Ecuador': 'Spanish',
    'Egypt': 'Arabic',
    'El Salvador': 'Spanish',
    'Estonia': 'Estonian',
    'Eswatini': 'English',
    'Finland': 'Finnish',
    'France': 'French',
    'Georgia': 'Georgian',
    'Germany': 'German',
    'Ghana': 'English',
    'Gibraltar': 'English',
    'Greece': 'Greek',
    'Guernsey': 'English',
    'Guinea': 'French',
    'Hong Kong': 'Chinese',
    'Hungary': 'Hungarian',
    'India': 'Hindi',
    'Indonesia': 'Indonesian',
    'Iran': 'Persian',
    'Iraq': 'Arabic',
    'Ireland': 'English',
    'Israel': 'Hebrew',
    'Italy': 'Italian',
    'Jamaica': 'English',
    'Japan': 'Japanese',
    'Jordan': 'Arabic',
    'Kazakhstan': 'Russian',
    'Kenya': 'English',
    'Kuwait': 'Arabic',
    'Latvia': 'Latvian',
    'Lebanon': 'Arabic',
    'Liechtenstein': 'German',
    'Lithuania': 'Lithuanian',
    'Luxembourg': 'French',
    'Madagascar': 'French',
    'Malawi': 'English',
    'Malaysia': 'Malay',
    'Maldives': 'Dhivehi',
    'Malta': 'English',
    'Mauritius': 'French',
    'Mexico': 'Spanish',
    'Moldova': 'Romanian',
    'Morocco': 'French',
    'Namibia': 'English',
    'Nepal': 'Nepali',
    'Netherlands': 'Dutch',
    'New Zealand': 'English',
    'Nicaragua': 'Spanish',
    'Nigeria': 'English',
    'North Macedonia': 'Macedonian',
    'Norway': 'Norwegian',
    'Oman': 'Arabic',
    'Panama': 'Spanish',
    'Peru': 'Spanish',
    'Philippines': 'English',
    'Poland': 'Polish',
    'Portugal': 'Portuguese',
    'Puerto Rico': 'Spanish',
    'Qatar': 'Arabic',
    'Romania': 'Romanian',
    'Russia': 'Russian',
    'Rwanda': 'English',
    'Saudi Arabia': 'Arabic',
    'Serbia': 'Serbian',
    'Singapore': 'English',
    'Slovakia': 'Slovak',
    'Slovenia': 'Slovenian',
    'South Africa': 'English',
    'South Korea': 'Korean',
    'Spain': 'Spanish',
    'Sweden': 'Swedish',
    'Switzerland': 'German',
    'Taiwan': 'Chinese',
    'Tanzania': 'Swahili',
    'Thailand': 'Thai',
    'Trinidad and Tobago': 'English',
    'Turkey': 'Turkish',
    'Uganda': 'English',
    'Ukraine': 'Ukrainian',
    'United Arab Emirates': 'Arabic',
    'United Kingdom': 'English',
    'United States': 'English',
    'Uruguay': 'Spanish',
    'Uzbekistan': 'Uzbek',
    'Venezuela': 'Spanish',
    'Vietnam': 'Vietnamese',
    'Yemen': 'Arabic',
    'Zambia': 'English',
    'Zimbabwe': 'English',
}

# Sources dropped from the evidence pool: research-firm SEO "market report" pages,
# press-release farms and stock-ticker aggregators (seen flooding local-language and
# topic results in the 2026-11 regional experiments). Lowercase substrings matched
# against the article's source name / URL. Trade-off: genuine company press releases
# also travel over these wires and are lost with them.
SOURCE_BLOCKLIST = sorted(set([s.lower() for s in RESEARCH_FIRM_SOURCES] + [
    "indexbox", "spherical insights", "straits research", "fortune business insights",
    "newscast.jp", "atpress", "@press", "dreamnews", "wiseguy", "marketbeat", "stocktitan",
    "simply wall st", "simplywall.st", "zacks", "证券之星", "东方财富", "stock titan",
    "der aktionär", "der aktionaer", "boerse express", "börse express",
    # 2026-11-26 search audit: research-firm sites that leaked into the pool, and Facebook
    # posts (Hindi edition) that are not a publisher
    "market data forecast", "marketdataforecast", "global market insights", "gminsights",
    "facebook",
]))
