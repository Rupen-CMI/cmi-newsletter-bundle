"""Per-country region data — the country-first evidence filter's lookup table.

`config.COUNTRIES` carries only 12 curated countries; every other lead resolved to
the "Global" profile and got unfiltered US-English news (an Indian lead could get
New Orleans news). This module covers EVERY country that appears in the leads table
plus the rest of the 8 regions used there (North America, Latin America, Europe,
Asia Pacific, Oceania, Middle East, Africa, Global), ~150 entries.

Region assignments deliberately match the values already populated in
cmi_newsletter.db's leads.region column (Turkey/Cyprus/Egypt -> Middle East,
Georgia -> Asia Pacific, Russia -> Europe, Central America & Caribbean ->
Latin America, ...) so region fallbacks in geo.filter_for_lead never pull in a
country the DB placed in a different region.

Detection design (regions.article_countries / countries_in_text):
  - word-boundary, case-insensitive matching of name/alias/nationality/city/
    currency tokens; multi-word tokens are matched and REMOVED longest-first, so
    "Papua New Guinea" (PG) can never also count for "Guinea" (GN), "North Korea"
    (KP) for "Korea" (KR), or "South Sudan" (SS) for "Sudan" (SD).
  - a small mask table kills known false-positive phrases that are NOT country
    names ("New Mexico", "Indian Ocean", "New South Wales", "North America",
    "New Jersey", "New York", "New England", "British Columbia",
    "Northern Ireland", "New Guinea", "Guinea Bissau").
  - deliberately NOT detection tokens: the bare words "Georgia" (US state) and
    "Jordan" (person name) — those countries are only detected via
    "Georgian"/"Tbilisi" and "Jordanian"/"Amman"/"Jordan's".
  - shared currency tokens ("€", "euro", "cfa franc", ...) map to ALL their
    countries, so one mention credits every user of that currency.
  - ambiguous city names are dropped, not kept: Cambridge, Birmingham, Victoria,
    Hamilton, Perth, Paris, Kingston, San Jose, Nice, Male, Newcastle, Cordoba,
    Douglas, Split, Syracuse.
  - English words that collide with currency names are NOT used as bare tokens:
    "real" (BR), "rand" (ZA), "rial" (IR), "try" (TRY), "won" (KRW), "dong"
    (VND), "kip" (LAK), "mad" (MAD) — only the prefixed/full forms match.

get_profile() returns the SAME shape as config.COUNTRIES entries (name, gn,
aliases, nationality, cities, currency, tld, outlets) plus "iso2" and "region";
config.get_country() uses this as its fallback before degrading to Global.
"""
from __future__ import annotations

import re

REGIONS = ("North America", "Latin America", "Europe", "Asia Pacific",
           "Oceania", "Middle East", "Africa", "Global")

# ISO2 codes where Google News publishes a dedicated ENGLISH edition, so we can use
# hl=en-XX / gl=XX / ceid=XX:en. Everything else falls back to the closest English
# edition for its region (FALLBACK_HL below).
_EN_EDITIONS = {
    "US", "GB", "CA", "AU", "NZ", "IE", "IN", "SG", "MY", "PH", "HK", "PK", "JP",
    "AE", "NG", "KE", "GH", "ZA", "TZ", "UG", "ZW",
}

_FALLBACK_HL = {
    "Europe": "en-GB", "Middle East": "en-AE", "Africa": "en-ZA",
    "Latin America": "en-419", "Asia Pacific": "en-SG", "Oceania": "en-AU",
    "North America": "en-US", "Global": "en-US",
}

_EUR = ["€", "euro", "eur"]   # shared across the eurozone entries

def _gn(iso2: str, region: str) -> tuple[str, str, str]:
    """(hl, gl, ceid) — English locale where Google News supports it, else the
    closest regional English edition, same pattern as config.COUNTRIES entries."""
    gl = iso2 if iso2 else "US"
    if iso2 in _EN_EDITIONS:
        hl = f"en-{iso2}"
    else:
        hl = _FALLBACK_HL.get(region, "en-US")
    return (hl, gl, f"{gl}:en")

# ── the table ────────────────────────────────────────────────────────────────
# (iso2, name, region, extra aliases, nationality, cities, currency tokens, tlds)
# name/aliases/nationality/cities/currency are all detection tokens.
# Deliberate omissions: bare "georgia", bare "jordan" (see module docstring).
_ROWS = [
    # ── North America ──
    ("US", "United States", "North America",
     ["u.s.", "u.s", "usa", "u.s.a.", "us-based", "america", "americans",
      "united states of america"],
     "american",
     ["new york", "san francisco", "boston", "chicago", "los angeles", "seattle",
      "austin", "silicon valley", "washington", "houston", "atlanta", "dallas",
      "denver", "nashville", "miami", "san diego", "philadelphia", "minneapolis"],
     [], []),
    ("CA", "Canada", "North America", ["canadians"], "canadian",
     ["toronto", "vancouver", "montreal", "ottawa", "calgary", "waterloo",
      "edmonton", "halifax"], ["c$", "cad", "canadian dollar"], [".ca"]),
    ("BM", "Bermuda", "North America", [], "bermudian", [], ["bmd"], [".bm"]),
    # ── Latin America (incl. Central America & Caribbean — matches leads.region) ──
    ("BR", "Brazil", "Latin America", ["brasil", "brazilians"], "brazilian",
     ["sao paulo", "rio de janeiro", "brasilia", "belo horizonte"],
     ["r$", "reais", "brl", "brazilian real"], [".br", ".com.br"]),
    ("MX", "Mexico", "Latin America", ["mexicans"], "mexican",
     ["mexico city", "guadalajara", "monterrey"],
     ["mxn", "mexican peso"], [".mx", ".com.mx"]),
    ("AR", "Argentina", "Latin America", ["argentinians"], "argentine",
     ["buenos aires"], ["argentine peso"], [".ar", ".com.ar"]),
    ("CL", "Chile", "Latin America", ["chileans"], "chilean",
     ["santiago", "valparaiso"], ["chilean peso"], [".cl"]),
    ("CO", "Colombia", "Latin America", ["colombians"], "colombian",
     ["bogota", "medellin"], ["colombian peso"], [".co", ".com.co"]),
    ("PE", "Peru", "Latin America", ["peruvians"], "peruvian",
     ["lima"], ["peruvian sol"], [".pe", ".com.pe"]),
    ("UY", "Uruguay", "Latin America", ["uruguayans"], "uruguayan",
     ["montevideo"], ["uruguayan peso"], [".uy", ".com.uy"]),
    ("VE", "Venezuela", "Latin America", ["venezuelans"], "venezuelan",
     ["caracas"], ["venezuelan bolivar"], [".ve"]),
    ("EC", "Ecuador", "Latin America", ["ecuadorians"], "ecuadorian",
     ["quito", "guayaquil"], [], [".ec"]),
    ("BO", "Bolivia", "Latin America", ["bolivians"], "bolivian",
     ["sucre"], ["boliviano"], [".bo"]),
    ("PY", "Paraguay", "Latin America", ["paraguayans"], "paraguayan",
     ["asuncion"], ["guarani"], [".py"]),
    ("CR", "Costa Rica", "Latin America", ["costa ricans"], "costa rican",
     [], ["costa rican colon"], [".cr", ".co.cr"]),
    ("PA", "Panama", "Latin America", ["panama canal", "panamanians"], "panamanian",
     [], ["balboa"], [".pa"]),
    ("GT", "Guatemala", "Latin America", ["guatemalans"], "guatemalan",
     ["guatemala city"], ["quetzal"], [".gt"]),
    ("HN", "Honduras", "Latin America", ["hondurans"], "honduran",
     ["tegucigalpa"], ["lempira"], [".hn"]),
    ("SV", "El Salvador", "Latin America", ["salvadorans"], "salvadoran",
     ["san salvador"], [], [".sv", ".com.sv"]),
    ("NI", "Nicaragua", "Latin America", ["nicaraguans"], "nicaraguan",
     ["managua"], ["nicaraguan cordoba"], [".ni", ".com.ni"]),
    ("CU", "Cuba", "Latin America", ["cubans"], "cuban",
     ["havana"], ["cuban peso"], [".cu"]),
    ("DO", "Dominican Republic", "Latin America", ["dominicans"], "dominican",
     ["santo domingo"], ["dominican peso"], [".do"]),
    ("HT", "Haiti", "Latin America", ["haitians"], "haitian",
     ["port-au-prince"], ["gourde"], [".ht"]),
    ("JM", "Jamaica", "Latin America", ["jamaicans"], "jamaican",
     [], ["jamaican dollar", "jmd"], [".jm"]),
    ("TT", "Trinidad and Tobago", "Latin America", ["trinidadians", "tobagonians",
     "trinidad", "tobago", "port of spain"], "trinidadian", [], [], [".tt"]),
    ("PR", "Puerto Rico", "Latin America", ["puerto ricans"], "puerto rican",
     [], [], []),
    ("BS", "Bahamas", "Latin America", ["bahamians", "the bahamas"], "bahamian",
     ["nassau"], ["bahamian dollar", "bsd"], [".bs"]),
    # ── Europe ──
    ("GB", "United Kingdom", "Europe",
     ["uk", "u.k.", "britain", "great britain", "british", "britons", "england",
      "scotland", "scots", "wales"], "british",
     ["london", "manchester", "edinburgh", "glasgow", "leeds", "bristol",
      "liverpool", "cardiff", "belfast", "nottingham"],
     ["£", "pound", "pounds", "gbp", "pence"], [".co.uk", ".uk"]),
    ("IE", "Ireland", "Europe", ["irish", "eire"], "irish",
     ["dublin", "cork"], _EUR, [".ie"]),
    ("DE", "Germany", "Europe", ["germans", "deutschland"], "german",
     ["berlin", "munich", "frankfurt", "hamburg", "stuttgart", "cologne"],
     _EUR, [".de"]),
    ("FR", "France", "Europe", ["france's", "frenchmen"], "french",
     ["lyon", "marseille", "toulouse", "bordeaux", "strasbourg", "lille"],
     _EUR, [".fr"]),
    ("NL", "Netherlands", "Europe", ["dutch", "holland", "the hague",
     "nederland"], "dutch",
     ["amsterdam", "rotterdam", "eindhoven"], _EUR, [".nl"]),
    ("BE", "Belgium", "Europe", ["belgians", "flemish", "wallonia"], "belgian",
     ["brussels", "antwerp", "bruges"], _EUR, [".be"]),
    ("CH", "Switzerland", "Europe", ["swiss"], "swiss",
     ["zurich", "geneva", "bern", "basel"], ["chf", "swiss francs"], [".ch"]),
    ("AT", "Austria", "Europe", ["austrians", "viennese"], "austrian",
     ["vienna", "salzburg"], _EUR, [".at"]),
    ("ES", "Spain", "Europe", ["spaniards", "spanish", "espana"], "spanish",
     ["madrid", "barcelona", "valencia", "seville"], _EUR, [".es", ".com.es"]),
    ("IT", "Italy", "Europe", ["italians"], "italian",
     ["rome", "milan", "naples", "turin", "florence"], _EUR, [".it"]),
    ("PT", "Portugal", "Europe", ["portuguese", "lisboa"], "portuguese",
     ["lisbon", "porto"], _EUR, [".pt"]),
    ("GR", "Greece", "Europe", ["greeks", "greek", "hellas"], "greek",
     ["athens", "thessaloniki"], _EUR, [".gr"]),
    ("SE", "Sweden", "Europe", ["swedes", "swedish"], "swedish",
     ["stockholm", "gothenburg", "malmo"], ["krona", "sek"], [".se"]),
    ("NO", "Norway", "Europe", ["norwegians", "norwegian"], "norwegian",
     ["oslo", "bergen"], ["norwegian krone", "nok"], [".no"]),
    ("DK", "Denmark", "Europe", ["danes", "danish"], "danish",
     ["copenhagen", "aarhus"], ["danish krone", "dkk"], [".dk"]),
    ("FI", "Finland", "Europe", ["finns", "finnish"], "finnish",
     ["helsinki", "espoo"], _EUR, [".fi"]),
    ("IS", "Iceland", "Europe", ["icelanders", "icelandic"], "icelandic",
     ["reykjavik"], ["icelandic krona"], [".is"]),
    ("PL", "Poland", "Europe", ["polska", "warszawa"], "",
     ["warsaw", "krakow", "gdansk", "wroclaw", "poznan"],
     ["zloty", "pln", "zlotys"], [".pl", ".com.pl"]),
    ("CZ", "Czech Republic", "Europe", ["czechia", "czechs", "praha"], "czech",
     ["prague"], ["koruna", "czk"], [".cz"]),
    ("SK", "Slovakia", "Europe", ["slovaks", "slovakian"], "slovak",
     ["bratislava"], _EUR, [".sk"]),
    ("HU", "Hungary", "Europe", ["hungarians"], "hungarian",
     ["budapest"], ["forint", "huf"], [".hu"]),
    ("RO", "Romania", "Europe", ["romanians"], "romanian",
     ["bucharest", "cluj"], ["leu", "ron"], [".ro"]),
    ("BG", "Bulgaria", "Europe", ["bulgarians"], "bulgarian",
     ["sofia"], ["lev", "bgn"], [".bg"]),
    ("UA", "Ukraine", "Europe", ["ukrainians"], "ukrainian",
     ["kyiv", "kiev", "odessa", "kharkiv"], ["hryvnia", "uah"], [".ua", ".com.ua"]),
    ("RU", "Russia", "Europe", ["russians", "kremlin"], "russian",
     ["moscow", "saint petersburg", "st petersburg", "novosibirsk"],
     ["ruble", "rouble", "rub"], [".ru", ".com.ru"]),
    ("BY", "Belarus", "Europe", ["belarusians"], "belarusian",
     ["minsk"], ["belarusian ruble"], [".by"]),
    ("MD", "Moldova", "Europe", ["moldovans"], "moldovan",
     ["chisinau"], ["moldovan leu", "mdl"], [".md"]),
    ("HR", "Croatia", "Europe", ["croats", "croatians"], "croatian",
     ["zagreb"], _EUR, [".hr"]),
    ("SI", "Slovenia", "Europe", ["slovenians", "slovene"], "slovenian",
     ["ljubljana"], _EUR, [".si"]),
    ("RS", "Serbia", "Europe", ["serbs", "serbians"], "serbian",
     ["belgrade"], ["serbian dinar"], [".rs"]),
    ("BA", "Bosnia and Herzegovina", "Europe",
     ["bosnia", "herzegovina", "bosnians"], "bosnian",
     ["sarajevo"], ["convertible mark"], [".ba"]),
    ("MK", "North Macedonia", "Europe", ["macedonians", "macedonian", "skopje"],
     "", [], ["denar"], [".mk"]),
    ("ME", "Montenegro", "Europe", ["montenegrins", "montenegrin"], "",
     ["podgorica"], _EUR, [".me"]),
    ("AL", "Albania", "Europe", ["albanians", "albanian"], "",
     ["tirana"], ["lek"], [".al"]),
    ("MT", "Malta", "Europe", ["maltese"], "",
     ["valletta"], _EUR, [".mt"]),
    ("LT", "Lithuania", "Europe", ["lithuanians"], "lithuanian",
     ["vilnius"], _EUR, [".lt"]),
    ("LV", "Latvia", "Europe", ["latvians"], "latvian",
     ["riga"], _EUR, [".lv"]),
    ("EE", "Estonia", "Europe", ["estonians"], "estonian",
     ["tallinn"], _EUR, [".ee"]),
    ("LU", "Luxembourg", "Europe", ["luxembourgers"], "luxembourgish",
     [], _EUR, [".lu"]),
    ("LI", "Liechtenstein", "Europe", ["liechtensteiners"], "",
     ["vaduz"], ["chf", "swiss francs"], [".li"]),
    ("MC", "Monaco", "Europe", [], "",
     ["monte carlo"], _EUR, [".mc"]),
    ("AD", "Andorra", "Europe", ["andorrans"], "andorran",
     [], _EUR, [".ad"]),
    ("SM", "San Marino", "Europe", ["sammarinese"], "",
     [], _EUR, [".sm"]),
    ("IM", "Isle of Man", "Europe", ["manx"], "",
     [], [], [".im"]),
    ("JE", "Jersey", "Europe", [], "",
     ["saint helier"], ["jersey pound"], [".je"]),
    ("GG", "Guernsey", "Europe", [], "",
     ["st peter port"], ["guernsey pound"], [".gg"]),
    ("GI", "Gibraltar", "Europe", ["gibraltarians"], "",
     [], ["gibraltar pound"], [".gi"]),
    # ── Asia Pacific (Georgia -> Asia Pacific per leads.region) ──
    ("IN", "India", "Asia Pacific", ["india's", "bharat", "desi"], "indian",
     ["new delhi", "mumbai", "delhi", "bengaluru", "bangalore", "chennai",
      "hyderabad", "pune", "kolkata", "gurugram", "gurgaon", "noida", "ahmedabad"],
     ["₹", "rupee", "rupees", "crore", "crores", "lakh", "inr"],
     [".in", ".co.in", ".gov.in"]),
    ("CN", "China", "Asia Pacific", ["chinese", "prc", "beijing's"], "chinese",
     ["beijing", "shanghai", "shenzhen", "guangzhou", "hangzhou"],
     ["yuan", "renminbi", "rmb", "cny"], [".cn", ".com.cn"]),
    ("JP", "Japan", "Asia Pacific", ["japanese"], "japanese",
     ["tokyo", "osaka", "kyoto", "yokohama", "nagoya"],
     ["¥", "yen", "jpy"], [".jp", ".co.jp"]),
    ("KR", "South Korea", "Asia Pacific", ["korean", "korea", "rok"], "south korean",
     ["seoul", "busan", "incheon"], ["krw", "korean won"], [".kr", ".co.kr"]),
    ("KP", "North Korea", "Asia Pacific", ["dprk", "pyongyang"], "north korean",
     [], ["north korean won"], []),
    ("TW", "Taiwan", "Asia Pacific", ["taiwanese"], "taiwanese",
     ["taipei", "kaohsiung"], ["twd", "nt$"], [".tw", ".com.tw"]),
    ("HK", "Hong Kong", "Asia Pacific", ["hongkong", "kowloon"], "hong konger",
     ["tsim sha tsui"], ["hk$", "hkd"], [".hk", ".com.hk"]),
    ("MO", "Macau", "Asia Pacific", ["macao"], "",
     [], ["mop"], [".mo"]),
    ("SG", "Singapore", "Asia Pacific", ["singaporeans"], "singaporean",
     [], ["s$", "sgd"], [".sg", ".com.sg"]),
    ("MY", "Malaysia", "Asia Pacific", ["malaysians"], "malaysian",
     ["kuala lumpur", "penang", "johor"], ["ringgit", "myr"], [".my", ".com.my"]),
    ("ID", "Indonesia", "Asia Pacific", ["indonesians"], "indonesian",
     ["jakarta", "surabaya", "bandung"], ["rupiah", "idr"], [".id", ".co.id"]),
    ("TH", "Thailand", "Asia Pacific", ["thais", "thai"], "thai",
     ["bangkok", "phuket", "chiang mai"], ["baht", "thb"], [".th", ".co.th"]),
    ("VN", "Vietnam", "Asia Pacific", ["viet nam", "vietnamese"], "vietnamese",
     ["hanoi", "ho chi minh", "saigon"], ["vnd", "vietnamese dong"],
     [".vn", ".com.vn"]),
    ("PH", "Philippines", "Asia Pacific", ["filipinos", "filipino", "philippine"],
     "filipino",
     ["manila", "quezon city", "cebu"], ["philippine peso"], [".ph", ".com.ph"]),
    ("PK", "Pakistan", "Asia Pacific", ["pakistani"], "pakistani",
     ["karachi", "lahore", "islamabad", "rawalpindi"], ["pakistani rupee"],
     [".pk", ".com.pk"]),
    ("BD", "Bangladesh", "Asia Pacific", ["bangladeshi"], "bangladeshi",
     ["dhaka", "chittagong"], ["taka", "bdt"], [".bd", ".com.bd"]),
    ("LK", "Sri Lanka", "Asia Pacific", ["sri lankan", "sri lankans"], "",
     ["colombo"], ["sri lankan rupee"], [".lk"]),
    ("NP", "Nepal", "Asia Pacific", ["nepalese", "nepali"], "nepali",
     ["kathmandu"], ["nepalese rupee"], [".np"]),
    ("MV", "Maldives", "Asia Pacific", ["maldivian"], "",
     [], ["rufiyaa", "mvr"], [".mv"]),
    ("BT", "Bhutan", "Asia Pacific", ["bhutanese"], "",
     ["thimphu"], ["ngultrum"], [".bt"]),
    ("MM", "Myanmar", "Asia Pacific", ["burmese", "rangoon"], "",
     ["yangon", "naypyidaw"], ["kyat"], [".mm"]),
    ("KH", "Cambodia", "Asia Pacific", ["cambodians"], "cambodian",
     ["phnom penh"], ["riel", "khr"], [".kh"]),
    ("LA", "Laos", "Asia Pacific", ["laotians", "laotian"], "",
     ["vientiane"], ["lak"], [".la"]),
    ("BN", "Brunei", "Asia Pacific", ["bruneians", "bruneian"], "",
     ["bandar seri begawan"], ["brunei dollar"], [".bn"]),
    ("TL", "Timor-Leste", "Asia Pacific", ["east timor", "timorese", "timor"],
     "", ["dili"], ["us dollar"], [".tl"]),
    ("MN", "Mongolia", "Asia Pacific", ["mongolians", "mongolian"], "",
     ["ulaanbaatar"], ["tugrik"], [".mn"]),
    ("KZ", "Kazakhstan", "Asia Pacific", ["kazakhs", "kazakh", "kazakhstani"],
     "", ["almaty", "astana", "nur-sultan"], ["tenge", "kzt"], [".kz"]),
    ("UZ", "Uzbekistan", "Asia Pacific", ["uzbeks", "uzbek", "uzbekistani"], "",
     ["tashkent"], [], [".uz"]),
    ("KG", "Kyrgyzstan", "Asia Pacific", ["kyrgyz", "kyrgyzstani"], "",
     ["bishkek"], [], [".kg"]),
    ("TJ", "Tajikistan", "Asia Pacific", ["tajiks", "tajikistani"], "",
     ["dushanbe"], [], [".tj"]),
    ("AF", "Afghanistan", "Asia Pacific", ["afghans", "afghan"], "afghan",
     ["kabul"], ["afghani"], [".af"]),
    ("GE", "Georgia", "Asia Pacific", ["georgians"], "georgian",
     ["tbilisi", "batumi"], ["lari"], [".ge"]),
    ("AZ", "Azerbaijan", "Asia Pacific", ["azeris", "azerbaijani"], "",
     ["baku"], ["manat"], [".az"]),
    # ── Oceania ──
    ("AU", "Australia", "Oceania", ["aussie", "australians"], "australian",
     ["sydney", "melbourne", "brisbane", "adelaide", "canberra"],
     ["a$", "aud", "australian dollar"], [".com.au", ".au"]),
    ("NZ", "New Zealand", "Oceania", ["nz", "kiwi dollar"], "new zealander",
     ["auckland", "wellington", "christchurch"], ["nz$", "nzd"],
     [".nz", ".co.nz"]),
    ("AS", "American Samoa", "Oceania", ["pago pago"], "",
     [], [], []),
    ("FJ", "Fiji", "Oceania", ["fijians", "fijian"], "",
     ["suva"], ["fijian dollar"], [".fj"]),
    ("PG", "Papua New Guinea", "Oceania", ["png", "port moresby",
     "papua new guinean"], "",
     [], ["kina"], [".pg"]),
    ("VU", "Vanuatu", "Oceania", ["ni-vanuatu"], "",
     ["port vila"], ["vatu"], [".vu"]),
    ("SB", "Solomon Islands", "Oceania", ["honiara", "solomon islanders"], "",
     [], ["solomon islands dollar"], [".sb"]),
    ("NC", "New Caledonia", "Oceania", ["noumea"], "",
     [], ["cfp franc"], []),
    ("WS", "Samoa", "Oceania", ["samoans", "samoan", "apia"], "",
     [], ["tala"], [".ws"]),
    ("TO", "Tonga", "Oceania", ["tongans", "tongan"], "",
     ["nuku'alofa"], ["paanga"], [".to"]),
    # ── Middle East (Turkey/Cyprus/Egypt -> Middle East per leads.region) ──
    ("AE", "United Arab Emirates", "Middle East",
     ["uae", "u.a.e.", "emirates", "emirati"], "emirati",
     ["dubai", "abu dhabi", "sharjah"], ["aed", "dirham", "dirhams"],
     [".ae", ".co.ae"]),
    ("SA", "Saudi Arabia", "Middle East", ["saudi", "saudis", "ksa"], "saudi",
     ["riyadh", "jeddah", "dammam"], ["riyal", "sar", "saudi riyal"], [".sa"]),
    ("QA", "Qatar", "Middle East", ["qataris"], "qatari",
     ["doha"], ["qatari riyal", "qar"], [".qa"]),
    ("KW", "Kuwait", "Middle East", ["kuwaitis"], "kuwaiti",
     ["kuwait city"], ["kuwaiti dinar", "kwd"], [".kw"]),
    ("BH", "Bahrain", "Middle East", ["bahrainis"], "bahraini",
     ["manama"], ["bahraini dinar", "bhd"], [".bh", ".com.bh"]),
    ("OM", "Oman", "Middle East", ["omanis"], "omani",
     ["muscat"], ["omani rial", "omr"], [".om", ".com.om"]),
    ("YE", "Yemen", "Middle East", ["yemenis"], "yemeni",
     ["sanaa", "aden"], ["yemeni rial"], []),
    ("IL", "Israel", "Middle East", ["israelis", "idf"], "israeli",
     ["jerusalem", "tel aviv", "haifa"], ["shekel", "shekels", "ils"],
     [".il", ".co.il"]),
    ("PS", "Palestine", "Middle East", ["palestinian", "palestinians", "gaza",
     "west bank", "ramallah"], "", [], [], [".ps"]),
    ("LB", "Lebanon", "Middle East", ["lebanese"], "",
     ["beirut"], ["lebanese pound", "lbp"], [".lb", ".com.lb"]),
    ("SY", "Syria", "Middle East", ["syrians", "syrian"], "syrian",
     ["damascus", "aleppo"], ["syrian pound"], []),
    ("IQ", "Iraq", "Middle East", ["iraqis"], "iraqi",
     ["baghdad", "basra", "mosul"], ["iraqi dinar"], [".iq"]),
    ("IR", "Iran", "Middle East", ["iranians"], "iranian",
     ["tehran"], ["irr", "iranian rial"], [".ir"]),
    ("JO", "Jordan", "Middle East", ["jordanian", "jordanians", "jordan's",
     "kingdom of jordan"], "jordanian",
     ["amman"], ["jordanian dinar", "jod"], [".jo"]),
    ("TR", "Turkey", "Middle East", ["turkish", "turks", "turkiye"], "turkish",
     ["istanbul", "ankara", "izmir"], ["turkish lira"], [".com.tr", ".tr"]),
    ("CY", "Cyprus", "Middle East", ["cypriots"], "cypriot",
     ["nicosia", "limassol"], _EUR, [".cy"]),
    ("EG", "Egypt", "Middle East", ["egyptians"], "egyptian",
     ["cairo", "alexandria"], ["egyptian pound", "egp"], [".eg", ".com.eg"]),
    # ── Africa ──
    ("ZA", "South Africa", "Africa", ["south africans", "gauteng"],
     "south african",
     ["johannesburg", "cape town", "pretoria", "durban"],
     ["zar", "south african rand"], [".co.za", ".za"]),
    ("NG", "Nigeria", "Africa", ["nigerians"], "nigerian",
     ["lagos", "abuja", "ibadan", "port harcourt", "kano"],
     ["₦", "naira", "ngn"], [".ng", ".com.ng"]),
    ("KE", "Kenya", "Africa", ["kenyans"], "kenyan",
     ["nairobi", "mombasa"], ["kenyan shilling", "kes"], [".ke", ".co.ke"]),
    ("GH", "Ghana", "Africa", ["ghanaian", "ghanaians"], "ghanaian",
     ["accra", "kumasi"], ["cedi", "ghs"], [".gh", ".com.gh"]),
    ("TZ", "Tanzania", "Africa", ["tanzanian", "tanzanians", "zanzibar"], "",
     ["dar es salaam", "dodoma"], ["tanzanian shilling", "tzs"],
     [".tz", ".co.tz"]),
    ("UG", "Uganda", "Africa", ["ugandan", "ugandans"], "",
     ["kampala"], ["ugandan shilling", "ugx"], []),
    ("RW", "Rwanda", "Africa", ["rwandan", "rwandans"], "",
     ["kigali"], ["rwandan franc", "rwf"], []),
    ("ET", "Ethiopia", "Africa", ["ethiopian", "ethiopians"], "",
     ["addis ababa"], ["birr", "etb"], [".et"]),
    ("ZM", "Zambia", "Africa", ["zambian", "zambians"], "",
     ["lusaka"], ["kwacha", "zmw"], [".zm"]),
    ("MW", "Malawi", "Africa", ["malawian", "malawians"], "",
     ["lilongwe", "blantyre"], ["malawi kwacha"], []),
    ("ZW", "Zimbabwe", "Africa", ["zimbabwean", "zimbabweans"], "",
     ["harare", "bulawayo"], ["zimbabwe dollar"], [".zw", ".co.zw"]),
    ("BW", "Botswana", "Africa", ["botswanan"], "",
     ["gaborone"], ["botswana pula"], [".co.bw", ".bw"]),
    ("NA", "Namibia", "Africa", ["namibian", "namibians"], "",
     ["windhoek"], ["namibian dollar"], [".na"]),
    ("MU", "Mauritius", "Africa", ["mauritian", "mauritians"], "",
     ["port louis"], ["mauritian rupee", "mur"], []),
    ("MG", "Madagascar", "Africa", ["malagasy", "madagascan"], "",
     ["antananarivo"], ["ariary"], []),
    ("MZ", "Mozambique", "Africa", ["mozambican", "mozambicans"], "",
     ["maputo"], ["metical"], []),
    ("MA", "Morocco", "Africa", ["moroccans", "moroccan", "marrakech"], "",
     ["casablanca", "rabat"], ["moroccan dirham"], []),
    ("DZ", "Algeria", "Africa", ["algerians"], "algerian",
     ["algiers", "oran"], ["algerian dinar", "dzd"], []),
    ("TN", "Tunisia", "Africa", ["tunisians"], "tunisian",
     ["tunis"], ["tunisian dinar", "tnd"], [".tn"]),
    ("LY", "Libya", "Africa", ["libyans"], "libyan",
     ["tripoli", "benghazi"], ["libyan dinar"], []),
    ("SD", "Sudan", "Africa", ["sudanese", "khartoum"], "",
     [], ["sudanese pound"], []),
    ("SS", "South Sudan", "Africa", ["south sudanese", "juba"], "",
     [], ["south sudanese pound"], []),
    ("NE", "Niger", "Africa", ["nigerien", "niamey"], "",
     [], ["cfa franc"], []),
    ("TD", "Chad", "Africa", ["chadian", "ndjamena"], "",
     [], ["cfa franc"], []),
    ("SN", "Senegal", "Africa", ["senegalese", "dakar"], "",
     [], ["cfa franc"], []),
    ("CI", "Ivory Coast", "Africa", ["cote d'ivoire", "ivorian", "abidjan"], "",
     [], ["cfa franc"], []),
    ("CM", "Cameroon", "Africa", ["cameroonian", "cameroonians", "douala"], "",
     ["yaounde"], ["cfa franc"], []),
    ("CD", "Democratic Republic of the Congo", "Africa",
     ["dr congo", "drc", "kinshasa"], "",
     [], ["congolese franc"], []),
    ("CG", "Republic of the Congo", "Africa", ["congolese", "brazzaville"], "",
     [], ["cfa franc"], []),
    ("AO", "Angola", "Africa", ["angolans", "luanda"], "",
     [], ["kwanza", "aoa"], []),
    ("GN", "Guinea", "Africa", ["guinean", "conakry"], "",
     [], ["guinean franc"], []),
    ("GQ", "Equatorial Guinea", "Africa", ["malabo"], "",
     [], ["cfa franc"], []),
    ("GW", "Guinea-Bissau", "Africa", ["bissau"], "",
     [], ["cfa franc"], []),
    ("GM", "Gambia", "Africa", ["gambian", "gambians", "the gambia", "banjul"],
     "", [], ["dalasi"], []),
    ("LR", "Liberia", "Africa", ["liberian", "liberians", "monrovia"], "",
     [], ["liberian dollar"], []),
    ("SL", "Sierra Leone", "Africa", ["freetown", "sierra leonean"], "",
     [], ["leone"], []),
    ("ML", "Mali", "Africa", ["malian", "bamako"], "",
     [], ["cfa franc"], []),
    ("BF", "Burkina Faso", "Africa", ["burkinabe", "ouagadougou"], "",
     [], ["cfa franc"], []),
    ("BJ", "Benin", "Africa", ["beninese", "cotonou"], "",
     [], ["cfa franc"], []),
    ("TG", "Togo", "Africa", ["togolese", "lome"], "",
     [], ["cfa franc"], []),
    ("GA", "Gabon", "Africa", ["gabonese", "libreville"], "",
     [], ["cfa franc"], []),
    ("BI", "Burundi", "Africa", ["burundian", "bujumbura"], "",
     [], ["burundian franc"], []),
    ("SO", "Somalia", "Africa", ["somali", "somalis", "mogadishu"], "",
     [], ["somali shilling"], []),
    ("DJ", "Djibouti", "Africa", ["djiboutian"], "",
     [], ["djiboutian franc"], []),
    ("ER", "Eritrea", "Africa", ["eritrean", "asmara"], "",
     [], ["nakfa"], []),
    ("CV", "Cape Verde", "Africa", ["cabo verde", "praia"], "",
     [], ["escudo"], []),
    ("SC", "Seychelles", "Africa", ["seychellois"], "",
     [], ["seychellois rupee"], []),
    ("MR", "Mauritania", "Africa", ["mauritanian", "nouakchott"], "",
     [], ["ouguiya"], []),
    ("SZ", "Eswatini", "Africa", ["swaziland", "swazi", "mbabane"], "",
     [], ["lilangeni"], [".co.sz", ".sz"]),
    ("LS", "Lesotho", "Africa", ["basotho", "maseru"], "",
     [], ["loti"], []),
]

# Countries whose bare NAME is too ambiguous to use as a detection token
# ("Georgia" = US state, "Jordan" = person name). Their aliases/cities carry
# detection instead; get_profile("Georgia") still resolves by name.
_NO_NAME_TOKEN = {"GE", "JO"}

# Ambiguous words that only false-positive inside a longer phrase which is NOT
# itself a country token. (Phrases that ARE country tokens — "papua new guinea",
# "north korea", "south sudan", "equatorial guinea", "american samoa" — are
# handled by longest-first token removal instead.)
_MASK_PHRASES = [
    ("new mexico", ["mexico"], ""),
    ("indian ocean", ["indian"], ""),
    ("new south wales", ["wales", "south wales"], ""),
    ("north america", ["america"], ""),
    ("latin america", ["america"], ""),
    ("south america", ["america"], ""),
    ("central america", ["america"], ""),
    ("north american", ["american"], ""),
    ("latin american", ["american"], ""),
    ("south american", ["american"], ""),
    ("central american", ["american"], ""),
    ("new england", ["england"], ""),
    ("british columbia", ["british"], ""),
    ("northern ireland", ["ireland"], ""),
    ("new jersey", ["jersey"], ""),
    ("new york", ["york"], ""),
    ("new guinea", ["guinea"], "(?<!papua )"),
    ("guinea bissau", ["guinea"], ""),
]


# ── Sub-national places (states, provinces, emirates, Länder) ────────────────
# A sub-national place name is decent evidence of its country on its own:
# "Ontario" means Canada, "Bavaria" Germany, "Maharashtra" India. These are
# matched (word-boundary, longest-first) BEFORE the country tokens and their
# matches are consumed, so:
#   - "New South Wales" credits AU and can never also credit the bare "wales"
#     (GB) token, "British Columbia" never "british" (GB), "Northern Ireland"
#     never "ireland" (IE), "Lower Saxony" never "saxony".
# Deliberately omitted: bare "Georgia" (US state / country — handled by the
# co-occurrence rule below), "Victoria" (a dropped-ambiguous city name),
# "Punjab" (India AND Pakistan), "acre" (an English word).
_SUBNAT_ROWS = [
    ("US", [
        "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
        "connecticut", "delaware", "florida", "hawaii", "idaho", "illinois",
        "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
        "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
        "missouri", "montana", "nebraska", "nevada", "new hampshire",
        "new jersey", "new mexico", "new york", "north carolina",
        "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
        "rhode island", "south carolina", "south dakota", "tennessee", "texas",
        "utah", "vermont", "virginia", "west virginia", "wisconsin", "wyoming",
    ]),
    ("CA", ["ontario", "quebec", "british columbia", "alberta", "manitoba",
            "saskatchewan", "nova scotia", "new brunswick", "newfoundland",
            "prince edward island"]),
    ("AU", ["new south wales", "queensland", "tasmania", "western australia",
            "south australia", "australian capital territory"]),
    ("IN", ["maharashtra", "gujarat", "tamil nadu", "karnataka", "kerala",
            "rajasthan", "uttar pradesh", "madhya pradesh", "west bengal",
            "andhra pradesh", "telangana", "odisha", "bihar", "haryana",
            "assam"]),
    ("CN", ["guangdong", "zhejiang", "jiangsu", "shandong", "sichuan",
            "fujian", "hunan", "hubei", "hebei", "xinjiang", "yunnan",
            "guangxi", "anhui", "guizhou", "liaoning", "shaanxi", "tibet"]),
    ("DE", ["bavaria", "baden-württemberg", "baden-wurttemberg",
            "baden württemberg", "saxony", "saxony-anhalt", "hesse",
            "north rhine-westphalia", "north rhine westphalia",
            "lower saxony", "thuringia", "brandenburg",
            "rhineland-palatinate", "rhineland palatinate",
            "schleswig-holstein", "mecklenburg-vorpommern", "saarland"]),
    ("BR", ["minas gerais", "rio grande do sul", "rio grande do norte",
            "bahia", "pernambuco", "ceara", "parana", "paraná",
            "santa catarina", "espirito santo", "goias", "amazonas",
            "distrito federal", "mato grosso", "maranhao", "paraiba"]),
    ("AE", ["ajman", "ras al khaimah", "fujairah", "umm al quwain"]),
    ("SA", ["neom", "hejaz", "asir", "najd"]),
    ("GB", ["northern ireland"]),
]

# Bare "Georgia" stays a deliberate non-token (module docstring) EXCEPT when a
# US context signal co-occurs: "State Funds 16 Georgia Shelters" is US news
# (Atlanta/county context), "Georgian wine exports" is the country and never
# fires (no bare word, no signal).
_GEORGIA_US_SIGNALS = ("atlanta", "savannah", "augusta", "county",
                       "state of georgia", "georgia's governor",
                       "georgia department", "georgia general assembly")


def _build() -> None:
    """Build the profiles, lookup maps and detection structures once at import."""
    global _PROFILES, _BY_KEY, _ORDERED_TOKENS, _PLAIN_TOKENS, _TLD_MAP
    global _SUBNAT_TOKENS, _GEORGIA_RX
    _PROFILES = {}
    for iso2, name, region, aliases, nationality, cities, currency, tlds in _ROWS:
        if iso2 in _PROFILES:
            continue    # duplicate rows are mistakes; first entry wins
        _PROFILES[iso2] = {
            "iso2": iso2, "name": name, "region": region,
            "gn": _gn(iso2, region),
            "aliases": [name.lower()] + [a.lower() for a in aliases],
            "nationality": (nationality or "").lower(),
            "cities": [c.lower() for c in cities],
            "currency": [c.lower() for c in currency],
            "tld": list(tlds),
            "outlets": [],
        }
    # lookup: name / iso2 / alias / nationality -> iso2
    _BY_KEY = {}
    for iso2, p in _PROFILES.items():
        _BY_KEY.setdefault(iso2.lower(), iso2)
        _BY_KEY.setdefault(p["name"].lower(), iso2)
        for a in p["aliases"]:
            _BY_KEY.setdefault(a, iso2)
        if p["nationality"]:
            _BY_KEY.setdefault(p["nationality"], iso2)

    # detection tokens: a token -> the set of countries it belongs to (shared
    # tokens like "€" or "cfa franc" credit all of their users). The country's
    # own name is a token too, unless it is on the ambiguous-name list.
    tokens = {}    # token -> set of iso2 (shared tokens credit all users)
    plain = {}     # symbols and other non-word tokens
    for iso2, p in _PROFILES.items():
        toks = list(p["aliases"])
        if p["nationality"]:
            toks.append(p["nationality"])
        if iso2 in _NO_NAME_TOKEN and toks and toks[0] == p["name"].lower():
            # "Georgia"/"Jordan" are too ambiguous as bare words (US state /
            # person name) — drop the name, keep alias/nationality/city tokens.
            toks = toks[1:]
        else:
            toks.append(p["name"].lower())
        toks += p["cities"] + p["currency"]
        for t in toks:
            t = t.strip()
            if not t:
                continue
            target = tokens if re.search(r"[0-9a-z]", t) else plain
            target.setdefault(t, set()).add(iso2)
    # longest first, so "papua new guinea" is consumed before "guinea" can match
    _ORDERED_TOKENS = [
        (re.compile(r"(?<![0-9a-z])" + re.escape(tok) + r"(?![0-9a-z])", re.I),
         isos)
        for tok, isos in sorted(tokens.items(), key=lambda kv: -len(kv[0]))]
    _PLAIN_TOKENS = [(tok, isos)
                     for tok, isos in sorted(plain.items(), key=lambda kv: -len(kv[0]))]

    # ccTLD map: ("com","au") -> AU etc.
    _TLD_MAP = {}
    for iso2, p in _PROFILES.items():
        for t in p["tld"]:
            labels = tuple(t.strip(".").lower().split("."))
            _TLD_MAP.setdefault(labels, iso2)

    # sub-national place tokens, longest phrase first so "Lower Saxony" is
    # consumed before "saxony" and "West Virginia" before "virginia"
    _SUBNAT_TOKENS = sorted(
        ((re.compile(r"(?<![0-9a-z])" + re.escape(phrase) + r"(?![0-9a-z])", re.I),
          iso2)
         for iso2, phrases in _SUBNAT_ROWS for phrase in phrases),
        key=lambda p: -len(p[0].pattern))
    _GEORGIA_RX = re.compile(r"(?<![0-9a-z])georgia(?![0-9a-z])", re.I)


_build()


# ── lookups ──────────────────────────────────────────────────────────────────
def get_profile(name_or_iso2: str) -> dict | None:
    """Country profile in the same shape as config.COUNTRIES entries (plus
    'iso2' and 'region'), or None when unknown. Accepts English names, common
    short forms ("UK", "UAE") and ISO2 codes ("PL", "SG")."""
    if not name_or_iso2:
        return None
    key = re.sub(r"\s+", " ", name_or_iso2.strip().lower())
    iso2 = _BY_KEY.get(key)
    if not iso2:
        return None
    return _PROFILES[iso2]


def region_of(iso2: str) -> str:
    """Region name for an ISO2 code, '' when unknown."""
    p = _PROFILES.get((iso2 or "").upper())
    return p["region"] if p else ""


def countries_in(region: str) -> list[str]:
    """ISO2 codes belonging to a region, sorted."""
    return sorted(i for i, p in _PROFILES.items() if p["region"] == region)


def name_of(iso2: str) -> str:
    p = _PROFILES.get((iso2 or "").upper())
    return p["name"] if p else ""


# ── detection ────────────────────────────────────────────────────────────────
def _normalize(text: str) -> str:
    t = (text or "").replace("’", "'").replace("`", "'").lower()
    return t


def _mask_phrases(t: str) -> str:
    for phrase, words, guard in _MASK_PHRASES:
        if phrase in t:
            # replace only the colliding words INSIDE the phrase, e.g.
            # "new mexico" -> "new qzxwq" so the bare "mexico" token can't match
            pat = re.compile("|".join(re.escape(w) for w in words))
            rx = re.compile(guard + re.escape(phrase))
            t = rx.sub(lambda m: pat.sub("qzxwq", m.group(0)), t)
    return t


def countries_in_text(text: str) -> set[str]:
    """ISO2 codes whose name/alias/nationality/city/currency tokens appear in the
    text (word-boundary, case-insensitive). Longest tokens consume their matches
    first, so compound country names never leak into their substrings. Sub-national
    place names (states, provinces, Länder, emirates) are matched and consumed
    BEFORE the country tokens, so "New South Wales" credits AU and can never also
    credit the bare "wales" (GB) token."""
    t = _normalize(text)
    if not t:
        return set()
    found: set[str] = set()
    # 1. sub-national places first: "new south wales" -> AU before the "wales"
    #    (GB) alias can match, "british columbia" -> CA before "british" (GB).
    for rx, iso2 in _SUBNAT_TOKENS:
        if rx.search(t):
            found.add(iso2)
            t = rx.sub(" ", t)   # consume: shorter phrases/tokens can't match
    # 2. bare "Georgia" -> US only when a US context signal co-occurs (see
    #    _GEORGIA_US_SIGNALS); "Georgian"/"Tbilisi" keep crediting GE as before.
    if _GEORGIA_RX.search(t) and any(s in t for s in _GEORGIA_US_SIGNALS):
        found.add("US")
        t = _GEORGIA_RX.sub(" ", t)
    # 3. false-positive phrase masking, then the country tokens themselves
    t = _mask_phrases(t)
    for rx, isos in _ORDERED_TOKENS:
        if rx.search(t):
            found |= isos
            t = rx.sub(" ", t)   # consume: shorter tokens can't match inside
    for tok, isos in _PLAIN_TOKENS:
        if tok in t:
            found |= isos
    return found


def _url_tld_countries(url: str) -> set[str]:
    """Countries whose ccTLD the URL's hostname ends with (".in" matches
    "example.co.in"; ".com" is not a country TLD so it matches nothing)."""
    out: set[str] = set()
    if not url:
        return out
    m = re.search(r"(?:https?://)?([^/?:#\s]+)", url.lower())
    if not m:
        return out
    host = m.group(1).split("@")[-1].split(":")[0].strip(".")
    labels = host.split(".") if "." in host else []
    for depth in (1, 2, 3):
        if len(labels) >= depth:
            iso2 = _TLD_MAP.get(tuple(labels[-depth:]))
            if iso2:
                out.add(iso2)
    return out


def article_countries(article) -> set[str]:
    """ISO2 codes an article mentions: title + summary + source text, plus the
    URL's ccTLD. Case-insensitive word-boundary matching (see module docstring
    for the ambiguity rules)."""
    text = " ".join([getattr(article, "title", "") or "",
                     getattr(article, "summary", "") or "",
                     getattr(article, "source", "") or ""])
    found = countries_in_text(text)
    found |= _url_tld_countries(getattr(article, "url", "") or "")
    return found
