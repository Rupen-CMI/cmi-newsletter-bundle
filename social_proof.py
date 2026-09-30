"""Pick social proof that matches the client's industry.

The failure this exists to prevent: a brief for Barsys (bar robots, consumer
appliances) showing Genentech, Alexion and Glenmark logos. Wrong logos are worse
than none — they tell the reader the sender did not look at their business.

Design:
  1. Classify the tenant profile into one broad vertical, deterministically.
     Keyword scoring first, LLM only to break a tie or when nothing scores.
  2. Serve logos and testimonials from that vertical's pool only.
  3. FAIL CLOSED. Below `MIN_ITEMS` verified entries for the vertical, return
     nothing. An empty block is honest; a mismatched one is not.

Every logo and quote in assets/social_proof.json is a real, published CMI client
taken from coherentmarketinsights.com/testimonials. Nothing here is invented.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

HERE = os.path.dirname(__file__)
ASSETS = os.path.join(HERE, "assets", "social_proof.json")

# Below this many verified items we show nothing rather than pad the row.
MIN_LOGOS = 5
MIN_QUOTES = 2

# Broad verticals, each with the words that identify a client in it. Scored by
# how many distinct keywords hit, so "hospitals and oncology centers" lands in
# life_sciences rather than matching a stray word elsewhere.
VERTICALS: dict[str, dict] = {
    "life_sciences": {
        "label": "life sciences and healthcare",
        "keywords": ["pharma", "pharmaceutic", "biotech", "biopharma", "oncology", "cancer",
                     "hospital", "clinic", "medical device", "diagnostic", "therapeut",
                     "cell therapy", "gene therapy", "car-t", "vaccine", "drug", "clinical",
                     "healthcare", "health care", "patient", "surgical", "dental", "life science",
                     "laboratory", "lab equipment", "cro", "cdmo", "apheresis", "genomic"],
    },
    "consumer": {
        "label": "consumer products and appliances",
        "keywords": ["consumer", "appliance", "beverage", "food", "drink", "cocktail", "bar ",
                     "kitchen", "household", "retail", "grocery", "cosmetic", "beauty",
                     "skincare", "personal care", "apparel", "textile", "furniture", "toy",
                     "restaurant", "hospitality", "hotel", "packaged goods", "cpg", "brewing",
                     "spirits", "alcohol", "coffee", "tea", "snack", "dairy", "nutrition"],
    },
    "technology": {
        "label": "technology and electronics",
        "keywords": ["software", "saas", "platform", "cloud", "ai ", "artificial intelligence",
                     "machine learning", "data analytics", "semiconductor", "electronic",
                     "computing", "hardware", "device maker", "telecom", "telecommunication",
                     "network", "cyber", "fintech app", "mobile", "app developer", "robotics",
                     "automation software", "display", "sensor", "iot"],
    },
    "industrial": {
        "label": "industrial and manufacturing",
        "keywords": ["manufactur", "industrial", "machinery", "equipment", "engineering",
                     "automotive", "aerospace", "defense", "defence", "construction",
                     "chemical", "materials", "metal", "steel", "plastic", "polymer",
                     "packaging", "logistics", "supply chain", "mining", "agricultur",
                     "energy", "oil", "gas", "renewable", "utility", "power grid", "tools"],
    },
    "services": {
        "label": "professional and financial services",
        "keywords": ["consult", "advisory", "bank", "insurance", "financial service",
                     "asset management", "private equity", "venture", "legal", "law firm",
                     "accounting", "market research", "staffing", "recruit", "education",
                     "university", "training", "government", "public sector", "media",
                     "marketing agency", "procurement"],
    },
}


def _load() -> dict:
    with open(ASSETS, encoding="utf-8") as f:
        return json.load(f)


def classify(profile, llm_module=None) -> tuple[str, str, bool]:
    """Return (vertical_key, how_it_was_decided, confident). Never raises."""
    haystack = " ".join([
        getattr(profile, "value_prop", "") or "",
        " ".join(getattr(profile, "industries", []) or []),
        getattr(profile, "company", "") or "",
    ]).lower()

    scores = {k: sum(1 for w in v["keywords"] if w in haystack)
              for k, v in VERTICALS.items()}
    best = max(scores, key=lambda k: scores[k])
    ranked = sorted(scores.values(), reverse=True)
    clear = scores[best] >= 2 and (len(ranked) < 2 or ranked[0] > ranked[1])
    if clear:
        return best, f"keywords ({scores[best]} hits)", True

    # Ambiguous or nothing scored: ask the model, but only to choose from the list.
    if llm_module is not None:
        try:
            data = llm_module.chat_json(
                "You classify a company into exactly one broad industry vertical. "
                "Answer with one key only.",
                f"COMPANY: {getattr(profile, 'company', '')}\n"
                f"WHAT THEY SELL: {getattr(profile, 'value_prop', '')}\n"
                f"THEIR CUSTOMERS: {', '.join(getattr(profile, 'industries', []) or [])}\n\n"
                f"KEYS: {', '.join(VERTICALS)}\n"
                'Return JSON {"vertical":"<one key>"}.',
                max_tokens=60, default={}) or {}
            pick = str(data.get("vertical", "")).strip().lower()
            if pick in VERTICALS:
                return pick, "classified by model", True
        except Exception:  # noqa: BLE001 — classification must never break a run
            pass

    # A single keyword hit is not evidence. "e-commerce retailers" alone should not
    # put a payments company in the consumer pool, so treat weak as unknown.
    if scores[best] > 0:
        return best, f"weak keyword match ({scores[best]} hit, not trusted)", False
    return "", "no keyword or model match", False


def for_profile(profile, llm_module=None, *, want_logos=True, want_quotes=True) -> dict:
    """Logos and testimonials for this client's vertical, or empty if unsure."""
    data = _load()
    key, why, confident = classify(profile, llm_module)
    pools = data.get("verticals", {})
    pool = pools.get(key, {}) if (key and confident) else {}

    logos = list(pool.get("logos") or [])
    quotes = list(pool.get("testimonials") or [])

    # Rotate which three quotes appear, seeded by the client name. Two clients in the
    # same vertical get different quotes, and the same client gets the same three
    # every time so a re-run is reproducible.
    if quotes:
        # Shuffle deterministically rather than rotating: a rotation makes two
        # neighbouring seeds share two of their three quotes.
        seed = int(hashlib.sha1((getattr(profile, "company", "") or "x").lower()
                                .encode("utf-8")).hexdigest(), 16)
        quotes = sorted(quotes, key=lambda q: hashlib.sha1(
            f'{seed}:{q["company"]}'.encode("utf-8")).hexdigest())

    label = VERTICALS[key]["label"] if key in VERTICALS else "unknown"
    notes = [f"vertical: {label} ({why})"]
    if not confident:
        notes.append("social proof hidden: the vertical could not be established "
                     "confidently, and wrong logos are worse than none")

    if not want_logos:
        logos = []
    elif len(logos) < MIN_LOGOS:
        notes.append(f"logos hidden: only {len(logos)} verified for this vertical "
                     f"(need {MIN_LOGOS})")
        logos = []
    if not want_quotes:
        quotes = []
    elif len(quotes) < MIN_QUOTES:
        notes.append(f"testimonials hidden: only {len(quotes)} verified for this vertical "
                     f"(need {MIN_QUOTES})")
        quotes = []

    return {"logos": logos[:20], "testimonials": quotes[:3],
            "vertical": key if confident else "",
            "vertical_label": label if confident else "",
            "confident": confident, "notes": notes}
