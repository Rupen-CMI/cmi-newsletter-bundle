"""Turn pipeline output into the template context for either brief type.

Two brief types, one template (`templates/cmi_brief.html`) because they are
structurally identical:

  signal  -- Potential Customer / Industry Outlook / Competition Outlook
  report  -- Your Market      / Industry Outlook / Competition Outlook

The only real difference is section 1: the signal brief profiles a prospect, the
report brief profiles the market the reader asked about.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

import llm
import social_proof
import voice

HERE = os.path.dirname(__file__)
DEMO_URL = "https://www.coherentmarketinsights.com/talkwithanalyst"

# An "Unsubscribe" that goes nowhere ("#") is worse than none: bulk filters look for a
# working opt-out, and a dead one invites spam reports. Point it at the site's contact
# page until a per-recipient unsubscribe endpoint exists.
UNSUB_URL = os.getenv("CMI_UNSUB_URL", "https://www.coherentmarketinsights.com/contact-us")


def _clean(s: str) -> str:
    """Strip the things the voice rules ban but a model still sometimes emits."""
    s = (s or "").strip()
    s = re.sub(r"<(?!/?[bi]\b)[^>]*>", "", s)          # only <b>/<i> survive
    s = re.sub(r"\s*[—–]\s*", ", ", s)                  # em dash -> comma
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"\s+([.,;:])", r"\1", s)
    return re.sub(r"\s{2,}", " ", s)


def _social(profile, include_logos: bool, include_quotes: bool) -> dict:
    """Vertical-matched social proof. Empty when the vertical is not established."""
    return social_proof.for_profile(profile, llm, want_logos=include_logos,
                                    want_quotes=include_quotes)


# ── LLM helpers ───────────────────────────────────────────────────────────────
def analyst_bullets(profile, signal: str, extra: str = "") -> list[str]:
    system = voice.bullet_system().format(company=profile.company, short=profile.short)
    user = (f"SELLER: {profile.company} — {profile.value_prop}\n"
            f"MARKET: {getattr(profile, 'market', 'Global')}\n"
            f"SIGNAL:\n{signal}\n{extra}\n\n"
            'Return JSON {"bullets":["<b>lead.</b> support.", "...", "..."]} with exactly 3.')
    data = llm.chat_json(system, user, max_tokens=900, default={}) or {}
    out = [_clean(b) for b in (data.get("bullets") or []) if str(b).strip()]
    return out[:3]


# The six engagement dimensions, in the order caps_projects asks for them. Used as the
# fallback when a returned title is too mangled to salvage a dimension from.
_DIMENSIONS = ("supplier and demand", "capacity and input supply", "pricing and access",
               "customer readiness and adoption", "competitor and pipeline", "market entry")


def caps_projects(profile) -> tuple[str, list[dict]]:
    """Six 'engagements' for the Inside CMI panel, named for the seller's industry."""
    system = (
        "You describe Coherent Market Insights research engagements. Name each one in the "
        "pattern '<industry> industry outlook, <dimension> landscape'. Describe the "
        "capability delivered, never a named client. Professional register.\n\n"
        "TITLE FORMAT IS FIXED. Every title is exactly: an industry phrase, then the "
        "words 'industry outlook', then a comma, then the dimension, then the word "
        "'landscape'. Worked examples of correct titles:\n"
        "  Cell and gene therapy industry outlook, supplier and demand landscape\n"
        "  Biologics manufacturing industry outlook, capacity and input supply landscape\n"
        "  Advanced therapies industry outlook, payer and reimbursement landscape\n"
        "A title that omits the industry, drops 'industry outlook', or reorders these "
        "parts is wrong. Vary the industry phrase across the six so they do not all "
        "repeat one label.\n"
        "The industry phrase names the SELLER'S OWN market and its adjacent segments "
        "(the product space CMI researches), not the seller's customer verticals. For a "
        "cell-therapy developer that means phrases like 'cell and gene therapy', "
        "'biologics manufacturing', 'advanced therapies', 'oncology therapeutics', "
        "'tertiary oncology care', 'specialty therapeutics'. A title built on a buyer "
        "vertical ('insurance and health financing industry outlook') is wrong.\n\n"
        "NEVER name the seller or its products. These are engagements delivered for "
        "OTHER, unnamed clients in the same space, and writing the seller's own product "
        "name into one turns it into a fabricated case study. Say 'the client', or name "
        "the capability. Concrete industry nouns (apheresis, viral vector, reimbursement "
        "precedent, cost per batch) are wanted; invented client facts are not.\n\n" + voice.active_style()
    )
    user = (f"SELLER: {profile.company} — {profile.value_prop}\n"
            f"THEIR INDUSTRY: {', '.join(profile.industries)}\n\n"
            'Return JSON {"industry":"<short industry phrase for the intro line>",'
            '"projects":[{"title":"...","body_html":"2 sentences, <b> allowed"}]} '
            "with exactly 6 projects, covering supplier and demand, manufacturing or "
            "capacity, pricing and access, customer readiness, competitor tracking, "
            "and market entry.\n"
            f"Do not write the words '{profile.company}' anywhere in the output.")
    data = llm.chat_json(system, user, max_tokens=1600, default={}) or {}
    projects = [{"title": _clean(p.get("title", "")), "body_html": _clean(p.get("body_html", ""))}
                for p in (data.get("projects") or [])][:6]
    # A curated caps_industry (company_overrides.json) names the seller's own market;
    # the model otherwise reaches for a buyer vertical here.
    import tenant as _tenant
    _ov = _tenant._overrides_for(getattr(profile, "company", ""))
    industry = ((_ov.get("caps_industry") or "").strip()
                or _clean(data.get("industry", ""))
                or ", ".join(profile.industries[:2]))

    # Repair what the model got wrong rather than shipping it. A smaller model drops the
    # '<industry> industry outlook, <dimension> landscape' shape and writes the seller's
    # product name into the body, which reads as an invented case study.
    _seller_bits = [b for b in re.split(r"[\s,]+", profile.company) if len(b) > 3]

    def _fix_title(t: str, i: int) -> str:
        if re.search(r"industry outlook,.*landscape\s*$", t, re.I):
            return t
        # Salvage the dimension the model chose; re-hang it on the required pattern.
        dim = re.sub(r"\b(outlook|landscape|industry)\b", " ", t, flags=re.I)
        dim = re.sub(r"[,\s]+", " ", dim).strip() or _DIMENSIONS[i % len(_DIMENSIONS)]
        return f"{industry} industry outlook, {dim.lower()} landscape"

    for i, p in enumerate(projects):
        p["title"] = _fix_title(p["title"], i)
        for bit in _seller_bits:                    # never let the seller's own name through
            p["body_html"] = re.sub(rf"\b{re.escape(bit)}\b", "the client", p["body_html"], flags=re.I)
    while len(projects) < 6:                       # never render an empty card
        projects.append(projects[len(projects) % max(1, len(projects))] if projects
                        else {"title": f"{industry} industry outlook, supplier and demand landscape",
                              "body_html": "Supplier base, demand and spending patterns mapped "
                                           "across the client's priority countries."})
    return industry, projects


def editorial(profile, sections: list[dict]) -> tuple[str, list[dict]]:
    """The lede and the three top-3 lines, written after the sections exist."""
    system = ("You write the opening of a market brief for one named client.\n\n"
              + voice.active_style())
    body = "\n".join(f"[{s['label']}] {s['title']}: {re.sub(r'<[^>]+>', '', s['body_html'])[:280]}"
                     for s in sections)
    user = (f"CLIENT: {profile.company} ({profile.short}) — {profile.value_prop}\n"
            f"MARKET: {getattr(profile, 'market', 'Global')}\n"
            f"VERTICALS TRACKED: {', '.join(profile.industries)}\n"
            f"SECTIONS:\n{body}\n\n"
            'Return JSON {"lede":"3 to 5 short sentences","top3":["...","...","..."]}. '
            "The lede opens with what CMI tracked, then one short sentence per signal, "
            "then one sentence on what they mean together. Each top3 line is a bolded "
            "claim in <b></b> then one plain sentence on why it matters. Same order as "
            "SECTIONS.")
    data = llm.chat_json(system, user, max_tokens=1000, default={}) or {}
    lede = _clean(data.get("lede", ""))
    lines = [_clean(x) for x in (data.get("top3") or [])][:3]
    while len(lines) < len(sections):
        i = len(lines)
        lines.append(f"<b>{sections[i]['title']}</b>")
    top3 = [{"cat": s["label"], "html": lines[i]} for i, s in enumerate(sections)]
    return lede, top3


# ── section builders ──────────────────────────────────────────────────────────
def _prospect(label, card: dict, cell_labels, *, flag="", title="") -> dict:
    """Map a brief_builder card onto the template shape.

    brief_builder emits company_name / narrative_html / left / right; the template
    wants title / body_html / cells. Anything already in template shape passes through,
    which is how the report brief's hand-built market card works.
    """
    title = title or card.get("company_name") or card.get("name") or "—"
    body = _clean(card.get("narrative_html") or card.get("body_html") or card.get("summary") or "")
    cells = card.get("cells") or []
    if not cells:
        left, right = card.get("left") or {}, card.get("right") or {}
        cells = [{"h": left.get("h") or cell_labels[0], "v": _clean(left.get("v", ""))},
                 {"h": right.get("h") or cell_labels[1], "v": _clean(right.get("v", ""))}]
    cells = [c for c in cells if (c.get("v") or "").strip()][:2]
    return {"label": label, "kind": "prospect",
            "flag": flag or _clean(card.get("flag", "")) or label,
            "title": _clean(title), "body_html": body, "cells": cells,
            # The template links the citation when a URL survives this far; an empty
            # string renders as plain text rather than a dead anchor.
            "source_url": (card.get("source_url") or "").strip(),
            "source_name": _clean(card.get("source_name") or "")}


def _story(label, title, body_html, *, source_url="", source_name="") -> dict:
    return {"label": label, "kind": "story", "title": _clean(title),
            "body_html": _clean(body_html), "flag": "", "cells": [],
            "source_url": (source_url or "").strip(),
            "source_name": _clean(source_name or "")}


def build_context(profile, brief: dict, kind: str, *, market_topic: str = "",
                  include_logos=True, include_quotes=True, log=print) -> dict:
    """`brief` is whatever brief_builder.build_brief produced."""
    now = datetime.now(timezone.utc)
    market = getattr(profile, "market", "Global")
    sections: list[dict] = []

    # brief_builder's own key names -- "customer"/"competition" are its debug dict,
    # the rendered cards live under these.
    cust = brief.get("potential_customer") or {}
    comp = brief.get("competition_outlook") or {}
    news = (brief.get("industry_outlook") or [{}])[0] or {}

    # -- section 1 -------------------------------------------------------------
    if kind == "report" and market_topic:
        log("    · section 1: your market")
        sections.append(_prospect(
            "Your Market",
            {"body_html": news.get("body_html", ""),
             "source_url": news.get("source_url", ""),
             "source_name": news.get("source_name", ""),
             "cells": [{"h": "What is changing", "v": _clean(news.get("headline", ""))}]},
            ("What is changing", "Why it matters for you"),
            flag=f"Your Market · {market_topic.lower()}", title=market_topic))
    elif cust:
        log("    · section 1: potential customer")
        sections.append(_prospect("Potential Customer", cust,
                                  ("Why now", "Who to target")))

    # -- section 2 -------------------------------------------------------------
    # Report briefs get Market Moves here; signal briefs keep Industry Outlook. The two
    # never both appear, so a report brief stays three sections long.
    moves = brief.get("market_moves") or {}
    if kind == "report" and moves:
        log("    · section 2: market moves")
        sections.append(_prospect("Market Moves", moves,
                                  ("Who is moving", "What it signals")))
    elif news:
        log("    · section 2: industry outlook")
        sections.append(_story("Industry Outlook", news.get("headline", ""),
                               news.get("body_html", "")))

    # -- section 3 -------------------------------------------------------------
    if comp:
        log("    · section 3: competition outlook")
        sections.append(_prospect("Competition Outlook", comp,
                                  ("Why it matters", "Their edge")))

    # -- analyst opinions ------------------------------------------------------
    for s in sections:
        log(f"    · analyst opinion: {s['label']}")
        s["bullets"] = analyst_bullets(
            profile, f"{s['title']}\n{re.sub(r'<[^>]+>', '', s['body_html'])}")

    log("    · editorial (lede + top 3)")
    lede, top3 = editorial(profile, sections)

    cta_set = voice.ctas(DEMO_URL, DEMO_URL)
    for i, s in enumerate(sections):
        s["cta"] = cta_set[i % len(cta_set)]

    log("    · Inside CMI projects")
    caps_industry, projects = caps_projects(profile)

    social = _social(profile, include_logos, include_quotes)
    for n in social["notes"]:
        log(f"    · social proof: {n}")

    kicker = ("Market Signal Brief" + (f" · {market}" if market != "Global" else "")
              if kind == "signal" else "Market Report Brief")
    source = ("CMI's market and competitive intelligence coverage" if kind == "signal"
              else f"CMI's {market_topic.lower()} and adjacent coverage" if market_topic
              else "CMI's market coverage")

    return {
        "company": profile.company,
        "recipient": profile.recipient,
        "brand": {"kicker": kicker,
                  "issue_date": now.strftime("%d %b %Y"),
                  "date_long": now.strftime("%A, %B %d, %Y"),
                  "prov_date": now.strftime("%b %d, %Y"),
                  "tagstrip_source": source,
                  "demo_url": DEMO_URL,
                  "unsub_url": UNSUB_URL},
        "lede": lede,
        "top3": top3,
        "sections": sections,
        "caps": {"industry": caps_industry, "projects": projects},
        "social": social,
    }
