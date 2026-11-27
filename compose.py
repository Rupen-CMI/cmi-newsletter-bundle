"""Turn pipeline output into the template context for either brief type.

Two brief types, one template (`templates/cmi_brief.html`):

  signal  -- Potential Customer / Industry Outlook / Competition Outlook  (3 sections)
  report  -- Your Market                          / Competition Outlook  (2 sections)

Section 1 differs in content (signal profiles a prospect, report profiles the market
the reader asked about) and report briefs drop the middle section (Market Moves)
entirely rather than filling it -- see the section-2 comment below. CTA copy/URLs
and the footer CTA also differ by kind; see voice.ctas/voice.footer_cta.
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
# 2026-10-19: moved off coherentmarketinsights.com to the new cmiinsights.info site,
# per the sample briefs in session_scratch/final_4_picks/fortinet_*.html. Each CTA
# appends its own path (see voice.ctas/voice.footer_cta) -- this is the site root.
SITE_ROOT = "https://www.cmiinsights.info"
DEMO_URL = f"{SITE_ROOT}/speakwithanalyst"

# 2026-10-19: matches the dedicated unsubscribe endpoint used in the same sample
# briefs, replacing the old contact-us page fallback.
UNSUB_URL = os.getenv("CMI_UNSUB_URL", "https://unsubs-kohl.vercel.app/")


def _clean(s: str) -> str:
    """Strip the things the voice rules ban but a model still sometimes emits."""
    # The model occasionally returns a list where a string field was asked for
    # (e.g. "v": ["a", "b"] instead of "v": "a, b") -- str() first so a truthy
    # non-string doesn't crash on the .strip() below (AttributeError: 'list'
    # object has no attribute 'strip').
    if not isinstance(s, str):
        s = str(s) if s else ""
    s = s.strip()
    s = re.sub(r"<(?!/?[bi]\b)[^>]*>", "", s)          # only <b>/<i> survive
    s = re.sub(r"\s*[—–]\s*", ", ", s)                  # em dash -> comma
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"\s+([.,;:])", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s)
    # The model sometimes emits a stray trailing </b> with no matching opener --
    # seen at scale in Campaign 5 analyst bullets (2026-10-27): it bolds the lead
    # clause correctly, then closes a SECOND, nonexistent <b> at the very end of
    # the plain-sentence tail. HTML renders a lone </b> harmlessly, but it is
    # still wrong/sloppy, so balance here rather than leave it in the output.
    opens, closes = s.count("<b>"), s.count("</b>")
    if closes > opens:
        for _ in range(closes - opens):
            idx = s.rfind("</b>")
            s = s[:idx] + s[idx + 4:]
    elif opens > closes:
        s += "</b>" * (opens - closes)
    return s


def _social(profile, include_logos: bool, include_quotes: bool) -> dict:
    """Vertical-matched social proof. Empty when the vertical is not established."""
    return social_proof.for_profile(profile, llm, want_logos=include_logos,
                                    want_quotes=include_quotes)


def prospect_heading(profile) -> str:
    """Section-1 heading by lead type: the one label that varies with who the lead is."""
    lt = (getattr(profile, "lead_type", "") or "seller").lower()
    if lt == "investor":
        return "Potential Deal"
    if lt in ("nonprofit", "education", "government"):
        return "Potential Partnership"
    return "Potential Customer"


# ── LLM helpers ───────────────────────────────────────────────────────────────
def _ctx(profile) -> str:
    sc = voice.seller_context(profile)
    return sc + "\n" if sc else ""


def analyst_bullets(profile, signal: str, extra: str = "") -> list[str]:
    system = voice.bullet_system().format(company=profile.company, short=profile.short)
    user = (f"SELLER: {profile.company} — {profile.value_prop}\n"
            f"MARKET: {getattr(profile, 'market', 'Global')}\n"
            f"{_ctx(profile)}"
            f"SIGNAL:\n{signal}\n{extra}\n\n"
            'Return JSON {"bullets":["<b>judgement.</b> support.", "...", "..."]} with exactly 3 bullets.')
    data = voice.chat_json_guarded(system, user, max_tokens=900, profile=profile) or {}
    out = [_clean(b) for b in (data.get("bullets") or []) if str(b).strip()]
    return out[:3]


def analyst_bullets_multi(profile, signals: list[str]) -> list[list[str]]:
    """Same output as calling analyst_bullets once per signal, but as one LLM call --
    each section's 3 bullets only ever depend on that section's own text, never on
    any other section, so a signal brief with 3 sections made 3 separate round-trips
    for what is really 3 independent instances of the same small task. Combined
    2026-10-14 to cut N-1 calls per brief (N = number of sections). Falls back to the
    one-call-per-signal path entirely if the combined response comes back malformed,
    rather than guessing which signals got bullets and which did not."""
    if not signals:
        return []
    if len(signals) == 1:
        return [analyst_bullets(profile, signals[0])]
    system = voice.bullet_system().format(company=profile.company, short=profile.short)
    blocks = "\n\n".join(f"[{i}] SIGNAL:\n{s}" for i, s in enumerate(signals))
    user = (f"SELLER: {profile.company} — {profile.value_prop}\n"
            f"MARKET: {getattr(profile, 'market', 'Global')}\n"
            f"{_ctx(profile)}\n"
            f"{blocks}\n\n"
            'Return JSON {"items":[{"index":0,"bullets":["<b>judgement.</b> support.", "...", "..."]}]} '
            "with one entry per SIGNAL, each with exactly 3 bullets. Each SIGNAL's bullets use only "
            "that SIGNAL.")
    data = voice.chat_json_guarded(system, user, max_tokens=900 * len(signals), profile=profile) or {}
    items = data.get("items") if isinstance(data, dict) else None
    out: list[list[str]] = [[] for _ in signals]
    seen = set()
    if items:
        for it in items:
            if not isinstance(it, dict):
                continue  # model occasionally returns a bare string instead of {"index":...}
            i = it.get("index")
            if isinstance(i, int) and 0 <= i < len(signals):
                bullets = [_clean(b) for b in (it.get("bullets") or []) if str(b).strip()]
                out[i] = bullets[:3]
                seen.add(i)
    # A combined call that only partially answered (some indices missing entirely) is
    # not trusted for the gaps -- fall back to the single-signal call for exactly the
    # sections it dropped, rather than shipping a brief with some sections missing
    # their analyst opinion entirely.
    for i, s in enumerate(signals):
        if i not in seen or not out[i]:
            out[i] = analyst_bullets(profile, s)
    return out


# 2026-10-19: cut from 6 engagement cards to 4, matching the cmiinsights.info redesign
# sample (final_4_picks/fortinet_*.html both show exactly Project 01-04). Dropped
# "competitor and pipeline" and "market entry" rather than any of the first four.
_DIMENSIONS = ("supplier and demand", "capacity and input supply", "pricing and access",
               "customer readiness and adoption")


def caps_projects(profile, *, kind: str = "signal", market_topic: str = "") -> tuple[str, list[dict]]:
    """Four real-looking past CMI engagements for the Inside CMI panel.

    `subject` is the market these four projects must actually be about: for a report
    brief, that is `market_topic` (the exact market the recipient asked about) --
    not the seller's own product space, which is irrelevant to a report brief and
    produced engagements about the wrong market entirely. For a signal brief, it is
    the seller's own sub-market (profile.industries / curated override), as before.
    Each of the four must cover a DIFFERENT sub-market or tightly-related market
    within that subject, not four rephrasings of the same umbrella term -- confirmed
    as a real quality problem 2026-10-12: cards all titled "X industry outlook,
    <dimension> landscape" read as filler, not distinct project case studies.
    """
    import tenant as _tenant
    _ov = _tenant._overrides_for(getattr(profile, "company", ""))
    subject = (market_topic if kind == "report" else
               (_ov.get("caps_industry") or "").strip() or ", ".join(profile.industries[:2]))

    system = (
        "You describe four real-looking past engagements Coherent Market Insights has "
        "delivered, to show the reader CMI's depth of experience in this exact space. "
        "These read like an actual project list on a case-study slide -- a scope and a "
        "deliverable -- not a marketing tagline and not four rephrasings of the same "
        "generic title.\n\n"
        "Each of the four projects must be about a DIFFERENT sub-market or tightly "
        "related market within the subject given, named explicitly in the title (e.g. "
        "for 'cybersecurity': network firewalls, SASE, OT/industrial security, "
        "zero-trust -- four different named segments, not 'cybersecurity' four "
        "times). Titles are short, concrete, and sound like a real study: 'Next-gen "
        "firewall capacity assessment, APAC region', 'SASE pricing benchmark across "
        "mid-market buyers'. Cover these four angles across the four projects: "
        "supplier and demand, manufacturing or capacity, pricing and access, and "
        "customer readiness -- but let each title's wording vary naturally; do not "
        "force every title into one repeated sentence pattern.\n\n"
        "NEVER name the seller or its products in a title or body. These are "
        "engagements delivered for OTHER, unnamed clients in the same space, and "
        "writing the seller's own product name into one turns it into a fabricated "
        "case study. Say 'the client', or name the capability. Concrete industry "
        "nouns (apheresis, viral vector, reimbursement precedent, cost per batch, "
        "zero-trust, SASE) are wanted; invented client facts are not.\n\n" + voice.active_style()
    )
    user = (f"SUBJECT (every project must be about THIS market and its sub-markets): {subject}\n"
            f"SELLER (context only, do not name in output): {profile.company} — {profile.value_prop}\n\n"
            'Return JSON {"industry":"<short phrase for the intro line, naming the subject>",'
            '"projects":[{"title":"...","body_html":"2 sentences, <b> allowed"}]} '
            "with exactly 4 projects, each about a different named sub-market of the "
            "subject above, covering supplier and demand, manufacturing or capacity, "
            "pricing and access, and customer readiness across the four.\n"
            f"Do not write the words '{profile.company}' anywhere in the output.")
    data = llm.chat_json(system, user, max_tokens=2000, default={}) or {}
    projects = [{"title": _clean(p.get("title", "")), "body_html": _clean(p.get("body_html", ""))}
                for p in (data.get("projects") or [])][:4]
    industry = _clean(data.get("industry", "")) or subject

    # Repair what the model got wrong rather than shipping it: strip the seller's own
    # name out of a title/body if it slipped through (a smaller model sometimes still
    # writes a fabricated case study naming the seller directly).
    _seller_bits = [b for b in re.split(r"[\s,]+", profile.company) if len(b) > 3]

    for i, p in enumerate(projects):
        for bit in _seller_bits:                    # never let the seller's own name through
            p["title"] = re.sub(rf"\b{re.escape(bit)}\b", "the client", p["title"], flags=re.I)
            p["body_html"] = re.sub(rf"\b{re.escape(bit)}\b", "the client", p["body_html"], flags=re.I)
        if not p["title"]:
            p["title"] = f"{industry} {_DIMENSIONS[i % len(_DIMENSIONS)]} study"
        # "2 sentences" in the prompt above is an instruction, not a guarantee -- the
        # model has been seen writing 3-4, which on the fixed-size capcard grid just
        # means more wrapped lines, not a bigger card -- confirmed 2026-10-22 against
        # the Fortinet sample (the cmiinsights.info redesign's actual reference,
        # 2 sentences per card) vs. a live Waters brief running noticeably longer.
        # Keep only the first 2 sentences deterministically rather than trust the
        # instruction alone, matching every other prompt-following gap this file
        # already hard-guards (seller-name leak above, the lede opener, etc.).
        sentences = re.findall(r"[^.!?]+[.!?]+(?:</b>)?|[^.!?]+$", p["body_html"])
        if len(sentences) > 2:
            p["body_html"] = "".join(sentences[:2]).strip()
    while len(projects) < 4:                       # never render an empty card
        projects.append(projects[len(projects) % max(1, len(projects))] if projects
                        else {"title": f"{industry} supplier and demand study",
                              "body_html": "Supplier base, demand and spending patterns mapped "
                                           "across the client's priority countries."})
    return industry, projects


_LEDE_COUNT_WORD = {1: "one", 2: "two", 3: "three", 4: "four"}


_GENERIC_LABEL = {"Potential Customer": "a potential customer",
                  "Potential Deal": "a potential deal",
                  "Potential Partnership": "a potential partnership",
                  "Industry Outlook": "an industry shift",
                  "Competition Outlook": "a competitor move"}

# Demonyms the lead's own country may bring into a theme phrase ("an Indian ...").
_DEMONYM = {"india": "Indian", "united states": "American", "usa": "American",
            "united kingdom": "British", "uk": "British", "china": "Chinese",
            "japan": "Japanese", "germany": "German", "france": "French",
            "australia": "Australian", "canada": "Canadian", "brazil": "Brazilian",
            "singapore": "Singaporean", "south korea": "Korean", "uae": "Emirati",
            "saudi arabia": "Saudi", "italy": "Italian", "spain": "Spanish"}


def _theme_ok(theme, market: str) -> str:
    """The theme phrase if it is safe to put in the lede, else "".

    A theme is 2 to 6 plain lowercase words: no digits or currency, no punctuation,
    and no capitalised word other than the lead's country and its demonym, so a
    company, product or person name can never get through."""
    if not isinstance(theme, str):
        return ""
    t = re.sub(r"\s+", " ", theme).strip()
    words = t.split(" ")
    if not t or not 2 <= len(words) <= 6:
        return ""
    if re.search(r"[\d$€£¥₹%<>&.,;:!?'’\"()/]", t):
        return ""
    allowed = set()
    mk = (market or "").strip()
    if mk and mk != "Global":
        allowed = set(mk.split())
        if mk.lower() in _DEMONYM:
            allowed.add(_DEMONYM[mk.lower()])
    for w in words:
        if w[:1].isupper() and w not in allowed:
            return ""
    return t


def _build_lede(profile, sections: list[dict], themes: list, opening: str) -> str:
    """The lede, assembled in code: fixed opener, one bolded label per section, an
    optional vetted theme phrase after it, and the country once at the end."""
    market = (getattr(profile, "market", "") or "").strip()
    titles = " ".join(s.get("title", "") for s in sections).lower()
    title_words = {w for w in re.findall(r"[a-z]+", titles) if len(w) > 3}
    parts = []
    used_market = False
    for i, s in enumerate(sections):
        art, noun = _GENERIC_LABEL.get(s["label"], "a market development").split(" ", 1)
        part = f"{art} <b>{noun}</b>"
        theme = _theme_ok(themes[i] if i < len(themes) else "", market)
        if theme and not title_words.isdisjoint(re.findall(r"[a-z]+", theme.lower())):
            theme = ""          # a word lifted from the section's own title
        if theme:
            part += f" {theme}"
            used_market = used_market or bool(market and market in theme)
        parts.append(part)
    if len(parts) > 1:
        parts = parts[:-1] + ["and " + parts[-1]]
    joined = (", " if len(parts) > 2 else " ").join(parts)
    where = f" in {market}" if market and market != "Global" and not used_market else ""
    return f"{opening} {joined}{where}.".replace("  ", " ")


def editorial(profile, sections: list[dict]) -> tuple[str, list[dict]]:
    """The lede and the three top-3 lines, written after the sections exist.

    The lede is built in code (see _build_lede); the model only supplies an optional
    short theme phrase per section and the top-3 lines."""
    system = ("You write the opening of a market brief for one named client.\n\n"
              + voice.active_style())
    body = "\n".join(f"[{s['label']}] {s['title']}: {re.sub(r'<[^>]+>', '', s['body_html'])[:280]}"
                     for s in sections)
    n = len(sections)
    count_word = _LEDE_COUNT_WORD.get(n, str(n))
    user = (f"CLIENT: {profile.company} ({profile.short}) — {profile.value_prop}\n"
            f"MARKET: {getattr(profile, 'market', 'Global')}\n"
            f"VERTICALS TRACKED: {', '.join(profile.industries)}\n"
            f"SECTIONS:\n{body}\n\n"
            'Return JSON {"themes":["...","..."],"top3":["...","...","..."]}, one entry per section '
            "in the order of SECTIONS.\n"
            "- themes: 2 to 6 lowercase words saying what the section is about, in general terms. "
            "No names and no numbers. Use \"\" if unsure.\n"
            "- top3: a short claim in <b></b>, then one plain sentence on why it matters.")
    opening = f"This week's brief covers {count_word} development{'s' if n != 1 else ''}:"

    data = voice.chat_json_guarded(system, user, max_tokens=1000, profile=profile) or {}
    themes = data.get("themes") if isinstance(data, dict) else None
    lede = _build_lede(profile, sections, themes if isinstance(themes, list) else [], opening)
    lines = [_clean(x) for x in ((data.get("top3") if isinstance(data, dict) else None) or [])][:3]
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
    cells = [{**c, "v": _clean(c.get("v", ""))} for c in cells]
    cells = [c for c in cells if c["v"]][:2]
    return {"label": label, "kind": "prospect",
            "flag": flag or _clean(card.get("flag", "")) or label,
            "title": _clean(title), "body_html": body, "cells": cells,
            # The template links the citation when a URL survives this far; an empty
            # string renders as plain text rather than a dead anchor.
            "source_url": _clean(card.get("source_url", "")),
            "source_name": _clean(card.get("source_name") or "")}


def _story(label, title, body_html, *, source_url="", source_name="") -> dict:
    return {"label": label, "kind": "story", "title": _clean(title),
            "body_html": _clean(body_html), "flag": "", "cells": [],
            "source_url": _clean(source_url),
            "source_name": _clean(source_name or "")}


def build_context(profile, brief: dict, kind: str, *, market_topic: str = "",
                  include_logos=True, include_quotes=True, log=print) -> dict:
    """`brief` is whatever brief_builder.build_brief produced."""
    now = datetime.now(timezone.utc)
    market = getattr(profile, "market", "Global")
    sections: list[dict] = []
    srcs: list[str] = []   # source article text per section, same order as sections

    # brief_builder's own key names -- "customer"/"competition" are its debug dict,
    # the rendered cards live under these.
    cust = brief.get("potential_customer") or {}
    comp = brief.get("competition_outlook") or {}
    news = (brief.get("industry_outlook") or [{}])[0] or {}

    # -- section 1 -------------------------------------------------------------
    # Only build "Your Market" when there is real content to put under the forced
    # headline. news can be {} when brief_builder.synthesize_news() found no usable
    # signal (empty industry pool) -- building the section anyway rendered a real
    # title with a completely blank body and stat grid, invisible to every existing
    # check (confirmed in 160/922 briefs in the 2026-10-08 regen, see EMPTY_SECTION
    # in validate_brief.py). Same honesty rule STRUCTURE already applies to
    # Competition Outlook/Industry Outlook: a section with nothing to say is
    # dropped, not faked -- STRUCTURE's missing-section warn/fail is the correct,
    # existing mechanism to flag a brief that is now too thin.
    if kind == "report" and market_topic and news.get("body_html"):
        log("    · section 1: your market")
        # 2026-10-22: the cmiinsights.info redesign's section-1 pill/flag reads the
        # bare topic name ("Wireless Access Point Market"), not a generic "Your
        # Market"/"Your Market · topic" wrapper -- confirmed against the Fortinet
        # industry-brief sample, which has neither string anywhere. "Your Market"
        # is kept ONLY as validate_brief.py's SECTION_ALIASES lookup key (the
        # STRUCTURE check's own section-1 identity), never rendered into the HTML.
        sections.append(_prospect(
            market_topic,
            {"body_html": news.get("body_html", ""),
             "source_url": news.get("source_url", ""),
             "source_name": news.get("source_name", ""),
             "cells": [{"h": "What is changing", "v": _clean(news.get("headline", ""))}]},
            ("What is changing", "Why it matters for you"),
            flag=market_topic, title=market_topic))
        srcs.append(news.get("_source_text", ""))
    elif kind == "report" and cust:
        # Fall back to a real Potential Customer card rather than drop the section
        # outright, if generation happened to find one.
        log("    · section 1: your market (fallback to potential customer)")
        sections.append(_prospect(prospect_heading(profile), cust,
                                  ("Why now", "Who to target")))
        srcs.append(cust.get("_source_text", ""))
    elif cust:
        log("    · section 1: potential customer")
        sections.append(_prospect(prospect_heading(profile), cust,
                                  ("Why now", "Who to target")))
        srcs.append(cust.get("_source_text", ""))

    # -- section 2 -------------------------------------------------------------
    # 2026-10-19: report briefs are capped at 2 sections (Your Market + Competition
    # Outlook), per the cmiinsights.info redesign sample -- Market Moves is dropped
    # entirely rather than filling a middle slot, so this whole section only ever
    # renders for a signal brief now.
    if kind != "report" and news:
        log("    · section 2: industry outlook")
        sections.append(_story("Industry Outlook", news.get("headline", ""),
                               news.get("body_html", ""),
                               source_url=news.get("source_url", ""),
                               source_name=news.get("source_name", "")))
        srcs.append(news.get("_source_text", ""))

    # -- section 3 -------------------------------------------------------------
    if comp:
        log("    · section 3: competition outlook")
        sections.append(_prospect("Competition Outlook", comp,
                                  ("Why it matters", "Their edge")))
        srcs.append(comp.get("_source_text", ""))

    # -- analyst opinions ------------------------------------------------------
    # One call for every section's bullets instead of one call per section -- see
    # analyst_bullets_multi's docstring. Falls back internally to the old
    # one-call-per-signal path for any section the combined call fails to answer.
    log(f"    · analyst opinions: {len(sections)} section(s) in one call")
    signals = []
    for s, src in zip(sections, srcs):
        prose = f"{s['title']}\n{re.sub(r'<[^>]+>', '', s['body_html'])}"
        # the article lets the analyst check units and figures against the source, not just the rewrite
        signals.append(f"ARTICLE: {src}\nSECTION TEXT: {prose}" if src else prose)
    all_bullets = analyst_bullets_multi(profile, signals)
    for s, bullets in zip(sections, all_bullets):
        s["bullets"] = bullets

    log("    · editorial (lede + top 3)")
    lede, top3 = editorial(profile, sections)

    cta_set = voice.ctas(kind, SITE_ROOT)
    topic_for_cta = market_topic or profile.company
    for i, s in enumerate(sections):
        cta = dict(cta_set[i % len(cta_set)])
        cta["q"] = cta["q"].format(topic=topic_for_cta, topic_lower=topic_for_cta.lower())
        s["cta"] = cta

    log("    · Inside CMI projects")
    caps_industry, projects = caps_projects(profile, kind=kind, market_topic=market_topic)

    social = _social(profile, include_logos, include_quotes)
    for n in social["notes"]:
        log(f"    · social proof: {n}")

    kicker = ("Market Signal Brief" + (f" · {market}" if market != "Global" else "")
              if kind == "signal" else "Market Report Brief")
    source = ("CMI's market and competitive intelligence coverage" if kind == "signal"
              else f"CMI's {market_topic.lower()} and adjacent coverage" if market_topic
              else "CMI's market coverage")

    foot_cta = voice.footer_cta(kind, SITE_ROOT)

    return {
        "company": profile.company,
        "recipient": profile.recipient,
        "brand": {"kicker": kicker,
                  "issue_date": now.strftime("%d %b %Y"),
                  "date_long": now.strftime("%A, %B %d, %Y"),
                  "prov_date": now.strftime("%b %d, %Y"),
                  "tagstrip_source": source,
                  "demo_url": foot_cta["url"],
                  "footer_cta_label": foot_cta["label"],
                  "footer_ctaq": foot_cta["ctaq"],
                  "unsub_url": UNSUB_URL},
        "lede": lede,
        "top3": top3,
        "sections": sections,
        "caps": {"industry": caps_industry, "projects": projects},
        "social": social,
    }
