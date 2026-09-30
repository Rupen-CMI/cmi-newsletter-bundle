"""LLM-backed role checks, using the fine-tune itself as the judge.

`validate_brief.py`'s existing checks are keyword rules: they catch a competitor that is
on the curated list, a regulator by name, a customer whose section text lacks any
buyer-vertical word. They cannot catch what needs judgement -- that Reckitt is VVF's
CUSTOMER rather than its rival, or that a "Cable Glands Market" brief is really about a
restaurant chain. Those are questions about meaning, and the model that writes the
briefs can also be asked to grade them.

Two checks, both deliberately one call:

  competitor_verdict  is this company a genuine rival in THIS market?
  customer_verdict    is this company a plausible BUYER of what the seller sells?

Both fail OPEN. A judge that is unavailable, slow, or returns nonsense must never block
a send -- validate_brief's deterministic rules stay the floor, and this only adds signal
on top. That matters because the judge is the same 7B whose factual recall we measured
as unreliable: it is good at reading a passage in front of it, weak at recalling facts
about companies it has never seen. Ask it only about text it can see.
"""
from __future__ import annotations

import json
import os
import re

_HTML = re.compile(r"<[^>]+>")


def _enabled() -> bool:
    """Off unless asked for. The checks cost a GPU call each, and a batch of 20 briefs
    would add 40 -- worth it before a send, wasteful during iteration."""
    return os.environ.get("CMI_LLM_VALIDATE", "").strip() == "1"


def _strict() -> bool:
    """Advisory by default: LLM findings are reported and counted, never blocking.

    Measured over four validation settings on the same 20 leads, the accuracy is
    identical in all of them -- the same briefs carry the same bad picks. What changes
    is only what you choose to BLOCK on, which is a shipping decision:

        rules only      15/20 pass   the rules cannot see a bad competitor at all
        strict           5/20 pass   any LLM role call fails the brief
        advisory         8/20 pass   role calls warn; only structure blocks  <- default
        agree-only       4/20 pass   two methods must reject

    Advisory is the default because a role call is a judgement a human can overrule,
    and because nothing is hidden: every finding still prints. Set CMI_LLM_STRICT=1 for
    a curated send where each brief must be defensible without review.
    """
    return os.environ.get("CMI_LLM_STRICT", "0").strip() == "1"


def _agree_only() -> bool:
    """CMI_LLM_AGREE=1 requires BOTH methods to reject before failing a brief.

    Method 1 reads the signal passage and method 3 reads the company's business; they
    fail in different directions, so demanding agreement raises the evidence bar. The
    cost is a second call per competitor, and the risk is that method 1's habitual
    "unknown" silences method 3 -- which is why 'unknown' is not counted as agreement
    either way.
    """
    return os.environ.get("CMI_LLM_AGREE", "").strip() == "1"


def _ask(system: str, user: str, default: dict) -> dict:
    try:
        import llm

        return llm.chat_json(system, user, max_tokens=300, default=default) or default
    except Exception:  # noqa: BLE001 -- a judge that breaks must not fail the brief
        return default


def _passage(text: str, label: str, span: int = 900) -> str:
    m = re.search(rf"{re.escape(label)}\s*·(.{{0,{span}}})", text, re.S)
    return _HTML.sub(" ", m.group(1)) if m else ""


def competitor_verdict(seller: str, seller_sells: str, market: str,
                       rival: str, passage: str) -> dict:
    """Is `rival` a genuine competitor to `seller` in `market`?

    Returns {"verdict": real|customer|adjacent|unrelated|unknown, "why": "..."}.
    'customer' is called out separately because that is the failure actually seen:
    the crawl put P&G and Reckitt in VVF's competitor slot when VVF manufactures
    their soap for them.
    """
    if not _enabled() or not rival:
        return {"verdict": "unknown", "why": "llm check disabled"}
    system = (
        "You judge whether one company competes with another. Answer only from the "
        "SIGNAL passage and the seller description given -- do not rely on outside "
        "knowledge, and say 'unknown' when the passage does not settle it.\n"
        "  real      = sells a directly substitutable product; a buyer picks one OR the other\n"
        "  customer  = BUYS FROM the seller, or the seller supplies/manufactures for them\n"
        "  adjacent  = same broad sector, different product; not a substitute\n"
        "  unrelated = different business entirely\n"
        "A contract manufacturer's clients are 'customer', never 'real', however large."
    )
    user = (f"SELLER: {seller} — {seller_sells}\n"
            f"MARKET IN QUESTION: {market}\n"
            f"COMPANY BEING CALLED A COMPETITOR: {rival}\n\n"
            f"SIGNAL PASSAGE:\n{passage[:900]}\n\n"
            'Return JSON {"verdict":"real|customer|adjacent|unrelated|unknown",'
            '"why":"one short sentence"}')
    out = _ask(system, user, {"verdict": "unknown", "why": "no response"})
    v = str(out.get("verdict", "unknown")).strip().lower()
    if v not in {"real", "customer", "adjacent", "unrelated", "unknown"}:
        v = "unknown"
    return {"verdict": v, "why": str(out.get("why", ""))[:160]}


def customer_verdict(seller: str, seller_sells: str, buyer: str, passage: str) -> dict:
    """Would `buyer` plausibly purchase what `seller` sells?

    Returns {"verdict": buyer|rival|supplier|unrelated|unknown, "why": "..."}.
    """
    if not _enabled() or not buyer:
        return {"verdict": "unknown", "why": "llm check disabled"}
    system = (
        "You judge whether a company named in a sales brief is a plausible BUYER of "
        "what the seller sells. Answer only from the passage and seller description.\n"
        "  buyer     = its own business would place a purchase order for this product\n"
        "  rival     = it sells the same thing\n"
        "  supplier  = it sells INTO the seller, not the other way round\n"
        "  unrelated = a real company in a trade with no use for this product\n"
        "Be strict. A large famous company in an unrelated trade is 'unrelated', not "
        "'buyer' -- a watchmaker does not buy oleochemicals."
    )
    user = (f"SELLER: {seller} — {seller_sells}\n"
            f"COMPANY IN THE POTENTIAL CUSTOMER SLOT: {buyer}\n\n"
            f"SIGNAL PASSAGE:\n{passage[:900]}\n\n"
            'Return JSON {"verdict":"buyer|rival|supplier|unrelated|unknown",'
            '"why":"one short sentence"}')
    out = _ask(system, user, {"verdict": "unknown", "why": "no response"})
    v = str(out.get("verdict", "unknown")).strip().lower()
    if v not in {"buyer", "rival", "supplier", "unrelated", "unknown"}:
        v = "unknown"
    return {"verdict": v, "why": str(out.get("why", ""))[:160]}


def competitor_verdict_by_business(seller: str, seller_sells: str, market: str,
                                   rival: str, passage: str) -> dict:
    """Second method: judge on what the two companies DO, not on the signal passage.

    `competitor_verdict` deliberately refuses outside knowledge, which makes it safe but
    blind: a TSMC chip-packaging headline never states whether TSMC competes in wood
    veneers, so it answers "unknown" and the bad pick ships. This variant asks the model
    to first say what the rival's business IS, then compare. That is factual recall,
    which this model is measurably weak at -- so `judge_brief` treats the two methods as
    a pair and only fails a brief when they agree.
    """
    if not _enabled() or not rival:
        return {"verdict": "unknown", "why": "llm check disabled"}
    system = (
        "You decide whether two companies compete. Work in two steps and show both.\n"
        "  1. state in a few words what the NAMED COMPANY's core business is\n"
        "  2. compare that to what the SELLER sells\n"
        "Verdicts:\n"
        "  real      = the two sell substitutable products; a buyer picks one OR the other\n"
        "  customer  = the named company would BUY from the seller\n"
        "  adjacent  = same broad sector, different product\n"
        "  unrelated = different industries entirely\n"
        "  unknown   = you genuinely do not know what the named company does\n"
        "Say 'unknown' rather than guessing at a company you have not heard of."
    )
    user = (
        "SELLER: " + seller + " — " + seller_sells + "\n"
        "MARKET IN QUESTION: " + market + "\n"
        "NAMED COMPANY: " + rival + "\n"
        "(context, may not describe their business: " + passage[:300] + ")\n\n"
        'Return JSON {"their_business":"...","verdict":"real|customer|adjacent|'
        'unrelated|unknown","why":"one short sentence"}'
    )
    out = _ask(system, user, {"verdict": "unknown", "why": "no response"})
    v = str(out.get("verdict", "unknown")).strip().lower()
    if v not in {"real", "customer", "adjacent", "unrelated", "unknown"}:
        v = "unknown"
    return {"verdict": v, "why": str(out.get("why", ""))[:160],
            "their_business": str(out.get("their_business", ""))[:120]}


def _describe_business(company: str) -> str:
    """What does this company do, in a few words? '' when the model does not know."""
    out = _ask("State what a company's core business is in under 15 words. "
               "Say 'unknown' if you are not sure.",
               "COMPANY: " + company + '\nReturn JSON {"business":"..."}',
               {"business": ""})
    b = str(out.get("business", "")).strip()
    return "" if b.lower().startswith("unknown") else b[:120]


def competitor_verdict_binary(seller_sells: str, rival: str, rival_business: str = "") -> dict:
    """Third method, and the one actually used: describe, then ask ONE yes/no question.

    Measured against the first two on five known cases (three real failures from live
    briefs plus two controls):

        method 1  judge from the signal passage only        1/5
        method 2  describe the business, then 5-way label   0/5
        method 3  describe the business, then yes/no        4/5

    The interesting part is WHY. Method 2's `their_business` answers were correct every
    time -- it knew TSMC makes semiconductors and Infosys sells IT services -- and then
    still returned "unknown" or "real" when asked to pick from five labels. The recall
    was never the weak link; the classification was. Collapsing the question to "could a
    buyer get the same thing here instead?" removes that step.

    The one miss is instructive too: it calls Galaxy Surfactants "no" for VVF, because
    surfactants and bar soap are not literally the same product even though the two
    firms compete for the same contract-manufacturing work. So this method is reliable
    at rejecting nonsense and conservative about confirming real rivals -- which is the
    right way round, since `judge_brief` only ever raises a warning on "no".
    """
    if not _enabled() or not rival:
        return {"verdict": "unknown", "why": "llm check disabled"}
    biz = rival_business or _describe_business(rival)
    if not biz:
        return {"verdict": "unknown", "why": "model does not know this company",
                "their_business": ""}
    out = _ask(
        "Answer one yes/no question. A buyer needs the SELLER'S PRODUCT. Could they buy "
        "from the OTHER COMPANY instead and get the same thing? Answer no if the other "
        "company is in a different industry.",
        "SELLER'S PRODUCT: " + seller_sells + "\n"
        "OTHER COMPANY: " + rival + " — " + biz + "\n"
        'Return JSON {"same_thing":"yes|no","why":"few words"}',
        {"same_thing": "", "why": ""})
    ans = str(out.get("same_thing", "")).strip().lower()
    verdict = {"yes": "real", "no": "unrelated"}.get(ans, "unknown")
    return {"verdict": verdict, "why": str(out.get("why", ""))[:160],
            "their_business": biz}


def judge_brief(text: str, profile, *, market: str = "") -> tuple[list[str], list[str]]:
    """Run both checks over a brief's visible text. Returns (fails, warns).

    Only a confident wrong answer becomes a FAIL: 'unknown' is silent, and 'adjacent'
    is a warning because a near-neighbour is a judgement call a human should make.
    """
    fails: list[str] = []
    warns: list[str] = []
    if not _enabled():
        return fails, warns

    # CMI_LLM_STRICT=0 routes every finding to warns instead: still reported, still
    # counted, but only a structural problem blocks the send.
    def sink(msg: str) -> None:
        (fails if _strict() else warns).append(msg)

    import validate_brief as vb

    sells = getattr(profile, "value_prop", "") or ""
    seller = getattr(profile, "company", "") or ""

    comp = vb.section_company(text, "Competition Outlook")
    if comp:
        # Method 1 reads the signal passage. It is the only one that can spot a
        # CUSTOMER wrongly filed as a rival, because that relationship is stated in
        # the text ("VVF manufactures for Reckitt").
        v = competitor_verdict(seller, sells, market or "the seller's market",
                               comp, _passage(text, "Competition Outlook"))
        if v["verdict"] == "customer":
            sink(f"LLM ROLE ERROR: Competition Outlook features '{comp}', which "
                 f"reads as a CUSTOMER of {seller}. {v['why']}")
        elif v["verdict"] == "unrelated":
            # In agree-only mode a single method is not enough to reject.
            second = competitor_verdict_binary(sells, comp) if _agree_only() else None
            if second is None or second["verdict"] == "unrelated":
                sink(f"LLM ROLE ERROR: '{comp}' is not in {seller}'s business at all. "
                     f"{v['why']}")
            else:
                warns.append(f"LLM: methods disagree on '{comp}' -- passage says "
                             f"unrelated, business check says {second['verdict']}.")
        elif v["verdict"] == "adjacent":
            warns.append(f"LLM: '{comp}' is adjacent, not a direct substitute. {v['why']}")
        elif v["verdict"] == "unknown" and sells:
            # The passage did not settle it -- which is the common case, and how TSMC
            # shipped as a wood-veneer rival. Fall back to the binary check, which is
            # reliable at rejecting an out-of-industry name (4/5 against method 1's 1/5).
            b = competitor_verdict_binary(sells, comp)
            if b["verdict"] == "unrelated":
                sink("LLM ROLE ERROR: '" + comp + "' (" + b.get("their_business", "") +
                     ") sells nothing a buyer of " + seller + "'s product would "
                     "substitute. " + b["why"])

    cust = vb.section_company(text, "Potential Customer")
    if cust:
        v = customer_verdict(seller, sells, cust, _passage(text, "Potential Customer"))
        if v["verdict"] in {"rival", "unrelated"}:
            sink(f"LLM ROLE ERROR: Potential Customer '{cust}' reads as "
                 f"{v['verdict']}, not a buyer. {v['why']}")
        elif v["verdict"] == "supplier":
            warns.append(f"LLM: '{cust}' looks like a supplier to {seller}. {v['why']}")

    return fails, warns
