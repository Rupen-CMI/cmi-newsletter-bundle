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


class JudgeUnavailable(Exception):
    """Raised by _ask_checked when the judge call itself errored (dead endpoint, bad
    response, etc). Distinct from the model genuinely answering 'unknown' -- callers
    that need to tell "the judge never ran" apart from "the judge ran and could not
    tell" should use _ask_checked instead of _ask. Added 2026-10-08: a dead GPU
    endpoint during the 2026-10-07 regen run made topic_alignment_verdict return
    {"verdict": "unknown", "why": "no response"} on every one of 13+ Cloudflare
    520/524 errors, indistinguishable in storage from a real 'aligned' pass -- so a
    brief that was never actually checked looked identical to one that passed."""


def _ask(system: str, user: str, default: dict) -> dict:
    try:
        import llm

        return llm.chat_json(system, user, max_tokens=300, default=default) or default
    except Exception:  # noqa: BLE001 -- a judge that breaks must not fail the brief
        return default


def _ask_checked(system: str, user: str, default: dict) -> dict:
    """Like _ask, but raises JudgeUnavailable instead of silently returning `default`
    on a broken call, so the caller can report "judge did not run" distinctly from
    a genuine low-confidence verdict."""
    try:
        import llm

        out = llm.chat_json(system, user, max_tokens=300, default=None)
    except Exception as e:  # noqa: BLE001
        raise JudgeUnavailable(str(e)) from e
    if not out:
        raise JudgeUnavailable("empty/no response from judge")
    return out


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


def competitor_signal_verdict(seller_sells: str, rival: str, passage: str) -> dict:
    """Is the SIGNAL passage itself actually describing a real competitive move in the
    seller's own category -- not just whether `rival` could ever be a competitor in the
    abstract (that is competitor_verdict_binary's job, from general knowledge of the
    company). This one reads the passage and decides if THIS claim is honest.

    Forces a decisive yes/no, no 'unknown' escape -- confirmed 2026-10-14 that the
    weaker 5-way check let real failures through by landing on 'adjacent' or 'unknown'
    rather than committing to an answer (a real case: seller and rival were both real
    companies in loosely related sectors, but the actual event reported -- a retail
    store opening -- had nothing to do with the seller's specific product; the passage's
    own hedged wording was enough for the weaker check to not commit). Describes the
    distinguishing PRINCIPLE rather than naming the specific companies/products involved
    in any one past case, deliberately -- a worked example risks the model pattern-
    matching on those specific companies/wording rather than learning the general test,
    and producing false negatives (or hallucinated similarity) on every other case that
    doesn't look like the example.
    """
    if not _enabled() or not rival:
        return {"verdict": "unknown", "why": "llm check disabled"}
    system = (
        "You judge whether a news passage actually describes a real competitive move "
        "in the seller's own specific product category -- not just whether the two "
        "companies are both real, or both named, or both operate somewhere in the "
        "same broad economy. You must answer YES or NO. There is no 'unknown' or "
        "'maybe' option: if the passage is ambiguous, weigh the test below and commit "
        "to the closer answer.\n\n"
        "The test: does the EVENT being reported -- the specific thing the rival did -- "
        "involve the same product or service the seller sells, or something a buyer "
        "would genuinely weigh as an alternative to it? Answer YES only if removing "
        "every category label and buzzword from the passage still leaves a real move "
        "in the seller's actual business. Answer NO if the event itself (a hire, an "
        "expansion into an unrelated line of business, a store opening, a stock-price "
        "move, a deal in a different market) has no bearing on the seller's specific "
        "product, even when the passage dresses it up with the seller's own category "
        "words ('manufacturing', 'AI', 'software', etc.) to make it sound connected -- "
        "judge the actual event, not which words appear near it. Also answer NO when "
        "the passage itself admits there is no real competitive news (hedging language "
        "that concedes the signal is weak, unclear, or not really about a competitor), "
        "even if the section is formatted as if it found something.\n"
    )
    user = (f"SELLER'S PRODUCT: {seller_sells}\n"
            f"COMPANY NAMED AS COMPETITOR: {rival}\n\n"
            f"SIGNAL PASSAGE:\n{passage[:900]}\n\n"
            'Return JSON {"real_signal":"yes|no","why":"one short sentence"}')
    out = _ask(system, user, {"real_signal": "", "why": "no response"})
    ans = str(out.get("real_signal", "")).strip().lower()
    verdict = {"yes": "real", "no": "unrelated"}.get(ans, "unknown")
    return {"verdict": verdict, "why": str(out.get("why", ""))[:160]}


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


def topic_alignment_verdict(email: str, kind: str, expected: str, text: str) -> dict:
    """Does this brief's ACTUAL content match what the recipient was promised?

    Added 2026-10-07 after a real failure: a lead who asked about the "Ship Breaking
    Market" (kind='report') was sent a brief entirely about hydraulic hammers,
    Salesforce acquiring an AI startup, and GPU-as-a-Service -- none of it related to
    ship breaking at all. Root cause was a separate bug (a blank-company lead silently
    fell back to CMI's own demo profile instead of raising), but nothing would have
    CAUGHT the resulting brief before it shipped -- every existing check validates the
    brief against its own `profile` object, which was itself wrong. This check instead
    validates against `expected`, which comes straight from the lead record in the DB
    (`leads.company` for a signal brief, `leads.report_name` for a report brief) --
    independent of whatever profile the generation step happened to build, so it is
    blind to a bad profile. For a report brief specifically, `expected` is the exact
    market/industry name the lead enquired about; it must be unambiguous in the
    content, not just thematically adjacent.

    Returns {"verdict": aligned|misaligned|unknown|judge_unavailable, "why": "..."}.
    `judge_unavailable` means the call itself failed (dead endpoint, malformed
    response) -- it is NOT the same as the model genuinely answering 'unknown', and
    callers must not treat it as a pass. See JudgeUnavailable's docstring for why
    this distinction exists.
    """
    if not _enabled() or not expected:
        return {"verdict": "unknown", "why": "llm check disabled or no expected topic"}
    kind_label = "an industry/market report the recipient specifically asked about" \
        if kind == "report" else "a sales-intelligence brief about the recipient's own company"
    system = (
        "You judge whether a newsletter's actual content matches what the recipient was "
        "promised. Read only the CONTENT given -- do not assume good faith; a brief that "
        "drifts onto unrelated companies or markets is a real failure, not a stylistic "
        "choice.\n"
        "  aligned     = the content is clearly and substantially about EXPECTED\n"
        "  misaligned  = the content is about something else, even if related/adjacent\n"
        "  unknown     = you cannot tell from the content given"
    )
    user = (
        f"This brief was sent to {email or 'a recipient'} as {kind_label}.\n"
        f"EXPECTED SUBJECT: {expected}\n\n"
        f"BRIEF CONTENT:\n{text[:2000]}\n\n"
        'Return JSON {"verdict":"aligned|misaligned|unknown","why":"one short sentence '
        'naming what the content was actually about if misaligned"}')
    try:
        out = _ask_checked(system, user, None)
    except JudgeUnavailable as e:
        return {"verdict": "judge_unavailable", "why": f"topic check did not run: {e}"[:200]}
    v = str(out.get("verdict", "unknown")).strip().lower()
    if v not in {"aligned", "misaligned", "unknown"}:
        v = "unknown"
    return {"verdict": v, "why": str(out.get("why", ""))[:200]}


def judge_brief(text: str, profile, *, market: str = "", email: str = "",
                kind: str = "", expected_subject: str = "") -> tuple[list[str], list[str]]:
    """Run all checks over a brief's visible text. Returns (fails, warns).

    Only a confident wrong answer becomes a FAIL: 'unknown' is silent, and 'adjacent'
    is a warning because a near-neighbour is a judgement call a human should make.

    `email`/`kind`/`expected_subject` enable the topic-alignment check (see
    topic_alignment_verdict) -- all three are optional so existing callers that only
    pass `text`/`profile`/`market` keep working exactly as before, just without it.
    """
    fails: list[str] = []
    warns: list[str] = []
    if not _enabled():
        return fails, warns

    if expected_subject:
        tv = topic_alignment_verdict(email, kind, expected_subject, text)
        if tv["verdict"] == "misaligned":
            fails.append(
                f"LLM TOPIC ERROR: brief does not match the expected subject "
                f"'{expected_subject}'. {tv['why']}")
        elif tv["verdict"] == "judge_unavailable":
            # Surfaced, never silent: a dead judge must not look like a pass. This is
            # a warn rather than a fail because the deterministic checks are still the
            # floor -- but it must show up somewhere so "never checked" is visible in
            # the stored validation record, distinct from "checked and aligned".
            warns.append(
                f"LLM TOPIC CHECK SKIPPED: topic-alignment judge did not run against "
                f"expected subject '{expected_subject}' ({tv['why']}). This brief's "
                "topic alignment has NOT been verified.")

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
        else:
            # The passage-based verdict ("real"/"adjacent"/"unrelated"/"unknown") was
            # measured at only 1/5 correct on real failures -- it is too easy for a
            # brief to talk its way to "adjacent" or "unknown" by mentioning the
            # seller's own category words in a sentence about an unrelated company
            # (confirmed 2026-10-14: Offeri/PriceSmart, a metal-fabrication software
            # seller paired with a Caribbean retail chain, passed clean because the
            # passage said "points to more competition... in the broader manufacturing
            # ecosystem" -- true-sounding but not actually about a competing product).
            # The binary method (competitor_verdict_binary) was measured at 4/5 by
            # asking one decisive yes/no question instead of a 5-way label, so it now
            # runs on every verdict except "customer", not only as a fallback for
            # "unknown" -- there is no verdict from the weaker method that should be
            # allowed to end the check without the stronger one's confirmation.
            b = competitor_verdict_binary(sells, comp)
            if b["verdict"] == "unrelated":
                sink("LLM ROLE ERROR: '" + comp + "' (" + b.get("their_business", "") +
                     ") sells nothing a buyer of " + seller + "'s product would "
                     "substitute. " + b["why"])
            else:
                # competitor_verdict_binary judges the COMPANY in the abstract (could a
                # buyer get the same thing from them, from general knowledge of their
                # business) -- it cannot catch a real competitor being cited for an
                # event that has nothing to do with the shared category (Offeri's rival
                # could genuinely compete with Offeri on quotation software and still be
                # cited here for an unrelated retail-store story). Read the actual
                # passage and demand a decisive yes/no on whether THIS claim is honest.
                s = competitor_signal_verdict(sells, comp, _passage(text, "Competition Outlook"))
                if s["verdict"] == "unrelated":
                    sink(f"LLM ROLE ERROR: the Competition Outlook signal about '{comp}' "
                         f"is not actually a competitive move in {seller}'s category. "
                         f"{s['why']}")
                elif v["verdict"] == "adjacent" and b["verdict"] != "real":
                    # Neither the binary company check nor the passage check confirmed
                    # a genuine rival -- still worth a warn.
                    warns.append(f"LLM: '{comp}' is adjacent, not a direct substitute. {v['why']}")

    cust = vb.section_company(text, "Potential Customer")
    if cust:
        v = customer_verdict(seller, sells, cust, _passage(text, "Potential Customer"))
        if v["verdict"] in {"rival", "unrelated"}:
            sink(f"LLM ROLE ERROR: Potential Customer '{cust}' reads as "
                 f"{v['verdict']}, not a buyer. {v['why']}")
        elif v["verdict"] == "supplier":
            warns.append(f"LLM: '{cust}' looks like a supplier to {seller}. {v['why']}")

    return fails, warns
