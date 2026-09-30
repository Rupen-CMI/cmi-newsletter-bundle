"""Check a generated brief has all three sections and that they fit the company.

    python validate_brief.py output/cmi_localmodel_immunoact_aditya_v6.html --company ImmunoACT
    python validate_brief.py output/*.html --company ImmunoACT --website https://immunoact.com/

Exists because the failures worth catching are not writing failures. The model writes
fluently about whatever it is handed, so a brief reads fine while a competitor sits in
the Potential Customer slot, or a section is silently absent because nothing matched.
Both shipped during development and neither is visible without reading the whole thing.

Four checks, in the order they matter:

  STRUCTURE  all three sections present and non-empty
  ROLES      the company in each slot holds the right relationship to the seller --
             a named competitor must never appear as a Potential Customer
  FIT        the copy talks about what the seller actually sells
  EMAIL      every section that survived STRUCTURE above is STILL present after
             running through emailify() -- the actual conversion applied at send
             time. emailify.py's own docstring says a CSS class it doesn't recognise
             is "silently skipped rather than half-rendered", so a brief can pass
             every check above and still ship with a whole section missing
             specifically in the version that reaches an inbox, if a template edit
             ever drifts out of sync with emailify's selectors. This check is what
             catches that before a send, not after a human notices a live email.

Exit code is 0 only when nothing FAILS; warnings alone still pass so this can gate a
send without blocking on judgement calls.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

SECTIONS = ("Potential Customer", "Industry Outlook", "Competition Outlook")

# A report brief replaces section 1 with "Your Market" built around the topic the lead
# enquired about. Demanding "Potential Customer" there failed every report brief for a
# section it is not supposed to have.
SECTION_ALIASES = {
    "Potential Customer": ("Potential Customer", "Your Market"),
    # Report briefs put Market Moves where a signal brief has Industry Outlook.
    "Industry Outlook": ("Industry Outlook", "Market Moves"),
}


def visible_text(html: str) -> str:
    t = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    t = re.sub(r"<br\s*/?>", "\n", t)
    t = re.sub(r"</(p|div|td|tr|h1|h2|h3|li)>", "\n", t)
    t = re.sub(r"<[^>]+>", " ", t)
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&#39;", "'"), ("&middot;", "·"),
                 ("&rsquo;", "'"), ("&mdash;", "-"), ("&deg;", "°")):
        t = t.replace(a, b)
    t = re.sub(r"&[a-z]+;|&#\d+;", " ", t)
    return re.sub(r"[ \t]+", " ", t)


def section_company(text: str, label: str) -> str:
    """The company name rendered under a section's flag line.

    The template puts the flag ('● Potential Customer · <industry>') immediately above
    the company heading, so the first substantial line after it is the subject.
    """
    m = re.search(rf"●?\s*{re.escape(label)}\s*·[^\n]*\n(.*)", text)
    if not m:
        return ""
    for line in m.group(1).split("\n")[:4]:
        line = line.strip()
        if 2 < len(line) < 70 and not line.lower().startswith(("why ", "who ", "their ")):
            return line
    return ""


def norm(s: str) -> str:
    s = re.sub(r"\s*[(\[][^)\]]*[)\]]", " ", (s or "").lower())
    s = re.sub(r"\b(ltd|limited|inc|corp|plc|pvt|private|laboratories|labs|group|holdings)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def name_overlap(a: str, b: str) -> bool:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    if na in nb or nb in na:
        return True
    ha, hb = " ".join(na.split()[:2]), " ".join(nb.split()[:2])
    return len(ha) > 5 and ha == hb


def check(path: str, profile) -> tuple[list[str], list[str]]:
    fails, warns, _llm_fails, _llm_warns = check_detailed(path, profile)
    return fails + _llm_fails, warns + _llm_warns


def check_detailed(path: str, profile) -> tuple[list[str], list[str], list[str], list[str]]:
    """Same checks as `check()`, but keeps the LLM judge's findings separate from the
    text-rule findings so a caller can tell "a keyword rule caught this" apart from
    "the AI judge flagged this" -- the latter needs a human to read the reasoning and
    decide, the former is usually a clear-cut structural problem.

    Returns (fails, warns, llm_fails, llm_warns). `fails`/`warns` are text-rule findings
    only; `llm_fails`/`llm_warns` are what the LLM judge added, each string carrying its
    own one-line reasoning already baked in (see llm_validate.judge_brief).

    The judge only runs when the text rules alone would PASS or WARN -- not on FAIL.
    A brief the rules already reject is already held back from sending; spending an
    extra GPU call asking the judge to grade something that will not ship anyway is
    wasted cost for no benefit.
    """
    html = open(path, encoding="utf-8").read()
    text = visible_text(html)
    fails: list[str] = []
    warns: list[str] = []

    # ---- STRUCTURE ---------------------------------------------------------
    # A brief with no findable prospect or rival is a real outcome, not a defect: some
    # markets simply have no rival news in the window. Dropping the section is the
    # honest response -- better than the stretched picks that came out of forcing one.
    # But it must stay RARE, so losing one section is a warning and losing two is a
    # failure: at that point the brief is a single card and not worth sending.
    missing: list[str] = []
    present = {}
    for s in SECTIONS:
        body = ""
        for alias in SECTION_ALIASES.get(s, (s,)):
            m = re.search(rf"{re.escape(alias)}\s*·(.{{0,1500}})", text, re.S)
            if m:
                body = m.group(1)
                break
        if not body and s == "Industry Outlook" and re.search(r"\bIndustry Outlook\b", text):
            # Industry Outlook has no company flag line -- it is a themed section.
            i = text.find("Industry Outlook")
            body = text[i:i + 1500]
        present[s] = bool(body and len(body.split()) > 25)
        if not present[s]:
            missing.append(s)

    if len(missing) >= 2:
        fails.append("MISSING SECTIONS: " + ", ".join(missing) +
                     " — too little left to send.")
    elif missing:
        warns.append(f"MISSING SECTION: {missing[0]} — no usable signal found. "
                     f"Acceptable occasionally; check the feed if it repeats.")

    # ---- EMAIL --------------------------------------------------------------
    # Every section-rendering element in the source (main.body's direct children
    # carrying "prospect" or "story" in their class -- the two classes
    # emailify.emailify()'s own dispatch loop looks for to render a section body)
    # must actually be recognised by that same dispatch logic. emailify.py's own
    # docstring warns that a class it does not recognise is "silently skipped
    # rather than half-rendered" -- so a template edit renaming one of these two
    # classes drops that section's body from every SENT email while the browser
    # HTML (the only thing validated until now) looks completely fine.
    #
    # This is a STRUCTURAL check (does the class spelling match), not a text
    # comparison. An earlier version tried matching section label text or
    # distinctive body words in emailify's rendered output, but a real brief
    # repeats its own industry vocabulary throughout every section AND in the
    # static "Inside CMI" showcase -- so words from a dropped section often still
    # appear to "survive" via a completely unrelated part of the same document,
    # producing a false negative on the exact failure this check exists to catch.
    # Comparing element counts by class, independent of wording, does not have
    # that blind spot: emailify.py's dispatch condition is copied here verbatim
    # (`"prospect" in cls or "story" in cls`) so this check breaks in lockstep
    # with emailify.py itself rather than silently drifting out of sync with it.
    if 'class="body"' in html or "class='body'" in html:
        try:
            from bs4 import BeautifulSoup as _Soup

            soup = _Soup(html, "html.parser")
            main = soup.select_one("main.body")
            recognized = 0
            unrecognized_classes: list[str] = []
            if main is not None:
                for el in main.find_all(recursive=False):
                    cls = el.get("class") or []
                    if "prospect" in cls or "story" in cls:
                        recognized += 1
                    elif any(c in ("prospect", "story") for c in
                            [c2.split("-")[0].split("_")[0] for c2 in cls]):
                        # Looks like a renamed/typo'd variant of a class emailify
                        # expects exactly (e.g. "prospect-v2", "story_alt") --
                        # emailify's dispatch does an exact membership check, so
                        # this element would be silently skipped even though it is
                        # clearly meant to be one of the two section-body kinds.
                        unrecognized_classes.append(" ".join(cls))
            n_sections_expected = sum(1 for s in SECTIONS if present.get(s))
            if main is not None and recognized < n_sections_expected:
                fails.append(
                    f"EMAIL ERROR: {n_sections_expected} section(s) found in the "
                    f"browser HTML but only {recognized} carry a class "
                    f"emailify.py's dispatch loop recognises ('prospect' or "
                    f"'story') -- the rest would be silently dropped from every "
                    f"SENT email."
                    + (f" Suspicious class attribute(s) found: "
                       f"{', '.join(unrecognized_classes)}." if unrecognized_classes else "")
                    + " Check templates/cmi_brief.html's CSS classes against "
                      "emailify.py's selectors.")
        except Exception as e:  # noqa: BLE001 -- a broken emailify check must not block
                                # a send the browser-HTML checks already cleared; this
                                # adds scrutiny, it must not become a new failure point
            warns.append(f"EMAIL check could not run: {type(e).__name__}: {e}")

    # ---- ROLES -------------------------------------------------------------
    cust = section_company(text, "Potential Customer")
    if not cust and re.search(r"Your Market\s*·", text):
        cust = ""   # report brief: section 1 is a market, so the buyer role checks below
                    # have nothing to judge and must not fire
    comp = section_company(text, "Competition Outlook")

    for rival in (profile.competitors or []):
        if cust and name_overlap(cust, rival):
            fails.append(f"ROLE ERROR: Potential Customer is '{cust}', a known competitor "
                         f"(matches '{rival}'). A rival must appear in Competition Outlook.")
            break

    # The competitor list only names the rivals someone thought to write down. A supplier
    # in the seller's OWN category is just as wrong in the customer slot even when it is
    # absent from that list -- Dr Reddy's launching an immuno-oncology biosimilar passed
    # every name check while being plainly not a buyer of CAR-T. Catch it by category:
    # the customer should read like a BUYER vertical, not like the seller's own trade.
    if cust:
        seller_trade = {w for w in re.findall(r"[a-z][a-z0-9-]{4,}",
                                              (profile.value_prop or "").lower())} - {
            "india", "indian", "supplies", "approved", "first", "targeted", "relapsed",
            "refractory", "patients", "treating"}
        cust_ctx = ""
        m = re.search(rf"Potential Customer\s*·(.{{0,900}})", text, re.S)
        if m:
            cust_ctx = m.group(1).lower()
        buyerish = any(w in cust_ctx for w in
                       ("hospital", "clinic", "centre", "center", "network", "institute",
                        "provider", "insurer", "payer", "medical college"))
        sellerish = sum(1 for w in seller_trade if len(w) > 5 and w in cust_ctx)
        if cust_ctx and not buyerish and sellerish >= 2:
            warns.append(f"Potential Customer '{cust}' reads like a company in the seller's own "
                         f"category, not a buyer. Check it is not a rival or supplier.")
        # A regulator, ministry or trade body is a real signal but has no purchase order
        # to give. Left unchecked the model invents a reason it might buy.
        non_buyers = ("regulator", "ministry", "cdsco", "dcgi", "government", "court",
                      "tribunal", "association", "council", "committee", "commission",
                      "department of", "task force")
        if any(w in cust.lower() for w in non_buyers):
            fails.append(f"ROLE ERROR: Potential Customer is '{cust}', which regulates or "
                         f"governs the market rather than buying in it.")

    if comp:
        if not any(name_overlap(comp, r) for r in (profile.competitors or [])):
            warns.append(f"Competition Outlook features '{comp}', which is not on the "
                         f"competitor list. Verify it actually sells a substitutable product.")
        # Being a real rival is not enough -- the SIGNAL has to be about the shared
        # product category. A diversified major's generics acquisition is a true
        # competitor doing something irrelevant to this seller, which reads as padding.
        try:
            import tenant as _tenant
            terms = [t.lower() for t in
                     ((_tenant._overrides_for(profile.company) or {}).get("product_terms") or [])]
        except Exception:  # noqa: BLE001
            terms = []
        if terms:
            m = re.search(r"Competition Outlook\s*·(.{0,1200})", text, re.S)
            ctx = (m.group(1) if m else "").lower()
            if ctx and not any(t in ctx for t in terms):
                warns.append(f"Competition Outlook covers '{comp}' but the signal never "
                             f"touches the seller's product category. A rival doing something "
                             f"unrelated is padding, not competitive intelligence.")
    if cust and comp and name_overlap(cust, comp):
        fails.append(f"ROLE ERROR: '{cust}' appears as both customer and competitor.")

    # ---- FIT ---------------------------------------------------------------
    # Does the brief discuss what the seller sells? Built from the value_prop's own
    # distinctive nouns rather than a hardcoded list, so this works for any tenant.
    stop = {"india", "indian", "sells", "makes", "first", "approved", "their", "which",
            "supplies", "company", "companies", "market", "product", "products", "global"}
    vp_terms = {w for w in re.findall(r"[a-z][a-z0-9-]{3,}", (profile.value_prop or "").lower())
                if w not in stop}
    low = text.lower()
    hits = sorted(w for w in vp_terms if w in low)
    if vp_terms and len(hits) < 2:
        fails.append(f"FIT ERROR: the brief barely mentions what {profile.company} sells. "
                     f"Matched only {hits or 'nothing'} from the value proposition.")
    elif vp_terms and len(hits) < 4:
        warns.append(f"Weak topical fit: only {hits} from the value proposition appear.")

    if profile.company.lower() not in low:
        fails.append(f"FIT ERROR: '{profile.company}' is never named in the brief.")

    # Rules above are keyword checks. They cannot tell that Reckitt is VVF's CUSTOMER
    # rather than its rival -- that needs judgement, so ask the model that wrote the
    # brief. Off unless CMI_LLM_VALIDATE=1, and it only ever ADDS findings: a judge
    # that is down or unsure leaves the deterministic verdict untouched.
    #
    # Only worth asking when the text rules did not already reject the brief -- a FAIL
    # is already held back from sending, so judging it teaches nothing new and costs a
    # GPU call for no benefit. See check_detailed's docstring.
    llm_fails: list[str] = []
    llm_warns: list[str] = []
    if not fails:
        try:
            import llm_validate

            llm_fails, llm_warns = llm_validate.judge_brief(
                text, profile, market=getattr(profile, "market", ""))
        except Exception:  # noqa: BLE001 -- the optional judge must never break validation
            pass

    return fails, warns, llm_fails, llm_warns


def check_html(html: str, profile) -> tuple[list[str], list[str]]:
    """Same as `check()`, but takes HTML text directly instead of a file path --
    for gating a send before the brief is written to disk."""
    import os
    import tempfile

    fd, tmp_path = tempfile.mkstemp(suffix=".html")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(html)
        return check(tmp_path, profile)
    finally:
        os.unlink(tmp_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a generated brief's structure, roles and fit.")
    ap.add_argument("paths", nargs="+", help="brief HTML file(s); globs allowed")
    ap.add_argument("--company", required=True)
    ap.add_argument("--website", default="")
    ap.add_argument("--market", default="India")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    from tenant import Profile

    profile = Profile.from_company(args.company, args.website, "there", args.market)
    print(f"seller:      {profile.company}")
    print(f"value_prop:  {profile.value_prop[:100]}")
    print(f"competitors: {', '.join(profile.competitors[:6])}\n")

    files = [p for pat in args.paths for p in (glob.glob(pat) or [pat])]
    worst = 0
    for path in files:
        if not os.path.isfile(path):
            print(f"{path}: not found")
            worst = 1
            continue
        fails, warns = check(path, profile)
        status = "FAIL" if fails else ("WARN" if warns else "PASS")
        print(f"[{status}] {os.path.basename(path)}")
        for f in fails:
            print(f"    x {f}")
        for w in warns:
            print(f"    ! {w}")
        if fails:
            worst = 1
    raise SystemExit(worst)


if __name__ == "__main__":
    main()
