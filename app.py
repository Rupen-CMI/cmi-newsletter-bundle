"""Coherent Market Insights — brief generator (Streamlit).

    streamlit run app.py

Type a company, pick a brief type, get the finished newsletter: preview it, run the
quality checks against it, download the HTML.

Two brief types:
  Signal Brief — Potential Customer / Industry Outlook / Competition Outlook
  Report Brief — Your Market       / Industry Outlook / Competition Outlook

Both use the same pipeline (tenant profile → collect news → score → LLM synthesis)
and the same template, so they cannot drift apart visually.
"""
from __future__ import annotations

import io
import os
import re
import sys
import traceback
from datetime import datetime, timezone

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import compose                     # noqa: E402
import config                      # noqa: E402
import geo                         # noqa: E402
import llm                         # noqa: E402
import render                      # noqa: E402
from brief_builder import build_brief   # noqa: E402
from collect import collect            # noqa: E402
from score import score_all, eligible  # noqa: E402
from tenant import Profile             # noqa: E402

OUT = os.path.join(HERE, "output")
TEMPLATE = "cmi_brief.html"

st.set_page_config(page_title="CMI Brief Generator", page_icon="📰", layout="wide")


# ── validator (optional, from the shared humanize toolkit) ───────────────────
def find_validator() -> str | None:
    for cand in (os.path.join(HERE, "..", "..", "wmr docs", "humanize"),
                 os.path.join(os.path.expanduser("~"), "Desktop", "wmr docs", "humanize")):
        if os.path.isfile(os.path.join(cand, "validate.py")):
            return os.path.abspath(cand)
    return None


def run_checks(path: str, company: str, competitors, industries, product) -> str:
    """Run the shared validator in-process. Returns its report, or a reason it was skipped."""
    vdir = find_validator()
    if not vdir:
        return "Validator not found (expected 'wmr docs/humanize/validate.py')."
    if vdir not in sys.path:
        sys.path.insert(0, vdir)
    try:
        import importlib
        import json as _json
        validate = importlib.import_module("validate")
        importlib.reload(validate)
        cfg_path = os.path.join(vdir, "structure.cmi.json")
        cfg = _json.load(open(cfg_path, encoding="utf-8")) if os.path.isfile(cfg_path) else {}
        prof = {"company": company, "own_category": product,
                "must_mention": [], "industries": list(industries),
                "competitors": list(competitors)}
        html = open(path, encoding="utf-8").read()
        res = validate.Results()
        validate.check_structure(html, cfg, res)
        validate.check_account(html, cfg, prof, res)
        validate.check_slop(html, cfg, res)
        return res.report()
    except Exception as e:  # noqa: BLE001 — checks must never break generation
        return f"Checks could not run: {type(e).__name__}: {e}"


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "client").lower()).strip("_") or "client"


# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Provider")
    st.caption(f"LLM: **{llm.PROVIDER}** "
               f"({'DeepSeek key set' if llm.DEEPSEEK_API_KEY else 'no DeepSeek key'}, "
               f"{'OpenAI fallback' if llm.OPENAI_API_KEY else 'no OpenAI key'})")
    if not (llm.DEEPSEEK_API_KEY or llm.OPENAI_API_KEY):
        st.error("No API key found. Put DEEPSEEK_API_KEY in a .env file next to app.py.")

    st.markdown("### Social proof")
    st.caption("Logos and quotes are picked from the vertical that matches the client: "
               "life sciences, consumer, technology, industrial or services. If the "
               "vertical cannot be established, both are hidden rather than guessed.")
    inc_logos = st.checkbox("Include client logo wall", value=True)
    inc_quotes = st.checkbox("Include client testimonials", value=True)
    st.caption("Certifications always appear — they are company-wide.")

    st.markdown("### Quality checks")
    run_validation = st.checkbox("Run checks after generating", value=True)


# ── main form ────────────────────────────────────────────────────────────────
st.title("Coherent Market Insights · Brief Generator")
st.caption("One company in, one finished newsletter out.")

col1, col2 = st.columns([3, 2])
with col1:
    kind_label = st.radio(
        "Brief type", ["Signal Brief", "Report Brief"], horizontal=True,
        help="Signal Brief leads with a potential customer. Report Brief leads with the "
             "market the reader asked about.")
    kind = "signal" if kind_label == "Signal Brief" else "report"

    company = st.text_input("Client company *", placeholder="ImmunoACT")
    website = st.text_input("Website (optional)", placeholder="https://immunoact.com")

with col2:
    recipient = st.text_input("Recipient first name", value="there", placeholder="Satyen")
    countries = sorted({c["name"] for c in config.COUNTRIES.values()}) \
        if hasattr(config, "COUNTRIES") else []
    market = st.selectbox("Market", ["Global", "India", "United States", "United Kingdom",
                                     "Germany", "Japan", "China", "Brazil", "Australia"]
                          if not countries else ["Global"] + countries, index=1)
    market_topic = ""
    if kind == "report":
        market_topic = st.text_input(
            "Market / report topic *", placeholder="Bone Marrow Transplant Market",
            help="The subject the reader asked about. It becomes section one.")

go = st.button("Generate brief", type="primary", use_container_width=True)

if go:
    if not company.strip():
        st.error("Enter a client company.")
        st.stop()
    if kind == "report" and not market_topic.strip():
        st.error("A Report Brief needs a market or report topic.")
        st.stop()

    os.makedirs(OUT, exist_ok=True)
    log_lines: list[str] = []

    def log(msg=""):
        log_lines.append(str(msg))

    status = st.status("Generating…", expanded=True)
    try:
        country = config.get_country(market)
        market_name = country["name"]

        status.write("Building the tenant profile…")
        profile = Profile.from_company(company.strip(), website.strip(),
                                       recipient.strip() or "there", market_name)
        status.write(f"Targets: {', '.join(profile.industries)}")
        status.write(f"Competitors: {', '.join(profile.competitors) or '—'}")

        status.write("Collecting live news…")
        articles = collect(profile, country)
        if not articles:
            status.update(label="No articles found", state="error")
            st.error("No articles came back. Check the network, then try again.")
            st.stop()
        status.write(f"{len(articles)} articles collected")

        if not geo.is_global(country):
            kept, fallback = geo.filter_country(articles, country)
            status.write(f"Geo filter → {market_name}: kept {len(kept)}/{len(articles)}"
                         f"{' (too few, kept all)' if fallback else ''}")
            articles = kept

        status.write("Scoring…")
        scored = score_all(articles, profile)
        keep = eligible(scored)
        status.write(f"{len(keep)}/{len(scored)} passed the minimum score")

        status.write("Writing the sections…")
        brief = build_brief(keep or scored, profile, log=log)

        status.write("Composing the brief…")
        ctx = compose.build_context(profile, brief[0] if isinstance(brief, tuple) else brief,
                                    kind, market_topic=market_topic.strip(),
                                    include_logos=inc_logos, include_quotes=inc_quotes,
                                    log=log)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        name = (f"cmi_{'signalbrief' if kind == 'signal' else 'reportbrief'}_"
                f"{slug(profile.company)}_{slug(market_name)}_{stamp}.html")
        path = os.path.join(OUT, name)
        html = render.render(ctx, TEMPLATE)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        status.update(label="Done", state="complete")

        st.success(f"Wrote `output/{name}`")
        st.session_state["html"] = html
        st.session_state["name"] = name
        st.session_state["path"] = path
        st.session_state["profile"] = profile
        st.session_state["log"] = log_lines
        st.session_state["run_checks"] = run_validation

    except Exception as e:  # noqa: BLE001
        status.update(label="Failed", state="error")
        st.error(f"{type(e).__name__}: {e}")
        st.code(traceback.format_exc(), language="text")
        st.stop()


# ── results ──────────────────────────────────────────────────────────────────
if st.session_state.get("html"):
    html = st.session_state["html"]
    name = st.session_state["name"]
    profile = st.session_state["profile"]

    st.download_button("Download HTML", data=html.encode("utf-8"), file_name=name,
                       mime="text/html", type="primary")

    tabs = st.tabs(["Preview", "Quality checks", "Pipeline log", "HTML"])

    with tabs[0]:
        st.components.v1.html(html, height=1400, scrolling=True)

    with tabs[1]:
        if st.session_state.get("run_checks"):
            report = run_checks(st.session_state["path"], profile.company,
                                profile.competitors, profile.industries,
                                getattr(profile, "value_prop", ""))
            failed = "[FAIL]" in report
            (st.error if failed else st.success)(
                "Checks failed — see below" if failed else "All checks passed")
            st.code(report, language="text")
            st.caption("Structure, account fit, AI-slop, grammar, voice, register and "
                       "readability. Run from the shared humanize toolkit.")
        else:
            st.info("Checks were switched off in the sidebar.")

    with tabs[2]:
        st.code("\n".join(st.session_state.get("log") or ["(no output)"]), language="text")

    with tabs[3]:
        st.code(html, language="html")
else:
    st.info("Fill in a company and press **Generate brief**.")
    with st.expander("What comes out"):
        st.markdown(
            "- Masthead, lede and a numbered top-three summary\n"
            "- Three sections, each with a news item, a 3-bullet **Analyst Opinion** "
            "carrying a dated provenance line, and a full-width CTA panel\n"
            "- **Inside CMI** panel with six engagements named for the client's industry\n"
            "- Optional client logo wall and testimonials\n"
            "- Certification band (ISO 9001, ISO 27001, ESOMAR, D-U-N-S, Clutch, Trustpilot)\n"
            "- Footer with demo and site CTAs")
