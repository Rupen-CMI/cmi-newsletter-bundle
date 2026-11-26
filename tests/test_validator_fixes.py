"""Unit tests for the Phase 5 validator work (2026-11 brief-quality review).

Every check is a small pure function (text -> messages) so it can be tested without a
browser or an LLM; the wiring tests then confirm check_detailed() reports the same
findings and that the fixed template blocks are never flagged.

Run with:
    venv/Scripts/python.exe -m pytest tests/test_validator_fixes.py -q -p no:cacheprovider
"""
from __future__ import annotations

import pytest

import config
import tenant
import validate_brief as vb


# ── fixtures ─────────────────────────────────────────────────────────────────
def _profile(**kw):
    """A non-pharma Profile; tests override only the field under test."""
    base = dict(company="FamPay", short="FamPay", recipient="Aditya",
                tagline="teen payments", value_prop="prepaid cards for teenagers",
                industries=["fintech", "retail banks"], competitors=["Slice", "Jupiter"],
                industry_keywords={"fintech": ["payments", "upi"]},
                trigger_queries=["fampay funding"], industry_queries=["upi growth"],
                buyer_functions=["retail banks", "payment processors"],
                market="India")
    base.update(kw)
    return tenant.Profile(**base)


_BODY = ("The company is expanding capacity across the region this quarter and has "
         "signalled further investment, with management pointing to sustained demand "
         "from domestic buyers and a broader order book than the prior year period.")
_BODY_2 = ("Regulators published a consultation on settlement rules, which the trade "
           "body said would shorten payment cycles for merchants over the next year.")
_BODY_3 = ("A rival opened two new offices in the region and said it would hire "
           "additional staff to serve existing clients through the coming quarter.")


def _brief(*, lede="This week's brief covers two signals: a potential customer in "
                  "Poland and the industry outlook on retail parks.",
           customer="Razorpay", customer_kind="fintech",
           body=_BODY, bullet="Cash reserves cover the announced build-out.",
           story_body=_BODY_2, competition="Slice", competition_body=_BODY_3,
           tail='<p class="agentline">Everything below comes from <b>CMI\'s '
                'analyst-verified research base</b>.</p>'):
    """A realistic slice of templates/cmi_brief.html around the model-written parts."""
    return f"""<html><body>
<header class="masthead"><div class="brandrow">Masthead copy</div></header>
<main class="body">
<h1 class="hello">Hello, Aditya.</h1>
<p class="lede">{lede}</p>
{tail}
<div class="top3"><div class="row"><div class="idx">1</div>
 <div class="txt"><span class="cat">Customer move</span>Short top3 line.</div></div></div>

<div class="seccta"><p class="q">Want accounts mapped?</p>
 <a class="btn" href="https://example.com">Get on a demo call</a></div>
<div class="prospect">
  <span class="flag">&#9679; Potential Customer &middot; {customer_kind}</span>
  <h2>{customer} raises funding</h2>
  <p>{body}</p>
  <div class="grid2"><div class="cell"><div class="h">Deal</div>
    <div class="v">Undisclosed</div></div></div>
</div>
<div class="analyst"><div class="tag">Analyst Opinion</div>
 <ul><li>{bullet}</li></ul>
 <div class="prov"><span class="src">CMI Intelligence</span>
  <span class="verified">&#10003; Analyst-reviewed</span></div></div>

<div class="seccta"><p class="q">Want these shifts tracked?</p>
 <a class="btn" href="https://example.com">Get on a demo call</a></div>
<div class="story">
  <span class="flag">&#9679; Industry Outlook</span>
  <h3>Payments outlook</h3>
  <p>{story_body}</p>
</div>
<div class="analyst"><div class="tag">Analyst Opinion</div>
 <ul><li>{bullet}</li></ul>
 <div class="prov"><span class="src">CMI Intelligence</span></div></div>

<div class="seccta"><p class="q">Want competitor movement flagged?</p>
 <a class="btn" href="https://example.com">Get on a demo call</a></div>
<div class="prospect">
  <span class="flag">&#9679; Competition Outlook &middot; {customer_kind}</span>
  <h2>{competition} expands</h2>
  <p>{competition_body}</p>
</div>
<div class="analyst"><div class="tag">Analyst Opinion</div>
 <ul><li>{bullet}</li></ul>
 <div class="prov"><span class="src">CMI Intelligence</span></div></div>

<div class="eyebrow"><span class="lbl">Inside Coherent Market Insights</span></div>
<div class="caps">
  <h3>Explore revenue-driven opportunities in <em>50+ countries</em>.</h3>
  <p class="capsub">A seamless, robust ecosystem with leverage and synergies.</p>
  <div class="capcard"><h4>A project title</h4>
    <p>Our pivotal, crucial, game-changer work unlocked the landscape.</p></div>
</div>
<div class="eyebrow"><span class="lbl">Client Testimonials</span></div>
<div class="tgrid"><div class="tcard">
  <p class="tq">A seamless robust experience; they will raise demand and lift sales.</p>
  <div class="tmeta">Client</div></div></div>
<div class="certs"><div class="c-h">Credibility &amp; certifications</div></div>
</main>
<footer class="foot">
  <p class="assembled">Prepared for Aditya &middot; FamPay</p>
  <p class="meta">Leverage our seamless ecosystem. Price sensitivity is stricter.</p>
</footer>
</body></html>"""


# ── LEDE_LEAK ────────────────────────────────────────────────────────────────
class TestLedeLeak:
    def test_blank_slot_fails(self):
        """Seen live: 'This week's brief covers two signals: the in India, ...'."""
        out = vb.lede_leaks("This week's brief covers two signals: the in India, "
                            "and a deal in Poland.")
        assert out and all("LEDE LEAK" in m for m in out)

    def test_leaked_instruction_fails(self):
        out = vb.lede_leaks("Two signals this week, each a short bolded label phrase "
                            "in optionally followed by a clause.")
        assert any("bolded label phrase" in m for m in out)
        assert any("optionally followed" in m for m in out)

    def test_json_instruction_leaks(self):
        assert vb.lede_leaks("Return a JSON object with the two signals.")
        assert vb.lede_leaks("Write two clauses, as indicated by the trigger.")

    def test_broken_clause_fails(self):
        """'a competitor move by in', 'where is active' -- a clause cut after its
        preposition."""
        assert vb.lede_leaks("The week covers a competitor move by in the market.")
        assert vb.lede_leaks("A customer where is active in Poland.")

    def test_clean_lede_passes(self):
        assert vb.lede_leaks(
            "This week's brief covers two signals: a potential customer in Poland "
            "and the industry outlook on retail parks.") == []

    def test_empty_bold_tag_fails(self):
        assert vb.lede_leaks("Two signals this week ", "<b></b> the first one.")


# ── OVERCLAIM ───────────────────────────────────────────────────────────────
class TestOverclaim:
    def test_inferred_demand_is_warned(self):
        out = vb.overclaim_warnings(
            "The FSSAI limits will raise testing demand for LC-MS/MS.")
        assert out and "OVERCLAIM" in out[0]

    def test_each_phrase_reported_once(self):
        text = ("Limits will raise testing demand. Limits will raise testing demand. "
                "Stricter rules also bring penalties.")
        out = vb.overclaim_warnings(text)
        assert sum("will raise/increase demand" in m for m in out) == 1
        assert sum("stricter" in m for m in out) == 1
        assert sum("penalt" in m for m in out) == 1

    def test_announced_fact_passes(self):
        assert vb.overclaim_warnings(
            "FSSAI published final residue limits for spices, effective July.") == []


# ── BANNED_WORD ─────────────────────────────────────────────────────────────
class TestBannedWord:
    def test_jargon_is_warned(self):
        out = vb.banned_word_warnings("India's deep-tech ecosystem is growing.")
        assert out and "ecosystem" in out[0]

    def test_whole_words_only(self):
        """'ecosystems'/'leveraged' are not the banned token itself."""
        assert vb.banned_word_warnings("These ecosystems and leveraged bets.") == []

    def test_clean_copy_passes(self):
        assert vb.banned_word_warnings(
            "Two payment processors added instant settlement in Poland.") == []


# ── DUPLICATE_SENTENCE ──────────────────────────────────────────────────────
class TestDuplicateSentence:
    def test_repeated_sentence_is_warned(self):
        s = "Retail parks in Poland added new tenants this quarter across the region."
        out = vb.duplicate_sentence_warnings(f"{s} {s}")
        assert out and "DUPLICATE SENTENCE" in out[0]

    def test_short_repeats_pass(self):
        assert vb.duplicate_sentence_warnings("Deal. Deal. Deal.") == []
        assert vb.duplicate_sentence_warnings(
            "Retail parks in Poland added tenants. Retail banks in Poland cut fees.") == []


# ── COMPANY_CASE ────────────────────────────────────────────────────────────
class TestCompanyCase:
    def test_lowercased_company_is_warned(self):
        out = vb.company_case_warnings("For waters, the picture is clear.", "Waters")
        assert out and "COMPANY CASE" in out[0]

    def test_correct_casing_passes(self):
        assert vb.company_case_warnings("For Waters, the picture is clear.", "Waters") == []

    def test_all_caps_or_all_lower_names_are_exempt(self):
        assert vb.company_case_warnings("bmw expanded.", "BMW") == []
        assert vb.company_case_warnings("acme expanded.", "acme") == []


# ── section aliases ─────────────────────────────────────────────────────────
class TestSectionAliases:
    @pytest.mark.parametrize("label", ["Potential Customer", "Potential Deal",
                                       "Potential Partnership"])
    def test_alias_is_accepted_as_section_one(self, label):
        assert label in vb.SECTION_ALIASES["Potential Customer"]
        assert label in vb.CUSTOMER_LABELS

    @pytest.mark.parametrize("label", ["Potential Deal", "Potential Partnership"])
    def test_alias_card_is_found_and_named(self, label):
        text = f"● {label} · chemicals\nAcme Labs Pvt\nwhy it matters"
        assert vb.section_company(text, vb.CUSTOMER_LABELS) == "Acme Labs Pvt"

    def test_alias_brief_has_no_missing_section(self):
        html = _brief(customer_kind="industrial chemicals").replace(
            "Potential Customer", "Potential Deal")
        fails, warns = vb.check_html(html, _profile())
        assert not any("MISSING SECTION" in f for f in fails)
        assert not any("MISSING SECTION" in w for w in warns)

    def test_partnership_alias_accepted(self):
        html = _brief(customer="University of Warsaw").replace(
            "Potential Customer", "Potential Partnership")
        fails, warns = vb.check_html(html, _profile())
        assert not any("MISSING SECTION" in f for f in fails)
        assert not any("MISSING SECTION" in w for w in warns)


# ── shared non-buyer list ───────────────────────────────────────────────────
class TestNonBuyerTerms:
    def test_union_is_shared(self):
        terms = config.NON_BUYER_TERMS
        for t in ("regulator", "regulatory authority", "ministry", "government",
                  "city of", "municipal", "municipality", "city council", "cdsco",
                  "dcgi", "fda", "court", "tribunal", "parliament", "association",
                  "council", "committee", "commission", "board of", "department of",
                  "task force", "research institute", "journal"):
            assert t in terms

    def test_government_body_in_customer_slot_fails(self):
        """'Ottawa City Government' is a signal, not a buyer."""
        html = _brief(customer="Ottawa City Government", customer_kind="municipal")
        fails, _ = vb.check_html(html, _profile())
        assert any("ROLE ERROR" in f and "Ottawa City Government" in f for f in fails)

    def test_real_company_in_customer_slot_passes(self):
        fails, _ = vb.check_html(_brief(customer="Razorpay"), _profile())
        assert not any("ROLE ERROR" in f and "Razorpay" in f for f in fails)


# ── same-category heuristic (profile-driven, not pharma-only) ───────────────
class TestSameCategoryHeuristic:
    def _retail_html(self):
        return _brief(
            customer="Retail parks operator",
            customer_kind="retail parks",
            body="Retail parks and the retail market expanded, said the operator.",
            bullet="The retail market added outlets.")

    def test_non_pharma_buyer_industry_suppresses_warning(self):
        """The old list was hospital/clinic-only, so a retail buyer of a retail
        researcher looked like the seller's own trade."""
        prof = _profile(industries=["Retail parks"],
                        buyer_functions=["retail park operators"],
                        value_prop="market reports on retail parks and shopping centres",
                        competitors=["Slice"])
        _, warns = vb.check_html(self._retail_html(), prof)
        assert not any("reads like a company" in w for w in warns)

    def test_profile_without_new_fields_does_not_crash(self):
        import types

        prof = types.SimpleNamespace(
            company="Widget Co", short="Widget", recipient="there", tagline="t",
            value_prop="market reports on retail parks and shopping centres",
            industries=[], competitors=[], market="Global")
        fails, warns = vb.check_html(self._retail_html(), prof)
        assert isinstance(fails, list) and isinstance(warns, list)

    def test_seller_category_word_still_warned(self):
        prof = _profile(industries=["Retail parks"],
                        value_prop="market reports on retail parks and shopping centres")
        html = _brief(customer="Market Insights Weekly", customer_kind="research",
                      body="Reports and market studies for the wider market.",
                      bullet="Reports and market studies for buyers.",
                      story_body="Regulators published a consultation on settlement "
                                 "rules for the year ahead.")
        _, warns = vb.check_html(html, prof)
        assert any("reads like a company" in w for w in warns)


# ── wiring + main-content scoping ───────────────────────────────────────────
class TestWiring:
    def test_lede_leak_is_a_fail(self):
        html = _brief(lede="This week's brief covers two signals: the in India, "
                           "and a deal in Poland.")
        fails, _ = vb.check_html(html, _profile())
        assert any("LEDE LEAK" in f for f in fails)

    def test_overclaim_and_jargon_are_warns(self):
        html = _brief(body="The FSSAI limits will raise testing demand, a seamless "
                           "ecosystem change.",
                      bullet="India's deep-tech ecosystem matters.")
        fails, warns = vb.check_html(html, _profile())
        assert any("OVERCLAIM" in w for w in warns)
        assert any("BANNED WORD" in w for w in warns)
        assert not any("OVERCLAIM" in f for f in fails)
        assert not any("BANNED WORD" in f for f in fails)

    def test_template_blocks_are_never_flagged(self):
        """Testimonials, the caps/projects showcase and the footer are fixed copy:
        jargon there is not a defect and must not be reported."""
        fails, warns = vb.check_html(_brief(), _profile())
        assert not any("BANNED WORD" in m for m in fails + warns)
        assert not any("OVERCLAIM" in m for m in fails + warns)
        assert not any("COMPANY CASE" in m for m in fails + warns)
        assert not any("DUPLICATE SENTENCE" in m for m in fails + warns)

    def test_banner_line_is_excluded(self):
        html = _brief(tail='<p class="agentline">Everything below comes from '
                           '<b>CMI\'s seamless ecosystem</b>.</p>')
        _, warns = vb.check_html(html, _profile())
        assert not any("BANNED WORD" in w for w in warns)

    def test_main_content_text_drops_static_tail(self):
        text = vb.main_content_text(_brief())
        assert "Razorpay" in text
        assert "Inside Coherent Market Insights" not in text
        assert "seamless robust experience" not in text
        assert "Credibility" not in text

    def test_duplicate_sentence_between_card_and_bullet_is_warned(self):
        sentence = ("Cash reserves comfortably cover the announced build-out across "
                    "the region this quarter.")
        html = _brief(body=sentence, bullet=sentence)
        _, warns = vb.check_html(html, _profile())
        assert any("DUPLICATE SENTENCE" in w for w in warns)

    def test_company_case_wired_from_profile(self):
        html = _brief(body="For fampay, the picture is clear on prepaid cards.",
                      bullet="Prepaid cards for teenagers remain the core offer.")
        _, warns = vb.check_html(html, _profile())
        assert any("COMPANY CASE" in w and "FamPay" in w for w in warns)
