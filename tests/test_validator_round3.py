"""Round-3 validator work: sentence-level over-claim judgement and the new
meta-commentary check.

`overclaim_warnings` and `meta_commentary_warnings` are pure text -> list[str],
so they are tested directly; the wiring tests then confirm `check_detailed()`
(the code path behind `check_html`) reports them as warns on the model-written
main content only.

Run with:
    venv/Scripts/python.exe -m pytest tests/test_validator_round3.py -q -p no:cacheprovider
"""
from __future__ import annotations

import tenant
import validate_brief as vb


# ── fixtures ─────────────────────────────────────────────────────────────────
_BODY = ("Regulators published a consultation on residue limits, with the trade "
         "body pointing to sustained demand from domestic buyers and a broader "
         "order book than the prior year period across the region.")
_BODY_2 = ("A rival opened two new offices in the region and said it would hire "
           "additional staff to serve existing clients through the coming quarter.")
_BODY_3 = ("The company is expanding capacity across the region this quarter and "
           "has signalled further investment in its testing laboratories.")


def _profile():
    return tenant.Profile(
        company="Waters", short="Waters", recipient="Sam",
        tagline="analytical instruments",
        value_prop="liquid chromatography.", industries=["Pharma"],
        competitors=["Thermo Fisher"], industry_keywords={}, trigger_queries=[],
        industry_queries=[], market="India", lead_type="seller",
        buyer_functions=["QC labs"],
        products=[{"name": "HPLC systems", "application": "testing"}])


def _brief(*, lede="This week's brief covers two signals: a potential customer in "
                   "Poland and the industry outlook on retail parks.",
           body=_BODY, bullet="Cash reserves cover the announced build-out.",
           story_body=_BODY_2, competition_body=_BODY_3):
    """A realistic slice of the template around the model-written parts."""
    return f"""<html><body>
<header class="masthead"><div class="brandrow">Masthead copy</div></header>
<main class="body">
<h1 class="hello">Hello, Sam.</h1>
<p class="lede">{lede}</p>
<div class="top3"><div class="row"><div class="idx">1</div>
 <div class="txt"><span class="cat">Customer move</span>Short top3 line.</div></div></div>
<div class="prospect">
  <span class="flag">&#9679; Potential Customer &middot; pharma</span>
  <h2>A customer raises funding</h2>
  <p>{body}</p>
</div>
<div class="analyst"><div class="tag">Analyst Opinion</div>
 <ul><li>{bullet}</li></ul>
 <div class="prov"><span class="src">CMI Intelligence</span></div></div>
<div class="story">
  <span class="flag">&#9679; Industry Outlook</span>
  <h3>Testing outlook</h3>
  <p>{story_body}</p>
</div>
<div class="prospect">
  <span class="flag">&#9679; Competition Outlook &middot; pharma</span>
  <h2>A rival expands</h2>
  <p>{competition_body}</p>
</div>
<div class="eyebrow"><span class="lbl">Inside Coherent Market Insights</span></div>
<div class="caps">
  <p class="capsub">No figures in the signal, so treat the risk as a scenario.</p>
</div>
<footer class="foot">Unsubscribe</footer>
</main></body></html>"""


# ── OVERCLAIM: sentence-level hedging ───────────────────────────────────────
class TestOverclaimHedges:
    def test_hedged_sentence_is_not_flagged(self):
        assert vb.overclaim_warnings(
            "Waters may see more inquiries from food labs.") == []

    def test_unhedged_price_sensitivity_is_flagged(self):
        out = vb.overclaim_warnings("Price sensitivity will rise as inflation bites.")
        assert out and out[0].startswith("OVERCLAIM:")
        assert "price sensitivity" in out[0]

    def test_unhedged_demand_claim_is_flagged(self):
        out = vb.overclaim_warnings(
            "The expansion will increase demand for graduates.")
        assert out and "will raise/increase demand" in out[0]

    def test_could_hedge_passes(self):
        assert vb.overclaim_warnings("It could lift demand for testing.") == []

    def test_each_hedge_word_qualifies_a_sentence(self):
        for hedge in ("could", "may", "might", "can", "possibly", "potentially",
                      "likely"):
            text = f"Stricter limits {hedge} raise testing demand."
            assert vb.overclaim_warnings(text) == [], hedge

    def test_hedge_in_another_sentence_does_not_save_the_claim(self):
        out = vb.overclaim_warnings(
            "The limits could tighten next year. Price sensitivity will rise as "
            "inflation bites.")
        assert len(out) == 1 and "price sensitivity" in out[0]

    def test_hedged_sentence_does_not_hide_an_unhedged_one(self):
        out = vb.overclaim_warnings(
            "Waters may see more inquiries from food labs. Stricter limits bring "
            "penalties.")
        assert len(out) == 2
        assert any("more inquiries" in w for w in out) is False
        assert any("stricter" in w for w in out)

    def test_state_as_fact_still_flagged(self):
        out = vb.overclaim_warnings(
            "The FSSAI limits will raise testing demand for LC-MS/MS.")
        assert out and "OVERCLAIM" in out[0]

    def test_each_pattern_reported_once(self):
        text = ("Price sensitivity will rise. Price sensitivity will rise again. "
                "Stricter rules also bring penalties.")
        out = vb.overclaim_warnings(text)
        assert sum("price sensitivity" in m for m in out) == 1
        assert sum("stricter" in m for m in out) == 1
        assert sum("penalt" in m for m in out) == 1

    def test_message_format_is_unchanged(self):
        out = vb.overclaim_warnings("The launch will increase demand for new systems.")
        assert len(out) == 1
        assert out[0].startswith("OVERCLAIM: the copy uses ")
        assert "'will increase demand'" in out[0]
        assert "'could'/'may'" in out[0]

    def test_announced_fact_passes(self):
        assert vb.overclaim_warnings(
            "FSSAI published final residue limits for spices, effective July.") == []

    def test_empty_text_passes(self):
        assert vb.overclaim_warnings("") == []
        assert vb.overclaim_warnings(None) == []


# ── META COMMENTARY ─────────────────────────────────────────────────────────
class TestMetaCommentary:
    def test_no_figures_plus_scenario_is_flagged(self):
        out = vb.meta_commentary_warnings(
            "No figures in the signal, so treat the risk as a scenario, not a "
            "forecast.")
        assert out and out[0].startswith("META COMMENTARY:")

    def test_sentence_reported_once_even_with_several_hits(self):
        out = vb.meta_commentary_warnings(
            "No figures in the signal, so treat the risk as a scenario.")
        assert len(out) == 1

    def test_the_signal_is_flagged(self):
        out = vb.meta_commentary_warnings(
            "The signal is a direct fit for its products.")
        assert out and out[0].startswith("META COMMENTARY:")
        assert "the signal" in out[0].lower()

    def test_fixed_lede_opener_is_not_flagged(self):
        assert vb.meta_commentary_warnings(
            "This week's brief covers two signals: a potential customer in Poland "
            "and the industry outlook on retail parks.") == []

    def test_each_pattern_is_flagged(self):
        cases = [
            "The signal names two labs.",
            "This signal is about testing capacity.",
            "No figures were released in the notice.",
            "That was not stated in the source.",
            "Treat this as a scenario for now.",
            "Demand is rising as indicated by the order book.",
            "The article does not name the buyer.",
        ]
        for text in cases:
            out = vb.meta_commentary_warnings(text)
            assert out and out[0].startswith("META COMMENTARY:"), text

    def test_indicated_by_variants(self):
        assert vb.meta_commentary_warnings(
            "Sales are rising as indicated by the order book.")
        assert vb.meta_commentary_warnings(
            "This is a testing theme as indicated by the trigger.")

    def test_market_copy_passes(self):
        assert vb.meta_commentary_warnings(
            "Waters opened a new application lab in Mumbai to support food "
            "testing customers.") == []

    def test_signals_plural_does_not_match(self):
        assert vb.meta_commentary_warnings(
            "Both signals point to stronger demand for testing capacity.") == []

    def test_case_insensitive(self):
        assert vb.meta_commentary_warnings("THE SIGNAL IS A DIRECT FIT.")

    def test_duplicate_sentence_reported_once(self):
        text = "The signal is a direct fit. The signal is a direct fit."
        assert len(vb.meta_commentary_warnings(text)) == 1

    def test_empty_text_passes(self):
        assert vb.meta_commentary_warnings("") == []
        assert vb.meta_commentary_warnings(None) == []

    def test_hedging_does_not_exempt_meta_commentary(self):
        out = vb.meta_commentary_warnings(
            "The signal may not give figures for the quarter.")
        assert out


# ── wiring into check_detailed (via check_html) ─────────────────────────────
class TestWiring:
    def test_meta_commentary_is_a_warn_not_a_fail(self):
        fails, warns = vb.check_html(
            _brief(body="The signal is a direct fit for its products."),
            _profile())
        assert any("META COMMENTARY" in w for w in warns)
        assert not any("META COMMENTARY" in f for f in fails)

    def test_meta_commentary_in_a_bullet_is_a_warn(self):
        fails, warns = vb.check_html(
            _brief(bullet="No figures in the signal, so treat the risk as a scenario."),
            _profile())
        assert any("META COMMENTARY" in w for w in warns)
        assert not any("META COMMENTARY" in f for f in fails)

    def test_overclaim_hedged_copy_is_quiet(self):
        fails, warns = vb.check_html(
            _brief(body="Waters may see more inquiries from food labs."),
            _profile())
        assert not any("OVERCLAIM" in m for m in fails + warns)

    def test_overclaim_unhedged_copy_is_a_warn(self):
        fails, warns = vb.check_html(
            _brief(body="Price sensitivity will rise as inflation bites."),
            _profile())
        assert any("OVERCLAIM" in w for w in warns)
        assert not any("OVERCLAIM" in f for f in fails)

    def test_fixed_template_tail_is_never_flagged(self):
        fails, warns = vb.check_html(_brief(), _profile())
        assert not any("META COMMENTARY" in m for m in fails + warns)
        assert not any("OVERCLAIM" in m for m in fails + warns)
