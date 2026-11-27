"""Content-quality fixes: product-term matching, competitive-move check, headline-name
guard, section heading by lead type, who-to-target fallback, lede repair, evidence rules.
All LLM calls are mocked."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import brief_builder as bb
import compose
import voice
from collect import Article


def _profile(**kw):
    base = dict(company="Waters", short="Waters", value_prop="liquid chromatography and mass spectrometry.",
                industries=["Pharma"], competitors=["Thermo Fisher"], market="India",
                lead_type="seller", buyer_functions=[], recipient="Sam",
                products=[{"name": n, "application": "testing"} for n in
                          ("UPLC-MS/MS", "HPLC systems", "LC-MS", "UHPLC",
                           "Empower chromatography software", "columns")])
    base.update(kw)
    return SimpleNamespace(**base)


def _art(title, summary=""):
    return Article(title=title, url="http://x", source="S", published=datetime.now(timezone.utc),
                   summary=summary, query="q", bucket="competitor", fingerprint=title)


LUCID = ("Quantum BioPharma signs study start-up agreement with Thermo Fisher's PPD "
         "for Phase 2 Lucid-MS trial in multiple sclerosis")


def _on_cat(profile, art):
    terms = bb._product_terms(profile)
    return bool(bb._term_hits(terms, f"{art.title} {art.summary}".lower()))


def test_lucid_ms_rejected_uhplc_accepted():
    p = _profile()
    assert not _on_cat(p, _art(LUCID))
    assert _on_cat(p, _art("Thermo Fisher launches new Vanquish UHPLC system"))


def test_short_term_needs_second_term():
    terms = ["ms", "lc-ms"]
    assert bb._term_hits(terms, "a lucid-ms trial") == 0
    assert bb._term_hits(["ms", "lc"], "ms and lc news") == 2


def _comp_args():
    a = _art("Thermo Fisher launches new Vanquish UHPLC system")
    return [a], [], _profile()


def test_check_false_skips_candidate():
    arts, peers, p = _comp_args()
    with mock.patch.object(bb.llm, "chat_json", return_value={"real_competitive_move": False}) as cj, \
            mock.patch.object(bb, "_write_card") as wc, \
            mock.patch.object(bb.free_research, "enrich_company", return_value={}):
        assert bb.build_competition(arts, peers, p, log=lambda *_: None) == (None, None)
    wc.assert_not_called()
    assert cj.call_args.kwargs["max_tokens"] == 40
    assert "Thermo Fisher" in cj.call_args.args[0] and "<" not in cj.call_args.args[1].split("Return")[0]


def test_check_true_proceeds():
    arts, peers, p = _comp_args()
    card = {"company_name": "Thermo Fisher"}
    with mock.patch.object(bb.llm, "chat_json", return_value={"real_competitive_move": True}), \
            mock.patch.object(bb, "_write_card", return_value=card) as wc, \
            mock.patch.object(bb.free_research, "enrich_company", return_value={}):
        out, art = bb.build_competition(arts, peers, p, log=lambda *_: None)
    assert out is card and wc.called


def test_headline_name_guard():
    for bad in ("Neogen raises Rs.600 crore", "Acme signs $5 deal", "Foo opens plant", "Bar Inc ₹500",
                "A B C D E F G", "Trailing,"):
        assert bb._looks_like_headline(bad), bad
    for ok in ("Dr. Reddy's Laboratories", "Waters Corporation", "Thermo Fisher"):
        assert not bb._looks_like_headline(ok), ok


def test_prospect_heading():
    h = compose.prospect_heading
    assert h(_profile()) == "Potential Customer"
    assert h(SimpleNamespace()) == "Potential Customer"
    assert h(_profile(lead_type="investor")) == "Potential Deal"
    for lt in ("nonprofit", "education", "government"):
        assert h(_profile(lead_type=lt)) == "Potential Partnership"


def test_who_to_target_fallback():
    assert bb._who_fallback(_profile(buyer_functions=["QC labs", "CROs", "Pharma R&D", "x"])) == \
        "QC labs, CROs, Pharma R&D"
    assert "Corporate Development" in bb._who_fallback(_profile())


def _sections():
    return [{"label": "Potential Customer", "title": "Acme Labs", "body_html": "b"},
            {"label": "Competition Outlook", "title": "Thermo Fisher", "body_html": "b"}]


def test_evidence_discipline_in_both_styles():
    assert "EVIDENCE DISCIPLINE" in voice.STYLE
    assert "could" in voice.STYLE_BRIEF and "Never invent a number" in voice.STYLE_BRIEF
    assert "something it can act on" not in voice.BULLET_SYSTEM


def test_seller_context_lines():
    txt = voice.seller_context(_profile(buyer_functions=["QC labs"]))
    assert "The seller's products: UPLC-MS/MS — testing" in txt
    assert "Country: India" in txt and "QC labs" not in txt
    assert "Who usually buys them: QC labs." in voice.seller_context(
        _profile(buyer_functions=["QC labs"]), buyers=True)
    assert voice.seller_context(SimpleNamespace()) == ""
