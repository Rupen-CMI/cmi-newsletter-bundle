"""Round 3: seller-identity check, one shared prose guard, no-source-reference prompt
lines, category-search articles reaching the competition check. LLM mocked."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import pytest

import brief_builder as bb
import compose
import voice
from collect import Article

OVER = "The launch will increase demand for new systems."
CLEAN = "The launch could add demand for new systems."


def _profile(**kw):
    base = dict(company="Ottawa Humane Society", short="OHS", value_prop="animal care.",
                industries=["Animal welfare"], competitors=["Acme Shelter"], market="Canada",
                lead_type="seller", buyer_functions=[], recipient="Sam", products=[])
    base.update(kw)
    return SimpleNamespace(**base)


def _art(title, composite=1.0, bucket="competitor"):
    a = Article(title=title, url="http://x", source="S", published=datetime.now(timezone.utc),
                summary="", query="q", bucket=bucket, fingerprint=title)
    a.composite = composite
    return a


# ── is_self ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", [
    "Ottawa Humane Society", "OHS", "OHS's", "Ottawa Humane Society's new clinic",
    "ottawa humane society", "Ottawa Humane", "The OHS"])
def test_is_self_true(name):
    assert bb.is_self(name, _profile())


@pytest.mark.parametrize("name", ["", "Acme Shelter", "Toronto Humane Society", "Ohio Shelter"])
def test_is_self_false(name):
    assert not bb.is_self(name, _profile())


def test_self_never_customer_or_peer():
    arts = [_art("OHS opens clinic", bucket="trigger"), _art("Acme Shelter raises", bucket="trigger")]
    reply = {"items": [{"index": 0, "company_name": "Ottawa Humane Society's", "relationship": "buyer"},
                       {"index": 1, "company_name": "Newco", "relationship": "peer"}]}
    with mock.patch.object(bb.llm, "chat_json", return_value=reply):
        rel, _ = bb.triage_triggers(arts, _profile())
    assert rel[0][1] == "self" and rel[1][1] == "peer"
    with mock.patch.object(bb.llm, "chat_json", return_value=reply), \
            mock.patch.object(bb, "_write_card", return_value={"company_name": "x"}), \
            mock.patch.object(bb.free_research, "enrich_company", return_value={}):
        card, art, peers = bb.build_customer(arts, _profile(), log=lambda *_: None)
    assert card is None or art.title != "OHS opens clinic"
    assert [n for _, n in peers] == ["Newco"]


def _run_comp(arts, peers, triage_reply, profile=None):
    judged = []

    def fake_check(a, comp, prof):
        judged.append((a.title, comp))
        return True

    with mock.patch.object(bb, "_competitive_move_check", side_effect=fake_check), \
            mock.patch.object(bb, "_write_card", side_effect=lambda a, n, *r: {"company_name": n}), \
            mock.patch.object(bb.free_research, "enrich_company", return_value={}), \
            mock.patch.object(bb.llm, "chat_json", return_value=triage_reply) as cj:
        out = bb.build_competition(arts, peers, profile or _profile(), log=lambda *_: None)
    return out, judged, cj


def test_self_peer_and_named_never_competitor():
    peers = [(_art("OHS opens clinic"), "OHS's")]
    out, judged, _ = _run_comp([], peers, {})
    assert out == (None, None) and judged == []
    # seller listed as its own competitor
    p = _profile(competitors=["Ottawa Humane Society"])
    out, judged, _ = _run_comp([_art("Ottawa Humane Society opens clinic")], [], {}, p)
    assert judged == []


def test_unattributed_competitor_article_reaches_check_only_as_labelled_peer():
    # Category-search articles are labelled once in build_customer's pool; build_competition
    # gets the peers and makes no triage call of its own.
    a = _art("Newco shelters expand", 2.0)
    out, judged, _ = _run_comp([a, _art("Some roundup", 1.0)], [], {"items": []})
    assert judged == [] and out == (None, None)
    out, judged, _ = _run_comp([a], [(a, "Newco Shelters"), (a, "Ministry of Health")], {"items": []})
    assert judged == [("Newco shelters expand", "Newco Shelters")]
    assert out[0]["company_name"] == "Newco Shelters"


def test_unattributed_self_and_non_peer_excluded():
    arts = [_art("OHS opens new clinic", 3.0), _art("Buyer news", 2.0)]
    reply = {"items": [{"index": 0, "company_name": "Ottawa Humane Society", "relationship": "peer"},
                       {"index": 1, "company_name": "Buyer Co", "relationship": "buyer"}]}
    out, judged, _ = _run_comp(arts, [], reply)
    assert out == (None, None) and judged == []


# ── shared prose guard ────────────────────────────────────────────────────────
def test_guard_on_bullets_keeps_clean_retry():
    replies = [{"bullets": [f"<b>x.</b> {OVER}"] * 3}, {"bullets": [f"<b>x.</b> {CLEAN}"] * 3}]
    with mock.patch.object(voice, "_flat_text", wraps=voice._flat_text), \
            mock.patch("llm.chat_json", side_effect=replies) as cj:
        out = compose.analyst_bullets(_profile(), "sig")
    assert cj.call_count == 2 and all("could add" in b for b in out)


def test_guard_on_multi_bullets():
    bad = {"items": [{"index": 0, "bullets": [OVER] * 3}, {"index": 1, "bullets": [CLEAN] * 3}]}
    good = {"items": [{"index": 0, "bullets": [CLEAN] * 3}, {"index": 1, "bullets": [CLEAN] * 3}]}
    with mock.patch("llm.chat_json", side_effect=[bad, good]) as cj:
        out = compose.analyst_bullets_multi(_profile(), ["a", "b"])
    assert cj.call_count == 2 and "could add" in out[0][0]


def test_guard_on_editorial_top3_and_banned_words():
    secs = [{"label": "Industry Outlook", "title": "T", "body_html": "b"}]
    bad = {"themes": ["x y"], "top3": ["<b>Shift.</b> The landscape will increase demand."]}
    good = {"themes": ["x y"], "top3": ["<b>Shift.</b> The market could add demand."]}
    with mock.patch("llm.chat_json", side_effect=[bad, good]) as cj:
        _, top3 = compose.editorial(_profile(), secs)
    assert cj.call_count == 2 and "landscape" not in top3[0]["html"]


def test_guard_flags_banned_after_masking_brand():
    prof = _profile(products=[{"name": "Empower"}])
    assert not voice.prose_flags({"a": "Empower 3 software ships."}, prof)
    assert voice.prose_flags({"a": "a robust plan"}, prof)


def test_guard_keeps_original_when_retry_empty():
    with mock.patch("llm.chat_json", side_effect=[{"a": OVER}, {}]):
        assert voice.chat_json_guarded("s", "u", max_tokens=5) == {"a": OVER}


def test_guard_uses_meta_check_when_available():
    import validate_brief
    fake = lambda t: ["META: ('the signal')"] if "the signal" in t.lower() else []
    with mock.patch.object(validate_brief, "meta_commentary_warnings", fake, create=True), \
            mock.patch("llm.chat_json", side_effect=[{"a": "The signal is a fit."}, {"a": "It fits."}]) as cj:
        out = voice.chat_json_guarded("s", "u", max_tokens=5)
    assert cj.call_count == 2 and out == {"a": "It fits."}


# ── prompts ───────────────────────────────────────────────────────────────────
def test_style_brief_has_banned_and_no_source_reference():
    assert 'Never refer to "the signal"' in voice.STYLE
    assert 'Never mention "the signal"' in voice.STYLE_BRIEF
    assert "Do not use:" in voice.STYLE_BRIEF and "landscape" in voice.STYLE_BRIEF \
        and "leverage" in voice.STYLE_BRIEF
