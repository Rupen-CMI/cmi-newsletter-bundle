"""Batch 1 writer copy: English-only guard, lede wording, no other-event headlines in the
card writer, article text reaching the Analyst Opinion call. All LLM calls are mocked."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import brief_builder as bb
import compose
import free_research
import voice
from collect import Article


def _profile():
    return SimpleNamespace(company="Waters", short="Waters", value_prop="liquid chromatography.",
                           industries=["Pharma"], competitors=["Thermo Fisher"], market="India",
                           lead_type="seller", buyer_functions=["QC labs"], recipient="Sam",
                           products=[{"name": "HPLC systems", "application": "testing"}])


def test_non_english_phrases():
    assert voice.non_english_phrases("Les entreprises des pays sont dans une phase est pour avec") != []
    assert voice.non_english_phrases("Das ist eine Studie und der Markt ist nicht klein mit der Branche") != []
    assert voice.non_english_phrases("東京の会社が新しい工場を建設すると発表しました") == ["non-English script"]
    assert voice.non_english_phrases("Danone and Société Générale signed a supply deal in Paris.") == []


def test_prose_flags_language():
    kinds = [k for k, _ in voice.prose_flags({"t": "Les ventes des produits sont pour une usine dans le sud"})]
    assert "language" in kinds


def test_lede_generic_label():
    secs = [{"label": "Wireless Access Point Market", "title": "Wireless Access Point Market", "body_html": "b"}]
    lede = compose._build_lede(_profile(), secs, [], "This week's brief covers one development:")
    assert "market development</b>" in lede and "signal" not in lede.lower()


def test_background_block_omits_headlines():
    enr = {"wikipedia": "Acme is a firm.", "financials": "Revenue 1bn", "recent_news": ["Other event"]}
    assert "Other event" not in free_research.background_block(enr)
    assert "WIKIPEDIA" in free_research.background_block(enr)
    assert "RECENT COMPANY HEADLINES" in free_research.context_block(enr)


@pytest.mark.parametrize("kind", ["customer", "competitor"])
def test_write_card_prompt_and_source_text(monkeypatch, kind):
    seen = {}

    def fake(system, user, **kw):
        seen["p"] = system + user
        return {}
    monkeypatch.setattr(bb, "_chat_guarded", fake)
    a = Article(title="Acme buys Beta", url="http://x", source="S",
                published=datetime.now(timezone.utc), summary="Deal worth 46.3 billion.")
    enr = {"wikipedia": "Acme is a firm.", "recent_news": ["Morrison stake"]}
    card = bb._write_card(a, "Acme", kind, enr, _profile())
    assert "RECENT COMPANY HEADLINES" not in seen["p"] and "Morrison" not in seen["p"]
    assert "events that are not in the SIGNAL" in seen["p"] and "Use FREE RESEARCH only to say who" in seen["p"]
    assert card["_source_text"].startswith("Acme buys Beta. Deal worth")


def test_analyst_signals_include_article(monkeypatch):
    got = {}

    class Stop(Exception):
        pass

    def fake(profile, signals):
        got["s"] = signals
        raise Stop
    monkeypatch.setattr(compose, "analyst_bullets_multi", fake)
    brief = {"industry_outlook": [{"headline": "H", "body_html": "<b>Body</b>", "source_url": "u",
                                   "source_name": "n", "_source_text": "Orig title. Orig 46.3 billion"}]}
    with pytest.raises(Stop):
        compose.build_context(_profile(), brief, "signal", log=lambda *_: None)
    assert got["s"][0].startswith("ARTICLE: Orig title. Orig 46.3 billion\nSECTION TEXT: H")
