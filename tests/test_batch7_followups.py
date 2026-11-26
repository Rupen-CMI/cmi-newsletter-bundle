"""Follow-ups to the Test15 run: batched buyer triage, untyped triggers, English fallback,
orphan sentences after a drop, and validator name/domain false alarms."""
from datetime import datetime, timezone
from types import SimpleNamespace

import brief_builder
import free_research
import llm
import voice
from collect import Article

NOW = datetime.now(timezone.utc)


def _art(title, **kw):
    return Article(title=title, url="https://x.example/" + title[:8].replace(" ", ""),
                   source="Wire", published=NOW, summary="", bucket="trigger", **kw)


def _prof(**kw):
    return SimpleNamespace(company="GrowCo", short="GrowCo", market="United Kingdom",
                           tagline="t", recipient="R", value_prop="greenhouse consulting",
                           industries=["Growers"], competitors=[], **kw)


def test_triage_continues_past_first_ten(monkeypatch):
    arts = [_art(f"Government grant scheme {i}") for i in range(25)]
    buyer = arts[17]
    buyer.title = "Tomato grower expands into cucumbers in the UK"
    calls = []

    def triage(cands, profile):
        batch = cands[:10]
        calls.append(len(batch))
        rel = {}
        for i, a in enumerate(batch):
            if a is buyer:
                a.would_buy, a.event_country = "glasshouse design", "United Kingdom"
                rel[i] = ("Tomato grower", "buyer")
            else:
                rel[i] = ("", "other")
        return rel, batch

    monkeypatch.setattr(brief_builder, "triage_triggers", triage)
    monkeypatch.setattr(brief_builder, "_rank_by_evidence", lambda t: t)
    monkeypatch.setattr(free_research, "enrich_company", lambda *a, **k: {})
    monkeypatch.setattr(brief_builder, "_write_card", lambda *a, **k: {"company": a[1]})
    card, art, _ = brief_builder.build_customer(arts, _prof(), log=lambda *a: None)
    assert art is buyer and card["company"] == "Tomato grower"
    assert calls == [10, 10]          # stopped once the buyer was found


def test_untyped_trigger_reaches_customer(monkeypatch):
    a = _art("Régénéterre s'allie à Eiffage pour produire de la terre végétale")
    a.trigger_type = ""
    seen = []
    monkeypatch.setattr(brief_builder, "build_customer",
                        lambda t, p, log=print: (seen.extend(t), (None, None, []))[1])
    monkeypatch.setattr(brief_builder, "build_competition", lambda *a, **k: (None, None))
    monkeypatch.setattr(brief_builder, "triage_and_synthesize_industry", lambda *a, **k: [])
    brief_builder.build_brief([a], _prof(), log=lambda *a: None)
    assert seen == [a]


def test_french_reply_is_translated(monkeypatch):
    french = {"headline": "Haies et bocage", "body_html":
              "Les haies et le bocage sont de plus en plus reconnus pour les sols agricoles "
              "dans le pays. Cela pourrait ouvrir de nouvelles opportunités pour les entreprises."}
    english = {"headline": "Hedges and bocage", "body_html":
               "French farms increasingly value hedges and bocage for their soil and their "
               "wildlife. Terra Innova could supply topsoil for these new hedgerow projects."}
    replies = iter([french, french, english])
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: next(replies))
    out = voice.chat_json_guarded("sys", "SIGNAL: Les haies et le bocage. Terra Innova",
                                  max_tokens=200)
    assert out["headline"] == "Hedges and bocage"


def test_bad_translation_is_not_used(monkeypatch):
    french = {"body_html": "Les haies et le bocage sont de plus en plus reconnus pour les sols "
                           "agricoles dans le pays. Cela pourrait ouvrir des opportunités."}
    replies = iter([french, french, {"other_key": "x"}])
    monkeypatch.setattr(llm, "chat_json", lambda *a, **k: next(replies))
    out = voice.chat_json_guarded("sys", "SIGNAL: Les haies", max_tokens=200)
    assert out == french


def test_orphan_sentence_dropped_with_its_parent():
    src = "Tokyo Tatemono invests 46.3 billion in offices."
    text = ("Tokyo Tatemono invests 46.3 billion yen in offices. That shift could lift demand. "
            "Tokyo Tatemono may need finance.")
    assert voice._drop_unsupported_sentences(text, src) == "Tokyo Tatemono may need finance."


def test_pointing_sentence_kept_when_parent_kept():
    src = "Tokyo Tatemono invests 46.3 billion in offices."
    text = "Tokyo Tatemono invests 46.3 billion in offices. That shift could lift demand."
    assert voice._drop_unsupported_sentences(text, src) == text


def _check(tmp_path, monkeypatch, body, company, topic, email=""):
    import validate_brief
    import tenant
    html = tmp_path / "b.html"
    html.write_text(f"<html><body><main class='body'><p>{body}</p></main></body></html>",
                    encoding="utf-8")
    prof = tenant.Profile(company=company, short=company, value_prop="hedge soil supply",
                          industries=["Farmers"], competitors=[], market="France", recipient="R",
                          tagline="t", industry_keywords=[], trigger_queries=[],
                          industry_queries=[])
    fails, _, _, _ = validate_brief.check_detailed(str(html), prof, email=email,
                                                   kind="signal", topic=topic)
    return fails


def test_squashed_lead_name_is_not_topic_error(tmp_path, monkeypatch):
    for topic, company in (("Vdlhapro", "VDL Hapro"), ("Pgigrup", "PGI Engineering"),
                           ("Bluematterconsulting", "Blue Matter")):
        fails = _check(tmp_path, monkeypatch, f"{company} sells hedge soil supply.", company, topic)
        assert not any(f.startswith("TOPIC ERROR") for f in fails), (topic, fails)


def test_unrelated_company_still_topic_error(tmp_path, monkeypatch):
    fails = _check(tmp_path, monkeypatch, "Coherent Market Insights sells hedge soil supply.",
                   "Coherent Market Insights", "Shipbreakers")
    assert any(f.startswith("TOPIC ERROR") for f in fails)
