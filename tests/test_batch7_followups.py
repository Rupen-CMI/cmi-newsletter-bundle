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


def _art(title, bucket="trigger", **kw):
    return Article(title=title, url="https://x.example/" + title[:8].replace(" ", ""),
                   source="Wire", published=NOW, summary="", bucket=bucket, **kw)


def _prof(**kw):
    kw.setdefault("industries", ["Growers"])
    return SimpleNamespace(company="GrowCo", short="GrowCo", market="United Kingdom",
                           tagline="t", recipient="R", value_prop="greenhouse consulting",
                           competitors=[], **kw)


def test_model_labels_top40_in_batches_and_picks_best(monkeypatch):
    arts = [_art(f"Farm grant scheme {i}") for i in range(45)]
    arts[3].title = "Chipotle opens a store in the UK"
    arts[17].title = "Tomato grower expands into cucumbers in the UK"
    calls, offered = [], []

    def triage(cands, profile):
        batch = cands[:10]
        calls.append(len(batch))
        rel = {i: ("", "other") for i in range(len(batch))}
        for i, a in enumerate(batch):
            if "Chipotle" in a.title:
                rel[i] = ("Chipotle", "buyer")
            if "Tomato" in a.title:
                rel[i] = ("Tomato grower", "buyer")
        return rel, batch

    def pick(buyers, profile):
        offered.extend(n for _, n in buyers)
        return [n for _, n in buyers].index("Tomato grower"), "United Kingdom", "glasshouse design"

    monkeypatch.setattr(brief_builder, "triage_triggers", triage)
    monkeypatch.setattr(brief_builder, "_pick_buyer", pick)
    monkeypatch.setattr(brief_builder, "_rank_by_evidence", lambda t: t)
    monkeypatch.setattr(free_research, "enrich_company", lambda *a, **k: {})
    monkeypatch.setattr(brief_builder, "_write_card", lambda *a, **k: {"company": a[1]})
    card, art, _ = brief_builder.build_customer(arts, _prof(), log=lambda *a: None)
    assert sorted(calls) == [10, 10, 10, 10]            # top 40 only, 10 per call
    assert offered == ["Chipotle", "Tomato grower"]     # every buyer goes to the pick call
    assert art is arts[17] and card["company"] == "Tomato grower"
    assert art.would_buy == "glasshouse design"


def test_untyped_trigger_reaches_customer(monkeypatch):
    a = _art("Régénéterre s'allie à Eiffage pour produire de la terre végétale")
    a.trigger_type = ""
    seen = []
    monkeypatch.setattr(brief_builder, "build_customer",
                        lambda t, p, log=print, **k: (seen.extend(t), (None, None, []))[1])
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


def test_event_items_ranked_before_event_less_ones(monkeypatch):
    plain = _art("Driver's licence scheme for Aussies expands")
    local = _art("MGTA investit dans le broyage des déblais")
    typed = _art("BHP unveils plan to double copper output", trigger_type="expansion")
    assert brief_builder._has_event(local) and brief_builder._has_event(typed)
    assert not brief_builder._has_event(plain)
    seen = []

    def triage(cands, profile):
        seen.extend(cands[:10])
        return {}, cands[:10]

    monkeypatch.setattr(brief_builder, "triage_triggers", triage)
    monkeypatch.setattr(brief_builder, "_rank_by_evidence", lambda t: t)
    brief_builder.build_customer([plain, local, typed], _prof(), log=lambda *a: None)
    assert seen[-1] is plain


def test_government_items_dropped_unless_lead_sells_to_government(monkeypatch):
    gov = _art("Government launches £28m farming innovation competition")
    firm = _art("Tomato grower expands into cucumbers in the UK")
    seen = []

    def triage(cands, profile):
        seen.append(list(cands[:10]))
        return {}, cands[:10]

    monkeypatch.setattr(brief_builder, "triage_triggers", triage)
    monkeypatch.setattr(brief_builder, "_rank_by_evidence", lambda t: t)
    brief_builder.build_customer([gov, firm], _prof(industries=["Growers"]), log=lambda *a: None)
    assert seen[-1] == [firm]
    brief_builder.build_customer([gov, firm], _prof(industries=["Municipalities"]), log=lambda *a: None)
    assert gov in seen[-1]


def test_one_pool_labels_buyer_and_competitor_news(monkeypatch):
    trig = [_art(f"Buyer news {i}") for i in range(30)]
    comp = [_art(f"Rival news {i}", bucket="competitor") for i in range(30)]
    seen = []

    def triage(cands, profile):
        seen.extend(cands[:10])
        return {}, cands[:10]

    monkeypatch.setattr(brief_builder, "triage_triggers", triage)
    monkeypatch.setattr(brief_builder, "_rank_by_evidence", lambda t: t)
    brief_builder.build_customer(trig, _prof(), log=lambda *a: None, competitor_arts=comp)
    assert len(seen) == 40
    assert sum(a.bucket == "competitor" for a in seen) == 20


def test_appointments_and_foreign_topic_stories_skipped():
    assert brief_builder._is_appointment(_art("BD Appoints Matthew Sassone as President of Pharmacy Automation"))
    assert brief_builder._is_appointment(_art("Acme names new CEO"))
    assert not brief_builder._is_appointment(_art("Acme opens plant in Ohio"))
