"""Competition Outlook: the card's own write-up is re-judged by the trained check, and
named-rival candidates get a batched event_country so foreign events are skipped.
All LLM calls are mocked."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import brief_builder as bb
from collect import Article


def _profile(**kw):
    base = dict(company="Seller", short="Seller", value_prop="widgets for labs.",
                industries=["Pharma"], competitors=["Acme Widgets"], market="Italy",
                lead_type="seller", buyer_functions=[], recipient="Sam",
                products=[{"name": "Zeta Widget", "application": "x"}])
    base.update(kw)
    return SimpleNamespace(**base)


def _art(title, composite=1.0, **kw):
    a = Article(title=title, url=f"http://x/{abs(hash(title))}", source="S",
                published=datetime.now(timezone.utc), summary="raw summary", query="q",
                bucket="competitor", fingerprint=title)
    a.composite = composite
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _is_check(system):
    return "Decide whether the SIGNAL is a real competitive move" in system


def _run(arts, *, check, country_reply, narratives=None, profile=None):
    """check(user_text) -> bool for every trained-check call; country_reply is what the
    batched country call returns. Returns (out, calls) with calls = [(kind, system, user)]."""
    calls = []

    def fake_chat(system, user, **kw):
        if _is_check(system):
            calls.append(("check", system, user))
            return {"real_competitive_move": check(user)}
        calls.append(("country", system, user))
        return country_reply

    def fake_write(a, n, *r):
        return {"company_name": n, "narrative_html": (narratives or {}).get(a.title, "")}

    with mock.patch.object(bb.llm, "chat_json", side_effect=fake_chat), \
            mock.patch.object(bb, "_write_card", side_effect=fake_write), \
            mock.patch.object(bb.free_research, "enrich_company", return_value={}):
        out = bb.build_competition(arts, [], profile or _profile(), log=lambda *_: None)
    return out, calls


UNKNOWN = {"items": []}


def test_selfcheck_false_skips_to_next_candidate():
    arts = [_art("Acme Widgets axe sale", 5), _art("Acme Widgets launches widget line", 1)]
    narr = {"Acme Widgets axe sale": "<b>Acme</b> sells tools, a different buyer and use case.",
            "Acme Widgets launches widget line": "Acme launches a widget line for labs."}
    out, calls = _run(arts, check=lambda u: "different buyer" not in u, country_reply=UNKNOWN,
                      narratives=narr)
    assert out[0]["company_name"] == "Acme Widgets"
    assert out[1].title == "Acme Widgets launches widget line"
    assert sum(1 for c in calls if c[0] == "check") == 4   # raw + own, for each candidate


def test_selfcheck_passing_card_returned():
    a = _art("Acme Widgets launches widget line")
    out, _ = _run([a], check=lambda u: True, country_reply=UNKNOWN,
                  narratives={a.title: "Acme launches a widget line."})
    assert out[1] is a and out[0]["narrative_html"]


def test_selfcheck_uses_trained_template_with_narrative_as_summary():
    a = _art("Acme Widgets launches widget line")
    _, calls = _run([a], check=lambda u: True, country_reply=UNKNOWN,
                    narratives={a.title: "<b>Acme</b> launches a widget line."})
    checks = [c for c in calls if c[0] == "check"]
    assert len(checks) == 2
    system, user = checks[1][1], checks[1][2]
    assert system == checks[0][1]                  # same system prompt as the raw check
    assert "Decide whether the SIGNAL is a real competitive move" in system
    assert "SUMMARY: Acme launches a widget line." in user and "<b>" not in user
    assert "raw summary" not in user
    assert a.summary == "raw summary"              # original article untouched


def test_foreign_event_skipped_same_and_unknown_kept():
    for reply, kept in (({"items": [{"i": 0, "event_country": "India"}]}, False),
                        ({"items": [{"i": 0, "event_country": "Italy"}]}, True),
                        ({"items": [{"i": 0, "event_country": "unknown"}]}, True),
                        ({}, True)):                      # failed call -> default {}
        a = _art("Acme Widgets expands plant")
        out, calls = _run([a], check=lambda u: True, country_reply=reply)
        assert (out[0] is not None) is kept, reply
        if not kept:
            assert not any(c[0] == "check" for c in calls)   # skipped before the check


def test_country_call_batched_once_and_skipped_when_known():
    arts = [_art("Acme Widgets one", 3), _art("Acme Widgets two", 2), _art("Acme Widgets three", 1)]
    _, calls = _run(arts, check=lambda u: False, country_reply=UNKNOWN)
    country = [c for c in calls if c[0] == "country"]
    assert len(country) == 1
    assert all(f"TITLE: {a.title}" in country[0][2] for a in arts)
    assert "event_country" in country[0][2]
    known = [_art(f"Acme Widgets item {i}", event_country="Italy") for i in range(2)]
    _, calls = _run(known, check=lambda u: False, country_reply=UNKNOWN)
    assert not any(c[0] == "country" for c in calls)
