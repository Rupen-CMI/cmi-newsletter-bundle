"""Fact guard: units, numbers and names the writer adds beyond its own input."""
import llm
import voice


def kinds(text, source):
    return voice.unsupported_facts(text, source)


def test_unit_added_is_flagged():
    assert ("unit", "46.3 billion yen") in kinds(
        "Output reached 46.3 billion yen.", "Report shows 46.3 billion added value")


def test_unit_kept_is_not_flagged():
    assert not [f for f in kinds("Output reached 46.3 billion.",
                                 "Report shows 46.3 billion added value") if f[0] == "unit"]


def test_unit_equivalents():
    assert not kinds("Up 15 percent.", "Up 15% last year")
    assert not kinds("A $5.1M deal.", "A $5.1 million deal")


def test_name_flags():
    src = "Sumitomo Mitsui Trust Bank buys stake in Indian PE firm"
    text = "Sumitomo Mitsui Trust Bank bought a stake in Morrison Capital Partners."
    assert ("name", "Morrison Capital Partners") in kinds(text, src)
    assert not [f for f in kinds("The Japan Times reported it.", "via The Japan Times")
                if f[0] == "name"]


def test_drop_only_offending_sentence():
    src = "Output 46.3 billion"
    body = "<b>Output</b> grew. Value hit 46.3 billion yen. Growth continues."
    out = voice._drop_unsupported_sentences({"body_html": body, "event_country": "Japan"}, src)
    assert out["body_html"] == "<b>Output</b> grew. Growth continues."
    assert out["event_country"] == "Japan"
    single = "Value hit 46.3 billion yen."
    assert voice._drop_unsupported_sentences(single, src) == single


def test_guard_drops_after_failed_retry(monkeypatch):
    calls = []
    reply = {"body": "Demand is steady. Value hit 46.3 billion yen."}

    def fake(system, user, **kw):
        calls.append(user)
        return dict(reply)
    monkeypatch.setattr(llm, "chat_json", fake)
    out = voice.chat_json_guarded("sys", "Output 46.3 billion", max_tokens=100)
    assert out == {"body": "Demand is steady."}
    assert len(calls) == 2 and "46.3 billion yen" in calls[1]


def test_clean_reply_single_call(monkeypatch):
    calls = []

    def fake(system, user, **kw):
        calls.append(1)
        return {"body": "Output reached 46.3 billion."}
    monkeypatch.setattr(llm, "chat_json", fake)
    out = voice.chat_json_guarded("sys", "Output 46.3 billion", max_tokens=100)
    assert out == {"body": "Output reached 46.3 billion."} and len(calls) == 1
