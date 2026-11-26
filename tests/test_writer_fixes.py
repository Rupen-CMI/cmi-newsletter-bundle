"""Writer fixes: deterministic lede, over-claim guard, plain-sentence prompt context,
evidence wording. All LLM calls are mocked."""
import re
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import brief_builder as bb
import compose
import voice
from collect import Article


def _profile(**kw):
    base = dict(company="Waters", short="Waters", value_prop="liquid chromatography.",
                industries=["Pharma"], competitors=["Thermo Fisher"], market="India",
                lead_type="seller", buyer_functions=["QC labs"], recipient="Sam",
                products=[{"name": "HPLC systems", "application": "testing"}])
    base.update(kw)
    return SimpleNamespace(**base)


def _sections(*labels):
    titles = {"Potential Customer": "Samsung Biologics", "Industry Outlook": "FSSAI rules tighten",
              "Competition Outlook": "Thermo Fisher"}
    return [{"label": l, "title": titles.get(l, "X"), "body_html": "b"} for l in labels]


ALL3 = ("Potential Customer", "Industry Outlook", "Competition Outlook")


def _lede(themes, profile=None, labels=ALL3):
    secs = _sections(*labels)
    reply = {"themes": themes, "top3": ["<b>a</b> b", "<b>c</b> d", "<b>e</b> f"]}
    with mock.patch.object(compose.llm, "chat_json", return_value=reply) as cj:
        lede, top3 = compose.editorial(profile or _profile(), secs)
    assert cj.call_count == 1
    return lede, top3


def test_lede_is_one_sentence_with_exact_opener():
    lede, top3 = _lede(["expanding production capacity", "", "launching a product line"])
    assert lede.startswith("This week's brief covers three developments: a <b>potential customer</b>")
    assert len(re.findall(r"[.!?]", re.sub(r"<[^>]+>", "", lede))) == 1 and lede.endswith(".")
    assert "<b>industry shift</b>" in lede and "and a <b>competitor move</b>" in lede
    assert lede.endswith("in India.") and len(top3) == 3


def test_lede_drops_proper_noun_theme():
    lede, _ = _lede(["Samsung Biologics expanding", "FSSAI rule change", "Thermo Fisher launching systems"])
    for bad in ("Samsung", "FSSAI", "Thermo", "Bisleri"):
        assert bad not in lede
    assert lede == ("This week's brief covers three developments: a <b>potential customer</b>, "
                    "an <b>industry shift</b>, and a <b>competitor move</b> in India.")


def test_lede_theme_rules():
    ok = compose._theme_ok
    assert ok("expanding production capacity", "India") == "expanding production capacity"
    assert ok("an Indian plant expansion", "India") == "an Indian plant expansion"
    assert ok("a plant expansion in India", "India")
    assert ok("a plant in Brazil", "India") == ""
    for bad in ("one two three four five six seven", "single", "raising $5 million",
                "ten new 2 sites", "expanding, quickly", "a company's plant", ""):
        assert ok(bad, "India") == "", bad
    assert ok(None, "India") == ""


def test_lede_no_possessive_or_blank_artifacts():
    lede, _ = _lede(["a competitor move early-stage", "", ""])
    assert "'s " not in lede.replace("This week's", "")
    assert not re.search(r"\b(the|a|an)\s+(in|of|for|by|where|on)\b", lede)
    assert "move'" not in lede


def test_lede_malformed_model_reply_still_builds():
    for reply in ({}, {"themes": "oops"}, {"themes": [None, 3, {}]}, None):
        with mock.patch.object(compose.llm, "chat_json", return_value=reply):
            lede, top3 = compose.editorial(_profile(), _sections("Potential Customer", "Competition Outlook"))
        assert lede == ("This week's brief covers two developments: a <b>potential customer</b> "
                        "and a <b>competitor move</b> in India.")
        assert len(top3) == 2


def test_lede_deal_partnership_labels_and_global():
    lede, _ = _lede([], profile=_profile(market="Global"), labels=("Potential Deal",))
    assert lede == "This week's brief covers one development: a <b>potential deal</b>."
    lede, _ = _lede([], labels=("Potential Partnership", "Competition Outlook"))
    assert "a <b>potential partnership</b>" in lede


def test_theme_with_country_not_duplicated():
    lede, _ = _lede(["a plant expansion in India", "", ""])
    assert lede.count("India") == 1


# ── over-claim guard ──────────────────────────────────────────────────────────
OVER = "The launch will increase demand for new systems."
CLEAN = "The launch could add demand for new systems."


def _card_reply(text):
    return {"narrative_html": text, "why_now": "x", "who_to_target": "QC", "recommended_action": "y",
            "approach_plan": [{"title": "t", "body_html": "b"}] * 3}


def _art():
    return Article(title="T", url="u", source="S", published=datetime.now(timezone.utc),
                   summary="s", query="q", bucket="trigger", fingerprint="T")


def test_overclaim_guard_regenerates_once_and_keeps_cleaner():
    import validate_brief
    assert validate_brief.overclaim_warnings(OVER)
    assert not validate_brief.overclaim_warnings(CLEAN)
    replies = [_card_reply(OVER), _card_reply(CLEAN)]
    with mock.patch.object(bb.llm, "chat_json", side_effect=replies) as cj:
        card = bb._write_card(_art(), "Acme", "customer", {}, _profile())
    assert cj.call_count == 2
    assert "could add demand" in card["narrative_html"]
    first, second = cj.call_args_list[0].args[1], cj.call_args_list[1].args[1]
    assert second.startswith(first) and "could or may" in second
    assert "will increase demand" in second


def test_overclaim_guard_keeps_original_when_retry_no_better():
    worse = OVER + " Stricter rules bring penalties and price sensitivity."
    replies = [_card_reply(OVER), _card_reply(worse)]
    with mock.patch.object(bb.llm, "chat_json", side_effect=replies) as cj:
        out = bb._chat_guarded("s", "u", max_tokens=10)
    assert cj.call_count == 2 and out["narrative_html"] == OVER


def test_no_retry_when_clean():
    with mock.patch.object(bb.llm, "chat_json", return_value=_card_reply(CLEAN)) as cj:
        bb._write_card(_art(), "Acme", "customer", {}, _profile())
    assert cj.call_count == 1


# ── prompt wording ────────────────────────────────────────────────────────────
def test_context_block_plain_sentences_and_buyers_only_in_customer_card():
    ctx = voice.seller_context(_profile(), buyers=True)
    assert not re.search(r"SELLER|BUYER|COUNTRY|PRODUCTS", ctx)
    prompts = {}
    for kind in ("customer", "competitor"):
        with mock.patch.object(bb.llm, "chat_json", return_value=_card_reply(CLEAN)) as cj:
            bb._write_card(_art(), "Acme", kind, {}, _profile())
        prompts[kind] = cj.call_args_list[0].args
    assert "Who usually buys them: QC labs" in prompts["customer"][1]
    assert "QC labs" not in prompts["competitor"][1]
    assert "QC labs" not in bb._ctx_prefix(_profile()) and "QC labs" not in compose._ctx(_profile())


def test_field_instructions_ask_only_for_what_signal_supports():
    with mock.patch.object(bb.llm, "chat_json", return_value=_card_reply(CLEAN)) as cj:
        bb._write_card(_art(), "Acme", "customer", {}, _profile())
    system, user = cj.call_args.args[:2]
    assert "create a need" not in user and "buying window" not in user
    assert "using only what the SIGNAL states" in user
    assert "BUYER FUNCTION" not in system + user


def test_voice_rules_drop_name_the_condition():
    for block in (voice.STYLE, voice.STYLE_BRIEF, voice.bullet_system(), voice.BULLET_SYSTEM):
        assert "name the condition" not in block and "plus the condition" not in block
        assert "BUYER FUNCTION" not in block and "SELLER PRODUCT" not in block
    assert '"could" or "may"' in voice.STYLE and '"could" or "may"' in voice.STYLE_BRIEF
