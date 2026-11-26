"""Batch 5: profile builder -- English values, enquiry-topic hint, generic and inactive rivals."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import llm
import rivals
import tenant


def _capture_from_company(monkeypatch, **kw):
    seen = {}

    def fake_chat_json(system, user, **k):
        seen["system"], seen["user"] = system, user
        return {}
    monkeypatch.setattr(llm, "chat_json", fake_chat_json)
    monkeypatch.setattr(tenant, "fetch_site_text", lambda *a, **k: "We make sunbeds and roof boxes. " * 20)
    tenant.Profile.from_company("VDL Hapro", "vdlhapro.com", "there", "Netherlands", **kw)
    return seen["system"] + "\n" + seen["user"]


def test_prompt_has_english_rule_and_topic_hint(monkeypatch):
    p = _capture_from_company(monkeypatch, topic_hint="Red Light Therapy Beds Market")
    assert "Write every value in English" in p
    assert "ENQUIRY:" in p and "Red Light Therapy Beds Market" in p


def test_no_topic_hint_for_market_subject(monkeypatch):
    p = _capture_from_company(monkeypatch, topic_hint="Red Light Therapy Beds Market",
                              subject_is_market=True)
    assert "ENQUIRY:" not in p


def test_profile_cache_passes_topic_hint(monkeypatch):
    seen = {}

    def fake(cls, company, website="", recipient="there", country_name="", conn=None, **kw):
        seen.update(kw)
        return tenant.Profile(company=company, short=company, recipient=recipient, tagline="",
                              value_prop="x", industries=[], competitors=[], industry_keywords={},
                              trigger_queries=[], industry_queries=[])
    monkeypatch.setattr(tenant.Profile, "from_company", classmethod(fake))
    tenant.ProfileCache().get("VDL Hapro", "vdlhapro.com", "there", "Netherlands",
                              topic="Red Light Therapy Beds Market")
    assert seen.get("topic_hint") == "Red Light Therapy Beds Market"


def _profile(**kw):
    d = dict(company="Australian Rain Technology", short="ART", recipient="there", tagline="",
             value_prop="Rainfall enhancement technology for farms.", industries=["Agriculture"],
             competitors=["Cloud Seeding Australia", "ChargePoint"], industry_keywords={},
             trigger_queries=["farm expansion"], industry_queries=["rainfall trends"],
             market="Australia", competitor_query_seeds=["cloud seeding"],
             competitor_aliases={"Cloud Seeding Australia": ["CSA"], "ChargePoint": []})
    d.update(kw)
    return tenant.Profile(**d)


def test_generic_rival():
    p = _profile()
    assert tenant._generic_rival("Cloud Seeding Australia", p)
    assert not tenant._generic_rival("ChargePoint", p)


def test_tidy_profile_drops_generic_rival_and_aliases():
    p = _profile()
    tenant.tidy_profile(p)
    assert p.competitors == ["ChargePoint"]
    assert "Cloud Seeding Australia" not in p.competitor_aliases


def test_inactive_rival_dropped(monkeypatch):
    p = _profile(competitors=["Real Rival", "Ghost Rival"], competitor_aliases={})
    monkeypatch.setattr(rivals, "rival_active", lambda prof, name, *a, **k: name == "Real Rival")
    tenant._drop_inactive_rivals(p, conn=object())
    assert p.competitors == ["Real Rival"]


def test_rival_check_error_keeps_all(monkeypatch):
    p = _profile(competitors=["Real Rival", "Ghost Rival"], competitor_aliases={})

    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(rivals, "rival_active", boom)
    tenant._drop_inactive_rivals(p, conn=object())
    assert p.competitors == ["Real Rival", "Ghost Rival"]


def test_rival_check_skipped_without_db(monkeypatch):
    p = _profile(competitors=["Ghost Rival"], competitor_aliases={})
    monkeypatch.setattr(rivals, "rival_active", lambda *a, **k: pytest.fail("must not run"))
    tenant._drop_inactive_rivals(p, conn=None)
    assert p.competitors == ["Ghost Rival"]
