"""Phase 1 seller-profile fixes: new fields, casing, competitor cleanup, caching.
No LLM / network: llm.chat_json and fetch_site_text are monkeypatched."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tenant  # noqa: E402
from tenant import Profile, ProfileCache  # noqa: E402

_OLD = dict(company="Acme", short="ACM", recipient="Bob", tagline="t", value_prop="v",
            industries=["x"], competitors=["y"], industry_keywords={"x": ["a"]},
            trigger_queries=["q"], industry_queries=["i"], market="India")


def _fake_llm(monkeypatch, data, site=""):
    monkeypatch.setattr(tenant.llm, "chat_json", lambda *a, **k: data)
    monkeypatch.setattr(tenant, "fetch_site_text", lambda *a, **k: site)


def test_old_dict_loads_with_defaults():
    p = Profile(**_OLD)
    assert p.lead_type == "seller" and p.products == [] and p.buyer_functions == []
    assert p.profile_confidence == "normal"


def test_from_dict_ignores_unknown_keys():
    p = Profile.from_dict({**_OLD, "bogus": 1})
    assert p.company == "Acme"
    assert Profile.from_dict({**_OLD, "products": [{"name": "X", "application": "y"}]}).products


def test_product_terms_merges_overrides(monkeypatch):
    monkeypatch.setattr(tenant, "_overrides_for",
                        lambda c: {"product_terms": ["Car-T", "uplc-ms/ms"]})
    p = Profile.from_dict({**_OLD, "products": [{"name": "UPLC-MS/MS", "application": "a"},
                                                {"name": "Columns", "application": "b"}]})
    assert tenant.product_terms_of(p) == ["uplc-ms/ms", "columns", "car-t"]


def test_casing_from_site_and_fallback(monkeypatch):
    assert tenant._fix_company_case("waters", "Welcome to Waters Corporation") == "Waters"
    assert tenant._fix_company_case("acme tools", "") == "Acme Tools"
    assert tenant._fix_company_case("McKenzie Labs", "") == "McKenzie Labs"
    assert tenant._fix_company_case("ACME TOOLS", "") == "Acme Tools"
    assert tenant._fix_company_case("ibm", "About IBM today") == "IBM"
    _fake_llm(monkeypatch, {}, site="Waters Corporation makes LC systems " * 10)
    assert Profile.from_company("waters", "waters.com").company == "Waters"


def test_competitor_cleanup():
    out = tenant._clean_competitors(["Usługi Obróbki Metalu", "metal fabrication services",
                                     "Agilent Technologies", "agilent technologies",
                                     "Shimadzu"])
    assert out == ["Agilent Technologies", "Shimadzu"]


def test_bad_types_and_lead_type(monkeypatch):
    _fake_llm(monkeypatch, {"lead_type": "wizard", "products": "nope",
                            "buyer_functions": 5, "target_industries": ["Labs"]})
    p = Profile.from_company("Acme", "acme.com")
    assert p.lead_type == "seller" and p.products == [] and p.buyer_functions == []
    assert p.profile_confidence == "low"
    _fake_llm(monkeypatch, {"lead_type": "Investor",
                            "products": [{"name": "A", "application": "b"}, "B", 3],
                            "buyer_functions": ["QC labs"]}, site="A and B products " + "x" * 300)
    p = Profile.from_company("Acme", "acme.com")
    assert p.lead_type == "investor" and [x["name"] for x in p.products] == ["A", "B"]
    assert p.buyer_functions == ["QC labs"] and p.profile_confidence == "normal"


def test_run_cache_same_object(monkeypatch):
    calls = []

    def fake(company, website="", recipient="there", country_name="", conn=None,
             subject_is_market=False):
        calls.append(company)
        return Profile(**{**_OLD, "company": company})

    monkeypatch.setattr(Profile, "from_company", staticmethod(fake))
    c = ProfileCache()
    a = c.get("Acme", "https://acme.com", "Bob", "India")
    b = c.get("Acme", "acme.com", "Bob", "India")
    assert a is b and len(calls) == 1
    assert c.get("Acme", "acme.com", "Bob", "Germany") is not a
