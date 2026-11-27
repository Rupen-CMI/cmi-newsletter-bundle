"""write_search_queries: separate small LLM call + deterministic check. llm.chat_json is mocked."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tenant


def _good(n=10):
    return {"trigger_queries": [f"buyer{i} opens plant" for i in range(n)],
            "industry_queries": ["new casting method", "robot welding line"],
            "competitor_query_seeds": ["pump makers merge"],
            "product_category": "industrial pumps", "local_queries": []}


def test_check_drops_long_vague_and_duplicates():
    d = {"trigger_queries": ["a b c d e", "dairy plant opens", "Dairy Plant Opens",
                             "market growth news", "farm builds barn"],
         "industry_queries": ["milk spending rises", "robot milking"],
         "competitor_query_seeds": ["pump makers merge"],
         "product_category": "very long product category name here"}
    out, problems = tenant._check_queries(d)
    assert out["trigger_queries"] == ["dairy plant opens", "farm builds barn"]
    assert out["industry_queries"] == ["robot milking"]
    assert out["product_category"] == "very long product"
    assert problems == {"too long", "vague"}


def test_check_local_queries_over_4_words_dropped_only_with_language():
    d = {"local_queries": ["jeden dwa", "jeden dwa trzy cztery piec"]}
    assert tenant._check_queries(d, "Polish")[0]["local_queries"] == ["jeden dwa"]
    assert tenant._check_queries(d, "")[0]["local_queries"] == []


def test_single_call_when_enough_buyer_queries(monkeypatch):
    calls = []
    monkeypatch.setattr(tenant.llm, "chat_json", lambda s, u, **k: calls.append(u) or _good())
    out = tenant.write_search_queries("Sells pumps.", ["Dairy"], ["Rival"], "Poland", "")
    assert len(calls) == 1
    assert len(out["trigger_queries"]) == 10 and out["product_category"] == "industrial pumps"
    assert "Poland" in calls[0] and "Dairy" in calls[0] and "Rival" in calls[0]


def test_retry_once_and_keep_better(monkeypatch):
    bad = {"trigger_queries": ["this one is far too long"] * 3 + ["market trends"] * 3,
           "industry_queries": ["x y"]}
    replies = iter([bad, _good(8)])
    calls = []
    monkeypatch.setattr(tenant.llm, "chat_json",
                        lambda s, u, **k: calls.append(u) or next(replies))
    out = tenant.write_search_queries("Sells pumps.", ["Dairy"], [], "Global", "")
    assert len(calls) == 2
    assert "too long" in calls[1] and "banned words" in calls[1]
    assert len(out["trigger_queries"]) == 8


def test_retry_worse_keeps_first(monkeypatch):
    replies = iter([_good(5), {}])
    monkeypatch.setattr(tenant.llm, "chat_json", lambda s, u, **k: next(replies))
    out = tenant.write_search_queries("v", ["Dairy"], [], "Global", "")
    assert len(out["trigger_queries"]) == 5


def test_local_language_in_prompt(monkeypatch):
    seen = []
    monkeypatch.setattr(tenant.llm, "chat_json", lambda s, u, **k: seen.append(u) or _good())
    tenant.write_search_queries("v", ["Dairy"], [], "Poland", "Polish")
    assert "local_queries" in seen[0] and "Polish" in seen[0]


def test_from_company_falls_back_when_query_call_fails(monkeypatch):
    def fake(system, user, **k):
        if "news-search queries" in system:
            raise RuntimeError("down")
        return {"value_prop": "v", "target_industries": ["Dairy"]}
    monkeypatch.setattr(tenant.llm, "chat_json", fake)
    monkeypatch.setattr(tenant, "fetch_site_text", lambda *a, **k: "")
    p = tenant.Profile.from_company("Acme", "", "there", "Canada")
    assert p.trigger_queries and p.industry_queries
    assert p.competitor_query_seeds == [] and p.product_category == ""


def test_from_company_fills_profile_fields(monkeypatch):
    def fake(system, user, **k):
        if "news-search queries" in system:
            return _good()
        assert "trigger_query_seeds" not in user and "product_category" not in user
        return {"value_prop": "v", "target_industries": ["Dairy"]}
    monkeypatch.setattr(tenant.llm, "chat_json", fake)
    monkeypatch.setattr(tenant, "fetch_site_text", lambda *a, **k: "")
    p = tenant.Profile.from_company("Acme", "", "there", "Canada")
    assert p.trigger_queries[0] == "buyer0 opens plant"
    assert p.competitor_query_seeds == ["pump makers merge"]
    assert p.product_category == "industrial pumps"
