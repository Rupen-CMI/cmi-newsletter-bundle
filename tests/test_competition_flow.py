"""Competition Outlook flow: the trained competitive-move check is the judge, string
matching only attributes and ranks. Tenant profile seeds + product grounding. LLM mocked."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import brief_builder as bb
import tenant
from collect import Article


def _profile(**kw):
    base = dict(company="Seller", short="Seller", value_prop="widgets for labs.",
                industries=["Pharma"], competitors=["Acme Widgets"], market="Global",
                lead_type="seller", buyer_functions=[], recipient="Sam",
                products=[{"name": "Zeta Widget", "application": "x"}])
    base.update(kw)
    return SimpleNamespace(**base)


def _art(title, composite=1.0):
    a = Article(title=title, url="http://x", source="S", published=datetime.now(timezone.utc),
                summary="", query="q", bucket="competitor", fingerprint=title)
    a.composite = composite
    return a


def _run(arts, peers, verdict, profile=None):
    judged = []

    def fake_check(a, comp, prof):
        judged.append((a.title, comp))
        return verdict(a.title)

    with mock.patch.object(bb, "_competitive_move_check", side_effect=fake_check), \
            mock.patch.object(bb, "_write_card", side_effect=lambda a, n, *r: {"company_name": n}), \
            mock.patch.object(bb.llm, "chat_json", return_value={}),             mock.patch.object(bb.free_research, "enrich_company", return_value={}):
        out = bb.build_competition(arts, peers, profile or _profile(), log=lambda *_: None)
    return out, judged


def test_peer_not_on_list_reaches_check():
    peer = (_art("Newco Labs expands plant"), "Newco Labs")
    out, judged = _run([], [peer], lambda t: True)
    assert judged == [("Newco Labs expands plant", "Newco Labs")]
    assert out[0]["company_name"] == "Newco Labs"


def test_non_buyer_peer_excluded():
    out, judged = _run([], [(_art("Ministry plan"), "Ministry of Health")], lambda t: True)
    assert out == (None, None) and judged == []


def test_off_keyword_story_selected_when_check_true():
    arts = [_art("Acme Widgets signs big distribution deal")]
    out, judged = _run(arts, [], lambda t: True)
    assert out[0]["company_name"] == "Acme Widgets"


def test_check_false_skipped_then_next_wins():
    arts = [_art("Acme Widgets hires CFO", 5), _art("Acme Widgets opens new site", 1)]
    out, judged = _run(arts, [], lambda t: "site" in t)
    assert [j[0] for j in judged] == ["Acme Widgets hires CFO", "Acme Widgets opens new site"]
    assert out[1].title == "Acme Widgets opens new site"


def test_none_pass_returns_none():
    assert _run([_art("Acme Widgets news")], [], lambda t: False)[0] == (None, None)


def test_on_category_ranked_first_when_both_pass():
    arts = [_art("Acme Widgets sells shoes", 9), _art("Acme Widgets launches Zeta Widget 2", 1)]
    out, judged = _run(arts, [], lambda t: True)     # pick call fails -> top-ranked
    assert out[1].title == "Acme Widgets launches Zeta Widget 2"


def test_top_candidates_judged_and_deduped():
    n = bb._COMPETITION_JUDGE_TOP
    arts = [_art(f"Acme Widgets story {i}", i) for i in range(n + 2)]
    arts.append(_art(f"acme widgets story {n + 1}!", 99))   # duplicate title
    out, judged = _run(arts, [], lambda t: False)
    assert len(judged) == n
    assert len({t.lower().strip("!") for t, _ in judged}) == n


def test_query_seeds_default_and_from_dict():
    base = dict(company="C", short="C", recipient="r", tagline="t", value_prop="v",
                industries=[], competitors=[], industry_keywords={}, trigger_queries=[],
                industry_queries=[])
    assert tenant.Profile.from_dict(base).competitor_query_seeds == []
    assert tenant.Profile.from_dict({**base, "competitor_query_seeds": ["a b"]}
                                    ).competitor_query_seeds == ["a b"]


def _from_company(llm_data, site_text):
    with mock.patch.object(tenant.llm, "chat_json", return_value=llm_data), \
            mock.patch.object(tenant, "fetch_site_text", return_value=site_text):
        return tenant.Profile.from_company("Seller Co", website="http://example.com")


def test_seeds_parsed_and_products_grounded():
    site = "Welcome. Our flagship Synapt G2 Si system and Quanta line. " * 5
    data = {"value_prop": "v", "competitor_query_seeds": ["category launches", "category funding"],
            "products": [{"name": "SYNAPT G2-Si", "application": "a"},
                         {"name": "Hallucinated Gizmo", "application": "b"},
                         {"name": "Quanta", "application": "c"}]}
    p = _from_company(data, site)
    assert p.competitor_query_seeds == ["category launches", "category funding"]
    assert [x["name"] for x in p.products] == ["SYNAPT G2-Si", "Quanta"]


def test_products_kept_when_no_site_text():
    p = _from_company({"products": [{"name": "Anything", "application": "a"}]}, "")
    assert [x["name"] for x in p.products] == ["Anything"]
    assert p.profile_confidence == "low"
