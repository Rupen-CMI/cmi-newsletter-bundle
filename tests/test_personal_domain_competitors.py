"""Personal-domain leads, competitor cleanup/scoring, query tidying, vm_connect guard.

Found on brief #20298: okarch@gmx.net was profiled as GMX (the mail provider), GMX was
listed as its own competitor, all 8 German queries searched GMX's own news, every
competitor search was anchored on "GMX Cloud", and "Germany" appeared twice in most
searches. A 40-brief sample also showed rivals localised for the wrong market
("Stryker India" for an Irish lead) and placeholder names ("Company A").

No network: fetches are faked, the embedding model is switched off where scoring runs.
"""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import collect as C
import competitor_score as cs
import config
import db
import enrich
import pipeline
import relevance
import tenant
import vm_connect
from collect import Article


def _profile(**kw):
    base = dict(company="gmx", short="gmx", market="Germany", competitors=[],
                trigger_queries=[], industry_queries=[], competitor_query_seeds=[],
                products=[], local_queries=[], local_competitor_names={},
                competitor_aliases={}, value_prop="", industry_keywords={})
    base.update(kw)
    return SimpleNamespace(**base)


def _art(title, days=1, summary="", url=None):
    return Article(title=title, url=url or f"http://x/{title}", source="S",
                   published=datetime.now(timezone.utc) - timedelta(days=days),
                   summary=summary, query="", bucket="competitor", fingerprint=title)


# ── personal domains ─────────────────────────────────────────────────────────
class TestCompanyInfo:
    def test_provider_name_is_not_a_company(self):
        assert enrich.company_info("okarch@gmx.net", "gmx", "https://gmx.net") == ("", "")
        assert enrich.company_info("a@gmail.com", "Gmail") == ("", "")
        assert enrich.company_info("a@icloud.com", "icloud.com") == ("", "")
        assert enrich.company_info("a@outlook.com", "") == ("", "")

    def test_sheet_company_kept_for_personal_email(self):
        assert enrich.company_info("t@aol.com", "GET Enterprises, Inc.") == \
            ("GET Enterprises, Inc.", "")

    def test_company_domain_gives_site_and_name(self):
        assert enrich.company_info("x@vvfltd.com", "") == ("Vvfltd", "https://vvfltd.com")
        assert enrich.company_info("x@vvfltd.com", "VVF") == ("VVF", "https://vvfltd.com")
        assert enrich.company_info("x@acme.com", "none") == ("Acme", "https://acme.com")

    def test_personal_website_is_not_a_domain(self):
        assert enrich.domain_of(website="https://gmx.net") == ""
        assert enrich.domain_of(email="a@web.de") == ""
        assert enrich.domain_of(website="https://www.acme.com/x") == "acme.com"


class TestReportOnly:
    def test_kinds(self):
        both = ["signal", "report"]
        assert pipeline.kinds_for_lead({"company": "Acme", "topic": "X"}, both) == both
        assert pipeline.kinds_for_lead({"company": "", "topic": "Genomic Biomarker Market"},
                                       ["signal"]) == ["report"]
        assert pipeline.kinds_for_lead({"company": "", "topic": ""}, both) == []

    def test_subject_without_company(self):
        assert db.brief_subject("", "report", "Genomic Biomarker Market") == \
            "Coherent Market Insights · Genomic Biomarker Market"
        assert db.brief_subject("Genomic Biomarker Market", "report",
                                "Genomic Biomarker Market") == \
            "Coherent Market Insights · Genomic Biomarker Market"
        assert db.brief_subject("Acme", "report", "X Market") == \
            "Coherent Market Insights · Acme · X Market"
        assert db.brief_subject("Acme", "signal") == "Coherent Market Insights · Acme's Brief"

    def test_cache_profiles_the_market(self, monkeypatch):
        seen = {}

        def fake(company, website="", recipient="there", country_name="", conn=None,
                 subject_is_market=False):
            seen.update(company=company, website=website, market=subject_is_market)
            return tenant.Profile(company=company, short=company, recipient=recipient,
                                  tagline="", value_prop="", industries=[], competitors=[],
                                  industry_keywords={}, trigger_queries=[],
                                  industry_queries=[], market=country_name)

        monkeypatch.setattr(tenant.Profile, "from_company", staticmethod(fake))
        tenant.ProfileCache().get("", "", "Oliver", "Germany",
                                  topic="Genomic Biomarker Market")
        assert seen == {"company": "Genomic Biomarker Market", "website": "", "market": True}


# ── competitor cleanup ───────────────────────────────────────────────────────
class TestSeller:
    def test_seller_forms(self):
        p = _profile()
        assert cs.is_seller("Gmx.net", p)
        assert cs.is_seller("GMX GmbH", p)
        assert cs.is_seller("GMX Lotto Online", p)
        assert not cs.is_seller("Mailbox.org", p)

    def test_website_label(self):
        p = _profile(company="Web", short="Web")
        assert cs.is_seller("Web.de", p, "https://web.de")
        p2 = _profile(company="United Internet AG", short="UIA")
        assert cs.is_seller("GMX Cloud", p2, "https://gmx.net")


class TestForeignSuffix:
    @pytest.mark.parametrize("name,lead,want", [
        ("Stryker India", "IE", "Stryker"),
        ("Grant Thornton Bharat", "IE", "Grant Thornton"),
        ("Columbia Sportswear Company India", "FR", "Columbia Sportswear Company"),
        ("Kollmorgen India", "IN", "Kollmorgen India"),
        ("Bank of America", "GB", "Bank of America"),
        ("India Post", "GE", "India Post"),
        ("Emirates", "IN", "Emirates"),
        ("Mondi SA", "SE", "Mondi SA"),
        ("Stryker India", "", "Stryker India"),
    ])
    def test_strip(self, name, lead, want):
        assert cs.strip_foreign_suffix(name, lead) == want


class TestCleanProfile:
    def test_gmx_case(self):
        p = _profile(
            competitors=["Web.de", "Mailbox.org", "Gmx.net", "Company A", "Tutanota"],
            local_competitor_names={"Web.de": "Web.de", "Gmx.net": "GMX"},
            competitor_aliases={"Gmx.net": ["GMX"], "Tutanota": ["Tuta", "GMX Mail"]},
            local_queries=["GMX Lotto Online", "E-Mail Anbieter Expansion"],
            competitor_query_seeds=["GMX launch", "email service launch"])
        out = cs.clean_profile(p, "https://gmx.net", "DE")
        assert p.competitors == ["Web.de", "Mailbox.org", "Tutanota"]
        assert out["seller"] == ["Gmx.net"] and out["placeholder"] == ["Company A"]
        assert p.local_competitor_names == {"Web.de": "Web.de"}
        assert p.competitor_aliases == {"Tutanota": ["Tuta"]}
        assert p.local_queries == ["E-Mail Anbieter Expansion"]
        assert p.competitor_query_seeds == ["email service launch"]

    def test_renamed_keys_follow(self):
        p = _profile(company="Cook Medical", short="CM",
                     competitors=["Stryker India", "Medtronic India", "Stryker"],
                     competitor_aliases={"Medtronic India": ["MDT"]})
        out = cs.clean_profile(p, "", "IE")
        assert p.competitors == ["Stryker", "Medtronic"]
        assert p.competitor_aliases == {"Medtronic": ["MDT"]}
        assert ["Stryker India", "Stryker"] in out["renamed"]


# ── query tidying ────────────────────────────────────────────────────────────
class TestTidy:
    def test_country_words_and_seed_duplicates(self):
        p = _profile(trigger_queries=["German telecom merger", "Germany's fintech funding",
                                      "telecom merger"],
                     industry_queries=["Germany cloud adoption"],
                     competitor_query_seeds=["German fintech funding", "email launch"])
        out = tenant.tidy_profile(p, country=config.get_country("Germany"))
        assert p.trigger_queries == ["telecom merger", "fintech funding"]
        assert p.industry_queries == ["cloud adoption"]
        assert p.competitor_query_seeds == ["email launch"]
        assert out["seed_duplicates"] == ["fintech funding"]

    def test_global_market_untouched(self):
        p = _profile(market="Global", trigger_queries=["German telecom merger"])
        tenant.tidy_profile(p)
        assert p.trigger_queries == ["German telecom merger"]

    def test_other_words_survive(self):
        assert tenant.strip_country_words("Mumbai hospital expansion India", "IN") == \
            "Mumbai hospital expansion"
        assert tenant.strip_country_words("US hospital expansion", "US") == \
            "US hospital expansion"


class TestPlanSearches:
    def test_country_not_doubled(self):
        p = _profile(trigger_queries=["German telecom merger", "fintech funding"],
                     competitors=["Kollmorgen India"])
        jobs, _ = C.plan_searches(p, config.get_country("Germany"))
        qs = [q for q, *_ in jobs]
        assert "German telecom merger" in qs and "fintech funding Germany" in qs
        assert "Kollmorgen India Germany" in qs
        jobs_in, _ = C.plan_searches(p, config.get_country("India"))
        assert "Kollmorgen India" in [q for q, *_ in jobs_in]

    def test_anchor_skips_seller_product(self):
        p = _profile(competitors=["Tutanota"],
                     products=[{"name": "GMX Cloud"}, {"name": "Secure Mailbox"}])
        qs = [q for q, *_ in C.plan_searches(p, config.get_country("Germany"))[0]]
        assert "Tutanota Secure Mailbox Germany" in qs
        assert not any("GMX Cloud" in q for q in qs)


# ── scoring ──────────────────────────────────────────────────────────────────
class TestScore:
    @pytest.fixture(autouse=True)
    def no_model(self):
        relevance.set_encoder(False)
        yield
        relevance.reset()

    def test_relevant_recent_local_rival_scores_high(self):
        p = _profile(value_prop="Free email and cloud storage for consumers",
                     competitor_query_seeds=["email service launch"])
        arts = [_art(f"Tutanota launches encrypted email service in Germany {i}", days=i)
                for i in range(1, 8)]
        row = cs.score_competitor(p, "Tutanota", config.get_country("Germany"), "DE",
                                  fetch=lambda q, loc: arts)
        assert row["n"] == 7 and row["method"] == "keywords"
        assert row["relevance"] == 1.0 and row["presence"] == 1.0
        assert row["verdict"] == "keep"

    def test_unrelated_old_foreign_rival_drops(self):
        p = _profile(value_prop="Free email and cloud storage for consumers")
        arts = [_art("Tanishq opens jewellery showroom in Mumbai", days=300)]
        row = cs.score_competitor(p, "Tanishq", config.get_country("Thailand"), "TH",
                                  fetch=lambda q, loc: arts)
        assert row["relevance"] == 0.0 and row["presence"] == 0.0
        assert row["verdict"] == "drop"

    def test_relevant_rival_outside_lead_market(self):
        p = _profile(value_prop="Gold jewellery retail for consumers")
        arts = [_art(f"Tanishq launches gold jewellery collection {i}", days=i)
                for i in range(1, 6)]
        asked = []

        def fetch(q, loc):
            asked.append(q)
            return arts if q == "Tanishq" and loc[0].startswith("en") else []

        row = cs.score_competitor(p, "Tanishq", config.get_country("Thailand"), "TH",
                                  fetch=fetch)
        assert row["relevance"] == 1.0 and row["presence_hits"] == 0
        assert row["verdict"] == "out of market"
        assert "Tanishq Thailand" in asked      # the direct presence search ran

    def test_global_rival_found_by_direct_search(self):
        p = _profile(value_prop="3D design software for engineers")
        plain = [_art(f"Autodesk launches 3D design software update {i}", days=i)
                 for i in range(1, 6)]
        local = [_art("Autodesk opens Copenhagen design office, Denmark", url="http://a"),
                 _art("Danish engineers adopt Autodesk software", url="http://b")]
        row = cs.score_competitor(
            p, "Autodesk", config.get_country("Denmark"), "DK",
            fetch=lambda q, loc: (local if q == "Autodesk Denmark"
                                  else plain if loc[0].startswith("en") else []))
        assert row["presence_hits"] == 2 and row["verdict"] == "keep"

    def test_no_news(self):
        row = cs.score_competitor(_profile(), "Nobody Inc", config.get_country("Germany"),
                                  "DE", fetch=lambda q, loc: [_art("unrelated headline")])
        assert row["n"] == 0 and row["verdict"] == "no news"

    def test_empty_search_is_no_data(self):
        row = cs.score_competitor(_profile(), "Autodesk", config.get_country("Denmark"),
                                  "DK", fetch=lambda q, loc: [])
        assert row["verdict"] == "no data"


class TestFetchRetry:
    def test_blocked_feed_is_retried(self, monkeypatch):
        good = SimpleNamespace(status=200, bozo=0, entries=[])
        feeds = [SimpleNamespace(status=503, bozo=1, entries=[]), good]
        calls = []
        monkeypatch.setattr(C.feedparser, "parse", lambda url: calls.append(url) or feeds.pop(0))
        monkeypatch.setattr(C.time, "sleep", lambda s: None)
        C._fetch_one("q", "trigger", datetime.now(timezone.utc), ("en-US", "US", "US:en"))
        assert len(calls) == 2


# ── vm_connect: never start a second engine ──────────────────────────────────
class TestVmGuard:
    PS = ("__ps__\n/home/gv/bin/python /home/gv/bin/vllm serve /home/merged "
          "--served-model-name cmi9b --port 8000 --quantization fp8\n")

    def test_parse(self, monkeypatch):
        monkeypatch.setattr(vm_connect, "_jl", lambda args, timeout=0: (0, self.PS))
        assert vm_connect.running_servers(1)[0]["model"] == "cmi9b"
        assert vm_connect.running_servers(1)[0]["port"] == "8000"

    def _ensure(self, monkeypatch, jl_out):
        booted = []
        monkeypatch.setattr(vm_connect, "status", lambda: {"serving": False, "error": ""})
        monkeypatch.setattr(vm_connect, "find_instance",
                            lambda: {"machine_id": 1, "status": "Running"})
        monkeypatch.setattr(vm_connect, "_jl", lambda args, timeout=0: (0, jl_out))
        monkeypatch.setattr(vm_connect, "_boot", lambda mid: booted.append(mid))
        st = vm_connect.ensure(wait_secs=0, log=lambda *_: None)
        return st, booted

    def test_other_server_blocks_boot(self, monkeypatch):
        st, booted = self._ensure(monkeypatch, self.PS)
        assert not booted and "cmi9b on port 8000" in st["error"]

    def test_ssh_failure_reported(self, monkeypatch):
        st, booted = self._ensure(monkeypatch, "Permission denied (publickey).")
        assert not booted and "jl exec failed" in st["error"]

    def test_clean_vm_boots(self, monkeypatch):
        st, booted = self._ensure(monkeypatch, "__ps__\n")
        assert booted == [1]
