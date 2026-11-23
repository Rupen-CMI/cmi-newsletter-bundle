"""Offline tests for the country-first evidence filter.

Covers the three pieces it is built from:

  regions.py     — the ~180-country table and its word-boundary country detector,
                   including every ambiguity the real lead data produced
                   (New Delhi vs New Orleans, Indian Ocean, Georgia the US state,
                   Jordan the person, Papua New Guinea vs Guinea, .co.in URLs)
  config.get_country — the 12 curated countries still win; everything else falls
                   back to regions.get_profile (accepting ISO2) before Global
  geo.filter_for_lead — the country-first filter that replaced filter_country's
                   fail-open behaviour for the lead pipeline

Everything here is offline: no network, no LLM, no SMTP.
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _art(title="X", summary="", source="Reuters", url="https://x.com/a",
         bucket="industry", query=""):
    from collect import Article

    return Article(title=title, url=url, source=source, summary=summary,
                   published=dt.datetime.now(dt.timezone.utc), bucket=bucket,
                   query=query)


def _country(name="India"):
    import config

    return config.get_country(name)


# ── regions.article_countries ────────────────────────────────────────────────
class TestArticleCountries:
    @pytest.mark.parametrize("title,expected", [
        ("New Delhi-based startup raises Series B", {"IN"}),
        ("Bostik launches adhesives in India", {"IN"}),
        ("Indian rupee steadies as Mumbai banks trade", {"IN"}),
        # "New Orleans" must not read as US-by-city... (no token matches) — and
        # definitely not India.
        ("New Orleans hospital expansion advances", set()),
        # US states now credit the US (sub-national places, 2026-11 fix):
        ("New Mexico solar manufacturer opens plant", {"US"}),
        ("Shipping lanes across the Indian Ocean widen", set()),
        # "Georgia" alone is still not a country token; with a US context signal
        # (Atlanta/county) it now counts as the US state.
        ("Georgia manufacturer to hire 300 workers", set()),
        ("State funds 16 Georgia shelters as Atlanta county expands capacity",
         {"US"}),
        ("Georgian wine exports rebound on Tbilisi demand", {"GE"}),
        ("Michael Jordan endorses a sneaker line", set()),
        ("Jordan's economy grows on Amman investment", {"JO"}),
        ("Jordanian bank opens branch in Amman", {"JO"}),
        ("Papua New Guinea expands port capacity", {"PG"}),
        ("Equatorial Guinea signs LNG deal", {"GQ"}),
        ("Guinea-Bissau holds elections in Bissau", {"GW"}),
        ("Guinea bauxite exports surge through Conakry", {"GN"}),
        ("North Korea fires a test missile", {"KP"}),
        ("South Korea semiconductor exports rise", {"KR"}),
        ("Niger uranium output climbs", {"NE"}),
        ("Nigeria banking reforms deepen", {"NG"}),
        ("New South Wales grid upgrade stalls", {"AU"}),
        ("Wales manufacturing grants extended", {"GB"}),
        ("Northern Ireland tech hiring picks up", {"GB"}),
        ("British Columbia forestry exports rebound", {"CA"}),
        ("Ontario manufacturing grants extended", {"CA"}),
        ("Quebec dairy co-op merges with rivals", {"CA"}),
        ("Dubai free-zone registrations climb", {"AE"}),
        ("Bavaria auto suppliers brace for slower demand", {"DE"}),
        ("Minas Gerais mining output rises", {"BR"}),
        ("Guangdong factory orders soften", {"CN"}),
        ("Maharashtra industrial policy attracts investors", {"IN"}),
        ("North America electronics demand softens", set()),
        ("U.S. chipmaker to invest $4 billion", {"US"}),
        ("Turkish lira steadies after Ankara remarks", {"TR"}),
        ("Poland hikes rates while the zloty firms", {"PL"}),
    ])
    def test_detection(self, title, expected):
        import regions

        got = regions.article_countries(_art(title=title))
        assert got == expected, f"{title!r}: got {sorted(got)}"

    def test_url_cctld_counts(self):
        import regions

        assert regions.article_countries(
            _art(title="Startup raises funds",
                 url="https://example.co.in/news")) == {"IN"}
        assert regions.article_countries(
            _art(title="Startup raises funds",
                 url="https://example.com.au/n")) == {"AU"}

    def test_foreign_article_names_both_countries(self):
        """Explicit-link rule input: the article names the lead's country too."""
        import regions

        got = regions.article_countries(
            _art(title="German supplier expands in India"))
        assert got == {"DE", "IN"}

    def test_table_covers_the_leads_table(self):
        """Every country_iso2 that actually appears in cmi_newsletter.db must
        resolve (read-only check; skipped when the DB file is absent)."""
        import regions

        db_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "cmi_newsletter.db")
        if not os.path.exists(db_path):
            pytest.skip("cmi_newsletter.db not present")
        import sqlite3

        conn = sqlite3.connect(db_path)
        try:
            isos = {r[0] for r in conn.execute(
                "SELECT DISTINCT country_iso2 FROM leads "
                "WHERE country_iso2 != ''")}
        finally:
            conn.close()
        missing = [i for i in sorted(isos) if regions.get_profile(i) is None]
        assert not missing, f"countries missing from regions table: {missing}"

    def test_regions_match_the_leads_table(self):
        """region_of() must agree with the region values the DB already carries
        for each ISO2 (read-only; skipped when the DB file is absent)."""
        import regions

        db_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "cmi_newsletter.db")
        if not os.path.exists(db_path):
            pytest.skip("cmi_newsletter.db not present")
        import sqlite3

        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT country_iso2, region FROM leads "
                "WHERE country_iso2 != '' AND country_source='stated' "
                "AND region != '' GROUP BY country_iso2, region").fetchall()
        finally:
            conn.close()
        mismatches = {}
        for iso2, region in rows:
            if len({r for i, r in rows if i == iso2}) == 1:
                want = regions.region_of(iso2)
                if want and want != region:
                    mismatches[iso2] = (want, region)
        assert not mismatches, f"region mismatches vs DB: {mismatches}"


# ── config.get_country ───────────────────────────────────────────────────────
class TestGetCountry:
    def test_curated_countries_still_win(self):
        import config

        assert config.get_country("India")["name"] == "India"
        assert config.get_country("India")["outlets"], "curated outlets kept"

    def test_unknown_and_empty_still_global(self):
        import config

        assert config.get_country("Atlantis")["name"] == "Global"
        assert config.get_country("")["name"] == "Global"

    @pytest.mark.parametrize("query,name,gl", [
        ("Poland", "Poland", "PL"),
        ("PL", "Poland", "PL"),
        ("Morocco", "Morocco", "MA"),
        ("South Korea", "South Korea", "KR"),
        ("UK", "United Kingdom", "GB"),
        ("UAE", "United Arab Emirates", "AE"),
    ])
    def test_wide_table_fallback(self, query, name, gl):
        import config

        prof = config.get_country(query)
        assert prof["name"] == name
        assert prof["gn"][1] == gl          # Google News gl=XX, e.g. ("..", "PL", "PL:en")
        assert prof["name"] != "Global"

    def test_iso2_of_a_curated_country_returns_the_curated_entry(self):
        import config

        assert config.get_country("IN")["name"] == "India"
        assert config.get_country("IN")["outlets"], "curated outlets kept via ISO2"


# ── geo.filter_for_lead ──────────────────────────────────────────────────────
class TestFilterForLead:
    def test_foreign_competitor_news_dropped(self):
        import geo

        kept, scope = geo.filter_for_lead(
            [_art(title="Brazilian coffee prices climb", bucket="competitor")],
            _country("India"))
        assert kept == []
        assert scope == "country"

    def test_foreign_news_naming_the_lead_country_kept(self):
        import geo

        a = _art(title="German supplier expands in India", bucket="trigger")
        kept, scope = geo.filter_for_lead([a], _country("India"))
        assert kept == [a]

    def test_in_country_kept_by_city_alias_and_cctld(self):
        import geo

        by_city = _art(title="Bengaluru startup raises Series B", bucket="trigger")
        by_tld = _art(title="Startup raises funds",
                      url="https://example.co.in/news", bucket="trigger")
        kept, scope = geo.filter_for_lead([by_city, by_tld], _country("India"))
        assert kept == [by_city, by_tld]
        assert scope == "country"

    def test_no_geo_industry_kept(self):
        """collect() appends the country name to every query, so a no-geo
        industry article found by that query is still country-anchored."""
        import geo

        a = _art(title="Global adhesives demand steadies", bucket="industry",
                 query="adhesives demand India")
        kept, _ = geo.filter_for_lead([a], _country("India"))
        assert kept == [a]

    def test_no_geo_industry_dropped_without_a_country_anchor(self):
        """2026-11 live run: "HD Hyundai and Anduril Launch Autonomous Vessel"
        reached an India lead — a no-geo industry article found by a query that
        names no country and carried by no outlet/TLD evidence is dropped."""
        import geo

        a = _art(title="HD Hyundai and Anduril launch autonomous vessel",
                 bucket="industry", query="autonomous vessels")
        kept, _ = geo.filter_for_lead([a], _country("India"))
        assert kept == []

    def test_no_geo_industry_kept_when_query_names_the_lead_country(self):
        import geo

        a = _art(title="HD Hyundai and Anduril launch autonomous vessel",
                 bucket="industry", query="autonomous vessels India")
        kept, _ = geo.filter_for_lead([a], _country("India"))
        assert kept == [a]

    def test_no_geo_industry_kept_by_demonym_in_query(self):
        import geo

        a = _art(title="Autonomous vessel pilots begin sea trials",
                 bucket="industry", query="autonomous vessels indian")
        kept, _ = geo.filter_for_lead([a], _country("India"))
        assert kept == [a]

    def test_no_geo_industry_kept_by_country_outlet(self):
        import geo

        a = _art(title="Global adhesives demand steadies",
                 source="Economic Times", bucket="industry", query="")
        kept, _ = geo.filter_for_lead([a], _country("India"))
        assert kept == [a]

    def test_no_geo_industry_kept_by_country_cctld(self):
        import geo

        a = _art(title="Global adhesives demand steadies",
                 url="https://example.in/news", bucket="industry", query="")
        kept, _ = geo.filter_for_lead([a], _country("India"))
        assert kept == [a]

    def test_no_geo_query_token_needs_word_boundaries(self):
        """A plain substring check would let "desi" match "design" or "uk" match
        "ukraine" — the anchor tokens must match as words."""
        import geo

        a = _art(title="Global adhesives demand steadies", bucket="industry",
                 query="adhesives designers design")     # contains "desi"
        kept, _ = geo.filter_for_lead([a], _country("India"))
        assert kept == []

    def test_no_geo_competitor_dropped(self):
        import geo

        a = _art(title="Global adhesives demand steadies", bucket="competitor")
        kept, _ = geo.filter_for_lead([a], _country("India"))
        assert kept == []

    def test_region_fallback_stays_inside_the_region(self):
        """Thin trigger/competitor pools may top up from the lead's REGION only:
        a Pakistan (Asia Pacific) article is addable for an India lead, a Brazil
        (Latin America) one never is, and the unfiltered pool is never used."""
        import geo

        arts = [
            _art(title="Mumbai fintech raises Series B", bucket="trigger"),
            _art(title="Chennai plant expansion announced", bucket="competitor"),
            _art(title="Karachi textile exporter secures funding",
                 bucket="trigger"),
            _art(title="Brazilian coffee prices climb", bucket="trigger"),
        ]
        kept, scope = geo.filter_for_lead(arts, _country("India"))
        titles = {a.title for a in kept}
        assert "Karachi textile exporter secures funding" in titles
        assert "Brazilian coffee prices climb" not in titles
        assert scope == "region"

    def test_no_region_fallback_when_country_evidence_is_plenty(self):
        import geo

        arts = [_art(title=f"Mumbai buyer {i} raises funding", bucket="trigger")
                for i in range(4)]
        arts.append(_art(title="Brazilian coffee prices climb", bucket="trigger"))
        kept, scope = geo.filter_for_lead(arts, _country("India"))
        assert all("Brazil" not in a.title for a in kept)
        assert scope == "country"

    def test_region_fallback_min_is_honoured(self):
        import geo

        arts = [
            _art(title="Mumbai fintech raises Series B", bucket="trigger"),
            _art(title="Karachi textile exporter secures funding",
                 bucket="trigger"),
        ]
        # threshold 1: the single surviving trigger article satisfies it
        # (Karachi was dropped pre-fallback) -> no fallback
        kept, scope = geo.filter_for_lead(arts, _country("India"),
                                          region_fallback_min=1)
        assert scope == "country"
        assert len(kept) == 1
        # threshold 2: one survivor is thin -> the same-region article is added
        kept, scope = geo.filter_for_lead(arts, _country("India"),
                                          region_fallback_min=2)
        assert scope == "region"
        assert len(kept) == 2

    def test_global_lead_unfiltered(self):
        import config
        import geo

        arts = [_art(title="Brazilian coffee prices climb", bucket="competitor"),
                _art(title="Anything at all", bucket="trigger")]
        kept, scope = geo.filter_for_lead(arts, config.COUNTRIES["Global"])
        assert kept == arts

    def test_topic_articles_are_industry_bucket_so_no_geo_is_kept(self):
        """collect_topic results come back bucket='industry'; a topic article
        found by a country-anchored query survives the filter even with no
        country word in its text."""
        import geo

        a = _art(title="Ship breaking market consolidates", bucket="industry",
                 query="ship breaking India")
        kept, _ = geo.filter_for_lead([a], _country("India"))
        assert kept == [a]


# ── collect.anchored_topic_queries ───────────────────────────────────────────
class TestAnchoredTopicQueries:
    def test_global_gets_no_anchored_queries(self):
        from collect import anchored_topic_queries

        assert anchored_topic_queries("Stem Cells Market", None) == []
        assert anchored_topic_queries(
            "Stem Cells Market", {"name": "Global"}) == []

    def test_country_name_appended_to_every_query(self):
        from collect import anchored_topic_queries, topic_queries

        qs = topic_queries("Stem Cells Market")
        aq = anchored_topic_queries("Stem Cells Market", {"name": "Morocco"})
        assert len(aq) == len(qs)
        assert f"{topic_queries('Stem Cells Market')[0]} Morocco" in aq

    def test_demonym_form_for_two_queries(self):
        from collect import anchored_topic_queries, topic_queries

        aq = anchored_topic_queries(
            "Autonomous Vessels", {"name": "India", "nationality": "Indian"})
        assert "autonomous vessels indian" in [q.lower() for q in aq]
        assert "autonomous vessels industry indian" in [q.lower() for q in aq]

    def test_georgia_country_anchor_present(self):
        """Bare "Georgia" is not a detection token — the anchor match must come
        from the profile's own alias list, which carries the country name."""
        import config
        import geo

        aq_len = len(geo.filter_for_lead(
            [_art(title="Pipeline deal signed", bucket="industry",
                  query="pipelines Georgia")],
            config.get_country("Georgia"))[0]) == 1
        assert aq_len


# ── validate_brief geography check ───────────────────────────────────────────
class TestValidateGeography:
    def _profile(self):
        import types

        return types.SimpleNamespace(
            company="TestCo", short="TestCo", recipient="there",
            tagline="t", value_prop="industrial widgets for manufacturers",
            industries=["manufacturing"], competitors=["RivalCo"], market="India")

    def _html(self, customer, industry, competition):
        return f"""
        <html><body>
        <div>● Potential Customer · widgets<br>{customer}</div>
        <div>● Industry Outlook · widgets<br>{industry}</div>
        <div>● Competition Outlook · widgets<br>{competition}</div>
        </body></html>"""

    def test_off_country_warn(self):
        import validate_brief as vb

        html = self._html(
            customer="TestCo buyer in Mumbai raises funding for Indian plants",
            industry="French banks in Paris adopt instant payment rails.",
            competition="TestCo rival RivalCo wins work in Bengaluru, India.")
        fails, warns = vb.check_html(html, self._profile(),
                                     lead_country="India")
        assert any("OFF-COUNTRY" in w and "Industry Outlook" in w for w in warns)
        assert not any("MIXED GEOGRAPHY" in f for f in fails)

    def test_mixed_geography_fail(self):
        import validate_brief as vb

        html = self._html(
            customer="TestCo buyer in Mumbai raises funding for Indian plants.",
            industry="French banks in Paris adopt instant payment rails.",
            competition="German rival in Berlin expands its card business.")
        fails, warns = vb.check_html(html, self._profile(),
                                     lead_country="India")
        assert any("MIXED GEOGRAPHY" in f for f in fails)
        assert any("OFF-COUNTRY" in w and "Industry Outlook" in w for w in warns)
        assert any("OFF-COUNTRY" in w and "Competition Outlook" in w
                   for w in warns)

    def test_explicit_link_is_allowed(self):
        import validate_brief as vb

        html = self._html(
            customer="TestCo buyer in Mumbai raises funding for Indian plants.",
            industry="German suppliers expand in India to serve Bengaluru.",
            competition="TestCo rival RivalCo wins work in Bengaluru, India.")
        fails, warns = vb.check_html(html, self._profile(),
                                     lead_country="India")
        assert not any("OFF-COUNTRY" in w for w in warns)
        assert not any("MIXED GEOGRAPHY" in f for f in fails)

    def test_check_skipped_without_lead_country(self):
        import validate_brief as vb

        html = self._html(
            customer="TestCo buyer raises funding for plants.",
            industry="French banks in Paris adopt instant payment rails.",
            competition="German rival expands its card business.")
        fails, warns = vb.check_html(html, self._profile())
        assert not any("OFF-COUNTRY" in w for w in warns)
        assert not any("MIXED GEOGRAPHY" in f for f in fails)

    def _profile_recipient(self, name):
        import types

        p = self._profile()
        p.recipient = name
        return p

    def test_salutation_and_first_name_city_ignored(self):
        """2026-11 live run: recipient first name "Sanaa" was read as Yemen's
        capital Sanaa. The salutation line and the first-name word must never
        count as geography."""
        import validate_brief as vb

        prof = self._profile_recipient("Sanaa Kaur")
        html = self._html(
            customer="Hello, Sanaa. TestCo buyer in Mumbai raises funding for Indian plants.",
            industry="Shipments through Sanaa port were steady this quarter.",
            competition="TestCo rival RivalCo wins work in Bengaluru, India.")
        fails, warns = vb.check_html(html, prof, lead_country="India")
        assert not any("OFF-COUNTRY" in w and "Industry Outlook" in w
                       for w in warns)
        assert not any("MIXED GEOGRAPHY" in f for f in fails)

    def test_prepared_for_and_brief_header_ignored(self):
        """Header/footer boilerplate lines are not news geography."""
        import validate_brief as vb

        prof = self._profile_recipient("Sanaa Kaur")
        html = self._html(
            customer="TestCo buyer in Mumbai raises funding for Indian plants.",
            industry="Prepared for Sanaa Kaur · Market Signal Brief · Indian ports steady.",
            competition="TestCo rival RivalCo wins work in Bengaluru, India.")
        fails, warns = vb.check_html(html, prof, lead_country="India")
        assert not any("MIXED GEOGRAPHY" in f for f in fails)
        assert not any("OFF-COUNTRY" in w for w in warns)
