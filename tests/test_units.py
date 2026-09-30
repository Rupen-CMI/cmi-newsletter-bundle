"""Unit tests for the pieces of the pipeline that carry real logic.

    pytest tests/ -q

Every test here runs offline: no GPU, no network, no SMTP. Each one uses its own
temporary database, so running the suite never touches cmi_newsletter.db.

The coverage is deliberately weighted toward the things that have actually broken in
this project rather than toward line count:

  - sqlite connections used across threads (failed all 20 domains once)
  - the send gate, which is the only irreversible step
  - free-mail domains being mistaken for company sites (the ImmunoACT misprofile)
  - the mail provider switch, since a wrong default sends real email the wrong way
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db as dbmod  # noqa: E402
import enrich  # noqa: E402


@pytest.fixture()
def conn(tmp_path):
    """A fresh database per test, so nothing leaks between tests or into production."""
    c = dbmod.get_conn(str(tmp_path / "t.db"))
    dbmod.init_db(c)
    yield c
    c.close()


def _profile(**kw):
    """A Profile with sane defaults; tests override only the field under test."""
    import tenant

    base = dict(company="FamPay", short="FamPay", recipient="Aditya",
                tagline="teen payments", value_prop="prepaid cards for teenagers",
                industries=["fintech"], competitors=["Slice", "Jupiter"],
                industry_keywords={"fintech": ["payments", "upi"]},
                trigger_queries=["fampay funding"], industry_queries=["upi growth"],
                market="India")
    base.update(kw)
    return tenant.Profile(**base)


# ── schema and helpers ────────────────────────────────────────────────────────
class TestSchema:
    def test_init_db_is_idempotent(self, conn):
        """init_db runs on every connection open, so a second call must be harmless."""
        dbmod.init_db(conn)
        dbmod.init_db(conn)
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"leads", "sends", "briefs", "runs", "company_intel"} <= names

    def test_upsert_lead_updates_rather_than_duplicating(self, conn):
        dbmod.upsert_lead(conn, email="a@b.com", name="A", company="Old")
        dbmod.upsert_lead(conn, email="a@b.com", name="A", company="New")
        rows = conn.execute("SELECT company FROM leads WHERE email='a@b.com'").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "New"

    def test_unsubscribe_is_respected(self, conn):
        assert dbmod.is_unsubscribed(conn, "x@y.com") is False
        dbmod.mark_unsubscribed(conn, "x@y.com", reason="test")
        assert dbmod.is_unsubscribed(conn, "x@y.com") is True


class TestIntelCache:
    def test_roundtrip_preserves_json_list_fields(self, conn):
        """industries/competitors/product_terms are stored as JSON text; a silent
        failure here would hand downstream code a string where it expects a list."""
        dbmod.put_intel(conn, domain="fampay.in", company="FamPay",
                        value_prop="teen cards", industries=["fintech"],
                        competitors=["Slice"], product_terms=["upi"],
                        raw_json="{}", source="homepage", ttl_days=90)
        got = dbmod.get_intel(conn, "fampay.in")
        assert got["industries"] == ["fintech"]
        assert got["competitors"] == ["Slice"]
        assert got["source"] == "homepage"

    def test_expired_row_is_not_returned(self, conn):
        """A negative TTL is the only way to simulate staleness without sleeping."""
        dbmod.put_intel(conn, domain="stale.com", company="S", source="thin",
                        ttl_days=-1)
        assert dbmod.get_intel(conn, "stale.com") is None

    def test_thin_domains_lists_only_thin(self, conn):
        dbmod.put_intel(conn, domain="thin.com", source="thin", ttl_days=7)
        dbmod.put_intel(conn, domain="rich.com", source="homepage", ttl_days=90)
        assert enrich.thin_domains(conn) == ["thin.com"]


# ── the failure that cost the most time ───────────────────────────────────────
class TestThreadSafety:
    def test_fetch_facts_holds_no_connection(self):
        """_fetch_facts must be callable from a pool. It was not, once: a shared
        sqlite connection inside the workers failed all 20 domains with
        'SQLite objects created in a thread can only be used in that thread'."""
        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(enrich._fetch_facts, "example.invalid", "X")
                    for _ in range(4)]
            for f in futs:
                assert f.result()["domain"] == "example.invalid"

    def test_unreachable_site_degrades_to_thin(self):
        """An unreachable site is a normal outcome, not an exception: enrichment is
        best-effort and must fall back to name-only inference."""
        facts = enrich._fetch_facts("no-such-host.invalid", "Nope")
        assert facts["source"] == "thin"


class TestDomainOf:
    @pytest.mark.parametrize("email", [
        "adityajha8906@gmail.com", "x@yahoo.com", "y@outlook.com", "z@rediffmail.com"])
    def test_free_mail_yields_no_domain(self, email):
        """gmail.com is not FamPay's website. Guessing from a free-mail address is
        how a lead once got profiled from its display name alone."""
        assert enrich.domain_of(email=email) == ""

    def test_company_email_yields_domain(self):
        assert enrich.domain_of(email="aditya@fampay.in") == "fampay.in"

    def test_website_beats_email_and_strips_www(self):
        assert enrich.domain_of(email="a@gmail.com",
                                website="https://www.fampay.in/about") == "fampay.in"


# ── the only irreversible step ────────────────────────────────────────────────
class TestSendGate:
    def _brief(self, conn, email, validation, status="generated"):
        bid = dbmod.save_brief(conn, email=email, company="C", kind="report",
                               html="<p>x</p>", status=status,
                               validation=json.dumps(validation))
        return bid

    def test_clean_brief_is_sendable(self, conn):
        self._brief(conn, "clean@x.com", {"fails": [], "warns": []})
        assert len(dbmod.sendable_briefs(conn)) == 1

    def test_warning_does_not_block(self, conn):
        """A warning is a judgement a human can overrule, not a defect."""
        self._brief(conn, "warn@x.com", {"fails": [], "warns": ["thin section"]})
        assert len(dbmod.sendable_briefs(conn)) == 1

    def test_hard_failure_blocks(self, conn):
        self._brief(conn, "fail@x.com", {"fails": ["missing 2 sections"], "warns": []})
        assert dbmod.sendable_briefs(conn) == []

    def test_already_sent_is_not_resent(self, conn):
        self._brief(conn, "sent@x.com", {"fails": []}, status="sent")
        assert dbmod.sendable_briefs(conn) == []

    def test_malformed_validation_json_does_not_crash_the_gate(self, conn):
        """Fail-open on unparseable JSON is deliberate, but it must not raise --
        one bad row would otherwise stop an entire batch from sending."""
        dbmod.save_brief(conn, email="bad@x.com", company="C", kind="report",
                         html="<p>x</p>", status="generated", validation="{not json")
        assert len(dbmod.sendable_briefs(conn)) == 1

    def test_run_id_scopes_the_gate(self, conn):
        rid = dbmod.start_run(conn, workers=2, kind="report")
        dbmod.save_brief(conn, email="a@x.com", company="C", kind="report",
                         html="<p>x</p>", status="generated",
                         validation="{}", run_id=rid)
        dbmod.save_brief(conn, email="b@x.com", company="C", kind="report",
                         html="<p>x</p>", status="generated", validation="{}")
        assert len(dbmod.sendable_briefs(conn, run_id=rid)) == 1
        assert len(dbmod.sendable_briefs(conn)) == 2


# ── structural validation ─────────────────────────────────────────────────────
# Sections are detected by "<heading> · <company>" and each body must carry more
# than 25 words, so a realistic fixture is required -- a bare <h2> matches nothing.
_LOREM = ("The company is expanding capacity across the region this quarter and has "
          "signalled further investment, with management pointing to sustained demand "
          "from domestic buyers and a broader order book than the prior year period.")

GOOD_HTML = f"""
<html><body>
<h2>Potential Customer · Razorpay</h2><p>{_LOREM}</p>
<h2>Industry Outlook · Payments</h2><p>{_LOREM}</p>
<h2>Competition Outlook · Slice</h2><p>{_LOREM}</p>
</body></html>
"""

ONE_SECTION_HTML = f"""
<html><body>
<h2>Potential Customer · Razorpay</h2><p>{_LOREM}</p>
</body></html>
"""


class TestStructureValidation:
    def test_all_three_sections_present_passes(self):
        import validate_brief

        fails, _ = validate_brief.check_html(GOOD_HTML, _profile())
        assert not [f for f in fails if "section" in f.lower()]

    def test_two_missing_sections_fail(self):
        """One missing section warns, two fail: some markets genuinely have no
        rival news, but a brief missing most of its body is not sendable."""
        import validate_brief

        fails, warns = validate_brief.check_html(ONE_SECTION_HTML, _profile())
        assert fails, "a brief with only one section must hard-fail"

    def test_visible_text_strips_markup(self):
        import validate_brief

        text = validate_brief.visible_text("<p>Hello <b>world</b></p>")
        assert "Hello" in text and "world" in text and "<b>" not in text

    def test_name_overlap_matches_variants(self):
        """'Bharat Forge Ltd' and 'Bharat Forge' are the same company; a strict
        equality check here would flag a correct brief."""
        import validate_brief

        assert validate_brief.name_overlap("Bharat Forge Ltd", "Bharat Forge")
        assert not validate_brief.name_overlap("Bharat Forge", "Tata Steel")


# ── mail provider selection ───────────────────────────────────────────────────
class TestMailProvider:
    def test_sendgrid_wins_when_key_present(self, monkeypatch):
        import mailer

        monkeypatch.setattr(mailer, "_env",
                            lambda n, d="": {"SENDGRID_WEB_API": "SG.x"}.get(n, d))
        assert mailer._provider() == "sendgrid"

    def test_smtp_when_no_key(self, monkeypatch):
        import mailer

        monkeypatch.setattr(mailer, "_env", lambda n, d="": d)
        assert mailer._provider() == "smtp"

    def test_explicit_override_wins(self, monkeypatch):
        """CMI_MAIL_PROVIDER must be able to force SMTP even with a key set --
        otherwise there is no way to test the SMTP path once SendGrid is configured."""
        import mailer

        monkeypatch.setattr(mailer, "_env", lambda n, d="": {
            "SENDGRID_WEB_API": "SG.x", "CMI_MAIL_PROVIDER": "smtp"}.get(n, d))
        assert mailer._provider() == "smtp"

    def test_sendgrid_payload_shape(self, monkeypatch):
        """The v3 API rejects a malformed body with a 400 that is easy to misread as
        an auth problem, so the shape is worth pinning."""
        import urllib.request

        import mailer

        seen = {}

        def fake_urlopen(req, timeout=30):
            seen["url"] = req.full_url
            seen["auth"] = req.headers.get("Authorization", "")
            seen["body"] = json.loads(req.data)

            class R:
                status = 202

                def __enter__(s): return s

                def __exit__(s, *a): pass
            return R()

        monkeypatch.setattr(mailer, "_env", lambda n, d="": {
            "SENDGRID_WEB_API": "SG.testkey", "SENDER_EMAIL": "from@cmi.com",
            "SENDER_NAME": "CMI"}.get(n, d))
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        mailer._send_via_sendgrid("<h1>hi</h1>", "to@x.com", "Subj", ["r@cmi.com"])

        assert seen["url"] == "https://api.sendgrid.com/v3/mail/send"
        assert seen["auth"] == "Bearer SG.testkey"
        body = seen["body"]
        assert body["personalizations"][0]["to"] == [{"email": "to@x.com"}]
        assert body["from"]["email"] == "from@cmi.com"
        assert [c["type"] for c in body["content"]] == ["text/plain", "text/html"]

    def test_missing_sender_email_is_a_clear_error(self, monkeypatch):
        """SendGrid answers an unverified/absent sender with 403 'access forbidden',
        which says nothing useful. Catch it before the request goes out."""
        import mailer

        monkeypatch.setattr(mailer, "_env", lambda n, d="": {
            "SENDGRID_WEB_API": "SG.x"}.get(n, d))
        with pytest.raises(RuntimeError, match="SENDER_EMAIL"):
            mailer._send_via_sendgrid("<p>x</p>", "to@x.com", "S", [])


# ── scoring ───────────────────────────────────────────────────────────────────
class TestScoring:
    def _article(self, hours_old: float, title="UPI payments volume rises",
                 bucket="industry"):
        import datetime as dt

        from collect import Article

        return Article(title=title, url="https://x.com/1", source="Test",
                       published=dt.datetime.now(dt.timezone.utc)
                       - dt.timedelta(hours=hours_old),
                       bucket=bucket, query="q")

    def test_freshness_decays_with_age(self):
        import score

        assert score.freshness(self._article(1)) > score.freshness(self._article(200))

    def test_competitor_lookback_spans_six_months(self):
        """Competitor news is scarce; a 24h window left Competition Outlook empty.
        The 182-day band is what fixed it, so it needs to stay non-zero."""
        import score

        assert score.freshness(self._article(24 * 150, bucket="competitor")) > 0


# ── links ─────────────────────────────────────────────────────────────────────
class TestLinks:
    """The brief cites sources in prose ("according to a Livemint report"). Those
    citations were not linked, and the unsubscribe href was a literal "#"."""

    def test_demo_url_is_the_live_analyst_path(self):
        """It is /talkwithanalyst, NOT /talktoanalyst -- the latter 404s. Checked
        against the live site; pinned here so a plausible-looking 'fix' cannot
        silently break every CTA in every brief."""
        import compose

        assert compose.DEMO_URL.endswith("/talkwithanalyst")

    def test_unsubscribe_is_not_a_dead_anchor(self):
        """A '#' unsubscribe is worse than none: bulk filters look for a working
        opt-out and a dead one invites spam reports."""
        import compose

        assert compose.UNSUB_URL.startswith("http")

    def test_prospect_card_carries_source_url(self):
        import compose

        sec = compose._prospect("Competition Outlook",
                                {"company_name": "PhonePe", "narrative_html": "Body.",
                                 "source_url": "https://news.google.com/X",
                                 "source_name": "Livemint"},
                                ("a", "b"))
        assert sec["source_url"] == "https://news.google.com/X"
        assert sec["source_name"] == "Livemint"

    def test_missing_source_url_stays_empty_not_hash(self):
        """No URL must render as plain text, never as another dead anchor."""
        import compose

        sec = compose._prospect("Competition Outlook",
                                {"company_name": "X", "narrative_html": "B."},
                                ("a", "b"))
        assert sec["source_url"] == ""

    def test_emailify_preserves_the_source_link(self):
        """emailify REBUILDS the provenance row rather than copying it, so a link
        added in the template is dropped unless it is re-emitted explicitly -- and
        the loss shows only in the email, not in the browser file.

        Targets _analyst directly: that is the function that rebuilds the row, so
        this pins the behaviour without depending on the whole document shape.
        """
        import bs4

        import emailify as E

        soup = bs4.BeautifulSoup(
            "<div class='analyst'><div class='tag'>Analyst Opinion</div>"
            "<ul><li>b</li></ul>"
            "<div class='prov'><span class='src'>CMI Intelligence</span>"
            "<span>Sep 23, 2026</span>"
            "<a class='srclink' href='https://news.google.com/ZZTEST'>Source: Livemint</a>"
            "<span class='verified'>ok</span></div></div>", "html.parser")
        out = E._analyst(soup.select_one(".analyst"))
        assert "ZZTEST" in out
        assert "Livemint" in out

    def test_template_uses_brand_unsub_url(self):
        import io
        import os

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        t = io.open(os.path.join(root, "templates", "cmi_brief.html"),
                    encoding="utf-8").read()
        assert "brand.unsub_url" in t
        assert 'class="unsub" href="#"' not in t


# ── geo filtering ─────────────────────────────────────────────────────────────
class TestGeo:
    def _art(self, title="X", summary="", source="Reuters", url="https://x.com/a"):
        import datetime as dt

        from collect import Article

        return Article(title=title, url=url, source=source, summary=summary,
                       published=dt.datetime.now(dt.timezone.utc), bucket="industry")

    def test_global_matches_everything(self):
        import config
        import geo

        assert geo.matches(self._art(title="Anything at all"),
                           config.COUNTRIES["Global"])

    def test_country_matches_on_alias_in_title(self):
        import config
        import geo

        assert geo.matches(self._art(title="Bharat Forge expands in India"),
                           config.get_country("India"))

    def test_country_rejects_unrelated_article(self):
        import config
        import geo

        assert not geo.matches(self._art(title="Brazilian coffee prices climb",
                                         source="Reuters", url="https://x.com/a"),
                               config.get_country("India"))

    def test_filter_falls_back_rather_than_returning_nothing(self):
        """A thin real newsletter is worse than an off-country dilution: when too
        few articles match, the full set is used and the flag reports it."""
        import config
        import geo

        arts = [self._art(title="Brazilian coffee prices climb") for _ in range(5)]
        kept, used_fallback = geo.filter_country(arts, config.get_country("India"))
        assert kept, "filtering must never return an empty list"
        assert used_fallback is True


class TestCountryResolution:
    def test_known_country_resolves(self):
        import config

        assert config.get_country("India")["name"] == "India"

    def test_unknown_and_empty_fall_back_to_global(self):
        """Leads carry free-text countries; an unrecognised one must degrade to
        Global rather than raising and losing the brief."""
        import config

        assert config.get_country("Atlantis")["name"] == "Global"
        assert config.get_country("")["name"] == "Global"


# ── LLM response handling ─────────────────────────────────────────────────────
class TestJsonParsing:
    def test_plain_json(self):
        import llm

        assert llm._parse_json('{"a": 1}') == {"a": 1}

    def test_fenced_json_is_tolerated(self):
        """Models wrap output in ```json fences despite being told not to."""
        import llm

        assert llm._parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_raw_control_character_is_repaired(self):
        """A literal newline inside a string value is the single most common cause
        of a dropped section; it is repaired rather than discarded."""
        import llm

        assert llm._parse_json('{"a": "line1\nline2"}')["a"].startswith("line1")

    def test_unrepairable_json_raises(self):
        """chat_json catches this and falls back to `default` -- but the parser
        itself must not silently return an empty dict and hide the failure."""
        import json as _json

        import llm

        with pytest.raises((_json.JSONDecodeError, ValueError)):
            llm._parse_json("not json at all")


# ── editorial noise filters ───────────────────────────────────────────────────
class TestResearchPR:
    def test_rival_research_firm_is_flagged(self):
        """CMI sells market reports, so a rival research firm's PR is competitor
        noise, not a market signal."""
        import config

        firm = next(iter(config.RESEARCH_FIRM_SOURCES))
        assert config.is_research_pr(firm, "Some market to reach $5bn by 2030")

    def test_generic_report_pr_is_flagged_by_title(self):
        import config

        assert config.is_research_pr(
            "PRNewswire",
            "Global Widget Market Size to Reach USD 5.2 Billion by 2032")

    def test_real_news_is_not_flagged(self):
        import config

        assert not config.is_research_pr("Reuters",
                                         "Bharat Forge wins defence order")


# ── validator role logic ──────────────────────────────────────────────────────
class TestValidatorRoles:
    def test_section_company_extracts_the_flagged_name(self):
        import validate_brief as vb

        text = "Competition Outlook · PhonePe\nPhonePe launched a UPI service."
        assert "PhonePe" in vb.section_company(text, "Competition Outlook")

    def test_norm_ignores_suffixes_and_case(self):
        import validate_brief as vb

        assert vb.norm("Bharat Forge Ltd.") == vb.norm("bharat forge ltd")

    def test_curated_competitor_in_customer_slot_is_caught(self):
        """The failure this exists for: a known RIVAL appearing as a Potential
        Customer. Slice is in the fixture profile's competitor list."""
        import validate_brief as vb

        html = (f"<html><body>"
                f"<h2>Potential Customer \u00b7 Slice</h2><p>{_LOREM}</p>"
                f"<h2>Industry Outlook \u00b7 Payments</h2><p>{_LOREM}</p>"
                f"<h2>Competition Outlook \u00b7 Jupiter</h2><p>{_LOREM}</p>"
                f"</body></html>")
        fails, warns = vb.check_html(html, _profile())
        assert fails or warns, "a curated rival in the customer slot must be flagged"


# ── enrichment ladder ─────────────────────────────────────────────────────────
class TestEnrichmentLadder:
    def test_curated_override_wins_over_scraping(self):
        """company_overrides.json is hand-checked and must beat anything fetched:
        naming competitors is factual recall a 7B answers inconsistently."""
        facts = enrich._fetch_facts("fampay.in", "FamPay")
        assert facts["source"] == "curated"
        assert "Junio" in facts["competitors"] or "Slice" in facts["competitors"]

    def test_curated_lookup_is_case_insensitive(self):
        import tenant

        assert tenant._overrides_for("BHARAT FORGE")
        assert tenant._overrides_for("bharat forge")


# ── vm connection ─────────────────────────────────────────────────────────────
class TestVmConnect:
    """The machine id changes on EVERY resume, so nothing may be keyed on it."""

    def test_finds_instance_by_name_not_id(self, monkeypatch):
        import vm_connect

        rows = [{"name": "other", "machine_id": 1},
                {"name": "cmi-serve-v1", "machine_id": 999, "status": "Running"}]
        monkeypatch.setattr(vm_connect, "_jl",
                            lambda *a, **k: (0, json.dumps(rows)))
        assert vm_connect.find_instance("cmi-serve-v1")["machine_id"] == 999

    def test_missing_instance_returns_none_not_raises(self, monkeypatch):
        """status() is a health check; it must degrade rather than explode."""
        import vm_connect

        monkeypatch.setattr(vm_connect, "_jl", lambda *a, **k: (0, "[]"))
        assert vm_connect.find_instance("nope") is None

    def test_endpoint_is_empty_while_paused(self):
        """`endpoints` is null on a paused box -- the usual reason the URL
        'disappears'."""
        import vm_connect

        assert vm_connect.endpoint_of({"endpoints": None}) == ""

    def test_endpoint_picks_the_serving_port(self):
        import vm_connect

        inst = {"endpoints": {"Port 7007": "https://a.example.net",
                              "Port 6006": "https://b.example.net"}}
        assert vm_connect.endpoint_of(inst) == "https://b.example.net/v1"

    def test_is_serving_requires_a_real_completion(self, monkeypatch):
        """/v1/models answering is NOT enough: it responds while the engine is still
        loading, and trusting it produced a full batch of empty briefs that passed."""
        import urllib.request

        import vm_connect

        class R:
            def __enter__(s): return s
            def __exit__(s, *a): pass
            def read(s): return b'{"object":"list","data":[{"id":"cmi-7b"}]}'

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: R())
        assert vm_connect.is_serving("https://x/v1") is False

    def test_is_serving_true_on_content(self, monkeypatch):
        import urllib.request

        import vm_connect

        class R:
            def __enter__(s): return s
            def __exit__(s, *a): pass
            def read(s): return b'{"choices":[{"message":{"content":"OK"}}]}'

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: R())
        assert vm_connect.is_serving("https://x/v1") is True

    def test_unreachable_endpoint_is_not_serving(self):
        import vm_connect

        assert vm_connect.is_serving("http://127.0.0.1:1/v1", timeout=2) is False

    def test_empty_url_is_not_serving(self):
        import vm_connect

        assert vm_connect.is_serving("") is False

    def test_write_base_url_replaces_rather_than_appends(self, tmp_path, monkeypatch):
        """Repeated healing must not leave a stack of stale LOCAL_LLM_BASE_URL lines:
        dotenv takes the last one, so a shadowed old value is very hard to spot."""
        import vm_connect

        env = tmp_path / ".env"
        env.write_text("A=1\nLOCAL_LLM_BASE_URL=https://old/v1\nB=2\n", encoding="utf-8")
        monkeypatch.setattr(os.path, "dirname", lambda _p: str(tmp_path))
        vm_connect.write_base_url("https://new/v1")
        vm_connect.write_base_url("https://newer/v1")
        text = env.read_text(encoding="utf-8")
        assert text.count("LOCAL_LLM_BASE_URL=") == 1
        assert "https://newer/v1" in text
        assert "A=1" in text and "B=2" in text

    def test_gpu_ladder_prefers_l4(self):
        """L4 is the tuned target; A30 is the same 24GB at the same spot price, and
        A100 is ~2x the cost -- so the order matters financially."""
        import vm_connect

        assert vm_connect.GPU_LADDER[0] == "L4"
        assert vm_connect.GPU_LADDER.index("A30") < vm_connect.GPU_LADDER.index("A100")


# ── api authentication ────────────────────────────────────────────────────────
class TestApiAuth:
    """The API can send real email, so exposing it unauthenticated is the worst
    failure available here. These pin the guards."""

    def _client(self, monkeypatch, key=""):
        import importlib

        monkeypatch.setenv("CMI_API_KEY", key)
        import api

        importlib.reload(api)
        from fastapi.testclient import TestClient

        return TestClient(api.app)

    def test_open_when_no_key_configured(self, monkeypatch):
        """Backwards compatible: the localhost-only default needs no setup."""
        c = self._client(monkeypatch, "")
        assert c.get("/stats").status_code == 200

    def test_rejects_missing_key(self, monkeypatch):
        c = self._client(monkeypatch, "s3cret")
        assert c.get("/stats").status_code == 401

    def test_rejects_wrong_key(self, monkeypatch):
        c = self._client(monkeypatch, "s3cret")
        assert c.get("/stats", headers={"X-API-Key": "nope"}).status_code == 401

    def test_accepts_x_api_key(self, monkeypatch):
        c = self._client(monkeypatch, "s3cret")
        assert c.get("/stats", headers={"X-API-Key": "s3cret"}).status_code == 200

    def test_accepts_bearer(self, monkeypatch):
        """Tools reach for different conventions; requiring one is a needless
        failure mode."""
        c = self._client(monkeypatch, "s3cret")
        r = c.get("/stats", headers={"Authorization": "Bearer s3cret"})
        assert r.status_code == 200

    def test_health_stays_open(self, monkeypatch):
        """A liveness probe that needs a credential cannot be used as one."""
        c = self._client(monkeypatch, "s3cret")
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["auth_required"] is True

    def test_send_endpoint_is_also_guarded(self, monkeypatch):
        """The irreversible endpoint must not be reachable without the key."""
        c = self._client(monkeypatch, "s3cret")
        r = c.post("/briefs/1/send", json={"confirm": True})
        assert r.status_code == 401


# ── connection threading ──────────────────────────────────────────────────────
class TestConnThreading:
    """A sqlite3 connection may only be used on the thread that created it. This has
    broken twice: once via a shared connection in a ThreadPoolExecutor (all 20
    enrichment domains failed), and once via @st.cache_resource in the dashboard,
    which caches per PROCESS while Streamlit reruns on a new thread each pass."""

    def test_same_thread_reuses_one_connection(self, tmp_path):
        import db as dbmod

        p = str(tmp_path / "t.db")
        assert dbmod.get_conn(p) is dbmod.get_conn(p)

    def test_different_threads_get_different_connections(self, tmp_path):
        import db as dbmod

        p = str(tmp_path / "t.db")
        dbmod.init_db(dbmod.get_conn(p))
        main_id = id(dbmod.get_conn(p))
        seen: list[int] = []

        def worker():
            seen.append(id(dbmod.get_conn(p)))

        threads = [__import__("threading").Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert seen and main_id not in seen

    def test_query_from_another_thread_does_not_raise(self, tmp_path):
        """The exact traceback the dashboard produced."""
        import sqlite3
        import threading

        import db as dbmod

        p = str(tmp_path / "t.db")
        dbmod.init_db(dbmod.get_conn(p))
        err: list[Exception] = []

        def worker():
            try:
                c = dbmod.get_conn(p)
                dbmod.init_db(c)
                c.execute("SELECT COUNT(*) FROM leads").fetchone()
            except sqlite3.Error as e:  # noqa: BLE001
                err.append(e)

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert not err, f"cross-thread use raised: {err}"

    def test_pool_of_threads_can_all_read(self, tmp_path):
        import db as dbmod

        p = str(tmp_path / "t.db")
        dbmod.init_db(dbmod.get_conn(p))

        def read(_):
            c = dbmod.get_conn(p)
            dbmod.init_db(c)
            return c.execute("SELECT COUNT(*) FROM leads").fetchone()[0]

        with cf.ThreadPoolExecutor(max_workers=6) as ex:
            assert list(ex.map(read, range(6))) == [0] * 6

    def test_dashboard_does_not_cache_the_connection(self):
        """st.cache_resource caches for the whole PROCESS, which is the wrong scope
        for a thread-bound object. Pinned because the decorator looks harmless."""
        import io
        import os

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src = io.open(os.path.join(root, "dashboard.py"), encoding="utf-8").read()
        head = src[:src.index("def _conn()")]
        assert not head.rstrip().endswith("@st.cache_resource")


# ── send panel ────────────────────────────────────────────────────────────────
class TestSendSubject:
    """One definition shared by pipeline, api and the dashboard -- three copies drift."""

    def test_report_names_the_market(self):
        """'FamPay's Brief' throws away the most relevant word in the email."""
        import db as dbmod

        s = dbmod.brief_subject("FamPay", "report", "Payment Gateway Market")
        assert "FamPay" in s and "Payment Gateway Market" in s

    def test_signal_falls_back_to_generic(self):
        import db as dbmod

        assert dbmod.brief_subject("Bharat Forge", "signal", "").endswith("'s Brief")

    def test_report_without_topic_does_not_dangle(self):
        import db as dbmod

        assert dbmod.brief_subject("X", "report", "").endswith("'s Brief")

    def test_missing_company_does_not_produce_empty_subject(self):
        import db as dbmod

        assert "Your Company" in dbmod.brief_subject("", "signal", "")


class TestSendDryRun:
    def test_dry_run_sends_nothing_and_records_nothing(self, conn, monkeypatch):
        """The dashboard's default. It must be impossible for this path to email."""
        import mailer
        import pipeline

        called = []
        monkeypatch.setattr(mailer, "send_html",
                            lambda *a, **k: called.append(a))
        dbmod.save_brief(conn, email="a@x.com", company="C", kind="report",
                         html="<p>x</p>", status="generated", validation={"fails": []})
        monkeypatch.setattr(pipeline, "_log", lambda *_a, **_k: None)
        out = pipeline.stage_send(conn, run_id=None, campaign="t", dry_run=True,
                                  limit=100)
        assert called == [], "dry run must not call the mailer"
        assert out["sent"] == 0
        assert conn.execute("SELECT COUNT(*) FROM sends").fetchone()[0] == 0

    def test_unsubscribed_is_skipped_not_emailed(self, conn, monkeypatch):
        import mailer
        import pipeline

        called = []
        monkeypatch.setattr(mailer, "send_html", lambda *a, **k: called.append(a))
        monkeypatch.setattr(pipeline, "_log", lambda *_a, **_k: None)
        dbmod.save_brief(conn, email="gone@x.com", company="C", kind="signal",
                         html="<p>x</p>", status="generated", validation={"fails": []})
        dbmod.mark_unsubscribed(conn, "gone@x.com")
        out = pipeline.stage_send(conn, run_id=None, campaign="t", dry_run=False,
                                  limit=100)
        assert called == []
        assert out["sent"] == 0 and out["skipped"] == 1

    def test_hard_failure_never_reaches_the_send_list(self, conn, monkeypatch):
        import mailer
        import pipeline

        called = []
        monkeypatch.setattr(mailer, "send_html", lambda *a, **k: called.append(a))
        monkeypatch.setattr(pipeline, "_log", lambda *_a, **_k: None)
        dbmod.save_brief(conn, email="bad@x.com", company="C", kind="signal",
                         html="<p>x</p>", status="generated",
                         validation={"fails": ["missing sections"]})
        out = pipeline.stage_send(conn, run_id=None, campaign="t", dry_run=False,
                                  limit=100)
        assert called == [] and out["sent"] == 0

    def test_one_failure_does_not_stop_the_batch(self, conn, monkeypatch):
        """A bad address must not cost the rest of the run."""
        import mailer
        import pipeline

        sent = []

        def flaky(html, to, subject, **k):
            if to == "boom@x.com":
                raise RuntimeError("smtp exploded")
            sent.append(to)

        monkeypatch.setattr(mailer, "send_html", flaky)
        monkeypatch.setattr(pipeline, "_log", lambda *_a, **_k: None)
        for e in ("ok1@x.com", "boom@x.com", "ok2@x.com"):
            dbmod.save_brief(conn, email=e, company="C", kind="signal",
                             html="<p>x</p>", status="generated",
                             validation={"fails": []})
        out = pipeline.stage_send(conn, run_id=None, campaign="t", dry_run=False,
                                  limit=100)
        assert out["sent"] == 2 and out["failed"] == 1
        assert set(sent) == {"ok1@x.com", "ok2@x.com"}

    def test_explicit_ids_send_exactly_those(self, conn, monkeypatch):
        """The count on the Send button must be the truth.

        `limit` slices the sendable set BEFORE the unsubscribe check, so an
        unsubscribed brief inside the slice silently eats a slot -- the button said
        'Send 2' and one email went out. brief_ids avoids the whole class of bug.
        """
        import mailer
        import pipeline

        sent = []
        monkeypatch.setattr(mailer, "send_html",
                            lambda html, to, subj, **k: sent.append(to))
        monkeypatch.setattr(pipeline, "_log", lambda *_a, **_k: None)
        for e in ("live1@x.com", "live2@x.com", "unsub@x.com"):
            dbmod.save_brief(conn, email=e, company="C", kind="signal",
                             html="<p>x</p>", status="generated",
                             validation={"fails": []})
        dbmod.mark_unsubscribed(conn, "unsub@x.com")

        rows = dbmod.sendable_briefs(conn)
        live = [r for r in rows if not dbmod.is_unsubscribed(conn, r["email"])]
        out = pipeline.stage_send(conn, run_id=None, campaign="t", dry_run=False,
                                  limit=len(live),
                                  brief_ids=[r["id"] for r in live])
        assert out["sent"] == len(live) == len(sent)

    def test_limit_alone_can_undercount(self, conn, monkeypatch):
        """Pins the old behaviour so the reason brief_ids exists stays visible."""
        import mailer
        import pipeline

        sent = []
        monkeypatch.setattr(mailer, "send_html",
                            lambda html, to, subj, **k: sent.append(to))
        monkeypatch.setattr(pipeline, "_log", lambda *_a, **_k: None)
        for e in ("live1@x.com", "live2@x.com", "unsub@x.com"):
            dbmod.save_brief(conn, email=e, company="C", kind="signal",
                             html="<p>x</p>", status="generated",
                             validation={"fails": []})
        dbmod.mark_unsubscribed(conn, "unsub@x.com")
        # sendable_briefs is id DESC, so the unsubscribed row is first and is
        # included in limit=2 -- leaving only one real send.
        out = pipeline.stage_send(conn, run_id=None, campaign="t", dry_run=False,
                                  limit=2)
        assert out["sent"] == 1 and out["skipped"] == 1
