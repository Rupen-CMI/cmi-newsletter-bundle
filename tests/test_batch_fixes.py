"""Seven code fixes: nbsp cleaning, evidence-ranked shortlists, short names, site decoding,
HQ country check, free-research in evidence, prompt-leak scrub. No network, no LLM."""
from datetime import datetime, timezone
from types import SimpleNamespace

import brief_builder as bb
import collect
import config
import enrich
import pipeline
import tenant
from collect import Article


def _art(title, rel, rec, comp=0.5):
    return Article(title=title, url="u/" + title, source="s", published=datetime.now(timezone.utc),
                   summary="", query="q", bucket="industry", fingerprint=title,
                   composite=comp, ev_relevancy=rel, ev_recency=rec)


# 1
def test_clean_unescapes_nbsp():
    assert collect._clean("Headline&nbsp;&nbsp;Publisher") == "Headline Publisher"
    assert collect._clean("A\xa0\xa0B <b>x</b> &amp; y") == "A B x & y"


def test_cluster_ignores_nbsp_token():
    import cluster
    arts = [_art(f"nbsp energy australia {i}", 1, 1) for i in range(8)]
    for a in arts:
        a.title = f"Energy&nbsp;Australia solar nbsp {a.title}"
    label, _ = cluster.dominant_theme(arts)
    assert "nbsp" not in label.lower().split(" / ")


# 2
def test_floor_and_ranking():
    a, b, c = _art("a", 0.05, 1.0, 0.9), _art("b", 0.9, 0.9, 0.1), _art("c", 0.5, 0.2, 0.3)
    assert config.MIN_EV_RELEVANCY == 0.15
    assert [x.title for x in bb._rank_by_evidence([a, b, c])] == ["b", "c"]
    low = [_art("x", 0.0, 0.0, 0.2), _art("y", 0.0, 0.0, 0.8)]
    assert [x.title for x in bb._rank_by_evidence(low)] == ["y", "x"]   # fallback: current order
    assert bb._above_floor([a]) == [a]


# 3
def test_short_names():
    s = tenant._short_name
    assert s("Bluematterconsulting") == "Bluematterconsulting"
    assert s("Thetravelportfolio") == "Thetravelportfolio"
    assert s("Kirey (Insirio)") == "Kirey"
    assert s("Mitsubishi Hc Capital") == "MHC"
    assert s("BIL Suisse") == "BS"


# 4
def test_decode_response_meta_charset_and_apparent():
    html = '<html><head><meta charset="utf-8"></head><body>三菱HC</body></html>'
    r = SimpleNamespace(content=html.encode("utf-8"), encoding="ISO-8859-1",
                        apparent_encoding="utf-8", text=html.encode("utf-8").decode("latin-1"))
    assert "三菱HC" in tenant._decode_response(r)
    r2 = SimpleNamespace(content="三菱".encode("utf-8"), encoding=None,
                         apparent_encoding="utf-8", text="x")
    assert tenant._decode_response(r2) == "三菱"
    r3 = SimpleNamespace(content=b"hi", encoding="utf-8", apparent_encoding="x", text="hi")
    assert tenant._decode_response(r3) == "hi"


# 5
def test_hq_country():
    ckt = "CKT Co. 158 Joongdae-ro, Songpa-gu Seoul, Korea Tel 02-123"
    assert enrich.hq_country_from_site(ckt, "ckt.com") == "KR"
    assert enrich.hq_country_from_site("hello world", "firma.de") == "DE"
    assert enrich.hq_country_from_site("hello", "shop.co.uk") == "GB"
    assert enrich.hq_country_from_site("hello", "acme.com") == ""


def test_country_check_logs_and_returns():
    lines = []
    lead = {"company": "CKT", "website": "ckt.de"}
    out = pipeline._country_check(lead, "India", conn=None, log=lines.append)
    assert out == {"stated": "IN", "website": "DE"}
    assert lines and "country check: CKT stated IN, website says DE" in lines[0]
    assert pipeline._country_check(lead, "Germany", conn=None, log=lines.append) is None


# 6
def test_evidence_used_has_free_research():
    dbg = {"customer": {"company": "X", "article": "t", "url": "u", "source": "s",
                        "published": "p", "free_research": "Wikipedia text"}}
    assert pipeline._evidence_used({}, dbg)["Potential Customer"]["free_research"] == "Wikipedia text"


# 7
def test_signal_scrub():
    f = bb._clean_frag
    assert f("The SIGNAL states Group14 raised €150 million to expand.") == "Group14 raised €150 million to expand."
    assert f("The signal does not state any policy change, only the current separation.") == ""
    assert f("The free research shows CARES also refers to a cardiac arrest registry") == ""
    assert f("Per the SIGNAL, the plant opens. It adds 50 jobs.") == "The plant opens. It adds 50 jobs."
    assert f("Acme opened a plant. The signal gives no figures.") == "Acme opened a plant."
