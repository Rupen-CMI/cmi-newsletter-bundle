from datetime import datetime, timezone

import pytest

import brief_builder
import collect

GN = "https://news.google.com/rss/articles/CBMiABC?oc=5"


def _a(url=GN, **kw):
    a = collect.Article(title="Firm opens new plant", url=url, source="Wire",
                        published=datetime.now(timezone.utc), summary="Firm opens new plant")
    for k, v in kw.items():
        setattr(a, k, v)
    return a


@pytest.mark.gn_live_ok
def test_resolve_gnews_parses_batchexecute(monkeypatch):
    class R:
        def __init__(self, text, code=200):
            self.text, self.status_code = text, code
    page = '<c-wiz data-n-a-sg="SIG" data-n-a-ts="1700000000"></c-wiz>'
    inner = '["garturlres","https://publisher.example/story",1]'
    import json
    body = ")]}'\n\n" + json.dumps([["wrb.fr", "Fbv4je", inner, None, None, None, "generic"],
                                     ["di", 1], ["af.httprm", 1, "x", 1]])
    monkeypatch.setattr(collect, "_GN_CACHE", {})
    monkeypatch.setattr(collect, "_GN_STATE", {"last": 0.0, "until": 0.0})
    monkeypatch.setattr(collect, "_GN_GAP", 0.0)
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: R(page))
    monkeypatch.setattr(requests, "post", lambda *a, **k: R(body))
    assert collect._resolve_gnews(GN) == "https://publisher.example/story"


@pytest.mark.gn_live_ok
def test_resolve_gnews_429_pauses(monkeypatch):
    class R:
        text, status_code = "", 429
    monkeypatch.setattr(collect, "_GN_CACHE", {})
    monkeypatch.setattr(collect, "_GN_STATE", {"last": 0.0, "until": 0.0})
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: R())
    assert collect._resolve_gnews(GN) == ""
    assert collect._GN_STATE["until"] > __import__("time").time()
    other = "https://news.google.com/rss/articles/CBMiOTHER"
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called during pause")))
    assert collect._resolve_gnews(other) == ""


def test_article_text_uses_resolved_url(monkeypatch):
    seen = {}
    monkeypatch.setattr(collect, "_resolve_gnews", lambda url, timeout=8: "https://pub.example/x")
    def page(url, timeout):
        seen["url"] = url
        return "The firm said on Monday it will open a new plant employing 300 people. " * 3
    monkeypatch.setattr(collect, "_page_text", page)
    a = _a()
    text, src = collect._article_text_src(a, 5)
    assert src == "page" and seen["url"] == "https://pub.example/x" and a.url == "https://pub.example/x"


def test_fill_text_skips_articles_with_text(monkeypatch):
    calls = []
    monkeypatch.setattr(collect, "_fill_one", lambda a, timeout=8: calls.append(a) or 1)
    done, todo = _a(text_source="page"), _a()
    assert collect.fill_text([done, todo, None]) == 1
    assert calls == [todo]


def test_ensure_text_off_in_frozen_replay(monkeypatch):
    calls = []
    monkeypatch.setattr(collect, "fill_text", lambda arts, max_workers=4: calls.append(arts))
    monkeypatch.setenv("CMI_EVIDENCE_DIR", "x")
    brief_builder._ensure_text([_a()])
    assert calls == []
    monkeypatch.delenv("CMI_EVIDENCE_DIR")
    monkeypatch.setattr(brief_builder.config, "FETCH_ARTICLE_TEXT", True)
    brief_builder._ensure_text([_a()])
    assert len(calls) == 1


def test_enrich_texts_does_not_resolve(monkeypatch):
    calls = []
    monkeypatch.setattr(collect, "_fill_one", lambda a, timeout=8, resolve=True: calls.append(resolve) or 0)
    a = _a(bucket="trigger")
    collect.enrich_texts([a])
    assert calls == [False]
