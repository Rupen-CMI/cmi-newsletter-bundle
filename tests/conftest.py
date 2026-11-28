import pytest


@pytest.fixture(autouse=True)
def _no_google_news_lookup(monkeypatch, request):
    """Tests never hit the live Google News resolver (tests/test_text_fetch.py opts in)."""
    if request.node.get_closest_marker("gn_live_ok"):
        return
    import collect
    monkeypatch.setattr(collect, "_resolve_gnews", lambda url, timeout=8: "")


def pytest_configure(config):
    config.addinivalue_line("markers", "gn_live_ok: allow the real collect._resolve_gnews")
