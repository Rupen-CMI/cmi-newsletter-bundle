"""Streaming + transient-retry behaviour of the local LLM path (no network)."""
import os
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm  # noqa: E402


def _chunk(text=None, usage=None):
    ch = [NS(delta=NS(content=text))] if text is not None else []
    return NS(choices=ch, usage=usage)


class FakeClient:
    def __init__(self, deltas=None, full="", fail_first=0, exc=None):
        self.calls = []
        self.deltas, self.full, self.fail_first, self.exc = deltas or [], full, fail_first, exc
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        if len(self.calls) <= self.fail_first:
            raise self.exc
        if kw.get("stream"):
            return iter([_chunk(d) for d in self.deltas]
                        + [_chunk(None, NS(prompt_tokens=1, completion_tokens=2, total_tokens=3))])
        return NS(usage=None, choices=[NS(message=NS(content=self.full))])


def _setup(monkeypatch, client):
    monkeypatch.setattr(llm, "_local_client", client)
    monkeypatch.setattr(llm, "PROVIDER", "local")
    monkeypatch.setattr(llm, "_log_token_usage", lambda u: None)
    monkeypatch.setattr(llm, "_log_training_pair", lambda *a, **k: None)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.delenv("LOCAL_LLM_STREAM", raising=False)


def test_stream_assembles_deltas(monkeypatch):
    c = FakeClient(deltas=["Hel", "lo ", "world"])
    _setup(monkeypatch, c)
    assert llm.chat("s", "u") == "Hello world"
    assert c.calls[0]["stream"] is True


def test_chat_json_parses_streamed(monkeypatch):
    c = FakeClient(deltas=['{"a": ', '1, "b"', ': "x"}'])
    _setup(monkeypatch, c)
    assert llm.chat_json("s", "u") == {"a": 1, "b": "x"}


def test_switch_off_uses_old_path(monkeypatch):
    c = FakeClient(full="plain text")
    _setup(monkeypatch, c)
    monkeypatch.setenv("LOCAL_LLM_STREAM", "0")
    assert llm.chat("s", "u") == "plain text"
    assert "stream" not in c.calls[0]


def test_524_retried_then_success(monkeypatch):
    e = Exception("Error code: 524 - timeout")
    e.status_code = 524
    c = FakeClient(deltas=["ok"], fail_first=1, exc=e)
    _setup(monkeypatch, c)
    assert llm.chat("s", "u") == "ok"
    assert len(c.calls) == 2


def test_non_transient_not_retried(monkeypatch):
    c = FakeClient(fail_first=5, exc=ValueError("bad"))
    _setup(monkeypatch, c)
    try:
        llm.chat("s", "u")
        assert False
    except RuntimeError:
        pass
    assert len(c.calls) == 1
