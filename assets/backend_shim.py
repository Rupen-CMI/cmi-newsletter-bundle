"""Minimal stand-ins for the `backend.*` package smartcrawl imports.

smartcrawl.py was written inside another project and pulls four things from it:

    backend.kit.utils.utc_now        - a timestamp
    backend.utils.domain.safe_fetch  - an httpx GET with sane defaults
    backend.email._llm.async_chat_client - an AsyncOpenAI-compatible client
    backend.cost.meter               - per-user LLM cost accounting

Only the first three do any work here; the cost meter is billing telemetry for that
project's users and is stubbed out. Registering these as real `backend.*` modules in
sys.modules lets smartcrawl import unmodified, so it stays in sync with upstream
instead of becoming a fork that has to be re-patched on every change.

    import backend_shim; backend_shim.install()
    from smartcrawl import smart_crawl
"""
from __future__ import annotations

import os
import sys
import types
from datetime import datetime, timezone


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


async def _safe_fetch(url: str, *, headers: dict | None = None, timeout: float = 15,
                      follow_redirects: bool = True, **kw):
    """httpx GET that returns None instead of raising.

    smartcrawl treats a falsy result as "this source had nothing", so every failure
    mode -- DNS, TLS, timeout, 4xx -- has to come back as None rather than an
    exception, or one dead sitemap probe aborts the whole crawl.
    """
    import httpx

    try:
        async with httpx.AsyncClient(follow_redirects=follow_redirects,
                                     timeout=timeout, verify=False) as c:
            r = await c.get(url, headers=headers or {})
            return r if r.status_code < 400 else None
    except Exception:  # noqa: BLE001 -- a failed probe is a normal outcome here
        return None


class _Settings:
    """Only the attributes smartcrawl reads."""

    def __init__(self) -> None:
        self.model_extract = os.environ.get("SMARTCRAWL_MODEL", "cmi-7b")


def _get_settings() -> _Settings:
    return _Settings()


def _async_chat_client(model: str):
    """AsyncOpenAI pointed at whichever endpoint serves `model`.

    Defaults to the self-hosted fine-tune: set LOCAL_LLM_BASE_URL and leave
    SMARTCRAWL_MODEL at its default so every call in the pipeline, extraction
    included, runs on our own model. The hosted branches remain only so the same
    shim still works if a key is deliberately supplied.
    """
    from openai import AsyncOpenAI

    if model.startswith("deepseek"):
        return AsyncOpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                           base_url="https://api.deepseek.com")
    if model.startswith("gpt-"):
        return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    base = os.environ.get("LOCAL_LLM_BASE_URL", "http://localhost:6006/v1")
    return AsyncOpenAI(api_key=os.environ.get("LOCAL_LLM_API_KEY", "local"),
                       base_url=base)


def _current_user_id() -> str:
    return "cmi-newsletter"


def _record(*a, **k) -> None:
    return None


def _estimate_llm_cost_inr(*a, **k) -> float:
    return 0.0


def install() -> None:
    """Register the shim modules under `backend.*` before smartcrawl is imported."""
    if "backend" in sys.modules:
        return

    def mod(name: str, **attrs) -> types.ModuleType:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    backend = mod("backend")
    backend.__path__ = []  # mark as a package so submodule imports resolve

    for pkg in ("backend.kit", "backend.utils", "backend.email", "backend.cost"):
        p = mod(pkg)
        p.__path__ = []

    mod("backend.kit.utils", utc_now=_utc_now)
    mod("backend.utils.domain", safe_fetch=_safe_fetch)
    mod("backend.email._llm", async_chat_client=_async_chat_client)
    mod("backend.config", get_settings=_get_settings)
    mod("backend.cost.meter", current_user_id=_current_user_id,
        record=_record, estimate_llm_cost_inr=_estimate_llm_cost_inr)
