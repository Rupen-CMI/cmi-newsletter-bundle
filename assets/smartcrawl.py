"""smart_crawl.py — Adaptive company intelligence crawler.

Phases:
  1. Homepage fetch (scrapling) -> extract all links
  2. LLM analyzes nav + user query -> prioritized crawl plan
  3. Parallel crawl: news sections (index->links->concurrent scrapling) +
     general sections (1-level scrapling, BFS fallback if blocked)
  4a. Direct article extraction (trafilatura) — raw full text, no LLM
  4b. LLM structured metadata extraction from general pages
"""
from __future__ import annotations
import asyncio, contextvars, gzip, hashlib, heapq, json, os, re, time, warnings
import html as _html
warnings.filterwarnings("ignore", "unclosed transport", ResourceWarning)  # suppress Windows asyncio Playwright cleanup noise
from datetime import datetime, timezone

from backend.kit.utils import utc_now
from typing import Any
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, quote

# async_chat_client() builds a NEW AsyncOpenAI-compatible client (and its httpx
# connection pool) on every call and nothing closes them. _run_extraction calls
# it per chunk, so an uncached call here leaks a pool per chunk — hundreds per
# autochain run. Cached per model because the provider is chosen from the model
# name (DeepSeek's endpoint via async_chat_client's base_url routing).
_extraction_clients: dict[str, object] = {}
def _get_extraction_client(model: str):
    if model not in _extraction_clients:
        from backend.email._llm import async_chat_client
        _extraction_clients[model] = async_chat_client(model)
    return _extraction_clients[model]
def _get_llm_model() -> str:
    from backend.config import get_settings
    return get_settings().model_extract
LLM_MODEL = _get_llm_model()

import logging as _logging
log = _logging.getLogger(__name__)

# ── Tunables (env-configurable; defaults preserve prior hardcoded behavior) ───
# Every crawl knob reads an env var so it can be tuned without a code change.
# Each default equals the literal it replaced — setting nothing changes nothing.
def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default

def _envf(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default

# Lightpanda JS-render browser
_LP_PORT            = _envi("LIGHTPANDA_PORT", 9223)
_LP_STARTUP_WAIT_S  = _envf("LIGHTPANDA_STARTUP_WAIT_S", 1.5)   # serve boot wait
_LP_WS_OPEN_TIMEOUT = _envf("LIGHTPANDA_WS_OPEN_TIMEOUT", 3.0)  # CDP WS connect timeout
_LP_WS_MAX_BYTES    = _envi("LIGHTPANDA_WS_MAX_MB", 20) * 1024 * 1024
_LP_SETTLE_S        = _envf("CRAWL_JS_SETTLE_S", 2.5)           # post-navigate settle
# Link extraction / tier escalation
_MIN_ANCHORS        = _envi("CRAWL_MIN_ANCHORS", 8)             # < this → escalate tier
_SHELL_THRESHOLD    = _envi("CRAWL_SHELL_THRESHOLD", 30_000)    # big-but-linkless SPA shell
# scrapling / playwright / camoufox removed 2026-06. Tier-1 is now direct
# curl_cffi (scrapling_get) and the JS tier is Lightpanda (js_browser_get);
# the leaky 700MB Firefox dependency chain is gone entirely.

# ── Lightpanda subprocess manager ─────────────────────────────────────────────
# Lightpanda is a Zig-based headless browser that uses ~3MB vs Chromium's ~300MB.
# It connects via CDP so Playwright can drive it with one line change.
# Binary path: LIGHTPANDA_BIN env var, PATH lookup, or the Dockerfile's own
# install location (/usr/local/bin/lightpanda) — never a shared /tmp path (see
# _LP_TRUSTED_BIN_FALLBACK below).
# WS URL: LIGHTPANDA_WS env var (must resolve to an allowlisted host — see
# _LP_ALLOWED_WS_HOSTS), or auto-started on ws://127.0.0.1:9223.
import subprocess as _subprocess
import shutil as _shutil

_LP_PROC: "_subprocess.Popen | None" = None
_LP_WS: str = ""

# /tmp is world-writable: any local user can pre-plant a file at the
# predictable "/tmp/lightpanda" path and have it executed as this process's
# JS-render subprocess (CWE-377 local-privesc pattern). The Dockerfile always
# sets LIGHTPANDA_BIN explicitly to this same admin-controlled path, so this
# fallback is only ever exercised in unconfigured/local environments — it must
# point somewhere that isn't shared, world-writable scratch space.
_LP_TRUSTED_BIN_FALLBACK = "/usr/local/bin/lightpanda"

# LIGHTPANDA_WS is operator config, not attacker-influenced input: restrict it
# to loopback hosts by default so an env var leak/injection can't redirect the
# crawler's CDP connection to an attacker-controlled browser endpoint.
_LP_ALLOWED_WS_HOSTS = {
    h.strip().lower()
    for h in os.environ.get("LIGHTPANDA_WS_ALLOWED_HOSTS", "127.0.0.1,localhost,::1").split(",")
    if h.strip()
}

def _lp_ws_host_allowed(ws_url: str) -> bool:
    try:
        return (urlparse(ws_url).hostname or "").lower() in _LP_ALLOWED_WS_HOSTS
    except Exception:
        return False

def _start_lightpanda() -> str:
    """Start Lightpanda CDP server if available. Returns WebSocket URL or ''."""
    global _LP_PROC, _LP_WS
    # Re-check liveness before trusting the cached URL. A bare `if _LP_WS` kept
    # returning the endpoint of a Lightpanda that had crashed or been killed, so
    # every later fetch failed its WS connect and the JS tier stayed dead for the
    # process lifetime with no restart and no log. `_LP_PROC is None` is allowed
    # because the LIGHTPANDA_WS override path below sets _LP_WS without spawning.
    if _LP_WS and (_LP_PROC is None or _LP_PROC.poll() is None):
        return _LP_WS
    if _LP_PROC is not None and _LP_PROC.poll() is not None:
        _logging.getLogger(__name__).warning(
            "[lightpanda] server exited (rc=%s) — respawning", _LP_PROC.returncode,
        )
        _LP_PROC = None
        _LP_WS = ""
    # Respect explicit override, but only for an allowlisted (loopback) host —
    # see _LP_ALLOWED_WS_HOSTS above.
    override = os.environ.get("LIGHTPANDA_WS", "")
    if override:
        if _lp_ws_host_allowed(override):
            _LP_WS = override
            return _LP_WS
        _logging.getLogger(__name__).warning(
            "[lightpanda] LIGHTPANDA_WS host not in allowlist %s — ignoring override: %s",
            sorted(_LP_ALLOWED_WS_HOSTS), override,
        )
    # Find binary
    bin_path = os.environ.get("LIGHTPANDA_BIN", "") or _shutil.which("lightpanda") or _LP_TRUSTED_BIN_FALLBACK
    if not os.path.isfile(bin_path) or not os.access(bin_path, os.X_OK):
        return ""
    try:
        _LP_PROC = _subprocess.Popen(
            [bin_path, "serve", "--host", "127.0.0.1", "--port", str(_LP_PORT)],
            stdout=_subprocess.DEVNULL,
            stderr=_subprocess.DEVNULL,
        )
        import time as _time
        _time.sleep(_LP_STARTUP_WAIT_S)
        if _LP_PROC.poll() is not None:
            _LP_PROC = None
            return ""
        _LP_WS = f"ws://127.0.0.1:{_LP_PORT}"
        _logging.getLogger(__name__).info("[lightpanda] started PID=%s ws=%s", _LP_PROC.pid, _LP_WS)
        return _LP_WS
    except Exception as e:
        _logging.getLogger(__name__).warning("[lightpanda] failed to start: %s", e)
        return ""


def _stop_lightpanda() -> None:
    """Terminate the managed Lightpanda server. Idempotent.

    Nothing used to reap this: the Popen handle sat in a module global with no
    terminate, no atexit, and no lifespan hook, so the server outlived every
    request and survived until the container died. Registered with atexit below
    and called from main.py's lifespan shutdown.
    """
    global _LP_PROC, _LP_WS
    proc, _LP_PROC, _LP_WS = _LP_PROC, None, ""
    if proc is None or proc.poll() is not None:
        return
    log = _logging.getLogger(__name__)
    try:
        proc.terminate()
        proc.wait(timeout=5)
        log.info("[lightpanda] terminated PID=%s", proc.pid)
    except _subprocess.TimeoutExpired:
        proc.kill()
        log.warning("[lightpanda] PID=%s ignored SIGTERM — killed", proc.pid)
    except Exception as e:
        log.warning("[lightpanda] shutdown failed for PID=%s: %s", proc.pid, e)


import atexit as _atexit
_atexit.register(_stop_lightpanda)


# JS polyfills injected before every page load — fixes Lightpanda's missing Web APIs
# that cause ClientDisconnected on Angular/React sites (missing DOMTokenList.supports,
# global, process, stub HTML elements). Returns true unconditionally — safe lie for
# feature-detection calls that only need a truthy answer.
_LP_POLYFILL = """
// Fix missing Web APIs (causes ClientDisconnected on Angular/React sites)
if (!DOMTokenList.prototype.supports) DOMTokenList.prototype.supports = function() { return true; };
if (typeof global === 'undefined') { try { globalThis.global = globalThis; } catch(e) {} }
if (typeof process === 'undefined') { try { globalThis.process = { env: {}, nextTick: function(cb){setTimeout(cb,0)} }; } catch(e) {} }
['HTMLFrameSetElement','HTMLDialogElement','HTMLDetailsElement','HTMLMarqueeElement',
 'HTMLPictureElement','HTMLTimeElement'].forEach(function(n){
    if (!globalThis[n]) { try { globalThis[n] = class extends HTMLElement {}; } catch(e) {} }
});
// Hide headless signals — prevent navigator.webdriver detection
try {
    Object.defineProperty(navigator, 'webdriver', { get: function(){ return false; }, configurable: true });
} catch(e) {}
// Spoof basic fingerprint properties so bot-detection JS gets plausible values
try {
    if (!window.chrome) { window.chrome = { runtime: {} }; }
    if (navigator.plugins && navigator.plugins.length === 0) {
        Object.defineProperty(navigator, 'plugins', { get: function(){ return [1,2,3,4,5]; } });
    }
} catch(e) {}
// ── Anti-detection gap-fillers (from Obscura's bootstrap.js, render-free only) ──
// SAFE additions: each only defines a property Lightpanda leaves missing — a
// real-Chrome value where its absence is itself a bot tell. We deliberately do
// NOT touch canvas/WebGL (Lightpanda has no renderer, so they yield nothing to
// fingerprint) nor override Function.prototype.toString (page libraries call it
// legitimately; breaking it lowers render yield).
try {
    if (!navigator.languages || navigator.languages.length === 0)
        Object.defineProperty(navigator, 'languages', { get: function(){ return ['en-US','en']; } });
} catch(e) {}
try {
    if (!navigator.hardwareConcurrency)
        Object.defineProperty(navigator, 'hardwareConcurrency', { get: function(){ return 8; } });
} catch(e) {}
try {
    if (navigator.deviceMemory === undefined)
        Object.defineProperty(navigator, 'deviceMemory', { get: function(){ return 8; } });
} catch(e) {}
try {
    if (!navigator.userAgentData) {
        Object.defineProperty(navigator, 'userAgentData', { get: function(){ return {
            brands: [{brand:'Chromium', version:'131'}, {brand:'Not_A Brand', version:'24'}],
            mobile: false, platform: 'Windows',
            getHighEntropyValues: function(){ return Promise.resolve({
                platform:'Windows', platformVersion:'10.0.0', architecture:'x86', bitness:'64',
                model:'', uaFullVersion:'131.0.0.0'
            }); }
        }; } });
    }
} catch(e) {}
"""

async def _lightpanda_cdp_fetch(lp_ws: str, url: str, timeout: float = 8.0) -> str:
    """Fetch page HTML via Lightpanda using raw CDP WebSocket.

    No Playwright/Chromium dependency — pure JSON-RPC over WebSocket.
    Lightpanda exposes a standard CDP endpoint; we drive it directly.
    """
    import json as _json
    try:
        import websockets.client as _wsc
    except ImportError:
        return ""

    html = ""
    _id_counter = 0

    try:
        async with _wsc.connect(lp_ws, max_size=_LP_WS_MAX_BYTES, open_timeout=_LP_WS_OPEN_TIMEOUT) as ws:
            pending: "dict[int, asyncio.Future]" = {}

            async def _send(method: str, params: dict | None = None) -> dict:
                nonlocal _id_counter
                _id_counter += 1
                msg_id = _id_counter
                fut: asyncio.Future = asyncio.get_event_loop().create_future()
                pending[msg_id] = fut
                await ws.send(_json.dumps({"id": msg_id, "method": method, "params": params or {}}))
                return await asyncio.wait_for(fut, timeout=timeout)

            async def _pump():
                async for raw in ws:
                    try:
                        msg = _json.loads(raw)
                        mid = msg.get("id")
                        if mid and mid in pending and not pending[mid].done():
                            pending[mid].set_result(msg.get("result", {}))
                    except Exception:
                        log.debug("CDP message parse failed", exc_info=True)

            pump_task = asyncio.create_task(_pump())  # rls-allow: CDP websocket reader; touches no database at all
            ctx_id = ""
            target_id = ""
            try:
                # Create a new tab, navigate, extract HTML
                ctx = await _send("Target.createBrowserContext")
                ctx_id = ctx.get("browserContextId", "")
                tgt = await _send("Target.createTarget", {"url": "about:blank", "browserContextId": ctx_id})
                target_id = tgt.get("targetId", "")
                sess = await _send("Target.attachToTarget", {"targetId": target_id, "flatten": True})
                session_id = sess.get("sessionId", "")

                async def _ssend(method: str, params: dict | None = None) -> dict:
                    nonlocal _id_counter
                    _id_counter += 1
                    msg_id = _id_counter
                    fut: asyncio.Future = asyncio.get_event_loop().create_future()
                    pending[msg_id] = fut
                    await ws.send(_json.dumps({"id": msg_id, "sessionId": session_id,
                                               "method": method, "params": params or {}}))
                    return await asyncio.wait_for(fut, timeout=timeout)

                await _ssend("Page.addScriptToEvaluateOnNewDocument", {"source": _LP_POLYFILL})
                await _ssend("Page.navigate", {"url": url})
                await asyncio.wait_for(asyncio.sleep(_LP_SETTLE_S), timeout=_LP_SETTLE_S + 0.5)  # let page settle
                result = await _ssend("Runtime.evaluate", {
                    "expression": "document.documentElement ? document.documentElement.outerHTML : ''",
                    "returnByValue": True,
                })
                html = result.get("result", {}).get("value") or ""
            finally:
                # CRITICAL (verified in Lightpanda src/cdp/CDP.zig): Lightpanda
                # allows exactly ONE browser context at a time, and closeTarget
                # does NOT free it. Without disposeBrowserContext the context
                # arena leaks AND the next createBrowserContext fails with
                # 'AlreadyExists' — every fetch after the first would break.
                # Dispose unconditionally (even on error) and BEFORE cancelling
                # the pump so these sends still get their replies.
                try:
                    if target_id:
                        await _send("Target.closeTarget", {"targetId": target_id})
                except Exception:
                    log.debug("CDP Target.closeTarget failed during cleanup", exc_info=True)
                try:
                    if ctx_id:
                        await _send("Target.disposeBrowserContext", {"browserContextId": ctx_id})
                except Exception:
                    log.debug("CDP Target.disposeBrowserContext failed during cleanup", exc_info=True)
                pump_task.cancel()
    except Exception as e:
        _logging.getLogger(__name__).debug("[lightpanda_cdp] %s: %s", url[:60], e)

    return html


# ── Browser concurrency cap + zombie reaper ──────────────────────────────────
# Why this exists: scrapling's StealthyFetcher.fetch()/DynamicFetcher.fetch()
# spawn Camoufox/Chromium child processes that can outlive the Python response
# object when the fetch times out or raises. On 2026-05-28 this leaked into a
# t3.medium OOM-kill of the uvicorn host. The cap bounds peak browser RAM, the
# reaper kills any child processes left over after each fetch.
import threading as _threading

class _CrossLoopSemaphore:
    """asyncio-compatible semaphore that works across multiple event loops.

    asyncio.Semaphore is event-loop-local. Each company's smart_crawl runs
    in its own asyncio.new_event_loop() (via asyncio.to_thread). Using a plain
    asyncio.Semaphore means each loop independently has its own count → two
    companies can each spin up CRAWL_BROWSER_CONCURRENCY browsers, doubling
    the real cap. This class uses threading.BoundedSemaphore underneath so
    the cap is enforced process-wide across all event loops.
    """
    def __init__(self, value: int) -> None:
        self._sem = _threading.BoundedSemaphore(value)

    async def __aenter__(self):
        # Run the blocking acquire in a thread so the event loop stays alive
        # while waiting. For browser tasks (5–15s each) the acquire wait is
        # negligible relative to the actual browser work.
        #
        # CANCEL-SAFETY (this used to be a permanent-deadlock bug):
        # asyncio.to_thread cannot interrupt the worker thread. A plain
        # `await asyncio.to_thread(acquire)` that gets cancelled leaves the
        # thread still running — it goes on to TAKE the permit, while
        # __aenter__ raises CancelledError so the `async with` body never runs
        # and __aexit__ never fires. The permit is then lost for the lifetime of
        # the process. With CRAWL_BROWSER_CONCURRENCY=2, two such cancellations
        # (a crawl timing out, a client disconnecting) deadlock every browser
        # fetch process-wide until restart.
        #
        # Shield the acquire so cancelling the await does not orphan it, then
        # hand the permit straight back when the thread eventually lands.
        fut = asyncio.ensure_future(asyncio.to_thread(self._sem.acquire))  # rls-allow: semaphore permit acquire, no DB access
        try:
            await asyncio.shield(fut)  # rls-allow: shields the same semaphore-only future above, no DB access
        except asyncio.CancelledError:
            def _release_orphaned_permit(f: "asyncio.Future") -> None:
                if f.cancelled() or f.exception() is not None:
                    return
                try:
                    self._sem.release()
                except ValueError:
                    # BoundedSemaphore refuses over-release. Reaching here means
                    # the permit was already returned by another path, so the
                    # accounting is intact — but it also means this callback's
                    # assumption was wrong, which is worth seeing if it recurs.
                    log.warning("browser semaphore: orphaned-permit release was "
                                "already accounted for — no permit lost")
            fut.add_done_callback(_release_orphaned_permit)
            raise
        return self

    async def __aexit__(self, *_):
        try:
            self._sem.release()
        except ValueError:
            # Defensive: a double-release would otherwise raise out of __aexit__
            # and mask whatever exception the body was already propagating.
            log.warning("browser semaphore over-released — ignoring")

_BROWSER_SEM = _CrossLoopSemaphore(int(os.environ.get("CRAWL_BROWSER_CONCURRENCY", "2")))
# Per-domain tier memory: host → (min tier that worked, time it was recorded).
# Updated on every successful fetch; consulted on the next call to skip known-useless
# tiers. TTL'd (below) rather than pinned for process lifetime — a single 403 (rate
# limit, transient WAF rule, one-off outage) used to promote a host to a higher tier
# permanently, with no re-probe ever happening until a container restart; a domain
# that recovers minutes later stayed on the expensive/degraded path for the process's
# entire remaining lifetime. No persistence across processes, no locking needed
# (single async process).
_DOMAIN_TIER_CACHE: dict[str, tuple[int, float]] = {}
_DOMAIN_TIER_CACHE_TTL = 6 * 3600  # 6h — long enough to skip repeat probes within a
                                    # crawl burst, short enough to re-verify a host
                                    # that was blocked only transiently.


def _cached_domain_tier(host: str) -> int:
    entry = _DOMAIN_TIER_CACHE.get(host)
    if entry is None:
        return 0
    tier, ts = entry
    if time.time() - ts > _DOMAIN_TIER_CACHE_TTL:
        _DOMAIN_TIER_CACHE.pop(host, None)
        return 0
    return tier


def _set_domain_tier(host: str, tier: int) -> None:
    _DOMAIN_TIER_CACHE[host] = (tier, time.time())
_BROWSER_PROC_PATTERNS = ("lightpanda",
                          # Legacy patterns — kept so orphans from old deployments are still reaped
                          "camoufox", "firefox", "chromium", "chrome", "headless_shell",
                          "playwright/driver", "geckodriver", "firefox-bin")


# A browser owned by a live fetch has its Lightpanda parent (ppid != 1)
# and is younger than this. Anything older OR reparented to init is garbage.
_MAX_BROWSER_LIFETIME_S = _envf("CRAWL_MAX_BROWSER_LIFETIME_S", 75.0)  # browser timeout + grace + buffer


def _reap_browser_children() -> int:
    """Kill leaked/orphaned browser processes. Returns count killed.

    The previous version used ``psutil.Process().children(recursive=True)`` and
    MISSED the real leak: when the Playwright Node.js driver crashes (the FF
    ``pageError.location.url`` null-check bug on JS-heavy sites), Linux reparents
    camoufox-bin and its tabs to PID 1 (init) — they leave our process subtree,
    so ``children()`` finds nothing while ~700MB of Camoufox keeps running.

    This scans the WHOLE process table (sees reparented orphans regardless of
    parent) and kills a browser proc only when it is provably garbage:
      • reparented to init (ppid == 1)  → orphan from a driver crash, OR
      • older than the max browser lifetime → hung past its own timeout.
    A browser still owned by a concurrent live fetch has ppid != 1 and age <
    lifetime, so this NEVER kills another in-flight crawl's browser. Safe to call
    from any browser tier's ``finally`` and at end-of-pipeline.
    """
    try:
        import psutil
    except ImportError:
        return 0
    killed = 0
    now = time.time()
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time", "ppid"]):
        try:
            info = proc.info
            name = (info.get("name") or "").lower()
            cmd = " ".join(info.get("cmdline") or []).lower()
            if not any(p in name or p in cmd for p in _BROWSER_PROC_PATTERNS):
                continue
            ppid = info.get("ppid")
            age = now - (info.get("create_time") if info.get("create_time") is not None else now)
            if ppid == 1 or age > _MAX_BROWSER_LIFETIME_S:
                try:
                    for ch in proc.children(recursive=True):
                        try:
                            ch.kill()
                            killed += 1
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    proc.kill()
                    killed += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return killed

# ── Constants / regexes ───────────────────────────────────────────────────────
CF_RE      = re.compile(r"just a moment|cf-browser-verification|enable javascript|human verification|attention required", re.I)
_SKIP_RE   = re.compile(
    r"/user/(login|register|password|logout|reset)(?:[/?]|$)"
    r"|/auth/(login|signup|register|logout)(?:[/?]|$)"
    r"|/signin(?:[/?]|$)|/signup(?:[/?]|$)"
    r"|/account/(login|register|logout|password|reset|verify)(?:[/?]|$)"
    # WordPress admin/login surface and the two demo posts every fresh WP install
    # ships with. Not "low value" — zero value: /wp-admin and /wp-login.php return
    # ~200-byte login shells, and hello-world / sample-page are Automattic's
    # placeholder copy, never the site's own content. Left in, they consume real
    # page budget (4 of 40 slots on vedhyatech.com) and feed placeholder text to
    # the extractor. Taxonomy and date archives are NOT skipped here — they are
    # thin but they do link to real posts, so they are demoted in _score instead.
    r"|/wp-admin(?:[/?]|$)|/wp-login\.php|/xmlrpc\.php"
    r"|/(hello-world|sample-page)(?:-\d+)?(?:[/?]|$)"
    r"|\.(zip|rar|7z|tar|gz|mp4|mp3|avi|mov|flv|wmv|jpg|jpeg|png|gif|bmp|webp|svg|ico|woff|woff2|ttf|eot)(\?.*)?$",
    re.I,
)
_DOC_RE    = re.compile(r"\.(pdf|doc|docx|xls|xlsx|ppt|pptx|csv)(\?.*)?$", re.I)
# CMS taxonomy + date-archive index pages (see _score priority 3).
_ARCHIVE_PATH_RE = re.compile(r"^/(category|tag|tags|author|archives?)/|^/(19|20)\d{2}(/\d{1,2})*/?$", re.I)
_H1_RE     = re.compile(r"<h1[^>]*>(.*?)</h1>", re.I | re.S)
_TAG_RE    = re.compile(r"<[^>]+>")
_A_HREF_RE    = re.compile(r'<a\s[^>]*href=["\']([^"\'#][^"\']*)["\'][^>]*>(.*?)</a>', re.I | re.S)
_CANONICAL_RE = re.compile(r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']', re.I)
_NEXT_PAGE_RE = re.compile(r'<link[^>]+rel=["\']next["\'][^>]+href=["\']([^"\']+)["\']', re.I)
_PAGINATION_QUERY_KEYS = {"page", "paged", "p", "pg", "offset", "start"}
_DATA_URL_RE  = re.compile(r'\bdata-(?:href|url|link|route|path)=["\']([^"\'#][^"\']*)["\']', re.I)
_JSONLD_RE    = re.compile(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.I | re.S)
_LOC_RE       = re.compile(r"<loc>(https?://[^<]+)</loc>")
_DATE_TEXT_RE = re.compile(
    r"\b(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?),?\s+(20\d{2})\b"
    r"|(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2}),?\s+(20\d{2})\b",
    re.I
)
_MONTH_MAP = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
# Hard cap on sitemap URL discovery — prevents OOM on e-commerce sitemaps with 100k+ SKU URLs.
# The seed cap in smart_crawl() only applies AFTER discovery; this caps the discovery list itself.
_SITEMAP_LOC_CAP = _envi("CRAWL_SITEMAP_LOC_CAP", 2000)

def _parse_text_date(text: str) -> str:
    """Extract ISO date from natural-language date in first 3000 chars of text."""
    for m in _DATE_TEXT_RE.finditer(text[:3000]):
        g = m.groups()
        try:
            if g[0]:  # DD Mon YYYY
                day, mon, yr = int(g[0]), _MONTH_MAP[g[1].lower()[:3]], int(g[2])
            else:      # Mon DD YYYY
                mon, day, yr = _MONTH_MAP[g[3].lower()[:3]], int(g[4]), int(g[5])
            return f"{yr:04d}-{mon:02d}-{day:02d}"
        except (ValueError, KeyError):
            continue
    return ""
_CAT_TYPE  = {"news":"news","press":"press_release","blog":"blog","events":"event","media":"news","other":"news"}
_NEWS_PATHS = ["/blog","/news","/newsroom","/press","/insights","/media","/updates","/announcements","/articles","/publications"]

# ── Crawl modes ───────────────────────────────────────────────────────────────
# `paths` ranks discovered URLs (priority 0 in _score). `strict_paths` decides
# whether it ALSO filters: True → anything off-list is dropped outright; False →
# off-list URLs stay in the queue behind the on-list ones and get crawled if the
# page budget allows.
#
# Strict filtering is right for the narrow modes (news/investor/leads/leadership)
# — there the point is to spend the whole budget on one section of the site. It
# is wrong for the "tell me everything about this company" modes, because no
# fixed list can name every way a site labels its own offerings. Measured on
# vedhyatech.com: discovery found 37 URLs, the company allow-list passed 3, and
# the 34 it dropped included /offerings/ (the entire product catalogue), /clients/,
# /certification/ and 6 /portfolio/<product> pages — while 7 of the 10 budgeted
# page slots went unused. Ranking keeps the on-list pages first without throwing
# away the rest of the site.
MODES: dict[str, dict] = {
    "full": {
        "paths":      [],
        "fields":     None,
        "max_pages":  80,
        "max_tokens": 5000,
        "label":      "full company intelligence",
        "include_docs": True,
    },
    "news": {
        "paths":      ["/blog", "/news", "/newsroom", "/press", "/insights",
                       "/media", "/updates", "/announcements", "/articles", "/publications"],
        "fields":     ["media"],
        "max_pages":  30,
        "max_tokens": 3000,
        "label":      "news articles and press coverage",
        "include_docs": False,
    },
    "investor": {
        "paths":      ["/investor", "/investors", "/ir", "/financials", "/annual",
                       "/ipo", "/sebi", "/bse", "/nse", "/disclosures", "/results", "/agm"],
        "fields":     ["media", "financials", "intel"],
        "max_pages":  40,
        "max_tokens": 6000,
        "label":      "investor relations, financial results, and disclosures",
        "include_docs": True,
    },
    "company": {
        "paths":      ["/about", "/about-us", "/company", "/team", "/leadership",
                       "/people", "/our-team", "/who-we-are", "/management",
                       "/products", "/product", "/solutions", "/services",
                       "/offerings", "/offering", "/portfolio", "/capabilities",
                       "/what-we-do", "/expertise", "/technology", "/technologies",
                       "/industries", "/sectors", "/markets", "/verticals",
                       "/customers", "/case-studies", "/case-study", "/clients",
                       "/partners", "/partnerships",
                       "/certification", "/certifications", "/accreditation",
                       "/contact", "/contact-us", "/locations", "/offices",
                       "/pricing", "/plans", "/why-us", "/why",
                       "/vision", "/mission", "/values"],
        # "financials" added so employee_count/key_numbers get extracted — real
        # site content (e.g. an Indiamart/Infocom directory-listing "Key Facts"
        # block: "No. of Employees: 03") was being silently dropped even though
        # it was right there in the crawled text, because the extraction schema
        # never asked for it. Verified live against fluidflowengg.com.
        "fields":     ["company", "location", "business", "people", "contact", "relationships", "intel", "financials"],
        "max_pages":  40,
        "max_tokens": 6000,
        "label":      "company profile, leadership, products, services, and client relationships",
        "include_docs": True,
        "strict_paths": False,
    },
    "leads": {
        "paths":      ["/team", "/leadership", "/people", "/our-team", "/management",
                       "/customers", "/case-studies", "/partners", "/partnerships",
                       "/clients", "/contact", "/contact-us"],
        "fields":     ["people", "contact", "relationships"],
        "max_pages":  25,
        "max_tokens": 2500,
        "label":      "contacts, customers, partnerships and relationships",
        "include_docs": False,
    },
    "leadership": {
        "paths":      ["/about", "/team", "/management", "/leadership", "/board",
                       "/directors", "/founders", "/executive", "/our-people",
                       "/people", "/who-we-are", "/our-team", "/governance"],
        "fields":     ["company", "people"],
        "max_pages":  25,
        "max_tokens": 3000,
        "label":      "leadership, board of directors, key executives and team",
        "include_docs": False,
    },
    "business": {
        "paths":      ["/about", "/about-us", "/company", "/team", "/leadership",
                       "/products", "/product", "/solutions", "/services",
                       "/customers", "/case-studies", "/partners", "/partnerships",
                       "/blog", "/news", "/press", "/insights", "/resources",
                       "/investors", "/investor", "/financials", "/contact",
                       "/pricing", "/plans"],
        "fields":     ["company", "business", "people", "media", "financials", "intel"],
        "max_pages":  60,
        "max_tokens": 5000,
        "label":      "business intelligence: products, services, industries, investment data, news and people",
        "include_docs": True,
        "strict_paths": False,
    },
}

# Universal valuable paths — boosted to priority 0 in EVERY mode (even "full"),
# because /about, /team, /contact, /pricing, /products are always high-signal
# pages for company intelligence regardless of what mode the caller picked.
_UNIVERSAL_VALUABLE_PATHS = (
    "/about", "/about-us", "/company", "/team", "/leadership", "/management",
    "/people", "/our-team", "/who-we-are", "/founders",
    "/products", "/product", "/solutions", "/services",
    "/offerings", "/offering", "/portfolio", "/capabilities",
    "/what-we-do", "/expertise", "/industries", "/sectors", "/verticals",
    "/customers", "/case-studies", "/case-study", "/clients",
    "/partners", "/partnerships",
    "/certification", "/certifications",
    "/contact", "/contact-us", "/locations", "/offices",
    "/pricing", "/plans",
    "/careers", "/jobs",
)

def _mode_schema(fields: list[str] | None) -> dict:
    """Return a subset of INTEL_SCHEMA containing only the requested top-level fields."""
    if not fields:
        return INTEL_SCHEMA
    req   = [f for f in INTEL_SCHEMA["required"]    if f in fields]
    props = {k: v for k, v in INTEL_SCHEMA["properties"].items() if k in fields}
    return {**INTEL_SCHEMA, "required": req, "properties": props}

# ── Output schema (compact JSON Schema for OpenAI Structured Output) ──────────
def _obj(req: list[str], **props: Any) -> dict:
    return {"type":"object","additionalProperties":False,"required":req,"properties":props}
def _s() -> dict: return {"type":"string"}
def _ni() -> dict: return {"type":["integer","null"]}
def _arr(item: Any = None, desc: str = "") -> dict:
    out = {"type":"array","items":item or _s()}
    if desc:
        out["description"] = desc
    return out

INTEL_SCHEMA = _obj(
    ["company","location","business","financials","people","media","relationships","contact","intel"],
    company=_obj(
        ["name","legal_name","description","tagline","founded_year","website","ticker","type","stock_exchange","regulatory_status"],
        name=_s(), legal_name=_s(), description=_s(), tagline=_s(), founded_year=_ni(),
        website=_s(), ticker=_s(), type=_s(), stock_exchange=_s(), regulatory_status=_s(),
    ),
    location=_obj(
        ["headquarters","offices","countries","manufacturing_locations","regions_served"],
        headquarters=_s(), offices=_arr(), countries=_arr(),
        manufacturing_locations=_arr(), regions_served=_arr(),
    ),
    business=_obj(
        ["industries","key_markets","products","services","certifications","awards","technology","competitive_advantages","regulatory_compliance"],
        industries=_arr(), key_markets=_arr(),
        products=_arr(_obj(["name","category","description","features","target_customers","pricing"],
            name=_s(), category=_s(), description=_s(), features=_arr(), target_customers=_s(), pricing=_s())),
        services=_arr(_obj(["name","category","description","target_customers"],
            name=_s(), category=_s(), description=_s(), target_customers=_s())),
        certifications=_arr(), awards=_arr(), technology=_arr(),
        competitive_advantages=_arr(), regulatory_compliance=_arr(),
    ),
    financials=_obj(
        ["revenue","employee_count","funding","key_numbers","market_cap","fiscal_year","recent_fundraise","profitability"],
        revenue=_s(), employee_count=_s(), funding=_s(), key_numbers=_arr(),
        market_cap=_s(), fiscal_year=_s(), recent_fundraise=_s(), profitability=_s(),
    ),
    people=_obj(["leadership","board","advisors"],
        leadership=_arr(_obj(["name","title","bio","linkedin"], name=_s(), title=_s(), bio=_s(), linkedin=_s())),
        board=_arr(_obj(["name","title","bio"], name=_s(), title=_s(), bio=_s())),
        advisors=_arr(_obj(["name","title"], name=_s(), title=_s())),
    ),
    media=_obj(["articles","press_releases","investor_presentations"],
        articles=_arr(_obj(
            ["title","date","type","url","author","full_text","summary","entities","figures","quotes"],
            title=_s(), date=_s(), type=_s(), url=_s(), author=_s(), full_text=_s(),
            summary=_s(), entities=_arr(), figures=_arr(), quotes=_arr())),
        press_releases=_arr(_obj(["title","date","url","summary","highlights"],
            title=_s(), date=_s(), url=_s(), summary=_s(),
            highlights=_arr())),
        investor_presentations=_arr(_obj(["title","date","url","summary","highlights"],
            title=_s(), date=_s(), url=_s(), summary=_s(),
            highlights=_arr())),
    ),
    relationships=_obj(
        ["partnerships","customers","competitors","subsidiaries","parent_company","distributors","suppliers","investors"],
        partnerships=_arr(desc="Named organisations this company works WITH: joint ventures, alliances, channel or technology partners."),
        customers=_arr(desc="Named organisations that BUY FROM this company. For a contract manufacturer these are the brands it produces for, and they are NOT competitors."),
        competitors=_arr(desc="Named companies that sell a DIRECTLY SUBSTITUTABLE product or service, so a buyer chooses this company OR that one and gets the same thing. A company that appears on the site as a client, partner, or brand served is NEVER a competitor, however large it is. If the site names no true rival, return an empty array rather than guessing."),
        subsidiaries=_arr(),
        parent_company=_s(), distributors=_arr(), suppliers=_arr(), investors=_arr(),
    ),
    contact=_obj(
        ["email","phone","address","linkedin","twitter","facebook","instagram","youtube","other_social"],
        email=_s(), phone=_s(), address=_s(), linkedin=_s(), twitter=_s(),
        facebook=_s(), instagram=_s(), youtube=_s(), other_social=_arr(),
    ),
    intel=_obj(
        ["key_insights","recent_developments","strategic_focus","risks_challenges","opportunities","market_position","growth_drivers"],
        key_insights=_arr(), recent_developments=_arr(), strategic_focus=_arr(),
        risks_challenges=_arr(), opportunities=_arr(), market_position=_s(), growth_drivers=_arr(),
    ),
)

# ── Tier-1: direct curl_cffi (real Chrome JA3/JA4 TLS + HTTP/2 fingerprint) ──────
# Was scrapling's AsyncFetcher; replaced 2026-06 with direct curl_cffi so the
# scrapling → playwright → camoufox dependency chain (and its ~700MB Firefox)
# is uninstalled entirely. curl_cffi IS what scrapling used under the hood for
# the TLS fingerprint, so the fingerprint-wall behaviour is unchanged.
#
# THREAD-LOCAL, not a single global: curl_cffi's AsyncSession binds to the
# asyncio event loop running at construction time (its AsyncCurl registers fd
# callbacks against that specific loop). Each company's smart_crawl() runs in
# its own asyncio.new_event_loop() via asyncio.to_thread (see
# _CrossLoopSemaphore above) — a single shared AsyncSession would get its
# libcurl multi-handle driven concurrently from multiple OS threads/loops,
# which is exactly the native memory-corruption pattern that segfaulted the
# backend mid-crawl (Kajal's tejasnetworks.com run, 2026-07-22, exit 139).
# One session per thread == one session per event loop, so each loop only
# ever drives its own AsyncCurl.
_CURL_SESSION_LOCAL = _threading.local()


def _get_curl_session() -> Any:
    session = getattr(_CURL_SESSION_LOCAL, "session", None)
    if session is None:
        try:
            from curl_cffi.requests import AsyncSession
        except ImportError:
            return None
        session = AsyncSession()
        _CURL_SESSION_LOCAL.session = session
    return session


_SAME_SITE_REDIRECT_CAP = 3   # matches backend.utils.domain._MAX_REDIRECTS


def _same_registrable_host(a: str, b: str) -> bool:
    """True if a and b are the same site modulo a `www.` prefix.

    Deliberately narrow — this is NOT a general redirect follower. It exists
    only to unblock the extremely common case of a bare-domain URL 301-ing to
    its `www.` canonical (or vice versa) on the SAME host, which discovery/nav
    links routinely produce. Anything that changes host beyond that (a real
    cross-domain redirect, which is exactly the SSRF vector allow_redirects=
    False exists to block) is left unfollowed, same as before this function
    existed — safe_fetch (backend/utils/domain.py) remains the place for
    genuine per-hop-validated cross-domain redirect handling.
    """
    return a.lower().removeprefix("www.") == b.lower().removeprefix("www.")


async def _curl_get_once(s: Any, url: str, timeout: int, verify: bool) -> str:
    # No custom headers: impersonate sets a full, internally-consistent browser
    # header set (sec-ch-ua, Accept, etc.) that MATCHES the JA3/TLS — adding
    # our own UA would create a UA↔TLS mismatch that WAFs flag.
    # allow_redirects=False: SSRF defense — an attacker-controlled domain could
    # 302 to an internal address. Cross-domain redirect hops are NOT followed
    # here — only a same-host `www.` canonicalization is (see
    # _same_registrable_host), bounded to _SAME_SITE_REDIRECT_CAP hops, reusing
    # this same curl_cffi session (no extra process/memory). Anything else stays
    # unfollowed; safe_fetch (backend.utils.domain, the Tier-0 path) is where
    # genuine per-hop-validated cross-domain redirects are handled.
    # Cloudflare on datacenter IPs blocks Chrome/Edge TLS fingerprints but passes
    # Safari17 (different JA3/JA4 hash). Verified on EC2 ap-south-1: chrome=403,
    # safari17_0=200 for volza.com and similar CF-protected B2B sites. No extra
    # process, no memory, no new dep — just a second curl_cffi request on the same
    # persistent AsyncSession.
    _IMPERSONATE_CHAIN = ("chrome", "safari17_0")
    current_url = url
    original_host = urlparse(url).hostname or ""
    for _hop in range(_SAME_SITE_REDIRECT_CAP + 1):
        r = None
        for impersonate in _IMPERSONATE_CHAIN:
            r = await s.get(current_url, impersonate=impersonate,
                            timeout=timeout, allow_redirects=False, verify=verify)
            if getattr(r, "status_code", 200) not in (403, 405, 429):
                break
        else:
            return ""
        status = getattr(r, "status_code", 200)
        if status in (301, 302, 303, 307, 308):
            loc = (r.headers or {}).get("location") or (r.headers or {}).get("Location")
            if not loc:
                return ""
            next_url = urljoin(current_url, loc)
            next_host = urlparse(next_url).hostname or ""
            if not _same_registrable_host(original_host, next_host):
                return ""   # cross-domain hop — not this tier's job, see docstring
            current_url = next_url
            continue
        html = (r.text if r is not None else "") or ""
        return html if len(html) > 500 else ""
    return ""


# Full browser headers — validated Tier-0 across 9 sites incl. Stripe, Shopify,
# Cloudflare, Tata, Siemens, Rakuten, Samsung, Notion. UA-only is NOT enough:
# Sec-Fetch-*, Accept-Language, Accept-Encoding matter for Cloudflare-protected sites.
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-Mode": "navigate",
    "Upgrade-Insecure-Requests": "1",
}


# Anti-bot detection: catch Cloudflare/PerimeterX/Akamai challenge pages BEFORE
# wasting time parsing them for anchors. From AWS IPs, Cloudflare 403s a lot of
# B2B sites (volza.com, etc.) — those return a short HTML challenge with 0–1
# anchors, which used to silently degrade into "0 seed URLs → empty extraction".
_ANTIBOT_RE = re.compile(
    r"attention required\s*\|\s*cloudflare|"
    r"just a moment|"
    r"checking your browser|"
    r"enable javascript and cookies|"
    r"cf-(?:chl-bypass|browser-verification|chl-opt)|"
    r"_cf_chl_opt|"
    r"ddos protection by cloudflare|"
    r"<title>\s*403\b|"
    r"access denied|"
    r"managed challenge",
    re.IGNORECASE,
)


def _looks_blocked(status: int, html: str) -> bool:
    """Return True if status/body indicates an anti-bot challenge or block.

    The escalation cue used to be "anchor count < 8", but a Cloudflare 403
    typically returns a 5KB challenge page with exactly 1 anchor — the
    cascade missed it. Detect the block by content signature, not symptom.
    """
    if status in (403, 429, 503):
        return True
    return bool(html and _ANTIBOT_RE.search(html[:4000]))


# NOTE: stealthy_get (Camoufox/Firefox via scrapling StealthyFetcher) was REMOVED
# 2026-06. It was a 700MB Firefox per fetch that (a) crashed on JS-error sites
# (Playwright FF `pageError.location.url` bug) and reparented to PID 1 → the OOM
# leak, and (b) couldn't beat our actual walls anyway (datacenter IP-rep needs a
# proxy; DataDome needs a proxy) — those go to the hardened Wayback fallback now.
# The JS-render tier is Lightpanda only (40-123MB, single managed process, no
# Playwright → no orphan leak). See js_browser_get + _lightpanda_cdp_fetch.


async def httpx_get(url: str, timeout: float = 12.0) -> str:
    """Tier-0: httpx + full browser headers. Fastest, works on most sites.

    Returns HTML on success (>500 bytes, no CF challenge), '' otherwise.
    Validated on 9 sites: beats scrapling on Stripe/Shopify/Cloudflare/Siemens/Zoho
    where scrapling returns 0 bytes due to curl_cffi TLS handshake issues.
    """
    from backend.utils.domain import safe_fetch
    try:
        r = await safe_fetch(url, headers=_BROWSER_HEADERS, timeout=timeout)
        if r.status_code != 200 or len(r.content) < 500:
            return ""
        html = r.text
        if CF_RE.search(html):  # Cloudflare challenge page — bail to scrapling
            return ""
        return html
    except Exception:
        return ""


async def scrapling_get(url: str, timeout: int = 15) -> str:
    """Tier-1: fast HTTP fetch via direct curl_cffi (real Chrome TLS fingerprint).

    Returns HTML or ''. Retries with TLS verification disabled when the server
    presents a cert chain curl_cffi can't validate (mis-served intermediates on
    marketing sites) — acceptable since we only scrape public HTML, no secrets.
    (Name kept for caller compatibility; no longer uses scrapling.)
    """
    s = _get_curl_session()
    if s is None:
        return ""
    try:
        return await _curl_get_once(s, url, timeout, verify=True)
    except Exception as e:
        msg = str(e).lower()
        if "ssl" in msg or "certificate" in msg or "tls" in msg:
            try:
                return await _curl_get_once(s, url, timeout, verify=False)
            except Exception as e2:
                if "DEBUG_FETCH" in os.environ:
                    log.debug("  [curl_cffi err after verify=False] %s -> %s: %s", url[:80], type(e2).__name__, str(e2)[:100])
        elif "DEBUG_FETCH" in os.environ:
            log.debug("  [curl_cffi err] %s -> %s: %s", url[:80], type(e).__name__, str(e)[:100])
    return ""

# Far-future timestamp → Wayback 302-redirects to the NEWEST snapshot in one
# call (verified 2.6s, 261KB for eximpedia from the AWS prod IP). A year prefix
# like "2026" instead resolves to the snapshot closest to Jan 1 — not newest.
_WAYBACK_NEWEST = "https://web.archive.org/web/29991231235959id_/{url}"
_WAYBACK_TS = "https://web.archive.org/web/{ts}id_/{url}"
# CDX = newest-first list of GOOD captures only (HTTP 200 + real HTML), dedup'd
# by content digest. Filters out the exact failure modes the bare id_ redirect
# happily lands on: SPN error stubs (e.g. eximpedia 2026-06-03 = 204
# application/http), archived 403 block pages, and 3xx redirect shells.
_WAYBACK_CDX = ("http://web.archive.org/cdx/search/cdx?url={u}&output=json"
                "&fl=timestamp&filter=statuscode:200&filter=mimetype:text/html"
                "&collapse=digest&limit=-8")
_WB_TS_RE = re.compile(r"/web/(\d{14})")
# Snapshot age was previously computed only for the log line (fresh/aging/stale
# labels below never fed a decision) — a snapshot from years ago passed the same
# `_wb_usable` content check as one from yesterday and was used identically,
# silently feeding stale company facts (leadership, products, even a since-closed
# business) into enrichment with no signal it happened. Anything older than this
# is rejected outright rather than merely logged as "stale"; age=None (timestamp
# unparseable) is treated as acceptable since we can't confirm it's actually old.
_WAYBACK_MAX_AGE_DAYS = 365
# archive.org throttles bursts hard (429/503/empty 200). On a multi-page crawl
# of a blocked domain, N pages each firing 2-3 archive requests concurrently
# self-throttles into a retry storm (observed: 6 pages -> 140s). Cap concurrent
# archive.org hits so we stay under its limiter — fewer retries, far faster wall
# clock. Lazily created so it binds to the running loop.
_WAYBACK_SEM: "asyncio.Semaphore | None" = None

# ── Wayback latency guards ────────────────────────────────────────────────────
# Wayback is correctly gated as a LAST RESORT (it only fires when the live tiers
# returned nothing or a block page). The problem was never that it runs too
# often — it is what each call costs when archive.org is unhappy:
#
#     3 attempts x 20s timeout + backoffs  ~=  69s for ONE url
#
# and wayback_get makes up to 3 such calls (newest probe -> CDX -> snapshot),
# all funnelled through a 2-permit semaphore. A handful of pages needing the
# archive therefore serialises into minutes of pure waiting — the "it keeps
# going on timeout" behaviour.
#
# Two guards, both env-tunable:
#   1. A shorter default timeout. 20s was chosen for a healthy archive.org; a
#      throttling one will not answer in 20s either, so waiting is wasted.
#   2. A per-crawl circuit breaker. archive.org failure is not per-URL — it is
#      "IA is throttling us right now". Once it has failed this many times in a
#      crawl, every later call would also fail at full timeout cost, so stop
#      asking and let the crawl finish with what the live tiers returned.
#      Kept in a ContextVar so two concurrent crawls cannot trip each other.
_WAYBACK_TIMEOUT_S  = _envf("CRAWL_WAYBACK_TIMEOUT_S", 8.0)
_WAYBACK_FAIL_LIMIT = _envi("CRAWL_WAYBACK_FAIL_LIMIT", 2)
# Hard ceiling on ONE wayback_get, covering all of its staged sub-fetches.
# This is the guard that actually bounds the crawl: the per-request timeout
# above only limits a single HTTP call, and the staged lookup makes several.
_WAYBACK_BUDGET_S   = _envf("CRAWL_WAYBACK_BUDGET_S", 15.0)
_wb_fail_count: "contextvars.ContextVar[int]" = contextvars.ContextVar(
    "wayback_fail_count", default=0
)


def _wayback_tripped() -> bool:
    """True once archive.org has failed enough times to stop being worth asking."""
    return _WAYBACK_FAIL_LIMIT > 0 and _wb_fail_count.get() >= _WAYBACK_FAIL_LIMIT


def _wayback_note(success: bool) -> None:
    """Reset the breaker on any success; step it on failure."""
    if success:
        _wb_fail_count.set(0)
        return
    n = _wb_fail_count.get() + 1
    _wb_fail_count.set(n)
    if n == _WAYBACK_FAIL_LIMIT:
        log.info("  [wayback] %d consecutive failures — skipping archive.org for "
                 "the rest of this crawl", n)


def _wayback_sem() -> "asyncio.Semaphore":
    """Loop-aware accessor — an asyncio.Semaphore belongs to ONE event loop.

    Each company's smart_crawl runs in its own asyncio.new_event_loop() (via
    asyncio.to_thread), so a semaphore created during one crawl and reused by
    the next raises

        RuntimeError: <asyncio.locks.Semaphore ...> is bound to a different event loop

    which is observed repeatedly in production. _LLM_SEM already re-creates
    itself on a loop change; this one did not. Recreating per loop is correct
    here: the semaphore only paces our own requests to archive.org, so a
    per-loop instance still bounds each crawl, and crawls are themselves capped
    by deep_research_concurrency.
    """
    global _WAYBACK_SEM
    try:
        cur_loop = asyncio.get_event_loop()
        if _WAYBACK_SEM is None or getattr(_WAYBACK_SEM, "_loop", cur_loop) is not cur_loop:
            _WAYBACK_SEM = asyncio.Semaphore(2)
    except RuntimeError:
        _WAYBACK_SEM = asyncio.Semaphore(2)
    return _WAYBACK_SEM


# Hosts whose newest capture we've already found broken (e.g. a recent failed SPN
# stub). A broken newest is domain-wide, so once seen we skip the wasted fast-path
# probe (+ its retry backoff) for every other page of that domain and go straight
# to the CDX 'newest GOOD capture' lookup.
_WB_BROKEN_NEWEST: "set[str]" = set()


def _wb_age_days(ts14: str | None) -> int | None:
    if not ts14:
        return None
    try:
        snapped = datetime.strptime(ts14, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        return max(0, (utc_now() - snapped).days)
    except ValueError:
        return None


def _wb_usable(text: str) -> bool:
    """Real page HTML only — reject SPN error stubs, redirect shells, and
    archived 403/anti-bot block pages that would otherwise pass as content."""
    if not text or len(text) < 1500:
        return False
    head = text[:4000]
    if _ANTIBOT_RE.search(head) or CF_RE.search(text[:2000]):
        return False
    return True


async def _wb_fetch(snap_url: str, timeout: float) -> tuple[str | None, int | None]:
    """GET one snapshot, hardened against archive.org's two real failure modes:

    1. Corrupt encoding — broken captures send a brotli/gzip header with an
       undecodable body, which raises on `.text`. `Accept-Encoding: identity`
       asks for raw bytes to dodge most; we still catch the rest.
    2. Rate-limiting — archive.org throttles bursts with 429/503 or an empty
       200. We retry up to 3× with linear backoff before giving up, so a
       transient throttle doesn't look like 'no snapshot'.
    3. Transient playback 404 — IA's playback layer serves a 404 for slow/deep
       snapshots that DO exist and render on a later attempt (confirmed by
       kronikier on real domains). 404 is retried with a longer 3s backoff.

    Genuine non-200 originals (archived 3xx redirects / 403 block pages) come
    back with that status preserved and are NOT in the retry set, so they're
    cleanly rejected by `status != 200`.
    """
    import httpx
    headers = {**_BROWSER_HEADERS, "Accept-Encoding": "identity"}
    # Transient IA failures worth retrying. 404 = deep-storage render lag, not
    # "gone"; 408/5xx = IA hiccups. 403 / 3xx are deliberately absent — an
    # archived block page or redirect is genuine, retrying won't change it.
    _RETRYABLE = (404, 408, 429, 500, 502, 503, 504)
    for attempt in range(3):
        try:
            async with _wayback_sem():
                async with httpx.AsyncClient(follow_redirects=False, timeout=timeout) as c:
                    r = await c.get(snap_url, headers=headers)
            if r.status_code in _RETRYABLE and attempt < 2:
                backoff = 3.0 if r.status_code == 404 else 1.5   # 404 backs off longer
                await asyncio.sleep(backoff * (attempt + 1))
                continue
            if r.status_code != 200:
                return None, None
            try:
                text = r.text
            except Exception:
                log.debug("Wayback response body decode failed (corrupt capture)", exc_info=True)
                return None, None  # corrupt capture (e.g. brotli decode failure)
            if not text and attempt < 2:
                await asyncio.sleep(1.5 * (attempt + 1))   # empty 200 = transient throttle
                continue
            m = _WB_TS_RE.search(str(r.url))
            return text, _wb_age_days(m.group(1) if m else None)
        except Exception:
            if attempt < 2:
                await asyncio.sleep(1.0)
                continue
            log.debug("Wayback fetch failed after retries", exc_info=True)
            return None, None
    return None, None


async def wayback_get(url: str, timeout: float | None = None) -> tuple[str, int | None]:
    """Hard wall-clock cap around the archive lookup.

    The staged lookup below can, worst case, chain 3 x (3 attempts x timeout +
    backoff) ~= 100s+ for a single URL before returning empty. Multiply by a
    2-permit semaphore and a few pages and the crawl stalls for minutes on a
    fallback that is, by definition, optional.

    Cancellation is safe here: the only lock involved is `_wayback_sem`, a plain
    asyncio.Semaphore held via `async with` (releases on cancel), and every httpx
    client is context-managed. This is NOT true of `_BROWSER_SEM`, which is
    thread-backed and cancel-unsafe — do not copy this pattern there.
    """
    try:
        return await asyncio.wait_for(
            _wayback_get_staged(url, timeout), timeout=_WAYBACK_BUDGET_S
        )
    except (asyncio.TimeoutError, asyncio.CancelledError):
        log.warning("  [wayback] %.0fs budget exceeded for %s — archive fallback "
                    "abandoned for this URL", _WAYBACK_BUDGET_S, url[:60])
        _wayback_note(False)
        return "", None


async def _wayback_get_staged(url: str, timeout: float | None = None) -> tuple[str, int | None]:
    """No-proxy fallback for IP-reputation blocks. Returns (html, snapshot_age_days).

    When a site 403s our datacenter egress IP (verified live: eximpedia 403s the
    AWS box), archive.org serves the page from its own non-blocked IP. The `id_`
    suffix returns raw HTML; the served URL carries the capture timestamp, so
    freshness is free (company data: ≤30d good, >60d stale). Browserless and
    leak-free — replaces the same-IP browser tiers that would also 403 and leak.
    Snapshots older than `_WAYBACK_MAX_AGE_DAYS` are rejected outright rather than
    accepted with only a "stale" log label — see that constant's comment.

    Two stages so the common case stays fast but broken archives still recover:
      1. Newest snapshot via far-future timestamp (~2.6s). Use it IF it's real
         HTML (not an SPN stub / archived 403 / redirect shell / corrupt encode).
      2. Else CDX for the newest GOOD capture (200 + text/html, digest-collapsed)
         and fetch those timestamps newest-first. Empty string if none usable.
    """
    import httpx
    # Circuit breaker — archive.org has already failed repeatedly this crawl, so
    # this call would almost certainly also burn a full timeout for nothing.
    if _wayback_tripped():
        return "", None
    if timeout is None:
        timeout = _WAYBACK_TIMEOUT_S
    host = urlparse(url if "://" in url else "https://" + url).netloc.removeprefix("www.")

    # Stage 1 — fast path: newest snapshot, validated. Skipped for hosts already
    # known to have a broken newest capture (avoids re-probing + retry backoff).
    if host not in _WB_BROKEN_NEWEST:
        text, age = await _wb_fetch(_WAYBACK_NEWEST.format(url=url), timeout)
        if _wb_usable(text or ""):
            _wayback_note(True)
            if age is None or age <= _WAYBACK_MAX_AGE_DAYS:
                return text or "", age
            # Usable HTML, just too old. CDX returns captures older than this one,
            # so every candidate it offers would fail the same age gate — walking
            # it would burn up to 8 more throttled fetches (each with retry
            # backoff) to reach a guaranteed empty result. Give up now, and do NOT
            # mark the host: the capture is fine, only its age disqualifies it, and
            # _WB_BROKEN_NEWEST means "newest capture is broken" domain-wide.
            return "", None
        _WB_BROKEN_NEWEST.add(host)

    # Stage 2 — robust path: newest GOOD capture via CDX (skips broken/403/3xx).
    # Strip scheme + query so the page key matches what archive.org indexed, then
    # percent-encode it in full (S7 audit: an unencoded '&' or '=' in the page key
    # would terminate the `url=` param early and inject attacker-controlled CDX
    # query parameters, e.g. `filter=`).
    page_key = url.split("://", 1)[-1].split("?", 1)[0]
    try:
        async with _wayback_sem():
            async with httpx.AsyncClient(timeout=timeout) as c:
                cr = await c.get(_WAYBACK_CDX.format(u=quote(page_key, safe="")),
                                 headers={"User-Agent": _BROWSER_HEADERS.get("User-Agent", "Mozilla/5.0")})
        rows = json.loads(cr.text) if cr.text.strip().startswith("[") else []
    except Exception:
        log.warning("Wayback CDX API fetch failed", exc_info=True)
        _wayback_note(False)
        return "", None
    # CDX returns ascending; rows[0] is the header. Try newest GOOD first.
    for row in reversed(rows[1:]):
        ts = row[0] if isinstance(row, list) else None
        if not ts:
            continue
        text, age = await _wb_fetch(_WAYBACK_TS.format(ts=ts, url=url), timeout)
        if _wb_usable(text or "") and (age is None or age <= _WAYBACK_MAX_AGE_DAYS):
            _wayback_note(True)
            return text or "", age
        # Each miss here costs another throttled, retried fetch. If archive.org
        # is refusing us, stop walking the remaining candidates.
        if _wayback_tripped():
            break
    _wayback_note(False)
    return "", None


async def _attempt_js_fetch(lp_ws: str, url: str, attempt: int) -> str:
    """Single Lightpanda CDP fetch attempt. Returns HTML or '' on any failure."""
    timeout = _TIMEOUT_BROWSER + (5 if attempt else 0)
    async with _BROWSER_SEM:
        try:
            html = await asyncio.wait_for(_lightpanda_cdp_fetch(lp_ws, url), timeout=timeout)
            if html and len(html) > 500:
                return html
        except (asyncio.TimeoutError, Exception) as e:
            _logging.getLogger(__name__).debug("[lightpanda] attempt=%d %s: %s", attempt, url[:60], e)
    return ""


async def js_browser_get(url: str) -> str:
    """Tier-2: JS execution via Lightpanda ONLY (Camoufox removed 2026-06).

    Lightpanda: lightweight Zig+V8 browser via CDP — ~40-123MB, single managed
    `lightpanda serve` process, no Playwright/Firefox → no orphan leak. Handles
    JS-DOM-mutation + XHR/fetch/WebSocket hydration (its strong suit). The ~15%
    it can't render (canvas/WebGL-gated, real-layout-gated, CORS-data, framesets)
    fall through to the hardened Wayback fallback in fetch_homepage — NOT to a
    heavy stealth browser, which couldn't beat our IP-rep/DataDome walls anyway.
    Retries once with an extended timeout: some SPAs finish hydrating in 15–20s but
    are consistent across attempts (same DOM, same render path).
    """
    lp_ws = _start_lightpanda()
    if not lp_ws:
        return ""
    for attempt in range(2):
        html = await _attempt_js_fetch(lp_ws, url, attempt)
        if html:
            return html
    return ""

# ── URL utils ─────────────────────────────────────────────────────────────────
def _norm(u: str) -> str:
    # Must mirror _extract_links()'s pagination-preserving `clean` computation —
    # every seen-set/dedup check downstream keys off this function, so if it
    # stripped pagination query keys while _extract_links() kept them, paginated
    # URLs would collapse back to the same key here and the crawl would never
    # advance past page 1 regardless of what _extract_links() returns.
    #
    # The host is canonicalized too (scheme forced, `www.` dropped, lowercased).
    # Sitemaps and nav routinely mix the two hostnames, and without this the same
    # page is fetched twice under both spellings. Measured on vedhyatech.com: 6 of
    # 40 page slots went to www/non-www pairs of /vision, /mission, /values,
    # /career and the homepage. This is a dedup KEY only — never a fetch URL — so
    # collapsing hosts here cannot redirect a request anywhere.
    p = urlparse(u)
    query = parse_qs(p.query)
    pagination_kv = {k: v for k, v in query.items() if k.lower() in _PAGINATION_QUERY_KEYS}
    kept_query = urlencode(pagination_kv, doseq=True)
    host = p.netloc.lower().removeprefix("www.")
    return p._replace(scheme="https", netloc=host,
                      query=kept_query, fragment="").geturl().rstrip("/")

_IMG_ALT_RE = re.compile(r'<img\b[^>]*?\balt=["\']([^"\']{2,80})["\'][^>]*>', re.I)
# _TAG_RE stops at the first ">", so a comment is only half-eaten and its body
# lands in the extracted text. Every page carrying a banner/dev comment leaked it
# into the LLM chunk (vedhyatech.com: "We are participating in Bengaluru Tech
# summit and are at Stall S34. -->").
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
# Words that describe the decoration, not the thing depicted. An alt made up
# entirely of these ("logo", "preloader image") carries no information; an alt
# that merely contains one ("Bharat Electronics logo") is the brand name.
_DECORATIVE_ALT_WORDS = {"logo", "icon", "image", "img", "photo", "picture", "banner",
                         "arrow", "spacer", "divider", "background", "bg", "thumbnail",
                         "preloader", "loader", "loading", "placeholder", "the", "a"}


def _is_decorative_alt(alt: str) -> bool:
    words = re.findall(r"[a-z]+", alt.lower())
    return not words or all(w in _DECORATIVE_ALT_WORDS for w in words)


def _txt(html: str) -> str:
    """Strip HTML tags, preserve paragraph breaks, collapse whitespace.

    `<img alt>` is promoted to a text line before the tags come off. "Our clients"
    strips, partner walls, certification badges and award rows are image grids —
    the only place the brand names exist is the alt attribute. Dropping it made
    those pages look empty: vedhyatech.com/clients/ yielded 553 chars of text and
    `relationships.customers = []` while the page visibly lists client logos.
    """
    html = re.sub(r"<script\b[^>]*>.*?</script>",   " ", html, flags=re.I | re.S)
    html = re.sub(r"<style\b[^>]*>.*?</style>",     " ", html, flags=re.I | re.S)
    html = re.sub(r"<noscript\b[^>]*>.*?</noscript>"," ", html, flags=re.I | re.S)
    html = _COMMENT_RE.sub(" ", html)
    html = _IMG_ALT_RE.sub(
        lambda m: "\n" if _is_decorative_alt(m.group(1)) else f"\n{m.group(1)}\n", html)
    html = re.sub(r"<(?:p|br|div|h[1-6]|li|tr|blockquote)(?:\s[^>]*)?>", "\n", html, flags=re.I)
    text = _TAG_RE.sub(" ", html)
    lines = [" ".join(ln.split()) for ln in text.splitlines()]
    cleaned = []
    for ln in lines:
        if not ln: cleaned.append(""); continue
        if len(ln) < 3: continue
        cleaned.append(ln)
    result = re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned))
    # Entities are unescaped LAST — doing it before the tag strip would let an
    # escaped &lt;script&gt; in page copy turn back into a real tag.
    return _html.unescape(result).strip()

_IMG_LINE_RE = re.compile(r"^\s*!?\[.*?\](?:\(.*?\))?\s*$", re.M)
_NAV_LINE_RE = re.compile(r"^\s*(?:Home|Menu|Skip to|Search|Share|Follow|Subscribe|Copyright|All rights|Privacy|Terms|Cookie|Accept|Reject|\d+\s+(?:min|read))\b.*$", re.I | re.M)
_SCRIPT_STRIP_RE = re.compile(r'<script\b[^>]*>.*?</script>', re.I | re.S)
_STYLE_STRIP_RE  = re.compile(r'<style\b[^>]*>.*?</style>',  re.I | re.S)
_JS_LINE_RE = re.compile(
    r'^\s*(?:\(function\b|window\.__|\{"@(?:context|type|graph)\b|\bvar\s+\w|\bjQuery\b|'
    r'document\.(?:write|createElement|querySelector)|c\[a\]=c\[a\]\|\|)', re.M)

def _clean_article_text(text: str, title: str) -> str:
    if not text: return ""
    lines = text.splitlines()
    title_norm = title.lower().strip()
    while lines and lines[0].strip().lower() == title_norm:
        lines.pop(0)
    text = "\n".join(lines)
    text = _JS_LINE_RE.sub("", text)
    text = _IMG_LINE_RE.sub("", text)
    text = _NAV_LINE_RE.sub("", text)
    text = re.sub(r"^\s*https?://\S+\s*$", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

_CF_EMAIL_RE = re.compile(
    r'<a[^>]+data-cfemail=["\']([0-9a-f]+)["\'][^>]*>.*?</a>',
    re.I | re.S,
)


def _decode_cf_email(encoded: str) -> str:
    """Decode Cloudflare's XOR email obfuscation (data-cfemail attribute).

    Cloudflare replaces every email address on protected pages with:
      <a href="/cdn-cgi/l/email-protection" data-cfemail="HEXHEX">[email&#160;protected]</a>
    The hex string is a simple XOR cipher: first byte = key, remaining bytes
    XOR'd with key give the original email character codes.
    """
    try:
        key = int(encoded[:2], 16)
        return "".join(chr(int(encoded[i:i + 2], 16) ^ key) for i in range(2, len(encoded), 2))
    except Exception:
        log.debug("CloudFlare email decode failed for encoded value: %r", encoded, exc_info=True)
        return ""


# Contact harvesting (ported from soxoj/kronikier): real emails/phones often
# live ONLY in mailto:/tel: hrefs (visible text is an image/icon) or behind
# [at]/[dot] obfuscation — both are dropped by tag-stripping + link-skipping.
# We surface them as plain text so the extractor sees them. A strict syntax
# gate means prose like "meet at noon on the dot" never becomes a fake email.
_MAILTO_HREF_RE = re.compile(r'href=["\']mailto:([^"\'?>]+)', re.I)
_TEL_HREF_RE    = re.compile(r'href=["\']tel:([^"\'>]+)', re.I)
_VALID_EMAIL_RE = re.compile(r'^[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,255}\.[A-Za-z]{2,24}$')
# user [at] domain [dot] tld  — brackets/parens/braces, English + Russian.
_OBFUS_EMAIL_RE = re.compile(
    r'([A-Za-z0-9._%+\-]{1,64})\s*[\[\(\{]?\s*(?:at|собака)\s*[\]\)\}]?\s*'
    r'([A-Za-z0-9.\-]{1,255}?)\s*[\[\(\{]?\s*(?:dot|точка)\s*[\]\)\}]?\s*([A-Za-z]{2,24})',
    re.I,
)


def _harvest_href_contacts(raw_html: str) -> str:
    import html as _htmllib
    emails: list[str] = []
    phones: list[str] = []
    for m in _MAILTO_HREF_RE.findall(raw_html):
        v = _htmllib.unescape(m).split("?", 1)[0].strip().lower()
        if _VALID_EMAIL_RE.match(v) and v not in emails:
            emails.append(v)
    for a, b, c in _OBFUS_EMAIL_RE.findall(raw_html):
        cand = f"{a}@{b}.{c}".lower()
        if _VALID_EMAIL_RE.match(cand) and cand not in emails:
            emails.append(cand)
    for t in _TEL_HREF_RE.findall(raw_html):
        v = t.strip()
        if len(re.sub(r"[^\d+]", "", v)) >= 7 and v not in phones:
            phones.append(v)
    parts = []
    if emails:
        parts.append("Contact emails: " + ", ".join(emails[:15]))
    if phones:
        parts.append("Contact phones: " + ", ".join(phones[:15]))
    return ("  ".join(parts) + "  ") if parts else ""


def _pre_clean_html(html: str) -> str:
    # Decode Cloudflare email obfuscation before stripping tags so the real
    # email address ends up in the extracted text.
    html = _CF_EMAIL_RE.sub(lambda m: _decode_cf_email(m.group(1)), html)
    # Surface mailto:/tel:/[at][dot] contacts the tag-strip would otherwise drop.
    harvested = _harvest_href_contacts(html)
    if harvested:
        html = harvested + html
    html = _SCRIPT_STRIP_RE.sub(' ', html)
    html = _STYLE_STRIP_RE.sub(' ', html)
    return html

def _extract_links(html: str, base: str) -> list[dict]:
    """Extract all crawlable same-domain links from an HTML page.

    Covers: <a href> anchors, <link rel=canonical>, <link rel=next> pagination,
    and SPA data-href/data-url/data-link/data-route/data-path attributes.
    """
    bh = urlparse(base).netloc.removeprefix("www.")
    base_norm = base.rstrip("/")
    seen: set[str] = set()
    links: list[dict] = []

    def _maybe_add(href: str, label: str) -> None:
        href = re.sub(r"[\r\n\t]", "", href).strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            return
        full = urljoin(base, href)
        p = urlparse(full)
        if p.netloc.removeprefix("www.") != bh:
            return
        # Query-string pagination (?page=2, ?paged=2, ?p=2) must survive the
        # dedup strip below, or every page-N link collapses to the bare page-1
        # path (already in `seen`/`base_norm`) and gets silently dropped —
        # WooCommerce/Shopify/WP default pagination never crawls past page 1.
        query = parse_qs(p.query)
        pagination_kv = {k: v for k, v in query.items() if k.lower() in _PAGINATION_QUERY_KEYS}
        kept_query = urlencode(pagination_kv, doseq=True)
        clean = p._replace(fragment="", query=kept_query).geturl()
        if clean in seen or clean == base_norm or _SKIP_RE.search(clean):
            return
        seen.add(clean)
        links.append({"url": clean, "text": label[:120]})

    for href, anchor in _A_HREF_RE.findall(html):
        _maybe_add(href, _TAG_RE.sub("", anchor).strip())
    for href in _CANONICAL_RE.findall(html):
        _maybe_add(href, "canonical")
    for href in _NEXT_PAGE_RE.findall(html):
        _maybe_add(href, "next page")
    for href in _DATA_URL_RE.findall(html):
        _maybe_add(href, "")
    return links

_SITEMAP_PATHS = [
    "/sitemap.xml", "/sitemap_index.xml", "/sitemap.xml.gz",
    "/news-sitemap.xml", "/sitemap_pages.xml",
    "/sitemap-images.xml", "/sitemap_news.xml",
    "/wp-sitemap.xml",
]

# Adaptive timeouts — bucketed by resource type (Fix 10a)
_TIMEOUT_FAST     = _envi("CRAWL_TIMEOUT_FAST", 8)       # HTML pages: most respond < 3s, 8s is p99
_TIMEOUT_DOC      = _envi("CRAWL_TIMEOUT_DOC", 25)       # PDFs / XLSX / DOCX: larger payloads
_TIMEOUT_SITEMAP  = _envi("CRAWL_TIMEOUT_SITEMAP", 10)   # sitemap.xml: mid-size XML
_TIMEOUT_BROWSER  = _envi("CRAWL_TIMEOUT_BROWSER", 20)   # Lightpanda JS-render ceiling
# Wall-clock ceiling for Phase-1 discovery (sitemap + feed + wp_rest combined).
# These run concurrently with the homepage fetch and are pure optimisation: the
# BFS is itself budget-capped, so extra seed URLs beyond what BFS can consume
# add latency but no extracted detail. Measured: 35s unbounded on a real domain.
_DISCOVERY_BUDGET_S = _envf("CRAWL_DISCOVERY_BUDGET_S", 12.0)

async def _sitemap_links(base_url: str, stats: dict | None = None) -> list[dict]:
    """Discover page URLs via robots.txt + 8 standard paths + trafilatura fallback.

    Recursively resolves sitemap indexes (sub-sitemap .xml URLs) one level deep.
    Writes per-source counts into `stats` dict if provided (for observability).
    """
    from backend.utils.domain import safe_fetch
    domain_url = base_url.rstrip("/")
    bh = urlparse(base_url).netloc.removeprefix("www.")

    # Tier 1: robots.txt Sitemap: directives (stdlib, no deps)
    # NB: RobotFileParser.read() is SYNC blocking I/O — must run in a thread
    # so it doesn't freeze the event loop and stall parallel discovery tasks.
    robot_sitemaps: list[str] = []
    crawl_delay = None
    def _robots_sync() -> tuple[list[str], int | None]:
        from urllib.robotparser import RobotFileParser
        rp = RobotFileParser()
        rp.set_url(f"{domain_url}/robots.txt")
        try:
            rp.read()
            sm  = list(rp.site_maps() or [])
            cd  = rp.crawl_delay("*")
            return sm, int(cd) if cd else None
        except Exception:
            log.debug("robots.txt parse failed for %s", domain_url, exc_info=True)
            return [], None
    try:
        robot_sitemaps, crawl_delay = await asyncio.wait_for(
            asyncio.to_thread(_robots_sync), timeout=5.0
        )
    except (asyncio.TimeoutError, Exception):
        pass

    if stats is not None:
        stats["robots_txt"] = {
            "fetched": bool(robot_sitemaps) or crawl_delay is not None,
            "sitemaps_declared": len(robot_sitemaps),
            "crawl_delay_seconds": int(crawl_delay) if crawl_delay else None,
        }

    async def _fetch_sitemap(sm_url: str) -> tuple[str, list[str]]:
        try:
            r = await safe_fetch(sm_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if r.status_code != 200:
                return (sm_url, [])
            body = r.content
            # A raw .xml.gz sitemap is often served without a Content-Encoding
            # header (the URL itself is the compressed artifact), so httpx never
            # auto-decompresses it and r.text is unreadable gzip bytes that never
            # match _LOC_RE. Decompress explicitly whenever the payload looks
            # gzip-magic'd, regardless of what the URL extension claims.
            if body[:2] == b"\x1f\x8b":
                try:
                    body = gzip.decompress(body)
                except Exception:
                    log.warning("gzip decompress failed for sitemap: %s", sm_url, exc_info=True)
                    return (sm_url, [])
            text = body.decode("utf-8", errors="ignore")
            locs = _LOC_RE.findall(text)
            return (sm_url, [l.strip() for l in locs])
        except Exception:
            log.debug("sitemap fetch failed: %s", sm_url, exc_info=True)
            return (sm_url, [])

    # Tier 2: fetch candidates concurrently. When robots.txt explicitly declared
    # Sitemap: directive(s), that's the standards-based, authoritative answer —
    # skip guessing the 8 standard paths entirely. Measured live on
    # fluidflowengg.com: robots.txt correctly named /sitemap.xml in 0.34s, but
    # the code still fired all 8 guessed paths anyway, 7 of which were
    # guaranteed 404s costing 1.5-4.6s each (TLS handshake + RTT to the site's
    # real host, not something retrying faster fixes) — ~5s of pure waste on
    # the critical path for any site that properly declares its sitemap. Only
    # fall back to guessing when robots.txt gave nothing (missing robots.txt,
    # no Sitemap: line, or the declared one 404s below and sub_to_fetch stays
    # empty — see the recursion/fallback logic further down, unchanged).
    candidates = list(dict.fromkeys(robot_sitemaps)) if robot_sitemaps else \
        [f"{domain_url}{p}" for p in _SITEMAP_PATHS]
    results = await asyncio.gather(*[_fetch_sitemap(c) for c in candidates])

    standard_paths_status: dict[str, int] = {}
    all_locs: list[str] = []
    indexes_found = 0
    sub_to_fetch: list[str] = []
    for sm_url, locs in results:
        path = urlparse(sm_url).path or sm_url
        standard_paths_status[path] = 200 if locs else 0
        if not locs:
            continue
        sub = [l for l in locs if l.rstrip("/").endswith((".xml", ".xml.gz"))]
        if sub:
            indexes_found += 1
        sub_to_fetch.extend(sub[:20])
        if len(all_locs) < _SITEMAP_LOC_CAP:
            all_locs.extend(l for l in locs if l not in sub)

    # Recurse into sub-sitemaps one level deep
    sub_fetched = 0
    if sub_to_fetch:
        # Deduplicate sub-sitemap URLs in case multiple indexes reference the same one
        sub_unique = list(dict.fromkeys(sub_to_fetch))[:50]
        sub_results = await asyncio.gather(*[_fetch_sitemap(s) for s in sub_unique], return_exceptions=True)
        for sr in sub_results:
            if isinstance(sr, tuple) and sr[1]:
                sub_fetched += 1
                remaining = _SITEMAP_LOC_CAP - len(all_locs)
                if remaining > 0:
                    all_locs.extend(sr[1][:remaining])

    # Tier 3: trafilatura as comprehensive fallback (handles recursive indexes natively)
    # NB: sitemap_search() is SYNC blocking I/O (network + parsing) — must run in
    # a thread with a hard timeout, otherwise it stalls the event loop for minutes
    # on slow sitemaps.
    trafilatura_count = 0
    def _traf_sitemap_sync() -> list[str]:
        try:
            from trafilatura.sitemaps import sitemap_search
            return list(sitemap_search(domain_url) or [])
        except Exception:
            log.debug("trafilatura sitemap_search failed for %s", domain_url, exc_info=True)
            return []
    try:
        fallback = await asyncio.wait_for(
            asyncio.to_thread(_traf_sitemap_sync), timeout=15.0
        )
        trafilatura_count = len(fallback)
        remaining = _SITEMAP_LOC_CAP - len(all_locs)
        if remaining > 0:
            all_locs.extend(fallback[:remaining])
    except (asyncio.TimeoutError, Exception):
        pass

    if stats is not None:
        stats["standard_paths"] = standard_paths_status
        stats["sitemap_index_recursion"] = {
            "indexes_found": indexes_found,
            "sub_sitemaps_fetched": sub_fetched,
            "total_leaf_urls": len(all_locs),
        }
        stats["trafilatura_fallback_urls"] = trafilatura_count

    # Dedupe + filter + format
    seen: set[str] = set()
    links: list[dict] = []
    for loc in all_locs:
        loc = loc.strip()
        if not loc.startswith("http"):
            continue
        if urlparse(loc).netloc.removeprefix("www.") != bh:
            continue
        if _SKIP_RE.search(loc) or _DOC_RE.search(loc):
            continue
        clean = urlparse(loc)._replace(fragment="", query="").geturl()
        if clean in seen:
            continue
        seen.add(clean)
        slug = urlparse(clean).path.rstrip("/").rsplit("/", 1)[-1]
        links.append({"url": clean, "text": slug.replace("-", " ").replace("_", " ")})
    return links


async def _extract_feed_links(
    feed_url: str, bh: str, seen: set[str], links: list[dict]
) -> None:
    """Fetch one RSS/Atom feed and append valid same-domain item URLs to `links`.

    Extracted from _feed_links to move the try-except out of the loop body.
    Mutates `seen` and `links` in place.
    """
    from backend.utils.domain import safe_fetch
    try:
        r = await safe_fetch(feed_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return
        for href in re.findall(r'<link>\s*(https?://[^<\s]+)\s*</link>', r.text):
            href = href.strip()
            if urlparse(href).netloc.removeprefix("www.") != bh or href in seen:
                continue
            if _SKIP_RE.search(href):
                continue
            seen.add(href)
            slug = urlparse(href).path.rstrip("/").rsplit("/", 1)[-1]
            links.append({"url": href, "text": slug.replace("-", " ").replace("_", " ")})
        for href in re.findall(
            r'<link[^>]+href=["\']([^"\']+)["\'][^>]+rel=["\']alternate["\']',
            r.text, re.I,
        ):
            if urlparse(href).netloc.removeprefix("www.") != bh or href in seen:
                continue
            if _SKIP_RE.search(href):
                continue
            seen.add(href)
            links.append({"url": href, "text": ""})
    except Exception:
        log.debug("feed link extraction failed for %s", feed_url, exc_info=True)


async def _feed_links(base_url: str) -> list[dict]:
    """Discover RSS/Atom feeds and extract their item URLs — no crawling required.

    `find_feed_urls()` is SYNC blocking network I/O — must run in a thread
    so it doesn't stall the event loop and block the parallel discovery fan-out.
    """
    def _find_feeds_sync() -> list[str]:
        try:
            from trafilatura.feeds import find_feed_urls
            return list(find_feed_urls(base_url) or [])
        except Exception:
            log.debug("trafilatura find_feed_urls failed for %s", base_url, exc_info=True)
            return []
    try:
        feed_urls = await asyncio.wait_for(
            asyncio.to_thread(_find_feeds_sync), timeout=12.0
        )
    except (asyncio.TimeoutError, Exception):
        return []
    if not feed_urls:
        return []
    bh = urlparse(base_url).netloc.removeprefix("www.")
    seen: set[str] = set()
    links: list[dict] = []
    for feed_url in feed_urls[:5]:
        await _extract_feed_links(feed_url, bh, seen, links)
    return links


async def _wp_rest_links(base_url: str, stats: dict | None = None) -> list[dict]:
    """WordPress REST API enumeration. Returns all post+page URLs via /wp-json/wp/v2.

    Detection via /wp-json/wp/v2/posts?per_page=1 probe. Paginated up to 2000
    posts and 2000 pages (20 pages × 100 per_page). Writes detection flags
    and per-endpoint counts into `stats` dict if provided.
    """
    from backend.utils.domain import safe_fetch
    base_url = base_url.rstrip("/")
    probe = f"{base_url}/wp-json/wp/v2/posts?per_page=1"
    try:
        r = await safe_fetch(probe, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        # Accept either json content-type OR a body that parses as a JSON list
        # (some WP installs return application/octet-stream or text/plain on
        # behind-CDN edges, but the body is still valid JSON.)
        ok = False
        if r.status_code == 200:
            ct = r.headers.get("content-type", "").lower()
            if "json" in ct:
                ok = True
            else:
                try:
                    body = r.json()
                    ok = isinstance(body, list)
                except Exception:
                    ok = False
        if not ok:
            if stats is not None:
                stats["wp_rest_detected"] = False
                stats["wp_rest_posts"] = 0
                stats["wp_rest_pages"] = 0
            return []
    except Exception:
        if stats is not None:
            stats["wp_rest_detected"] = False
            stats["wp_rest_posts"] = 0
            stats["wp_rest_pages"] = 0
        return []

    bh = urlparse(base_url).netloc.removeprefix("www.")
    links: list[dict] = []
    counts = {"posts": 0, "pages": 0}
    _wp_deadline = time.perf_counter() + 45  # hard 45 s cap across all paginated requests
    for endpoint, key in (("/wp-json/wp/v2/posts", "posts"), ("/wp-json/wp/v2/pages", "pages")):
        page = 1
        while page <= 20:
            if time.perf_counter() > _wp_deadline:
                log.debug("_wp_rest_links: 45s budget exceeded, stopping pagination for %s", base_url)
                break
            url = f"{base_url}{endpoint}?per_page=100&page={page}&_fields=link,title"
            try:
                r = await safe_fetch(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code != 200:
                    break
                items = r.json()
                if not isinstance(items, list) or not items:
                    break
                for item in items:
                    link = item.get("link", "")
                    if not link or urlparse(link).netloc.removeprefix("www.") != bh:
                        continue
                    title_rendered = ""
                    t = item.get("title")
                    if isinstance(t, dict):
                        title_rendered = t.get("rendered") or ""
                    title = _TAG_RE.sub("", title_rendered).strip()
                    links.append({"url": link, "text": title[:120]})
                    counts[key] += 1
                if len(items) < 100:
                    break
                page += 1
            except Exception:
                break

    if stats is not None:
        stats["wp_rest_detected"] = True
        stats["wp_rest_posts"] = counts["posts"]
        stats["wp_rest_pages"] = counts["pages"]
    return links


def _jsonld_urls(html: str, bh: str) -> list[str]:
    """Extract same-domain URLs embedded in JSON-LD structured data blocks."""
    urls: list[str] = []
    for block in _JSONLD_RE.findall(html):
        try:
            data = json.loads(block)
        except Exception:
            continue
        if isinstance(data, dict):
            items = data.get("@graph", [data])
        elif isinstance(data, list):
            items = data
        else:
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            for field in ("url", "mainEntityOfPage", "@id"):
                val = item.get(field)
                if isinstance(val, dict):
                    val = val.get("@id") or val.get("url")
                if isinstance(val, str) and val.startswith("http"):
                    if urlparse(val).netloc.removeprefix("www.") == bh:
                        urls.append(val)
    return urls

# ── Phase 1: Homepage ─────────────────────────────────────────────────────────
async def fetch_homepage(base_url: str) -> tuple[str, list[dict], str, bool]:
    """Fetch homepage and extract nav links. Returns (text, links, html, escalated).

    Cascade (Camoufox/Firefox removed 2026-06):
      Tier 0:   httpx + full browser headers — fastest, works on most sites
      Tier 1:   direct curl_cffi (real Chrome JA3/JA4 TLS) — fingerprint-gated sites
      Tier 1.5: hardened Wayback — IP-reputation blocks (datacenter 403)
      Tier 2:   Lightpanda (CDP, ~40-123MB, no leak) — JS-rendered SPA shells
      Final:    Wayback catch-all — anything still unrendered/blocked

    Escalation criterion: anchor-count < 8 OR `_looks_blocked()` is true (anti-bot
    signal). Without the blocked-check, a Cloudflare 403 page (~5KB, 1 anchor)
    used to slip past the cascade silently.

    _DOMAIN_TIER_CACHE skips tiers already known to fail for this host, so repeated
    crawls of the same domain go directly to the working tier without wasted probes —
    for _DOMAIN_TIER_CACHE_TTL, after which the host is re-probed from tier 0 in case
    a transient block (rate limit, one-off WAF rule) has since cleared.
    """
    host = urlparse(base_url).netloc.removeprefix("www.")
    start_tier = _cached_domain_tier(host)

    html = ""
    links: list[dict] = []
    blocked = True

    # Tier 0
    if start_tier <= 0:
        html = await httpx_get(base_url, timeout=_TIMEOUT_FAST)
        links = _extract_links(html, base_url) if html else []
        blocked = _looks_blocked(0, html) if html else True
        # Flaw fix: only short-circuit at Tier 0 when we actually extracted links.
        # Large HTML alone is NOT a signal links were found — TLS-fingerprint-gated
        # SSR sites (e.g. eximpedia.app) serve a big shell to plain httpx with 0
        # anchors, but the full page (200KB+, 200+ links) to the scrapling/curl_cffi
        # Chrome-fingerprint tier below. Requiring len(links) >= 8 forces escalation
        # to Tier 1 instead of returning nav=0.
        if html and not blocked and len(links) >= _MIN_ANCHORS:
            _set_domain_tier(host, 0)
            return _txt(html), links, html, False

    # Tier 1: scrapling fallback (bot-protected sites)
    if start_tier <= 1:
        s_html = await scrapling_get(base_url)
        s_links = _extract_links(s_html, base_url) if s_html else []
        s_blocked = _looks_blocked(0, s_html) if s_html else True
        # Promote scrapling result only when it yields more anchors — HTML size
        # alone can be larger for a CF challenge page (no nav links) than for a
        # clean Tier-0 response, which would overwrite the better Tier-0 result.
        if s_html and not s_blocked and len(s_links) > len(links):
            html, links, blocked = s_html, s_links, False
        # Same fix as Tier 0: a big SPA shell with < 8 links must escalate to the JS
        # browser tier instead of short-circuiting here on HTML size alone.
        if html and not blocked and len(links) >= _MIN_ANCHORS:
            _set_domain_tier(host, 1)
            return _txt(html), links, html, False

    # Tier 1.5: IP-reputation block. Both httpx and scrapling came back blocked
    # (403 / anti-bot), so our egress IP is blocked at the edge — the browser
    # tiers below share that IP and would also 403 while leaking RAM (the OOM
    # path). Pull the newest archive.org snapshot instead: served from archive's
    # non-blocked IP, browserless, ~3s. Skip the browsers entirely on success.
    # Guard: only meaningful when we actually ran the IP-bound tiers (start_tier<=1).
    # If start_tier==2 we jumped here with `blocked=True` as an uninitialised default
    # — Wayback would fire erroneously before Lightpanda, which we already know works.
    if start_tier <= 1 and blocked:
        wb_html, wb_age = await wayback_get(base_url)
        wb_links = _extract_links(wb_html, base_url) if wb_html else []
        if wb_html and len(wb_links) >= _MIN_ANCHORS:
            fresh = "fresh" if (wb_age is not None and wb_age <= 30) else (
                "aging" if (wb_age is not None and wb_age <= 60) else "stale")
            log.info("[%s] wayback bypass: %d links, snapshot age=%sd (%s)", base_url, len(wb_links), wb_age, fresh)
            return _txt(wb_html), wb_links, wb_html, True

    # Tier 2: True SPA shell — js_browser_get handles LP retry internally
    escalated = False
    js_html = await js_browser_get(base_url)
    js_blocked = _looks_blocked(0, js_html) if js_html else True
    if js_html and not js_blocked and len(js_html) > len(html or ""):
        js_links = _extract_links(js_html, base_url)
        if len(js_links) > len(links):
            _set_domain_tier(host, 2)
            escalated = True
            return _txt(js_html), js_links, js_html, escalated

    # Final catch-all: every live tier failed to yield real nav. If we haven't
    # already pulled the archive (non-403 path — e.g. a JS shell the browsers
    # couldn't render), try the newest snapshot before giving up with nav=0.
    # `not blocked` ≡ the old `not wb_tried` — Wayback is only retried here when
    # the IP-rep branch above was skipped (unblocked path).
    if not blocked and len(links) < _MIN_ANCHORS:
        wb_html, wb_age = await wayback_get(base_url)
        wb_links = _extract_links(wb_html, base_url) if wb_html else []
        if wb_html and len(wb_links) > len(links):
            log.info("[%s] wayback final fallback: %d links, snapshot age=%sd", base_url, len(wb_links), wb_age)
            return _txt(wb_html), wb_links, wb_html, True

    return (_txt(html) if html else "", links, html or "", escalated)

# ── Document fetch + parse ────────────────────────────────────────────────────
async def _fetch_document(url: str) -> tuple[bytes, str]:
    from backend.utils.domain import safe_fetch
    ext = urlparse(url).path.rsplit(".", 1)[-1].lower() if "." in urlparse(url).path else ""
    try:
        r = await safe_fetch(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if r.status_code == 200:
            return r.content, ext
    except Exception:
        pass
    return b"", ext

def _parse_doc(data: bytes, ext: str) -> str:
    ext = ext.lower().lstrip(".")
    try:
        if ext == "pdf":
            import pymupdf
            doc = pymupdf.open(stream=data, filetype="pdf")
            pages_text: list[str] = []
            for page in doc:
                parts: list[str] = []
                try:
                    tabs = page.find_tables()
                    table_rects = [t.bbox for t in tabs.tables]
                    for blk in page.get_text("blocks"):
                        bx0, by0, bx1, by1, txt = blk[0], blk[1], blk[2], blk[3], blk[4].strip()
                        if not txt: continue
                        in_table = any(bx0 >= r[0]-5 and by0 >= r[1]-5 and bx1 <= r[2]+5 and by1 <= r[3]+5
                                       for r in table_rects)
                        if not in_table:
                            parts.append(txt)
                    for tab in tabs.tables:
                        rows = tab.extract()
                        if rows:
                            parts.append("\n".join(" | ".join(str(c or "").strip() for c in row) for row in rows))
                except Exception:
                    parts = [page.get_text()]
                pages_text.append("\n".join(parts))
            return "\n\n".join(pages_text).strip()
        if ext in ("doc", "docx"):
            import io; from docx import Document
            return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs if p.text.strip())
        if ext in ("xls", "xlsx"):
            import io, openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
            lines = []
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    row_text = " | ".join(str(c) for c in row if c is not None)
                    if row_text.strip(): lines.append(row_text)
            return "\n".join(lines)
        if ext in ("ppt", "pptx"):
            import io; from pptx import Presentation
            prs = Presentation(io.BytesIO(data))
            return "\n".join(s.text.strip() for sl in prs.slides for s in sl.shapes if hasattr(s, "text") and s.text.strip())
        if ext == "csv":
            return data.decode("utf-8", errors="replace")
    except Exception:
        pass
    return ""

def extract_articles(pages: list[dict]) -> list[dict]:
    import trafilatura
    _EMPTY = {"summary": "", "entities": [], "figures": [], "quotes": []}
    arts: list[dict] = []
    for p in pages:
        url, cat = p["url"], p.get("category", "news")
        if p.get("doc_bytes"):
            text = _parse_doc(p["doc_bytes"], p.get("doc_ext", ""))
            if not text or len(text) < 100: continue
            fname = urlparse(url).path.rsplit("/", 1)[-1]
            title = re.sub(r"\s+", " ", re.sub(r"[_%-]", " ", fname.rsplit(".", 1)[0])).strip().title()
            dm = re.search(r"(\d{2})(\d{2})(\d{4})", fname)
            arts.append({"title": title or fname, "date": f"{dm.group(3)}-{dm.group(2)}-{dm.group(1)}" if dm else "",
                         "type": _CAT_TYPE.get(cat, "document"), "url": url, "author": "", "full_text": text, **_EMPTY})
            continue
        html = p.get("html", "")
        if not html: continue
        html_clean = _pre_clean_html(html)
        text = trafilatura.extract(html_clean, url=url, include_tables=True, favor_precision=True, no_fallback=False)
        if not text or len(text) < 150:
            # Fallback when trafilatura under-extracts. Was scrapling's
            # Adaptor.get_all_text — replaced 2026-06 with our own tag-stripping
            # text cleaner (_txt) so scrapling can be uninstalled.
            text = _txt(html_clean)
            if not text or len(text) < 150:
                continue
        meta = trafilatura.extract_metadata(html, default_url=url)
        mt = (meta.title if meta else None) or ""
        title = ""
        m = _H1_RE.search(html)
        if m:
            h1 = _TAG_RE.sub("", m.group(1)).strip()
            if h1 and len(h1) > 5 and h1.lower() not in {"news","blog","press","events","media"}: title = h1
        if not title and mt and len(mt) > 10 and " " in mt: title = mt
        if not title:
            slug = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
            title = re.sub(r"[-_]", " ", slug).strip().title() if len(slug) > 5 else ""
        if not title: continue
        clean_text = _clean_article_text(text, title)
        if len(clean_text) < 100: continue
        date_str = str(meta.date)[:10] if meta and meta.date else ""
        if not date_str:
            date_str = _parse_text_date(clean_text)
        arts.append({"title": title, "date": date_str,
                     "type": _CAT_TYPE.get(cat, "news"), "url": url,
                     "author": str(meta.author) if meta and meta.author else "",
                     "full_text": clean_text, **_EMPTY})
    arts.sort(key=lambda a: a.get("date", ""), reverse=True)
    return arts

# ── Phase 4b: LLM metadata extraction (chunked + merged) ─────────────────────
def _norm_name(s: str) -> str:
    s = re.sub(r"\b(?:mr|mrs|ms|dr|shri|smt|late|former|prof)\.?\s*", "", s.lower())
    return re.sub(r"[^a-z0-9]", "", s)

def _merge_intel(base: dict, patch: dict) -> dict:
    if not base: return patch
    if not patch: return base
    for key, val in patch.items():
        if key not in base:
            base[key] = val
        elif isinstance(val, dict):
            base[key] = _merge_intel(base.get(key, {}), val)
        elif isinstance(val, list):
            existing_names: set[str] = set()
            try:
                existing_exact = {json.dumps(e, sort_keys=True) for e in base.get(key, [])}
            except (TypeError, ValueError):
                existing_exact = {id(e) for e in base.get(key, [])}
            for e in base.get(key, []):
                if isinstance(e, dict) and e.get("name"):
                    existing_names.add(_norm_name(e["name"]))
            for item in val:
                try:
                    exact_key = json.dumps(item, sort_keys=True)
                except (TypeError, ValueError):
                    exact_key = id(item)
                if exact_key in existing_exact:
                    continue
                # Name dedup applies to ANY named entity, not just people. The
                # `"title" in item` guard restricted it to leadership rows, so two
                # chunks that both saw the same product emitted it twice with
                # slightly different wording — the exact-JSON check above cannot
                # catch that. vedhyatech.com returned Ekalavya VT-HDL-G and -L
                # twice each in a 12-product list for exactly this reason.
                if isinstance(item, dict) and item.get("name"):
                    nm = _norm_name(item["name"])
                    if nm in existing_names:
                        continue
                    existing_names.add(nm)
                base.setdefault(key, []).append(item)
                existing_exact.add(exact_key)
        elif isinstance(val, str) and val and not base.get(key):
            base[key] = val
        elif isinstance(val, (int, float)) and val and not base.get(key):
            base[key] = val
    return base

_SOURCE_PLATFORMS = {
    "wikipedia", "wikimedia", "linkedin", "crunchbase", "bloomberg", "reuters",
    "forbes", "zoominfo", "owler", "pitchbook", "tracxn", "moneycontrol",
    "yourstory", "inc42", "techcrunch", "venturebeat", "google", "google news",
    "bing", "duckduckgo", "facebook", "twitter", "x.com", "youtube", "instagram",
    "glassdoor", "indeed", "ambitionbox", "businesswire", "prnewswire",
    "the economic times", "economic times", "livemint", "mint", "business standard",
    "hindu business line", "the hindu", "ndtv", "cnbc", "ft", "financial times",
    "wsj", "wall street journal", "nyt", "new york times", "medium", "substack",
    "quora", "reddit", "github", "yelp", "trustpilot", "g2", "capterra",
}
_SOURCE_CATEGORIES = {
    "online encyclopedia", "encyclopedia", "reference site", "reference",
    "news aggregator", "news outlet", "news website", "news",
    "business directory", "directory", "search engine", "social network",
    "social media", "professional network", "blogging platform", "blog",
    "video platform", "review site", "data provider",
}

def _strip_source_artifacts(merged: dict) -> None:
    """Drop product/service entries that name the SOURCE platform rather than the
    focal company's own offering. Belt-and-suspenders for the prompt rule."""
    biz = merged.get("business") or {}
    for key in ("products", "services"):
        items = biz.get(key)
        if not isinstance(items, list):
            continue
        cleaned: list = []
        for it in items:
            if not isinstance(it, dict):
                cleaned.append(it)
                continue
            name = (it.get("name") or "").strip().lower()
            cat  = (it.get("category") or "").strip().lower()
            if name in _SOURCE_PLATFORMS or cat in _SOURCE_CATEGORIES:
                continue
            cleaned.append(it)
        biz[key] = cleaned

_LLM_SEM: asyncio.Semaphore | None = None

# ── Extraction provider: DeepSeek only ─────────────────────────────────────────
# DeepSeek has no strict json_schema mode, so response_format stays json_object;
# the schema itself is emitted inside the static prefix built by _extract_chunk,
# where it is cacheable instead of being re-billed after the content on every
# chunk. Appending it again here would both duplicate it and break the cache.
# OpenAI is kept for embeddings only (see backend/config.py) — no LLM chat path
# in this file uses it.


async def _run_extraction(domain: str, prompt: str, schema: dict, max_tokens: int
                          ) -> tuple[dict, int, int, float]:
    """One extraction attempt. Raises on failure."""
    model = LLM_MODEL
    api = _get_extraction_client(model)
    source = "deepseek" if model.lower().startswith("deepseek-") else "openai"
    response_format: dict = {"type": "json_object"}

    # Local-only: a CLI-spawn-backed proxy (see backend.email._llm's
    # DEEPSEEK_BASE_URL override) has a much higher realistic per-turn latency
    # (observed 29-38s, vs real DeepSeek/OpenAI's low-single-digit-second calls)
    # — a fixed 30s here raced against that and timed out extraction on roughly
    # half of real calls. EXTRACTION_TIMEOUT_S only widens the timeout when that
    # override is actually active; unset (prod / real DeepSeek) keeps 30s exactly
    # as before.
    _extraction_timeout = 30
    if os.environ.get("DEEPSEEK_BASE_URL"):
        _extraction_timeout = int(os.environ.get("EXTRACTION_TIMEOUT_S", "90"))
    _extra_body: dict = {}
    if source == "deepseek":
        # Per-user_id isolation (KVCache / concurrency scheduling — see
        # api-docs.deepseek.com/quick_start/rate_limit). Sanitized to DeepSeek's
        # required charset since our own user ids are Auth0-style ("auth0|abc123").
        import re as _re
        from backend.cost.meter import current_user_id
        _uid = _re.sub(r"[^a-zA-Z0-9\-_]", "", current_user_id())[:512]
        if _uid:
            _extra_body["user_id"] = _uid
    resp = await api.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        response_format=response_format,
        temperature=0, max_tokens=max_tokens, timeout=_extraction_timeout,
        **({"extra_body": _extra_body} if _extra_body else {}),
    )

    tin = tout = 0
    cost = 0.0
    try:
        from backend.cost.meter import record, estimate_llm_cost_inr
        _u   = resp.usage
        tin  = (getattr(_u, "prompt_tokens", 0) or 0) if _u else 0
        tout = (getattr(_u, "completion_tokens", 0) or 0) if _u else 0
        # DeepSeek reports the prefix-cache split and bills hits at ~3% of the
        # miss rate. Pricing `tin` wholesale as a cache MISS (what this did
        # before) both overstates spend and makes the static-prefix ordering in
        # the prompt above look like it saved nothing. prompt_tokens is the
        # TOTAL, so the fresh count is total - hits; prefer the reported miss
        # figure when present. Absent on OpenAI, where both stay 0 and this
        # reduces to the previous behaviour.
        _hit  = ((getattr(_u, "prompt_cache_hit_tokens", 0) or 0) if _u else 0)
        _miss = ((getattr(_u, "prompt_cache_miss_tokens", 0) or 0) if _u else 0)
        if _hit > tin:
            _hit = tin
        _fresh = _miss if _miss else max(0, tin - _hit)
        cost = estimate_llm_cost_inr(model, _fresh, tout, cache_read_tokens=_hit)
        # tokens_in stays the TOTAL so per-run token counts remain comparable
        # with historic rows; only the pricing splits hit vs miss.
        record(source=source, operation="smart_crawl_extraction",
               cost_inr=cost, tokens_in=tin, tokens_out=tout, model_name=model)
    except Exception as _ce:
        log.warning("smart_crawl cost record failed for %s: %s", domain, _ce)
    return json.loads(resp.choices[0].message.content or "{}"), tin, tout, cost


async def _extract_chunk(domain: str, chunk: str, until: str | None,
                         schema: dict, mode_label: str, max_tokens: int = 5000
                         ) -> tuple[dict, int, int, float]:
    """Returns (intel_dict, tokens_in, tokens_out, cost_inr).

    Tokens and cost surface into extraction_stats so callers can show real
    LLM spend per crawl without having to query the cost meter DB.
    """
    date_note = f"\nOnly include media.articles published on or before {until}." if until else ""
    prompt = (
        # ── STATIC PREFIX — keep everything below byte-identical ──────────────
        # DeepSeek caches identical leading text automatically and bills it at
        # $0.007/1M instead of $0.22/1M (31x cheaper). The dynamic domain/mode
        # line used to sit here, at roughly char 60, which broke the prefix and
        # made all ~2000 tokens of rules a cache MISS on every chunk of every
        # company. It now lives in the dynamic tail just above "Content:".
        # NEVER interpolate a per-company value above the rules block.
        f"Extract structured company metadata from the web content below.\n\n"
        "Rules:\n"
        "- Extract ONLY what is explicitly stated — never invent or guess.\n"
        "- The CONTENT below may be sourced from multiple pages (the focal company's\n"
        "  own site, Wikipedia, news articles, third-party profiles, etc.).\n"
        "  Every field you extract must describe the FOCAL COMPANY (the focal domain\n"
        "  given below, just above the content),\n"
        "  NEVER the source page or its publisher.\n"
        "- business.products / business.services — STRICT:\n"
        "    * Only list products/services that the focal company itself manufactures,\n"
        "      sells, builds, or operates as its own offering.\n"
        "    * NEVER list the source platform (Wikipedia, LinkedIn, Crunchbase, Bloomberg,\n"
        "      Reuters, ZoomInfo, Owler, PitchBook, news outlets, encyclopedias, directories)\n"
        "      as a product — they are sources of information, not products of the focal company.\n"
        "    * If the chunk only describes the focal company in third-person prose without\n"
        "      naming concrete offerings, return [] — do NOT invent products from source metadata.\n"
        "    * Categories like 'Online Encyclopedia', 'News Aggregator', 'Business Directory',\n"
        "      'Reference Site' are red flags that you have extracted the source, not a product.\n"
        "- media.articles: leave as [] — articles are filled separately via trafilatura.\n"
        "- Include ALL products/services with full detail (name, category, features, pricing).\n"
        "- List ALL named executives, directors, board members, advisors (include Late/former).\n"
        "- financials.key_numbers: capture EVERY exact figure (revenue, margins, headcount, CAGR).\n"
        "- financials.employee_count + scale signals: aggressively extract company-size cues from\n"
        "  hero stats, trust strips, counter widgets, 'about us' badges, and footers. Patterns:\n"
        "    * 'Trusted by 500+ enterprises', '10,000+ customers', '1M+ users' → mention in key_numbers\n"
        "      AND, if it refers to client/customer count, treat it as scale evidence.\n"
        "    * '100+ employees', 'team of 250', 'over 1,000 professionals' → employee_count (verbatim).\n"
        "    * '25 years of experience', 'since 1995', 'established 1980' → company.founded_year + key_numbers.\n"
        "    * 'Offices in Mumbai, Delhi, Bangalore', 'present in 30 countries' → location.offices /\n"
        "      location.countries (list every named city / country verbatim).\n"
        "    * 'Pan-India presence', 'global reach across APAC' → location.regions_served.\n"
        "  Copy numbers/phrases VERBATIM — '500+', '10,000+ customers', 'over 1,000'. Do NOT invent.\n"
        "- media.press_releases / investor_presentations: write a detailed multi-sentence summary;\n"
        "  populate highlights with ALL key findings (financial tables, milestones, decisions).\n"
        "  ONLY add entries whose URL appears verbatim in this content chunk — no hallucination.\n"
        "- contact: extract ALL contact details — email address, phone number, physical address,\n"
        "  AND social media URLs (LinkedIn, Twitter, Facebook, Instagram, YouTube).\n"
        "  Look in footers, 'Contact Us' sections, page headers, and schema.org markup.\n"
        "  Copy email/phone/address VERBATIM — do not reformat or abbreviate.\n"
        "- relationships.customers: ALL named customers / client logos / brands this company works with.\n"
        "  This is critical — every brand name in a 'Our clients' / 'Trusted by' / logo strip / case study\n"
        "  / testimonial belongs here. Examples to look for: Unilever, HSBC, Amazon, Nestle, Walmart, etc.\n"
        "- relationships.investors: all named investors.\n"
        "- business.industries — CRITICAL DISAMBIGUATION:\n"
        "  Populate with the CUSTOMER verticals this company SELLS TO or SERVES, NOT the industry the\n"
        "  company itself operates in. For vendors, agencies, SaaS, consultancies, service businesses the\n"
        "  target industries are the CLIENT verticals, not the company's own sector.\n"
        "  Examples:\n"
        "    * A digital-ad agency: industries = ['Retail E-commerce', 'Financial Services', 'CPG',\n"
        "      'Gaming'] — NOT ['Digital Advertising', 'Marketing', 'Media'].\n"
        "    * A B2B SaaS for warehouses: industries = ['Logistics', 'Manufacturing', 'Retail'] —\n"
        "      NOT ['Software', 'SaaS'].\n"
        "    * A market-research firm: industries = its clients' sectors — NOT ['Research', 'Consulting'].\n"
        "  Derivation rules (in order):\n"
        "    1. If the page has an explicit 'Industries we serve' / 'Industries' / 'Sectors' / 'Verticals'\n"
        "       section, copy the labels verbatim.\n"
        "    2. Otherwise, roll up relationships.customers: each unique customer's industry becomes an\n"
        "       entry in business.industries (e.g. Amazon → Retail E-commerce, HSBC → Financial Services,\n"
        "       Nestle → CPG).\n"
        "    3. If the company is a consumer brand / D2C / publisher with no B2B clients, use the\n"
        "       consumer channel it sells through ('Retail E-commerce', 'Grocery Retail'), NOT its product\n"
        "       category.\n"
        "  3-6 entries max. If there's genuinely no signal, leave empty.\n"
        "- business.key_markets: geographic markets ('India', 'GCC', 'APAC', 'North America'), NOT\n"
        "  industries. Keep this separate from business.industries.\n"
        "- intel: synthesis and actionable insights only.\n\n"
        # The JSON schema is constant for a given mode, so it belongs in the
        # cached prefix. _run_extraction used to append it AFTER the content on
        # the DeepSeek path, which re-billed the whole schema at full cache-miss
        # rate on every chunk of every company. Stating it here also satisfies
        # DeepSeek's json_object requirement that the prompt contain the word
        # "json" and show the expected shape.
        "Return ONLY JSON matching this schema:\n"
        f"{json.dumps(schema, separators=(',', ':'))}\n\n"
        # ── DYNAMIC TAIL — everything from here varies per chunk/company. ──────
        # Must stay BELOW the rules so the static prefix above stays cacheable.
        f"Focal company domain: {domain}  Focus: {mode_label}{date_note}\n\n"
        f"Content:\n{chunk}"
    )
    global _LLM_SEM
    # Re-create the semaphore if it was created in a different event loop.
    # _smart_crawl_isolated runs in a thread with its own ProactorEventLoop;
    # if _LLM_SEM was created in the main (uvicorn) loop it's the wrong object
    # here and async with will deadlock. Checking the loop id handles this.
    try:
        cur_loop = asyncio.get_event_loop()
        if _LLM_SEM is None or getattr(_LLM_SEM, "_loop", cur_loop) is not cur_loop:
            _LLM_SEM = asyncio.Semaphore(3)
    except RuntimeError:
        _LLM_SEM = asyncio.Semaphore(3)
    async with _LLM_SEM:
        try:
            return await _run_extraction(domain, prompt, schema, max_tokens)
        except Exception as e:
            # DeepSeek is down: the crawl fetched pages but extracts nothing, so
            # every downstream field (products, industries, contact, size) comes
            # back empty. That is data loss, not a degradation — log it as such.
            log.error("  [LLM extraction FAILED] %s: %s", LLM_MODEL, str(e)[:160])
            return {}, 0, 0, 0.0


async def final_extract(domain: str, pages: list[dict],
                        until: str | None = None, mode: str = "full",
                        ) -> tuple[dict, dict]:
    """Returns (merged_intel, llm_stats). llm_stats has chunk/token/cost totals."""
    CHUNK      = 50_000
    mode_cfg   = MODES.get(mode, MODES["full"])
    schema     = _mode_schema(mode_cfg["fields"])
    mode_label = mode_cfg["label"]

    HTML_CAP = {"full": 8_000,  "news": 5_000, "investor": 8_000,  "company": 12_000, "leads": 5_000}
    DOC_CAP  = {"full": 15_000, "news": 8_000, "investor": 20_000, "company": 12_000, "leads": 8_000}
    html_cap = HTML_CAP.get(mode, 8_000)
    doc_cap  = DOC_CAP.get(mode, 15_000)

    # ── Cross-page boilerplate dedup ──────────────────────────────────────────
    # Nav bars, footers and cookie notices repeat verbatim on every page, so a
    # 40-page crawl can pay for the same menu 40 times. Strip repeats, but keep
    # the FIRST occurrence: footers carry the registered address and phone that
    # the `contact` field depends on, so deleting them outright loses real
    # signal. Only engages at >=4 pages, where "appears on most pages" is
    # actually evidence of boilerplate rather than a small-site coincidence.
    _seen_bp: set[str] = set()
    _bp_lines: set[str] = set()
    if len(pages) >= 4:
        _freq: dict[str, int] = {}
        for _p in pages:
            for _ln in {l.strip() for l in _p.get("text", "").splitlines() if l.strip()}:
                _freq[_ln] = _freq.get(_ln, 0) + 1
        _threshold = max(2, int(len(pages) * 0.6))
        _bp_lines = {l for l, n in _freq.items() if n >= _threshold}

    _bp_chars_saved = 0

    def _dedup_boilerplate(text: str) -> str:
        """Drop boilerplate lines already emitted by an earlier page."""
        nonlocal _bp_chars_saved
        if not _bp_lines:
            return text
        out: list[str] = []
        for ln in text.splitlines():
            s = ln.strip()
            if s and s in _bp_lines:
                if s in _seen_bp:
                    _bp_chars_saved += len(ln) + 1
                    continue
                _seen_bp.add(s)
            out.append(ln)
        return "\n".join(out)

    chunks: list[str] = []
    current = ""
    _seen_pages: set[str] = set()
    _pages_skipped = 0
    for p in pages:
        cap  = doc_cap if p.get("is_doc") else html_cap
        # Boilerplate is removed BEFORE the cap so the cap spends its budget on
        # unique content instead of a menu that repeats on every page.
        text = _dedup_boilerplate(p.get("text", ""))[:cap]
        # Skip pages that carry no unique text once boilerplate is gone — a
        # duplicate URL (trailing slash, tracking param) contributes nothing but
        # still costs a full chunk's worth of tokens.
        _sig = hashlib.sha256(text.strip().encode("utf-8", errors="ignore")).hexdigest()
        if not text.strip() or _sig in _seen_pages:
            _pages_skipped += 1
            continue
        _seen_pages.add(_sig)
        header = f"=== PAGE: {p['url']} ===\n"
        body   = text + "\n\n"
        if len(header) + len(body) > CHUNK:
            if current:
                chunks.append(current); current = ""
            pos = 0
            while pos < len(body):
                chunks.append(header + body[pos:pos + CHUNK])
                pos += CHUNK
        elif current and len(current) + len(header) + len(body) > CHUNK:
            chunks.append(current)
            current = header + body
        else:
            current += header + body
    if current:
        chunks.append(current)

    # SHA-256 chunk dedup — a last-resort net for wholly identical chunks.
    # Hash the FULL chunk, not a 2000-char prefix: every chunk opens with
    # "=== PAGE: <unique-url> ===", so a prefix hash never collided and this
    # pass silently never fired. The real boilerplate saving now happens
    # per-page above (_dedup_boilerplate); this only catches exact repeats.
    seen_hashes: set[str] = set()
    deduped_chunks: list[str] = []
    skipped_dup = 0
    for c in chunks:
        h = hashlib.sha256(c.encode("utf-8", errors="ignore")).hexdigest()
        if h in seen_hashes:
            skipped_dup += 1
            continue
        seen_hashes.add(h)
        deduped_chunks.append(c)

    log.debug("  -> %d LLM chunks (%s chars total; %d dup-skipped)", len(deduped_chunks), f"{sum(len(c) for c in deduped_chunks):,}", skipped_dup)
    results = await asyncio.gather(
        *[_extract_chunk(domain, c, until, schema, mode_label, mode_cfg["max_tokens"])
          for c in deduped_chunks]
    )
    merged: dict = {}
    tot_in = tot_out = 0
    tot_cost = 0.0
    for intel, tin, tout, cost in results:
        merged = _merge_intel(merged, intel)
        tot_in += tin
        tot_out += tout
        tot_cost += cost
    _strip_source_artifacts(merged)
    llm_stats = {
        "chunks_total":   len(chunks),
        "chunks_deduped": skipped_dup,
        "chunks_sent":    len(deduped_chunks),
        # Savings from the pre-chunk pass, so the optimisation is measurable
        # rather than assumed. bp_chars_saved is raw chars of repeated
        # nav/footer text that never reached the LLM (~4 chars ≈ 1 token).
        "pages_skipped":    _pages_skipped,
        "bp_chars_saved":   _bp_chars_saved,
        "tokens_in":      tot_in,
        "tokens_out":     tot_out,
        "cost_inr":       round(tot_cost, 4),
    }
    return merged, llm_stats

# ── Main orchestrator ─────────────────────────────────────────────────────────
async def smart_crawl(
    domain: str,
    until_date: str | None = None,
    max_pages: int = 0,
    max_news_articles: int = 200,
    mode: str = "full",
) -> dict[str, Any]:
    t0 = time.perf_counter()
    domain   = domain.strip().removeprefix("http://").removeprefix("https://").rstrip("/")
    bh       = domain.removeprefix("www.")
    base_url = f"https://{domain}"

    mode_cfg   = MODES.get(mode, MODES["full"])
    mode_paths = mode_cfg["paths"]
    if max_pages == 0:
        max_pages = mode_cfg["max_pages"]

    log.info("[%s] Mode: %s", domain, mode)

    # ── Phase 1: Homepage + parallel discovery (sitemap + feed + WP REST) ──
    t_phase = time.perf_counter()
    discovery_stats: dict[str, Any] = {}
    wp_stats: dict[str, Any] = {}

    sitemap_task = asyncio.create_task(_sitemap_links(base_url, stats=discovery_stats))  # rls-allow: pure HTTP fetcher; no DB
    feed_task    = asyncio.create_task(_feed_links(base_url))  # rls-allow: pure HTTP fetcher; no DB
    wp_task      = asyncio.create_task(_wp_rest_links(base_url, stats=wp_stats))  # rls-allow: pure HTTP fetcher; no DB

    log.info("[%s] Phase 1: Homepage + parallel discovery (sitemap/feed/wp_rest)", domain)
    t_hp = time.perf_counter()
    try:
        hp_text, nav_links, hp_html, nav_escalated = await fetch_homepage(base_url)
        homepage_ms = int((time.perf_counter() - t_hp) * 1000)

        # Bound the three discovery probes on wall clock. Measured live on
        # avigloplast.com, Phase 1 was 35s of an 80s crawl — 44% of the total,
        # and the only phase with no budget: _sitemap_links alone probes
        # robots + 8 standard paths + up to 50 sub-sitemaps at _TIMEOUT_SITEMAP
        # each, and feed/wp_rest burn their full timeouts to return 0 results on
        # sites that have neither.
        #
        # Whatever finished is kept; stragglers are cancelled. This is safe to
        # cancel — these three only use httpx via safe_fetch. fetch_homepage is
        # deliberately NOT bounded here: its cascade holds _BROWSER_SEM, which is
        # thread-backed and cancel-unsafe (cancelling it leaks a permit and
        # deadlocks browser fetches process-wide).
        #
        # The tasks were created before the homepage fetch, so they have already
        # had that time to run — charge it against their budget rather than
        # granting a fresh full window on top.
        _disc_left = max(0.5, _DISCOVERY_BUDGET_S - (time.perf_counter() - t_phase))
        _disc_done, _disc_pending = await asyncio.wait(
            {sitemap_task, feed_task, wp_task}, timeout=_disc_left
        )
        for _t in _disc_pending:
            _t.cancel()
        if _disc_pending:
            log.info("[%s] discovery budget %.0fs hit — %d/3 probes finished",
                     domain, _DISCOVERY_BUDGET_S, len(_disc_done))

        def _disc_result(t: "asyncio.Task") -> list:
            if t not in _disc_done or t.cancelled():
                return []
            try:
                r = t.result()
            except Exception:
                # A discovery probe raising is recoverable (the crawl proceeds on
                # the remaining seed sources) but not nothing: it means sitemap /
                # feed / wp_rest contributed zero URLs for a reason worth seeing.
                log.warning("[%s] discovery probe failed — continuing without it",
                            domain, exc_info=True)
                return []
            return r if isinstance(r, list) else []

        sitemap_links = _disc_result(sitemap_task)
        feed_links    = _disc_result(feed_task)
        wp_links      = _disc_result(wp_task)
    finally:
        # Cancel any tasks that didn't complete (e.g. if fetch_homepage raised
        # or smart_crawl was cancelled mid-fetch). cancel() on done tasks is a no-op.
        for _disc_task in (sitemap_task, feed_task, wp_task):
            if not _disc_task.done():
                _disc_task.cancel()
    if isinstance(sitemap_links, BaseException): sitemap_links = []
    if isinstance(feed_links,    BaseException): feed_links    = []
    if isinstance(wp_links,      BaseException): wp_links      = []
    discovery_ms = int((time.perf_counter() - t_phase) * 1000)

    # Record discovery counts
    discovery_stats["nav_urls"] = len(nav_links)
    discovery_stats["nav_urls_js_escalated"] = nav_escalated
    discovery_stats["feed_urls"] = len(feed_links)
    discovery_stats["jsonld_urls"] = 0  # filled incrementally in BFS
    discovery_stats["discovery_elapsed_ms"] = discovery_ms
    discovery_stats.setdefault("wp_rest_detected", wp_stats.get("wp_rest_detected", False))
    discovery_stats.setdefault("wp_rest_posts", wp_stats.get("wp_rest_posts", 0))
    discovery_stats.setdefault("wp_rest_pages", wp_stats.get("wp_rest_pages", 0))

    # Seed union with per-URL source tracking
    seed_with_source: list[tuple[str, dict]] = []
    for l in nav_links:      seed_with_source.append(("nav",     l))
    for l in sitemap_links:  seed_with_source.append(("sitemap", l))
    for l in feed_links:     seed_with_source.append(("feed",    l))
    for l in wp_links:       seed_with_source.append(("wp_rest", l))

    seen_seeds: set[str] = set()
    seed: list[tuple[str, dict]] = []
    for src, l in seed_with_source:
        u = _norm(l["url"])
        if u in seen_seeds:
            continue
        seen_seeds.add(u)
        seed.append((src, l))

    # ── OOM guard (post-2026-05-29 mass-orphan incident) ──────────────────
    # A large sitemap (e.g. 553+ URLs) produced a seed list big enough that the
    # priority-queue scoring + per-URL state pinned RAM and OOM-killed the whole
    # uvicorn host mid-autochain — leaving 122 companies frozen at 'researching'.
    # We only ever crawl `max_pages` of these, so cap the seed at a generous
    # multiple of max_pages (nav links first — highest signal — already lead the
    # list). This bounds peak memory without losing crawl quality, since the BFS
    # would never reach the tail anyway.
    _SEED_CAP = max(60, max_pages * 8)
    if len(seed) > _SEED_CAP:
        log.info("[%s] seed capped %d -> %d (OOM guard)", domain, len(seed), _SEED_CAP)
        seed = seed[:_SEED_CAP]
    discovery_stats["total_seed"] = len(seed)

    log.info("[%s] nav=%d sitemap=%d feed=%d wp_rest=%d -> seed=%d", domain, len(nav_links), len(sitemap_links), len(feed_links), len(wp_links), len(seed))
    if not seed and not hp_text:
        return {
            "domain": domain, "mode": mode, "error": "homepage_fetch_failed",
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "discovery": discovery_stats, "data": {},
        }

    # ── Phase 2 setup: priority queue + observability state ──────────────
    news_sp = set(_NEWS_PATHS)
    # _norm(base_url) must be seeded too: every downstream membership test keys off
    # _norm, which drops `www.`, so the raw spellings alone never match on a www
    # domain and the homepage gets refetched and re-extracted as a second page.
    seen: set[str] = {_norm(base_url), base_url, base_url.rstrip("/"), base_url + "/"}
    raw_pages: list[dict] = [{
        "url": base_url, "text": hp_text, "is_doc": False, "html": hp_html,
    }]
    source_of: dict[str, str] = {_norm(base_url): "homepage"}
    crawl_trace: list[dict] = []

    def _in_mode(url: str) -> bool:
        is_doc = bool(_DOC_RE.search(url))
        if is_doc:
            return mode_cfg.get("include_docs", True)
        if not mode_paths or not mode_cfg.get("strict_paths", True):
            return True
        p = urlparse(url).path.lower()
        return any(p == mp or p.startswith(mp + "/") or p.startswith(mp + "-") for mp in mode_paths)

    def _score(url: str) -> int:
        """heapq priority: 0 = high-signal path, 1 = shallow, 2 = deep, 3 = archive.

        Priority 0 covers BOTH mode-specific paths (e.g. /investor for investor
        mode) AND universal valuable paths (/about, /team, /pricing, etc.) that
        are always high-signal regardless of mode. Without this, company-mode
        crawls of product-heavy nav (Vercel-style) skip the about/team pages.

        Priority 3 is the CMS index surface — /category/, /tag/, /author/ and
        date archives. These carry almost no prose of their own (they are lists of
        titles) but they do link onward to real posts, so they are demoted rather
        than skipped: they get crawled only once every real page is exhausted,
        which is exactly the right trade in news mode and a no-op in company mode.
        """
        p = urlparse(url).path.lower()
        if mode_paths and any(p == mp or p.startswith(mp + "/") for mp in mode_paths):
            return 0
        if any(p == vp or p.startswith(vp + "/") for vp in _UNIVERSAL_VALUABLE_PATHS):
            return 0
        if _ARCHIVE_PATH_RE.search(p):
            return 3
        if p.count("/") <= 2:
            return 1
        return 2

    # heapq of (priority, monotonic_counter, url) — counter breaks ties for stability
    pq: list[tuple[int, int, str]] = []
    _counter = 0
    for src, l in seed:
        url = l["url"]
        nm = _norm(url)
        if nm in seen or not _in_mode(url):
            continue
        seen.add(nm)
        source_of[nm] = src
        heapq.heappush(pq, (_score(url), _counter, url))
        _counter += 1

    log.info("[%s] Phase 2: BFS crawl (mode=%s, budget=%d pages, pq=%d)", domain, mode, max_pages, len(pq))

    # ── Phase 2: priority BFS with full per-URL tracing ───────────────────
    t_phase = time.perf_counter()
    # fetch_sem: per-crawl HTTP concurrency. Was 24 but with deep_research_
    # concurrency=2 that meant 48 simultaneous requests from the same EC2 IP
    # — a strong bot-detection signal. 8 per company × 2 = 16 total, much safer.
    fetch_sem        = asyncio.Semaphore(8)
    browser_js_calls = 0
    BROWSER_JS_CAP   = 15
    jsonld_total     = 0

    # Aggregate counters
    cs_fetched = cs_attempted = 0
    cs_pw_fb = cs_cf = cs_skip_re = cs_too_small = cs_dup = cs_docs = 0
    cs_bytes = 0
    cs_fetch_ms_list: list[int] = []
    cs_by_priority: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    cs_by_source: dict[str, int] = {}
    cs_docs_by_type: dict[str, int] = {}

    async def _fetch_one(url: str, priority: int, source: str) -> dict | None:
        nonlocal browser_js_calls, jsonld_total
        nonlocal cs_fetched, cs_attempted, cs_pw_fb, cs_cf, cs_skip_re
        nonlocal cs_too_small, cs_dup, cs_docs, cs_bytes
        trace_row: dict[str, Any] = {
            "url": url, "source": source, "priority": priority,
            "status_code": None, "fetch_method": "skipped",
            "bytes": 0, "fetch_ms": 0, "cf_blocked": False,
            "skip_reason": None, "text_chars": 0, "used_for_llm": False,
        }
        if _SKIP_RE.search(url):
            trace_row["skip_reason"] = "skip_re"
            cs_skip_re += 1
            crawl_trace.append(trace_row)
            return None
        cs_attempted += 1
        cs_by_priority[priority] = cs_by_priority.get(priority, 0) + 1
        cs_by_source[source] = cs_by_source.get(source, 0) + 1

        async with fetch_sem:
            t_f = time.perf_counter()
            if _DOC_RE.search(url):
                data, ext = await _fetch_document(url)
                trace_row["fetch_method"] = "doc_fetch"
                trace_row["fetch_ms"] = int((time.perf_counter() - t_f) * 1000)
                trace_row["bytes"] = len(data) if data else 0
                cs_fetch_ms_list.append(trace_row["fetch_ms"])
                if not data:
                    trace_row["skip_reason"] = "doc_fetch_failed"
                    crawl_trace.append(trace_row)
                    return None
                text = _parse_doc(data, ext)
                if not text:
                    trace_row["skip_reason"] = "doc_parse_failed"
                    crawl_trace.append(trace_row)
                    return None
                trace_row["status_code"] = 200
                trace_row["text_chars"] = len(text)
                cs_docs += 1
                cs_docs_by_type[ext] = cs_docs_by_type.get(ext, 0) + 1
                cs_bytes += len(data)
                cs_fetched += 1
                crawl_trace.append(trace_row)
                return {"url": url, "text": text, "is_doc": True, "child_links": [], "_trace": trace_row}

            # HTML fetch — adaptive timeout
            html = await scrapling_get(url, timeout=_TIMEOUT_FAST)
            trace_row["fetch_method"] = "scrapling"
            if (not html or len(html) < 2000) and not CF_RE.search((html or "")[:500]):
                # scrapling returned NOTHING → the live IP is blocked at the edge.
                # The same-IP browser would also 403 (and Camoufox crashes/leaks
                # on JS-error sites), so go straight to the archive and skip it.
                # A sparse-but-present body is instead a real JS shell → let the
                # browser render it for fresh data, with Wayback as the fallback.
                if not html:
                    wb_html, _wb_age = await wayback_get(url)
                    if wb_html and len(wb_html) >= 1000 and not _ANTIBOT_RE.search(wb_html[:4000]):
                        html = wb_html
                        trace_row["fetch_method"] = "wayback"
                if (trace_row["fetch_method"] != "wayback"
                        and (not html or len(html) < 2000)
                        and browser_js_calls < BROWSER_JS_CAP):
                    browser_js_calls += 1
                    try:
                        js_html = await asyncio.wait_for(
                            js_browser_get(url), timeout=_TIMEOUT_BROWSER
                        )
                    except (asyncio.TimeoutError, Exception):
                        js_html = ""
                    if js_html and len(js_html) > len(html or ""):
                        html = js_html
                        trace_row["fetch_method"] = "lightpanda_camoufox"
                        cs_pw_fb += 1

            # Final per-page archive fallback: a JS shell the browser couldn't
            # render, or a challenge page slipped through — recover the snapshot.
            _pg_blocked = bool(html and (CF_RE.search(html[:2000]) or _ANTIBOT_RE.search(html[:4000])))
            if trace_row["fetch_method"] != "wayback" and (_pg_blocked or not html or len(html) < 300):
                wb_html, _wb_age = await wayback_get(url)
                if wb_html and len(wb_html) >= 1000 and not _ANTIBOT_RE.search(wb_html[:4000]):
                    html = wb_html
                    trace_row["fetch_method"] = "wayback"

            trace_row["fetch_ms"] = int((time.perf_counter() - t_f) * 1000)
            cs_fetch_ms_list.append(trace_row["fetch_ms"])

            if html and (CF_RE.search(html[:2000]) or _ANTIBOT_RE.search(html[:4000])):
                trace_row["cf_blocked"] = True
                trace_row["skip_reason"] = "cf_blocked"
                cs_cf += 1
                crawl_trace.append(trace_row)
                return None
            if not html or len(html) < 300:
                trace_row["skip_reason"] = "too_small"
                cs_too_small += 1
                crawl_trace.append(trace_row)
                return None

            trace_row["status_code"] = 200
            trace_row["bytes"] = len(html)
            cs_bytes += len(html)

            text = _txt(html)
            trace_row["text_chars"] = len(text)
            cs_fetched += 1

            # Child links: <a href>, canonical, pagination, data-* + JSON-LD URLs
            dom_children = _extract_links(html, url)
            jsonld_here = _jsonld_urls(html, bh)
            if jsonld_here:
                jsonld_total += len(jsonld_here)
                known_urls = {lk["url"] for lk in dom_children}
                for ju in jsonld_here:
                    if ju not in known_urls:
                        dom_children.append({"url": ju, "text": "jsonld"})

            child_links = [
                lk["url"] for lk in dom_children
                if urlparse(lk["url"]).netloc.removeprefix("www.") == bh
                and not _SKIP_RE.search(lk["url"])
                and _norm(lk["url"]) not in seen
                and _in_mode(lk["url"])
            ]
            crawl_trace.append(trace_row)
            return {
                "url": url, "text": text, "is_doc": False,
                "child_links": child_links, "html": html, "_trace": trace_row,
            }

    # Priority-BFS: pop highest-priority URLs in parallel batches.
    #
    # History: this was 24, cut to 6 on 2026-06-02 because 24 concurrent
    # _fetch_one calls each spawned Chromium (~300MB) → ~9GB peak → OOM kills.
    # That justification is now obsolete: playwright/camoufox/Chromium were
    # removed in 2026-06 and the JS tier is Lightpanda (~40MB), which is ALSO
    # capped independently at CRAWL_BROWSER_CONCURRENCY (default 2) by
    # _BROWSER_SEM. So BATCH no longer governs browser memory at all — the
    # semaphore does. What BATCH actually governs is how many cheap curl_cffi /
    # httpx fetches run concurrently, which cost kilobytes, not megabytes.
    #
    # Holding it at 6 was therefore costing wall-clock for no memory benefit.
    #
    # Default is 8, NOT the old 24 and not 12: the prod host runs ~1.4GB free
    # (3.7GB total, backend container already ~1GB of its 3GB limit), and each
    # in-flight fetch still holds its response body in memory even on the cheap
    # tier. 8 is a deliberate, conservative step up from 6 — measure headroom
    # with `docker stats` during a crawl before raising it further.
    # Raise via CRAWL_BFS_BATCH; no redeploy needed.
    BATCH = _envi("CRAWL_BFS_BATCH", 8)
    # Wall-clock budget for the fetch phase. Nothing bounded this before: 40
    # pages x (8s fast tier + 20s browser tier) x 2 attempts, plus wayback and
    # sitemap fallbacks, has no ceiling — which is exactly why an onboarding
    # scan can appear to "load forever".
    #
    # On expiry we stop QUEUEING more pages and fall through to extraction with
    # whatever was already fetched, rather than cancelling and losing the lot.
    # Partial intel beats an infinite spinner, and the pages are priority-
    # ordered (_score), so the ones we already have are the ones that matter.
    # 0 disables the budget entirely.
    _bfs_budget   = _envf("CRAWL_BFS_BUDGET_S", 25.0)
    _bfs_deadline = (t_phase + _bfs_budget) if _bfs_budget > 0 else float("inf")
    _bfs_truncated = False
    while pq and len(raw_pages) < max_pages:
        if time.perf_counter() > _bfs_deadline:
            _bfs_truncated = True
            log.info("  [bfs-budget] %.0fs budget reached — proceeding with %d pages "
                     "(%d left queued)", _bfs_budget, len(raw_pages), len(pq))
            break
        remaining = max_pages - len(raw_pages)
        batch_size = min(BATCH, remaining)
        batch: list[tuple[int, int, str]] = []
        while pq and len(batch) < batch_size:
            batch.append(heapq.heappop(pq))
        results = await asyncio.gather(*[
            _fetch_one(url, prio, source_of.get(_norm(url), "child_link"))
            for prio, _, url in batch
        ])
        for (prio, _, url), r in zip(batch, results):
            if not r:
                continue
            url_path   = urlparse(url).path.lower()
            keep_html  = mode in ("news", "investor") or any(url_path.startswith(sp) for sp in news_sp)
            raw_pages.append({
                "url": url, "text": r["text"], "is_doc": r["is_doc"],
                "html": r.get("html", "") if keep_html else "",
                "_trace": r.get("_trace"),
            })
            for cl in r.get("child_links", []):
                nm = _norm(cl)
                if nm in seen:
                    continue
                seen.add(nm)
                source_of[nm] = "child_link" if "jsonld" not in source_of.get(nm, "") else "jsonld"
                heapq.heappush(pq, (_score(cl), _counter, cl))
                _counter += 1
        # Early-termination for news: bail if remaining pq has zero mode-path hits
        if mode == "news" and len(raw_pages) >= max(20, max_pages // 2):
            if not any(p == 0 for p, _, _ in pq):
                log.debug("  [early-exit] news mode — no more mode-path URLs in queue")
                break
        log.debug("  fetched %d/%d pages, %d queued...", len(raw_pages), max_pages, len(pq))

    bfs_ms = int((time.perf_counter() - t_phase) * 1000)
    discovery_stats["jsonld_urls"] = jsonld_total
    # Surfaced so a short crawl is distinguishable from a budget-truncated one.
    discovery_stats["bfs_budget_truncated"] = _bfs_truncated
    discovery_stats["bfs_queued_unfetched"] = len(pq)

    total      = len(raw_pages)
    blob_chars = sum(len(p["text"]) for p in raw_pages)

    crawl_stats = {
        "pages_fetched": cs_fetched,
        "pages_attempted": cs_attempted,
        "fetch_success_rate": round(cs_fetched / cs_attempted, 3) if cs_attempted else 0.0,
        "browser_js_fallbacks": cs_pw_fb,
        "browser_js_budget_remaining": BROWSER_JS_CAP - browser_js_calls,
        "cf_blocked": cs_cf,
        "skipped_skip_re": cs_skip_re,
        "skipped_too_small": cs_too_small,
        "skipped_duplicate": cs_dup,
        "docs_fetched": cs_docs,
        "docs_by_type": cs_docs_by_type,
        "by_priority": cs_by_priority,
        "by_source": cs_by_source,
        "avg_fetch_ms": int(sum(cs_fetch_ms_list) / len(cs_fetch_ms_list)) if cs_fetch_ms_list else 0,
        "p95_fetch_ms": int(sorted(cs_fetch_ms_list)[int(len(cs_fetch_ms_list) * 0.95)]) if cs_fetch_ms_list else 0,
        "total_bytes": cs_bytes,
    }

    # ── Phase 3: Article extraction ───────────────────────────────────────
    t_phase = time.perf_counter()
    if mode in ("news", "investor"):
        news_raw = [{"url": p["url"], "html": p.get("html") or p["text"], "category": "news"}
                    for p in raw_pages if not p["is_doc"]]
    else:
        news_raw = [{"url": p["url"], "html": p.get("html") or p["text"], "category": "news"}
                    for p in raw_pages
                    if any(urlparse(p["url"]).path.lower().rstrip("/").startswith(sp) for sp in news_sp)]

    log.info("[%s] %d pages, %s chars (%d for article extraction)", domain, total, f"{blob_chars:,}", len(news_raw))
    log.info("[%s] Phase 3: Article extraction", domain)
    articles = extract_articles(news_raw)
    if until_date and articles:
        articles = [a for a in articles if not a["date"] or a["date"] >= until_date]
    log.debug("  -> %d articles", len(articles))
    article_ms = int((time.perf_counter() - t_phase) * 1000)

    # ── Phase 4: LLM extraction ───────────────────────────────────────────
    t_phase = time.perf_counter()
    llm_pages = [p for p in raw_pages if p["text"] and len(p["text"]) >= 400]  # raised 100 -> 400 (Fix 8c)
    # Mark which pages made it to LLM in trace
    llm_urls = {p["url"] for p in llm_pages}
    for row in crawl_trace:
        if row["url"] in llm_urls:
            row["used_for_llm"] = True
    log.info("[%s] Phase 4: LLM metadata (%d pages, mode=%s)", domain, len(llm_pages), mode)
    data, llm_stats = await final_extract(domain, llm_pages, until_date, mode)
    data.setdefault("media", {})["articles"] = articles
    llm_ms = int((time.perf_counter() - t_phase) * 1000)

    # De-hallucinate press_releases / investor_presentations
    fetched_norms = {_norm(p["url"]) for p in raw_pages}
    for section in ("press_releases", "investor_presentations"):
        entries = data.get("media", {}).get(section, [])
        if not entries:
            continue
        valid = [e for e in entries if not e.get("url") or _norm(e["url"]) in fetched_norms]
        removed = len(entries) - len(valid)
        if removed:
            log.debug("  [dehallu] %s: removed %d unverified URL entries", section, removed)
        data["media"][section] = valid

    elapsed = round(time.perf_counter() - t0, 1)
    log.info("[%s] Done in %ss", domain, elapsed)

    # ── Extraction / quality / diagnostic aggregates ──────────────────────
    ex_articles = articles
    extraction_stats = {
        "articles_found":        len(ex_articles),
        "articles_with_date":    sum(1 for a in ex_articles if a.get("date")),
        "articles_with_author":  sum(1 for a in ex_articles if a.get("author")),
        "llm_pages":             len(llm_pages),
        "llm_chunks_total":      llm_stats["chunks_total"],
        "llm_chunks_deduped":    llm_stats["chunks_deduped"],
        "llm_chunks_sent":       llm_stats["chunks_sent"],
        "llm_tokens_in":         llm_stats["tokens_in"],
        "llm_tokens_out":        llm_stats["tokens_out"],
        "llm_cost_inr":          llm_stats["cost_inr"],
        "extraction_elapsed_ms": article_ms + llm_ms,
    }

    # Quality: completeness over INTEL_SCHEMA top-level fields + entity counts
    def _non_empty(v: Any) -> bool:
        if v is None or v == "" or v == [] or v == {}:
            return False
        return True
    top_fields = list(INTEL_SCHEMA["properties"].keys())
    populated = 0
    fields_total = 0
    empty_sections: list[str] = []
    for f in top_fields:
        val = data.get(f)
        if isinstance(val, dict):
            if any(_non_empty(x) for x in val.values()):
                populated += sum(1 for x in val.values() if _non_empty(x))
            else:
                empty_sections.append(f)
            fields_total += len(val) if val else len(INTEL_SCHEMA["properties"][f].get("properties", {}))
        else:
            fields_total += 1
            if _non_empty(val):
                populated += 1
    quality = {
        "fields_populated": populated,
        "fields_total": fields_total,
        "completeness": round(populated / fields_total, 3) if fields_total else 0.0,
        "entity_counts": {
            "products":       len((data.get("business") or {}).get("products", []) or []),
            "services":       len((data.get("business") or {}).get("services", []) or []),
            "leadership":     len((data.get("people")   or {}).get("leadership", []) or []),
            "board":          len((data.get("people")   or {}).get("board", []) or []),
            "articles":       len((data.get("media")    or {}).get("articles", []) or []),
            "press_releases": len((data.get("media")    or {}).get("press_releases", []) or []),
            "partnerships":   len((data.get("relationships") or {}).get("partnerships", []) or []),
            "customers":      len((data.get("relationships") or {}).get("customers", []) or []),
        },
        "empty_sections": empty_sections,
    }

    diagnostics = {
        "had_robots_txt": discovery_stats.get("robots_txt", {}).get("fetched", False),
        "had_sitemap":    len(sitemap_links) > 0,
        "had_feed":       len(feed_links) > 0,
        "is_wordpress":   discovery_stats.get("wp_rest_detected", False),
        "uses_cloudflare": cs_cf > 0,
        "is_spa":         nav_escalated,
        "warnings":       [],
    }
    if len(nav_links) < _MIN_ANCHORS:  diagnostics["warnings"].append("sparse nav")
    if not sitemap_links:   diagnostics["warnings"].append("no sitemap found")
    if cs_cf > 0:           diagnostics["warnings"].append(f"{cs_cf} pages CF-blocked")

    phases_ms = {
        "discovery": discovery_ms,
        "homepage_fetch": homepage_ms,
        "bfs_crawl": bfs_ms,
        "article_extraction": article_ms,
        "llm_extraction": llm_ms,
        "total": int((time.perf_counter() - t0) * 1000),
    }

    raw_summary = [{"url": p["url"], "chars": len(p["text"]), "is_doc": p["is_doc"]} for p in raw_pages]
    return {
        "domain": domain, "mode": mode, "elapsed_s": elapsed,
        "pages_fetched": total, "blob_chars": blob_chars, "until_date": until_date,
        "raw_pages": raw_summary,
        "discovery": discovery_stats,
        "crawl_stats": crawl_stats,
        "crawl_trace": crawl_trace,
        "extraction_stats": extraction_stats,
        "quality": quality,
        "diagnostics": diagnostics,
        "phases_ms": phases_ms,
        "data": data,
    }
