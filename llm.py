"""DeepSeek-first LLM wrapper with automatic OpenAI fallback.

Mirrors the mother repo's `email/_llm.py` idea (DeepSeek is OpenAI-API-compatible, so we
use the `openai` SDK with a custom base_url). Provider order is DeepSeek → OpenAI so the
newsletter "uses DeepSeek" as requested, but still runs if the DeepSeek balance is 0.
"""
from __future__ import annotations
import json
import os
import re
import sys
import threading
from datetime import datetime, timezone

from openai import OpenAI
from dotenv import dotenv_values

import config

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_LOG_PATH = os.path.join(HERE, "training_data", "live_calls.jsonl")
_training_log_lock = threading.Lock()


def _safe_print(msg: str) -> None:
    """print() that never raises on a console codec that can't encode the message
    (Windows cp1252 choking on an arrow or other non-ASCII character in an error)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(str(msg).encode(enc, errors="replace").decode(enc, errors="replace"))

# Load keys (values never printed): sibling coherent-newsletter/.env, then a local .env
# in this folder which takes precedence if present, then process env.
_vals = {}
_shared_env = os.path.join(os.path.dirname(__file__), "..", "coherent-newsletter", ".env")
try:
    _vals = dotenv_values(_shared_env) or {}
except Exception:
    _vals = {}
_local_env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_local_env):
    _vals = {**_vals, **(dotenv_values(_local_env) or {})}


def _key(name: str) -> str:
    return os.getenv(name) or _vals.get(name) or ""


DEEPSEEK_API_KEY = _key("DEEPSEEK_API_KEY")
OPENAI_API_KEY = _key("OPENAI_API_KEY")
PROVIDER = _key("LLM_PROVIDER") or config.LLM["provider"]

# Explicit per-request timeout on every client -- without one, a hung request to an
# overloaded server blocks its worker thread for the SDK's own default (several
# minutes) with no error, silently draining a batch's effective concurrency. See
# config.LLM['timeout_secs'] for the full reasoning.
try:
    _LLM_TIMEOUT = float(_key("CMI_LLM_TIMEOUT_SECS") or config.LLM["timeout_secs"])
except (TypeError, ValueError):
    _LLM_TIMEOUT = config.LLM["timeout_secs"]

_ds_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=config.LLM["deepseek_base_url"],
                    timeout=_LLM_TIMEOUT) if DEEPSEEK_API_KEY else None
_oa_client = OpenAI(api_key=OPENAI_API_KEY, timeout=_LLM_TIMEOUT) if OPENAI_API_KEY else None

# Self-hosted fine-tune. Any OpenAI-compatible server works (vLLM, TGI, llama.cpp);
# the API key is unused by most of them but the SDK requires a non-empty string.
LOCAL_BASE_URL = _key("LOCAL_LLM_BASE_URL") or config.LLM["local_base_url"]
LOCAL_MODEL = _key("LOCAL_LLM_MODEL") or config.LLM["local_model"]
_local_client = OpenAI(api_key=_key("LOCAL_LLM_API_KEY") or "local", base_url=LOCAL_BASE_URL,
                       timeout=_LLM_TIMEOUT)

# Tracks which provider actually served calls this run (for the run report).
# pipeline.stage_generate calls chat()/chat_json() concurrently from up to `workers`
# threads, and a bare `dict[key] += 1` is a read-then-write, not one atomic step -- two
# threads can both read the old value before either writes back, losing an increment.
# _STATS_LOCK serialises the handful of lines that touch STATS; this is a few
# dict/int ops per LLM call, not the call itself, so it costs nothing worth measuring
# even at 40-way concurrency.
_STATS_LOCK = threading.Lock()
STATS = {"deepseek": 0, "openai": 0, "local": 0, "failed": 0, "provider_used": None}


_detectors_mod = None      # cached module; None once looked up = unavailable
_detectors_lookup_done = False


def _voice_detectors():
    """The shared humanize checker, or None if it isn't reachable.

    Imported lazily and cached: logging runs on every LLM call, and a failed import
    must cost nothing after the first attempt.
    """
    global _detectors_mod, _detectors_lookup_done
    if _detectors_lookup_done:
        return _detectors_mod
    _detectors_lookup_done = True
    for cand in (os.path.join(HERE, "..", "..", "wmr docs", "humanize"),
                 os.path.join(os.path.expanduser("~"), "Desktop", "wmr docs", "humanize")):
        if os.path.isfile(os.path.join(cand, "detectors.py")):
            try:
                sys.path.insert(0, os.path.abspath(cand))
                import detectors  # noqa: PLC0415
                _detectors_mod = detectors
                break
            except Exception:  # noqa: BLE001
                _detectors_mod = None
    return _detectors_mod


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _prose_of(text: str) -> str:
    """The human-readable copy inside a JSON payload, for voice scoring.

    Most pipeline calls are json_mode, so scoring the raw string would grade JSON
    punctuation rather than writing. The prose the voice rules govern lives in the
    string VALUES; short values (enum keys, labels) carry no prose and are skipped.
    """
    t = text.strip()
    if t.startswith("```"):
        try:
            t = t.split("```", 2)[1].removeprefix("json").strip()
        except IndexError:
            pass
    try:
        obj = json.loads(t)
    except (json.JSONDecodeError, ValueError):
        return text
    out: list[str] = []

    def walk(node):
        if isinstance(node, str):
            if len(node.split()) >= 4:
                out.append(_HTML_TAG_RE.sub("", node))
        elif isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(obj)
    return "\n".join(out)


def voice_score(output: str) -> int | None:
    """Slop score for a model output (0 = clean, higher = worse), or None if the
    checker is unavailable. Same detector that gates production copy."""
    det = _voice_detectors()
    if det is None:
        return None
    try:
        prose = _prose_of(output)
        if not prose.strip():
            return 0
        return det.detect(prose).score
    except Exception:  # noqa: BLE001
        return None


def _log_training_pair(system: str, user: str, output: str, *, provider: str, json_mode: bool) -> None:
    """Append a real (system, user, output) triple to training_data/live_calls.jsonl
    when CMI_LOG_TRAINING_DATA=1 — every successfully-served call becomes free
    fine-tuning data going forward, at zero extra LLM cost. Never raises: a logging
    failure must not break brief generation.

    Logged output is VOICE-FILTERED. The first fine-tune trained on whatever DeepSeek
    produced, so it inherited DeepSeek's own rule-breaking almost exactly: the corpus
    averaged 0.91 triads and 0.35 em-dashes per example, and the trained model then
    reproduced both. A student cannot beat its teacher when the teacher's every output
    defines "correct", so the corpus itself has to be the clean subset. Set
    CMI_TRAINING_MAX_SLOP to change the bar (default 8), or 0 to disable filtering.
    """
    if _key("CMI_LOG_TRAINING_DATA") != "1":
        return
    try:
        max_slop = int(_key("CMI_TRAINING_MAX_SLOP") or 8)
    except ValueError:
        max_slop = 8

    score = voice_score(output) if max_slop > 0 else None
    if score is not None and score > max_slop:
        return  # too many voice violations to teach from

    try:
        os.makedirs(os.path.dirname(TRAINING_LOG_PATH), exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "json_mode": json_mode,
            "voice_score": score,
            "system": system,
            "user": user,
            "output": output,
        }
        with _training_log_lock:
            with open(TRAINING_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — logging must never break the pipeline
        pass


def _call(client, model, messages, json_mode, temperature, max_tokens):
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    r = client.chat.completions.create(**kwargs)
    return r.choices[0].message.content or ""


def chat(system: str, user: str, *, json_mode: bool = False,
         temperature: float | None = None, max_tokens: int = 1200) -> str:
    """One completion against the configured provider only. No DeepSeek/OpenAI
    fallback: if the LLM call fails, that failure must stop the run rather than
    silently substitute a different provider or degrade to empty content."""
    temp = config.LLM["temperature"] if temperature is None else temperature
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if PROVIDER == "local":
        name, client, model = "local", _local_client, LOCAL_MODEL
    elif PROVIDER == "deepseek":
        name, client, model = "deepseek", _ds_client, config.LLM["deepseek_model"]
    else:
        name, client, model = "openai", _oa_client, config.LLM["openai_model"]

    if client is None:
        with _STATS_LOCK:
            STATS["failed"] += 1
        raise RuntimeError(f"LLM provider {name!r} is not configured (missing client/key)")

    try:
        out = _call(client, model, messages, json_mode, temp, max_tokens)
        with _STATS_LOCK:
            STATS[name] += 1
            STATS["provider_used"] = STATS["provider_used"] or name
        _log_training_pair(system, user, out, provider=name, json_mode=json_mode)
        return out
    except Exception as e:  # noqa: BLE001 — re-raised below; a failed call must not be swallowed
        with _STATS_LOCK:
            STATS["failed"] += 1
        raise RuntimeError(f"LLM provider {name!r} failed: {type(e).__name__}: {e}") from e


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    # tolerate ```json fences
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1].removeprefix("json").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Models occasionally emit a literal control character (raw newline/tab)
        # inside a string value instead of the escaped \n / \t -- strip those out
        # rather than discarding an otherwise well-formed response.
        repaired = re.sub(r'(?<!\\)[\x00-\x1f]+', " ", raw)
        return json.loads(repaired)


def chat_json(system: str, user: str, *, temperature: float | None = None,
              max_tokens: int = 1200, default: dict | None = None) -> dict:
    """chat() in JSON mode, parsed. Retries once on a JSON *parse* failure only (a
    single bad control character shouldn't permanently drop a whole section) --
    but a failed call to the LLM itself (provider down, timeout, etc.) is NOT a
    parse failure and must propagate immediately. Silently degrading to `default`
    on a real outage is how a dead endpoint produced hundreds of hollow briefs
    that still looked structurally valid -- an outage must stop the run, not
    disappear as an empty section."""
    # A failed call (provider down, timeout, etc.) is not caught here -- it propagates
    # straight out of chat_json, same as chat(). Only a JSON *parse* failure on an
    # otherwise-successful call gets a retry-then-degrade.
    raw = chat(system, user, json_mode=True, temperature=temperature, max_tokens=max_tokens)
    try:
        return _parse_json(raw)
    except Exception as e:  # noqa: BLE001 — genuine parse failure, one retry is reasonable
        _safe_print(f"  [llm] json parse failed, retrying once: {type(e).__name__}: {str(e)[:80]}")

    raw = chat(system, user, json_mode=True, temperature=temperature, max_tokens=max_tokens)
    try:
        return _parse_json(raw)
    except Exception as e:  # noqa: BLE001 — second parse failure: degrade, do not fail the run
        _safe_print(f"  [llm] json parse/degrade: {type(e).__name__}: {str(e)[:80]}")
        return default if default is not None else {}
