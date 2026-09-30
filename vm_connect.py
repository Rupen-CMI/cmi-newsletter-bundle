"""Find, heal, and record the connection to the model endpoint.

    python vm_connect.py --status            # what is the VM doing right now
    python vm_connect.py --ensure            # get to a serving endpoint, whatever it takes
    python vm_connect.py --ensure --spot     # same, but resume onto spot
    python vm_connect.py --watch             # keep it serving, forever

The problem this solves: **the machine id changes on every resume**, and the endpoint
URL is written into `.env`. Resume the box and `LOCAL_LLM_BASE_URL` silently points at
a dead host; a spot reclaim does the same thing without anyone touching it. That has
already cost a full generation run here -- four briefs produced against a 404 endpoint
and still reported PASS, because empty sections are structurally valid.

So nothing in this module trusts a stored id or URL. Every call rediscovers the
instance by NAME through `jl list --json`, and readiness always means "a real chat
completion came back", never "the process is up" or "/v1/models answered" -- that
endpoint can respond while the engine is still loading.

The ladder, cheapest first:

    endpoint already completes      -> done, no API calls
    instance Running, not serving   -> run the boot script (install + serve in tmux)
    instance Paused                 -> resume (NEW id), then boot
    GPU unavailable               -> try the fallbacks in GPU_LADDER
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

INSTANCE_NAME = os.getenv("CMI_VM_NAME", "cmi-serve-v1")
MODEL_NAME = os.getenv("LOCAL_LLM_MODEL", "cmi-7b")
SERVE_PORT = "6006"
BOOT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "ops", "boot_serve.sh")

# L4 is the tuned target (Rs.41.31 on-demand / Rs.27.54 spot). A30 is the same 24 GB at
# the same spot price and is a straight substitute when L4 capacity is gone; A100 is
# ~2x the price and only worth it when nothing else is free.
GPU_LADDER = ["L4", "A30", "A100"]

_JL_CANDIDATES = [
    "jl",
    os.path.expandvars(r"%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.12"
                       r"_qbz5n2kfra8p0\LocalCache\local-packages\Python312\Scripts\jl.exe"),
]


def _log(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(str(msg).encode(enc, "replace").decode(enc, "replace"), flush=True)


def jl_path() -> str:
    """Locate the jl CLI. It is not on PATH in a fresh shell on this machine."""
    for c in _JL_CANDIDATES:
        if c == "jl":
            try:
                subprocess.run([c, "--version"], capture_output=True, timeout=20)
                return c
            except (OSError, subprocess.SubprocessError):
                continue
        elif os.path.isfile(c):
            return c
    raise RuntimeError("jl CLI not found -- pip install jarvislabs")


def _jl(args: list[str], timeout: int = 600) -> tuple[int, str]:
    try:
        p = subprocess.run([jl_path(), *args], capture_output=True, text=True,
                           timeout=timeout, encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 1, "timeout"
    except Exception as e:  # noqa: BLE001
        return 1, f"{type(e).__name__}: {e}"


# ── discovery ─────────────────────────────────────────────────────────────────
def find_instance(name: str = INSTANCE_NAME) -> dict | None:
    """The current instance row, found by NAME.

    Never by id: `jl resume` returns a NEW machine id every time, so any id written
    down anywhere is stale the moment the box is resumed.
    """
    code, out = _jl(["list", "--json"], timeout=120)
    if code != 0:
        return None
    try:
        rows = json.loads(out[out.index("["):out.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    for r in rows:
        if r.get("name") == name:
            return r
    return rows[0] if len(rows) == 1 else None


def endpoint_of(inst: dict) -> str:
    """The OpenAI-compatible base URL for a running instance, or ''.

    `endpoints` is null while paused, which is the usual source of confusion when the
    URL 'disappears'.
    """
    eps = inst.get("endpoints") or {}
    if isinstance(eps, dict):
        for k, v in eps.items():
            if SERVE_PORT in str(k) and v:
                return f"{str(v).rstrip('/')}/v1"
    for v in (eps.values() if isinstance(eps, dict) else eps or []):
        if v:
            return f"{str(v).rstrip('/')}/v1"
    return ""


# ── readiness ─────────────────────────────────────────────────────────────────
def is_serving(base_url: str, timeout: int = 20) -> bool:
    """True only when a real completion comes back.

    `/v1/models` is NOT sufficient: it answers while the engine is still loading, and
    trusting it is how a whole batch got generated against a dead endpoint.
    """
    if not base_url:
        return False
    import urllib.error
    import urllib.request

    payload = json.dumps({"model": MODEL_NAME, "max_tokens": 4,
                          "messages": [{"role": "user", "content": "ping"}]}).encode()
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer local"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return b'"content"' in r.read()
    except (urllib.error.URLError, OSError, ValueError):
        return False


def current_base_url() -> str:
    from dotenv import dotenv_values

    here = os.path.dirname(os.path.abspath(__file__))
    vals = dotenv_values(os.path.join(here, ".env"))
    return os.getenv("LOCAL_LLM_BASE_URL") or vals.get("LOCAL_LLM_BASE_URL") or ""


def write_base_url(url: str) -> None:
    """Persist the discovered endpoint into .env, rewriting only that one line.

    The whole file is rewritten in place rather than appended to, so repeated healing
    cannot leave a stack of stale LOCAL_LLM_BASE_URL lines behind -- dotenv takes the
    last one, and a shadowed old value is very hard to spot.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, ".env")
    lines, seen = [], False
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("LOCAL_LLM_BASE_URL="):
                    lines.append(f"LOCAL_LLM_BASE_URL={url}\n")
                    seen = True
                else:
                    lines.append(line)
    if not seen:
        lines.append(f"LOCAL_LLM_BASE_URL={url}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


# ── status ────────────────────────────────────────────────────────────────────
def status() -> dict:
    """A full picture for the dashboard. Never raises -- it is a health check."""
    out = {"instance": None, "machine_id": None, "vm_status": "unknown",
           "gpu": "", "is_spot": None, "endpoint": "", "serving": False,
           "env_url": current_base_url(), "env_matches": False, "error": ""}
    try:
        inst = find_instance()
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        # The CLI may be missing, but a live endpoint in .env still works.
        out["serving"] = is_serving(out["env_url"])
        out["endpoint"] = out["env_url"] if out["serving"] else ""
        return out

    if not inst:
        out["vm_status"] = "not found"
        out["serving"] = is_serving(out["env_url"])
        return out

    out.update(instance=inst.get("name"), machine_id=inst.get("machine_id"),
               vm_status=inst.get("status", "unknown"), gpu=inst.get("gpu_type", ""),
               is_spot=inst.get("is_spot"))
    ep = endpoint_of(inst)
    out["endpoint"] = ep
    out["env_matches"] = bool(ep) and ep == out["env_url"]
    # Check the endpoint the code will actually use, not just the discovered one.
    out["serving"] = is_serving(ep or out["env_url"])
    return out


# ── healing ───────────────────────────────────────────────────────────────────
def _boot(machine_id: int) -> None:
    """Upload and run the boot script inside tmux.

    tmux matters: a plain nohup/setsid through `jl exec` dies when the exec session is
    torn down at timeout, and the symptom is a server that answers for a few minutes
    and then 404s mid-run.
    """
    if os.path.isfile(BOOT_SCRIPT):
        _jl(["upload", str(machine_id), BOOT_SCRIPT, "/home/boot_serve.sh"], timeout=300)
    _jl(["exec", str(machine_id), "--", "bash", "-lc",
         "sed -i 's/\\r$//' /home/boot_serve.sh; "
         "tmux kill-session -t boot 2>/dev/null; "
         "tmux new-session -d -s boot 'bash /home/boot_serve.sh'"], timeout=300)


def _resume(machine_id: int, spot: bool) -> int | None:
    """Resume, trying each GPU in the ladder. Returns the NEW machine id."""
    for gpu in GPU_LADDER:
        args = ["resume", str(machine_id), "--gpu", gpu, "--yes"]
        if spot:
            args.insert(3, "--spot")
        _log(f"  resume on {gpu}{' (spot)' if spot else ''}...")
        code, out = _jl(args, timeout=900)
        if "not available" in out.lower():
            _log(f"  {gpu} unavailable")
            continue
        inst = find_instance()
        if inst and inst.get("status") == "Running":
            return inst.get("machine_id")
        if code == 0:
            return machine_id
    return None


def ensure(*, spot: bool = False, wait_secs: int = 900, log=_log) -> dict:
    """Get to a serving endpoint and record it. Returns the final status().

    Idempotent and cheap on the happy path: if the endpoint already completes, this
    makes no API calls at all.
    """
    st = status()
    if st["serving"]:
        if st["endpoint"] and not st["env_matches"]:
            write_base_url(st["endpoint"])
            log(f"  endpoint moved -> {st['endpoint']} (.env updated)")
        log("  already serving")
        return st

    inst = find_instance()
    if not inst:
        st["error"] = (f"no instance named {INSTANCE_NAME!r}. Create one:\n"
                       f"  jl create --gpu L4 --spot --template pytorch "
                       f"--storage 60 --name {INSTANCE_NAME} --yes")
        log("  " + st["error"])
        return st

    mid = inst.get("machine_id")
    if inst.get("status") != "Running":
        log(f"  instance {mid} is {inst.get('status')} -- resuming")
        mid = _resume(mid, spot)
        if not mid:
            st = status()
            st["error"] = "no GPU available on any of " + ", ".join(GPU_LADDER)
            log("  " + st["error"])
            return st
        log(f"  running as {mid}")

    log("  booting server (install if needed, then serve in tmux)...")
    _boot(mid)

    deadline = time.time() + wait_secs
    while time.time() < deadline:
        inst = find_instance()
        if inst and inst.get("status") != "Running":
            # A spot box can be reclaimed while we are waiting for it to boot.
            log("  instance stopped while booting -- resuming again")
            mid = _resume(inst.get("machine_id"), spot)
            if mid:
                _boot(mid)
            continue
        ep = endpoint_of(inst or {}) or current_base_url()
        if is_serving(ep):
            write_base_url(ep)
            log(f"  serving at {ep}")
            return status()
        time.sleep(20)

    st = status()
    st["error"] = f"endpoint did not come up within {wait_secs}s"
    log("  " + st["error"])
    return st


def pause(log=_log) -> dict:
    """Pause the instance. `/home/merged` survives; the server does not.

    Exists because the cost asymmetry here is brutal: an idle L4 is Rs.27-41/hr for
    nothing, and the single easiest way to burn the balance is forgetting to do this
    after a run.
    """
    inst = find_instance()
    if not inst:
        return {"paused": False, "error": f"no instance named {INSTANCE_NAME!r}"}
    mid = inst.get("machine_id")
    if inst.get("status") != "Running":
        log(f"  already {inst.get('status')}")
        return {"paused": True, "machine_id": mid, "already": True}
    code, out = _jl(["pause", str(mid), "--yes"], timeout=600)
    ok = code == 0 or "paused" in out.lower()
    log(f"  {'paused' if ok else 'pause failed'} {mid}")
    return {"paused": ok, "machine_id": mid, "detail": out[-200:] if not ok else ""}


def watch(interval: int = 120, *, spot: bool = False) -> None:
    """Keep the endpoint alive. Re-heals after a reclaim, a pause, or an id change."""
    _log(f"watching {INSTANCE_NAME} every {interval}s -- Ctrl-C to stop")
    while True:
        try:
            st = ensure(spot=spot, log=lambda m: None)
            flag = "OK  " if st["serving"] else "DOWN"
            _log(f"[{time.strftime('%H:%M:%S')}] {flag} {st['vm_status']:<10}"
                 f"id={st['machine_id']} {st['endpoint'] or st['error'][:50]}")
        except KeyboardInterrupt:
            _log("stopped")
            return
        except Exception as e:  # noqa: BLE001 -- a watchdog must not die
            _log(f"[{time.strftime('%H:%M:%S')}] watcher error: {type(e).__name__}: {e}")
        time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description="Find, heal and record the model endpoint.")
    ap.add_argument("--status", action="store_true", help="report and exit")
    ap.add_argument("--ensure", action="store_true", help="get to a serving endpoint")
    ap.add_argument("--watch", action="store_true", help="keep it serving")
    ap.add_argument("--pause", action="store_true",
                    help="pause the instance (stops billing; keeps /home/merged)")
    ap.add_argument("--spot", action="store_true", help="resume onto spot pricing")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--wait", type=int, default=900)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    if args.pause:
        r = pause()
        print(json.dumps(r, indent=2))
        raise SystemExit(0 if r.get("paused") else 1)
    if args.watch:
        watch(args.interval, spot=args.spot)
    elif args.ensure:
        st = ensure(spot=args.spot, wait_secs=args.wait)
        raise SystemExit(0 if st["serving"] else 1)
    else:
        st = status()
        print(json.dumps(st, indent=2))
        raise SystemExit(0 if st["serving"] else 1)


if __name__ == "__main__":
    main()
