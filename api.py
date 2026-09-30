"""Read/write HTTP API over cmi_newsletter.db, for the dashboard and for logging.

    pip install fastapi uvicorn
    python api.py                      # http://127.0.0.1:8080/docs

Reads are open. The one endpoint that needs care is POST /briefs/{id}/send: it puts a
real email in front of a real person, so it

  - requires an explicit {"confirm": true} in the body,
  - refuses any brief whose validation carries a hard failure,
  - refuses an unsubscribed address,
  - and is idempotent per (email, campaign) through the existing sends unique index.

Binding to 127.0.0.1 by default is deliberate: nothing here authenticates a caller, so
it must not be reachable from another machine without a proxy in front that does.
"""
from __future__ import annotations

import json
import os
import sys

try:
    from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
    from fastapi.responses import HTMLResponse
except ImportError:  # noqa: BLE001
    raise SystemExit("pip install fastapi uvicorn")

import db

app = FastAPI(title="CMI newsletter API", version="1.1")


# ── auth ──────────────────────────────────────────────────────────────────────
# Set CMI_API_KEY to require a key on every request. It is OPTIONAL so that the
# localhost-only default keeps working with no setup -- but the moment the API is
# bound anywhere other than 127.0.0.1, a key is mandatory (enforced in __main__).
# Without one, anyone who can reach the port can send email as you.
API_KEY = os.environ.get("CMI_API_KEY", "")


def require_key(x_api_key: str = Header(default="", alias="X-API-Key"),
                authorization: str = Header(default="")) -> None:
    """Accept the key as `X-API-Key: <key>` or `Authorization: Bearer <key>`.

    Both are supported because command-line tools and SDKs reach for different
    conventions; requiring one specific header is a needless failure mode.
    """
    if not API_KEY:
        return
    supplied = x_api_key or (authorization[7:] if authorization.startswith("Bearer ")
                             else "")
    # Constant-time compare: a plain == leaks key length and prefix through timing.
    import hmac

    if not supplied or not hmac.compare_digest(supplied, API_KEY):
        raise HTTPException(401, "missing or invalid API key "
                                 "(send X-API-Key or Authorization: Bearer)")


@app.middleware("http")
async def _auth_middleware(request, call_next):
    """Require the key on everything except /health and the docs.

    Done as middleware rather than a per-route dependency on purpose: a route added
    later is protected by default, instead of being silently public because someone
    forgot to add Depends(require_key) to it.
    """
    open_paths = {"/health", "/docs", "/openapi.json", "/redoc"}
    if API_KEY and request.url.path not in open_paths:
        supplied = request.headers.get("x-api-key", "")
        if not supplied:
            auth = request.headers.get("authorization", "")
            supplied = auth[7:] if auth.startswith("Bearer ") else ""
        import hmac

        from fastapi.responses import JSONResponse

        if not supplied or not hmac.compare_digest(supplied, API_KEY):
            return JSONResponse(
                {"detail": "missing or invalid API key "
                           "(send X-API-Key or Authorization: Bearer)"},
                status_code=401)
    return await call_next(request)


@app.get("/health")
def health():
    """Unauthenticated liveness probe. Deliberately reveals nothing about the data."""
    return {"ok": True, "auth_required": bool(API_KEY)}


def _conn():
    c = db.get_conn()
    db.init_db(c)
    return c


def _rows(rows) -> list[dict]:
    return [dict(r) for r in rows]


def _validation(row) -> dict:
    try:
        return json.loads(row["validation"] or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


# ── leads ─────────────────────────────────────────────────────────────────────
@app.get("/leads")
def leads(campaign: str = "", status: str = "", limit: int = Query(200, le=2000)):
    c = _conn()
    q = "SELECT id,email,name,company,market,report_name,campaign,kind,schedule_status,created_at FROM leads"
    where, args = [], []
    if campaign:
        where.append("campaign = ?"); args.append(campaign)
    if status:
        where.append("schedule_status = ?"); args.append(status)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    return _rows(c.execute(q, args).fetchall())


@app.post("/leads/import")
def leads_import(xlsx: str = Body(..., embed=True), campaign: str = Body("default", embed=True),
                 kind: str = Body("report", embed=True), limit: int = Body(500, embed=True)):
    """Import a lead sheet already on this machine. Takes a path, not an upload: the
    dashboard and the API run beside each other, and a path avoids a second copy."""
    if not os.path.isfile(xlsx):
        raise HTTPException(404, f"no such file: {xlsx}")
    c = _conn()
    sys.argv = ["x"]
    import batch_generate as bg

    rows = bg.read_leads(xlsx, limit)
    added = 0
    for l in rows:
        seen = c.execute("SELECT 1 FROM leads WHERE email = ?", (l["email"],)).fetchone()
        db.upsert_lead(c, email=l["email"], name=l["name"], company=l["company"],
                       market=l["country"], report_name=l.get("topic", ""),
                       source="api", campaign=campaign, kind=kind)
        added += seen is None
    return {"read": len(rows), "new": added, "existing": len(rows) - added}


# ── briefs ────────────────────────────────────────────────────────────────────
@app.get("/briefs")
def briefs(status: str = "", run_id: int | None = None, limit: int = Query(200, le=2000)):
    c = _conn()
    out = []
    for r in db.list_briefs(c, status=status, run_id=run_id, limit=limit):
        d = dict(r)
        v = _validation(r)
        d["fails"] = v.get("fails", [])
        d["warns"] = v.get("warns", [])
        d.pop("validation", None)
        out.append(d)
    return out


@app.get("/briefs/{brief_id}/html", response_class=HTMLResponse)
def brief_html(brief_id: int):
    row = db.get_brief(_conn(), brief_id)
    if row is None:
        raise HTTPException(404, "no such brief")
    return HTMLResponse(row["html"] or "<p>(empty)</p>")


@app.post("/briefs/{brief_id}/send")
def brief_send(brief_id: int, confirm: bool = Body(False, embed=True),
               campaign: str = Body("api", embed=True)):
    """Send one brief. Guarded, because this is the only irreversible endpoint."""
    if not confirm:
        raise HTTPException(400, "pass {\"confirm\": true} — this sends a real email")

    c = _conn()
    row = db.get_brief(c, brief_id)
    if row is None:
        raise HTTPException(404, "no such brief")
    if row["status"] == "sent":
        return {"status": "already_sent", "brief_id": brief_id}

    fails = _validation(row).get("fails", [])
    if fails:
        raise HTTPException(409, {"error": "brief has hard validation failures",
                                  "fails": fails})
    if db.is_unsubscribed(c, row["email"]):
        raise HTTPException(409, "recipient has unsubscribed")

    import mailer

    subject = db.brief_subject(row["company"], row["kind"], row["topic"])
    try:
        mailer.send_html(row["html"], row["email"], subject,
                         filename_hint=row["email"], conn=c)
    except Exception as e:  # noqa: BLE001
        db.record_send(c, email=row["email"], status="failed",
                       detail=f"{type(e).__name__}: {e}", campaign=campaign)
        raise HTTPException(502, f"send failed: {type(e).__name__}: {e}")

    db.record_send(c, email=row["email"], status="sent", subject=subject, campaign=campaign)
    db.set_brief_status(c, brief_id, "sent")
    db.set_lead_status(c, row["email"], "sent")
    return {"status": "sent", "brief_id": brief_id, "email": row["email"]}


# ── runs ──────────────────────────────────────────────────────────────────────
@app.get("/runs")
def runs(limit: int = Query(50, le=500)):
    return _rows(db.list_runs(_conn(), limit))


@app.get("/runs/{run_id}")
def run_detail(run_id: int):
    c = _conn()
    row = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "no such run")
    d = dict(row)
    d["briefs"] = len(db.list_briefs(c, run_id=run_id, limit=10000))
    d["sendable"] = len(db.sendable_briefs(c, run_id=run_id))
    return d


# ── intel ─────────────────────────────────────────────────────────────────────
@app.get("/intel/{domain}")
def intel(domain: str):
    row = db.get_intel(_conn(), domain)
    if row is None:
        raise HTTPException(404, "not cached")
    row.pop("raw_json", None)      # the dump is large and rarely what a caller wants
    return row


@app.get("/intel")
def intel_list(source: str = "", limit: int = Query(200, le=2000)):
    c = _conn()
    q = "SELECT domain, company, source, fetched_at, expires_at FROM company_intel"
    args = []
    if source:
        q += " WHERE source = ?"; args.append(source)
    q += " ORDER BY fetched_at DESC LIMIT ?"
    args.append(limit)
    return _rows(c.execute(q, args).fetchall())


# ── vm health ─────────────────────────────────────────────────────────────────
@app.get("/vm/status")
def vm_status():
    """Is the model reachable right now? Never raises -- it is a health check, and a
    dashboard that 500s when the VM is down is useless exactly when it is needed."""
    try:
        import vm_connect

        return vm_connect.status()
    except Exception as e:  # noqa: BLE001
        return {"serving": False, "vm_status": "error",
                "error": f"{type(e).__name__}: {e}"}


@app.post("/vm/reconnect")
def vm_reconnect(spot: bool = Body(False, embed=True),
                 wait: int = Body(900, embed=True)):
    """Resume/boot until the endpoint serves, then record it in .env.

    Not guarded the way sending is: this starts a GPU (a cost), but it is reversible
    and idempotent -- if the endpoint already answers it does nothing at all.
    """
    import vm_connect

    st = vm_connect.ensure(spot=spot, wait_secs=wait)
    if not st.get("serving"):
        raise HTTPException(503, {"error": st.get("error") or "endpoint did not come up",
                                  "status": st})
    return st


# ── analytics ─────────────────────────────────────────────────────────────────
@app.get("/analytics")
def analytics(days: int = Query(30, le=365)):
    """Everything the dashboard charts, in one round trip."""
    c = _conn()
    one = lambda q, a=(): c.execute(q, a).fetchone()[0]  # noqa: E731
    rows = lambda q, a=(): [dict(r) for r in c.execute(q, a).fetchall()]  # noqa: E731
    since = f"-{int(days)} days"

    briefs_by_status = rows(
        "SELECT status, COUNT(*) AS n FROM briefs GROUP BY status ORDER BY n DESC")
    briefs_by_kind = rows(
        "SELECT kind, COUNT(*) AS n FROM briefs GROUP BY kind ORDER BY n DESC")
    sends_by_status = rows(
        "SELECT status, COUNT(*) AS n FROM sends GROUP BY status ORDER BY n DESC")
    daily = rows(
        "SELECT date(created_at) AS day, COUNT(*) AS briefs FROM briefs "
        "WHERE created_at >= date('now', ?) GROUP BY day ORDER BY day", (since,))
    sends_daily = rows(
        # NB: sends uses `sent_at`, not `created_at` like briefs.
        "SELECT date(sent_at) AS day, COUNT(*) AS sends FROM sends "
        "WHERE sent_at >= date('now', ?) GROUP BY day ORDER BY day", (since,))

    # Validation verdicts live as JSON on the row, so they are counted in Python
    # rather than SQL -- the same reason sendable_briefs does its filtering there.
    verdicts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    fail_reasons: dict[str, int] = {}
    for r in c.execute("SELECT validation FROM briefs WHERE validation IS NOT NULL"):
        try:
            v = json.loads(r[0] or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(v, dict):
            continue
        fails, warns = v.get("fails") or [], v.get("warns") or []
        verdicts["FAIL" if fails else ("WARN" if warns else "PASS")] += 1
        for f in fails:
            key = str(f).split("--")[0].split(":")[0][:60].strip()
            fail_reasons[key] = fail_reasons.get(key, 0) + 1

    gen = c.execute("SELECT AVG(gen_seconds), MIN(gen_seconds), MAX(gen_seconds) "
                    "FROM briefs WHERE gen_seconds > 0").fetchone()
    return {
        "totals": {
            "leads": one("SELECT COUNT(*) FROM leads"),
            "briefs": one("SELECT COUNT(*) FROM briefs"),
            "sent": one("SELECT COUNT(*) FROM briefs WHERE status='sent'"),
            "sendable_now": len(db.sendable_briefs(c)),
            "sends": one("SELECT COUNT(*) FROM sends"),
            "unsubscribes": one("SELECT COUNT(*) FROM unsubscribes"),
            "intel_cached": one("SELECT COUNT(*) FROM company_intel"),
            "intel_thin": one("SELECT COUNT(*) FROM company_intel WHERE source='thin'"),
            "runs": one("SELECT COUNT(*) FROM runs"),
        },
        "briefs_by_status": briefs_by_status,
        "briefs_by_kind": briefs_by_kind,
        "sends_by_status": sends_by_status,
        "verdicts": verdicts,
        "top_failures": sorted(fail_reasons.items(), key=lambda kv: -kv[1])[:8],
        "daily_briefs": daily,
        "daily_sends": sends_daily,
        "gen_seconds": {"avg": round(gen[0] or 0, 1), "min": gen[1] or 0,
                        "max": gen[2] or 0},
        "recent_runs": rows("SELECT * FROM runs ORDER BY id DESC LIMIT 10"),
    }


@app.get("/stats")
def stats():
    c = _conn()
    one = lambda q: c.execute(q).fetchone()[0]  # noqa: E731
    return {
        "leads": one("SELECT COUNT(*) FROM leads"),
        "briefs": one("SELECT COUNT(*) FROM briefs"),
        "briefs_sent": one("SELECT COUNT(*) FROM briefs WHERE status='sent'"),
        "sends": one("SELECT COUNT(*) FROM sends"),
        "unsubscribes": one("SELECT COUNT(*) FROM unsubscribes"),
        "intel_cached": one("SELECT COUNT(*) FROM company_intel"),
        "intel_thin": one("SELECT COUNT(*) FROM company_intel WHERE source='thin'"),
        "runs": one("SELECT COUNT(*) FROM runs"),
    }


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("CMI_API_HOST", "127.0.0.1")
    # Binding beyond loopback without a key would expose an unauthenticated endpoint
    # that can send real email. Refuse rather than warn.
    if host not in ("127.0.0.1", "localhost") and not API_KEY:
        raise SystemExit(
            f"refusing to bind {host} without CMI_API_KEY set.\n"
            "  generate one:  python -c \"import secrets; print(secrets.token_urlsafe(32))\"\n"
            "  then set CMI_API_KEY to it before starting the API.")
    uvicorn.run(app, host=host, port=int(os.environ.get("CMI_API_PORT", "8080")))
