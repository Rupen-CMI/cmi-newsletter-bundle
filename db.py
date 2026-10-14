"""SQLite storage for the CMI newsletter system: non-secret settings, contacts, and
send history. Actual credentials (SMTP host/user/password) stay in `.env`, never in
this DB — see mailer.py. This file only tracks things that are safe to keep in a
plain SQLite file: sender display name/from-address, reply-to list, recipients
uploaded for campaigns, and a log of what was sent to whom.

    from db import get_conn, init_db
    conn = get_conn()
    init_db(conn)
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "cmi_newsletter.db")


def _now() -> str:
    """UTC ISO timestamp. One definition so every table's timestamps sort together."""
    return datetime.now(timezone.utc).isoformat()


# (db_path, thread_id) -> connection. Keyed by thread because sqlite3 connections
# are not shareable across threads; see get_conn.
_CONNS: dict[tuple[str, int], sqlite3.Connection] = {}


def get_conn(db_path: str | None = None) -> sqlite3.Connection:
    """A connection for the CURRENT thread.

    sqlite3 binds a connection to the thread that created it, so a single shared
    connection breaks the moment anything multi-threaded touches it: Streamlit reruns
    each script pass on a different thread, and a ThreadPoolExecutor obviously does
    too. Both have already caused
    `ProgrammingError: SQLite objects created in a thread can only be used in that
    same thread` here.

    Connections are therefore cached PER THREAD rather than globally. Callers can keep
    calling get_conn() freely -- they get the same object within a thread, and a fresh
    one in a new thread -- so no caller needs to reason about this.
    """
    key = (db_path or DB_PATH, threading.get_ident())
    conn = _CONNS.get(key)
    if conn is not None:
        try:
            conn.execute("SELECT 1")
            return conn
        except sqlite3.Error:
            # Closed or otherwise unusable -- drop it and open a fresh one.
            _CONNS.pop(key, None)
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL lets readers run while a writer holds the file, which is what makes a
    # dashboard usable while a batch is writing briefs.
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA busy_timeout = 5000")
    except sqlite3.Error:
        pass  # a read-only or unusual filesystem -- not fatal
    _CONNS[key] = conn
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS settings (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS leads (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        email           TEXT NOT NULL UNIQUE,
        name            TEXT,
        company         TEXT,
        job_title       TEXT,
        market          TEXT DEFAULT 'Global',
        report_name     TEXT,
        source          TEXT DEFAULT 'manual',    -- manual | xlsx_import | api
        schedule_status TEXT DEFAULT 'queued',    -- queued | sent | failed | unsubscribed | paused
        campaign        TEXT DEFAULT 'default',   -- which send_status/sends campaign label this lead belongs to
        kind            TEXT DEFAULT 'signal',    -- signal | report
        last_attempt_at TEXT,
        created_at      TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS sends (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        email       TEXT NOT NULL,
        status      TEXT NOT NULL,           -- sent | failed | unsubscribed | skipped
        subject     TEXT,
        detail      TEXT,
        campaign    TEXT,                    -- optional batch label
        sent_at     TEXT NOT NULL
    );

    CREATE UNIQUE INDEX IF NOT EXISTS idx_sends_email_campaign
        ON sends(email, campaign);

    CREATE TABLE IF NOT EXISTS unsubscribes (
        email        TEXT PRIMARY KEY,
        unsubscribed_at TEXT NOT NULL,
        reason       TEXT
    );

    -- Scraped company facts, cached per DOMAIN rather than per brief. smartcrawl takes
    -- ~20 minutes per company on the local model, and a lead list shares far fewer
    -- domains than it has rows, so this is the difference between hours and minutes.
    -- It is also what stops the model guessing: profile inference from a company NAME
    -- alone once produced "immunoassay CDMO" for a CAR-T maker.
    CREATE TABLE IF NOT EXISTS company_intel (
        domain        TEXT PRIMARY KEY,
        company       TEXT,
        value_prop    TEXT,
        industries    TEXT,      -- json array
        competitors   TEXT,      -- json array
        product_terms TEXT,      -- json array
        raw_json      TEXT,      -- full smartcrawl dump, so fields can be re-derived
        source        TEXT,      -- smartcrawl | homepage | curated
        fetched_at    TEXT NOT NULL,
        expires_at    TEXT       -- refresh after N days rather than never
    );

    -- Every generated brief, whether or not it was sent. Holding the HTML means a
    -- re-send never regenerates, and a validation verdict stays attached to the exact
    -- document it judged.
    CREATE TABLE IF NOT EXISTS briefs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        email       TEXT NOT NULL,
        company     TEXT,
        kind        TEXT,        -- signal | report
        topic       TEXT,        -- the enquired market, for report briefs
        html        TEXT,
        status      TEXT,        -- generated | validated | sent | failed
        validation  TEXT,        -- json {"fails": [...], "warns": [...]}
        llm_calls   INTEGER,
        gen_seconds REAL,
        run_id      INTEGER,
        created_at  TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_briefs_email  ON briefs(email);
    CREATE INDEX IF NOT EXISTS idx_briefs_status ON briefs(status);
    CREATE INDEX IF NOT EXISTS idx_briefs_run    ON briefs(run_id);

    CREATE TABLE IF NOT EXISTS runs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at  TEXT NOT NULL,
        finished_at TEXT,
        workers     INTEGER,
        kind        TEXT,
        n_briefs    INTEGER DEFAULT 0,
        n_pass      INTEGER DEFAULT 0,
        n_warn      INTEGER DEFAULT 0,
        n_fail      INTEGER DEFAULT 0,
        wall_secs   REAL,
        cost_inr    REAL,
        endpoint    TEXT,
        notes       TEXT
    );
    """)
    conn.commit()
    _migrate_leads_columns(conn)
    _migrate_briefs_columns(conn)
    _migrate_sends_columns(conn)


def _migrate_sends_columns(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the first release to any sends table created by
    an older version of init_db, so upgrading never requires deleting the DB file."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(sends)")}
    additions = {
        # SendGrid's X-Message-Id, set only when status='sent'. Without this there is
        # no way to later match a reply or a bounce/complaint webhook back to a
        # specific send -- ported from the sibling newsletter-cc-cl project's
        # campaign.py, which already tracks this for exactly that reason.
        "message_id": "TEXT",
    }
    for col, decl in additions.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE sends ADD COLUMN {col} {decl}")
    conn.commit()


def _migrate_briefs_columns(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the first release to any briefs table created by
    an older version of init_db, so upgrading never requires deleting the DB file."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(briefs)")}
    additions = {
        # The tenant.Profile built at generation time (one LLM call to derive
        # competitors/value_prop/etc) serialized as JSON. Without this, any later,
        # separate-process validation pass has no way to re-check FIT-ERROR-style
        # rules against the seller's profile except by calling the LLM again --
        # duplicate spend for data that was already computed once. Saving it here
        # lets validation reload the exact profile that produced this brief, at
        # zero additional LLM cost, no matter how much later it runs.
        "profile_json": "TEXT",
    }
    for col, decl in additions.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE briefs ADD COLUMN {col} {decl}")
    conn.commit()


def _migrate_leads_columns(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the first release to any leads table created by
    an older version of init_db, so upgrading never requires deleting the DB file."""
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(leads)")}
    additions = {
        "schedule_status": "TEXT DEFAULT 'queued'",
        "campaign": "TEXT DEFAULT 'default'",
        "kind": "TEXT DEFAULT 'signal'",
        "last_attempt_at": "TEXT",
        # Persisted ground truth for which company a lead's brief should be about --
        # a signal brief's generation step reads THIS, not a value re-derived from the
        # email inline at generation time (see tenant.Profile.from_company's
        # 2026-10-07 incident docstring: a wrong/blank company silently produced a
        # brief about CMI itself). No separate "real" website source exists in this
        # schema yet, so this is populated from the email's own domain for now --
        # still real value, because it gives validate_brief.py's DOMAIN check and
        # generation itself ONE persisted value to agree on, instead of each
        # recomputing it ad hoc and silently drifting apart.
        "website": "TEXT DEFAULT ''",
    }
    for col, decl in additions.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE leads ADD COLUMN {col} {decl}")
    conn.commit()


# ── settings (non-secret config: from-name, reply-to list, etc.) ──────────────
def get_setting(conn: sqlite3.Connection, key: str, default: str = "") -> str:
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO settings(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()


# ── leads / contacts ───────────────────────────────────────────────────────────
def upsert_lead(conn: sqlite3.Connection, *, email: str, name: str = "", company: str = "",
                 job_title: str = "", market: str = "Global", report_name: str = "",
                 source: str = "manual", campaign: str = "default", kind: str = "signal",
                 schedule_status: str = "queued", commit: bool = True) -> None:
    """`commit=False` lets a caller importing many rows at once (pipeline.stage_import)
    wrap the whole batch in one transaction instead of committing -- a synchronous
    disk flush -- after every single row. Every other caller keeps the old behaviour
    (commit immediately) by not passing this."""
    conn.execute(
        """INSERT INTO leads(email, name, company, job_title, market, report_name, source,
                              campaign, kind, schedule_status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(email) DO UPDATE SET
             name=excluded.name, company=excluded.company, job_title=excluded.job_title,
             market=excluded.market, report_name=excluded.report_name, source=excluded.source,
             campaign=excluded.campaign, kind=excluded.kind""",
        (email.strip().lower(), name, company, job_title, market, report_name, source,
         campaign, kind, schedule_status, datetime.now(timezone.utc).isoformat()),
    )
    if commit:
        conn.commit()


def list_leads(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM leads ORDER BY created_at DESC").fetchall()


def queued_leads(conn: sqlite3.Connection, campaign: str | None = None) -> list[sqlite3.Row]:
    """Leads still waiting to be sent (schedule_status='queued'), optionally filtered
    to one campaign label. Used by scheduler.py to decide who's next in line."""
    if campaign:
        return conn.execute(
            "SELECT * FROM leads WHERE schedule_status = 'queued' AND campaign = ? ORDER BY created_at",
            (campaign,),
        ).fetchall()
    return conn.execute(
        "SELECT * FROM leads WHERE schedule_status = 'queued' ORDER BY created_at"
    ).fetchall()


def set_lead_status(conn: sqlite3.Connection, email: str, status: str) -> None:
    conn.execute(
        "UPDATE leads SET schedule_status = ?, last_attempt_at = ? WHERE email = ?",
        (status, datetime.now(timezone.utc).isoformat(), email.strip().lower()),
    )
    conn.commit()


# ── send history / suppression ────────────────────────────────────────────────
def is_unsubscribed(conn: sqlite3.Connection, email: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM unsubscribes WHERE email = ?", (email.strip().lower(),)
    ).fetchone()
    return row is not None


def mark_unsubscribed(conn: sqlite3.Connection, email: str, reason: str = "") -> None:
    conn.execute(
        "INSERT OR REPLACE INTO unsubscribes(email, unsubscribed_at, reason) VALUES (?, ?, ?)",
        (email.strip().lower(), datetime.now(timezone.utc).isoformat(), reason),
    )
    conn.commit()


def record_send(conn: sqlite3.Connection, *, email: str, status: str, subject: str = "",
                 detail: str = "", campaign: str = "", message_id: str = "") -> None:
    conn.execute(
        """INSERT INTO sends(email, status, subject, detail, campaign, sent_at, message_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(email, campaign) DO UPDATE SET
             status=excluded.status, subject=excluded.subject, detail=excluded.detail,
             sent_at=excluded.sent_at, message_id=excluded.message_id""",
        (email.strip().lower(), status, subject, detail, campaign,
         datetime.now(timezone.utc).isoformat(), message_id or None),
    )
    conn.commit()


def send_status(conn: sqlite3.Connection, email: str, campaign: str = "") -> str | None:
    row = conn.execute(
        "SELECT status FROM sends WHERE email = ? AND campaign = ?",
        (email.strip().lower(), campaign),
    ).fetchone()
    return row["status"] if row else None


def send_history(conn: sqlite3.Connection, limit: int = 500) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM sends ORDER BY sent_at DESC LIMIT ?", (limit,)
    ).fetchall()


def send_counts(conn: sqlite3.Connection) -> dict[str, int]:
    rows = conn.execute("SELECT status, COUNT(*) AS n FROM sends GROUP BY status").fetchall()
    return {r["status"]: r["n"] for r in rows}


if __name__ == "__main__":
    c = get_conn()
    init_db(c)
    print(f"Initialized {DB_PATH}")


# ── company_intel ─────────────────────────────────────────────────────────────
def get_intel(conn: sqlite3.Connection, domain: str) -> dict | None:
    """Cached company facts for a domain, or None when absent or expired.

    Expiry is checked on read rather than swept on a schedule: a stale row is simply
    treated as a miss, so a refresh happens the next time that company is touched.
    """
    if not domain:
        return None
    row = conn.execute("SELECT * FROM company_intel WHERE domain = ?",
                       (domain.lower(),)).fetchone()
    if row is None:
        return None
    exp = row["expires_at"]
    if exp and exp < _now():
        return None
    out = dict(row)
    for k in ("industries", "competitors", "product_terms"):
        try:
            out[k] = json.loads(out[k]) if out[k] else []
        except (json.JSONDecodeError, TypeError):
            out[k] = []
    return out


def put_intel(conn: sqlite3.Connection, *, domain: str, company: str = "",
              value_prop: str = "", industries: list | None = None,
              competitors: list | None = None, product_terms: list | None = None,
              raw_json: str = "", source: str = "homepage",
              ttl_days: int = 90) -> None:
    """Upsert one domain's facts. `source` records where they came from, because a
    curated row must never be silently overwritten by a weaker automatic one."""
    now = datetime.now(timezone.utc)
    conn.execute(
        """INSERT INTO company_intel
             (domain, company, value_prop, industries, competitors, product_terms,
              raw_json, source, fetched_at, expires_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(domain) DO UPDATE SET
             company=excluded.company, value_prop=excluded.value_prop,
             industries=excluded.industries, competitors=excluded.competitors,
             product_terms=excluded.product_terms, raw_json=excluded.raw_json,
             source=excluded.source, fetched_at=excluded.fetched_at,
             expires_at=excluded.expires_at""",
        (domain.lower(), company, value_prop,
         json.dumps(industries or [], ensure_ascii=False),
         json.dumps(competitors or [], ensure_ascii=False),
         json.dumps(product_terms or [], ensure_ascii=False),
         raw_json, source, now.isoformat(),
         (now + timedelta(days=ttl_days)).isoformat()))
    conn.commit()


def stale_domains(conn: sqlite3.Connection, domains: list[str]) -> list[str]:
    """Which of these domains have no usable cached intel. The enrichment stage only
    needs to work on these."""
    return [d for d in dict.fromkeys(x.lower() for x in domains if x)
            if get_intel(conn, d) is None]


# ── briefs ────────────────────────────────────────────────────────────────────
def _validation_json(validation) -> str:
    """Serialise a validation verdict, accepting either a dict or JSON text.

    Callers pass both. Running json.dumps over a string double-encodes it, so the
    value read back is a str rather than a dict -- and `sendable_briefs` then raises
    AttributeError and takes down the whole batch's send step.
    """
    if validation is None:
        return "{}"
    if isinstance(validation, str):
        try:
            json.loads(validation)          # already JSON: store it as-is
            return validation
        except json.JSONDecodeError:
            return "{}"                     # unparseable: a neutral verdict, not a crash
    return json.dumps(validation, ensure_ascii=False)


def save_brief(conn: sqlite3.Connection, *, email: str, company: str = "",
               kind: str = "signal", topic: str = "", html: str = "",
               status: str = "generated", validation: dict | None = None,
               llm_calls: int = 0, gen_seconds: float = 0.0,
               run_id: int | None = None, profile_json: str = "") -> int:
    """Store a generated brief and return its id.

    Briefs are append-only: a regenerate writes a new row rather than replacing the
    old one, so what was actually sent stays reconstructable even after the pipeline
    changes.

    `profile_json` is the tenant.Profile that produced this brief (dataclasses.asdict,
    then json.dumps) -- saving it here means any later validation pass, even in a
    separate process/run, can reload the exact profile without a second LLM call to
    re-derive it. Optional so existing callers (send_one.py, run_all40.py) keep
    working unchanged.
    """
    cur = conn.execute(
        """INSERT INTO briefs
             (email, company, kind, topic, html, status, validation,
              llm_calls, gen_seconds, run_id, created_at, profile_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (email, company, kind, topic, html, status,
         _validation_json(validation),
         llm_calls, gen_seconds, run_id, _now(), profile_json))
    conn.commit()
    return int(cur.lastrowid)


def set_brief_status(conn: sqlite3.Connection, brief_id: int, status: str) -> None:
    conn.execute("UPDATE briefs SET status = ? WHERE id = ?", (status, brief_id))
    conn.commit()


def get_brief(conn: sqlite3.Connection, brief_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM briefs WHERE id = ?", (brief_id,)).fetchone()


def latest_brief(conn: sqlite3.Connection, email: str,
                 kind: str = "") -> sqlite3.Row | None:
    """Most recent brief for a lead, so a send never regenerates what already exists."""
    if kind:
        return conn.execute(
            "SELECT * FROM briefs WHERE email=? AND kind=? ORDER BY id DESC LIMIT 1",
            (email, kind)).fetchone()
    return conn.execute(
        "SELECT * FROM briefs WHERE email=? ORDER BY id DESC LIMIT 1", (email,)).fetchone()


def list_briefs(conn: sqlite3.Connection, *, status: str = "", run_id: int | None = None,
                limit: int = 200) -> list[sqlite3.Row]:
    q = "SELECT id, email, company, kind, topic, status, validation, gen_seconds, created_at FROM briefs"
    where, args = [], []
    if status:
        where.append("status = ?"); args.append(status)
    if run_id is not None:
        where.append("run_id = ?"); args.append(run_id)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    return conn.execute(q, args).fetchall()


def sendable_briefs(conn: sqlite3.Connection, run_id: int | None = None) -> list[sqlite3.Row]:
    """Briefs clear to send: generated or validated, and carrying no hard failure.

    The `fails` check is done here rather than in SQL because validation is stored as
    JSON -- and because the gate belongs next to the definition of 'sendable', not
    scattered across callers.
    """
    rows = list_briefs(conn, run_id=run_id, limit=10000)
    out = []
    for r in rows:
        if r["status"] not in ("generated", "validated"):
            continue
        try:
            v = json.loads(r["validation"] or "{}")
        except json.JSONDecodeError:
            v = {}
        if not isinstance(v, dict):
            v = {}
        if v.get("fails"):
            continue
        out.append(r)
    return out


# ── runs ──────────────────────────────────────────────────────────────────────
def start_run(conn: sqlite3.Connection, *, workers: int = 0, kind: str = "",
              endpoint: str = "", notes: str = "") -> int:
    cur = conn.execute(
        """INSERT INTO runs (started_at, workers, kind, endpoint, notes)
           VALUES (?,?,?,?,?)""", (_now(), workers, kind, endpoint, notes))
    conn.commit()
    return int(cur.lastrowid)


def finish_run(conn: sqlite3.Connection, run_id: int, *, n_briefs: int = 0,
               n_pass: int = 0, n_warn: int = 0, n_fail: int = 0,
               wall_secs: float = 0.0, cost_inr: float = 0.0) -> None:
    conn.execute(
        """UPDATE runs SET finished_at=?, n_briefs=?, n_pass=?, n_warn=?, n_fail=?,
                           wall_secs=?, cost_inr=? WHERE id=?""",
        (_now(), n_briefs, n_pass, n_warn, n_fail, wall_secs, cost_inr, run_id))
    conn.commit()


def list_runs(conn: sqlite3.Connection, limit: int = 50) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()


def brief_subject(company: str, kind: str = "", topic: str = "") -> str:
    """The subject line for a brief. One definition, because pipeline.py, api.py and
    the dashboard all send, and three copies drift.

    A report brief names the market the lead enquired about: "FamPay's Brief" throws
    away the single most relevant word in the email.
    """
    company = (company or "Your Company").strip()
    topic = (topic or "").strip()
    if kind == "report" and topic:
        return f"Coherent Market Insights · {company} · {topic}"
    return f"Coherent Market Insights · {company}'s Brief"
