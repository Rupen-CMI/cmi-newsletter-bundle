"""CMI campaign dashboard — send a single test brief, upload a contact list and
batch-send, see settings and history.

    streamlit run campaign_dashboard.py

Sending goes through mailer.py (SMTP). Its config comes from .env:
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD   required
  SENDER_EMAIL, SENDER_NAME                         optional

Non-secret settings (Reply-To list) are stored in cmi_newsletter.db and editable
below. Sent/failed/unsubscribed state also persists there, so re-running the same
file is safe — already-sent rows (per campaign label) are skipped automatically.
"""
from __future__ import annotations

import os
import queue
import tempfile
import threading
import time

import streamlit as st

import campaign
import db as dbmod
import mailer
import send_one

st.set_page_config(page_title="CMI · Campaign Dashboard", page_icon="📤", layout="wide")

st.markdown("""
<style>
  html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: "Comic Sans MS", "Comic Sans", cursive !important;
  }
  .stApp { background-color: #FFFFFF !important; color: #000000 !important; }
  h1, h2, h3, h4, h5, h6, p, span, label, div { color: #000000 !important; }
  [data-testid="stHeader"] { background-color: #FFFFFF !important; }
  [data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 2px solid #000000 !important; }
  .stButton > button, .stButton > button * {
    background-color: #FFFFFF !important; color: #000000 !important;
    border: 2px solid #000000 !important; border-radius: 6px !important;
    font-family: "Comic Sans MS", cursive !important; font-weight: bold !important;
  }
  .stButton > button:hover,
  .stButton > button:hover *,
  .stButton > button:focus,
  .stButton > button:focus *,
  .stButton > button:active,
  .stButton > button:active * {
    background-color: #EEEEEE !important; color: #000000 !important;
    border-color: #000000 !important;
  }
  input, textarea, select, .stTextInput input, .stNumberInput input {
    background-color: #FFFFFF !important; color: #000000 !important;
    border: 2px solid #000000 !important;
  }
  [data-testid="stMetricValue"], [data-testid="stMetricLabel"] { color: #000000 !important; }
  [data-testid="stDataFrame"] { border: 2px solid #000000 !important; }
  hr, [data-testid="stDivider"] { border-color: #000000 !important; }
  .stAlert { border: 2px solid #000000 !important; background-color: #FFFFFF !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 style='margin-bottom:2px'>Coherent Market Insights · Campaign Dashboard</h1>"
    "<p style='color:#6C6A62;margin-top:2px'>Send a brief, upload a contact list, track delivery history.</p>",
    unsafe_allow_html=True,
)

# Fetched per rerun rather than held at module scope: Streamlit reruns on a different
# thread each time, and a sqlite3 connection is bound to its creating thread.
# dbmod.get_conn caches per thread, so this stays cheap.
conn = dbmod.get_conn()
dbmod.init_db(conn)

try:
    mailer._smtp_config()
    smtp_ok = True
except RuntimeError as e:
    smtp_ok = False
    st.error(str(e), icon="🚫")

st.divider()
st.subheader("⚙️ Settings")
reply_to_current = dbmod.get_setting(conn, "reply_to", ", ".join(mailer.DEFAULT_REPLY_TO))
reply_to_input = st.text_input("Reply-To addresses (comma-separated)", value=reply_to_current)
if st.button("Save settings"):
    dbmod.set_setting(conn, "reply_to", reply_to_input)
    st.success("Saved.")

if smtp_ok:
    cfg = mailer._smtp_config()
    st.caption(f"Sending from **{cfg['from_name']} <{cfg['from_email']}>** via **{cfg['host']}:{cfg['port']}**")

st.divider()

st.subheader("✉️ Send one test email")
t1, t2, t3 = st.columns([2, 2, 2])
test_email = t1.text_input("Send test to", placeholder="you@example.com")
test_name = t2.text_input("Recipient first name", value="there")
test_company = t3.text_input("Client company (blank = CMI default profile)", placeholder="ImmunoACT")
t4, t5 = st.columns([2, 2])
test_market = t4.text_input("Market", value="Global")
test_kind = t5.selectbox("Brief type", ["signal", "report"])
test_report = ""
if test_kind == "report":
    test_report = st.text_input("Report / market topic *", placeholder="Bone Marrow Transplant Market")

if st.button("Send test email", disabled=not test_email or not smtp_ok):
    with st.spinner("Generating a live brief and sending…"):
        try:
            send_one.send_one(email=test_email, company=test_company, recipient=test_name or "there",
                               market=test_market or "Global", kind=test_kind, report_name=test_report,
                               campaign_label="dashboard-test")
            st.success(f"Sent to {test_email}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Send failed: {type(e).__name__}: {e}")

st.divider()

st.subheader("🌍 Send to leads already in the DB, by country")
st.caption(
    "Leads imported via import_leads.py (or a previous xlsx upload below) live in "
    "cmi_newsletter.db. Pick a country to send only to queued leads from that market — "
    "useful for sending India today and leaving the rest for the scheduler."
)

db_markets = sorted({r["market"] for r in conn.execute(
    "SELECT DISTINCT market FROM leads WHERE schedule_status = 'queued'"
)})

if not db_markets:
    st.caption("No queued leads in the DB yet — import an xlsx with import_leads.py, or upload one below.")
else:
    dm1, dm2 = st.columns([2, 2])
    picked_market = dm1.selectbox("Country", db_markets)
    db_leads = conn.execute(
        "SELECT * FROM leads WHERE schedule_status = 'queued' AND market = ? ORDER BY created_at",
        (picked_market,),
    ).fetchall()
    dm2.metric(f"Queued leads in {picked_market}", len(db_leads))

    st.dataframe(
        [{"email": r["email"], "name": r["name"], "company": r["company"], "job_title": r["job_title"]}
         for r in db_leads],
        use_container_width=True, hide_index=True,
    )

    dc1, dc2, dc3 = st.columns([2, 2, 2])
    db_delay = dc1.number_input("Seconds between sends", min_value=0.0, value=12.0, step=1.0, key="db_delay")
    db_subject = dc2.text_input("Subject line", value="Coherent Market Insights · Your Brief", key="db_subject")
    db_kind = dc3.selectbox("Brief type", ["signal", "report"], key="db_kind")

    if "db_send_running" not in st.session_state:
        st.session_state.db_send_running = False

    if st.button(f"▶ Send now to all {len(db_leads)} queued {picked_market} leads",
                 type="primary", disabled=not db_leads or not smtp_ok or st.session_state.db_send_running):
        st.session_state.db_send_running = True
        progress_bar = st.progress(0.0)
        log_box = st.container()
        sent_n = failed_n = 0
        for idx, lead in enumerate(db_leads):
            try:
                send_one.send_one(
                    email=lead["email"], company=lead["company"] or "", recipient=lead["name"] or "there",
                    market=lead["market"] or "Global", kind=db_kind, report_name=lead["report_name"] or "",
                    subject=db_subject, campaign_label=lead["campaign"] or "default",
                )
                dbmod.set_lead_status(conn, lead["email"], "sent")
                sent_n += 1
                with log_box:
                    st.write(f"sent      — {lead['email']}")
            except Exception as e:  # noqa: BLE001
                dbmod.set_lead_status(conn, lead["email"], "failed")
                failed_n += 1
                with log_box:
                    st.write(f"failed    — {lead['email']} ({type(e).__name__}: {e})")
            progress_bar.progress((idx + 1) / len(db_leads))
            if idx < len(db_leads) - 1:
                time.sleep(db_delay)
        st.session_state.db_send_running = False
        st.success(f"Done — sent {sent_n}, failed {failed_n} out of {len(db_leads)} {picked_market} leads.")
        st.rerun()

st.divider()

st.subheader("📤 Batch campaign (upload a fresh xlsx)")
uploaded = st.file_uploader("Contact list (.xlsx)", type=["xlsx"])

if uploaded:
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    import openpyxl
    wb = openpyxl.load_workbook(tmp_path, read_only=True)
    sheet = st.selectbox("Sheet", wb.sheetnames)

    try:
        contacts = campaign.read_contacts(tmp_path, sheet=sheet)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    campaign_label = st.text_input("Campaign label", value=uploaded.name)

    already_sent = {r["email"] for r in conn.execute(
        "SELECT email FROM sends WHERE status='sent' AND campaign=?", (campaign_label,)
    )}
    already_unsub = {r["email"] for r in conn.execute("SELECT email FROM unsubscribes")}

    to_send = [c for c in contacts if c.email.lower() not in already_sent and c.email.lower() not in already_unsub]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows in sheet", len(contacts))
    m2.metric("Already sent (this campaign)", len(already_sent & {c.email.lower() for c in contacts}))
    m3.metric("Unsubscribed", len(already_unsub & {c.email.lower() for c in contacts}))
    m4.metric("Will send now", len(to_send))

    st.dataframe(
        [{"email": c.email, "name": c.name, "company": c.company, "market": c.market} for c in contacts[:50]],
        use_container_width=True, hide_index=True,
    )
    if len(contacts) > 50:
        st.caption(f"Showing first 50 of {len(contacts)} rows.")

    st.divider()
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    test_limit = c1.number_input("Limit this run to first N contacts (0 = all)", min_value=0, value=min(3, len(to_send)))
    delay = c2.number_input("Seconds between sends", min_value=0.0, value=12.0, step=1.0)
    subject = c3.text_input("Subject line", value="Coherent Market Insights · Your Brief")
    kind = c4.selectbox("Brief type ", ["signal", "report"])

    if "campaign_thread" not in st.session_state:
        st.session_state.campaign_thread = None
        st.session_state.stop_event = None
        st.session_state.progress_queue = None

    running = st.session_state.campaign_thread is not None and st.session_state.campaign_thread.is_alive()

    b1, b2 = st.columns([1, 1])
    start_clicked = b1.button("▶ Start", type="primary", disabled=not to_send or running or not smtp_ok)
    stop_clicked = b2.button("⏹ Stop", disabled=not running)

    if start_clicked:
        stop_event = threading.Event()
        progress_q: queue.Queue = queue.Queue()
        limit = test_limit or None

        def _worker():
            def _on_progress(event):
                progress_q.put(event)
            result = campaign.run_campaign(
                tmp_path, sheet=sheet, limit=limit, delay_seconds=delay,
                subject=subject, kind=kind, campaign_label=campaign_label,
                on_progress=_on_progress, should_stop=stop_event.is_set,
            )
            progress_q.put({"email": "", "status": "__done__", "detail": result})

        t = threading.Thread(target=_worker, daemon=True)
        st.session_state.stop_event = stop_event
        st.session_state.progress_queue = progress_q
        st.session_state.campaign_thread = t
        t.start()
        st.rerun()

    if stop_clicked and st.session_state.stop_event:
        st.session_state.stop_event.set()
        st.info("Stopping after the current send finishes…")

    if running or (st.session_state.progress_queue and not st.session_state.progress_queue.empty()):
        limit = test_limit or None
        total = min(limit, len(to_send)) if limit else len(to_send)
        progress_bar = st.progress(0.0)
        log_box = st.container()
        done_count = 0
        final_result = None

        deadline = time.time() + 8
        while time.time() < deadline:
            try:
                event = st.session_state.progress_queue.get(timeout=0.5)
            except queue.Empty:
                if not (st.session_state.campaign_thread and st.session_state.campaign_thread.is_alive()):
                    break
                continue
            if event["status"] == "__done__":
                final_result = event["detail"]
                break
            done_count += 1
            progress_bar.progress(min(1.0, done_count / max(1, total)))
            with log_box:
                st.write(f"{event['status']:12} — {event['email']}" + (f" ({event.get('detail')})" if event.get("detail") else ""))

        if final_result is not None:
            st.session_state.campaign_thread = None
            stopped_note = " (stopped early)" if final_result.get("stopped") else ""
            st.success(f"Done{stopped_note} — sent {final_result['sent']}, failed {final_result['failed']}, "
                       f"skipped {final_result['skipped']}, unsubscribed {final_result['unsubscribed']}")
        else:
            st.rerun()

st.divider()
st.subheader("📊 Send history")
counts = dbmod.send_counts(conn)
rows = dbmod.send_history(conn)

h1, h2, h3, h4 = st.columns(4)
h1.metric("Total sent", counts.get("sent", 0))
h2.metric("Failed", counts.get("failed", 0))
h3.metric("Unsubscribed", counts.get("unsubscribed", 0))
h4.metric("Total tracked", sum(counts.values()))

if rows:
    st.dataframe(
        [{"email": r["email"], "status": r["status"], "campaign": r["campaign"],
          "detail": r["detail"], "when": r["sent_at"]} for r in rows],
        use_container_width=True, hide_index=True,
    )
else:
    st.caption("No sends recorded yet.")

conn.close()
