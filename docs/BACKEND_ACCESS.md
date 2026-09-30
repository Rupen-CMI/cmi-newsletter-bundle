# Connecting to the backend, and letting Claude Code drive it

How to reach the CMI API yourself, how to authenticate it, and how to give Claude Code
(or any agent) safe access to it.

**Read the security note first.** This API can send real email to real people. Until
you set a key it has *no authentication at all* and is safe only because it binds to
`127.0.0.1`.

---

## 1. Start it

```bash
python api.py                  # http://127.0.0.1:8080/docs
```

| var | default | meaning |
|---|---|---|
| `CMI_API_HOST` | `127.0.0.1` | loopback only. Changing this **requires** a key |
| `CMI_API_PORT` | `8080` | |
| `CMI_API_KEY` | *(unset)* | when set, every route except `/health` and the docs needs it |

Check it is alive — `/health` never needs a credential, which is what makes it usable
as a probe:

```bash
curl -s http://127.0.0.1:8080/health
# {"ok":true,"auth_required":false}
```

`auth_required: false` means anyone who can reach the port can send email as you.
On a laptop that is fine. Anywhere else, set a key.

---

## 2. Turn on authentication

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

```powershell
$env:CMI_API_KEY = "<paste>"      # PowerShell
python api.py
```
```bash
export CMI_API_KEY="<paste>"      # bash
python api.py
```

Put it in `.env` to make it stick. It is a credential — treat it like the SendGrid key,
and never paste it into a script, a markdown file, or a chat window.

Both header styles work, because different tools reach for different conventions:

```bash
curl -H "X-API-Key: $CMI_API_KEY"            http://127.0.0.1:8080/stats
curl -H "Authorization: Bearer $CMI_API_KEY" http://127.0.0.1:8080/stats
```

Verified behaviour:

| request | with no key configured | with a key configured |
|---|---|---|
| `GET /health` | 200 | 200 |
| `GET /docs` | 200 | 200 |
| `GET /stats` no header | 200 | **401** |
| `GET /stats` wrong key | 200 | **401** |
| `GET /stats` correct key | 200 | 200 |
| `POST /briefs/{id}/send` no key | (guarded by `confirm`) | **401** |

Two design points worth knowing:

- **Auth is middleware, not a per-route dependency.** A route added later is protected
  by default rather than silently public because someone forgot a decorator.
- **The key compare is constant-time** (`hmac.compare_digest`). A plain `==` leaks key
  length and prefix through timing.
- **Binding beyond loopback without a key is refused outright**, not warned about:

```
$ CMI_API_HOST=0.0.0.0 python api.py
refusing to bind 0.0.0.0 without CMI_API_KEY set.
```

---

## 3. The endpoints

| method | path | what | reversible? |
|---|---|---|---|
| GET | `/health` | liveness, no auth | yes |
| GET | `/vm/status` | is the model serving, machine id, GPU, whether `.env` matches | yes |
| POST | `/vm/reconnect` | resume/boot the VM, record the endpoint | yes, but **starts a billable GPU** |
| GET | `/analytics` | totals, verdicts, top failures, daily activity, throughput | yes |
| GET | `/stats` | bare counts | yes |
| GET | `/leads?campaign=&status=` | lead list | yes |
| POST | `/leads/import` | import an xlsx already on this machine | yes |
| GET | `/briefs?status=&run_id=` | brief list with fails/warns | yes |
| GET | `/briefs/{id}/html` | render one brief | yes |
| POST | `/briefs/{id}/send` | **sends a real email** | **NO** |
| GET | `/runs`, `/runs/{id}` | run history | yes |
| GET | `/intel`, `/intel/{domain}` | cached company facts | yes |

`POST /briefs/{id}/send` is the only irreversible one, and it is guarded four ways: the
API key, an explicit `{"confirm": true}`, a refusal on any hard validation failure, and
a refusal for unsubscribed recipients.

```bash
curl -X POST http://127.0.0.1:8080/briefs/17/send \
  -H "X-API-Key: $CMI_API_KEY" -H "Content-Type: application/json" \
  -d '{"confirm": true, "campaign": "manual"}'
```

Without `confirm` you get `400 — this sends a real email`.

---

## 4. Giving Claude Code access

Claude Code reaches your backend the same way you would: HTTP with the key in a header.
There is no Anthropic credential involved — **you issue the key, and you can revoke it.**

### The setup

1. Start the API with `CMI_API_KEY` set.
2. Put the key somewhere Claude Code can read at run time — `.env` is already the
   pattern here, and it is gitignored.
3. Tell Claude what the base URL is and that the key is in `.env`.

Then ask in plain language: *"check whether the VM is connected"*, *"show me the
analytics"*, *"list briefs that failed validation"*. Claude will run curl or a short
Python snippet against the API.

### Never paste the key into the conversation

Have Claude read it at run time instead. This keeps the secret out of the transcript:

```python
from dotenv import dotenv_values
import urllib.request, json

key = dotenv_values(".env")["CMI_API_KEY"]
req = urllib.request.Request("http://127.0.0.1:8080/analytics",
                             headers={"X-API-Key": key})
print(json.load(urllib.request.urlopen(req))["totals"])
```

This matters. Earlier in this project a SendGrid key was printed into tool output by a
masking routine that only redacted names containing KEY/PASSWORD/SECRET/TOKEN — the var
was named `Sendfrid_web_api` and slipped through. Anything printed to a terminal is in
the scrollback and the transcript. Read secrets, never echo them.

### What to let an agent do unsupervised

| | endpoints | why |
|---|---|---|
| **Safe** | `/health` `/vm/status` `/analytics` `/stats` `/leads` `/briefs` `/briefs/{id}/html` `/runs` `/intel` | read-only; worst case is a stale number |
| **Costs money** | `/vm/reconnect` | starts a GPU at ₹27–41/hr. Fine to allow, but **pause it afterwards**: `python vm_connect.py --pause`, or the ⏸ button on the dashboard |
| **Never unsupervised** | `POST /briefs/{id}/send` | irreversible. A wrong brief to a real prospect cannot be recalled |

The dashboard's **Send** panel is the human equivalent: dry-run by default, and it
requires typing `SEND` to arm. Prefer it over the API when a person is driving.

If you want an agent to have read-only access with no way to send, run a second
instance on another port behind a proxy that allows only `GET`. The application has one
key and does not do per-key scopes — do not pretend otherwise by convention alone.

### A read-only key is not a feature yet

There is deliberately no "read-only key" in the code. One key, all routes. If you need
scoped access, the honest options are a `GET`-only reverse proxy, or adding scopes
properly — not handing out the same key and hoping.

---

## 5. Exposing it beyond your laptop

The default is loopback because nothing here does user management, rate limiting, or
TLS. If you genuinely need remote access, in order of preference:

1. **SSH tunnel** — no exposure at all, nothing to misconfigure:
   ```bash
   ssh -L 8080:127.0.0.1:8080 user@host
   ```
2. **A reverse proxy** (Caddy/nginx) terminating TLS, with the API still bound to
   `127.0.0.1` behind it. Keep `CMI_API_KEY` set as well — defence in depth.
3. **`CMI_API_HOST=0.0.0.0` with a key.** Works, but it is plain HTTP: the key crosses
   the network in the clear. Only on a trusted private network.

Never put this on the public internet on plain HTTP. The key would be sniffable, and
the endpoint behind it sends email.

---

## 6. Quick recipes

```bash
# is the model up?
curl -s -H "X-API-Key: $CMI_API_KEY" localhost:8080/vm/status | jq '{serving,machine_id,vm_status}'

# bring it up if not (starts a GPU — pause it later)
curl -s -X POST -H "X-API-Key: $CMI_API_KEY" -H "Content-Type: application/json" \
     -d '{"spot": true}' localhost:8080/vm/reconnect

# what has been produced
curl -s -H "X-API-Key: $CMI_API_KEY" localhost:8080/analytics | jq '.totals, .verdicts'

# briefs blocked by validation
curl -s -H "X-API-Key: $CMI_API_KEY" localhost:8080/briefs | jq '[.[] | select(.fails | length > 0)]'

# read one
curl -s -H "X-API-Key: $CMI_API_KEY" localhost:8080/briefs/17/html > brief.html
```

---

## 7. Verifying the guards still hold

```bash
python -m pytest tests/test_units.py -k ApiAuth -q
```

Seven tests cover: open when unconfigured, 401 on missing and wrong keys, both header
styles accepted, `/health` staying open, and the send endpoint being guarded. They run
offline against a `TestClient` — no server, no GPU.
