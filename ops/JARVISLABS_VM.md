# JarvisLabs GPU instance: spawn, connect, serve the model, start/stop

Sections 1–6 create the instance and get a shell on it. Sections 7–10 set up and
serve the production model (`cmi9b`). Running generation itself is in
`.claude/skills/brief-generation-campaign/SKILL.md` Step 6.

All commands run from the laptop in **Git Bash** (not PowerShell). Prefix with
`export PYTHONIOENCODING=utf-8;` if `jl` prints a Unicode error.

## 0. One-time prerequisites (already done on this laptop)

| What | Where / check |
|---|---|
| `jl` CLI installed and logged in | `jl status --json` prints the account (kashmir@coherentmarketinsights.com). If not: `jl login` (or `jl setup`). |
| SSH key pair | `~/.ssh/jarvis_new` (private) and `~/.ssh/jarvis_new.pub` |
| That key registered on the account as `cmi-newsletter-work` | `jl ssh-key list --json` shows a key ending `...ILactofEnEvbNyHheLT2P0VHjttXQHhM8qsGrhBIavsV`. If missing: `jl ssh-key add ~/.ssh/jarvis_new.pub --name cmi-newsletter-work` |

Keys are attached when an instance is **created**. Add the key before `jl create`,
or the new instance won't accept it.

## 1. Check balance

```bash
jl status --json        # read balance.balance (INR)
```

Rough cost: L4 / A30 spot ₹27.54/hr, on-demand ₹38.9–41.3/hr. Ask before
spawning if the balance looks short for the job.

## 2. Check free capacity

```bash
jl gpus --json
```

Look for `gpu_type` L4 or A30, `workload_type: "container"`, and
`num_free_devices > 0`.

## 3. Create the instance

Try in this order and stop at the first one that succeeds:

```bash
jl create --gpu L4 --storage 100 --yes --json           # L4 on-demand (preferred: can be paused safely)
jl create --gpu A30 --storage 100 --yes --json          # A30 on-demand
jl create --gpu L4 --spot --storage 100 --yes --json    # L4 spot (cheaper, can be preempted)
jl create --gpu A30 --spot --storage 100 --yes --json   # A30 spot
```

The skill lists spot first to save cost. Use on-demand when the instance should
survive a pause: a paused **spot** instance can be reclaimed and its disk lost.

From the JSON output, note:
- `machine_id`: used by every later `jl` command
- `public_ip`: used for ssh
- `endpoints`: public HTTPS URLs for the instance's ports. The one ending in
  `...1.notebooksn.jarvislabs.net` maps to port **6006**, which serves the model API.

Creation takes about 1–2 minutes; `status` should read `Running`.
To see the details again later:

```bash
jl list --json
jl get <machine_id> --json
```

## 4. Connect

```bash
K=~/.ssh/jarvis_new; IP=<public_ip>
ssh -i $K -o StrictHostKeyChecking=no root@$IP
```

Or, without the IP: `jl ssh <machine_id>`.

Run one command without a shell:
```bash
ssh -i $K -o StrictHostKeyChecking=no root@$IP 'nvidia-smi'
```

Copy files:
```bash
scp -i $K local_file root@$IP:/home/
scp -i $K root@$IP:/home/remote_file .
```

The `~/.ssh/config` entry for `*.notebooksn.jarvislabs.net` already uses this
key. For a raw IP that isn't in the config, keep the `-i ~/.ssh/jarvis_new` flag.

## 5. First checks after connecting

```bash
nvidia-smi                  # GPU visible; memory used should be ~0 MiB
df -h / /home               # /home is the persistent 100 GB disk
```

If `nvidia-smi` shows VRAM in use with no process of yours, the card is dirty.
Destroy the instance and create another.

**Disk rule:** only `/home` survives a pause. `/` (including pip packages
installed into the default conda env) is wiped on resume. Keep models, code and
logs under `/home`. Reinstall Python packages after every resume.

## 6. Stop, resume, destroy

```bash
jl pause   <machine_id> --yes --json    # stops billing for GPU; /home kept
jl resume  <machine_id> --yes --json    # same machine_id; the public IP can change, so re-read it
jl destroy <machine_id> --yes --json    # permanent; everything is lost
```

- Before pausing, delete lead data from the VM (DB copies, outputs). Keep code
  and models.
- After `resume`, read `public_ip` and `endpoints` again from the JSON. Then
  reinstall packages, since the root disk was wiped.
- Paused spot instances may vanish (`jl resume` → "Instance not found").
- Never leave a finished instance running: it bills by the hour while idle.

---

## 7. Set up the model (new instance: about 15 minutes)

**The model:** Qwen3.5-9B base + the fine-tuned **v4** LoRA adapter, merged into one
model, served FP8 under the name `cmi9b` with 16k context and thinking off.
Never use the old v1 `cmi-7b`.

**The order is: install the base model first, then add the adapter on top.**

1. **Download the base model** `Qwen/Qwen3.5-9B` from Hugging Face into `/home/qwen35base` (step 7b).
   This is the plain, untuned model (19 GB).
2. **Upload the v4 adapter** from the laptop into `/home/v4adapter` (step 7a). The adapter
   is not a full model; it holds only the fine-tuned changes, and it is useless without the base.
3. **Merge the adapter into the base** with `merge_v4.py` (step 7c). This writes one combined
   model to `/home/merged`. vLLM serves this merged copy, not the base and not the adapter.

Steps 1 and 2 can run in any order (7a and 7b below run together). Step 3 needs both done.

Files on the laptop (repo root):

| File | What it does |
|---|---|
| `training_data/v4_adapter/` | The v4 adapter (~340 MB) |
| `ops/merge_v4.py` | Merges `/home/v4adapter` into `/home/qwen35base` and writes `/home/merged` |
| `ops/serve_cmi9b.sh` | Starts vLLM on port 6006 serving `/home/merged` as `cmi9b` |

Run from the repo root in Git Bash:

```bash
K=~/.ssh/jarvis_new; IP=<public_ip>
S="ssh -i $K -o StrictHostKeyChecking=no root@$IP"

# 7a. upload the adapter and the two scripts
scp -i $K -r training_data/v4_adapter root@$IP:/home/v4adapter
scp -i $K ops/merge_v4.py ops/serve_cmi9b.sh root@$IP:/home/

# 7b. in parallel: install vLLM + PEFT, and download the base model (19 GB)
$S "nohup sh -c 'pip install vllm peft > /home/pip.log 2>&1; pip install -q \"huggingface_hub>=1.5,<2.0\" >> /home/pip.log 2>&1; echo PIPDONE >> /home/pip.log' >/dev/null 2>&1 &"
$S "cd /home && HF_HUB_DISABLE_XET=1 setsid nohup hf download Qwen/Qwen3.5-9B --local-dir /home/qwen35base > /home/dl_base.log 2>&1 < /dev/null &"
```

Wait until both are done (about 3–6 minutes):

```bash
$S "grep PIPDONE /home/pip.log; grep Downloaded /home/dl_base.log"   # both lines must print
```

The `huggingface_hub<2.0` pin must come **after** the installs. Version 2.x breaks
the transformers import ("huggingface-hub>=1.5.0,<2.0 is required").

```bash
# 7c. merge the adapter into the base (about 3–5 minutes, runs on CPU)
$S "cd /home && setsid nohup python merge_v4.py > /home/merge_v4.log 2>&1 < /dev/null &"
$S "cat /home/merge_v4.log"
```

Done when the log shows **both**:
- `lora modules attached: 400`. Any other number means the adapter didn't load; stop and check.
- `merged -> /home/merged`

## 8. Serve the model

```bash
$S "bash /home/serve_cmi9b.sh"
```

The script runs vLLM in the background with these settings: `--quantization fp8`,
`--served-model-name cmi9b`, `--max-model-len 16384`, `--gpu-memory-utilization 0.90`,
`--max-num-seqs 32`, thinking off, port `6006`. The log is `/home/vllm.log`.

Wait for startup (1–3 minutes):

```bash
$S "grep -c 'Application startup complete' /home/vllm.log"   # 1 = ready
```

- Lines starting `[rank0]:W ... Cubin file saved by TritonBundler not found` are
  **warnings** from an old compile cache. Ignore them.
- A real failure stops the vLLM process. Check with
  `$S "pgrep -f '[v]llm.entrypoints' || tail -30 /home/vllm.log"`.

**Context must be 16384, not 4096.** Long profile and writer prompts overflow 4k
and fail with HTTP 400 "maximum context length".

## 9. Check the server and point the laptop at it

On the instance:

```bash
$S "curl -s localhost:6006/v1/models"
# must show "id":"cmi9b" and "max_model_len":16384

$S "curl -s localhost:6006/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{\"model\":\"cmi9b\",\"messages\":[{\"role\":\"user\",\"content\":\"Capital of France in one word?\"}],\"max_tokens\":10}'"
# content must be "Paris", with no <think> text
```

From the laptop, use the public endpoint (the `endpoints` URL ending `...1.notebooksn.jarvislabs.net`):

```bash
curl -s https://<endpoint-host>/v1/models
```

Then set the repo's `.env`. Edit only these two lines and never print the file;
it holds secrets:

```bash
sed -i 's|^LOCAL_LLM_BASE_URL=.*|LOCAL_LLM_BASE_URL=https://<endpoint-host>/v1|; s|^LOCAL_LLM_MODEL=.*|LOCAL_LLM_MODEL=cmi9b|' .env
grep -E "^LOCAL_LLM_(BASE_URL|MODEL)=" .env
```

The endpoint sits behind Cloudflare, which cuts requests that are silent for about
100 s. `llm.py` streams responses to avoid this (`LOCAL_LLM_STREAM=0` turns it off;
don't). It retries 520/522/524 errors twice.

## 10. Restart, and after a resume

**Restart the server on the same instance:**

```bash
$S "pkill -9 -f '[v]llm.entrypoints'; pkill -9 -f '[E]ngineCore'"
$S "nvidia-smi --query-gpu=memory.used --format=csv"   # wait until < 500 MiB
$S "bash /home/serve_cmi9b.sh"
```

Use the bracket patterns exactly. A plain `pkill -f vllm` also matches and kills
your own ssh command. Starting vLLM before GPU memory is freed fails with "Free
memory on device ... less than desired GPU memory utilization".

**After `jl resume`:** `/home` (adapter, base, `/home/merged`, scripts) is still there.
The root disk is wiped, so vLLM is gone. Skip the download and merge:

```bash
# re-read public_ip and endpoints from: jl get <machine_id> --json
$S "nohup sh -c 'pip install vllm > /home/pip.log 2>&1; pip install -q \"huggingface_hub>=1.5,<2.0\" >> /home/pip.log 2>&1; echo PIPDONE >> /home/pip.log; bash /home/serve_cmi9b.sh' >/dev/null 2>&1 &"
# about 5 minutes for pip, then 1–3 minutes for startup; check as in sections 8 and 9
```

If the endpoint host changed, update `.env` (section 9).
