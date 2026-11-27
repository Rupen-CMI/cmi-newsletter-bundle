# JarvisLabs GPU instance: spawn, connect, start/stop

Covers only creating the instance and getting a shell on it. Model setup and
serving are in `.claude/skills/brief-generation-campaign/SKILL.md` Step 5.

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
