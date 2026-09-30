#!/usr/bin/env bash
# Install vLLM (a pause wipes site-packages) and serve the merged model.
# Runs fully detached: every step happens inside tmux, so an SSH drop or a
# `jl exec` timeout cannot kill it halfway. Idempotent -- safe to re-run.
set -u
LOG=/home/boot_serve.log
exec >> "$LOG" 2>&1
echo "=== boot $(date) ==="

if ! pip show vllm > /dev/null 2>&1; then
  echo "installing vllm..."
  pip install vllm
fi
echo "vllm: $(pip show vllm 2>/dev/null | sed -n 2p)"

if tmux has-session -t vllm 2>/dev/null; then
  echo "tmux session already running"
else
  tmux new-session -d -s vllm \
    "cd /home && python -m vllm.entrypoints.openai.api_server \
      --model /home/merged --served-model-name cmi-7b \
      --max-model-len 4096 --gpu-memory-utilization 0.94 --max-num-seqs 64 \
      --enable-prefix-caching --enable-chunked-prefill --port 6006 \
      > /home/vllm.log 2>&1"
  echo "tmux session started"
fi
