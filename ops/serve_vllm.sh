cd /home
pkill -9 -f vllm.entrypoints 2>/dev/null; sleep 3
# Aggressive batching for the 40-brief run.
#   --max-num-seqs 64   : double the earlier 32. The KV cache held 88k tokens at 0.92
#                         utilisation, which is ~21 concurrent 4k requests -- but our
#                         real prompts are far shorter than 4k, so more fit than that
#                         figure suggests. vLLM queues anything over the limit rather
#                         than failing, so setting it high costs nothing.
#   --gpu-memory-utilization 0.94 : whatever is left after weights IS the KV cache, and
#                         cache size is what decides concurrency. 0.94 is as far as is
#                         safe before CUDA graph capture starts failing.
#   --enable-prefix-caching : every brief sends the same voice.STYLE prefix on all 11
#                         calls; computed once instead of 11x per brief.
#   --enable-chunked-prefill : long prompts stop blocking short ones behind them, which
#                         matters when 40 briefs each fire a 2k-token prompt at once.
nohup python -m vllm.entrypoints.openai.api_server \
  --model /home/merged \
  --served-model-name cmi-7b \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.94 \
  --max-num-seqs 64 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --port 6006 > /home/vllm.log 2>&1 &
sleep 20
echo "vllm launched (max-num-seqs 64, util 0.94, chunked prefill)"
tail -2 /home/vllm.log
