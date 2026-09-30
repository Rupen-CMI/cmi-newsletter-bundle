for i in $(seq 1 45); do
  if curl -s -m 5 http://localhost:6006/v1/models 2>/dev/null | grep -q cmi-7b; then
    echo VLLM-READY
    grep -aoE "GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9]+ tokens per request: [0-9.]+x" /home/vllm.log | tail -2
    exit 0; fi
  if grep -aqE "Traceback|Error|CUDA out of memory" /home/vllm.log 2>/dev/null; then echo FAILED; grep -aE "Error|error" /home/vllm.log | tail -6; exit 1; fi
  sleep 15
done
echo TIMEOUT; tail -4 /home/vllm.log
