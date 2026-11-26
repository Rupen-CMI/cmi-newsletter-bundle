cd /home
nohup python -m vllm.entrypoints.openai.api_server --model /home/merged --served-model-name cmi9b \
  --quantization fp8 --max-model-len 16384 --gpu-memory-utilization 0.90 --max-num-seqs 32 \
  --enable-prefix-caching --enable-chunked-prefill \
  --default-chat-template-kwargs '{"enable_thinking": false}' --port 6006 > /home/vllm.log 2>&1 &
