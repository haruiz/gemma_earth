vllm serve haruiz/gemmaearth \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image":1}' \
  --gpu-memory-utilization 0.90
