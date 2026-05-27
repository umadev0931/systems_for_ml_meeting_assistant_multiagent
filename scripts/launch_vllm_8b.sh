#!/usr/bin/env bash
# Launch the single Llama-3.1-8B vLLM OpenAI-compatible server on :8001.
#
# Prefix caching is enabled so that the shared transcript prefix sent by the
# three workers is reused (and reported as usage.prompt_tokens_details.cached_
# tokens, and via the /metrics prefix-cache counters). Prometheus /metrics is
# served at http://localhost:8001/metrics.
#
# Target hardware: Cloud TPU v5e single-host via the tpu-inference plugin.
set -euo pipefail

MODEL="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8001}"

# TPU backend (tpu-inference plugin). On GPU, drop VLLM_TARGET_DEVICE.
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-tpu}"

exec vllm serve "${MODEL}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --tensor-parallel-size "${TP_SIZE:-4}" \
  --enable-prefix-caching \
  --gpu-memory-utilization "${MEM_UTIL:-0.90}"

# Notes:
#   * --enable-prefix-caching is required to observe KV prefix-cache reuse.
#   * Scrape http://localhost:${PORT}/metrics for KV-cache + queue state.
#   * For a 70B comparison you would launch a second server with TP=4/8 on a
#     different port; this experiment intentionally uses 8B only.
