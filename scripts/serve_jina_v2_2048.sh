#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Runbook & Launcher Script for serving jinaai/jina-embeddings-v2-small-en
# with TPU v6e 4-Layer Fused FP32 Megakernel & max-model-len = 2048.
# Also starts the megakernel_proxy.py adapter with truncate_prompt_tokens = 2048.

set -euo pipefail

MODEL_ID="${MODEL_ID:-jinaai/jina-embeddings-v2-small-en}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-40}"
DTYPE="${DTYPE:-float32}"
PROXY_PORT="${PROXY_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8001}"

echo "=========================================================================="
echo " Starting Jina Embeddings v2 Setup & Serving Script (Max Len: ${MAX_MODEL_LEN})"
echo " Model ID:           ${MODEL_ID}"
echo " Max Model Len:      ${MAX_MODEL_LEN} (strictly capped at 2,048 tokens)"
echo " Max Batched Tokens: ${MAX_BATCHED_TOKENS}"
echo " Max Num Seqs:       ${MAX_NUM_SEQS}"
echo " Precision Dtype:    ${DTYPE}"
echo " Proxy Port:         0.0.0.0:${PROXY_PORT} (truncate_prompt_tokens=${MAX_MODEL_LEN})"
echo " vLLM Engine Port:   ${VLLM_HOST}:${VLLM_PORT}"
echo "=========================================================================="

# 1. Apply environment setup & shims
if [ -f "scripts/setup_jina_env.py" ]; then
    echo "[1/3] Running environment setup & shims..."
    python3 scripts/setup_jina_env.py
fi

# 2. Start Megakernel Adapter Proxy (with truncate_prompt_tokens=2048 & micro-batching)
if [ -f "models/JinaEmbedding/vLLM/megakernel/megakernel_proxy.py" ]; then
    echo "[2/3] Launching Megakernel Adapter Proxy on port ${PROXY_PORT} (truncate_prompt_tokens=${MAX_MODEL_LEN})..."
    VLLM_URL="http://${VLLM_HOST}:${VLLM_PORT}" \
    python3 models/JinaEmbedding/vLLM/megakernel/megakernel_proxy.py "${PROXY_PORT}" &
fi

# 3. Serve Model via vLLM Pooling Runner with --max-model-len 2048
echo "[3/3] Launching vllm serve on ${VLLM_HOST}:${VLLM_PORT} with --max-model-len ${MAX_MODEL_LEN}..."
exec vllm serve "${MODEL_ID}" \
    --runner pooling \
    --convert embed \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs="${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
    --dtype "${DTYPE}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    "$@"
