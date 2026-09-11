#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Runbook & Launcher Script for serving jinaai/jina-embeddings-v2-small-en
# with JAX ALiBi Flash-Attention kernels & max-model-len = 8192 on Cloud TPU.
#
# Equivalent script matching docs/models/jina_embeddings_v2.md configured for 8192 context length.

set -euo pipefail

MODEL_ID="${MODEL_ID:-jinaai/jina-embeddings-v2-small-en}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-8192}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "=========================================================================="
echo " Starting Jina Embeddings v2 Setup & Serving Script (Max Len: ${MAX_MODEL_LEN})"
echo " Model ID:           ${MODEL_ID}"
echo " Max Model Len:      ${MAX_MODEL_LEN}"
echo " Max Batched Tokens: ${MAX_BATCHED_TOKENS}"
echo " Host/Port:          ${HOST}:${PORT}"
echo "=========================================================================="

# 1. Apply environment setup & shims
if [ -f "scripts/setup_jina_env.py" ]; then
    echo "[1/4] Running environment setup & shims..."
    python3 scripts/setup_jina_env.py
fi

# 2. Handle HF_TOKEN if placeholder or invalid
if [ "${HF_TOKEN:-}" = "test" ]; then
    echo "[2/4] Unsetting invalid placeholder HF_TOKEN to allow public model download..."
    unset HF_TOKEN
fi

# 3. Optional validation test execution flag
if [ "${RUN_VALIDATION_TESTS:-false}" = "true" ]; then
    echo "[3/4] Running model parity & ALiBi kernel validation tests..."
    pytest tests/kernels/encoder_alibi_kernel_test.py -v
    pytest tests/models/jax/test_jina_bert.py -v -rs
    pytest tests/e2e/test_jina_embeddings.py -v -rs
else
    echo "[3/4] Skipping validation tests (set RUN_VALIDATION_TESTS=true to enable)."
fi

# 4. Serve Model via vLLM Pooling Runner with 8192 max model length
echo "[4/4] Launching vllm serve on ${HOST}:${PORT} with --max-model-len ${MAX_MODEL_LEN}..."
exec vllm serve "${MODEL_ID}" \
    --runner pooling \
    --convert embed \
    --trust-remote-code \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
    --dtype float32 \
    --host "${HOST}" \
    --port "${PORT}" \
    "$@"
