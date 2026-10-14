#!/bin/bash
# Starts the PANW /prompt_c2 HTTP adapter on :8000 and vLLM on :8001
# with the Opus JinaBert Megakernel v2 + Fused Device Pooling (commit d3439062) flags.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/jina_megakernel_logs}"
mkdir -p "${LOG_DIR}"

export USE_JINA_BERT_MEGAKERNEL=1
export JINA_BERT_MEGAKERNEL_VERSION=v2
export JINA_BERT_MEGAKERNEL_PRECISION=default
export TPU_POOLING_FAST_PATH=1
export JINA_BERT_DEVICE_POOLING=1

echo "[0/2] Applying idempotent JinaBert transformers v5 + vLLM ModelRegistry shims (setup_jina_env.py)..."
python3 "${SCRIPT_DIR}/setup_jina_env.py"

echo "[1/2] Starting /prompt_c2 adapter proxy on :8000 (logs: ${LOG_DIR}/proxy.log)..."
python3 "${SCRIPT_DIR}/stock_adapter_proxy.py" 8000 > "${LOG_DIR}/proxy.log" 2>&1 &
PROXY_PID=$!
trap "kill ${PROXY_PID} 2>/dev/null || true" EXIT

echo "[2/2] Starting vllm serve on :8001 (logs: ${LOG_DIR}/vllm_serve.log)..."
exec vllm serve jinaai/jina-embeddings-v2-small-en \
    --runner pooling \
    --convert embed \
    --trust-remote-code \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --dtype float32 \
    --host 0.0.0.0 \
    --port 8001 > "${LOG_DIR}/vllm_serve.log" 2>&1
