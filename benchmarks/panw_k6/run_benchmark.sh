#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Automated PANW k6 Benchmark Runner for Jina Embeddings v2 on TPU v5e
# ─────────────────────────────────────────────────────────────────────────────

POD_NAME=$(kubectl get pods -l app=jina-embeddings-v2 -o jsonpath={.items[0].metadata.name})
echo "Found active Jina TPU pod: $POD_NAME"

echo "=== [1/3] Copying benchmark scripts to Pod ==="
kubectl cp k6_ray_serve_test.js "$POD_NAME:/workspace/k6_ray_serve_test.js"
kubectl cp analyze_k6_results.py "$POD_NAME:/workspace/analyze_k6_results.py"
kubectl exec "$POD_NAME" -- mkdir -p /workspace/results

echo "=== [2/3] Running 20 RPS Benchmark (Fixed Payloads: 1K, 2K, 5K, 7K) ==="
kubectl exec "$POD_NAME" -- bash -c "cd /workspace && rm -f /workspace/results/raw.ndjson && k6 run --out json=/workspace/results/raw.ndjson -e SUITE=payload_size -e ENDPOINT=prompt_c2 -e STAGE_DURATION=30s -e PAYLOAD_RPS=20 -e PAYLOAD_VUS=40 k6_ray_serve_test.js"

echo "=== [3/3] Running 40 RPS Stress Benchmark ==="
kubectl exec "$POD_NAME" -- bash -c "cd /workspace && rm -f /workspace/results/raw_40rps.ndjson && k6 run --out json=/workspace/results/raw_40rps.ndjson -e SUITE=payload_size -e ENDPOINT=prompt_c2 -e STAGE_DURATION=20s -e PAYLOAD_RPS=40 -e PAYLOAD_VUS=80 k6_ray_serve_test.js"

echo "=== Analyzing Results inside Pod ==="
kubectl exec "$POD_NAME" -- python3 /workspace/analyze_k6_results.py /workspace/results/raw.ndjson
kubectl exec "$POD_NAME" -- python3 /workspace/analyze_k6_results.py /workspace/results/raw_40rps.ndjson

echo "=== Downloading results to local results/ directory ==="
mkdir -p ../results
kubectl cp "$POD_NAME:/workspace/results" ../results/

echo "Benchmark Complete! Summary generated in ../results/."
