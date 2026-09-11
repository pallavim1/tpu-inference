#!/usr/bin/env bash
# Step 4: Run Baseline Sweeps (Up to 40 RPS)
set -euo pipefail

DURATION="${DURATION:-60s}"

echo "=========================================================================="
echo " 4. Running Baseline Benchmarks (Up to 40 RPS, Duration: ${DURATION})"
echo "=========================================================================="

kubectl exec cpu-benchmark-runner -- bash -c "
  cd /workspace && \
  k6 run --no-thresholds \
    --out json=/workspace/cpu_to_tpu_results/baseline_40rps.ndjson \
    -e SUITE=payload_size \
    -e ENDPOINT=prompt_c2 \
    -e HTTP_URL=http://jina-embedding-service:8000 \
    -e PAYLOAD_RPS=40 \
    -e STAGE_DURATION=${DURATION} \
    k6_ray_serve_test.js
"
echo "✅ Baseline benchmarks completed!"
