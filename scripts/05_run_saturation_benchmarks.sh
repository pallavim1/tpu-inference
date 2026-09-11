#!/usr/bin/env bash
# Step 5: Run Full Saturation Benchmark Suite
set -euo pipefail

DURATION="${DURATION:-60s}"
PHASE="${PHASE:-all}"

echo "=========================================================================="
echo " 5. Running Full Saturation Benchmark Suite (Duration: ${DURATION}, Phase: ${PHASE})"
echo "=========================================================================="

kubectl exec cpu-benchmark-runner -- python3 /workspace/run_cpu_to_tpu_saturation.py --duration "${DURATION}" --phase "${PHASE}"
echo "✅ Full Saturation Benchmark Suite completed!"
