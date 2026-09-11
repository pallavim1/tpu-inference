#!/usr/bin/env bash
# Step 6: Export Results & Excel Sheets
set -euo pipefail

OUTPUT_EXCEL="${OUTPUT_EXCEL:-./jina_embeddings_v2_tpu_v5e_benchmarks.xlsx}"

echo "=========================================================================="
echo " 6. Exporting Benchmark Telemetry & Excel Reports"
echo "=========================================================================="

kubectl exec cpu-benchmark-runner -- python3 /workspace/generate_consolidated_excel.py 2>/dev/null || true
kubectl cp cpu-benchmark-runner:/workspace/jina_embeddings_v2_tpu_v5e_benchmarks.xlsx "${OUTPUT_EXCEL}" 2>/dev/null || true

echo "✅ Results exported to ${OUTPUT_EXCEL}!"
