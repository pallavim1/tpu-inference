#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 05_run_saturation_benchmarks.sh
# Runs full saturation sweeps until saturation for each payload (1K, 2K, 5K, 7K)
# Automatically generates timestamped reports and pulls deliverables locally
# ==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

DURATION="60s"
PHASE="all"
TS="$(date +%Y%m%d_%H%M%S)"
LOCAL_RESULTS_DIR="./results/run_${TS}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --duration|-d)
      DURATION="$2"
      shift 2
      ;;
    --phase|-p)
      PHASE="$2"
      shift 2
      ;;
    --timestamp|-t)
      TS="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --duration <time>  Stage duration per load point (default: 60s)"
      echo "  --phase <all|multi|2k|1k>  Run specific phase (default: all)"
      echo "  --timestamp <str>  Custom timestamp identifier"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}  Running Live TPU Saturation Benchmark Suite                                  ${NC}"
echo -e "${BLUE}  Run Identifier: ${TS} | Phase: ${PHASE} | Duration: ${DURATION}            ${NC}"
echo -e "${BLUE}==============================================================================${NC}"

# 1. Launch Benchmark Runner on Pod
echo -e "\n${YELLOW}[1/3] Kicking off saturation suite on CPU benchmark runner pod...${NC}"
kubectl exec -it cpu-benchmark-runner -- python3 /workspace/run_timestamped_benchmark_suite.py \
  --timestamp "${TS}" \
  --duration "${DURATION}" \
  --phase "${PHASE}"

# 2. Package & Download Artifacts
echo -e "\n${YELLOW}[2/3] Extracting generated artifacts and reports to workstation...${NC}"
mkdir -p "${LOCAL_RESULTS_DIR}"

kubectl exec cpu-benchmark-runner -- bash -c "
  cd /workspace/benchmark_runs/run_${TS} && \
  tar -czf /workspace/summary_results_${TS}.tar.gz *.json *.xlsx *.md 2>/dev/null || true
"

kubectl cp "cpu-benchmark-runner:/workspace/summary_results_${TS}.tar.gz" "/tmp/summary_results_${TS}.tar.gz"
tar -xzf "/tmp/summary_results_${TS}.tar.gz" -C "${LOCAL_RESULTS_DIR}"

# Copy consolidated workbook
kubectl cp "cpu-benchmark-runner:/workspace/jina_embeddings_v2_tpu_v5e_benchmarks.xlsx" "${LOCAL_RESULTS_DIR}/jina_embeddings_v2_tpu_v5e_benchmarks_${TS}.xlsx" 2>/dev/null || true

echo -e "${GREEN}✓ Artifacts downloaded to: ${LOCAL_RESULTS_DIR}${NC}"
ls -lh "${LOCAL_RESULTS_DIR}"

echo -e "\n${BLUE}==============================================================================${NC}"
echo -e "${GREEN}✓ Saturation benchmark sweep complete for Run [${TS}]!${NC}"
echo -e "${BLUE}  Deliverables stored in: ${LOCAL_RESULTS_DIR}${NC}"
echo -e "${BLUE}==============================================================================${NC}"
