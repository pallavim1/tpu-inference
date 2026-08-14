#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 04_run_baseline_benchmarks.sh
# Runs PANW baseline benchmark sweeps (up to 40 RPS) from the CPU runner pod
# ==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

MODE="fixed"
RPS="20"
DURATION="60s"
PAYLOAD=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --rps|-r)
      RPS="$2"
      shift 2
      ;;
    --duration|-d)
      DURATION="$2"
      shift 2
      ;;
    --matrix|-m)
      MODE="matrix"
      shift
      ;;
    --payload|-p)
      PAYLOAD="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --rps <num>        Target RPS for fixed suite (default: 20)"
      echo "  --duration <time>  Stage duration (default: 60s)"
      echo "  --matrix           Run full 48-stage matrix (1, 5, 7, 10, 20, 30, 40 RPS x 4 sizes)"
      echo "  --payload <1k|2k|5k|7k>  Run single payload only"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

TS=$(date +%Y%m%d_%H%M%S)

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}  Running Baseline Benchmarks (Up to 40 RPS)                                  ${NC}"
echo -e "${BLUE}  Mode: ${MODE} | Target RPS: ${RPS} | Stage Duration: ${DURATION}             ${NC}"
echo -e "${BLUE}==============================================================================${NC}"

if [ "$MODE" == "matrix" ]; then
    echo -e "\n${YELLOW}Starting Full 48-Stage Baseline Matrix Sweep across 1K, 2K, 5K, 7K payloads...${NC}"
    kubectl exec -it cpu-benchmark-runner -- bash -c "
      cd /workspace && \
      k6 run --no-thresholds \
        --out json=/workspace/results/baseline_matrix_${TS}.ndjson \
        -e SUITE=rps_payload \
        -e ENDPOINT=prompt_c2 \
        -e HTTP_URL=http://jina-embedding-service:8000 \
        -e STAGE_DURATION=${DURATION} \
        k6_ray_serve_test.js
        
      python3 /workspace/analyze_k6_results.py /workspace/results/baseline_matrix_${TS}.ndjson --out /workspace/results/baseline_matrix_${TS}_summary
    "
else
    echo -e "\n${YELLOW}Running Fixed Payload Baseline Suite at ${RPS} RPS...${NC}"
    kubectl exec -it cpu-benchmark-runner -- bash -c "
      cd /workspace && \
      k6 run --no-thresholds \
        --out json=/workspace/results/baseline_${RPS}rps_${TS}.ndjson \
        -e SUITE=payload_size \
        -e ENDPOINT=prompt_c2 \
        -e HTTP_URL=http://jina-embedding-service:8000 \
        -e PAYLOAD_RPS=${RPS} \
        -e STAGE_DURATION=${DURATION} \
        k6_ray_serve_test.js
        
      python3 /workspace/analyze_k6_results.py /workspace/results/baseline_${RPS}rps_${TS}.ndjson --out /workspace/results/baseline_${RPS}rps_${TS}_summary
    "
fi

echo -e "\n${BLUE}==============================================================================${NC}"
echo -e "${GREEN}✓ Baseline benchmark sweep complete!${NC}"
echo -e "${BLUE}==============================================================================${NC}"
