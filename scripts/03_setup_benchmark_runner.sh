#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 03_setup_benchmark_runner.sh
# Sets up the client benchmark runner pod on the dedicated CPU Node Pool
# ==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." 2>/dev/null || pwd)"
DEPLOY_DIR="${BASE_DIR}/deploy"
BENCH_DIR="${BASE_DIR}/benchmarks"

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}  Step 3: Setting Up Benchmark Runner on Dedicated CPU Node Pool              ${NC}"
echo -e "${BLUE}==============================================================================${NC}"

# 1. Deploy Pod
echo -e "\n${YELLOW}[1/4] Deploying cpu-benchmark-runner pod...${NC}"
kubectl apply -f "${DEPLOY_DIR}/cpu_benchmark_runner.yaml"
kubectl wait --for=condition=ready pod/cpu-benchmark-runner --timeout=180s
echo -e "${GREEN}✓ cpu-benchmark-runner pod is Ready.${NC}"

# 2. Install Dependencies
echo -e "\n${YELLOW}[2/4] Installing k6 load generator, Python dependencies, and utilities...${NC}"
kubectl exec cpu-benchmark-runner -- bash -c "
  apt-get update -qq && apt-get install -y -qq gnupg curl ca-certificates git python3-pip
  
  # Install k6
  if ! command -v k6 &>/dev/null; then
    gpg -k 2>/dev/null || true
    gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D7AAB9732172C82409E5
    echo 'deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main' | tee /etc/apt/sources.list.d/k6.list
    apt-get update -qq && apt-get install -y -qq k6
  fi
  
  # Install Python packages
  pip install --upgrade pip -q
  pip install openpyxl pandas -q
  mkdir -p /workspace/results /workspace/benchmark_runs
"
echo -e "${GREEN}✓ Dependencies installed (k6 $(kubectl exec cpu-benchmark-runner -- k6 version | head -n 1)).${NC}"

# 3. Copy Test Scripts
echo -e "\n${YELLOW}[3/4] Copying benchmark suites and telemetry scripts into runner pod...${NC}"
for file in "${BENCH_DIR}"/*; do
  if [ -f "$file" ]; then
    kubectl cp "$file" "cpu-benchmark-runner:/workspace/$(basename "$file")"
  fi
done
echo -e "${GREEN}✓ Benchmark scripts synced to /workspace on runner pod.${NC}"

# 4. Network Health Check
echo -e "\n${YELLOW}[4/4] Testing internal GKE network connectivity from CPU runner to TPU pod...${NC}"
RESP=$(kubectl exec cpu-benchmark-runner -- curl -s -X POST http://jina-embedding-service:8000/prompt_c2 \
  -H "Content-Type: application/json" \
  -d '{"text": "Network Connectivity Verification"}' || true)

if echo "$RESP" | grep -q "embedding"; then
    echo -e "${GREEN}✓ Internal network connectivity VERIFIED! Round-trip inference successful.${NC}"
else
    echo -e "${RED}✗ Warning: Inference request failed: ${RESP}${NC}"
fi

echo -e "\n${BLUE}==============================================================================${NC}"
echo -e "${GREEN}✓ Benchmark Runner is ready for execution!${NC}"
echo -e "${BLUE}  Next options:${NC}"
echo -e "${BLUE}    - Run baseline (up to 40 RPS):   ./scripts/04_run_baseline_benchmarks.sh${NC}"
echo -e "${BLUE}    - Run saturation suite:          ./scripts/05_run_saturation_benchmarks.sh${NC}"
echo -e "${BLUE}==============================================================================${NC}"
