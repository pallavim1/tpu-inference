#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_all_end_to_end.sh
# Master automated script to provision infrastructure, deploy TPU service,
# set up runner pod, and execute full saturation benchmarks.
# ==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}  Palo Alto Networks (PANW) - TPU v5e Automated End-to-End Benchmarking       ${NC}"
echo -e "${BLUE}==============================================================================${NC}"

# Step 1: Provision Infrastructure
echo -e "\n${YELLOW}>>> STEP 1: Provisioning GKE Cluster & Nodepools <<<${NC}"
"${SCRIPT_DIR}/01_provision_infrastructure.sh"

# Step 2: Deploy TPU Serving Workload
echo -e "\n${YELLOW}>>> STEP 2: Deploying Jina Embeddings vLLM Service on TPU v5e <<<${NC}"
"${SCRIPT_DIR}/02_deploy_tpu_workload.sh" --precision fp16

# Step 3: Setup Benchmark Runner Pod
echo -e "\n${YELLOW}>>> STEP 3: Setting Up CPU Benchmark Runner Pod <<<${NC}"
"${SCRIPT_DIR}/03_setup_benchmark_runner.sh"

# Step 4: Run Full Saturation Suite
echo -e "\n${YELLOW}>>> STEP 4: Executing Full Saturation Benchmark Suite <<<${NC}"
"${SCRIPT_DIR}/05_run_saturation_benchmarks.sh" --duration 60s --phase all

echo -e "\n${BLUE}==============================================================================${NC}"
echo -e "${GREEN}✓ All End-to-End Steps Successfully Completed!${NC}"
echo -e "${BLUE}==============================================================================${NC}"
