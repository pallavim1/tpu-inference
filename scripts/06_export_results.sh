#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 06_export_results.sh
# Downloads benchmark deliverables and Excel sheets from runner pod to workstation
# ==============================================================================

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

DEST_DIR="${1:-./downloaded_results}"
mkdir -p "${DEST_DIR}"

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}  Exporting Benchmark Deliverables from GKE Pod                               ${NC}"
echo -e "${BLUE}  Destination: ${DEST_DIR}                                                    ${NC}"
echo -e "${BLUE}==============================================================================${NC}"

# Download Excel workbook
echo -e "\n${YELLOW}[1/2] Fetching consolidated Excel workbook...${NC}"
kubectl cp cpu-benchmark-runner:/workspace/jina_embeddings_v2_tpu_v5e_benchmarks.xlsx "${DEST_DIR}/jina_embeddings_v2_tpu_v5e_benchmarks.xlsx" 2>/dev/null || true

# Download all run directories
echo -e "\n${YELLOW}[2/2] Fetching summary telemetry and reports...${NC}"
kubectl exec cpu-benchmark-runner -- bash -c "cd /workspace && tar -czf /tmp/all_summaries.tar.gz benchmark_runs/ results/ 2>/dev/null || true"
kubectl cp cpu-benchmark-runner:/tmp/all_summaries.tar.gz "${DEST_DIR}/all_summaries.tar.gz" 2>/dev/null || true
tar -xzf "${DEST_DIR}/all_summaries.tar.gz" -C "${DEST_DIR}" 2>/dev/null || true
rm -f "${DEST_DIR}/all_summaries.tar.gz"

echo -e "\n${GREEN}✓ Export complete! Files saved to: ${DEST_DIR}${NC}"
ls -lh "${DEST_DIR}"
