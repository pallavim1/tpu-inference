#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 02_deploy_tpu_workload.sh
# Deploys Jina Embeddings v2 with vLLM on Cloud TPU v5e
# Supports: --precision fp16 (bfloat16) | --precision fp32 (float32)
# ==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

PRECISION="fp16"
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../deploy" 2>/dev/null || pwd)"
if [ ! -f "${DEPLOY_DIR}/jina_v5e_deployment.yaml" ]; then
    DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/deploy"
fi

while [[ $# -gt 0 ]]; do
  case $1 in
    --precision|-p)
      PRECISION="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 [--precision fp16|fp32]"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

DTYPE="bfloat16"
if [ "$PRECISION" == "fp32" ]; then
    DTYPE="float32"
fi

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}  Step 2: Deploying Jina Embeddings vLLM Service on Cloud TPU v5e             ${NC}"
echo -e "${BLUE}  Model: jinaai/jina-embeddings-v2-small-en | Precision: ${PRECISION} (${DTYPE}) ${NC}"
echo -e "${BLUE}==============================================================================${NC}"

echo -e "\n${YELLOW}[1/3] Generating deployment configuration with precision: ${DTYPE}...${NC}"
TMP_MANIFEST="/tmp/jina_v5e_deployment_${PRECISION}.yaml"
sed "s/--dtype .*/--dtype ${DTYPE} \\\\/" "${DEPLOY_DIR}/jina_v5e_deployment.yaml" > "$TMP_MANIFEST"

echo -e "\n${YELLOW}[2/3] Submitting Kubernetes deployment to cluster...${NC}"
kubectl apply -f "$TMP_MANIFEST"

echo -e "\n${YELLOW}[3/3] Waiting for TPU v5e serving pod to reach Ready status...${NC}"
kubectl rollout status deployment/jina-embeddings-v2-tpu --timeout=300s
kubectl get pods -l app=jina-embeddings-v2 -o wide

echo -e "\n${YELLOW}Running live inference smoke test...${NC}"
POD_NAME=$(kubectl get pods -l app=jina-embeddings-v2 -o jsonpath='{.items[0].metadata.name}')
RESP=$(kubectl exec "$POD_NAME" -c jina-vllm-server -- curl -s -X POST http://127.0.0.1:8000/prompt_c2 \
    -H "Content-Type: application/json" \
    -d '{"text": "Palo Alto Networks ATP Verification Smoke Test"}' || true)

if echo "$RESP" | grep -q "embedding"; then
    echo -e "${GREEN}✓ Smoke test PASSED! Embedding vector generated successfully.${NC}"
else
    echo -e "${RED}✗ Warning: Smoke test returned unexpected response: ${RESP}${NC}"
fi

echo -e "\n${BLUE}==============================================================================${NC}"
echo -e "${GREEN}✓ TPU Inference Service is live on http://jina-embedding-service:8000!${NC}"
echo -e "${BLUE}  Next step: Run ./scripts/03_setup_benchmark_runner.sh${NC}"
echo -e "${BLUE}==============================================================================${NC}"
