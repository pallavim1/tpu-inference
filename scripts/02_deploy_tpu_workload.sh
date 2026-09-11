#!/usr/bin/env bash
# Step 2: Deploy TPU v5e Inference Service
set -euo pipefail

MANIFEST_PATH="${MANIFEST_PATH:-panw-benchmark/jina_v5e_deployment.yaml}"

echo "=========================================================================="
echo " 2. Deploying Jina Embeddings v2 vLLM Service on TPU v5e"
echo "=========================================================================="

if [ -f "$MANIFEST_PATH" ]; then
    kubectl apply -f "$MANIFEST_PATH"
elif [ -f "deploy/jina_v5e_deployment.yaml" ]; then
    kubectl apply -f deploy/jina_v5e_deployment.yaml
elif [ -f "jina_v5e_deployment.yaml" ]; then
    kubectl apply -f jina_v5e_deployment.yaml
else
    echo "Deployment manifest not found!"
    exit 1
fi

kubectl rollout status deployment/jina-embeddings-v2-tpu --timeout=300s
kubectl get pods -l app=jina-embeddings-v2 -o wide
echo "✅ TPU v5e Inference Workload Deployed & Healthy!"
