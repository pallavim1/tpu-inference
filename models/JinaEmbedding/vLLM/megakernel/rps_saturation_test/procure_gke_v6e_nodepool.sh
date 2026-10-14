#!/usr/bin/env bash
# Helper script to procure a 1-chip TPU v6e (ct6e-standard-1t) GKE nodepool
# in us-east5-a or us-east5-b (with us-east1-d fallback) and deploy the
# TPU server pod (jina-opus-v6e-test) + CPU load-generator pod (panw-v6e-benchmark-runner).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT="${PROJECT:-northam-ce-mlai-tpu}"
CLUSTER="${CLUSTER:-pm-panw-v6e-ga}"
REGION="${REGION:-us-east5}"
ZONE="${ZONE:-us-east5-a}"
NODEPOOL_NAME="${NODEPOOL_NAME:-tpu-v6e-1t-pool}"
NUM_NODES="${NUM_NODES:-1}"

echo "=== Step 1: Checking / Procuring 1-chip TPU v6e (ct6e-standard-1t) Nodepool ==="
echo "Project: ${PROJECT} | Cluster: ${CLUSTER} | Zone: ${ZONE}"

if gcloud container node-pools describe "${NODEPOOL_NAME}" \
  --cluster="${CLUSTER}" --region="${REGION}" --project="${PROJECT}" >/dev/null 2>&1; then
  echo "Nodepool ${NODEPOOL_NAME} already exists in ${CLUSTER}."
else
  echo "Creating nodepool ${NODEPOOL_NAME} (ct6e-standard-1t) in ${ZONE}..."
  gcloud container node-pools create "${NODEPOOL_NAME}" \
    --cluster="${CLUSTER}" \
    --region="${REGION}" \
    --node-locations="${ZONE}" \
    --project="${PROJECT}" \
    --machine-type=ct6e-standard-1t \
    --num-nodes="${NUM_NODES}"
fi

echo "=== Step 2: Deploying TPU Server Pod & CPU Benchmark Runner Pod ==="
kubectl apply -f "${SCRIPT_DIR}/jina_v6e_benchmark_pods.yaml"

echo "Waiting for jina-opus-v6e-test and panw-v6e-benchmark-runner to reach Running..."
kubectl wait --for=condition=Ready pod/jina-opus-v6e-test --timeout=600s
kubectl wait --for=condition=Ready pod/panw-v6e-benchmark-runner --timeout=300s

kubectl get pods -o wide jina-opus-v6e-test panw-v6e-benchmark-runner
