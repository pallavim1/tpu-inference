#!/bin/bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Cluster & TPU Node Pool Provisioning for Jina Embeddings v2 on TPU v5e
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ID="northam-ce-mlai-tpu"
REGION="europe-west4"
ZONE="europe-west4-b"
CLUSTER_NAME="pm-panw-jina-cluster"
VPC_NAME="pm-panw-jina-vpc"
SUBNET_NAME="pm-panw-jina-subnet"
TPU_POOL_NAME="pm-panw-jina-tpu-pool"

echo "=== [1/5] Setting active project: $PROJECT_ID ==="
gcloud config set project "$PROJECT_ID"

echo "=== [2/5] Creating VPC and Subnet ==="
if ! gcloud compute networks describe "$VPC_NAME" --project="$PROJECT_ID" &>/dev/null; then
    gcloud compute networks create "$VPC_NAME" \
        --project="$PROJECT_ID" \
        --subnet-mode=custom
fi

if ! gcloud compute networks subnets describe "$SUBNET_NAME" --region="$REGION" --project="$PROJECT_ID" &>/dev/null; then
    gcloud compute networks subnets create "$SUBNET_NAME" \
        --project="$PROJECT_ID" \
        --region="$REGION" \
        --network="$VPC_NAME" \
        --range=10.240.0.0/20 \
        --secondary-range=pm-panw-jina-pods=10.241.0.0/16,pm-panw-jina-services=10.242.0.0/20
fi

if ! gcloud compute firewall-rules describe pm-panw-jina-allow-internal --project="$PROJECT_ID" &>/dev/null; then
    gcloud compute firewall-rules create pm-panw-jina-allow-internal \
        --project="$PROJECT_ID" \
        --network="$VPC_NAME" \
        --allow=tcp,udp,icmp \
        --source-ranges=10.240.0.0/20,10.241.0.0/16,10.242.0.0/20
fi

echo "=== [3/5] Creating GKE Cluster: $CLUSTER_NAME ==="
if ! gcloud container clusters describe "$CLUSTER_NAME" --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then
    gcloud container clusters create "$CLUSTER_NAME" \
        --project="$PROJECT_ID" \
        --zone="$ZONE" \
        --release-channel=rapid \
        --network="$VPC_NAME" \
        --subnetwork="$SUBNET_NAME" \
        --cluster-secondary-range-name=pm-panw-jina-pods \
        --services-secondary-range-name=pm-panw-jina-services \
        --num-nodes=1 \
        --machine-type=e2-standard-4 \
        --enable-ip-alias
fi

echo "=== [4/5] Creating TPU v5e Node Pool: $TPU_POOL_NAME ==="
if ! gcloud container node-pools describe "$TPU_POOL_NAME" --cluster="$CLUSTER_NAME" --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then
    gcloud container node-pools create "$TPU_POOL_NAME" \
        --project="$PROJECT_ID" \
        --cluster="$CLUSTER_NAME" \
        --zone="$ZONE" \
        --node-locations="$ZONE" \
        --machine-type=ct5lp-hightpu-1t \
        --tpu-topology=1x1 \
        --num-nodes=1
fi

echo "=== [5/5] Configuring kubectl context & deploying workload ==="
gcloud container clusters get-credentials "$CLUSTER_NAME" --zone="$ZONE" --project="$PROJECT_ID"
kubectl apply -f jina_v5e_deployment.yaml

echo "=== Deployment submitted! Check pod status with: kubectl get pods -w ==="
