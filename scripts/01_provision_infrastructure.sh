#!/usr/bin/env bash
# Step 1: Provision GKE Cluster & Nodepools
set -euo pipefail

export PROJECT_ID="${PROJECT_ID:-northam-ce-mlai-tpu}"
export REGION="${REGION:-europe-west4}"
export ZONE="${ZONE:-europe-west4-b}"
export VPC_NAME="${VPC_NAME:-pm-panw-jina-vpc}"
export SUBNET_NAME="${SUBNET_NAME:-pm-panw-jina-subnet}"
export CLUSTER_NAME="${CLUSTER_NAME:-pm-panw-jina-cluster}"
export TPU_POOL_NAME="${TPU_POOL_NAME:-pm-panw-jina-tpu-pool}"
export CPU_POOL_NAME="${CPU_POOL_NAME:-cpu-benchmark-pool}"

echo "=========================================================================="
echo " 1. Provisioning GKE Infrastructure & Nodepools"
echo " Project: $PROJECT_ID | Zone: $ZONE | Cluster: $CLUSTER_NAME"
echo "=========================================================================="

gcloud config set project "$PROJECT_ID"
gcloud services enable container.googleapis.com tpu.googleapis.com compute.googleapis.com

if ! gcloud compute networks describe "$VPC_NAME" --project="$PROJECT_ID" &>/dev/null; then
    gcloud compute networks create "$VPC_NAME" --project="$PROJECT_ID" --subnet-mode=custom
fi

if ! gcloud compute networks subnets describe "$SUBNET_NAME" --project="$PROJECT_ID" --region="$REGION" &>/dev/null; then
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

if ! gcloud container node-pools describe "$CPU_POOL_NAME" --cluster="$CLUSTER_NAME" --zone="$ZONE" --project="$PROJECT_ID" &>/dev/null; then
    gcloud container node-pools create "$CPU_POOL_NAME" \
        --project="$PROJECT_ID" \
        --cluster="$CLUSTER_NAME" \
        --zone="$ZONE" \
        --node-locations="$ZONE" \
        --machine-type=n2-standard-8 \
        --num-nodes=1
fi

gcloud container clusters get-credentials "$CLUSTER_NAME" --zone="$ZONE" --project="$PROJECT_ID"
echo "✅ Infrastructure provisioned and credentials retrieved!"
