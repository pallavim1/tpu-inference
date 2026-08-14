#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# 01_provision_infrastructure.sh
# End-to-end infrastructure provisioning for Jina Embeddings on Cloud TPU v5e
# ==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ID="${PROJECT_ID:-northam-ce-mlai-tpu}"
REGION="${REGION:-europe-west4}"
ZONE="${ZONE:-europe-west4-b}"
CLUSTER_NAME="${CLUSTER_NAME:-pm-panw-jina-cluster}"
VPC_NAME="${VPC_NAME:-pm-panw-jina-vpc}"
SUBNET_NAME="${SUBNET_NAME:-pm-panw-jina-subnet}"
TPU_POOL_NAME="${TPU_POOL_NAME:-pm-panw-jina-tpu-pool}"
CPU_POOL_NAME="${CPU_POOL_NAME:-cpu-benchmark-pool}"

echo -e "${BLUE}==============================================================================${NC}"
echo -e "${BLUE}  Step 1: Provisioning GKE Cluster & Nodepools for TPU Inference Benchmarking  ${NC}"
echo -e "${BLUE}  Project: ${PROJECT_ID} | Region: ${REGION} | Zone: ${ZONE}${NC}"
echo -e "${BLUE}==============================================================================${NC}"

# 1. Project & APIs
echo -e "\n${YELLOW}[1/6] Setting active project and enabling GCP APIs...${NC}"
gcloud config set project "${PROJECT_ID}"
gcloud services enable container.googleapis.com tpu.googleapis.com compute.googleapis.com

# 2. VPC & Subnet
echo -e "\n${YELLOW}[2/6] Configuring VPC and Subnets...${NC}"
if ! gcloud compute networks describe "${VPC_NAME}" --project="${PROJECT_ID}" &>/dev/null; then
    gcloud compute networks create "${VPC_NAME}" \
        --project="${PROJECT_ID}" \
        --subnet-mode=custom
    echo -e "${GREEN}✓ Created VPC: ${VPC_NAME}${NC}"
else
    echo -e "${GREEN}✓ VPC ${VPC_NAME} already exists.${NC}"
fi

if ! gcloud compute networks subnets describe "${SUBNET_NAME}" --region="${REGION}" --project="${PROJECT_ID}" &>/dev/null; then
    gcloud compute networks subnets create "${SUBNET_NAME}" \
        --project="${PROJECT_ID}" \
        --region="${REGION}" \
        --network="${VPC_NAME}" \
        --range=10.240.0.0/20 \
        --secondary-range=pm-panw-jina-pods=10.241.0.0/16,pm-panw-jina-services=10.242.0.0/20
    echo -e "${GREEN}✓ Created Subnet: ${SUBNET_NAME}${NC}"
else
    echo -e "${GREEN}✓ Subnet ${SUBNET_NAME} already exists.${NC}"
fi

if ! gcloud compute firewall-rules describe pm-panw-jina-allow-internal --project="${PROJECT_ID}" &>/dev/null; then
    gcloud compute firewall-rules create pm-panw-jina-allow-internal \
        --project="${PROJECT_ID}" \
        --network="${VPC_NAME}" \
        --allow=tcp,udp,icmp \
        --source-ranges=10.240.0.0/20,10.241.0.0/16,10.242.0.0/20
    echo -e "${GREEN}✓ Created internal firewall rules.${NC}"
fi

# 3. GKE Cluster
echo -e "\n${YELLOW}[3/6] Creating GKE Cluster: ${CLUSTER_NAME}...${NC}"
if ! gcloud container clusters describe "${CLUSTER_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" &>/dev/null; then
    gcloud container clusters create "${CLUSTER_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}" \
        --release-channel=rapid \
        --network="${VPC_NAME}" \
        --subnetwork="${SUBNET_NAME}" \
        --cluster-secondary-range-name=pm-panw-jina-pods \
        --services-secondary-range-name=pm-panw-jina-services \
        --num-nodes=1 \
        --machine-type=e2-standard-4 \
        --enable-ip-alias
    echo -e "${GREEN}✓ Created GKE cluster: ${CLUSTER_NAME}${NC}"
else
    echo -e "${GREEN}✓ GKE cluster ${CLUSTER_NAME} already exists.${NC}"
fi

# 4. TPU v5e Nodepool
echo -e "\n${YELLOW}[4/6] Provisioning Cloud TPU v5e Node Pool: ${TPU_POOL_NAME}...${NC}"
if ! gcloud container node-pools describe "${TPU_POOL_NAME}" --cluster="${CLUSTER_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" &>/dev/null; then
    gcloud container node-pools create "${TPU_POOL_NAME}" \
        --project="${PROJECT_ID}" \
        --cluster="${CLUSTER_NAME}" \
        --zone="${ZONE}" \
        --node-locations="${ZONE}" \
        --machine-type=ct5lp-hightpu-1t \
        --tpu-topology=1x1 \
        --num-nodes=1
    echo -e "${GREEN}✓ Created TPU v5e Node Pool: ${TPU_POOL_NAME}${NC}"
else
    echo -e "${GREEN}✓ TPU v5e Node Pool ${TPU_POOL_NAME} already exists.${NC}"
fi

# 5. Dedicated CPU Benchmark Nodepool
echo -e "\n${YELLOW}[5/6] Provisioning Dedicated CPU Benchmark Node Pool: ${CPU_POOL_NAME}...${NC}"
if ! gcloud container node-pools describe "${CPU_POOL_NAME}" --cluster="${CLUSTER_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" &>/dev/null; then
    gcloud container node-pools create "${CPU_POOL_NAME}" \
        --project="${PROJECT_ID}" \
        --cluster="${CLUSTER_NAME}" \
        --zone="${ZONE}" \
        --node-locations="${ZONE}" \
        --machine-type=n2-standard-8 \
        --num-nodes=1
    echo -e "${GREEN}✓ Created CPU Benchmark Node Pool: ${CPU_POOL_NAME}${NC}"
else
    echo -e "${GREEN}✓ CPU Benchmark Node Pool ${CPU_POOL_NAME} already exists.${NC}"
fi

# 6. Fetch Credentials
echo -e "\n${YELLOW}[6/6] Fetching kubectl credentials...${NC}"
gcloud container clusters get-credentials "${CLUSTER_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}"
echo -e "${GREEN}✓ Active Kubernetes context updated.${NC}"

echo -e "\n${BLUE}==============================================================================${NC}"
echo -e "${GREEN}✓ Infrastructure provisioning complete!${NC}"
echo -e "${BLUE}  Next step: Run ./scripts/02_deploy_tpu_workload.sh${NC}"
echo -e "${BLUE}==============================================================================${NC}"
