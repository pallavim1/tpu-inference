# GKE Infrastructure Guide: Provisioning Cloud TPU v5e with DWS Flex-Start

This standalone guide provides step-by-step instructions to create a Google Kubernetes Engine (GKE) cluster and provision a **Dynamic Workload Scheduler (DWS) Flex-Start Cloud TPU v5e Node Pool**.

---

## 1. Overview: DWS Flex-Start for Cloud TPU v5e

**Dynamic Workload Scheduler (DWS) Flex-Start** is a Google Cloud consumption model designed to optimize accelerator access and cost efficiency for time-flexible AI/ML workloads:

* **Economics:** Up to **~50% discount** compared to standard on-demand pricing.
* **Duration:** Workloads run continuously for up to **7 days** per provisioning run.
* **Acquisition Model:** Dynamic queued scheduling—GKE requests TPU capacity and provisions nodes automatically as soon as resources become available.
* **Scale-to-Zero:** Starts at `0 nodes` ($0 compute cost) and automatically spins up TPU instances upon workload submission.
* **Supported Accelerator Shape:** `ct5lp-hightpu-1t` (1x Cloud TPU v5e chip / 16 vCPUs / 32 GB host memory / 16 GB HBM).
* **Supported Zones:** `europe-west4-b`, `us-west4-a`, `us-central1-a`.

---

## 2. Prerequisites & Environment Setup

### A. Set Environment Variables
```bash
export PROJECT_ID="<YOUR_GCP_PROJECT_ID>"
export REGION="europe-west4"
export ZONE="europe-west4-b"
export CLUSTER_NAME="tpu-flex-cluster"
export VPC_NAME="tpu-flex-vpc"
export SUBNET_NAME="tpu-flex-subnet"
export TPU_POOL_NAME="tpu-v5e-dws-flex-pool"

# Set active project
gcloud config set project "$PROJECT_ID"
```

### B. Enable Required Google Cloud APIs
```bash
gcloud services enable \
    container.googleapis.com \
    tpu.googleapis.com \
    compute.googleapis.com
```

---

## 3. Step 1: Create VPC Network & Subnet

Create a custom VPC network with dedicated secondary IP ranges for GKE Pods and Services:

```bash
# 1. Create VPC Network
gcloud compute networks create "$VPC_NAME" \
    --project="$PROJECT_ID" \
    --subnet-mode=custom

# 2. Create Subnet with Pod and Service secondary IP ranges
gcloud compute networks subnets create "$SUBNET_NAME" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --network="$VPC_NAME" \
    --range=10.240.0.0/20 \
    --secondary-range=gke-pods=10.241.0.0/16,gke-services=10.242.0.0/20

# 3. Create Internal Firewall Rule
gcloud compute firewall-rules create "${VPC_NAME}-allow-internal" \
    --project="$PROJECT_ID" \
    --network="$VPC_NAME" \
    --allow=tcp,udp,icmp \
    --source-ranges=10.240.0.0/20,10.241.0.0/16,10.242.0.0/20
```

---

## 4. Step 2: Create GKE Cluster

Create a standard GKE cluster with VPC-native IP aliasing and a lightweight management node pool:

```bash
gcloud container clusters create "$CLUSTER_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --release-channel=rapid \
    --network="$VPC_NAME" \
    --subnetwork="$SUBNET_NAME" \
    --cluster-secondary-range-name=gke-pods \
    --services-secondary-range-name=gke-services \
    --num-nodes=1 \
    --machine-type=e2-standard-4 \
    --enable-ip-alias
```

Fetch cluster credentials for `kubectl`:
```bash
gcloud container clusters get-credentials "$CLUSTER_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID"
```

---

## 5. Step 3: Create DWS Flex-Start TPU v5e Node Pool

Provision the autoscaling DWS Flex-Start node pool. The pool initializes at **0 nodes**, so you do not incur charges until workloads are scheduled:

```bash
gcloud container node-pools create "$TPU_POOL_NAME" \
    --project="$PROJECT_ID" \
    --cluster="$CLUSTER_NAME" \
    --zone="$ZONE" \
    --node-locations="$ZONE" \
    --machine-type=ct5lp-hightpu-1t \
    --reservation-affinity=none \
    --enable-autoscaling \
    --num-nodes=0 \
    --min-nodes=0 \
    --max-nodes=2 \
    --flex-start
```

### Parameter Reference:
* **`--machine-type=ct5lp-hightpu-1t`**: Specifies 1x Cloud TPU v5e accelerator chip.
* **`--flex-start`**: Enables DWS Flex-Start dynamic queued allocation and discounted pricing.
* **`--enable-autoscaling --min-nodes=0 --max-nodes=2`**: Allows GKE to dynamically provision up to 2 TPU nodes when needed and scale to zero when idle.
* **`--reservation-affinity=none`**: Ensures node allocation draws from DWS dynamic capacity rather than a static reservation.

---

## 6. Step 4: Verify Node Pool Status

Verify that the node pool is successfully created and active in GKE:

```bash
gcloud container node-pools describe "$TPU_POOL_NAME" \
    --cluster="$CLUSTER_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID"
```

*Expected Output:*
```yaml
autoscaling:
  enabled: true
  flexStart: true
  maxNodeCount: 2
  minNodeCount: 0
machineType: ct5lp-hightpu-1t
name: tpu-v5e-dws-flex-pool
status: RUNNING
```

---

## 7. Next Steps: Targeting the DWS Flex Node Pool

Your GKE cluster is now equipped with an active DWS Flex-Start TPU v5e node pool. 

To schedule any containerized TPU workload onto this pool, ensure your Kubernetes pod/deployment manifest includes:

```yaml
spec:
  nodeSelector:
    cloud.google.com/gke-nodepool: tpu-v5e-dws-flex-pool
  tolerations:
  - key: "google.com/tpu"
    operator: "Exists"
    effect: "NoSchedule"
  containers:
  - name: my-tpu-app
    resources:
      limits:
        google.com/tpu: "1"
```

When this manifest is submitted, GKE will automatically queue the request, acquire the TPU v5e instance via DWS Flex-Start, and execute the container.
