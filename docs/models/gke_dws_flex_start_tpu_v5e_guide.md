# Step-by-Step Customer Guide: Using DWS Flex-Start with Cloud TPU v5e on GKE

This guide provides end-to-end instructions for provisioning a Google Kubernetes Engine (GKE) cluster with a **Dynamic Workload Scheduler (DWS) Flex-Start Cloud TPU v5e Node Pool**, deploying AI/ML workloads, and leveraging dynamic provisioning with up to **50% cost savings** compared to on-demand pricing.

---

## 1. Overview: What is DWS Flex-Start?

**Dynamic Workload Scheduler (DWS) Flex-Start** is a Google Cloud consumption model designed for workloads that can run with flexible start times. 

* **How it works:** You create an autoscaling node pool configured with `--flex-start`. When you schedule a TPU workload, GKE queues the request and automatically provisions TPU v5e VMs as soon as capacity becomes available.
* **Duration:** Workloads run for up to **7 days** per run.
* **Cost Efficiency:** Up to **50% discount** compared to standard on-demand pricing.
* **Supported Machine Types:** `ct5lp-hightpu-1t` (Single-chip TPU v5e / 1x1 topology).
* **Recommended Regions/Zones:** `europe-west4-b`, `us-west4-a`, `us-central1-a`.

---

## 2. Prerequisites & Project Setup

### A. Set Environment Variables
```bash
export PROJECT_ID="northam-ce-mlai-tpu"        # Replace with your GCP Project ID
export REGION="europe-west4"
export ZONE="europe-west4-b"
export CLUSTER_NAME="pm-panw-jina-cluster"
export VPC_NAME="pm-panw-jina-vpc"
export SUBNET_NAME="pm-panw-jina-subnet"
export TPU_POOL_NAME="tpu-v5e-dws-flex-pool"

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

Create a custom VPC network with secondary IP ranges for GKE Pods and Services:

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
    --secondary-range=pm-panw-jina-pods=10.241.0.0/16,pm-panw-jina-services=10.242.0.0/20

# 3. Create Internal Firewall Rule
gcloud compute firewall-rules create "${VPC_NAME}-allow-internal" \
    --project="$PROJECT_ID" \
    --network="$VPC_NAME" \
    --allow=tcp,udp,icmp \
    --source-ranges=10.240.0.0/20,10.241.0.0/16,10.242.0.0/20
```

---

## 4. Step 2: Create GKE Cluster

Create a standard GKE cluster with VPC-native IP aliasing and a small management pool:

```bash
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
```

Fetch cluster credentials for `kubectl`:
```bash
gcloud container clusters get-credentials "$CLUSTER_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID"
```

---

## 5. Step 3: Create DWS Flex-Start TPU v5e Node Pool

Create the autoscaling DWS Flex-Start node pool. It starts at **0 nodes** so that you incur zero cost until a workload is scheduled:

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
    --max-nodes=1 \
    --flex-start
```

### Key Flag Explanations:
* **`--machine-type=ct5lp-hightpu-1t`**: Specifies 1x Cloud TPU v5e chip (16 vCPUs, 32 GB host memory, 16 GB HBM TPU).
* **`--flex-start`**: Enables DWS Flex-Start pricing and queued provisioning.
* **`--enable-autoscaling --min-nodes=0 --max-nodes=1`**: Allows GKE to scale from zero when pods are submitted and scale back to zero when idle.
* **`--reservation-affinity=none`**: Ensures the pool provisions via DWS dynamic capacity rather than targeting a dedicated static reservation.

Verify the node pool creation:
```bash
gcloud container node-pools describe "$TPU_POOL_NAME" \
    --cluster="$CLUSTER_NAME" \
    --zone="$ZONE"
```
*Expected: `status: RUNNING` with `autoscaling.flexStart: true`.*

---

## 6. Step 4: Deploy Workload Targeting DWS Flex Pool

To run workloads on the DWS Flex node pool, include the appropriate `nodeSelector` and `tolerations` in your Kubernetes pod or deployment spec.

### Example: Jina Embeddings vLLM Serving on DWS Flex-Start TPU

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jina-embeddings-v2-tpu-flex
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jina-embeddings-v2-flex
  template:
    metadata:
      labels:
        app: jina-embeddings-v2-flex
    spec:
      nodeSelector:
        cloud.google.com/gke-nodepool: tpu-v5e-dws-flex-pool
      tolerations:
      - key: "google.com/tpu"
        operator: "Exists"
        effect: "NoSchedule"
      containers:
      - name: vllm-tpu-server
        image: python:3.12-slim
        command:
          - /bin/bash
          - -c
          - |
            pip install vllm-tpu==0.26.0
            exec vllm serve jinaai/jina-embeddings-v2-small-en \
              --runner pooling \
              --convert embed \
              --trust-remote-code \
              --max-model-len 2048 \
              --dtype bfloat16 \
              --host 0.0.0.0 \
              --port 8000
        resources:
          requests:
            google.com/tpu: "1"
            cpu: "8"
            memory: "20Gi"
          limits:
            google.com/tpu: "1"
            cpu: "16"
            memory: "30Gi"
        ports:
        - containerPort: 8000
```

Apply the deployment:
```bash
kubectl apply -f deployment_flex.yaml
```

---

## 7. Step 5: Monitor Scale-Up & Verification

1. **Watch GKE Cluster Autoscaler trigger DWS Flex-Start node provisioning:**
   ```bash
   kubectl get pods -w
   ```
   *The pod will transition from `Pending` ➔ `ContainerCreating` ➔ `Running` as GKE provisions the Flex-Start TPU node.*

2. **Verify the active TPU node:**
   ```bash
   kubectl get nodes -l cloud.google.com/gke-nodepool=tpu-v5e-dws-flex-pool
   ```

3. **Check live vLLM inference logs:**
   ```bash
   kubectl logs -f -l app=jina-embeddings-v2-flex
   ```

4. **Scale to zero when complete:**
   ```bash
   kubectl delete deployment jina-embeddings-v2-tpu-flex
   ```
   *After deleting the workload, GKE automatically tears down the TPU node, stopping all compute billing.*

---

## 8. Summary Comparison: DWS Flex-Start vs. Other Consumption Models

| Parameter | **DWS Flex-Start** | **On-Demand** | **Spot / Preemptible** | **1-Yr / 3-Yr CUD Reservation** |
| :--- | :---: | :---: | :---: | :---: |
| **Pricing Discount** | **~50% Off** 💰 | Baseline (0%) | ~70% Off | 30% to 55% Off |
| **Provisioning Model** | Queued / Dynamic | Immediate | Opportunistic (preemptible) | Guaranteed / Reserved |
| **Max Run Duration** | **Up to 7 Days** | Unlimited | Subject to preemption | 1 to 3 Years |
| **Scale-From-Zero** | ✅ Supported | ✅ Supported | ✅ Supported | ❌ Fixed Cost |
| **Best For** | Benchmarking, batch tests, model fine-tuning, flexible serving | Production steady-state without commitment | Fault-tolerant batch training | High-volume 24/7 production |
