# End-to-End Benchmarking Guide: Jina Embeddings v2 on Cloud TPU v5e
## Comprehensive Runbook from Infrastructure Provisioning to Saturation Benchmarks

**Workload:** Palo Alto Networks (PANW) ATP Embedding Inference (`POST /prompt_c2`)  
**Model:** `jinaai/jina-embeddings-v2-small-en` (512-dim, JinaBert)  
**Serving Engine:** vLLM (`v0.26.0`) on Google Cloud TPU v5e (`ct5lp-hightpu-1t` / 1x Chip)  
**Client Environment:** Dedicated CPU Node Pool (`cpu-benchmark-runner` on `n2-standard-8`)  
**Network Path:** GKE Cluster Network (`europe-west4-b`) via `http://jina-embedding-service:8000/prompt_c2`  
**Target SLA:** Strict Tail Latency $P_{99} < 50\text{ ms}$  

---

## Table of Contents
1. [Google Cloud Project & Prerequisites](#1-google-cloud-project--prerequisites)
2. [GKE Cluster & Node Pool Provisioning](#2-gke-cluster--node-pool-provisioning)
3. [Deploying Jina Embeddings vLLM Service on TPU v5e](#3-deploying-jina-embeddings-vllm-service-on-tpu-v5e)
4. [Setting Up the CPU Benchmark Runner Pod](#4-setting-up-the-cpu-benchmark-runner-pod)
5. [Running Baseline Benchmarks (Up to 40 RPS)](#5-running-baseline-benchmarks-up-to-40-rps)
6. [Running Full Saturation Benchmarks (Until Saturation for Each Payload)](#6-running-full-saturation-benchmarks-until-saturation-for-each-payload)
7. [Automated One-Click Saturation Benchmark Runner](#7-automated-one-click-saturation-benchmark-runner)
8. [Telemetry Parsing & Consolidated Excel Generation](#8-telemetry-parsing--consolidated-excel-generation)
9. [Master Saturation & Headroom Reference](#9-master-saturation--headroom-reference)

---

## 1. Google Cloud Project & Prerequisites

### A. Configure Active Project & Google Cloud SDK
```bash
export PROJECT_ID="northam-ce-mlai-tpu"
export REGION="europe-west4"
export ZONE="europe-west4-b"
export PATH="/usr/local/google/home/pallaviam/google-cloud-sdk/bin:$PATH"

# Set active GCP project
gcloud config set project "$PROJECT_ID"
```

### B. Enable Required GCP APIs
```bash
gcloud services enable \
    container.googleapis.com \
    tpu.googleapis.com \
    compute.googleapis.com
```

---

## 2. GKE Cluster & Node Pool Provisioning

### A. Create Custom VPC Network, Subnet & Firewall
```bash
export VPC_NAME="pm-panw-jina-vpc"
export SUBNET_NAME="pm-panw-jina-subnet"

# 1. Create VPC
gcloud compute networks create "$VPC_NAME" \
    --project="$PROJECT_ID" \
    --subnet-mode=custom

# 2. Create Subnet with secondary IP ranges for GKE Pods and Services
gcloud compute networks subnets create "$SUBNET_NAME" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --network="$VPC_NAME" \
    --range=10.240.0.0/20 \
    --secondary-range=pm-panw-jina-pods=10.241.0.0/16,pm-panw-jina-services=10.242.0.0/20

# 3. Allow internal cluster traffic
gcloud compute firewall-rules create pm-panw-jina-allow-internal \
    --project="$PROJECT_ID" \
    --network="$VPC_NAME" \
    --allow=tcp,udp,icmp \
    --source-ranges=10.240.0.0/20,10.241.0.0/16,10.242.0.0/20
```

### B. Create GKE Cluster
```bash
export CLUSTER_NAME="pm-panw-jina-cluster"

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

### C. Create Cloud TPU v5e Node Pool
```bash
export TPU_POOL_NAME="pm-panw-jina-tpu-pool"

gcloud container node-pools create "$TPU_POOL_NAME" \
    --project="$PROJECT_ID" \
    --cluster="$CLUSTER_NAME" \
    --zone="$ZONE" \
    --node-locations="$ZONE" \
    --machine-type=ct5lp-hightpu-1t \
    --tpu-topology=1x1 \
    --num-nodes=1
```

### D. Create Dedicated CPU Benchmark Node Pool
To ensure client load generation does not contend with TPU resources, create a dedicated CPU nodepool:
```bash
export CPU_POOL_NAME="cpu-benchmark-pool"

gcloud container node-pools create "$CPU_POOL_NAME" \
    --project="$PROJECT_ID" \
    --cluster="$CLUSTER_NAME" \
    --zone="$ZONE" \
    --node-locations="$ZONE" \
    --machine-type=n2-standard-8 \
    --num-nodes=1
```

### E. Get Cluster Credentials
```bash
gcloud container clusters get-credentials "$CLUSTER_NAME" \
    --zone="$ZONE" \
    --project="$PROJECT_ID"
```

---

## 3. Deploying Jina Embeddings vLLM Service on TPU v5e

### A. Apply Deployment Manifest
The deployment includes:
1. `jina-adapter-script` (ConfigMap containing async high-performance aiohttp adapter proxy on port 8000)
2. `jina-embeddings-v2-tpu` (Deployment running vLLM with `--model jinaai/jina-embeddings-v2-small-en --max-model-len 8192 --max-num-batched-tokens 8192` on TPU v5e)
3. `jina-embedding-service` (Kubernetes ClusterIP Service on port 8000)

```bash
kubectl apply -f deploy/jina_v5e_deployment.yaml
```

---

## 4. Setting Up the CPU Benchmark Runner Pod

### A. Deploy Benchmark Runner Pod on CPU Node Pool
```bash
kubectl apply -f deploy/cpu_benchmark_runner.yaml
kubectl wait --for=condition=ready pod/cpu-benchmark-runner --timeout=120s
```

---

## 5. Running Baseline Benchmarks (Up to 40 RPS)

```bash
kubectl exec -it cpu-benchmark-runner -- bash -c "
  cd /workspace && \
  k6 run --no-thresholds \
    --out json=/workspace/results/baseline_matrix.ndjson \
    -e SUITE=rps_payload \
    -e ENDPOINT=prompt_c2 \
    -e HTTP_URL=http://jina-embedding-service:8000 \
    -e STAGE_DURATION=60s \
    k6_ray_serve_test.js
"
```

---

## 6. Running Full Saturation Benchmarks (Until Saturation for Each Payload)

### Phase 1: Multi-Payload Saturation Sweep (50 to 90 RPS)
Sweeps all payloads (1K, 2K, 5K, 7K, 8K) through 50, 60, 70, 80, and 90 RPS (60s each):
```bash
kubectl exec -it cpu-benchmark-runner -- bash -c "
  cd /workspace
  for rps in 50 60 70 80 90; do
    echo '>>> Running Multi-Payload @ '\$rps' RPS <<<'
    k6 run --no-thresholds \
      --out json=/workspace/results/multi_\${rps}rps.ndjson \
      -e SUITE=payload_size \
      -e ENDPOINT=prompt_c2 \
      -e HTTP_URL=http://jina-embedding-service:8000 \
      -e PAYLOAD_RPS=\$rps \
      -e STAGE_DURATION=60s \
      k6_high_rps_saturation_test.js
      
    python3 /workspace/analyze_k6_results.py /workspace/results/multi_\${rps}rps.ndjson --out /workspace/results/multi_\${rps}rps_summary
    sleep 5
  done
"
```

---

## 7. Automated One-Click Saturation Benchmark Runner

To run all saturation stages automatically in the background with isolated timestamped directories:

```bash
kubectl exec -it cpu-benchmark-runner -- python3 /workspace/run_cpu_to_tpu_saturation.py --duration 60s --phase all
```

Or execute directly via Python runner script:
```bash
python3 scripts/benchmarking/run_jina_v2_alibi_saturation_benchmark.py \
  --endpoint http://jina-embedding-service:8000/prompt_c2 \
  --duration 60s \
  --max-model-len 8192
```

---

## 8. Telemetry Parsing & Consolidated Excel Generation

```bash
python3 /workspace/analyze_k6_results.py /workspace/results/multi_60rps.ndjson --out /workspace/results/multi_60rps_summary
python3 /workspace/generate_consolidated_excel.py
```

---

## 9. Master Saturation & Headroom Reference (8192 Max Model Len)

| Payload Size | Approx Tokens | L4 Baseline Max RPS ($P_{99} < 50\text{ ms}$) | TPU v5e Max RPS ($P_{99} < 50\text{ ms}$) | Exact Saturation Boundary | Headroom vs. 20 RPS Baseline | Headroom vs. 40 RPS Peak |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB** (`1024 B`) | ~200 | **40 RPS** | **160 RPS** ($P_{99}=28.9\text{ms}$) | **180 RPS** ($P_{99}=1,019.7\text{ms}$) | **8.0x Margin** | **4.0x Margin** |
| **2 KB** (`2048 B`) | ~400 | **40 RPS** | **90 RPS** ($P_{99}=32.3\text{ms}$) | **95 RPS** ($P_{99}=1,362.1\text{ms}$) | **4.5x Margin** | **2.25x Margin** |
| **5 KB** (`5120 B`) | ~1,000 | **20 RPS** | **80 RPS** ($P_{99}=37.9\text{ms}$) | **90 RPS** ($P_{99}=149.3\text{ms}$) | **4.0x Margin** | **2.0x Margin** |
| **7 KB** (`7168 B`) | ~1,400 | **10 RPS** | **80 RPS** ($P_{99}=44.9\text{ms}$) | **90 RPS** ($P_{99}=126.3\text{ms}$) | **4.0x Margin** | **2.0x Margin** |
