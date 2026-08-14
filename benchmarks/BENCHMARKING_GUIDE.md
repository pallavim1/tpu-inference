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
2. `jina-embeddings-v2-tpu` (Deployment running vLLM with `--model jinaai/jina-embeddings-v2-small-en` on TPU v5e)
3. `jina-embedding-service` (Kubernetes ClusterIP Service on port 8000)

```bash
kubectl apply -f deploy/jina_v5e_deployment.yaml
```

### B. Toggle Model Precision (FP32 vs FP16 / BF16)
To change precision, update the `args` section in `deploy/jina_v5e_deployment.yaml`:
* **For FP16 / BF16 Precision (Recommended)**:
  ```yaml
  args:
    - --model=jinaai/jina-embeddings-v2-small-en
    - --dtype=bfloat16
    - --port=8001
  ```
* **For FP32 Precision**:
  ```yaml
  args:
    - --model=jinaai/jina-embeddings-v2-small-en
    - --dtype=float32
    - --port=8001
  ```

### C. Verify Pod Readiness
```bash
kubectl rollout status deployment/jina-embeddings-v2-tpu --timeout=300s
kubectl get pods -l app=jina-embeddings-v2 -o wide
```

---

## 4. Setting Up the CPU Benchmark Runner Pod

### A. Deploy Benchmark Runner Pod on CPU Node Pool
```bash
kubectl apply -f deploy/cpu_benchmark_runner.yaml
kubectl wait --for=condition=ready pod/cpu-benchmark-runner --timeout=120s
```

### B. Install Dependencies inside Runner Pod
```bash
kubectl exec cpu-benchmark-runner -- bash -c "
  apt-get update && apt-get install -y gnupg curl ca-certificates git python3-pip
  
  # Install k6
  gpg -k 2>/dev/null || true
  gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D7AAB9732172C82409E5
  echo 'deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main' | tee /etc/apt/sources.list.d/k6.list
  apt-get update && apt-get install -y k6
  
  # Install Python telemetry and Excel dependencies
  pip install --upgrade pip
  pip install openpyxl pandas
  mkdir -p /workspace/results /workspace/benchmark_runs
"
```

### C. Copy Benchmark Scripts into Runner Pod
```bash
kubectl cp benchmarks/k6_ray_serve_test.js cpu-benchmark-runner:/workspace/k6_ray_serve_test.js
kubectl cp benchmarks/k6_high_rps_saturation_test.js cpu-benchmark-runner:/workspace/k6_high_rps_saturation_test.js
kubectl cp benchmarks/k6_1kb_saturation_test.js cpu-benchmark-runner:/workspace/k6_1kb_saturation_test.js
kubectl cp benchmarks/k6_2kb_saturation_test.js cpu-benchmark-runner:/workspace/k6_2kb_saturation_test.js
kubectl cp benchmarks/analyze_k6_results.py cpu-benchmark-runner:/workspace/analyze_k6_results.py
kubectl cp benchmarks/generate_consolidated_excel.py cpu-benchmark-runner:/workspace/generate_consolidated_excel.py
kubectl cp benchmarks/run_cpu_to_tpu_saturation_fp16.py cpu-benchmark-runner:/workspace/run_cpu_to_tpu_saturation_fp16.py
```

### D. Verify Live Endpoint Reachability from CPU Node Pool
```bash
kubectl exec cpu-benchmark-runner -- curl -s -X POST http://jina-embedding-service:8000/prompt_c2 \
  -H "Content-Type: application/json" \
  -d '{"text": "Palo Alto Networks ATP Verification Smoke Test"}' | head -c 200
```
*Expected output: valid JSON containing 512-dim embedding representation.*

---

## 5. Running Baseline Benchmarks (Up to 40 RPS)

This step reproduces the customer's baseline test matrix (**1, 5, 7, 10, 20, 30, 40 RPS**) across 1K, 2K, 5K, and 7K payloads.

### Option A: Run Full 48-Stage Baseline Matrix Sweep
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

### Option B: Run Fixed Payload Baseline Suites (20 RPS / 40 RPS)
```bash
# 20 RPS Baseline across all 4 payloads:
kubectl exec -it cpu-benchmark-runner -- bash -c "
  cd /workspace && \
  k6 run --no-thresholds \
    --out json=/workspace/results/baseline_20rps.ndjson \
    -e SUITE=payload_size \
    -e ENDPOINT=prompt_c2 \
    -e HTTP_URL=http://jina-embedding-service:8000 \
    -e PAYLOAD_RPS=20 \
    -e STAGE_DURATION=60s \
    k6_ray_serve_test.js
"

# 40 RPS Baseline across all 4 payloads:
kubectl exec -it cpu-benchmark-runner -- bash -c "
  cd /workspace && \
  k6 run --no-thresholds \
    --out json=/workspace/results/baseline_40rps.ndjson \
    -e SUITE=payload_size \
    -e ENDPOINT=prompt_c2 \
    -e HTTP_URL=http://jina-embedding-service:8000 \
    -e PAYLOAD_RPS=40 \
    -e STAGE_DURATION=60s \
    k6_ray_serve_test.js
"
```

---

## 6. Running Full Saturation Benchmarks (Until Saturation for Each Payload)

### Phase 1: Multi-Payload Saturation Sweep (50 to 90 RPS)
Sweeps all 4 payloads through 50, 60, 70, 80, and 90 RPS (60s each):
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
* **7 KB Saturation Result**: **80 RPS** ($P_{99} = 44.9\text{ ms}$, ✅ PASS). Saturates at 90 RPS ($P_{99} = 126.3\text{ ms}$).
* **5 KB Saturation Result**: **80 RPS** ($P_{99} = 37.9\text{ ms}$, ✅ PASS). Saturates at 90 RPS ($P_{99} = 149.3\text{ ms}$).

---

### Phase 2: Dedicated 2 KB Saturation Sweep (90 to 110 RPS)
```bash
kubectl exec -it cpu-benchmark-runner -- bash -c "
  cd /workspace
  for rps in 90 95 100 110; do
    echo '>>> Running 2 KB @ '\$rps' RPS <<<'
    k6 run --no-thresholds \
      --out json=/workspace/results/2kb_\${rps}rps.ndjson \
      -e SUITE=payload_size \
      -e ENDPOINT=prompt_c2 \
      -e HTTP_URL=http://jina-embedding-service:8000 \
      -e PAYLOAD_RPS=\$rps \
      -e PAYLOAD_VUS=200 \
      -e STAGE_DURATION=60s \
      k6_2kb_saturation_test.js
      
    python3 /workspace/analyze_k6_results.py /workspace/results/2kb_\${rps}rps.ndjson --out /workspace/results/2kb_\${rps}rps_summary
    sleep 5
  done
"
```
* **2 KB Saturation Result**: **90 RPS** ($P_{99} = 32.3\text{ ms}$, ✅ PASS). Saturates at 95 RPS ($P_{99} = 1,362.1\text{ ms}$).

---

### Phase 3: Dedicated 1 KB Saturation Sweep (100 to 220 RPS)
```bash
kubectl exec -it cpu-benchmark-runner -- bash -c "
  cd /workspace
  for rps in 100 120 140 160 180 190 200 220; do
    echo '>>> Running 1 KB @ '\$rps' RPS <<<'
    k6 run --no-thresholds \
      --out json=/workspace/results/1kb_\${rps}rps.ndjson \
      -e SUITE=payload_size \
      -e ENDPOINT=prompt_c2 \
      -e HTTP_URL=http://jina-embedding-service:8000 \
      -e PAYLOAD_RPS=\$rps \
      -e PAYLOAD_VUS=250 \
      -e STAGE_DURATION=60s \
      k6_1kb_saturation_test.js
      
    python3 /workspace/analyze_k6_results.py /workspace/results/1kb_\${rps}rps.ndjson --out /workspace/results/1kb_\${rps}rps_summary
    sleep 5
  done
"
```
* **1 KB Saturation Result**: **160 RPS** ($P_{99} = 28.9\text{ ms}$, ✅ PASS). Physical saturation boundary reached at 177 RPS / 180 RPS ($P_{99} = 1,019.7\text{ ms}$).

---

## 7. Automated One-Click Saturation Benchmark Runner

To run all 17 saturation stages automatically in the background with isolated timestamped directories:

```bash
kubectl exec -it cpu-benchmark-runner -- python3 /workspace/run_timestamped_benchmark_suite.py --duration 60s --phase all
```

---

## 8. Telemetry Parsing & Consolidated Excel Generation

### A. Parse Raw k6 Telemetry to Summary JSON and Excel
```bash
python3 /workspace/analyze_k6_results.py /workspace/results/my_test.ndjson --out /workspace/results/my_test_summary
```

### B. Generate Multi-Tab Excel Workbook
```bash
python3 /workspace/generate_consolidated_excel.py
```
*Produces `/workspace/jina_embeddings_v2_tpu_v5e_benchmarks.xlsx` with three styled tabs:*
1. `prompt-c2 Onnx FP 32 v5e`
2. `prompt-c2 Onnx FP 16 v5e`
3. `Performance-$ Analysis`

### C. Copy Results to Workstation
```bash
kubectl cp cpu-benchmark-runner:/workspace/jina_embeddings_v2_tpu_v5e_benchmarks.xlsx ./jina_embeddings_v2_tpu_v5e_benchmarks.xlsx
```

---

## 9. Master Saturation & Headroom Reference

| Payload Size | Approx Tokens | L4 Baseline Max RPS ($P_{99} < 50\text{ ms}$) | TPU v5e Max RPS ($P_{99} < 50\text{ ms}$) | Exact Saturation Boundary | Headroom vs. 20 RPS Baseline | Headroom vs. 40 RPS Peak |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB** (`1024 B`) | ~200 | **40 RPS** | **160 RPS** ($P_{99}=28.9\text{ms}$) | **180 RPS** ($P_{99}=1,019.7\text{ms}$) | **8.0x Margin** | **4.0x Margin** |
| **2 KB** (`2048 B`) | ~400 | **40 RPS** | **90 RPS** ($P_{99}=32.3\text{ms}$) | **95 RPS** ($P_{99}=1,362.1\text{ms}$) | **4.5x Margin** | **2.25x Margin** |
| **5 KB** (`5120 B`) | ~1,000 | **20 RPS** | **80 RPS** ($P_{99}=37.9\text{ms}$) | **90 RPS** ($P_{99}=149.3\text{ms}$) | **4.0x Margin** | **2.0x Margin** |
| **7 KB** (`7168 B`) | ~1,400 | **10 RPS** | **80 RPS** ($P_{99}=44.9\text{ms}$) | **90 RPS** ($P_{99}=126.3\text{ms}$) | **4.0x Margin** | **2.0x Margin** |
