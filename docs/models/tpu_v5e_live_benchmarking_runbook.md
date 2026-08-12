# TPU v5e Live Benchmarking Runbook: Jina Embeddings v2

**Model:** `jinaai/jina-embeddings-v2-small-en` (512-dim Embeddings, JinaBert)  
**Cluster:** `pm-panw-jina-cluster`  
**Region/Zone:** `europe-west4-b`  
**GCP Project:** `northam-ce-mlai-tpu`  
**Target SLA:** P99 Round-Trip Latency < 50 ms  
**Load Tool:** Grafana k6 (`constant-arrival-rate`, open-loop) + `analyze_k6_results.py`

---

## 1. Overview & Architecture

This runbook provides step-by-step instructions for connecting to the Google Kubernetes Engine (GKE) TPU v5e cluster, verifying the live model deployment, and executing customer-equivalent (PANW ATP) saturation benchmark sweeps across all four payload sizes (**1 KB**, **2 KB**, **5 KB**, and **7 KB**) up to **190 RPS**.

All benchmark load generation, model inference, and statistical percentile analysis run **100% inside the TPU v5e pod** using localhost loopback (`127.0.0.1:8000`), eliminating network hops and measuring pure accelerator latency.

---

## 2. Prerequisites & Cluster Connection

Run the following commands on your workstation terminal to configure Google Cloud SDK and set your Kubernetes context to the TPU v5e cluster:

```bash
# 1. Export Google Cloud SDK binaries to PATH
export PATH="/usr/local/google/home/pallaviam/google-cloud-sdk/bin:$PATH"

# 2. Fetch cluster credentials
gcloud container clusters get-credentials pm-panw-jina-cluster \
    --zone=europe-west4-b \
    --project=northam-ce-mlai-tpu

# 3. Verify cluster connectivity and list running TPU pods
kubectl get pods -l app=jina-embeddings-v2 -o wide
```

---

## 3. Live Smoke Test (Direct Real-Time Inference)

Send a single test request directly into the live TPU v5e container to verify that the model server (`vllm`) and protocol adapter (`/prompt_c2`) are active and responding:

```bash
# Get pod name
POD=$(kubectl get pods -l app=jina-embeddings-v2 -o jsonpath='{.items[0].metadata.name}')

# Send test embedding request
kubectl exec $POD -- bash -c 'curl -s -X POST http://127.0.0.1:8000/prompt_c2 \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Smoke test payload for Jina Embeddings v2 on Google Cloud TPU v5e\"}"' | head -c 250
```

*Expected output: JSON containing valid 512-dim embedding float values (`{"id": "embd-...", "data": [{"embedding": [-0.35, 0.12, ...]}]}`).*

---

## 4. How to Run the Saturation Benchmarks

### Option A: Interactive In-Pod Shell (Recommended)

Running inside the container shell allows you to monitor and control tests directly:

```bash
# 1. Exec into an interactive shell inside the TPU Pod
POD=$(kubectl get pods -l app=jina-embeddings-v2 -o jsonpath='{.items[0].metadata.name}')
kubectl exec -it $POD -- bash

# 2. Navigate to the benchmark workspace
cd /workspace

# 3. Launch the full saturation suite in the background (60s stage duration)
nohup python3 -u /workspace/in_pod_saturation_benchmark.py --duration 60s > /workspace/in_pod_benchmark_60s.log 2>&1 &

# 4. Stream real-time progress and live latency tables
tail -f /workspace/in_pod_benchmark_60s.log
```
*(Press `Ctrl+C` at any time to exit the log viewer; the benchmark process will continue running in the background).*

---

### Option B: One-Click Local Workstation Trigger

If you prefer to kick off the benchmark from your workstation without entering the pod shell:

```bash
POD=$(kubectl get pods -l app=jina-embeddings-v2 -o jsonpath='{.items[0].metadata.name}')

# Launch detached in pod
kubectl exec $POD -- bash -c "nohup python3 -u /workspace/in_pod_saturation_benchmark.py --duration 60s > /workspace/in_pod_benchmark_60s.log 2>&1 < /dev/null &"

# Stream live updates
kubectl exec -it $POD -- tail -f /workspace/in_pod_benchmark_60s.log
```

---

### Option C: Run an Individual Ad-Hoc Scenario

To test a specific payload size and RPS level (for example, **7 KB payload at 60 RPS** for 60 seconds):

```bash
POD=$(kubectl get pods -l app=jina-embeddings-v2 -o jsonpath='{.items[0].metadata.name}')

# 1. Run k6 load test inside the pod
kubectl exec -it $POD -- bash -c 'cd /workspace && \
  k6 run --out json=/workspace/manual_7k_60rps.ndjson \
  -e SUITE=payload_size -e ENDPOINT=prompt_c2 -e STAGE_DURATION=60s -e PAYLOAD_RPS=60 \
  k6_high_rps_saturation_test.js'

# 2. Run analysis script and print latency percentile table
kubectl exec -it $POD -- python3 /workspace/analyze_k6_results.py \
  /workspace/manual_7k_60rps.ndjson --out /workspace/manual_7k_60rps_summary
```

---

## 5. Monitoring & Output Verification

### Check Status Anytime:
```bash
POD=$(kubectl get pods -l app=jina-embeddings-v2 -o jsonpath='{.items[0].metadata.name}')

# View all completed latency tables printed so far
kubectl exec $POD -- grep -A 10 "Results for" /workspace/in_pod_benchmark_60s.log

# List all generated .xlsx Excel reports and .json summaries
kubectl exec $POD -- ls -lh /workspace/live_results_60s/
```

### Copy Reports to Your Workstation:
To copy the entire results folder (including `.xlsx` spreadsheets) to your local environment:

```bash
POD=$(kubectl get pods -l app=jina-embeddings-v2 -o jsonpath='{.items[0].metadata.name}')

mkdir -p /usr/local/google/home/pallaviam/panw-benchmark/in_pod_live_results_60s
kubectl cp $POD:/workspace/live_results_60s /usr/local/google/home/pallaviam/panw-benchmark/in_pod_live_results_60s
```

---

## 6. Process Management & Troubleshooting

### Stopping / Cancelling an Active Benchmark Run:
To immediately stop any running `k6` or Python benchmark processes inside the pod:

```bash
POD=$(kubectl get pods -l app=jina-embeddings-v2 -o jsonpath='{.items[0].metadata.name}')

kubectl exec $POD -- python3 -c '
import os, signal
for pid_dir in os.listdir("/proc"):
    if pid_dir.isdigit():
        try:
            with open(f"/proc/{pid_dir}/cmdline", "rb") as f:
                cmd = f.read().decode("latin1")
                if ("k6" in cmd or "in_pod_saturation" in cmd or "analyze_k6" in cmd) and "python3 -c" not in cmd:
                    os.kill(int(pid_dir), signal.SIGTERM)
                    print(f"Stopped PID {pid_dir}")
        except Exception:
            pass
'
```

---

## 7. Master Saturation Reference Matrix (1 TPU v5e Chip)

| Payload Size | Character / Byte Range | Approx Token Count | Max Sustainable Throughput (<50ms P99 SLA) | Achieved P99 at Max RPS | Exact Saturation Boundary (P99 > 50ms) | Headroom vs. 20 RPS Baseline | Headroom vs. 40 RPS Peak |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB (1024 B)** | Short Queries | ~200 Tokens | **180 RPS** | **35.9 ms** | **190 RPS** (P99 = 771.4 ms) | **9.0x Margin** | **4.5x Margin** |
| **2 KB (2048 B)** | Standard Prompts | ~400 Tokens | **90 RPS** | **28.9 ms** | **95 RPS** (P99 = 363.8 ms) | **4.5x Margin** | **2.25x Margin** |
| **5 KB (5120 B)** | Large Context | ~1,000 Tokens | **80 RPS** | **49.5 ms** | **90 RPS** (P99 = 2,094.6 ms) | **4.0x Margin** | **2.0x Margin** |
| **7 KB (7168 B)** | Max Sequence Length | ~1,400 Tokens | **60 RPS** | **41.8 ms** | **70 RPS** (P99 = 59.4 ms) | **3.0x Margin** | **1.5x Margin** |
