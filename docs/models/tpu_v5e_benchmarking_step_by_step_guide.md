# Step-by-Step Benchmarking Guide: Jina Embeddings v2 on Cloud TPU v5e
## From Baseline (up to 40 RPS) to Full Saturation Sweep Across All Payloads

**Model:** `jinaai/jina-embeddings-v2-small-en` (512-dim, JinaBert)  
**Serving Engine:** vLLM (`v0.26.0`) on Google Cloud TPU v5e (`ct5lp-hightpu-1t` / 1x Chip)  
**Client Environment:** Dedicated CPU Node Pool (`cpu-benchmark-runner` on `n2-standard-8`)  
**Network Path:** GKE Cluster Network (`europe-west4-b`) via `http://jina-embedding-service:8000/prompt_c2`  
**Target SLA:** Strict Tail Latency $P_{99} < 50\text{ ms}$  

---

## 1. Prerequisites & Environment Setup

### A. Connect to GKE Cluster
```bash
export PATH="/usr/local/google/home/pallaviam/google-cloud-sdk/bin:$PATH"

gcloud container clusters get-credentials pm-panw-jina-cluster \
    --zone=europe-west4-b \
    --project=northam-ce-mlai-tpu
```

### B. Verify Serving and Benchmark Runner Pods
```bash
kubectl get pods -o wide
```
*Expected: `jina-embeddings-v2-tpu-...` (Running on TPU pool) and `cpu-benchmark-runner` (Running on CPU pool).*

### C. Live Smoke Test (Endpoint Verification)
```bash
kubectl exec cpu-benchmark-runner -- curl -s -X POST http://jina-embedding-service:8000/prompt_c2 \
  -H "Content-Type: application/json" \
  -d '{"text": "Palo Alto Networks ATP Verification Smoke Test"}' | head -c 200
```

---

## 2. Part 1: Running Baseline Benchmarks (Up to 40 RPS)

This test reproduces PANW's standard evaluation matrix (**1, 5, 7, 10, 20, 30, 40 RPS**) across all four payload sizes (**1 KB, 2 KB, 5 KB, 7 KB**).

### Option A: Run Full 48-Stage Baseline Matrix (Automated)
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

### Option B: Run Fixed Payload Size Suite at Baseline (20 RPS or 40 RPS)
To evaluate 1K, 2K, 5K, and 7K sequentially at a fixed throughput (e.g. 20 RPS or 40 RPS):
```bash
# Run 20 RPS Baseline across all 4 payloads (60s each):
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

# Run 40 RPS Baseline across all 4 payloads (60s each):
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

### Option C: Run Single Ad-Hoc Payload & RPS Stage
```bash
# Example: 7 KB payload at 40 RPS for 60 seconds
kubectl exec -it cpu-benchmark-runner -- bash -c "
  cd /workspace && \
  k6 run --no-thresholds \
    --out json=/workspace/results/adhoc_7kb_40rps.ndjson \
    -e ENDPOINT=prompt_c2 \
    -e HTTP_URL=http://jina-embedding-service:8000 \
    -e RPS=40 \
    -e VUS=80 \
    -e DURATION=60s \
    -e MIN_SIZE=7168 \
    -e MAX_SIZE=7168 \
    k6_ray_serve_test.js
"
```

---

## 3. Part 2: Running Saturation Benchmarks (Until Saturation for Each Payload)

To determine the maximum certified SLA capacity ($P_{99} < 50\text{ ms}$) and the exact physical hardware saturation boundary, run the phased saturation suite.

### Phase 1: Multi-Payload Saturation Sweep (50 to 90 RPS)
Sweeps all 4 payloads (1K, 2K, 5K, 7K) through 50, 60, 70, 80, and 90 RPS (60s each):
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
* **7 KB Saturation Result**: 80 RPS ($P_{99} = 44.9\text{ ms}$, ✅ PASS). Saturates at 90 RPS ($P_{99} = 126.3\text{ ms}$).
* **5 KB Saturation Result**: 80 RPS ($P_{99} = 37.9\text{ ms}$, ✅ PASS). Saturates at 90 RPS ($P_{99} = 149.3\text{ ms}$).

---

### Phase 2: Dedicated 2 KB Saturation Sweep (90 to 110 RPS)
Sweeps 2 KB payloads through 90, 95, 100, and 110 RPS:
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
* **2 KB Saturation Result**: 90 RPS ($P_{99} = 32.3\text{ ms}$, ✅ PASS). Saturates at 95 RPS ($P_{99} = 1,362.1\text{ ms}$).

---

### Phase 3: Dedicated 1 KB Saturation Sweep (100 to 220 RPS)
Sweeps 1 KB payloads through 100, 120, 140, 160, 180, 190, 200, and 220 RPS:
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
* **1 KB Saturation Result**: 160 RPS ($P_{99} = 28.9\text{ ms}$, ✅ PASS). Physical saturation boundary reached at 177 RPS / 180 RPS ($P_{99} = 1,019.7\text{ ms}$).

---

## 4. Part 3: Automated One-Click Saturation Runner (All Phases with Timestamping)

To execute the entire 17-stage saturation sweep automatically in the background and generate isolated timestamped outputs:

```bash
kubectl exec -it cpu-benchmark-runner -- python3 /workspace/run_timestamped_benchmark_suite.py --duration 60s --phase all
```

---

## 5. Part 4: Switching Between FP32 and FP16 / BF16 Precision

To toggle precision on the TPU v5e deployment:

### To Switch to Native 16-bit Precision (`bfloat16`):
In `jina_v5e_deployment.yaml`, ensure the vLLM container start command uses:
```yaml
args:
  - --model=jinaai/jina-embeddings-v2-small-en
  - --dtype=bfloat16
  - --port=8001
```
Apply and wait:
```bash
kubectl apply -f deploy/jina_v5e_deployment.yaml
kubectl rollout status deployment/jina-embeddings-v2-tpu
```

### To Switch to FP32 Precision (`float32`):
```yaml
args:
  - --model=jinaai/jina-embeddings-v2-small-en
  - --dtype=float32
  - --port=8001
```

---

## 6. Part 5: Parsing Telemetry & Exporting Consolidated Deliverables

### A. Parse Raw k6 Telemetry to Summary JSON and Excel:
```bash
python3 /workspace/analyze_k6_results.py /workspace/results/my_test.ndjson --out /workspace/results/my_test_summary
```

### B. Generate Consolidated Multi-Tab Excel Workbook:
```bash
python3 /workspace/generate_consolidated_excel.py
```
*Creates `/workspace/jina_embeddings_v2_tpu_v5e_benchmarks.xlsx` with separate tabs for `prompt-c2 Onnx FP 32 v5e`, `prompt-c2 Onnx FP 16 v5e`, and `Performance-$ Analysis`.*

---

## 7. Master Saturation & Headroom Reference Matrix

| Payload Size | Approx Tokens | L4 Baseline Max RPS ($P_{99} < 50\text{ ms}$) | TPU v5e Max RPS ($P_{99} < 50\text{ ms}$) | Exact Saturation Boundary | Headroom vs. 20 RPS Baseline | Headroom vs. 40 RPS Peak |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB** (`1024 B`) | ~200 | **40 RPS** | **160 RPS** ($P_{99}=28.9\text{ms}$) | **180 RPS** ($P_{99}=1,019.7\text{ms}$) | **8.0x Margin** | **4.0x Margin** |
| **2 KB** (`2048 B`) | ~400 | **40 RPS** | **90 RPS** ($P_{99}=32.3\text{ms}$) | **95 RPS** ($P_{99}=1,362.1\text{ms}$) | **4.5x Margin** | **2.25x Margin** |
| **5 KB** (`5120 B`) | ~1,000 | **20 RPS** | **80 RPS** ($P_{99}=37.9\text{ms}$) | **90 RPS** ($P_{99}=149.3\text{ms}$) | **4.0x Margin** | **2.0x Margin** |
| **7 KB** (`7168 B`) | ~1,400 | **10 RPS** | **80 RPS** ($P_{99}=44.9\text{ms}$) | **90 RPS** ($P_{99}=126.3\text{ms}$) | **4.0x Margin** | **2.0x Margin** |
