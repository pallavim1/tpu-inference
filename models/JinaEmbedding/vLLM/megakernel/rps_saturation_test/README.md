# Jina Embeddings v2 (`jina-embeddings-v2-small-en`) on TPU v6e (FP32) — End-to-End Benchmark & Replication Guide

This directory contains the complete, self-contained scripts, Kubernetes manifests, and step-by-step instructions to:
1. **Procure a 1-chip TPU v6e (`ct6e-standard-1t`)** node in GKE (`us-east5-a` or `us-east5-b`, with `us-east1-d` as fallback) and deploy the server + benchmark runner pods.
2. **Start `vllm serve`** (`--dtype float32 --max-model-len 2048 --max-num-batched-tokens 2048`) with the JinaBERT Pallas Megakernel + Fused Device Mean Pooling path and the `/prompt_c2` HTTP adapter proxy.
3. **Run all 3 benchmark test cases** across exact `1,024`-token (`1KB`) and `2,048`-token (`2KB`) payloads:
   - **Test Case 1 — Batch Request Testing:** Single HTTP `POST /prompt_c2` request containing $N \in \{1, 4, 8, 16\}$ prompts (`{"text": [p_1, ..., p_N]}`).
   - **Test Case 2 — Concurrent HTTP Request Testing:** Closed-loop `k6 constant-vus` with $\text{VUS} \in \{1, 4, 8, 16\}$ concurrent virtual users (`30s` per step).
   - **Test Case 3 — Dedicated Open-Loop RPS Saturation:** Open-loop `k6 constant-arrival-rate` (`60s` per RPS step on the 10-RPS grid) to measure maximum sustainable throughput under the **$\text{p99} < 50\text{ ms}$ SLA**.

---

## Files in This Directory

| File | Description |
| :--- | :--- |
| [`procure_gke_v6e_nodepool.sh`](./procure_gke_v6e_nodepool.sh) | Creates a 1-chip TPU v6e (`ct6e-standard-1t`) nodepool in `us-east5-a` or `us-east5-b` (or `us-east1-d`) and deploys the server + benchmark runner pods. |
| [`jina_v6e_benchmark_pods.yaml`](./jina_v6e_benchmark_pods.yaml) | Kubernetes Pod manifest for `jina-opus-v6e-test` (1x TPU v6e server pod) and `panw-v6e-benchmark-runner` (CPU load-generator pod). |
| [`start_server.sh`](./start_server.sh) | Applies JinaBERT environment shims (`setup_jina_env.py`), starts `stock_adapter_proxy.py` on `:8000` (`POST /prompt_c2`), and launches `vllm serve` on `:8001` with all Megakernel + Fused Pooling flags enabled. |
| [`setup_jina_env.py`](./setup_jina_env.py) | Idempotent setup script that ensures the cached HuggingFace JinaBERT model config is compatible with `transformers` and `vLLM`. |
| [`stock_adapter_proxy.py`](./stock_adapter_proxy.py) | Lightweight `aiohttp` adapter on `:8000` exposing `POST /prompt_c2` (`{"text": ...}`) and forwarding to `http://127.0.0.1:8001/v1/embeddings`. |
| [`build_token_pools.py`](./build_token_pools.py) | Generates deterministic pools of 25,000 unique prompts (`pools/pool_1024tok.json` and `pools/pool_2048tok.json`) verified with `AutoTokenizer` to tokenize to **exact `1,024` and `2,048` tokens** (including `[CLS]` and `[SEP]`). |
| [`run_rps_saturation_test.py`](./run_rps_saturation_test.py) | Automated benchmark driver that runs Preflight token verification, **Test Case 1 (Batch $N=1..16$)**, **Test Case 2 (`k6` Concurrent $\text{VUS}=1..16$)**, and **Test Case 3 (Dedicated 1K & 2K RPS Saturation)** while verifying exact token counts via `:8001/metrics`. |
| [`k6_1kb_saturation_test_1024tok.js`](./k6_1kb_saturation_test_1024tok.js) | `k6` `constant-arrival-rate` saturation script for `1,024`-token payloads (`250` pre-allocated VUs, `60s` stage duration). |
| [`k6_2kb_saturation_test_2048tok.js`](./k6_2kb_saturation_test_2048tok.js) | `k6` `constant-arrival-rate` saturation script for `2,048`-token payloads (`200` pre-allocated VUs, `60s` stage duration). |
| [`analyze_k6_results.py`](./analyze_k6_results.py) | Parses raw `k6` `.ndjson` output and computes per-stage `achieved_rps`, `min`, `p50`, `avg`, `p90`, `p95`, `p99`, `max`, and error counts. |

---

## Step 1 — Procure a 1-Chip TPU v6e (`ct6e-standard-1t`) in GKE & Deploy Pods

We verified immediate on-demand `ct6e-standard-1t` (`v6e-1`, `1x1` topology) availability in **`us-east5-a`** and **`us-east5-b`** (with **`us-east1-d`** as an additional fallback zone).

### Option A: Automated Nodepool Creation + Pod Deployment
```bash
git clone -b jina-v2-opus-megakernel https://github.com/pallavim1/tpu-inference.git
cd tpu-inference/models/JinaEmbedding/vLLM/megakernel/rps_saturation_test

# Customize your GKE cluster/zone if needed (defaults: CLUSTER=pm-panw-v6e-ga, REGION=us-east5, ZONE=us-east5-a):
export PROJECT="northam-ce-mlai-tpu"
export CLUSTER="<YOUR_GKE_CLUSTER>"
export REGION="us-east5"
export ZONE="us-east5-a"   # or us-east5-b (or us-east1-d if using a us-east1 cluster)

bash procure_gke_v6e_nodepool.sh
```

### Option B: If Using an Existing `ct6e-standard-1t` Nodepool (e.g., `pm-panw-v6e-ga-a` in `us-east5-a` or `pm-panw-v6e-ga-b` in `us-east5-b`)
Simply deploy the TPU server pod and CPU benchmark runner pod directly:
```bash
kubectl apply -f jina_v6e_benchmark_pods.yaml
kubectl wait --for=condition=Ready pod/jina-opus-v6e-test pod/panw-v6e-benchmark-runner --timeout=600s
```

---

## Step 2 — Clone the Branch & Start `vllm serve` + `/prompt_c2` Adapter on the TPU v6e Pod

Open a shell on the `jina-opus-v6e-test` pod, clone the **`jina-v2-opus-megakernel`** branch, install `tpu-inference`, and run `start_server.sh`:

```bash
kubectl exec -it jina-opus-v6e-test -- bash

# Inside the TPU v6e pod:
git clone -b jina-v2-opus-megakernel https://github.com/pallavim1/tpu-inference.git /workspace/tpu-inference
cd /workspace/tpu-inference
pip install -e . --no-build-isolation

# Launch the /prompt_c2 adapter (:8000) and vllm serve (:8001):
cd /workspace/tpu-inference/models/JinaEmbedding/vLLM/megakernel/rps_saturation_test
bash start_server.sh
```

`start_server.sh` automatically sets:
- `USE_JINA_BERT_MEGAKERNEL=1`
- `JINA_BERT_MEGAKERNEL_VERSION=v2`
- `JINA_BERT_MEGAKERNEL_PRECISION=default`
- `TPU_POOLING_FAST_PATH=1`
- `JINA_BERT_DEVICE_POOLING=1`
- `JINA_BERT_FUSED_POOLING=1`
- `TPU_ENCODER_INPUT_FAST_PATH=1`

Wait until `/tmp/jina_megakernel_logs/vllm_serve.log` shows `Application startup complete` and `curl -s http://127.0.0.1:8000/health` returns `{"status": "ok"}`.

---

## Step 3 — Run All 3 Test Cases from the CPU Benchmark Runner

First, get the TPU server pod IP from your workstation:
```bash
TPU_IP=$(kubectl get pod jina-opus-v6e-test -o jsonpath='{.status.podIP}')
echo "TPU Server Pod IP: ${TPU_IP}"
```

Next, open a shell on `panw-v6e-benchmark-runner` (or any CPU VM/pod in the same zone with `k6`, `aiohttp`, and `transformers` installed) and clone the repository:
```bash
kubectl exec -it panw-v6e-benchmark-runner -- bash

# Inside the CPU benchmark runner pod (one-time setup if using a fresh python:3.11-slim container):
apt-get update && apt-get install -y git curl gnupg ca-certificates
curl -fsSL https://dl.k6.io/key.gpg | gpg --dearmor -o /usr/share/keyrings/k6-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" > /etc/apt/sources.list.d/k6.list
apt-get update && apt-get install -y k6
pip install aiohttp transformers

git clone -b jina-v2-opus-megakernel https://github.com/pallavim1/tpu-inference.git /workspace/tpu-inference
cd /workspace/tpu-inference/models/JinaEmbedding/vLLM/megakernel/rps_saturation_test
```

### Run All 3 Test Cases End-to-End (Single Command)
```bash
export TPU_IP="<TPU_POD_IP>"

python3 run_rps_saturation_test.py \
  --base-url "http://${TPU_IP}:8000" \
  --metrics-url "http://${TPU_IP}:8001/metrics"
```
*(On first run, `run_rps_saturation_test.py` automatically generates `pools/pool_1024tok.json` and `pools/pool_2048tok.json` and verifies via `:8001/metrics` that every `1KB` prompt tokenizes to exactly `1,024` server tokens and every `2KB` prompt tokenizes to `2,048` server tokens.)*

---

### Or Run Each of the 3 Test Cases Individually

#### Test Case 1 — Batch Request Testing (Single HTTP Request Containing $N \in \{1, 4, 8, 16\}$ Prompts)
Sends a single HTTP `POST /prompt_c2` request containing $N = 1, 4, 8, 16$ prompts (`{"text": [p_1, ..., p_N]}`) for both `1KB (1,024 tok)` and `2KB (2,048 tok)`:
```bash
python3 run_rps_saturation_test.py \
  --test-case 1 \
  --base-url "http://${TPU_IP}:8000" \
  --metrics-url "http://${TPU_IP}:8001/metrics"
```

#### Test Case 2 — Concurrent HTTP Request Testing (`k6 constant-vus`, $\text{VUS} \in \{1, 4, 8, 16\}$)
Runs `k6` with `1, 4, 8, 16` concurrent virtual users (`30s` per step, 1 prompt per HTTP request) for both `1KB (1,024 tok)` and `2KB (2,048 tok)`:
```bash
python3 run_rps_saturation_test.py \
  --test-case 2 \
  --base-url "http://${TPU_IP}:8000" \
  --metrics-url "http://${TPU_IP}:8001/metrics"
```

#### Test Case 3 — Dedicated Open-Loop RPS Saturation (`k6 constant-arrival-rate`, `60s` per step, $\text{p99} < 50\text{ ms}$ SLA)
Sweeps open-loop arrival rates to find the maximum SLA-passing throughput ($\text{p99} < 50\text{ ms}$):
```bash
# Option 3A: Quick 15-minute check of the 10-RPS saturation knee (480..550 RPS for 1KB, 280..370 RPS for 2KB):
python3 run_rps_saturation_test.py \
  --knee-only \
  --base-url "http://${TPU_IP}:8000" \
  --metrics-url "http://${TPU_IP}:8001/metrics"

# Option 3B: Full saturation sweep (100..550 RPS for 1KB and 90..370 RPS for 2KB):
python3 run_rps_saturation_test.py \
  --test-case 3 \
  --base-url "http://${TPU_IP}:8000" \
  --metrics-url "http://${TPU_IP}:8001/metrics"
```

---

## Reference Benchmark Results on 1x TPU v6e (`ct6e-standard-1t`, FP32) vs. TPU v5e & L4 GPU

### 1. Test Case 1 — Batch Request Testing ($N \in \{1, 4, 8, 16\}$ Prompts per Single HTTP Request)
| Payload Size | Batch ($N$) | **TPU v6e Throughput** | **TPU v6e `p50`** | **TPU v6e `p99`** | TPU v5e Throughput / `p50` / `p99` | L4 GPU Throughput / `p50` / `p99` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (`1,024` tok)** | **1** | **139.4/s** | **7.1 ms** | **8.1 ms** | 85.8/s / 11.5 ms / 12.7 ms | 43.2/s / 20.7 ms / 23.5 ms |
| **1KB (`1,024` tok)** | **4** | **235.1/s** | **16.9 ms** | **18.1 ms** | 187.4/s / 21.2 ms / 22.8 ms | 103.4/s / 38.1 ms / 42.9 ms |
| **1KB (`1,024` tok)** | **8** | **259.5/s** | **30.6 ms** | **32.3 ms** | 187.6/s / 42.5 ms / 44.5 ms | 135.1/s / 58.7 ms / 66.3 ms |
| **1KB (`1,024` tok)** | **16** | **272.9/s** | **58.5 ms** | **59.6 ms** | 186.8/s / 85.4 ms / 89.9 ms | 165.2/s / 96.5 ms / 107.6 ms |
| **2KB (`2,048` tok)** | **1** | **115.5/s** | **8.5 ms** | **9.5 ms** | 59.5/s / 16.5 ms / 17.7 ms | 39.0/s / 24.3 ms / 28.2 ms |
| **2KB (`2,048` tok)** | **4** | **153.2/s** | **25.9 ms** | **27.1 ms** | 97.7/s / 40.7 ms / 42.8 ms | 75.7/s / 52.1 ms / 58.2 ms |
| **2KB (`2,048` tok)** | **8** | **165.3/s** | **48.1 ms** | **50.5 ms** | 96.6/s / 82.6 ms / 86.0 ms | 90.5/s / 87.4 ms / 97.3 ms |
| **2KB (`2,048` tok)** | **16** | **172.4/s** | **92.1 ms** | **98.0 ms** | 96.5/s / 165.9 ms / 171.2 ms | 101.1/s / 156.9 ms / 175.5 ms |

### 2. Test Case 2 — Concurrent HTTP Request Testing (`k6 constant-vus`, $\text{VUS} \in \{1, 4, 8, 16\}$)
| Payload Size | Concurrency (`VUS`) | **TPU v6e Throughput** | **TPU v6e `p50`** | **TPU v6e `p99`** | TPU v5e Throughput / `p50` / `p99` | L4 GPU Throughput / `p50` / `p99` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1KB (`1,024` tok)** | **1** | **134.1/s** | **6.6 ms** | **7.1 ms** | 85.8/s / 11.5 ms / 12.7 ms | 43.2/s / 20.7 ms / 23.5 ms |
| **1KB (`1,024` tok)** | **4** | **394.5/s** | **9.3 ms** | **12.0 ms** | 187.4/s / 21.2 ms / 22.8 ms | 103.4/s / 38.1 ms / 42.9 ms |
| **1KB (`1,024` tok)** | **8** | **519.2/s** | **14.3 ms** | **19.7 ms** | 187.6/s / 42.5 ms / 44.5 ms | 135.1/s / 58.7 ms / 66.3 ms |
| **1KB (`1,024` tok)** | **16** | **546.4/s** | **28.4 ms** | **37.4 ms** | 186.8/s / 85.4 ms / 89.9 ms | 165.2/s / 96.5 ms / 107.6 ms |
| **2KB (`2,048` tok)** | **1** | **105.8/s** | **8.0 ms** | **9.0 ms** | 59.5/s / 16.5 ms / 17.7 ms | 39.0/s / 24.3 ms / 28.2 ms |
| **2KB (`2,048` tok)** | **4** | **294.8/s** | **12.3 ms** | **15.6 ms** | 97.7/s / 40.7 ms / 42.8 ms | 75.7/s / 52.1 ms / 58.2 ms |
| **2KB (`2,048` tok)** | **8** | **374.0/s** | **19.7 ms** | **26.0 ms** | 96.6/s / 82.6 ms / 86.0 ms | 90.5/s / 87.4 ms / 97.3 ms |
| **2KB (`2,048` tok)** | **16** | **302.3/s** | **51.2 ms** | **58.2 ms** | 96.5/s / 165.9 ms / 171.2 ms | 101.1/s / 156.9 ms / 175.5 ms |

### 3. Test Case 3 — Dedicated Open-Loop RPS Saturation ($\text{p99} < 50\text{ ms}$ SLA, 10-RPS Grid)
| Payload Tier | Hardware Setup | Max SLA-Passing RPS | `p50` @ Max RPS | `p99` @ Max RPS | RPS Gain vs. L4 GPU | RPS Gain vs. TPU v5e | Cost / 1M Requests (40% Util.) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1K (`1,024` tok)** | L4 GPU (`g2-standard-4`, $\$0.70/\text{hr}$) | `70 /s` | — | — | Baseline | — | $\$6.94$ |
| **1K (`1,024` tok)** | TPU v5e (`ct5lp-hightpu-1t`, $\$1.20/\text{hr}$) | `180 /s` *(187 max)* | `19.9 ms` | `29.6 ms` | `+157.1%` (`2.57×`) | Baseline | $\$4.46$ |
| **1K (`1,024` tok)** | **TPU v6e (`ct6e-standard-1t`, $\$2.70/\text{hr}$)** | **`530 /s`** | **`13.1 ms`** | **`47.9 ms`** | **`+657.1%` (`7.57×`)** | **`+194.4%` (`2.94×`)** | **$\$3.54$ (`-20.6%` vs v5e)** |
| **2K (`2,048` tok)** | L4 GPU (`g2-standard-4`, $\$0.70/\text{hr}$) | `40 /s` | — | — | Baseline | — | $\$12.15$ |
| **2K (`2,048` tok)** | TPU v5e (`ct5lp-hightpu-1t`, $\$1.20/\text{hr}$) | `90 /s` | `17.2 ms` | `26.5 ms` | `+125.0%` (`2.25×`) | Baseline | $\$9.26$ |
| **2K (`2,048` tok)** | **TPU v6e (`ct6e-standard-1t`, $\$2.70/\text{hr}$)** | **`350 /s`** | **`11.7 ms`** | **`25.2 ms`** | **`+775.0%` (`8.75×`)** | **`+288.9%` (`3.89×`)** | **$\$5.36$ (`-42.1%` vs v5e)** |
