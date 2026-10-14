# Jina Embeddings v2 (`jina-embeddings-v2-small-en`) TPU v6e FP32 — RPS Saturation & Validation Test Kit

This directory contains the self-contained benchmark and RPS saturation validation suite used to produce the **`d3439062`** (`jina-v2-opus-megakernel`) TPU v6e FP32 results (**`530 RPS`** at `1,024` tokens and **`350 RPS`** at `2,048` tokens under `p99 < 50 ms` SLA).

---

## Files in This Directory

| File | Purpose |
| :--- | :--- |
| [`start_server.sh`](./start_server.sh) | Starts the `/prompt_c2` adapter proxy (`:8000`) and `vllm serve` (`:8001`) with `USE_JINA_BERT_MEGAKERNEL=1`, `JINA_BERT_MEGAKERNEL_VERSION=v2`, `TPU_POOLING_FAST_PATH=1`, `JINA_BERT_DEVICE_POOLING=1`, `--dtype float32`, `--max-model-len 2048`, `--max-num-batched-tokens 2048`. |
| [`stock_adapter_proxy.py`](./stock_adapter_proxy.py) | Lightweight `aiohttp` proxy on `:8000` exposing `POST /prompt_c2` (`{"text": ...}`) and forwarding to `http://127.0.0.1:8001/v1/embeddings` with `truncate_prompt_tokens: 2048`. |
| [`build_token_pools.py`](./build_token_pools.py) | Builds deterministic pools of random-character prompts (`pools/pool_1024tok.json` and `pools/pool_2048tok.json`) verified with `AutoTokenizer` to tokenize to **exact `1,024` and `2,048` tokens** (including `[CLS]` and `[SEP]`). |
| [`k6_1kb_saturation_test_1024tok.js`](./k6_1kb_saturation_test_1024tok.js) | `k6` `constant-arrival-rate` saturation script for `1,024`-token payloads (`250` pre-allocated VUs, `60s` stage duration). |
| [`k6_2kb_saturation_test_2048tok.js`](./k6_2kb_saturation_test_2048tok.js) | `k6` `constant-arrival-rate` saturation script for `2,048`-token payloads (`200` pre-allocated VUs, `60s` stage duration). |
| [`analyze_k6_results.py`](./analyze_k6_results.py) | Parses raw `k6` `.ndjson` output and computes per-stage `achieved_rps`, `min`, `p50`, `avg`, `p90`, `p95`, `p99`, `max`, and error counts. |
| [`run_rps_saturation_test.py`](./run_rps_saturation_test.py) | End-to-end Python driver that runs Preflight token verification, Single-HTTP Batch (`N=1,4,8,16`), `k6` Concurrent (`C=1,4,8,16`), and `1KB` / `2KB` Dedicated RPS Saturation sweeps while scraping `vllm:prompt_tokens_total` and `vllm:request_success_total` on `:8001/metrics`. |

---

## Step-by-Step Instructions to Validate a PR Branch

### 1. Prerequisites
- **Hardware:** 1x TPU v6e chip (`ct6e-standard-1t` VM or GKE pod with `google.com/tpu: 1`).
- **Tools:** `k6` (v0.50+ installed in `PATH`) and Python packages `aiohttp`, `transformers`, `pytest`.

### 2. Run Offline Correctness & Step Microbenchmark (Optional Sanity Check)
From the root of your `tpu-inference` checkout with the megakernel PR changes installed (`pip install -e . --no-build-isolation`):

```bash
export USE_JINA_BERT_MEGAKERNEL=1
export JINA_BERT_MEGAKERNEL_VERSION=v2
export TPU_POOLING_FAST_PATH=1
export JINA_BERT_DEVICE_POOLING=1

# 1. Numerical & end-to-end TPU correctness tests (must pass 28/28, cosine_sim >= 0.999999)
pytest -v tests/models/jax/test_jina_bert.py

# 2. Offline step microbenchmark across all 10 shapes (B=1..128, S=128..2048)
python scripts/measure_jina_forward.py --out /tmp/measure_jina_forward.json
```

### 3. Start `vllm serve` + `/prompt_c2` Adapter Proxy
In Terminal 1 (on the TPU v6e host/pod):

```bash
cd models/JinaEmbedding/vLLM/megakernel/rps_saturation_test
bash start_server.sh
```
Wait until `/tmp/jina_megakernel_logs/vllm_serve.log` shows `Application startup complete` and `curl -s http://127.0.0.1:8000/health` returns `{"status": "ok"}`.

### 4. Run the RPS Saturation Test
In Terminal 2 (on a CPU load-generator node or the same host):

#### Option A — Quick Saturation Knee Check (~15 mins)
Tests the 10-RPS grid right around the `d3439062` saturation ceiling (`480, 500, 510, 520, 530, 540, 550 RPS` for `1,024` tokens; `280, 300, 320, 340, 350, 360, 370 RPS` for `2,048` tokens):

```bash
cd models/JinaEmbedding/vLLM/megakernel/rps_saturation_test
python3 run_rps_saturation_test.py \
    --knee-only \
    --base-url http://127.0.0.1:8000 \
    --metrics-url http://127.0.0.1:8001/metrics
```
*(If running `k6` from a separate CPU VM/pod, replace `127.0.0.1` with the TPU pod/VM IP).*

#### Option B — Full Dedicated RPS Saturation Sweep (`100 -> 550 RPS` for 1K & `90 -> 370 RPS` for 2K)
```bash
python3 run_rps_saturation_test.py \
    --saturation-only \
    --base-url http://127.0.0.1:8000 \
    --metrics-url http://127.0.0.1:8001/metrics
```

#### Option C — Complete 4-Part Benchmark Suite (Batch `N=1..16` + Concurrency `C=1..16` + Full 1K & 2K Saturation)
```bash
python3 run_rps_saturation_test.py \
    --base-url http://127.0.0.1:8000 \
    --metrics-url http://127.0.0.1:8001/metrics
```

---

## Expected Reference Results (`d3439062` on 1x TPU v6e Chip, FP32)

| Test | Payload (`Tokens`) | Expected Result (`d3439062` Reference) | SLA / Pass Criterion |
| :--- | :---: | :--- | :--- |
| **Preflight Check** | `1,024` & `2,048` tok | `prompt_tokens == 1024` / `2048`, `dim == 512` | Exact token count & 512-d output |
| **Single-HTTP Batch (`N=1..16`)** | `1,024` tok (`1KB`) | `P50 = 6.9 ms` (`N=1`), `9.2 ms` (`N=4`), `13.4 ms` (`N=8`), `23.7 ms` (`N=16`) | `0` errors |
| **Single-HTTP Batch (`N=1..16`)** | `2,048` tok (`2KB`) | `P50 = 8.4 ms` (`N=1`), `12.6 ms` (`N=4`), `21.2 ms` (`N=8`), `39.3 ms` (`N=16`) | `0` errors |
| **`k6` Concurrent (`VUS=8`)** | `1,024` tok (`1KB`) | **`559.1 req/s`** (`P50 = 13.7 ms`, `p99 = 19.6 ms`) | `0` errors |
| **`k6` Concurrent (`VUS=8`)** | `2,048` tok (`2KB`) | **`374.1 req/s`** (`P50 = 17.6 ms`, `p99 = 24.4 ms`) | `0` errors |
| **1KB Dedicated Saturation** | `1,024` tok (`1KB`) | **`530 RPS` PASS** (`P50 = 13.1 ms`, `p99 = 47.9 ms`); `540 RPS` SATURATED (`p99 = 121.1 ms`) | **`p99 < 50 ms`** (`>= 520–530 RPS`) |
| **2KB Dedicated Saturation** | `2,048` tok (`2KB`) | **`350 RPS` PASS** (`P50 = 11.7 ms`, `p99 = 25.2 ms`); `360 RPS` SATURATED (`p99 = 89.3 ms`) | **`p99 < 50 ms`** (`>= 340–350 RPS`) |
