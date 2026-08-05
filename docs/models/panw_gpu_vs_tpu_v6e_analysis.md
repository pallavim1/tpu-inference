# Jina Embeddings v2: PANW GPU vs. Google Cloud TPU v6e Benchmark Comparison & Analysis

## 1. Overview
This document provides a comparative analysis between **Palo Alto Networks (PANW) GPU benchmark results** and **Google Cloud TPU v6e (Trillium)** benchmark results for `jinaai/jina-embeddings-v2-small-en` (and related JinaBert embedding models).

It highlights the key differences in evaluation methodology, performance metrics, hardware utilization, and outlines the required test suite enhancements to conduct a 1:1 "apples-to-apples" RPS load benchmark on TPU v6e.

---

## 2. PANW GPU Benchmark Results Summary

PANW evaluated Jina embedding serving under varying **Target Requests Per Second (RPS)** (1 to 40 RPS) across 4 payload sizes (1K, 2K, 5K, 7K bytes).

| Scenario | Payload Size | Approx Tokens | Target RPS | Achieved RPS | p50 Latency (ms) | p90 Latency (ms) | p99 Latency (ms) | Max Latency (ms) | GPU Util (%) | GPU RAM (GB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `prompt_c2_1024b` | 1K (1024B) | ~200 | 1 - 40 RPS | 1.02 - 40.02 | **17.2 - 18.9** | 18.5 - 20.4 | 22.8 - 26.8 | 24.8 - 169.9 | 0.0% - 17.0% | 2.72 / 4.71 |
| `prompt_c2_2048b` | 2K (2048B) | ~400 | 1 - 40 RPS | 1.02 - 40.02 | **23.3 - 25.0** | 25.1 - 27.5 | 31.4 - 46.5 | 35.4 - 193.0 | 0.0% - 30.0% | 2.72 / 4.71 |
| `prompt_c2_5120b` | 5K (5120B) | ~1,000 | 1 - 30 RPS | 1.02 - 30.02 | **30.8 - 32.8** | 34.8 - 36.3 | 40.4 - 61.5 | 51.1 - 194.8 | 0.0% - 22.0% | 2.72 / 4.71 |
| `prompt_c2_5120b` | 5K (5120B) | ~1,000 | 40 RPS | 40.02 | 46.3 | 66.8 | 145.3 | 283.3 | 30.2% | 2.72 / 4.71 |
| `prompt_c2_7168b` | 7K (7168B) | ~1,400 | 1 - 20 RPS | 1.02 - 20.02 | **35.9 - 37.7** | 39.4 - 42.6 | 45.5 - 70.2 | 46.0 - 181.2 | 0.0% - 15.0% | 2.72 / 4.71 |
| `prompt_c2_7168b` | 7K (7168B) | ~1,400 | 30 RPS | 30.01 | 50.4 | 65.7 | 115.7 | 243.2 | 19.3% | 2.72 / 4.71 |
| `prompt_c2_7168b` | 7K (7168B) | ~1,400 | 40 RPS | **33.14 (Sat.)** | **191.2** | **603.0** | **1298.8** | 2181.1 | 24.9% | 2.72 / 4.71 |

### Key Observations on GPU:
1. **Low Concurrency Latency**: At 1-10 RPS, p50 latency ranges from **18.2 ms** (1K) to **36.6 ms** (7K).
2. **Endpoint Saturation**: On large payloads (7K / 1,400 tokens), the GPU endpoint saturates at ~33 RPS, causing p90 latency to jump from 65.7 ms to **603 ms** and p99 to **1,298 ms**.
3. **Resource Footprint**: Consumes ~2.72 GB GPU memory (out of 4.71 GB slice limit) and under 0.5 CPU cores.

---

## 3. Initial TPU v6e Benchmark Results (vLLM Engine)

Our initial TPU v6e tests measured raw engine throughput and latency across fixed batch sizes ($BS = 16, 32, 64, 128$) and token sequence lengths ($128, 512, 1000$).

| Model | Batch Size | Sequence Length | Throughput (emb/sec) | p50 Latency (ms) | p90 Latency (ms) | p99 Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `jina-embeddings-v2-small-en` | **16** | **128** | **738.92** | **21.62** | 21.88 | 22.29 |
| `jina-embeddings-v2-small-en` | 16 | 512 | 137.73 | 116.19 | 116.25 | 116.27 |
| `jina-embeddings-v2-small-en` | 16 | 1000 | 71.98 | 222.31 | 222.65 | 223.25 |
| `jina-embeddings-v2-small-en` | **32** | 128 | 665.68 | 46.10 | 46.57 | 71.77 |
| `jina-embeddings-v2-small-en` | 32 | 512 | 140.96 | 226.95 | 227.13 | 227.37 |
| `jina-embeddings-v2-small-en` | 32 | 1000 | 72.62 | 438.17 | 440.73 | 468.47 |
| `jina-embeddings-v2-small-en` | **64** | 128 | 486.13 | 131.64 | 131.74 | 131.83 |
| `jina-embeddings-v2-small-en` | 64 | 512 | 142.25 | 449.94 | 450.08 | 450.25 |
| `jina-embeddings-v2-small-en` | 64 | 1000 | 73.43 | 871.52 | 872.18 | 872.37 |
| `jina-embeddings-v2-small-en` | **128** | 128 | 494.85 | 258.55 | 259.28 | 259.66 |
| `jina-embeddings-v2-small-en` | 128 | 512 | 142.94 | 895.49 | 896.16 | 896.59 |
| `jina-embeddings-v2-small-en` | 128 | 1000 | 73.65 | 1737.49 | 1739.13 | 1739.62 |

---

## 4. Architectural & Methodological Comparison

| Feature / Metric | PANW GPU Benchmark | Current TPU v6e Benchmark |
| :--- | :--- | :--- |
| **Benchmark Mode** | Online RPS Load Testing (Locust/HTTP) | Direct Offline Batch Engine Testing |
| **Input Specification** | Exact byte sizes (1K, 2K, 5K, 7K bytes) | Fixed token lengths (128, 512, 1000 tokens) |
| **Concurrency Model** | Variable requests/sec (1 to 40 RPS) | Synchronous fixed batch sizes (16 to 128) |
| **Serving Endpoint** | HTTP OpenAI Embeddings API (`/v1/embeddings`) | Direct `vLLM` Python Engine (`LLM.embed()`) |
| **Max Context Length** | ~1,400 tokens (7K bytes) | Capped at 1,024 tokens |

---

## 5. Required Test Modifications for 1:1 Benchmark Alignment

To execute a direct "apples-to-apples" comparison matching PANW's GPU suite:

1. **Implement Asynchronous RPS Load Generator**:
   - Deploy an HTTP load test script (using `aiohttp` or `vLLM`'s `benchmark_serving.py`) targeting the vLLM OpenAI API endpoint (`http://localhost:8000/v1/embeddings`).
   - Run tests at fixed Target RPS levels: **1, 5, 7, 10, 20, 30, and 40 RPS**.

2. **Match Payload String Byte Sizes**:
   - Construct payload inputs by byte length matching PANW specifications:
     - **1K**: 1,024 bytes
     - **2K**: 2,048 bytes
     - **5K**: 5,120 bytes
     - **7K**: 7,168 bytes

3. **Expand TPU Context Window**:
   - Launch `vllm serve` with `--max-model-len 2048` to support the 7K payload (~1,400 tokens) without triggering sequence context length errors.

4. **Collect Standard Metric Parity**:
   - Record **Achieved RPS**, **Error Count**, **Min / p50 / Avg / p90 / p95 / p99 / Max Latency (ms)**, and track CPU/RAM utilization.
