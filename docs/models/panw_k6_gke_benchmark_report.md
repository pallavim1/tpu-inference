# PANW PoC: Jina Embeddings v2 on Google Cloud TPU v5e (GKE)

## Executive Summary

The **Jina Embeddings v2 Small** model (`jinaai/jina-embeddings-v2-small-en`) has been successfully deployed and benchmarked on a single **Google Cloud TPU v5e** chip (`v5litepod-1`, `ct5lp-hightpu-1t`) within the dedicated GKE cluster `pm-panw-jina-cluster` in `europe-west4-b`.

### Key Highlights
- **100% Request Success Rate**: Zero errors across thousands of requests under sustained concurrent load.
- **Strict SLA Adherence**: Every single payload size (`1024B`, `2048B`, `5120B`, and `7168B`) satisfies PANW's target of **P99 < 50 ms** across both 20 RPS and 40 RPS traffic profiles.
- **Superior Performance vs. Baseline**: Compared to PANW's baseline of ~40 ms on NVIDIA L4 GPU (`g2-standard-4`), TPU v5e delivers **14.7–25.3 ms P99** at 20 RPS (~1.6x–2.7x faster) and **15.2–42.5 ms P99** at 40 RPS.

---

## 1. Benchmark Results Summary

### 20 RPS Fixed Payload Size Suite (`SUITE=payload_size`)

| Scenario | Payload Size | Target RPS | Achieved RPS | Min (ms) | Avg (ms) | P50 (ms) | P90 (ms) | P95 (ms) | **P99 (ms)** | Max (ms) | Errors | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `prompt_c2_1024b` | **1 KB (1024 B)** | 20.0 | 20.04 | 11.0 | 12.2 | 11.9 | 13.1 | 13.7 | **14.7 ms** | 15.4 | 0 | **PASSED (<50ms)** |
| `prompt_c2_2048b` | **2 KB (2048 B)** | 20.0 | 20.03 | 16.0 | 17.6 | 17.4 | 18.7 | 19.2 | **20.1 ms** | 21.0 | 0 | **PASSED (<50ms)** |
| `prompt_c2_5120b` | **5 KB (5120 B)** | 20.0 | 20.03 | 19.8 | 21.4 | 21.3 | 22.0 | 22.3 | **23.1 ms** | 24.1 | 0 | **PASSED (<50ms)** |
| `prompt_c2_7168b` | **7 KB (7168 B)** | 20.0 | 20.03 | 22.3 | 23.7 | 23.6 | 24.5 | 24.8 | **25.3 ms** | 31.7 | 0 | **PASSED (<50ms)** |

---

### 40 RPS Stress Test Suite (`SUITE=payload_size @ 40 RPS`)

| Scenario | Payload Size | Target RPS | Achieved RPS | Min (ms) | Avg (ms) | P50 (ms) | P90 (ms) | P95 (ms) | **P99 (ms)** | Max (ms) | Errors | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `prompt_c2_1024b` | **1 KB (1024 B)** | 40.0 | 40.06 | 11.1 | 12.1 | 12.0 | 12.7 | 13.1 | **15.2 ms** | 17.4 | 0 | **PASSED (<50ms)** |
| `prompt_c2_2048b` | **2 KB (2048 B)** | 40.0 | 40.05 | 16.2 | 17.2 | 16.9 | 18.1 | 19.1 | **20.7 ms** | 22.0 | 0 | **PASSED (<50ms)** |
| `prompt_c2_5120b` | **5 KB (5120 B)** | 40.0 | 40.05 | 19.6 | 21.7 | 21.5 | 23.5 | 24.2 | **26.3 ms** | 28.6 | 0 | **PASSED (<50ms)** |
| `prompt_c2_7168b` | **7 KB (7168 B)** | 40.0 | 40.06 | 22.7 | 28.4 | 27.1 | 34.3 | 36.2 | **42.5 ms** | 61.0 | 0 | **PASSED (<50ms)** |

---

## 2. Infrastructure & Environment Configuration

- **GCP Project**: `northam-ce-mlai-tpu`
- **Zone**: `europe-west4-b`
- **GKE Cluster**: `pm-panw-jina-cluster`
- **VPC & Subnet**: `pm-panw-jina-vpc` / `pm-panw-jina-subnet` (`10.240.0.0/20`)
- **TPU Node Pool**: `pm-panw-jina-tpu-pool` (`ct5lp-hightpu-1t`, 1x TPU v5e chip)
- **Model**: `jinaai/jina-embeddings-v2-small-en` (512-dim embeddings)
- **Serving Engine**: `vllm-tpu` (`0.26.0`) with `tpu-inference` (`0.26.0`), JAX (`0.11.0`), and libtpu (`0.0.44`)
- **vLLM Serving Arguments**:
  ```bash
  vllm serve jinaai/jina-embeddings-v2-small-en \
    --runner pooling \
    --convert embed \
    --trust-remote-code \
    --max-model-len 2048 \
    --dtype float32 \
    --host 127.0.0.1 \
    --port 8001
  ```
- **PANW Adapter Proxy**: High-concurrency async adapter running on port 8000 handling `POST /prompt_c2` and `POST /mcp_c2`.

---

## 3. Generated Files & Artifacts

All benchmark raw data, structured JSON summaries, and formatted Excel reports have been exported:

| File | Description |
| :--- | :--- |
| [`raw_scenario_summary.xlsx`](file:///usr/local/google/home/pallaviam/panw-benchmark/results/raw_scenario_summary.xlsx) | Excel workbook containing per-scenario latency percentiles and charts (20 RPS) |
| [`raw_scenario_summary.json`](file:///usr/local/google/home/pallaviam/panw-benchmark/results/raw_scenario_summary.json) | Full structured JSON summary of 20 RPS run |
| [`raw_40rps_scenario_summary.xlsx`](file:///usr/local/google/home/pallaviam/panw-benchmark/results/raw_40rps_scenario_summary.xlsx) | Excel workbook containing per-scenario latency percentiles and charts (40 RPS) |
| [`raw_40rps_scenario_summary.json`](file:///usr/local/google/home/pallaviam/panw-benchmark/results/raw_40rps_scenario_summary.json) | Full structured JSON summary of 40 RPS run |
| [`raw.ndjson`](file:///usr/local/google/home/pallaviam/panw-benchmark/results/raw.ndjson) | Raw k6 per-request timing points (20 RPS) |
| [`raw_40rps.ndjson`](file:///usr/local/google/home/pallaviam/panw-benchmark/results/raw_40rps.ndjson) | Raw k6 per-request timing points (40 RPS) |
