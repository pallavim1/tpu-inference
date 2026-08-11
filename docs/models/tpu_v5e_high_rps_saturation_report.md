# TPU v5e High-RPS Saturation & Scaling Benchmark Report
## Model: Jina Embeddings v2 Small (`jinaai/jina-embeddings-v2-small-en`)
## Target SLO: P99 Round-Trip Latency < 50 ms

---

## 1. Executive Summary

To determine the maximum throughput capacity and exact latency saturation boundaries of a **single Google Cloud TPU v5e chip** (`ct5lp-hightpu-1t`), an incremental load sweep was executed from **50 RPS to 90 RPS** across all four Palo Alto Networks (PANW) payload sizes (`1024B`, `2048B`, `5120B`, and `7168B`).

### Key Findings & Max Capacity Limits (P99 < 50ms SLA):

| Payload Size | Character / Byte Range | Max RPS Passing P99 < 50ms SLA | Achieved P99 at Max RPS | Saturation Point (P99 > 50ms) |
| :--- | :--- | :---: | :---: | :---: |
| **1 KB (1024 B)** | ~200 Tokens | **> 90 RPS** | **16.7 ms** | *Not saturated at 90 RPS (Sub-17ms P99)* |
| **2 KB (2048 B)** | ~400 Tokens | **> 90 RPS** | **28.9 ms** | *Not saturated at 90 RPS (Sub-29ms P99)* |
| **5 KB (5120 B)** | ~1,000 Tokens | **80 RPS** | **49.5 ms** | **90 RPS** (Saturated @ 1,157 ms P50) |
| **7 KB (7168 B)** | ~1,400 Tokens | **60 RPS** | **41.8 ms** | **70 RPS** (Exceeded @ 59.4 ms P99) |

---

## 2. Comprehensive Per-RPS Benchmark Breakdown

### 50 RPS Stage Breakdown (`PAYLOAD_RPS=50`)

| Scenario | Payload Size | Target RPS | Achieved RPS | Min (ms) | Avg (ms) | P50 (ms) | P90 (ms) | P95 (ms) | **P99 (ms)** | Max (ms) | Errors | SLA Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `prompt_c2_1024b` | **1 KB (1024 B)** | 50.0 | 50.06 | 10.6 | 11.5 | 11.3 | 12.1 | 12.6 | **15.2 ms** | 16.8 | 0 | **PASSED (<50ms)** |
| `prompt_c2_2048b` | **2 KB (2048 B)** | 50.0 | 50.06 | 15.6 | 16.7 | 16.4 | 17.7 | 18.6 | **20.8 ms** | 24.8 | 0 | **PASSED (<50ms)** |
| `prompt_c2_5120b` | **5 KB (5120 B)** | 50.0 | 50.04 | 19.2 | 23.9 | 23.4 | 27.3 | 28.5 | **33.8 ms** | 54.5 | 0 | **PASSED (<50ms)** |
| `prompt_c2_7168b` | **7 KB (7168 B)** | 50.0 | 50.05 | 22.9 | 28.9 | 28.9 | 31.8 | 33.6 | **35.6 ms** | 36.8 | 0 | **PASSED (<50ms)** |

---

### 60 RPS Stage Breakdown (`PAYLOAD_RPS=60`)

| Scenario | Payload Size | Target RPS | Achieved RPS | Min (ms) | Avg (ms) | P50 (ms) | P90 (ms) | P95 (ms) | **P99 (ms)** | Max (ms) | Errors | SLA Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `prompt_c2_1024b` | **1 KB (1024 B)** | 60.0 | 60.05 | 10.6 | 11.5 | 11.3 | 11.9 | 12.4 | **14.9 ms** | 16.3 | 0 | **PASSED (<50ms)** |
| `prompt_c2_2048b` | **2 KB (2048 B)** | 60.0 | 60.08 | 15.6 | 17.2 | 16.8 | 18.8 | 20.4 | **22.4 ms** | 23.8 | 0 | **PASSED (<50ms)** |
| `prompt_c2_5120b` | **5 KB (5120 B)** | 60.0 | 60.06 | 20.1 | 26.3 | 25.4 | 30.4 | 31.3 | **35.4 ms** | 59.3 | 0 | **PASSED (<50ms)** |
| `prompt_c2_7168b` | **7 KB (7168 B)** | 60.0 | 60.06 | 22.1 | 30.1 | 29.6 | 35.2 | 36.8 | **41.8 ms** | 68.8 | 0 | **PASSED (<50ms)** |

---

### 70 RPS Stage Breakdown (`PAYLOAD_RPS=70`)

| Scenario | Payload Size | Target RPS | Achieved RPS | Min (ms) | Avg (ms) | P50 (ms) | P90 (ms) | P95 (ms) | **P99 (ms)** | Max (ms) | Errors | SLA Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `prompt_c2_1024b` | **1 KB (1024 B)** | 70.0 | 70.07 | 10.6 | 11.6 | 11.4 | 12.0 | 12.6 | **15.5 ms** | 21.5 | 0 | **PASSED (<50ms)** |
| `prompt_c2_2048b` | **2 KB (2048 B)** | 70.0 | 70.06 | 15.9 | 19.6 | 19.4 | 21.2 | 23.0 | **27.1 ms** | 34.9 | 0 | **PASSED (<50ms)** |
| `prompt_c2_5120b` | **5 KB (5120 B)** | 70.0 | 70.08 | 20.4 | 25.5 | 24.7 | 29.6 | 30.4 | **32.3 ms** | 53.4 | 0 | **PASSED (<50ms)** |
| `prompt_c2_7168b` | **7 KB (7168 B)** | 70.0 | 70.06 | 24.2 | 37.2 | 37.6 | 45.7 | 49.7 | **59.4 ms** | 62.0 | 0 | **EXCEEDED (>50ms)** |

---

### 80 RPS Stage Breakdown (`PAYLOAD_RPS=80`)

| Scenario | Payload Size | Target RPS | Achieved RPS | Min (ms) | Avg (ms) | P50 (ms) | P90 (ms) | P95 (ms) | **P99 (ms)** | Max (ms) | Errors | SLA Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `prompt_c2_1024b` | **1 KB (1024 B)** | 80.0 | 80.08 | 10.5 | 11.6 | 11.4 | 12.3 | 12.8 | **15.7 ms** | 19.1 | 0 | **PASSED (<50ms)** |
| `prompt_c2_2048b` | **2 KB (2048 B)** | 80.0 | 80.06 | 16.2 | 18.9 | 18.2 | 21.9 | 23.7 | **26.3 ms** | 32.8 | 0 | **PASSED (<50ms)** |
| `prompt_c2_5120b` | **5 KB (5120 B)** | 80.0 | 79.93 | 0.8 | 30.5 | 29.5 | 38.8 | 41.1 | **49.5 ms** | 58.8 | 1 | **PASSED (<50ms)** |
| `prompt_c2_7168b` | **7 KB (7168 B)** | 80.0 | 75.09 | 0.9 | 818.7 | 820.2 | 1420.5 | 1474.2 | **1523.2 ms** | 1540.9 | 2 | **SATURATED** |

---

### 90 RPS Stage Breakdown (`PAYLOAD_RPS=90`)

| Scenario | Payload Size | Target RPS | Achieved RPS | Min (ms) | Avg (ms) | P50 (ms) | P90 (ms) | P95 (ms) | **P99 (ms)** | Max (ms) | Errors | SLA Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `prompt_c2_1024b` | **1 KB (1024 B)** | 90.0 | 90.07 | 10.8 | 12.3 | 12.1 | 13.4 | 14.2 | **16.7 ms** | 19.9 | 0 | **PASSED (<50ms)** |
| `prompt_c2_2048b` | **2 KB (2048 B)** | 90.0 | 90.06 | 15.8 | 19.7 | 18.7 | 24.3 | 26.2 | **28.9 ms** | 34.5 | 0 | **PASSED (<50ms)** |
| `prompt_c2_5120b` | **5 KB (5120 B)** | 90.0 | 82.16 | 23.2 | 1152.2 | 1157.6 | 1997.4 | 2048.5 | **2094.6 ms** | 2109.9 | 0 | **SATURATED** |
| `prompt_c2_7168b` | **7 KB (7168 B)** | 90.0 | 75.06 | 0.9 | 2099.8 | 2496.7 | 3166.8 | 3219.8 | **3273.1 ms** | 3292.1 | 6 | **SATURATED** |

---

## 3. Latency Progression Summary (P99 vs. RPS)

| Payload Size | 20 RPS | 40 RPS | 50 RPS | 60 RPS | 70 RPS | 80 RPS | 90 RPS | SLA Max Passing Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB (1024 B)** | 14.7 ms | 15.2 ms | 15.2 ms | 14.9 ms | 15.5 ms | 15.7 ms | **16.7 ms** | **> 90 RPS** (Sub-17ms P99) |
| **2 KB (2048 B)** | 20.1 ms | 20.7 ms | 20.8 ms | 22.4 ms | 27.1 ms | 26.3 ms | **28.9 ms** | **> 90 RPS** (Sub-29ms P99) |
| **5 KB (5120 B)** | 23.1 ms | 26.3 ms | 33.8 ms | 35.4 ms | 32.3 ms | **49.5 ms** | *2,094 ms (Sat)* | **80 RPS** (49.5 ms P99) |
| **7 KB (7168 B)** | 25.3 ms | 42.5 ms | 35.6 ms | **41.8 ms** | *59.4 ms* | *1,523 ms (Sat)* | *3,273 ms (Sat)* | **60 RPS** (41.8 ms P99) |

---

## 4. Hardware Sizing & Capacity Recommendations for PANW

1. **For 1 KB & 2 KB Payload Dominant Traffic**:
   - A single TPU v5e chip comfortably serves up to **90+ RPS** with ultra-low latencies (**16.7 ms P99** on 1K, **28.9 ms P99** on 2K).
   - Capacity headroom: **> 4.5x** over PANW baseline average requirement (20 RPS) and **> 2.25x** over peak (40 RPS).

2. **For 5 KB Payload Traffic**:
   - Maximum sustainable capacity under 50 ms P99 SLA is **80 RPS** (**49.5 ms P99**).
   - Capacity headroom: **4.0x** over average traffic (20 RPS) and **2.0x** over peak traffic (40 RPS).

3. **For 7 KB Payload Traffic**:
   - Maximum sustainable capacity under 50 ms P99 SLA is **60 RPS** (**41.8 ms P99**).
   - Capacity headroom: **3.0x** over average traffic (20 RPS) and **1.5x** over peak traffic (40 RPS).
   - Saturation begins at 70 RPS (59.4 ms P99).
