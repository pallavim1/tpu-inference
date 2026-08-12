# Live Benchmark Report: Jina Embeddings v2 on Cloud TPU v5e (FP16 / BF16 Precision)

**Target Model**: `jinaai/jina-embeddings-v2-small-en` (512-dim)  
**Precision**: `bfloat16` (Native 16-bit Floating Point on Cloud TPU v5e)  
**SLA Criterion**: Strict Tail Round-Trip Latency $P_{99} < 50\text{ ms}$ across GKE Nodepools  
**Client Environment**: Dedicated CPU Nodepool (`n2-standard-8`, `cpu-benchmark-pool`)  
**Server Environment**: Google Cloud TPU v5e (`ct5lp-hightpu-1t`, `pm-panw-jina-tpu-pool`)  
**Network Path**: Real GKE Cluster Network (`europe-west4-b`) via Kubernetes ClusterIP Service  
**Stage Duration**: 60 seconds per load point  

---

## 1. Executive Summary & Precision Comparison (FP16 vs. FP32 vs. NVIDIA L4)

1. **Substantial Tail Latency ($P_{99}$) Reduction with 16-bit Precision**:
   * Operating under native 16-bit precision (`bfloat16`) reduces execution overhead and kernel memory bandwidth demands.
   * **7 KB Payload at 80 RPS**: $P_{99}$ drops from **49.10 ms** (FP32) to **48.00 ms** (FP16), providing increased headroom below the 50 ms SLA threshold.
   * **5 KB Payload at 80 RPS**: $P_{99}$ drops from **41.90 ms** (FP32) to **39.50 ms** (FP16).
   * **2 KB Payload at 80 RPS**: $P_{99}$ drops from **26.30 ms** (FP32) to **24.80 ms** (FP16).
   * **1 KB Payload at 160 RPS**: $P_{99}$ drops from **30.60 ms** (FP32) to **28.90 ms** (FP16).

2. **Throughput Multipliers over NVIDIA L4 Baseline ($P_{99} < 50\text{ ms}$)**:
   * **1 KB**: **160 RPS** vs. 40 RPS on L4 (**4.0x Throughput**)
   * **2 KB**: **80 RPS** vs. 40 RPS on L4 (**2.0x Throughput**)
   * **5 KB**: **80 RPS** vs. 20 RPS on L4 (**4.0x Throughput**)
   * **7 KB**: **80 RPS** vs. 10 RPS on L4 (**8.0x Throughput**)

---

## 2. Comprehensive Live Test Results Table (FP16 / BF16)

| Scenario / Payload | Target RPS | Achieved RPS | $P_{50}$ (ms) | $P_{90}$ (ms) | $P_{95}$ (ms) | $P_{99}$ Latency (ms) | Max Latency (ms) | SLA Status ($P_{99} < 50\text{ ms}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB** (`1024B`) | 50 | 50.10 | 12.00 | 12.60 | 13.10 | **14.20** | 205.20 | ✅ PASS |
| **1 KB** (`1024B`) | 60 | 60.02 | 11.90 | 12.50 | 12.80 | **14.00** | 22.20 | ✅ PASS |
| **1 KB** (`1024B`) | 70 | 70.03 | 11.80 | 12.40 | 12.70 | **13.90** | 20.10 | ✅ PASS |
| **1 KB** (`1024B`) | 80 | 80.03 | 11.90 | 12.50 | 12.90 | **14.40** | 22.80 | ✅ PASS |
| **1 KB** (`1024B`) | 90 | 90.03 | 12.30 | 13.50 | 14.00 | **15.60** | 19.30 | ✅ PASS |
| **1 KB** (`1024B`) | 100 | 100.03 | 13.50 | 14.80 | 15.00 | **15.60** | 19.70 | ✅ PASS |
| **1 KB** (`1024B`) | 120 | 120.03 | 13.40 | 14.30 | 14.80 | **16.90** | 24.50 | ✅ PASS |
| **1 KB** (`1024B`) | 140 | 140.03 | 14.20 | 18.30 | 19.20 | **20.90** | 31.10 | ✅ PASS |
| **1 KB** (`1024B`) | 160 | 160.03 | 20.20 | 26.50 | 27.40 | **28.90** | 32.60 | ✅ PASS |
| **1 KB** (`1024B`) | 180 | 176.83 | 389.40 | 1017.90 | 1051.80 | **1127.40** | 1137.60 | ⚠️ SATURATED (HW Limit ~177 RPS) |
| **2 KB** (`2048B`) | 50 | 50.02 | 17.10 | 17.90 | 18.30 | **19.70** | 25.00 | ✅ PASS |
| **2 KB** (`2048B`) | 60 | 60.03 | 17.70 | 19.10 | 19.70 | **21.60** | 27.90 | ✅ PASS |
| **2 KB** (`2048B`) | 70 | 70.02 | 19.90 | 21.00 | 21.30 | **22.20** | 29.70 | ✅ PASS |
| **2 KB** (`2048B`) | 80 | 80.01 | 19.00 | 21.10 | 22.40 | **24.80** | 30.80 | ✅ PASS |
| **2 KB** (`2048B`) | 90 | 90.01 | 22.90 | 65.90 | 69.80 | **77.10** | 79.80 | ⚠️ SATURATED ($P_{99} > 50\text{ ms}$) |
| **2 KB** (`2048B`) | 95 | 90.83 | 1439.80 | 2382.70 | 2438.50 | **2457.00** | 2465.00 | ⚠️ SATURATED (HW Limit ~91 RPS) |
| **5 KB** (`5120B`) | 50 | 50.02 | 22.40 | 25.20 | 25.90 | **29.40** | 43.30 | ✅ PASS |
| **5 KB** (`5120B`) | 60 | 60.01 | 23.00 | 27.20 | 27.90 | **29.00** | 33.60 | ✅ PASS |
| **5 KB** (`5120B`) | 70 | 70.02 | 24.20 | 28.80 | 29.70 | **32.60** | 46.70 | ✅ PASS |
| **5 KB** (`5120B`) | 80 | 80.01 | 28.90 | 35.80 | 36.90 | **39.50** | 52.60 | ✅ PASS |
| **5 KB** (`5120B`) | 90 | 88.77 | 370.90 | 823.90 | 873.90 | **902.20** | 915.50 | ⚠️ SATURATED (HW Limit ~89 RPS) |
| **7 KB** (`7168B`) | 50 | 50.01 | 24.10 | 28.10 | 29.50 | **33.60** | 37.70 | ✅ PASS |
| **7 KB** (`7168B`) | 60 | 60.01 | 27.70 | 31.20 | 31.80 | **33.20** | 38.40 | ✅ PASS |
| **7 KB** (`7168B`) | 70 | 70.01 | 27.40 | 35.50 | 37.20 | **41.20** | 52.90 | ✅ PASS |
| **7 KB** (`7168B`) | 80 | 80.01 | 31.30 | 37.70 | 40.10 | **48.00** | 61.10 | ✅ PASS |
| **7 KB** (`7168B`) | 90 | 88.26 | 555.50 | 1171.50 | 1214.80 | **1241.00** | 1261.50 | ⚠️ SATURATED (HW Limit ~88 RPS) |

---

## 3. Tiered Performance-per-Dollar Analysis by Payload Size

### Pricing Tiers Confirmed by PANW:
* **NVIDIA L4 (`g2-standard-4`)**: 1-Year CUD = **$0.45/hr** | 3-Year CUD = **$0.32/hr**
* **TPU v5e (`ct5lp-hightpu-1t`)**: 1-Year CUD = **$0.84/hr** | 3-Year CUD = **$0.54/hr**
* **TPU v6e (`ct6e-standard-4t`)**: 1-Year CUD = **$1.89/hr** | 3-Year CUD = **$1.22/hr**

---

### A. 1 KB Payload (`1024B`) Tier Analysis

| Metric | NVIDIA L4 GPU | Cloud TPU v5e (FP16) | Advantage (TPU vs. L4) |
| :--- | :---: | :---: | :---: |
| **Max Certified RPS ($P_{99} < 50\text{ ms}$)** | **40 RPS** | **160 RPS** | **4.0x Throughput** |
| **$P_{99}$ Latency at Capacity** | 22.80 ms | 28.90 ms | Meets Strict SLA |
| **Hourly Cost (1-Year CUD)** | $0.45 / hr | $0.84 / hr | — |
| **Hourly Cost (3-Year CUD)** | $0.32 / hr | $0.54 / hr | — |
| **RPS per Dollar (1-Year CUD)** | 88.9 RPS / $ | **190.5 RPS / $** | **+114% Cost Efficiency** |
| **RPS per Dollar (3-Year CUD)** | 125.0 RPS / $ | **296.3 RPS / $** | **+137% Cost Efficiency** |
| **Cost per 1M Embeddings (1-Yr)** | $3.125 | **$1.458** | **53.3% Cost Savings** |
| **Cost per 1M Embeddings (3-Yr)** | $2.222 | **$0.938** | **57.8% Cost Savings** |

---

### B. 2 KB Payload (`2048B`) Tier Analysis

| Metric | NVIDIA L4 GPU | Cloud TPU v5e (FP16) | Advantage (TPU vs. L4) |
| :--- | :---: | :---: | :---: |
| **Max Certified RPS ($P_{99} < 50\text{ ms}$)** | **40 RPS** | **80 RPS** | **2.0x Throughput** |
| **$P_{99}$ Latency at Capacity** | 46.50 ms | 24.80 ms | **46.7% Lower Tail Latency** |
| **RPS per Dollar (1-Year CUD)** | 88.9 RPS / $ | **95.2 RPS / $** | **+7.1% Cost Efficiency** |
| **RPS per Dollar (3-Year CUD)** | 125.0 RPS / $ | **148.1 RPS / $** | **+18.5% Cost Efficiency** |
| **Cost per 1M Embeddings (1-Yr)** | $3.125 | **$2.917** | **6.7% Cost Savings** |
| **Cost per 1M Embeddings (3-Yr)** | $2.222 | **$1.875** | **15.6% Cost Savings** |

---

### C. 5 KB Payload (`5120B`) Tier Analysis

| Metric | NVIDIA L4 GPU | Cloud TPU v5e (FP16) | Advantage (TPU vs. L4) |
| :--- | :---: | :---: | :---: |
| **Max Certified RPS ($P_{99} < 50\text{ ms}$)** | **20 RPS** | **80 RPS** | **4.0x Throughput** |
| **$P_{99}$ Latency at Capacity** | 49.10 ms | 39.50 ms | **19.6% Lower Tail Latency** |
| **RPS per Dollar (1-Year CUD)** | 44.4 RPS / $ | **95.2 RPS / $** | **+114% Cost Efficiency** |
| **RPS per Dollar (3-Year CUD)** | 62.5 RPS / $ | **148.1 RPS / $** | **+137% Cost Efficiency** |
| **Cost per 1M Embeddings (1-Yr)** | $6.250 | **$2.917** | **53.3% Cost Savings** |
| **Cost per 1M Embeddings (3-Yr)** | $4.444 | **$1.875** | **57.8% Cost Savings** |

---

### D. 7 KB Payload (`7168B`) Tier Analysis

| Metric | NVIDIA L4 GPU | Cloud TPU v5e (FP16) | Advantage (TPU vs. L4) |
| :--- | :---: | :---: | :---: |
| **Max Certified RPS ($P_{99} < 50\text{ ms}$)** | **10 RPS** | **80 RPS** | **8.0x Throughput** |
| **$P_{99}$ Latency at Capacity** | 46.40 ms | 48.00 ms | Meets Strict SLA |
| **RPS per Dollar (1-Year CUD)** | 22.2 RPS / $ | **95.2 RPS / $** | **+328% Cost Efficiency** |
| **RPS per Dollar (3-Year CUD)** | 31.3 RPS / $ | **148.1 RPS / $** | **+374% Cost Efficiency** |
| **Cost per 1M Embeddings (1-Yr)** | $12.500 | **$2.917** | **76.7% Cost Savings** |
| **Cost per 1M Embeddings (3-Yr)** | $8.889 | **$1.875** | **78.9% Cost Savings** |

---

## 4. Architectural Summary for Palo Alto Networks

1. **Large Payload Scaling Advantage**:
   On longer payloads (5 KB and 7 KB), NVIDIA L4 quickly saturates at 20 RPS and 10 RPS due to GPU memory bandwidth and serial execution constraints. In contrast, Cloud TPU v5e matrix multiply units (MXUs) process 80 RPS effortlessly across all sequence lengths.
2. **Cluster Footprint Reduction**:
   Serving 1,000 RPS of mixed 5KB–7KB ATP traffic requires:
   * **NVIDIA L4**: **100 GPU pods** ($45.00/hr on 1-yr CUD)
   * **Cloud TPU v5e**: **13 TPU pods** ($10.92/hr on 1-yr CUD)
   * **Annual Infrastructure Savings**: **~$298,500 / year** (75.7% total TCO reduction)
