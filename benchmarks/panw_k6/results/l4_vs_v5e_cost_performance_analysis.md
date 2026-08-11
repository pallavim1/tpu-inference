# $/Performance & TCO Analysis: NVIDIA L4 vs. Google Cloud TPU v5e

**Model:** `jinaai/jina-embeddings-v2-small-en` (512-dim Embeddings)  
**Workload:** PANW PoC Endpoint Benchmarking (`POST /prompt_c2`)  
**Target SLA:** P99 Latency < 50 ms  

---

## 1. Executive Summary

This document presents a comprehensive cost-to-performance ($/perf) and Total Cost of Ownership (TCO) analysis comparing **NVIDIA L4 GPU** (`g2-standard-4`) and **Google Cloud TPU v5e** (`ct5lp-hightpu-1t` / `v5litepod-1`), with additional projections for **Cloud TPU v6e (Trillium)**.

The analysis is based on empirical benchmarking results across four payload sizes (**1 KB**, **2 KB**, **5 KB**, and **7 KB**) at varying traffic concurrency levels (**1 to 40 RPS**).

```
P99 Latency at 7 KB Payload @ 40 RPS (Target SLA: < 50 ms):
NVIDIA L4: [████████████████████████████████████████████████████] 1,298.8 ms (FAILED SLA)
TPU v5e:   [█] 42.5 ms (PASSED SLA, 30.6x faster)

Throughput Achieved at 7 KB Payload @ 40 RPS Target:
NVIDIA L4: [████████████████] 33.14 RPS (17.2% dropped load / queue buildup)
TPU v5e:   [████████████████████] 40.06 RPS (100% full capacity delivered)
```

### Key Takeaways:
1. **SLA Compliance & Stability at Scale**:
   - **Small Payloads (1K–2K)**: Both L4 and TPU v5e successfully meet the **P99 < 50 ms** SLA at 20 RPS and 40 RPS. TPU v5e delivers **~1.5x–2.2x faster P99 latency** with significantly tighter jitter bounds (Max spike: 17–22 ms on v5e vs. 170–193 ms on L4).
   - **Large Payloads (5K–7K)**: L4 experiences severe latency degradation and saturation:
     - At **5K @ 40 RPS**, L4 P99 degrades to **145.3 ms** (nearly 3x over SLA), whereas TPU v5e remains at **26.3 ms** (**5.5x faster**).
     - At **7K @ 40 RPS**, L4 **completely collapses**: achieved RPS drops to **33.14** (17% dropped/queued load), and P99 explodes to **1,298.8 ms** (26x over SLA). TPU v5e effortlessly maintains **40.06 RPS** with **42.5 ms P99** (**30.6x faster**).
2. **SLA-Constrained Capacity per Accelerator**:
   - **1 KB**: L4 = 40 RPS | TPU v5e = 40+ RPS (1.0x capacity ratio)
   - **2 KB**: L4 = ~35 RPS | TPU v5e = 40+ RPS (1.1x capacity ratio)
   - **5 KB**: L4 = **20 RPS** | TPU v5e = **40+ RPS** (**2.0x capacity on v5e**)
   - **7 KB**: L4 = **10 RPS** | TPU v5e = **40+ RPS** (**4.0x capacity on v5e**)
3. **Bottom-Line TCO (1,000 RPS Production Workload)**:
   - For **5 KB payloads**, TPU v5e delivers **15.6% net monthly savings** ($9,855/mo vs. $11,680/mo on 3-yr CUD) and **50% fewer nodes** (25 vs. 50).
   - For **7 KB payloads**, TPU v5e delivers **57.8% net monthly savings** ($9,855/mo vs. $23,360/mo on 3-yr CUD) and **75% fewer nodes** (25 vs. 100).
   - For a **Uniform Payload Mix (25% each)**, TPU v5e delivers **17.3% net monthly savings** ($9,855/mo vs. $11,914/mo) and **51% fewer nodes** (25 vs. 51).

---

## 2. Pricing & Cost Model

The analysis applies Google Cloud Committed Use Discount (CUD) rates per chip-hour:

| Accelerator | 1-Year CUD ($/hr) | 3-Year CUD ($/hr) | Monthly Cost per Chip (1-Yr CUD) | Monthly Cost per Chip (3-Yr CUD) |
| :--- | :---: | :---: | :---: | :---: |
| **NVIDIA L4 GPU** (`g2-standard-4`) | **$0.45** | **$0.32** | $328.50 | $233.60 |
| **Cloud TPU v5e** (`ct5lp-hightpu-1t`) | **$0.84** | **$0.54** | $613.20 | $394.20 |
| **Cloud TPU v6e (Trillium)** | **$1.89** | **$1.22** | $1,379.70 | $890.60 |

*Note: Monthly costs calculated at standard 730 hours/month.*

---

## 3. Side-by-Side Performance & Latency Matrix

### 3.1 20 RPS Sustained Traffic Profile

| Payload | Target RPS | Accelerator | Achieved RPS | Min (ms) | P50 (ms) | Avg (ms) | P90 (ms) | P95 (ms) | **P99 (ms)** | Max (ms) | **P99 SLA (<50ms)** |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB** | 20 | **NVIDIA L4** | 20.02 | 14.9 | 17.7 | 17.9 | 19.3 | 20.1 | **24.3** | 128.3 | ✅ PASS |
| | 20 | **TPU v5e** | 20.04 | 11.0 | **11.9** | 12.2 | 13.1 | 13.7 | **14.7** | 15.4 | ✅ **PASS (1.65x faster)** |
| **2 KB** | 20 | **NVIDIA L4** | 20.02 | 19.8 | 23.6 | 23.9 | 25.5 | 26.8 | **31.4** | 139.6 | ✅ PASS |
| | 20 | **TPU v5e** | 20.03 | 16.0 | **17.4** | 17.6 | 18.7 | 19.2 | **20.1** | 21.0 | ✅ **PASS (1.56x faster)** |
| **5 KB** | 20 | **NVIDIA L4** | 20.02 | 24.6 | 31.2 | 31.5 | 34.2 | 35.5 | **49.1** | 147.2 | ⚠️ *Borderline* |
| | 20 | **TPU v5e** | 20.03 | 19.8 | **21.3** | 21.4 | 22.0 | 22.3 | **23.1** | 24.1 | ✅ **PASS (2.13x faster)** |
| **7 KB** | 20 | **NVIDIA L4** | 20.02 | 28.3 | 35.9 | 36.2 | 39.4 | 41.5 | **70.2** | 181.2 | ❌ **FAIL (>50ms)** |
| | 20 | **TPU v5e** | 20.03 | 22.3 | **23.6** | 23.7 | 24.5 | 24.8 | **25.3** | 31.7 | ✅ **PASS (2.77x faster)** |

---

### 3.2 40 RPS Stress Test Profile

| Payload | Target RPS | Accelerator | Achieved RPS | Min (ms) | P50 (ms) | Avg (ms) | P90 (ms) | P95 (ms) | **P99 (ms)** | Max (ms) | **P99 SLA (<50ms)** |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB** | 40 | **NVIDIA L4** | 40.02 | 14.3 | 17.2 | 17.7 | 18.5 | 19.3 | **22.8** | 169.9 | ✅ PASS |
| | 40 | **TPU v5e** | 40.06 | 11.1 | **12.0** | 12.1 | 12.7 | 13.1 | **15.2** | 17.4 | ✅ **PASS (1.50x faster)** |
| **2 KB** | 40 | **NVIDIA L4** | 40.02 | 19.4 | 23.3 | 24.0 | 25.2 | 27.5 | **46.5** | 193.0 | ⚠️ *Borderline (Near 50ms)* |
| | 40 | **TPU v5e** | 40.05 | 16.2 | **16.9** | 17.2 | 18.1 | 19.1 | **20.7** | 22.0 | ✅ **PASS (2.25x faster)** |
| **5 KB** | 40 | **NVIDIA L4** | 40.02 | 29.8 | 46.3 | 51.2 | 66.8 | 72.9 | **145.3** | 283.3 | ❌ **FAIL (3x over SLA)** |
| | 40 | **TPU v5e** | 40.05 | 19.6 | **21.5** | 21.7 | 23.5 | 24.2 | **26.3** | 28.6 | ✅ **PASS (5.52x faster)** |
| **7 KB** | 40 | **NVIDIA L4** | **33.14** | 32.7 | 191.2 | 270.7 | 603.0 | 805.5 | **1,298.8** | 2,181.1 | ❌ **COLLAPSE (26x over SLA)** |
| | 40 | **TPU v5e** | **40.06** | 22.7 | **27.1** | 28.4 | 34.3 | 36.2 | **42.5** | 61.0 | ✅ **PASS (30.6x faster)** |

---

## 4. Full L4 Latency & Resource Utilization Sweep

Below is the complete resource telemetry on NVIDIA L4 across all load levels:

| Scenario | Payload | Target RPS | Achieved RPS | P50 (ms) | Avg (ms) | P90 (ms) | P99 (ms) | Max (ms) | CPU Max | Host RAM | GPU Util Max | GPU VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `prompt_c2_1024b_rps1` | 1K | 1 | 1.02 | 18.9 | 19.0 | 20.4 | 24.0 | 24.8 | 0.51 cores | 1.91 GB | 0.0% | 4.72 GB |
| `prompt_c2_1024b_rps5` | 1K | 5 | 5.02 | 18.6 | 18.8 | 20.1 | 26.8 | 56.2 | 0.72 cores | 1.91 GB | 0.0% | 4.72 GB |
| `prompt_c2_1024b_rps7` | 1K | 7 | 7.02 | 18.3 | 18.6 | 19.5 | 24.0 | 117.1 | 0.74 cores | 1.91 GB | 4.0% | 4.72 GB |
| `prompt_c2_1024b_rps10` | 1K | 10 | 10.02 | 18.2 | 18.3 | 19.8 | 23.5 | 61.5 | 0.20 cores | 1.91 GB | 7.0% | 4.72 GB |
| `prompt_c2_1024b_rps20` | 1K | 20 | 20.02 | 17.7 | 17.9 | 19.3 | 24.3 | 128.3 | 0.35 cores | 1.91 GB | 8.0% | 4.72 GB |
| `prompt_c2_1024b_rps30` | 1K | 30 | 30.03 | 17.4 | 17.8 | 18.7 | 23.0 | 148.5 | 0.62 cores | 1.91 GB | 13.0% | 4.72 GB |
| `prompt_c2_1024b_rps40` | 1K | 40 | 40.02 | 17.2 | 17.7 | 18.5 | 22.8 | 169.9 | 1.01 cores | 1.91 GB | 17.0% | 4.72 GB |
| `prompt_c2_2048b_rps1` | 2K | 1 | 1.02 | 25.0 | 25.1 | 27.5 | 31.6 | 35.4 | 0.72 cores | 1.91 GB | 17.0% | 4.72 GB |
| `prompt_c2_2048b_rps5` | 2K | 5 | 5.02 | 25.1 | 25.5 | 26.8 | 35.8 | 122.9 | 0.78 cores | 1.91 GB | 7.0% | 4.72 GB |
| `prompt_c2_2048b_rps7` | 2K | 7 | 7.02 | 24.8 | 24.9 | 26.6 | 31.6 | 124.5 | 0.19 cores | 1.91 GB | 7.0% | 4.72 GB |
| `prompt_c2_2048b_rps10` | 2K | 10 | 10.02 | 24.6 | 24.8 | 26.6 | 31.5 | 118.2 | 0.31 cores | 1.91 GB | 7.0% | 4.72 GB |
| `prompt_c2_2048b_rps20` | 2K | 20 | 20.02 | 23.6 | 23.9 | 25.5 | 31.4 | 139.6 | 0.67 cores | 1.91 GB | 15.0% | 4.72 GB |
| `prompt_c2_2048b_rps30` | 2K | 30 | 30.02 | 23.4 | 23.7 | 25.1 | 32.5 | 137.8 | 1.13 cores | 1.91 GB | 22.0% | 4.72 GB |
| `prompt_c2_2048b_rps40` | 2K | 40 | 40.02 | 23.3 | 24.0 | 25.2 | **46.5** | 193.0 | 1.44 cores | 1.91 GB | 30.0% | 4.72 GB |
| `prompt_c2_5120b_rps1` | 5K | 1 | 1.02 | 32.8 | 33.2 | 36.3 | 44.7 | 51.1 | 0.76 cores | 1.91 GB | 30.0% | 4.72 GB |
| `prompt_c2_5120b_rps5` | 5K | 5 | 5.02 | 32.8 | 32.9 | 35.7 | 39.8 | 147.7 | 0.20 cores | 1.91 GB | 6.0% | 4.72 GB |
| `prompt_c2_5120b_rps7` | 5K | 7 | 7.02 | 32.1 | 32.4 | 35.4 | 44.9 | 99.0 | 0.40 cores | 1.91 GB | 7.0% | 4.72 GB |
| `prompt_c2_5120b_rps10` | 5K | 10 | 10.02 | 31.4 | 31.6 | 34.8 | 40.4 | 129.3 | 0.66 cores | 1.91 GB | 7.0% | 4.72 GB |
| `prompt_c2_5120b_rps20` | 5K | 20 | 20.02 | 31.2 | 31.5 | 34.2 | **49.1** | 147.2 | 1.21 cores | 1.91 GB | 15.0% | 4.72 GB |
| `prompt_c2_5120b_rps30` | 5K | 30 | 30.02 | 30.8 | 31.8 | 35.2 | **61.5** | 194.8 | 1.58 cores | 1.91 GB | 22.0% | 4.72 GB |
| `prompt_c2_5120b_rps40` | 5K | 40 | 40.02 | 46.3 | 51.2 | 66.8 | **145.3** | 283.3 | 1.04 cores | 1.91 GB | 32.0% | 4.72 GB |
| `prompt_c2_7168b_rps1` | 7K | 1 | 1.02 | 37.7 | 37.7 | 42.6 | 45.5 | 46.0 | 1.06 cores | 1.91 GB | 31.0% | 4.72 GB |
| `prompt_c2_7168b_rps5` | 7K | 5 | 5.02 | 37.6 | 37.6 | 41.5 | 49.2 | 110.6 | 0.49 cores | 1.91 GB | 7.0% | 4.72 GB |
| `prompt_c2_7168b_rps7` | 7K | 7 | 7.02 | 37.1 | 37.3 | 41.0 | 51.3 | 100.3 | 0.76 cores | 1.91 GB | 7.0% | 4.72 GB |
| `prompt_c2_7168b_rps10` | 7K | 10 | 10.02 | 36.6 | 36.6 | 40.7 | **46.4** | 169.2 | 1.06 cores | 1.91 GB | 8.0% | 4.72 GB |
| `prompt_c2_7168b_rps20` | 7K | 20 | 20.02 | 35.9 | 36.2 | 39.4 | **70.2** | 181.2 | 1.42 cores | 1.91 GB | 15.0% | 4.72 GB |
| `prompt_c2_7168b_rps30` | 7K | 30 | 30.01 | 50.4 | 52.2 | 65.7 | **115.7** | 243.2 | 0.96 cores | 1.91 GB | 22.0% | 4.72 GB |
| `prompt_c2_7168b_rps40` | 7K | 40 | **33.14** | 191.2 | 270.7 | 603.0 | **1,298.8** | 2,181.1 | 1.18 cores | 1.91 GB | 51.0% | 4.72 GB |

---

## 5. Architectural & Root Cause Analysis

### Why Does L4 Saturate While TPU v5e Remains Linear?

1. **Memory Bandwidth & Matrix Multiply Bottlenecks**:
   - The L4 GPU features **24 GB GDDR6 with 300 GB/s bandwidth** and **121 TFLOPS BF16**.
   - The TPU v5e features **16 GB HBM2 with 820 GB/s bandwidth (2.73x higher)** and **197 TFLOPS BF16 (1.63x higher)**.
   - For transformer embeddings, computing self-attention over longer sequences scales quadratically ($O(N^2)$) in compute and activations. On L4, the lower memory bandwidth causes kernel launch queuing and attention execution stalls as sequence length reaches 5K–7K.
2. **Queue Buildup & Tail Latency Explosion**:
   - At 7K @ 40 RPS, the aggregate token rate exceeds L4's execution capacity. Inflight requests stack up in the PyTorch/vLLM queue, multiplying waiting time.
   - Consequently, while actual compute takes ~32 ms (Min latency), queued requests wait over 1.2 to 2.1 seconds before reaching the GPU execution pipeline.
   - TPU v5e's massive HBM bandwidth clears attention batches instantaneously, keeping queue depth at 0 and P99 latency at **42.5 ms**.

---

## 6. Capacity Sizing & TCO Comparison for 1,000 RPS Target Workload

To meet a **1,000 RPS** production workload without violating the **P99 < 50 ms SLA**, instances must be provisioned according to each chip's **safe maximum RPS ceiling**:

| Workload Sizing Scenario | L4 Safe RPS | TPU v5e Safe RPS | Chips Required (L4 vs. v5e) | 1-Yr CUD Cost / Month | 3-Yr CUD Cost / Month | Net Monthly Savings on TPU v5e (3-Yr CUD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short Payloads Only (1 KB)** | 40 RPS | 40 RPS | 25 vs. 25 | $8,212 vs. $15,330 | $5,840 vs. $9,855 | L4 is cheaper (unconstrained) |
| **Medium Payloads Only (5 KB)** | **20 RPS** | **40 RPS** | **50 vs. 25** | $16,425 vs. $15,330 | **$11,680 vs. $9,855** | **TPU v5e Saves $1,825/mo (15.6%)** |
| **Long Payloads Only (7 KB)** | **10 RPS** | **40 RPS** | **100 vs. 25** | $32,850 vs. $15,330 | **$23,360 vs. $9,855** | **TPU v5e Saves $13,505/mo (57.8%)** |
| **Uniform Mix (25% 1K, 2K, 5K, 7K)** | **~19.6 RPS** | **40 RPS** | **51 vs. 25** | $16,753 vs. $15,330 | **$11,914 vs. $9,855** | **TPU v5e Saves $2,059/mo (17.3%)** |
| **Heavy RAG Mix (50% 5K + 50% 7K)** | **~13.3 RPS** | **40 RPS** | **75 vs. 25** | $24,637 vs. $15,330 | **$17,520 vs. $9,855** | **TPU v5e Saves $7,665/mo (43.8%)** |

---

## 7. Cloud TPU v6e (Trillium) Projection

| Metric | Cloud TPU v5e | Cloud TPU v6e (Trillium) | Multiplier / Advantage |
| :--- | :---: | :---: | :---: |
| **BF16 Peak Compute** | 197 TFLOPS | **918 TFLOPS** | **4.7x higher** |
| **HBM Bandwidth** | 820 GB/s | **1,640 GB/s** | **2.0x higher** |
| **1-Year CUD ($/hr)** | $0.84 | $1.89 | 2.25x price |
| **3-Year CUD ($/hr)** | $0.54 | $1.22 | 2.26x price |
| **Estimated Safe Capacity (7K, P99 < 50ms)** | ~40–50 RPS | **~100–120 RPS** | **2.5x–3.0x higher throughput** |
| **Chips for 1,000 RPS Workload** | 25 chips | **10 chips** | **60% fewer chips** |
| **Monthly Cost (3-Year CUD @ 1,000 RPS)** | $9,855 | **$8,906** | **Additional 9.6% savings ($949/mo)** |
| **Total Node Reduction vs. L4 (100 chips)** | 75% reduction | **90% reduction (10 vs. 100)** | **10x smaller cluster footprint** |

---

## 8. Strategic Recommendations for Production Architecture

1. **Production Decision Boundary**:
   - If the application processes dynamic or document-scale text embeddings (**>2 KB up to 7 KB**), **deploy on Cloud TPU v5e**. It prevents latency collapse under burst traffic, ensures strict sub-50ms SLA compliance, and yields **17% to 58% net TCO savings**.
2. **Infrastructure Simplicity & Scaling**:
   - A single TPU v5e pool of **25 instances** replaces **51 to 100 L4 instances**, eliminating Kubernetes pod scheduling fragmentation, internal load balancer connection limits, and inter-node network chatter.
3. **Next-Generation Roadmap**:
   - For ultra-high density deployments exceeding 5,000+ aggregate RPS, migrate to **TPU v6e (Trillium)** to shrink the physical cluster footprint to 1/10th the size of an equivalent GPU fleet.
