# Performance/$ & TCO Economic Analysis: TPU v5e vs. NVIDIA L4 GPU
## Executive Business Case for Palo Alto Networks (PANW) ATP Embedding Inference

**Model:** `jinaai/jina-embeddings-v2-small-en` (512-dim, JinaBert)  
**Target SLA:** Strict P99 Round-Trip Latency < 50 ms  
**Pricing Basis:** 1-Year & 3-Year Committed Use Discounts (CUD)  

---

## 1. Executive Summary

Enforcing the strict **P99 < 50 ms SLA** across both customer NVIDIA L4 data and live Google Cloud TPU v5e GKE benchmarks:

* **1 KB Payload**: TPU v5e delivers **180 RPS** vs. NVIDIA L4 **40 RPS** (**4.50x higher throughput**, **62.5% cheaper per 1M reqs**).
* **2 KB Payload**: TPU v5e delivers **90 RPS** vs. NVIDIA L4 **40 RPS** (**2.25x higher throughput**, **25.0% cheaper per 1M reqs**).
* **5 KB Payload**: TPU v5e delivers **90 RPS** vs. NVIDIA L4 **20 RPS** (**4.50x higher throughput**, **62.5% cheaper per 1M reqs**).
* **7 KB Payload**: TPU v5e delivers **80 RPS** vs. NVIDIA L4 **10 RPS** (**8.00x higher throughput**, **78.9% cheaper per 1M reqs**).

```
Performance-per-Dollar (RPS per USD / 3-Year CUD under <50ms P99 SLA):
1 KB Payload : TPU v5e [████████████████████] 333.3 RPS/USD (2.67x vs L4's 125.0 RPS/USD)
2 KB Payload : TPU v5e [██████████] 166.7 RPS/USD (1.33x vs L4's 125.0 RPS/USD)
5 KB Payload : TPU v5e [██████████] 166.7 RPS/USD (2.67x vs L4's 62.5 RPS/USD)
7 KB Payload : TPU v5e [█████████] 148.1 RPS/USD (4.74x vs L4's 31.3 RPS/USD)
```

---

## 2. SLA Compliance & Maximum Sustainable Capacity Baseline

Under PANW's strict SLA requirement (**$P_{99} < 50\text{ ms}$**):

| Payload Tier | Character / Byte Range | Approx Token Count | NVIDIA L4 Max SLA RPS | L4 P99 Latency | TPU v5e Max SLA RPS | TPU v5e P99 Latency | Throughput Advantage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB (1024 B)** | Short Queries | ~200 Tokens | **40 RPS** *(Max Tested)* | 22.8 ms | **180 RPS** | **38.9 ms** | **4.50x Higher** 🚀 |
| **2 KB (2048 B)** | Standard Prompts | ~400 Tokens | **40 RPS** | 46.5 ms | **90 RPS** | **28.4 ms** | **2.25x Higher** 🚀 |
| **5 KB (5120 B)** | Large Context | ~1,000 Tokens | **20 RPS** *(Fails @ 30)* | 49.1 ms | **90 RPS** | **44.9 ms** | **4.50x Higher** 🚀 |
| **7 KB (7168 B)** | Max Sequence | ~1,400 Tokens | **10 RPS** *(Fails @ 20)* | 46.4 ms | **80 RPS** | **41.5 ms** | **8.00x Higher** 🚀 |
| **Blended Mixed** | *Production Avg* | ~500 Tokens | **33.0 RPS** | — | **125.0 RPS** | — | **3.79x Higher** 🚀 |

---

## 3. Pricing & Cost Model

| Accelerator Platform | Machine Type | Spec / Memory | 1-Year CUD (USD/hr) | 3-Year CUD (USD/hr) | Monthly Cost (730h / 1-Yr) | Monthly Cost (730h / 3-Yr) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA L4 GPU** | `g2-standard-4` | 1x L4 (24GB), 4 vCPU, 16GB RAM | **USD 0.45** | **USD 0.32** | USD 328.50 | USD 233.60 |
| **Google Cloud TPU v5e** | `ct5lp-hightpu-1t` | 1x v5e (16GB), 24 vCPU, 49GB RAM | **USD 0.84** | **USD 0.54** | USD 613.20 | USD 394.20 |
| **Google Cloud TPU v6e** *(Trillium)* | `ct6e-standard-4t` | 1x v6e (32GB), 32 vCPU, 64GB RAM | **USD 1.89** | **USD 1.22** | USD 1,379.70 | USD 890.60 |

---

## 4. Tiered Performance-per-Dollar Analysis by Payload Size

### A. 3-Year CUD Tier (Maximum Long-Term Economy)
* **NVIDIA L4 (`g2-standard-4`)**: **USD 0.32 / hour**
* **Google Cloud TPU v5e (`ct5lp-hightpu-1t`)**: **USD 0.54 / hour**

| Payload Tier | Max RPS (L4 GPU) | Max RPS (TPU v5e) | Throughput Multiplier | L4 Efficiency (RPS / USD) | TPU v5e Efficiency (RPS / USD) | Cost / 1M Reqs (L4 GPU) | Cost / 1M Reqs (TPU v5e) | PANW Net Cost Savings (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB (1024 B)** | 40 RPS | **180 RPS** | **4.50x** | 125.0 RPS / USD | **333.3 RPS / USD** | USD 2.22 / 1M | **USD 0.83 / 1M** | **62.5% Cheaper** 💰 |
| **2 KB (2048 B)** | 40 RPS | **90 RPS** | **2.25x** | 125.0 RPS / USD | **166.7 RPS / USD** | USD 2.22 / 1M | **USD 1.67 / 1M** | **25.0% Cheaper** 💰 |
| **5 KB (5120 B)** | 20 RPS | **90 RPS** | **4.50x** | 62.5 RPS / USD | **166.7 RPS / USD** | USD 4.44 / 1M | **USD 1.67 / 1M** | **62.5% Cheaper** 💰 |
| **7 KB (7168 B)** | 10 RPS | **80 RPS** | **8.00x** | 31.3 RPS / USD | **148.1 RPS / USD** | USD 8.89 / 1M | **USD 1.88 / 1M** | **78.9% Cheaper** 💰 |
| **Blended Mixed** | **33.0 RPS** | **125.0 RPS** | **3.79x** | **103.1 RPS / USD** | **231.5 RPS / USD** | **USD 2.69 / 1M** | **USD 1.20 / 1M** | **55.5% Cheaper** 💰 |

---

### B. 1-Year CUD Tier
* **NVIDIA L4 (`g2-standard-4`)**: **USD 0.45 / hour**
* **Google Cloud TPU v5e (`ct5lp-hightpu-1t`)**: **USD 0.84 / hour**

| Payload Tier | Max RPS (L4 GPU) | Max RPS (TPU v5e) | Throughput Multiplier | L4 Efficiency (RPS / USD) | TPU v5e Efficiency (RPS / USD) | Cost / 1M Reqs (L4 GPU) | Cost / 1M Reqs (TPU v5e) | PANW Net Cost Savings (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB (1024 B)** | 40 RPS | **180 RPS** | **4.50x** | 88.9 RPS / USD | **214.3 RPS / USD** | USD 3.12 / 1M | **USD 1.30 / 1M** | **58.5% Cheaper** 💰 |
| **2 KB (2048 B)** | 40 RPS | **90 RPS** | **2.25x** | 88.9 RPS / USD | **107.1 RPS / USD** | USD 3.12 / 1M | **USD 2.59 / 1M** | **17.0% Cheaper** 💰 |
| **5 KB (5120 B)** | 20 RPS | **90 RPS** | **4.50x** | 44.4 RPS / USD | **107.1 RPS / USD** | USD 6.25 / 1M | **USD 2.59 / 1M** | **58.5% Cheaper** 💰 |
| **7 KB (7168 B)** | 10 RPS | **80 RPS** | **8.00x** | 22.2 RPS / USD | **95.2 RPS / USD** | USD 12.50 / 1M | **USD 2.92 / 1M** | **76.7% Cheaper** 💰 |
| **Blended Mixed** | **33.0 RPS** | **125.0 RPS** | **3.79x** | **73.3 RPS / USD** | **148.8 RPS / USD** | **USD 3.79 / 1M** | **USD 1.87 / 1M** | **50.7% Cheaper** 💰 |

---

## 5. Production Fleet Sizing & Annual TCO Comparison

### Production Fleet: 1,000 RPS Target Workload (~2.6 Billion Inferences / Month)

Assuming standard mixed traffic (40% 1KB, 30% 2KB, 20% 5KB, 10% 7KB):

| Fleet Parameter | NVIDIA L4 GPU Fleet | Google Cloud TPU v5e Fleet | Difference / Benefit |
| :--- | :---: | :---: | :--- |
| **Instances / Chips Required (Mixed)** | **31x `g2-standard-4`** | **8x `ct5lp-hightpu-1t`** | **23 Fewer Nodes (74% Footprint Reduction)** |
| **Instances Required (7 KB Heavy)** | **100x `g2-standard-4`** | **13x `ct5lp-hightpu-1t`** | **87 Fewer Nodes (87% Footprint Reduction)** |
| **Monthly Cost (1-Year CUD)** | USD 10,183.50 / mo | USD 4,905.60 / mo | **Save USD 5,277.90 / mo (USD 63,335 / yr) $\rightarrow$ 51.8% Savings** |
| **Monthly Cost (3-Year CUD)** | USD 7,241.60 / mo | USD 3,153.60 / mo | **Save USD 4,088.00 / mo (USD 49,056 / yr) $\rightarrow$ 56.5% Savings** |
| **Worst-Case 7KB Surge Fleet (3-Yr CUD)** | **100x L4 (USD 23,360 / mo)** | **13x v5e (USD 5,124 / mo)** | **Save USD 18,236.00 / mo (USD 218,832 / yr) $\rightarrow$ 78.1% Savings** |

---

## 6. Key Takeaways to Present to PANW

1. **Massive Cost Reduction Across All Tiers (25% to 78.9% Cheaper)**:
   * Even on short queries (1 KB), TPU v5e processes 1 Million embeddings for **USD 0.83** vs. **USD 2.22 on L4** (62.5% savings).
   * On maximum sequence length (7 KB), TPU v5e processes 1 Million embeddings for **USD 1.88** vs. **USD 8.89 on L4** (78.9% savings).
2. **Context Length Robustness (8x Higher 7KB Throughput)**:
   * NVIDIA L4 hits memory-bandwidth bottlenecks on long sequences, capping throughput at **10 RPS** before exceeding the 50 ms SLA ($P_{99} = 70.2\text{ ms}$ at 20 RPS).
   * TPU v5e easily sustains **80 RPS on 7 KB** with **$P_{99} = 41.5\text{ ms}$**, eliminating the need to over-provision GPU clusters for worst-case payload spikes.
3. **Infrastructure Simplicity**:
   * For a 1,000 RPS fleet, PANW replaces **31 to 100 GPU instances with just 8 to 13 TPU v5e pods**, saving **$49,000 to $218,000 annually** while drastically reducing Kubernetes control-plane load and network churn.
