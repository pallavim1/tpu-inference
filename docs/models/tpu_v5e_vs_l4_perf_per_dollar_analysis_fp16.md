# Performance/$ & TCO Economic Analysis: TPU v5e (FP16 / BF16) vs. NVIDIA L4 GPU
## Executive Business Case for Palo Alto Networks (PANW) ATP Embedding Inference

**Model:** `jinaai/jina-embeddings-v2-small-en` (512-dim, JinaBert)  
**Precision:** `bfloat16` (Native 16-bit Floating Point on Cloud TPU v5e)  
**Target SLA:** Strict Tail Latency $P_{99} < 50\text{ ms}$ across GKE Nodepools  
**Client Environment:** Dedicated CPU Node Pool (`n2-standard-8` in `cpu-benchmark-pool`)  
**Serving Environment:** Google Cloud TPU v5e (`ct5lp-hightpu-1t` in `pm-panw-jina-tpu-pool`)  
**Network Path:** Real GKE Cluster Network (`europe-west4-b`) via Kubernetes ClusterIP Service  
**Stage Duration:** 60 seconds per load point  

---

## 1. Executive Summary

Enforcing the customer's strict **$P_{99} < 50\text{ ms}$ SLA** across both customer NVIDIA L4 baseline data and live Google Cloud TPU v5e GKE network benchmarks in **FP16 / BF16 precision**:

* **1 KB Payload**: TPU v5e delivers **160 RPS** vs. NVIDIA L4 **40 RPS** (**4.00x higher throughput**, **53.3% cheaper per 1M reqs**).
* **2 KB Payload**: TPU v5e delivers **80 RPS** vs. NVIDIA L4 **40 RPS** (**2.00x higher throughput**, **6.7% cheaper per 1M reqs**, with **46.7% lower tail latency**).
* **5 KB Payload**: TPU v5e delivers **80 RPS** vs. NVIDIA L4 **20 RPS** (**4.00x higher throughput**, **53.3% cheaper per 1M reqs**).
* **7 KB Payload**: TPU v5e delivers **80 RPS** vs. NVIDIA L4 **10 RPS** (**8.00x higher throughput**, **76.7% cheaper per 1M reqs**).
* **Blended Mixed Workload**: TPU v5e delivers **112.0 RPS** vs. NVIDIA L4 **33.0 RPS** (**3.39x higher throughput**, **50.3% cheaper per 1M reqs**).

```
Performance-per-Dollar (RPS per USD / 3-Year CUD under <50ms P99 SLA):
1 KB Payload : TPU v5e (FP16) [████████████████████] 296.3 RPS/USD (2.37x vs L4's 125.0 RPS/USD)
2 KB Payload : TPU v5e (FP16) [██████████] 148.1 RPS/USD (1.18x vs L4's 125.0 RPS/USD)
5 KB Payload : TPU v5e (FP16) [██████████] 148.1 RPS/USD (2.37x vs L4's 62.5 RPS/USD)
7 KB Payload : TPU v5e (FP16) [██████████] 148.1 RPS/USD (4.73x vs L4's 31.3 RPS/USD)
Blended Avg  : TPU v5e (FP16) [██████████████] 207.4 RPS/USD (2.01x vs L4's 103.1 RPS/USD)
```

---

## 2. Unit Pricing Reference

| Platform | Machine Type / Accelerator | On-Demand ($/hr) | 1-Year CUD ($/hr) | 3-Year CUD ($/hr) | Monthly Cost (730h / 1-Yr) | Monthly Cost (730h / 3-Yr) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA L4 GPU** | `g2-standard-4` (1x L4 24GB) | $0.70 | **$0.45** | **$0.32** | USD 328.50 | USD 233.60 |
| **Google Cloud TPU v5e** | `ct5lp-hightpu-1t` (1x TPU v5e 16GB) | $1.20 | **$0.84** | **$0.54** | USD 613.20 | USD 394.20 |
| **Google Cloud TPU v6e** | `ct6e-standard-4t` (1x TPU v6e 32GB) | $2.70 | **$1.89** | **$1.22** | USD 1,379.70 | USD 890.60 |

---

## 3. Blended Traffic Methodology & Mathematical Derivation (FP16)

### A. Traffic Mix Weights ($w_i$)
* **1 KB (`1024 B`) — 40% (`0.40`)**: High-frequency short queries, hashes, URLs, and discrete lookups.
* **2 KB (`2048 B`) — 30% (`0.30`)**: Standard event telemetry, headers, and metadata prompts.
* **5 KB (`5120 B`) — 20% (`0.20`)**: Multi-line logs, aggregated telemetry, and code blocks.
* **7 KB (`7168 B`) — 10% (`0.10`)**: Full-context payloads, file extractions, and large forensic buffers.

$$\sum_{i=1}^{4} w_i = 0.40 + 0.30 + 0.20 + 0.10 = 1.0\quad (100\%)$$

### B. Weighted Blended Sustainable Throughput ($RPS_{\text{blended}}$)
$$\text{Blended RPS} = \sum_{i} \left( w_i \times \text{Max RPS}_i \right)$$

* **For NVIDIA L4 GPU**:
  $$\text{RPS}_{\text{L4, blended}} = (0.40 \times 40) + (0.30 \times 40) + (0.20 \times 20) + (0.10 \times 10) = 16.0 + 12.0 + 4.0 + 1.0 = \mathbf{33.0\text{ RPS}}$$

* **For Google Cloud TPU v5e (FP16 / BF16)**:
  $$\text{RPS}_{\text{TPU FP16, blended}} = (0.40 \times 160) + (0.30 \times 80) + (0.20 \times 80) + (0.10 \times 80) = 64.0 + 24.0 + 16.0 + 8.0 = \mathbf{112.0\text{ RPS}}$$

$$\text{Throughput Advantage} = \frac{112.0\text{ RPS}}{33.0\text{ RPS}} = \mathbf{3.39\times\text{ Higher Throughput per Pod}}$$

### C. Blended Performance-per-Dollar Formula (RPS / USD)
$$\text{RPS / USD} = \frac{\text{Blended RPS}}{\text{Hourly Cost (USD)}}$$

* **3-Year CUD ($0.32/hr L4 vs. $0.54/hr TPU v5e)**:
  * L4 Efficiency: $\frac{33.0}{0.32} = \mathbf{103.1\text{ RPS / USD}}$
  * TPU v5e Efficiency: $\frac{112.0}{0.54} = \mathbf{207.4\text{ RPS / USD}}$ (**+101.2% Cost Efficiency**)

* **1-Year CUD ($0.45/hr L4 vs. $0.84/hr TPU v5e)**:
  * L4 Efficiency: $\frac{33.0}{0.45} = \mathbf{73.3\text{ RPS / USD}}$
  * TPU v5e Efficiency: $\frac{112.0}{0.84} = \mathbf{133.3\text{ RPS / USD}}$ (**+81.9% Cost Efficiency**)

### D. Blended Cost per 1 Million Embeddings Formula
$$\text{Cost per 1M Reqs} = \frac{\text{Hourly Cost}}{\text{Blended RPS} \times 3,600\text{ s/hr}} \times 1,000,000$$

* **3-Year CUD**:
  * L4 Cost / 1M: $\frac{\$0.32}{33.0 \times 3600} \times 10^6 = \mathbf{\$2.69\text{ per 1M}}$
  * TPU v5e Cost / 1M: $\frac{\$0.54}{112.0 \times 3600} \times 10^6 = \mathbf{\$1.34\text{ per 1M}}$ (**50.3% Net Cost Savings**)

* **1-Year CUD**:
  * L4 Cost / 1M: $\frac{\$0.45}{33.0 \times 3600} \times 10^6 = \mathbf{\$3.79\text{ per 1M}}$
  * TPU v5e Cost / 1M: $\frac{\$0.84}{112.0 \times 3600} \times 10^6 = \mathbf{\$2.08\text{ per 1M}}$ (**45.0% Net Cost Savings**)

---

## 4. Tiered Performance-per-Dollar Analysis Tables (FP16)

### A. 3-Year CUD Tier (Maximum Long-Term Economy)
* **NVIDIA L4 (`g2-standard-4`)**: **USD 0.32 / hour**
* **Google Cloud TPU v5e (`ct5lp-hightpu-1t`)**: **USD 0.54 / hour**

| Payload Tier | Max RPS (L4 GPU) | Max RPS (TPU v5e) | Throughput Multiplier | L4 Efficiency (RPS / USD) | TPU v5e Efficiency (RPS / USD) | Cost / 1M Reqs (L4 GPU) | Cost / 1M Reqs (TPU v5e) | PANW Net Cost Savings (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB (1024 B)** | 40 RPS | **160 RPS** | **4.00x** | 125.0 RPS / USD | **296.3 RPS / USD** | USD 2.22 / 1M | **USD 0.94 / 1M** | **57.8% Cheaper** 💰 |
| **2 KB (2048 B)** | 40 RPS | **80 RPS** | **2.00x** | 125.0 RPS / USD | **148.1 RPS / USD** | USD 2.22 / 1M | **USD 1.88 / 1M** | **15.6% Cheaper** 💰 |
| **5 KB (5120 B)** | 20 RPS | **80 RPS** | **4.00x** | 62.5 RPS / USD | **148.1 RPS / USD** | USD 4.44 / 1M | **USD 1.88 / 1M** | **57.8% Cheaper** 💰 |
| **7 KB (7168 B)** | 10 RPS | **80 RPS** | **8.00x** | 31.3 RPS / USD | **148.1 RPS / USD** | USD 8.89 / 1M | **USD 1.88 / 1M** | **78.9% Cheaper** 💰 |
| **Blended Mixed** | **33.0 RPS** | **112.0 RPS** | **3.39x** | **103.1 RPS / USD** | **207.4 RPS / USD** | **USD 2.69 / 1M** | **USD 1.34 / 1M** | **50.3% Cheaper** 💰 |

---

### B. 1-Year CUD Tier
* **NVIDIA L4 (`g2-standard-4`)**: **USD 0.45 / hour**
* **Google Cloud TPU v5e (`ct5lp-hightpu-1t`)**: **USD 0.84 / hour**

| Payload Tier | Max RPS (L4 GPU) | Max RPS (TPU v5e) | Throughput Multiplier | L4 Efficiency (RPS / USD) | TPU v5e Efficiency (RPS / USD) | Cost / 1M Reqs (L4 GPU) | Cost / 1M Reqs (TPU v5e) | PANW Net Cost Savings (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB (1024 B)** | 40 RPS | **160 RPS** | **4.00x** | 88.9 RPS / USD | **190.5 RPS / USD** | USD 3.12 / 1M | **USD 1.46 / 1M** | **53.3% Cheaper** 💰 |
| **2 KB (2048 B)** | 40 RPS | **80 RPS** | **2.00x** | 88.9 RPS / USD | **95.2 RPS / USD** | USD 3.12 / 1M | **USD 2.92 / 1M** | **6.7% Cheaper** 💰 |
| **5 KB (5120 B)** | 20 RPS | **80 RPS** | **4.00x** | 44.4 RPS / USD | **95.2 RPS / USD** | USD 6.25 / 1M | **USD 2.92 / 1M** | **53.3% Cheaper** 💰 |
| **7 KB (7168 B)** | 10 RPS | **80 RPS** | **8.00x** | 22.2 RPS / USD | **95.2 RPS / USD** | USD 12.50 / 1M | **USD 2.92 / 1M** | **76.7% Cheaper** 💰 |
| **Blended Mixed** | **33.0 RPS** | **112.0 RPS** | **3.39x** | **73.3 RPS / USD** | **133.3 RPS / USD** | **USD 3.79 / 1M** | **USD 2.08 / 1M** | **45.0% Cheaper** 💰 |

---

## 5. Production Fleet Sizing & Annual TCO Comparison (FP16)

### Production Fleet: 1,000 RPS Target Workload (~2.6 Billion Inferences / Month)

Using the derived blended capacity per pod ($\text{Pods Required} = \lceil 1,000 / \text{Blended RPS} \rceil$):

| Fleet Parameter | NVIDIA L4 GPU Fleet | Google Cloud TPU v5e Fleet (FP16) | Difference / Benefit |
| :--- | :---: | :---: | :--- |
| **Instances / Chips Required (Mixed)** | **31x `g2-standard-4`** | **9x `ct5lp-hightpu-1t`** | **22 Fewer Nodes (71% Footprint Reduction)** |
| **Instances Required (7 KB Heavy)** | **100x `g2-standard-4`** | **13x `ct5lp-hightpu-1t`** | **87 Fewer Nodes (87% Footprint Reduction)** |
| **Monthly Cost (1-Year CUD)** | USD 10,183.50 / mo | USD 5,518.80 / mo | **Save USD 4,664.70 / mo (USD 55,976 / yr) $\rightarrow$ 45.8% Savings** |
| **Monthly Cost (3-Year CUD)** | USD 7,241.60 / mo | USD 3,547.80 / mo | **Save USD 3,693.80 / mo (USD 44,326 / yr) $\rightarrow$ 51.0% Savings** |
| **Worst-Case 7KB Surge Fleet (3-Yr CUD)** | **100x L4 (USD 23,360 / mo)** | **13x v5e (USD 5,124 / mo)** | **Save USD 18,236.00 / mo (USD 218,832 / yr) $\rightarrow$ 78.1% Savings** |
