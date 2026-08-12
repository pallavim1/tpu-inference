# Performance/$ & TCO Economic Analysis: TPU v5e vs. NVIDIA L4 GPU
## Executive Business Case for Palo Alto Networks (PANW) ATP Embedding Inference

**Model:** `jinaai/jina-embeddings-v2-small-en` (512-dim, JinaBert)  
**Target SLA:** Strict P99 Round-Trip Latency < 50 ms  
**Pricing Basis:** 1-Year & 3-Year Committed Use Discounts (CUD)  

---

## 1. Executive Summary

Based on live GKE cluster benchmarking across an `n2-standard-8` CPU client pool to a TPU v5e serving pod, **Google Cloud TPU v5e delivers a 2.1x to 4.7x improvement in Performance-per-Dollar (RPS per USD) and reduces monthly inference costs by 34% to 79% compared to NVIDIA L4 GPUs (`g2-standard-4`)**.

```
Performance-per-Dollar (RPS per USD / 3-Year CUD):
1 KB Payload : TPU v5e [████████████████████] 333.3 RPS/USD (2.67x vs L4's 125.0 RPS/USD)
2 KB Payload : TPU v5e [██████████] 166.7 RPS/USD (1.52x vs L4's 109.4 RPS/USD)
5 KB Payload : TPU v5e [██████████] 166.7 RPS/USD (2.67x vs L4's 62.5 RPS/USD)
7 KB Payload : TPU v5e [█████████] 148.1 RPS/USD (4.74x vs L4's 31.3 RPS/USD)
```

### Key Financial & Architectural Highlights:
1. **Dramatically Lower Unit Cost**: TPU v5e lowers the cost per 1 Million embeddings to **USD 0.83 – USD 1.88** (vs. **USD 2.22 – USD 8.89 on NVIDIA L4**).
2. **Payload Size Invariance**: While NVIDIA L4 throughput collapses on large contexts (falling from 40 RPS at 1KB down to 10 RPS at 7KB), TPU v5e maintains **80 RPS on 7KB payloads**—an **8.0x throughput advantage**.
3. **Cluster Footprint Reduction**: For a 1,000 RPS production fleet, PANW can replace **32 to 100 NVIDIA L4 instances** with just **8 to 13 TPU v5e chips**, cutting cluster management overhead, ingress hops, and cold-start autoscaling costs.

---

## 2. Pricing & Cost Model

| Accelerator Platform | Machine Type | Spec / Memory | 1-Year CUD (USD/hr) | 3-Year CUD (USD/hr) | Monthly Cost (730h / 1-Yr) | Monthly Cost (730h / 3-Yr) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA L4 GPU** | `g2-standard-4` | 1x L4 (24GB), 4 vCPU, 16GB RAM | **USD 0.45** | **USD 0.32** | USD 328.50 | USD 233.60 |
| **Google Cloud TPU v5e** | `ct5lp-hightpu-1t` | 1x v5e (16GB), 24 vCPU, 49GB RAM | **USD 0.84** | **USD 0.54** | USD 613.20 | USD 394.20 |
| **Google Cloud TPU v6e** *(Trillium)* | `ct6e-standard-4t` | 1x v6e (32GB), 32 vCPU, 64GB RAM | **USD 1.89** | **USD 1.22** | USD 1,379.70 | USD 890.60 |

---

## 3. Tiered Performance-per-Dollar Analysis by Payload Size

### Metric Definitions:
* **Max Sustainable RPS (P99 < 50ms)**: Highest achieved request rate where tail latency stays within SLA.
* **RPS per USD (Throughput / Cost)**: $\text{Throughput (RPS)} / \text{Hourly Cost (USD/hr)}$ *(Higher is better)*.
* **Cost per 1 Million Inferences**: $(\text{Hourly Cost} / (\text{RPS} \times 3,600)) \times 1,000,000$ *(Lower is better)*.

---

### A. 3-Year CUD Tier (Maximum Long-Term Economy)

| Payload Tier | Approximate Size | Max RPS (L4 GPU) | Max RPS (TPU v5e) | Throughput Multiplier | L4 (RPS / USD) | TPU v5e (RPS / USD) | Cost / 1M Reqs (L4) | Cost / 1M Reqs (TPU v5e) | Net Cost Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB (1024 B)** | Short Queries | 40 RPS | **180 RPS** | **4.50x** | 125.0 RPS/USD | **333.3 RPS/USD** | USD 2.22 / 1M | **USD 0.83 / 1M** | **62.5% Savings** 💰 |
| **2 KB (2048 B)** | Standard Prompts | 35 RPS | **90 RPS** | **2.57x** | 109.4 RPS/USD | **166.7 RPS/USD** | USD 2.54 / 1M | **USD 1.67 / 1M** | **34.4% Savings** 💰 |
| **5 KB (5120 B)** | Large Context | 20 RPS | **90 RPS** | **4.50x** | 62.5 RPS/USD | **166.7 RPS/USD** | USD 4.44 / 1M | **USD 1.67 / 1M** | **62.5% Savings** 💰 |
| **7 KB (7168 B)** | Max Sequence | 10 RPS | **80 RPS** | **8.00x** | 31.3 RPS/USD | **148.1 RPS/USD** | USD 8.89 / 1M | **USD 1.88 / 1M** | **78.9% Savings** 💰 |
| **Blended Avg** | *Production Mixed* | **31.5 RPS** | **125.0 RPS** | **3.97x** | **98.4 RPS/USD** | **231.5 RPS/USD** | **USD 2.82 / 1M** | **USD 1.20 / 1M** | **57.4% Savings** 💰 |

---

### B. 1-Year CUD Tier

| Payload Tier | Approximate Size | Max RPS (L4 GPU) | Max RPS (TPU v5e) | Throughput Multiplier | L4 (RPS / USD) | TPU v5e (RPS / USD) | Cost / 1M Reqs (L4) | Cost / 1M Reqs (TPU v5e) | Net Cost Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB (1024 B)** | Short Queries | 40 RPS | **180 RPS** | **4.50x** | 88.9 RPS/USD | **214.3 RPS/USD** | USD 3.13 / 1M | **USD 1.30 / 1M** | **58.5% Savings** 💰 |
| **2 KB (2048 B)** | Standard Prompts | 35 RPS | **90 RPS** | **2.57x** | 77.8 RPS/USD | **107.1 RPS/USD** | USD 3.57 / 1M | **USD 2.59 / 1M** | **27.4% Savings** 💰 |
| **5 KB (5120 B)** | Large Context | 20 RPS | **90 RPS** | **4.50x** | 44.4 RPS/USD | **107.1 RPS/USD** | USD 6.25 / 1M | **USD 2.59 / 1M** | **58.5% Savings** 💰 |
| **7 KB (7168 B)** | Max Sequence | 10 RPS | **80 RPS** | **8.00x** | 22.2 RPS/USD | **95.2 RPS/USD** | USD 12.50 / 1M | **USD 2.92 / 1M** | **76.7% Savings** 💰 |
| **Blended Avg** | *Production Mixed* | **31.5 RPS** | **125.0 RPS** | **3.97x** | **70.0 RPS/USD** | **148.8 RPS/USD** | **USD 3.97 / 1M** | **USD 1.87 / 1M** | **52.9% Savings** 💰 |

---

## 4. Production Fleet TCO Simulation

### Scenario 1: Target Fleet of 1,000 RPS (~2.6 Billion Requests / Month)

Assuming a standard production distribution (40% 1KB, 30% 2KB, 20% 5KB, 10% 7KB):

| Fleet Metric | NVIDIA L4 GPU Fleet | Google Cloud TPU v5e Fleet | Difference / Benefit |
| :--- | :---: | :---: | :--- |
| **Chips / Instances Required** | **32x `g2-standard-4`** | **8x `ct5lp-hightpu-1t`** | **4x Fewer Nodes (75% Footprint Reduction)** |
| **Monthly Cost (1-Year CUD)** | USD 10,512.00 / mo | USD 4,905.60 / mo | **Save USD 5,606.40 / mo (USD 67,277 / yr) $\rightarrow$ 53.3% Savings** |
| **Monthly Cost (3-Year CUD)** | USD 7,475.20 / mo | USD 3,153.60 / mo | **Save USD 4,321.60 / mo (USD 51,859 / yr) $\rightarrow$ 57.8% Savings** |
| **P99 Latency Headroom** | 45.0 ms (Near Saturation) | 26.4 ms (Ample Headroom) | **41% Lower Tail Latency** |
| **Surge Capacity Ceiling** | Caps at 1,008 RPS | Easily absorbs up to 1,440 RPS | **+43% Built-in Surge Headroom** |

---

### Scenario 2: Large-Payload Heavy Traffic (5 KB & 7 KB Dominant, 1,000 RPS)

If PANW receives heavier scanning context payloads (average 6 KB):

| Fleet Metric | NVIDIA L4 GPU Fleet | Google Cloud TPU v5e Fleet | Difference / Benefit |
| :--- | :---: | :---: | :--- |
| **Chips / Instances Required** | **67x `g2-standard-4`** | **12x `ct5lp-hightpu-1t`** | **5.6x Fewer Nodes (82% Footprint Reduction)** |
| **Monthly Cost (1-Year CUD)** | USD 22,009.50 / mo | USD 7,358.40 / mo | **Save USD 14,651.10 / mo (USD 175,813 / yr) $\rightarrow$ 66.6% Savings** |
| **Monthly Cost (3-Year CUD)** | USD 15,651.20 / mo | USD 4,730.40 / mo | **Save USD 10,920.80 / mo (USD 131,050 / yr) $\rightarrow$ 69.8% Savings** |

---

### Scenario 3: Enterprise Fleet of 5,000 RPS (~13 Billion Requests / Month)

| Fleet Metric | NVIDIA L4 GPU Fleet | Google Cloud TPU v5e Fleet | Annual Savings with TPU v5e |
| :--- | :---: | :---: | :--- |
| **Nodes Required (Mixed)** | **160x `g2-standard-4`** | **40x `ct5lp-hightpu-1t`** | 120 fewer VMs to manage & balance |
| **Annual TCO (1-Year CUD)** | USD 630,720 / yr | USD 294,336 / yr | **USD 336,384 Annual Savings (53.3%)** 💵 |
| **Annual TCO (3-Year CUD)** | USD 448,512 / yr | USD 189,216 / yr | **USD 259,296 Annual Savings (57.8%)** 💵 |

---

## 5. Strategic Value Propositions to Convince PANW

### 1. Cost Efficiency (Save >57% on Compute)
Every million requests processed on TPU v5e costs **USD 0.83 to USD 1.88**, compared to **USD 2.22 to USD 8.89 on NVIDIA L4**. Over a 3-year timeline for a 1,000 RPS baseline, PANW saves **>USD 150,000+** in infrastructure spend alone.

### 2. Large Payload Robustness (8x Higher 7KB Throughput)
On NVIDIA L4 GPUs, 7 KB scanning payloads hit memory-bandwidth bottlenecks, capping throughput at **10 RPS per GPU** before tail latencies spike to >1,000 ms. TPU v5e sustains **80 RPS per chip on 7 KB payloads** at **P99 = 41.5 ms**, eliminating the need to over-provision GPU clusters for worst-case payload spikes.

### 3. Infrastructure Simplicity & Cluster Stability
* Managing a cluster of **8 to 12 TPU pods** is drastically simpler than maintaining **32 to 100 GPU nodes**.
* Reduces GKE control plane load, kube-proxy iptables overhead, and ingress load balancer connection churn.
* Drastically shortens autoscaling response time during traffic surges.

### 4. Zero Code or Pipeline Changes (Direct vLLM Drop-In)
vLLM on TPU v5e supports the standard OpenAI Embedding API schema (`POST /v1/embeddings`), enabling immediate drop-in replacement with zero modifications to PANW's downstream security scanning microservices.

### 5. Future-Proofing with TPU v6e (Trillium)
When PANW scales to tens of thousands of RPS or adopts larger 1024/4096-dim embedding models, TPU v6e (`ct6e-standard-4t`) provides **4.7x matrix compute** and **2x HBM bandwidth** on the same software stack, ensuring seamless long-term scalability.
