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

```
Performance-per-Dollar (RPS per USD / 3-Year CUD under <50ms P99 SLA):
1 KB Payload : TPU v5e (FP16) [████████████████████] 296.3 RPS/USD (2.37x vs L4's 125.0 RPS/USD)
2 KB Payload : TPU v5e (FP16) [██████████] 148.1 RPS/USD (1.18x vs L4's 125.0 RPS/USD)
5 KB Payload : TPU v5e (FP16) [██████████] 148.1 RPS/USD (2.37x vs L4's 62.5 RPS/USD)
7 KB Payload : TPU v5e (FP16) [██████████] 148.1 RPS/USD (4.73x vs L4's 31.3 RPS/USD)
```

---

## 2. Unit Pricing Reference

| Platform | Machine Type / Accelerator | On-Demand ($/hr) | 1-Year CUD ($/hr) | 3-Year CUD ($/hr) |
| :--- | :--- | :---: | :---: | :---: |
| **NVIDIA L4 GPU** | `g2-standard-4` (1x L4 24GB) | $0.70 | **$0.45** | **$0.32** |
| **Google Cloud TPU v5e** | `ct5lp-hightpu-1t` (1x TPU v5e 16GB) | $1.20 | **$0.84** | **$0.54** |
| **Google Cloud TPU v6e** | `ct6e-standard-4t` (1x TPU v6e 32GB) | $2.70 | **$1.89** | **$1.22** |

---

## 3. Tiered Performance-per-Dollar Analysis by Payload Size (FP16)

### A. 1 KB Payload (`1024B`) — Short Ingestion Queries

| Metric | NVIDIA L4 GPU | Cloud TPU v5e (FP16) | Advantage (TPU vs. L4) |
| :--- | :---: | :---: | :---: |
| **Max Certified RPS ($P_{99} < 50\text{ ms}$)** | **40 RPS** | **160 RPS** | **4.0x Higher Throughput** |
| **$P_{99}$ Latency at Max Capacity** | 22.80 ms | 28.90 ms | Meets Strict SLA |
| **RPS per Dollar (1-Year CUD)** | 88.9 RPS / $ | **190.5 RPS / $** | **+114% Cost Efficiency** |
| **RPS per Dollar (3-Year CUD)** | 125.0 RPS / $ | **296.3 RPS / $** | **+137% Cost Efficiency** |
| **Cost per 1M Embeddings (1-Yr CUD)** | $3.125 | **$1.458** | **53.3% Cost Savings** |
| **Cost per 1M Embeddings (3-Yr CUD)** | $2.222 | **$0.938** | **57.8% Cost Savings** |

---

### B. 2 KB Payload (`2048B`) — Standard Security Prompts

| Metric | NVIDIA L4 GPU | Cloud TPU v5e (FP16) | Advantage (TPU vs. L4) |
| :--- | :---: | :---: | :---: |
| **Max Certified RPS ($P_{99} < 50\text{ ms}$)** | **40 RPS** | **80 RPS** | **2.0x Higher Throughput** |
| **$P_{99}$ Latency at Max Capacity** | 46.50 ms | 24.80 ms | **46.7% Lower Tail Latency** |
| **RPS per Dollar (1-Year CUD)** | 88.9 RPS / $ | **95.2 RPS / $** | **+7.1% Cost Efficiency** |
| **RPS per Dollar (3-Year CUD)** | 125.0 RPS / $ | **148.1 RPS / $** | **+18.5% Cost Efficiency** |
| **Cost per 1M Embeddings (1-Yr CUD)** | $3.125 | **$2.917** | **6.7% Cost Savings** |
| **Cost per 1M Embeddings (3-Yr CUD)** | $2.222 | **$1.875** | **15.6% Cost Savings** |

---

### C. 5 KB Payload (`5120B`) — Heavy Telemetry Context

| Metric | NVIDIA L4 GPU | Cloud TPU v5e (FP16) | Advantage (TPU vs. L4) |
| :--- | :---: | :---: | :---: |
| **Max Certified RPS ($P_{99} < 50\text{ ms}$)** | **20 RPS** | **80 RPS** | **4.0x Higher Throughput** |
| **$P_{99}$ Latency at Max Capacity** | 49.10 ms | 39.50 ms | **19.6% Lower Tail Latency** |
| **RPS per Dollar (1-Year CUD)** | 44.4 RPS / $ | **95.2 RPS / $** | **+114% Cost Efficiency** |
| **RPS per Dollar (3-Year CUD)** | 62.5 RPS / $ | **148.1 RPS / $** | **+137% Cost Efficiency** |
| **Cost per 1M Embeddings (1-Yr CUD)** | $6.250 | **$2.917** | **53.3% Cost Savings** |
| **Cost per 1M Embeddings (3-Yr CUD)** | $4.444 | **$1.875** | **57.8% Cost Savings** |

---

### D. 7 KB Payload (`7168B`) — Maximum Sequence Length

| Metric | NVIDIA L4 GPU | Cloud TPU v5e (FP16) | Advantage (TPU vs. L4) |
| :--- | :---: | :---: | :---: |
| **Max Certified RPS ($P_{99} < 50\text{ ms}$)** | **10 RPS** | **80 RPS** | **8.0x Higher Throughput** |
| **$P_{99}$ Latency at Max Capacity** | 46.40 ms | 48.00 ms | Meets Strict SLA |
| **RPS per Dollar (1-Year CUD)** | 22.2 RPS / $ | **95.2 RPS / $** | **+328% Cost Efficiency** |
| **RPS per Dollar (3-Year CUD)** | 31.3 RPS / $ | **148.1 RPS / $** | **+374% Cost Efficiency** |
| **Cost per 1M Embeddings (1-Yr CUD)** | $12.500 | **$2.917** | **76.7% Cost Savings** |
| **Cost per 1M Embeddings (3-Yr CUD)** | $8.889 | **$1.875** | **78.9% Cost Savings** |

---

## 4. Production Cluster TCO (1,000 RPS Mixed Workload)

To serve a production cluster target of **1,000 RPS** of mixed 5KB–7KB traffic under the strict $P_{99} < 50\text{ ms}$ SLA:

| Metric | NVIDIA L4 GPU Fleet | Cloud TPU v5e Fleet (FP16) | Savings / Impact |
| :--- | :---: | :---: | :---: |
| **Pods Required** | **100 pods** (10 RPS/pod) | **13 pods** (80 RPS/pod) | **87% Smaller Footprint** |
| **Hourly Cost (1-Year CUD)** | $45.00 / hr | **$10.92 / hr** | **$34.08 / hr Savings** |
| **Monthly Cost (1-Year CUD)** | $32,850 / mo | **$7,972 / mo** | **$24,878 / mo Savings** |
| **Annual Cost (1-Year CUD)** | $394,200 / yr | **$95,659 / yr** | **$298,541 / yr Savings (75.7%)** |
| **Annual Cost (3-Year CUD)** | $280,320 / yr | **$61,495 / yr** | **$218,825 / yr Savings (78.1%)** |
