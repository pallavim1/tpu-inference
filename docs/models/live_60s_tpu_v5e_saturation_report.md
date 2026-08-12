# TPU v5e Live Saturation Benchmark Report (60s Sustained Stages)

**Model:** `jinaai/jina-embeddings-v2-small-en` (512-dim Embeddings, JinaBert)  
**Cluster:** `pm-panw-jina-cluster` (`europe-west4-b`)  
**Hardware:** 1x Google Cloud TPU v5e Chip (`ct5lp-hightpu-1t` / `v5litepod-1` on GKE)  
**Workload Target:** Palo Alto Networks (PANW) ATP Inference (`POST /prompt_c2`)  
**Target SLA:** P99 Round-Trip Latency < 50 ms  
**Execution Environment:** In-Pod Localhost Loopback (`127.0.0.1:8000`), 60 seconds per test stage  

---

## 1. Master Saturation Matrix (60s Sustained Duration)

| Payload Size | Character / Byte Range | Approx Token Count | Max Sustainable Throughput (<50ms P99 SLA) | Achieved P99 at Max RPS | Exact Saturation Boundary (P99 > 50ms) | Headroom vs. 20 RPS Baseline | Headroom vs. 40 RPS Peak |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB (1024 B)** | Short Queries | ~200 Tokens | **160 RPS** | **30.7 ms** | **180 RPS** (P99 = 103.8 ms) | **8.0x Margin** | **4.0x Margin** |
| **2 KB (2048 B)** | Standard Prompts | ~400 Tokens | **90 RPS** | **23.3–28.4 ms** | **95 RPS** (P99 = 988.4 ms) | **4.5x Margin** | **2.25x Margin** |
| **5 KB (5120 B)** | Large Context | ~1,000 Tokens | **70 RPS** | **32.1 ms** | **80 RPS** (P99 = 53.0 ms) | **3.5x Margin** | **1.75x Margin** |
| **7 KB (7168 B)** | Max Sequence Length | ~1,400 Tokens | **60 RPS** | **40.9 ms** | **70 RPS** (P99 = 56.3 ms) | **3.0x Margin** | **1.5x Margin** |

---

## 2. Complete P99 Latency Progression (50 to 190 RPS)

| Target RPS | 1 KB (1024 B) | 2 KB (2048 B) | 5 KB (5120 B) | 7 KB (7168 B) | 50ms SLA Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **50 RPS** | **14.9 ms** | **21.0 ms** | **32.8 ms** | **35.6 ms** | ✅ All Passed (All < 36 ms) |
| **60 RPS** | **15.0 ms** | **22.5 ms** | **34.7 ms** | **40.9 ms** | ✅ **7K Ceiling Passed (60 RPS)** |
| **70 RPS** | **16.1 ms** | **26.5 ms** | **32.1 ms** | *56.3 ms (Sat)* | ✅ **5K Ceiling Passed (70 RPS)** |
| **80 RPS** | **16.0 ms** | **25.9 ms** | *53.0 ms (Sat)* | *2,769.6 ms (Sat)* | ⚠️ 5K/7K Saturated |
| **90 RPS** | **16.8 ms** | **28.4 ms** | *4,030.0 ms (Sat)* | *7,547.6 ms (Sat)* | ✅ **2K Ceiling Passed (90 RPS)** |
| **95 RPS** | **17.0 ms** | *988.4 ms (Sat)* | *Saturated* | *Saturated* | ⚠️ **2K Saturation Boundary** |
| **100 RPS** | **17.7 ms** | *3,237.0 ms (Sat)* | *Saturated* | *Saturated* | ⚠️ 1K Passed (17.7 ms) |
| **120 RPS** | **19.2 ms** | *Saturated* | *Saturated* | *Saturated* | ⚠️ 1K Passed (19.2 ms) |
| **140 RPS** | **20.8 ms** | *Saturated* | *Saturated* | *Saturated* | ⚠️ 1K Passed (20.8 ms) |
| **160 RPS** | **30.7 ms** | *Saturated* | *Saturated* | *Saturated* | ✅ **1K Ceiling Passed (160 RPS)** |
| **180 RPS** | *103.8 ms (Sat)* | *Saturated* | *Saturated* | *Saturated* | ⚠️ **1K Saturation Boundary** |
| **190 RPS** | *2,295.4 ms (Sat)*| *Saturated* | *Saturated* | *Saturated* | ⚠️ 1K Heavy Saturation |

---

## 3. Full Scenario Breakdown from Live In-Pod Test Run

### Phase 1: Multi-Payload Sweep (50 to 90 RPS)
```
[50 RPS @ 60s]
  1024B : Achieved= 50.02 RPS | P50= 11.5 ms | P90= 12.1 ms | P95= 12.6 ms | P99= 14.9 ms | PASS ✅
  2048B : Achieved= 50.02 RPS | P50= 16.3 ms | P90= 17.7 ms | P95= 19.1 ms | P99= 21.0 ms | PASS ✅
  5120B : Achieved= 50.02 RPS | P50= 23.5 ms | P90= 27.3 ms | P95= 28.7 ms | P99= 32.8 ms | PASS ✅
  7168B : Achieved= 50.00 RPS | P50= 29.2 ms | P90= 31.9 ms | P95= 33.3 ms | P99= 35.6 ms | PASS ✅

[60 RPS @ 60s]
  1024B : Achieved= 60.02 RPS | P50= 11.4 ms | P90= 12.0 ms | P95= 12.5 ms | P99= 15.0 ms | PASS ✅
  2048B : Achieved= 60.02 RPS | P50= 16.9 ms | P90= 18.7 ms | P95= 20.1 ms | P99= 22.5 ms | PASS ✅
  5120B : Achieved= 60.03 RPS | P50= 25.7 ms | P90= 30.3 ms | P95= 31.3 ms | P99= 34.7 ms | PASS ✅
  7168B : Achieved= 60.02 RPS | P50= 29.9 ms | P90= 35.1 ms | P95= 36.7 ms | P99= 40.9 ms | PASS ✅ (7K Max)

[70 RPS @ 60s]
  1024B : Achieved= 70.02 RPS | P50= 11.3 ms | P90= 11.8 ms | P95= 12.3 ms | P99= 16.1 ms | PASS ✅
  2048B : Achieved= 70.02 RPS | P50= 19.2 ms | P90= 21.0 ms | P95= 23.0 ms | P99= 26.5 ms | PASS ✅
  5120B : Achieved= 70.02 RPS | P50= 24.4 ms | P90= 29.2 ms | P95= 30.0 ms | P99= 32.1 ms | PASS ✅ (5K Max)
  7168B : Achieved= 70.03 RPS | P50= 34.2 ms | P90= 44.1 ms | P95= 46.1 ms | P99= 56.3 ms | SATURATED ⚠️

[80 RPS @ 60s]
  1024B : Achieved= 80.02 RPS | P50= 11.5 ms | P90= 12.1 ms | P95= 12.6 ms | P99= 16.0 ms | PASS ✅
  2048B : Achieved= 80.02 RPS | P50= 18.1 ms | P90= 21.3 ms | P95= 23.4 ms | P99= 25.9 ms | PASS ✅
  5120B : Achieved= 80.01 RPS | P50= 29.1 ms | P90= 38.3 ms | P95= 41.5 ms | P99= 53.0 ms | SATURATED ⚠️
  7168B : Achieved= 76.24 RPS | P50= 2188.8 ms| P90= 2611.1 ms| P95= 2674.2 ms| P99= 2769.6 ms| SATURATED ⚠️

[90 RPS @ 60s]
  1024B : Achieved= 90.02 RPS | P50= 11.9 ms | P90= 13.0 ms | P95= 13.5 ms | P99= 16.8 ms | PASS ✅
  2048B : Achieved= 90.02 RPS | P50= 18.3 ms | P90= 23.6 ms | P95= 25.5 ms | P99= 28.4 ms | PASS ✅ (2K Max)
  5120B : Achieved= 82.28 RPS | P50= 2688.5 ms| P90= 3836.3 ms| P95= 3932.8 ms| P99= 4030.0 ms| SATURATED ⚠️
  7168B : Achieved= 74.78 RPS | P50= 4734.6 ms| P90= 7164.3 ms| P95= 7377.1 ms| P99= 7547.6 ms| SATURATED ⚠️
```

### Phase 2: Dedicated 2 KB Saturation Sweep (90 to 100 RPS)
```
[90 RPS @ 60s]  : Achieved= 90.03 RPS | P50=  18.6 ms | P90=  21.2 ms | P95=  21.9 ms | P99=   23.3 ms | PASS ✅
[95 RPS @ 60s]  : Achieved= 93.56 RPS | P50= 480.9 ms | P90= 878.3 ms | P95= 957.2 ms | P99=  988.4 ms | SATURATED ⚠️
[100 RPS @ 60s] : Achieved= 93.23 RPS | P50= 2192.5 ms| P90= 3110.5 ms| P95= 3196.3 ms| P99= 3237.0 ms | SATURATED ⚠️
```

### Phase 3: Dedicated 1 KB Saturation Sweep (100 to 190 RPS)
```
[100 RPS @ 60s] : Achieved= 100.02 RPS | P50=  13.0 ms | P90=  14.5 ms | P95=  15.1 ms | P99=   17.7 ms | PASS ✅
[120 RPS @ 60s] : Achieved= 120.03 RPS | P50=  12.6 ms | P90=  13.6 ms | P95=  15.0 ms | P99=   19.2 ms | PASS ✅
[140 RPS @ 60s] : Achieved= 140.04 RPS | P50=  12.7 ms | P90=  17.0 ms | P95=  18.6 ms | P99=   20.8 ms | PASS ✅
[160 RPS @ 60s] : Achieved= 160.04 RPS | P50=  18.5 ms | P90=  25.0 ms | P95=  26.2 ms | P99=   30.7 ms | PASS ✅
[180 RPS @ 60s] : Achieved= 180.03 RPS | P50=  26.0 ms | P90=  70.4 ms | P95=  85.8 ms | P99=  103.8 ms | SATURATED ⚠️
[190 RPS @ 60s] : Achieved= 182.49 RPS | P50= 1412.6 ms| P90= 2214.6 ms| P95= 2259.9 ms| P99= 2295.4 ms | SATURATED ⚠️
```

---

## 4. Key Takeaways & Comparison with NVIDIA L4 GPU

1. **Sustained 60s Capacity Limits (P99 < 50 ms)**:
   * **1 KB**: Up to **160 RPS** (30.7 ms P99) $\rightarrow$ **4.0x NVIDIA L4's 40 RPS limit**.
   * **2 KB**: Up to **90 RPS** (23.3–28.4 ms P99) $\rightarrow$ **2.6x NVIDIA L4's 35 RPS limit**.
   * **5 KB**: Up to **70 RPS** (32.1 ms P99) $\rightarrow$ **3.5x NVIDIA L4's 20 RPS limit**.
   * **7 KB**: Up to **60 RPS** (40.9 ms P99) $\rightarrow$ **6.0x NVIDIA L4's 10 RPS limit**.

2. **Headroom vs. Production SLAs**:
   * A single TPU v5e chip easily satisfies PANW's baseline traffic requirement (20 RPS) with **3.0x to 8.0x margin**.
   * Peak traffic (40 RPS) across all payload sizes runs sub-41 ms P99 on a single TPU v5e chip.
