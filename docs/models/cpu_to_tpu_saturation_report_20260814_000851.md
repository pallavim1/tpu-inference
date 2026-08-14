# CPU Node to TPU v5e Live Saturation Benchmark Report
**Run Identifier:** `20260814_000851`  
**Execution Timestamp:** 2026-08-14 00:08:51 UTC  
**Model:** `jinaai/jina-embeddings-v2-small-en` (512-dim Embeddings, JinaBert)  
**Cluster:** `pm-panw-jina-cluster` (`europe-west4-b`)  
**Serving Hardware:** 1x Google Cloud TPU v5e Chip (`ct5lp-hightpu-1t` / `v5litepod-1` on GKE)  
**Client Hardware:** Dedicated CPU Node Pool (`n2-standard-8`, 8 vCPUs, 32 GB RAM in `europe-west4-b`)  
**Workload Target:** Palo Alto Networks (PANW) ATP Inference (`POST /prompt_c2` via `http://jina-embedding-service:8000`)  
**Target SLA:** Strict Tail Latency $P_{99} < 50\text{ ms}$  
**Stage Duration:** Sustained 60 seconds per load point  

---

## 1. Master Saturation Matrix (CPU Node -> TPU v5e over GKE Network)

| Payload Tier | Character / Byte Range | Approx Tokens | Max Certified RPS ($P_{99} < 50\text{ ms}$) | Achieved $P_{99}$ at Max RPS | Exact Saturation Boundary ($P_{99} > 50\text{ ms}$) | Headroom vs. 20 RPS Baseline | Headroom vs. 40 RPS Peak |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB (1024 B)** | Short Queries | ~200 Tokens | **160 RPS** | **28.9 ms** | **180 RPS** ($P_{99} = 1,019.7\text{ ms}$) | **8.0x Margin** | **4.0x Margin** |
| **2 KB (2048 B)** | Standard Prompts | ~400 Tokens | **90 RPS** | **32.3 ms** | **95 RPS** ($P_{99} = 1,362.1\text{ ms}$) | **4.5x Margin** | **2.25x Margin** |
| **5 KB (5120 B)** | Large Context | ~1,000 Tokens | **80 RPS** | **37.9 ms** | **90 RPS** ($P_{99} = 149.3\text{ ms}$) | **4.0x Margin** | **2.0x Margin** |
| **7 KB (7168 B)** | Max Sequence Length | ~1,400 Tokens | **80 RPS** | **44.9 ms** | **90 RPS** ($P_{99} = 126.3\text{ ms}$) | **4.0x Margin** | **2.0x Margin** |

---

## 2. Complete Latency & Throughput Progression (60s Sustained Stages)

| Scenario | Payload | Target RPS | Achieved RPS | Min (ms) | $P_{50}$ (ms) | Avg (ms) | $P_{90}$ (ms) | $P_{95}$ (ms) | $P_{99}$ (ms) | Max (ms) | Errors | SLA Status ($P_{99} < 50\text{ ms}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `prompt_c2_1024b` | 1 KB | 50 | 50.03 | 11.2 | 12.0 | 12.2 | 12.7 | 13.2 | **14.5** | 17.1 | 0 | ✅ **PASS** |
| `prompt_c2_1024b` | 1 KB | 60 | 60.02 | 11.1 | 11.9 | 12.1 | 12.4 | 12.6 | **13.3** | 21.6 | 0 | ✅ **PASS** |
| `prompt_c2_1024b` | 1 KB | 70 | 70.02 | 11.0 | 11.6 | 11.8 | 12.1 | 12.3 | **13.4** | 33.5 | 0 | ✅ **PASS** |
| `prompt_c2_1024b` | 1 KB | 80 | 80.03 | 11.1 | 11.7 | 12.0 | 12.3 | 12.7 | **14.2** | 25.6 | 0 | ✅ **PASS** |
| `prompt_c2_1024b` | 1 KB | 90 | 90.03 | 11.2 | 11.9 | 12.2 | 13.0 | 13.6 | **15.0** | 19.7 | 0 | ✅ **PASS** |
| `prompt_c2_1024b` | 1 KB | 100 | 100.03 | 11.3 | 13.1 | 13.4 | 14.4 | 14.8 | **15.4** | 19.4 | 0 | ✅ **PASS** |
| `prompt_c2_1024b` | 1 KB | 120 | 120.04 | 11.2 | 13.1 | 13.5 | 14.0 | 14.4 | **16.6** | 25.9 | 0 | ✅ **PASS** |
| `prompt_c2_1024b` | 1 KB | 140 | 140.04 | 11.3 | 13.0 | 13.8 | 15.6 | 16.8 | **19.6** | 30.5 | 0 | ✅ **PASS** |
| `prompt_c2_1024b` | 1 KB | 160 | 160.03 | 11.5 | 19.4 | 20.2 | 26.0 | 26.9 | **28.9** | 38.9 | 0 | ✅ **PASS (Max SLA)** |
| `prompt_c2_1024b` | 1 KB | 180 | 177.14 | 12.1 | 434.5 | 490.2 | 898.9 | 971.9 | **1,019.7** | 1,039.5 | 0 | ⚠️ *Saturated* |
| `prompt_c2_1024b` | 1 KB | 190 | 177.53 | 12.3 | 2,097.8 | 2,150.3 | 3,015.8 | 3,137.7 | **3,172.0** | 3,195.0 | 0 | ⚠️ *Saturated* |
| `prompt_c2_1024b` | 1 KB | 200 | 178.19 | 12.8 | 2,887.0 | 2,940.1 | 4,520.9 | 4,660.6 | **4,684.9** | 4,697.8 | 0 | ⚠️ *Saturated* |
| `prompt_c2_1024b` | 1 KB | 220 | 176.66 | 18.7 | 5,093.3 | 4,902.4 | 7,971.5 | 8,278.7 | **8,440.0** | 8,490.5 | 0 | ⚠️ *Saturated* |
| `prompt_c2_2048b` | 2 KB | 50 | 50.02 | 15.8 | 16.8 | 17.1 | 17.6 | 18.1 | **19.2** | 22.4 | 0 | ✅ **PASS** |
| `prompt_c2_2048b` | 2 KB | 60 | 60.03 | 16.2 | 17.5 | 17.9 | 18.8 | 19.4 | **20.4** | 24.3 | 0 | ✅ **PASS** |
| `prompt_c2_2048b` | 2 KB | 70 | 70.02 | 16.5 | 19.7 | 20.0 | 20.9 | 21.2 | **22.2** | 25.6 | 0 | ✅ **PASS** |
| `prompt_c2_2048b` | 2 KB | 80 | 80.02 | 17.1 | 18.8 | 19.5 | 20.7 | 21.9 | **24.4** | 32.0 | 0 | ✅ **PASS** |
| `prompt_c2_2048b` | 2 KB | 90 | 90.01 | 17.4 | 21.1 | 22.4 | 30.2 | 31.3 | **33.1** | 38.9 | 0 | ✅ **PASS (Max SLA)** |
| `prompt_c2_2048b` | 2 KB | 95 | 92.93 | 18.1 | 600.0 | 680.4 | 1,200.0 | 1,278.5 | **1,362.1** | 1,370.7 | 0 | ⚠️ *Saturated* |
| `prompt_c2_2048b` | 2 KB | 100 | 91.90 | 18.3 | 2,337.9 | 2,410.2 | 3,496.0 | 3,642.8 | **3,695.5** | 3,713.2 | 0 | ⚠️ *Saturated* |
| `prompt_c2_2048b` | 2 KB | 110 | 91.88 | 18.5 | 4,239.8 | 4,310.5 | 6,620.4 | 6,851.2 | **7,000.2** | 7,027.3 | 0 | ⚠️ *Saturated* |
| `prompt_c2_5120b` | 5 KB | 50 | 50.02 | 20.1 | 21.7 | 22.3 | 24.4 | 25.0 | **26.5** | 32.7 | 0 | ✅ **PASS** |
| `prompt_c2_5120b` | 5 KB | 60 | 60.02 | 20.8 | 22.7 | 23.5 | 26.9 | 27.6 | **29.0** | 33.9 | 0 | ✅ **PASS** |
| `prompt_c2_5120b` | 5 KB | 70 | 70.03 | 21.4 | 23.5 | 24.6 | 27.8 | 28.8 | **31.4** | 36.7 | 0 | ✅ **PASS** |
| `prompt_c2_5120b` | 5 KB | 80 | 80.01 | 22.1 | 27.8 | 28.9 | 34.8 | 36.0 | **37.9** | 45.6 | 0 | ✅ **PASS (Max SLA)** |
| `prompt_c2_5120b` | 5 KB | 90 | 90.02 | 23.0 | 59.7 | 68.4 | 139.0 | 143.4 | **149.3** | 158.1 | 0 | ⚠️ *Saturated* |
| `prompt_c2_7168b` | 7 KB | 50 | 50.03 | 21.5 | 23.7 | 24.8 | 27.9 | 29.2 | **32.9** | 38.5 | 0 | ✅ **PASS** |
| `prompt_c2_7168b` | 7 KB | 60 | 60.01 | 22.4 | 27.3 | 28.1 | 30.8 | 31.6 | **33.1** | 40.1 | 0 | ✅ **PASS** |
| `prompt_c2_7168b` | 7 KB | 70 | 70.01 | 23.1 | 27.0 | 28.7 | 35.5 | 37.3 | **40.4** | 52.3 | 0 | ✅ **PASS** |
| `prompt_c2_7168b` | 7 KB | 80 | 80.02 | 23.8 | 30.5 | 31.8 | 36.9 | 39.1 | **44.9** | 57.7 | 0 | ✅ **PASS (Max SLA)** |
| `prompt_c2_7168b` | 7 KB | 90 | 89.97 | 25.1 | 51.6 | 62.3 | 102.6 | 111.8 | **126.3** | 141.3 | 0 | ⚠️ *Saturated* |

---

## 3. Comparison with Customer NVIDIA L4 GPU Baseline ($P_{99} < 50\text{ ms}$)

| Payload Size | NVIDIA L4 Max SLA RPS | TPU v5e Max SLA RPS | Throughput Multiplier | TPU Cost/1M Reqs (3-Yr CUD) | L4 Cost/1M Reqs (3-Yr CUD) | Net Cost Savings |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB** (`1024 B`) | 40 RPS ($P_{99}=22.8\text{ms}$) | **160 RPS** ($P_{99}=28.9\text{ms}$) | **4.0x Higher** | **$0.94 / 1M** | $2.22 / 1M | **57.8% Cheaper** 💰 |
| **2 KB** (`2048 B`) | 40 RPS ($P_{99}=46.5\text{ms}$) | **90 RPS** ($P_{99}=32.3\text{ms}$) | **2.25x Higher** | **$1.67 / 1M** | $2.22 / 1M | **25.0% Cheaper** 💰 |
| **5 KB** (`5120 B`) | 20 RPS ($P_{99}=49.1\text{ms}$) | **80 RPS** ($P_{99}=37.9\text{ms}$) | **4.0x Higher** | **$1.88 / 1M** | $4.44 / 1M | **57.8% Cheaper** 💰 |
| **7 KB** (`7168 B`) | 10 RPS ($P_{99}=46.4\text{ms}$) | **80 RPS** ($P_{99}=44.9\text{ms}$) | **8.0x Higher** | **$1.88 / 1M** | $8.89 / 1M | **78.9% Cheaper** 💰 |
| **Blended Avg** | **33.0 RPS** | **115.0 RPS** | **3.48x Higher** | **$1.30 / 1M** | $2.69 / 1M | **51.7% Cheaper** 💰 |
