# End-to-End Benchmarking Guide: Jina Embeddings v2 on Cloud TPU v5e
## Abstracted Modular Workflow (Steps 1 – 6)

**Workload:** Palo Alto Networks (PANW) ATP Embedding Inference (`POST /prompt_c2`)  
**Model:** `jinaai/jina-embeddings-v2-small-en` (512-dim, JinaBert)  
**Serving Engine:** vLLM (`v0.26.0`) on Google Cloud TPU v5e (`ct5lp-hightpu-1t` / 1x Chip)  
**Client Environment:** Dedicated CPU Node Pool (`cpu-benchmark-runner` on `n2-standard-8`)  
**Network Path:** GKE Cluster Network (`europe-west4-b`) via `http://jina-embedding-service:8000/prompt_c2`  
**Target SLA:** Strict Tail Latency $P_{99} < 50\text{ ms}$  

---

## 🚀 Quickstart: Executive Abstracted Workflow

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│  1. Provision GKE Cluster & Nodepools   ──►  ./scripts/01_provision_infrastructure.sh   │
│  2. Deploy TPU v5e Inference Service    ──►  ./scripts/02_deploy_tpu_workload.sh        │
│  3. Setup Benchmark Runner Pod          ──►  ./scripts/03_setup_benchmark_runner.sh     │
│  4. Run Baseline Sweeps (Up to 40 RPS)  ──►  ./scripts/04_run_baseline_benchmarks.sh    │
│  5. Run Full Saturation Benchmark Suite ──►  ./scripts/05_run_saturation_benchmarks.sh  │
│  6. Export Results & Excel Sheets       ──►  ./scripts/06_export_results.sh             │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Execution Details

### Step 1: Provision Infrastructure & GKE Cluster
Provisions VPC networks, subnets, GKE cluster, Cloud TPU v5e node pool, and dedicated CPU benchmark runner node pool:
```bash
./scripts/01_provision_infrastructure.sh
```

---

### Step 2: Deploy TPU v5e Inference Service
Deploys `vllm serve` with `--max-model-len 8192` and `--max-num-batched-tokens 8192` targeting JAX ALiBi Flash-Attention embedding kernels:
```bash
./scripts/02_deploy_tpu_workload.sh
```

---

### Step 3: Setup Benchmark Runner Pod
Spawns the isolated `cpu-benchmark-runner` pod on the CPU node pool and installs k6 load testing tools:
```bash
./scripts/03_setup_benchmark_runner.sh
```

---

### Step 4: Run Baseline Benchmarks (Up to 40 RPS)
Executes baseline load sweeps across 1K, 2K, 5K, 7K, and 8K token payloads at 40 RPS constant arrival rate:
```bash
./scripts/04_run_baseline_benchmarks.sh
```

---

### Step 5: Run Full Saturation Benchmark Suite
Runs the one-click saturation benchmark runner suite across 50 RPS, 60 RPS, 70 RPS, 80 RPS, and 90 RPS target stages:
```bash
./scripts/05_run_saturation_benchmarks.sh
```

---

### Step 6: Export Results & Excel Telemetry Workbook
Parses raw telemetry into JSON metrics and exports the consolidated multi-tab Excel workbook:
```bash
./scripts/06_export_results.sh
```

---

## Master Saturation & Headroom Reference (8192 Max Model Len)

| Payload Size | Approx Tokens | L4 Baseline Max RPS ($P_{99} < 50\text{ ms}$) | TPU v5e Max RPS ($P_{99} < 50\text{ ms}$) | Exact Saturation Boundary | Headroom vs. 20 RPS Baseline | Headroom vs. 40 RPS Peak |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB** (`1024 B`) | ~200 | **40 RPS** | **160 RPS** ($P_{99}=28.9\text{ms}$) | **180 RPS** ($P_{99}=1,019.7\text{ms}$) | **8.0x Margin** | **4.0x Margin** |
| **2 KB** (`2048 B`) | ~400 | **40 RPS** | **90 RPS** ($P_{99}=32.3\text{ms}$) | **95 RPS** ($P_{99}=1,362.1\text{ms}$) | **4.5x Margin** | **2.25x Margin** |
| **5 KB** (`5120 B`) | ~1,000 | **20 RPS** | **80 RPS** ($P_{99}=37.9\text{ms}$) | **90 RPS** ($P_{99}=149.3\text{ms}$) | **4.0x Margin** | **2.0x Margin** |
| **7 KB** (`7168 B`) | ~1,400 | **10 RPS** | **80 RPS** ($P_{99}=44.9\text{ms}$) | **90 RPS** ($P_{99}=126.3\text{ms}$) | **4.0x Margin** | **2.0x Margin** |
