# End-to-End Benchmarking Guide: Jina Embeddings v2 on Cloud TPU v5e
## Automated Scripts & Complete Runbook for Palo Alto Networks (PANW)

**Workload:** Palo Alto Networks (PANW) ATP Embedding Inference (`POST /prompt_c2`)  
**Model:** `jinaai/jina-embeddings-v2-small-en` (512-dim, JinaBert)  
**Serving Engine:** vLLM (`v0.26.0`) on Google Cloud TPU v5e (`ct5lp-hightpu-1t` / 1x Chip)  
**Client Environment:** Dedicated CPU Node Pool (`cpu-benchmark-runner` on `n2-standard-8`)  
**Network Path:** GKE Cluster Network (`europe-west4-b`) via `http://jina-embedding-service:8000/prompt_c2`  
**Target SLA:** Strict Tail Latency $P_{99} < 50\text{ ms}$  

---

## 🚀 Quick Start: Automated Script Execution (Recommended)

Instead of running individual commands manually, execute each phase using the self-contained automation scripts in `scripts/`:

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                AUTOMATED SCRIPT WORKFLOW                                 │
│                                                                                          │
│  1. Provision GKE Cluster & Nodepools   ──►  ./scripts/01_provision_infrastructure.sh   │
│  2. Deploy TPU v5e Inference Service    ──►  ./scripts/02_deploy_tpu_workload.sh        │
│  3. Setup Benchmark Runner Pod          ──►  ./scripts/03_setup_benchmark_runner.sh     │
│  4. Run Baseline Sweeps (Up to 40 RPS)  ──►  ./scripts/04_run_baseline_benchmarks.sh    │
│  5. Run Full Saturation Benchmark Suite ──►  ./scripts/05_run_saturation_benchmarks.sh  │
│  6. Export Results & Excel Sheets       ──►  ./scripts/06_export_results.sh             │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

> [!TIP]
> **One-Click Master Run**: To execute steps 1 through 5 end-to-end in a single command, run:
> ```bash
> ./scripts/run_all_end_to_end.sh
> ```

---

## Step-by-Step Script Execution

### Step 1: Provision GKE Cluster & Nodepools
Creates the VPC, subnets, GKE cluster (`pm-panw-jina-cluster`), Cloud TPU v5e nodepool (`pm-panw-jina-tpu-pool`), and dedicated CPU nodepool (`cpu-benchmark-pool`):

```bash
./scripts/01_provision_infrastructure.sh
```

---

### Step 2: Deploy Jina Embeddings vLLM Service on TPU v5e
Deploys the vLLM model server with the high-performance async adapter proxy on port 8000, verifies pod readiness, and runs a live smoke test:

```bash
# Default: FP16 / BF16 Precision (Recommended)
./scripts/02_deploy_tpu_workload.sh --precision fp16

# Alternative: FP32 Precision
./scripts/02_deploy_tpu_workload.sh --precision fp32
```

---

### Step 3: Setup Benchmark Runner on CPU Node Pool
Provisions the `cpu-benchmark-runner` pod on the isolated CPU nodepool, installs `k6`, `openpyxl`, and `pandas`, syncs all benchmark suites to `/workspace`, and tests internal GKE network reachability:

```bash
./scripts/03_setup_benchmark_runner.sh
```

---

### Step 4: Run Baseline Benchmarks (Up to 40 RPS)
Runs the PANW baseline evaluation matrix (1, 5, 7, 10, 20, 30, 40 RPS) across 1K, 2K, 5K, and 7K payloads:

```bash
# Run 20 RPS fixed baseline across all 4 payloads:
./scripts/04_run_baseline_benchmarks.sh --rps 20

# Run 40 RPS baseline across all 4 payloads:
./scripts/04_run_baseline_benchmarks.sh --rps 40

# Run full 48-stage matrix sweep (1, 5, 7, 10, 20, 30, 40 RPS x 4 payload sizes):
./scripts/04_run_baseline_benchmarks.sh --matrix
```

---

### Step 5: Run Full Saturation Benchmarks (Until Saturation for Each Payload)
Executes all 17 saturation stages (60s each) across 1K, 2K, 5K, and 7K payloads, automatically creates isolated timestamped directories (`/workspace/benchmark_runs/run_<timestamp>/`), parses telemetry, and downloads the deliverables locally:

```bash
# Run all phases (Multi-payload 50-90 RPS, 2K 90-110 RPS, 1K 100-220 RPS):
./scripts/05_run_saturation_benchmarks.sh --duration 60s --phase all

# Run specific phase only:
./scripts/05_run_saturation_benchmarks.sh --phase multi   # 50 to 90 RPS sweep (1K, 2K, 5K, 7K)
./scripts/05_run_saturation_benchmarks.sh --phase 2k      # 90 to 110 RPS sweep (2K only)
./scripts/05_run_saturation_benchmarks.sh --phase 1k      # 100 to 220 RPS sweep (1K only)
```

---

### Step 6: Export Results & Excel Workbooks
Downloads all generated telemetry, JSON summaries, and multi-tab Excel workbooks from the cluster to your local workstation:

```bash
./scripts/06_export_results.sh ./my_local_results
```

---

## 📊 Master Saturation & Headroom Reference Matrix

All results measured from the dedicated CPU nodepool pod across the GKE internal cluster network with sustained 60-second stages:

| Payload Size | Approx Tokens | NVIDIA L4 Max SLA RPS ($P_{99} < 50\text{ ms}$) | TPU v5e Max SLA RPS ($P_{99} < 50\text{ ms}$) | Exact Saturation Boundary ($P_{99} > 50\text{ ms}$) | Headroom vs. 20 RPS Baseline | Headroom vs. 40 RPS Peak | Net Cost Savings (3-Yr CUD) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 KB** (`1024 B`) | ~200 | **40 RPS** | **160 RPS** ($P_{99}=28.9\text{ms}$) | **180 RPS** ($P_{99}=1,019.7\text{ms}$) | **8.0x Margin** | **4.0x Margin** | **57.8% Cheaper** 💰 |
| **2 KB** (`2048 B`) | ~400 | **40 RPS** | **90 RPS** ($P_{99}=32.3\text{ms}$) | **95 RPS** ($P_{99}=1,362.1\text{ms}$) | **4.5x Margin** | **2.25x Margin** | **25.0% Cheaper** 💰 |
| **5 KB** (`5120 B`) | ~1,000 | **20 RPS** | **80 RPS** ($P_{99}=37.9\text{ms}$) | **90 RPS** ($P_{99}=149.3\text{ms}$) | **4.0x Margin** | **2.0x Margin** | **57.8% Cheaper** 💰 |
| **7 KB** (`7168 B`) | ~1,400 | **10 RPS** | **80 RPS** ($P_{99}=44.9\text{ms}$) | **90 RPS** ($P_{99}=126.3\text{ms}$) | **4.0x Margin** | **2.0x Margin** | **78.9% Cheaper** 💰 |
| **Blended Avg** | ~500 | **33.0 RPS** | **115.0 RPS** | — | **5.75x Margin** | **2.88x Margin** | **51.7% Cheaper** 💰 |

---

## 📁 Repository Directory Structure

```
models/JinaEmbedding/vLLM/
├── scripts/
│   ├── run_all_end_to_end.sh            # One-click master execution script
│   ├── 01_provision_infrastructure.sh   # GKE Cluster, TPU v5e pool & CPU pool setup
│   ├── 02_deploy_tpu_workload.sh        # Deploy vLLM service on TPU v5e (FP16/FP32)
│   ├── 03_setup_benchmark_runner.sh     # Setup runner pod, k6, dependencies & scripts
│   ├── 04_run_baseline_benchmarks.sh    # Run baseline benchmarks (up to 40 RPS)
│   ├── 05_run_saturation_benchmarks.sh  # Run saturation sweeps (until saturation)
│   └── 06_export_results.sh             # Export all results & Excel workbooks locally
├── deploy/
│   ├── jina_v5e_deployment.yaml         # Kubernetes Deployment & Service with vLLM proxy
│   ├── cpu_benchmark_runner.yaml        # Benchmark runner pod manifest
│   └── cluster_setup.sh                 # Cluster setup bash script
├── benchmarks/
│   ├── BENCHMARKING_GUIDE.md            # This end-to-end guide
│   ├── k6_ray_serve_test.js             # PANW baseline test suite
│   ├── k6_high_rps_saturation_test.js  # Multi-payload saturation load script
│   ├── k6_1kb_saturation_test.js        # 1 KB dedicated saturation script (100–220 RPS)
│   ├── k6_2kb_saturation_test.js        # 2 KB dedicated saturation script (90–110 RPS)
│   ├── run_timestamped_benchmark_suite.py # Timestamped benchmark runner
│   ├── analyze_k6_results.py            # Latency percentile & telemetry parser
│   └── generate_consolidated_excel.py  # Multi-tab Excel workbook builder
├── reports/
│   ├── cpu_to_tpu_saturation_report_20260814_000851.md  # Latest timestamped report
│   ├── jina_embeddings_v2_tpu_v5e_benchmarks_20260814_000851.xlsx # Multi-tab Excel workbook
│   ├── tpu_v5e_vs_l4_perf_per_dollar_analysis.md        # FP32 Economic Model
│   └── tpu_v5e_vs_l4_perf_per_dollar_analysis_fp16.md   # FP16 Economic Model
└── results/
    ├── fp32/                            # Raw JSON summaries and Excel sheets (FP32)
    └── fp16/                            # Raw JSON summaries and Excel sheets (FP16)
```
