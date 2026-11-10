# Kimi K3.1 Pro on Cloud TPU v7x (Ironwood) — End-to-End Validation & Benchmark Report

> [!NOTE]
> Independent field validation of the public llm-d `kimi-k3-1` guide (SGLang on TorchTPU) on **Spot** Cloud TPU v7x capacity, run 2026-11-09/10. These are not official Google benchmarks. Spot preemptions limited the serving windows; numbers can differ on reserved capacity or newer images. Deployment guide: [README.md](README.md).

> [!NOTE]
> **Status (2026-11-10 16:00Z):** Stages 1A, 1B, 1C, 2A, 2B and 3 are **done and benchmarked**. **Stage 4/5 (full 93-layer model) has been served twice:**
> - Attempt #4: `Ready` at 14:18Z, served for ~24 min.
> - Attempt #6: two replicas `Ready` at 15:45Z and 15:46Z, served for ~8 min.
>
> Across those windows it passed the G1–G4 validation gates and the 477K-token context check (G4X) through the `llm-d` router, and completed most throughput sweeps (Section 5.5). The guide's `inference-perf` benchmark (G5B) needs a longer window: about 15–30 min of serving after a ~24–34 min startup, while most Spot slices live ~40 min. It, the shared-prefix and decode-heavy sweeps are queued and run automatically on the next slice. Section 5.3 covers the startup attempts, the shared compile-cache results and the Spot adaptation. Four separate single-host nodes can't stand in for a slice (Section 5.1). Stage 1C (Section 6) adds expanded single-host sweeps, llm-d routing A/B and `inference-perf` results on the 12-layer configuration.

## 1. Environment & Stack Fingerprint

| Parameter | Verified Value |
| --- | --- |
| **GCP Project / Zone** | `<your-project>` · `us-central1-c` |
| **GKE Cluster** | a GKE Standard cluster (`1.36.5-gke.1982000`, `RAPID`, VPC MTU `8896`, Workload Identity, Gateway API, GCS FUSE CSI) |
| **TPU Hardware** | Cloud TPU v7x (Ironwood) `tpu7x-standard-4t` (`4` chips / `8` TensorCores / `224` vCPUs / `960 GB` host DRAM per VM, `97,012 MB` HBM per TensorCore) |
| **Cluster Node Pools** | `tpu7x-2x2x1-spot` (single-host, fixed `5 × tpu7x-standard-4t` = `20` Spot chips target; Spot preemptions keep 3–5 up at a time) + `tpu7x-2x2x4-spot` / `-2` / `-3` (multi-host `2x2x4` slice pools via `HIGH_THROUGHPUT` placement policies; Spot slices granted intermittently from 10:10Z, each reclaimed after ~34–40+ min) + on-demand CPU pools (`default-pool`, `cpu-infra-pool`) |
| **Model** | `moonshotai/Kimi-K3.1-Pro` (`KimiK31ForConditionalGeneration`, 2.8T Hybrid: 69 KDA linear-attention + 24 Sparse-MLA DSA top-2048 layers, 896 routed MXFP4 experts top-16 LatentMoE, `1,454.13 GiB` / `1.56 TB` across 96 `.safetensors` shards) |
| **Container Image** | `docker.io/lmsysorg/sglang@sha256:84fa8579cfb609d383a64fdd64c0036045140c905b7bd8a5ee9e6cadfe3cf433` (`dev-kimi-k31-pro-tpu`) |
| **Runtime Versions** | `PyTorch 2.11.0+cpu` (`torch_tpu`), `libtpu 0.0.50` (`TFRT TPU v7x`), `sglang 0.5.22`, `sglang_tpu 0.1.1` |
| **Orchestration & Routing** | `LeaderWorkerSet v1.0.0` + `llm-d-router-standalone v0` (`approx-prefix-cache-producer`, `prefix-cache-affinity-filter`, `token-load-scorer`) |

---

## 2. Stage 1A — TorchTPU Runtime & MXU `torch.compile` Smoke Test (`PASSED`)

Executed on a single `tpu7x-standard-4t` Spot node (`2x2x1`):

| Check | Result |
| --- | --- |
| **`PrivateUse1` Backend Registration** | Renamed cleanly to `tpu`; PJRT loaded `/usr/local/lib/python3.12/site-packages/libtpu/libtpu.so` (`0.0.50`) |
| **Visible TensorCores (`torch.accelerator.device_count()`)** | **`8` TensorCores** (`4` chips × `2` TensorCores/chip; `97.0 GB` HBM3e per TensorCore = `776 GB` HBM/host) |
| **`torch.compile(backend="tpu", dynamic=False)` MXU Kernel** | BF16 `[1024 × 7168] @ [7168 × 7168] + SiLU`: **`0.151 ms`** avg step latency (~**`697 TFLOP/s`** achieved on a single TensorCore) |
| **Tier-3 Compilation Cache Speedup** | `VL EXTEND (tokens=4096, bs=4)` compile dropped from **`74.94 s` cold → `7.80 s` warm (`9.61×` speedup)** |

---

## 3. Stage 1B — 12-Layer Kimi K3.1 Pro Full-Kernel Benchmark (`TP=8, EP=8`, Single Host `2x2x1`) (`PASSED`)

Validated one complete `attn_res_block_size=12` hybrid residual block of `moonshotai/Kimi-K3.1-Pro` (Layer 1 dense + KDA, 8 KDA + 896-expert top-16 MXFP4 LatentMoE layers `[2,3,5,6,7,9,10,11]`, and 3 Gated Sparse-MLA + DSA top-2048 + MXFP4 LatentMoE layers `[4,8,12]`, plus full 27-layer MoonViT-V2 vision tower) with `--attention-backend=pallas_tpu_mla_sparse --kv-cache-dtype=fp8_e4m3 --page-size=128 --mamba-ssm-dtype=bfloat16 --enable-hierarchical-cache --hicache-io-backend=raiden --hicache-mem-layout=page_first_fused --hicache-write-policy=write_through --hicache-ratio=2 --reasoning-parser=kimi_k3`.

### 3.1 Startup, Memory, and Graph Partitioning Metrics

| Metric | Measured Value |
| --- | --- |
| **CPU Fallback Partitions** | **`0` CPU fallbacks** across all 29 vision buckets, prefill (`4096`), and all 5 decode buckets (`bs=[1,2,4,8,16]`) (`Split graph into 1 partitions, 0 of which run on CPU`) |
| **Engine Startup Breakdown** | `load_weight=4.11 s`, `kv_cache_allocation=0.59 s`, `decode_compile=463.31 s`, `scheduler_e2e=506.49 s` |
| **HBM Footprint (`--mem-fraction-static=0.50`, `--max-total-tokens=524288`)** | **`41,513 / 97,012 MB`** per TensorCore |
| **Host RSS After Heap Trim (`TP0..TP7`)** | **`34.01 – 34.95 GiB`** per rank (`~274.3 GiB` total across 8 ranks + `HiCache raiden` `KV + MAMBA` host DRAM pools) |
| **Post-Startup XLA Graph Recompilations** | **`0`** across 206 benchmark requests and `22,373` decode steps |

### 3.2 Functional, Reasoning Parser, & Multi-Chunk Prefix-Cache Gates

| Gate | Workload | Cold Latency | Warm Prefix-Hit Latency | Cached Tokens (`UnifiedRadixCache` + `HiCache raiden`) | Speedup / Result |
| --- | --- | --- | --- | --- | --- |
| **Gate 2** | `/v1/chat/completions` (`110` in / `64` out, `kimi_k3` parser) | `1,241.5 ms` | — | `reasoning_tokens: 64` cleanly extracted | **PASS** |
| **Gate 3** | Multi-turn `5,127`-token prompt + `16` decode tokens | `1,028.9 ms` | **`873.1 ms`** | **`5,120 / 5,127`** (exact 128-token page boundary) | **`1.18×` PASS** |
| **Gate 4** | Long-context `22,657`-token chunked prefill (`6 × 4096`) + `32` decode | `884.0 ms` (~`25.6K` prefill tok/s) | **`275.0 ms`** | **`20,480 / 22,657`** (`KV + KDA Mamba` state hit) | **`3.21×` PASS** |

### 3.3 `sglang.benchmark.serving` Concurrency & Shared-Prefix Sweep (Stage 1B, `1× tpu7x-standard-4t`)

| Workload (`in / out`) | Max Concurrency (`C`) | Prompts | Req/s | Input `tok/s` | Output `tok/s` (Peak) | Total `tok/s` | Median TTFT (`P99`) | Median TPOT (`P95`) | Median E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uncached `1024 / 128` | `1` | `8` | `1.55` | `1,586.6` | **`198.3`** (`214.0`) | `1,784.9` | **`106.4 ms`** (`146.7 ms`) | **`4.14 ms`** (`4.30 ms`) | `634.7 ms` |
| Uncached `1024 / 128` | `4` | `16` | `4.58` | `4,691.9` | **`586.5`** (`796.0`) | `5,278.3` | **`142.0 ms`** (`835.0 ms`) | **`4.35 ms`** (`4.46 ms`) | `700.9 ms` |
| Uncached `1024 / 128` | `8` | `32` | `7.60` | `7,780.0` | **`972.5`** (`1,375.0`) | `8,752.5` | **`264.6 ms`** (`1,144.1 ms`) | **`4.43 ms`** (`10.60 ms`) | `821.2 ms` |
| Uncached `1024 / 128` | `16` | `64` | `11.66` | `11,942.1` | **`1,492.8`** (**`2,034.0`**) | `13,434.9` | **`377.6 ms`** (`1,338.6 ms`) | **`6.40 ms`** (`12.67 ms`) | `1,177.6 ms` |
| **Shared Prefix `3072+512 / 128`** | **`16`** | **`48`** | **`13.72`** | **`49,156.1`** | **`1,755.6`** (`1,824.0`) | **`50,911.7`** | **`416.9 ms`** (`560.6 ms`) | **`5.84 ms`** (`7.49 ms`) | `1,063.4 ms` |

- **Prometheus `/metrics` Validation**: `sglang:cache_hit_rate = 0.8571` (**`85.71%`**, matching the exact theoretical `3072 / 3584 = 85.71%` shared-prefix fraction across `UnifiedRadixCache` + `HiCache raiden`).

---

## 4. Stage 2 — 1.56 TB Checkpoint Hydration Throughput Benchmarks (`PASSED`)

| Hydration Stage | Architecture | Total Volume | Sustained Throughput | Wall-Clock Time |
| --- | --- | --- | --- | --- |
| **Stage 2A: Hugging Face → Regional GCS** (`gs://<model-bucket>`) | 12-pod Indexed K8s `Job` across `6 × n2d-standard-32` (`aria2 -x16` → `24 GiB` `emptyDir: Memory` RAM disk → parallel composite `gcloud storage cp`) | `112` files (`96` `.safetensors`), **`1,454.13 GiB` (`1.56 TB`)** | **`~4.0 – 4.5 GiB/s`** (`331 MiB/s`/worker, `49 s` per `15.82 GiB` shard) | **`10m 00s`** |
| **Stage 2B: Regional GCS → `ReadOnlyMany` `hyperdisk-ml`** (`2,000 GiB`, `6,400 MiB/s`) | Direct RWO CSI staging on 224-vCPU `tpu7x-standard-4t` (`32 × 4` parallel `gcloud storage cp`) → clean detach → GCE `READ_ONLY_MANY` → static `ROX` PV/PVC (`kimi-k3-1-models`) | `112` files (`96` `.safetensors`), **`1,454.13 GiB` (`1.56 TB`)** | **`2.6 GiB/s` avg** (up to `6.25 GiB/s` disk ceiling) | **`10m 28s` (`628 s`)** |

---

## 5. Stage 4 & 5 — Full 93-Layer `moonshotai/Kimi-K3.1-Pro` (`2x2x4`, `TP=32 / EP=8`, `llm-d`) — Served on a Spot Slice

The full model ran with the 1.56 TB checkpoint on a `ReadOnlyMany` Hyperdisk ML volume, behind the `llm-d` router, as a `LeaderWorkerSet` (`TP=32 / EP=8`, `nnodes=4`, guide config verbatim). It needs all 16 chips in **one** `2x2x4` ICI slice. Spot granted slices only intermittently and reclaimed each after ~34–40 min, so the first three startups were preempted before `Ready`. **Attempt #4 reached `Ready` at 14:18:10Z**, served for ~24 min and ran most Stage 5 gates (Sections 5.3–5.5) before its slice was preempted at 14:42Z.

| Signal | Observation |
| --- | --- |
| `2x2x4` slice requests | 06:30–07:32Z: 13 attempts (52 VM creates) all failed with `ZONE_RESOURCE_POOL_EXHAUSTED` (`STOCKOUT`). From 10:10Z, slices were granted intermittently: 10:10, 12:02, 12:28 and 13:40Z, plus one on the second pool that was preempted at 13:42Z before it could be used. Lifetimes before preemption: 40.0, 39.3, 33.7, **62.0** (13:40Z slice, served the full model for ~24 min) and 4.2 min. |
| Spot reclaim rate | Our single-host nodes were preempted 7–15 times/hour (01:00–07:00Z). |
| DWS Flex-Start | Flex-Start requests were queued but not fulfilled during the test window. |
| Other zones | No usable `tpu7x` Spot capacity or quota for this project in other zones during the test window. |

### 5.1 Attempts to Run the Full Model on 4 Separate Single-Host Nodes (16 Chips)

| Attempt | Configuration | Result |
| --- | --- | --- |
| 1 (07:58Z) | Guide config unchanged (`TP=32, EP=8, nnodes=4`) on 4 × `2x2x1` nodes | All ranks exit at startup: `ValueError: Topology does not match the host and device counts`. `sglang_tpu` requires all devices in one slice ("partial slices ... are unsupported"). |
| 2 (08:03Z) | `--no-tpu-node-normalization` + MegaScale DCN settings (4 slices of `2x2x1`) | All 32 ranks rendezvoused, then failed in TorchTPU: `create_process_group(): invalid world_size configuration: expected 1 or 8, but got 32`. |

**Conclusion:** with this SGLang/TorchTPU build, the full model can't span separate single-host slices. It needs a `2x2x4` slice from Spot, a reservation, on-demand capacity, or DWS Flex-Start.

### 5.2 Stage 4 Startup on a Granted Spot `2x2x4` Slice (10:10Z, preempted at 10:50Z)

Spot granted a `2x2x4` slice (`tpu7x-2x2x4-spot-2`, 4 nodes Ready 10:10:40–10:10:58Z). The armed `LeaderWorkerSet` (guide config verbatim) bound to it automatically, and startup progressed as follows:

| Startup phase | Measured |
| --- | --- |
| Multi-host init (`TORCH_TPU_TOPOLOGY=2,2,4,2`, 4 hosts × 8 TensorCores, node-order normalization) | `Init parallel` **6.4 s**; two-level Pallas all-reduce enabled (`TP 32 = 4 hosts x 8 ranks`) |
| **1.56 TB weight load** from the `ROX` Hyperdisk ML volume | **375.2 s**; HBM after load **60.2 GB / 97.0 GB** per TensorCore; static-fp8 GEMV binding freed 10.9 GB of bf16 weights |
| KV cache (`fp8_e4m3`, page 128) | **532,352 tokens** (10.66 GB per TensorCore), exactly the guide's router sizing |
| KDA / Mamba state cache | `max_size=708`, 69 layers, 9.59 GB per TensorCore |
| Graph partitioning | `Split graph into 1 partitions, 0 of which run on CPU` |
| Vision precompile (29 buckets) | ~5 min (done 10:24Z) |
| Prefill compile (`VL EXTEND`, 4096 tokens, bs=8) | **456.2 s**; HBM 86.2 / 97.0 GB |
| Decode compile (`VL DECODE`) | bs=1 117.8 s · bs=2 126.6 s · bs=4 134.1 s · bs=8 134.4 s · bs=16 146.8 s · bs=32 194.6 s · bs=64 still compiling |
| Outcome | **All 4 slice VMs preempted together at 10:50:41Z** (`compute.instances.preempted`), ~40 min after the slice was granted and seconds before startup would have finished. 90.3 GB of compiled programs were already saved to the GCS Tier-3 cache, so the next start should skip most compilation. The MIG immediately queued a replacement slice. |

### 5.3 Startup Attempts on Spot Slices and the First Full-Model `Ready`

| Attempt | Slice granted | Weight load | Compile | Outcome |
| --- | --- | ---: | --- | --- |
| #1 | 10:10Z (pool `-2`) | 375 s | prefill 456 s; decode bs 1–32 | Preempted at 10:50:41Z (40.0 min), during the bs=64 decode compile |
| #2 | 12:02Z (pool `-2`) | 339 s | prefill 453 s; decode 115–230 s | All compiles finished ~12:39Z; preempted at 12:41:35Z (39.3 min), just before `Ready` |
| #3 | 12:28Z (pool `-1`) | 337 s | restarted at 12:51Z | Preempted at 13:01:52Z (33.7 min) |
| **#4** | **13:40Z (pool `-2`)** | **316 s** | **prefill 463 s; decode 1,078 s** | **`Ready` at 14:18:10Z**; served ~24 min (first request 14:21:09Z); preempted at 14:42:26Z (slice lifetime 62.0 min) |
| #5 | 14:38Z (pool `-1`) | — | — | Preempted at 14:42:23Z after ~4 min, during pod startup |
| #6 | **15:11Z, two slices at once** (pools `-1` and `-2`, one replica each) | 572 s / 565 s (both replicas read the same volume at once) | **First run on the shared Tier-3 cache.** Prefill 167 s / 139 s (was 463 s). Decode buckets 110–163 s each (was 116–229 s). Total compile 1,048 s (was 1,541 s). | Replicas `Ready` at 15:44:50Z and 15:46:27Z (33.7 min after the grant). G4X done. Both slices preempted at 15:52:57Z (lifetime 41.9 min) |
| #7 | 16:56Z (pool `-2`), first run with the Spot adaptation | 313 s (single replica) | — | Preempted at 17:05:41Z (~9 min), during the vision precompile |

> [!NOTE]
> **Spot adaptation (from attempt #7 on, disclosed deviation):** Most slices lived ~40 min (40.0, 39.3, 33.7, 41.9, 41.9 min; one 62 min). A ~34 min startup leaves only a few minutes of serving. So, while no replica was running, the `LeaderWorkerSet` was changed in two ways:
> - `replicas: 1`, because two replicas loading 1.56 TB each from the same `ReadOnlyMany` volume slowed the weight load from 316 s to ~570 s.
> - `--tpu-compile-bs 1 8 16 32 64` instead of the guide's `1 2 4 8 16 32 64`. This saves ~4.6 min of decode compile; batches of 2–7 pad to the bs=8 graph.
> - From attempt #8 on: `SGLANG_TPU_VIT_PRECOMPILE_SIZES=""`. Per `sglang_tpu/envs.py`, set-but-empty disables the vision-encoder precompile ladder, saving ~4 min. Serving is text-only in this configuration; all Stage 5 workloads are text.
>
> All results before this note use the guide's configuration unchanged.

**Attempt #4 startup breakdown** (`Engine startup timings`: `load_weight=315.71 s`, `kv_cache_allocation=0.55 s`, decode graph capture `1,759.66 s`, `scheduler_e2e=2,126.33 s`):

| Phase | Time (UTC) | Duration |
| --- | --- | ---: |
| Slice VMs created → pods started | 13:40:24 → 13:40:52 | ~0.5 min |
| Multi-host init (`Init parallel`) | 13:42:48 | 7.7 s |
| 1.56 TB weight load (Hyperdisk ML `ROX`) | → 13:48:05 | 315.7 s |
| Vision precompile | → 13:51:50 | ~3.75 min |
| Prefill compile (`VL EXTEND`, 4096 tokens, bs=8) | 13:51:57 → 13:59:41 | 463.2 s |
| Decode compile, bs = 1 / 2 / 4 / 8 / 16 / 32 / 64 | 13:59:41 → 14:17:38 | 115.6 / 125.0 / 131.3 / 134.7 / 146.8 / 195.3 / 228.9 s |
| Server `fired up and ready to roll` | 14:18:10 | **37.7 min after the slice was granted** |
| First request (completes the final global warmup, then `torch.compile` frozen with `fail_on_recompile`) | 14:20:25 → 14:21:09 | 44 s |

- HBM after compile: **86.7 / 97.0 GB** per TensorCore. Host RSS: ~63–65 GiB per rank after heap trim (~564 GiB per host pod).
- 1,130 XLA programs per rank, ~705 s of XLA compile time per rank. None of them came from the compile cache (next bullet).

> [!WARNING]
> **Why attempt #4 compiled from scratch:** the shared Tier-3 cache template (applied at 12:51Z) never reached group 0. The `LeaderWorkerSet` rolling update (`maxSurge=0`, `maxUnavailable=1`) updates the highest group index first. Group 1 got the new template but never became `Ready` (no second slice), so the rollout stalled and every recreation of group 0 reused the old revision, whose cache directories are keyed by pod index. Node normalization assigns ranks per slice, so those per-pod directories missed.
>
> The shared directory **would have hit**: all 160 programs that attempt #4 compiled by 14:09Z (40 per host) were already in it under the same content-keyed names. A watcher switched the rollout to `maxUnavailable=2` when the 14:42Z preemption disrupted group 0, so both groups use the shared-cache revision from attempt #6 on.
>
> **Measured effect (attempt #6):** every XLA program was a cache hit; no new objects were written. But the cache removes only XLA's own compile step. Prefill compile dropped ~3× (463 s → 139–167 s). Each decode bucket still took ~110–160 s, mostly dynamo tracing and AOT lowering. Startup was therefore ~34 min, partly because two replicas loaded weights at the same time, vs the hoped-for ~15 min.

### 5.4 Full-Model Validation Gates (through the `llm-d` router)

| Gate | Test | Result |
| --- | --- | --- |
| **G1** serve check | `/v1/completions`, "The capital of France is", 16 tokens, temperature 0 | **PASS**: HTTP 200, " Paris." (44 s, includes the final warmup) |
| **G2** chat + reasoning parser | 17 × 23 | **PASS**: `391` (37 reasoning tokens, 1.14 s) |
| | 9:40 → 13:05 trip, minutes | **PASS**: `205` (74 reasoning tokens, 1.41 s) |
| | Python one-liner for the sum of squares 1..10 | **PASS**: `sum(i**2 for i in range(1, 11))`, "Result: **385**" (119 reasoning tokens, 2.00 s) |
| | Tool call with `--grammar-backend=none` | **As expected**: HTTP 400, "Grammar-based generation ... is not supported" |
| **G3** multi-turn prefix | 5,127-token prompt, 16 output tokens | **PASS**: cold 2,875 ms → warm **850 ms**, 5,120 cached tokens (3.4×) |

**G4 / G4X: long-context prefix cache** (single request, 16 output tokens). The G4 ramp builds each longer prompt by extending the previous one. So each "first request" only prefills the tokens beyond the previous, shorter prompt, which is already cached. G4X ran first on a fresh replica, so it is a true cold start.

| Prompt tokens | First request | New tokens prefilled (rate) | Repeat with cached prefix | Cached tokens | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3,895 | 1,082 ms | 3,895 (cold) | 852 ms | 3,840 | 1.3× |
| 34,057 | 7,297 ms | ~30.2K (~4.1K tok/s) | 900 ms | 34,048 | 8.1× |
| 145,227 | 25,446 ms | ~111K (~4.4K tok/s) | 1,001 ms | 145,152 | 25.4× |
| 293,465 | 34,049 ms | ~148K (~4.4K tok/s) | 1,147 ms | 293,376 | 29.7× |
| **477,653** (G4X, 97% of `--context-length=491520`) | **222,645 ms**, true cold; includes the one-time first-request warmup (~44 s) | 477,653 (~2.7K tok/s average after warmup) | **1,554 ms** | 477,568 | **143×** |

The first ~500K-token G4 attempt returned HTTP 400 because it exceeded the context limit (the harness's token estimate overshot). G4X re-ran it at 477,653 tokens.

### 5.5 Full-Model Throughput (`sglang.benchmark.serving` through the `llm-d` router, 1 replica = 16 chips)

| Test (`in / out`) | Prompts | Req/s | Output tok/s (peak) | Total tok/s | Mean TTFT (P99) | Mean TPOT (P99) | Mean E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024/256`, C=1 | 8 | 0.34 | **87.7** (124) | 438 | 840 ms (850 ms) | **8.1 ms** (8.2 ms) | 2.91 s |
| `1024/256`, C=4 | 16 | 0.69 | **175.8** (214) | 879 | 1.44 s (1.80 s) | 17.1 ms (19.6 ms) | 5.80 s |
| `1024/256`, C=8 | 32 | 0.98 | **250.1** (358) | 1,250 | 2.62 s (3.40 s) | 21.7 ms (29.1 ms) | 8.16 s |
| `1024/256`, C=16 | 64 | 1.42 | **362.4** (552) | 1,812 | 3.71 s (4.70 s) | 29.6 ms (40.7 ms) | 11.25 s |
| `1024/256`, C=32 | 128 | 1.71 | **437.1** (801) | 2,186 | 7.21 s (8.86 s) | 45.0 ms (68.5 ms) | 18.68 s |
| `1024/256`, C=64 | 256 | 1.81 | **463.3** (944) | 2,316 | 14.82 s (25.23 s) | 73.8 ms (126.6 ms) | 33.64 s |
| `4096/256`, C=1 | 8 | 0.27 | **69.1** (123) | 1,174 | 1.58 s (1.70 s) | 8.3 ms (8.4 ms) | 3.70 s |
| `4096/256`, C=4 | 16 | 0.47 | **119.4** (215) | 2,030 | 4.29 s (4.55 s) | 16.7 ms (17.1 ms) | 8.55 s |
| `4096/256`, C=16 | 64 | 0.73 | **187.8** (535) | 3,192 | 14.26 s (15.80 s) | 29.5 ms (74.2 ms) | 21.77 s |

The `4096/256` C=64 run was cut off by the 14:42Z preemption. On the next slices (attempt #6, Ready at 15:45Z), G4X finished (Section 5.4) and the guide's `inference-perf` run (G5B) started at 15:49Z. The shared-prefix sweep, a decode-heavy sweep (G5C) and the `4096/256` C=64 rerun are queued behind it.

A quick probe run first (G5Q: C=1/16/64, same workload) agreed within ~5%: 91.6 / 347.7 / 435.7 output tok/s.

**Attempt #8 (Spot-adapted config, `Ready` 18:40Z):** the shared-prefix sweep (G5A3) and a decode-heavy sweep (G5C). One replica of 16 chips, through the `llm-d` router.

| Test (`in / out`) | Prompts | Req/s | Output tok/s (peak) | Total tok/s | Mean TTFT (P99) | Mean TPOT (P99) | Mean E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Shared prefix `3072+1024/256` (16 groups), C=16 | 64 | 1.00 | **255.2** (540) | 4,428 | 9.46 s (15.30 s) | 25.7 ms (61.1 ms) | 16.02 s |
| Shared prefix, C=32 | 128 | 1.42 | **363.7** (748) | 6,313 | 13.28 s (23.00 s) | 36.0 ms (63.8 ms) | 22.47 s |
| Shared prefix, C=64 | 256 | 1.46 | **373.3** (696) | 6,480 | 22.37 s (46.88 s) | 77.4 ms (118.4 ms) | 42.10 s |
| Decode-heavy `128/1024`, C=1 | 4 | 0.11 | **113.2** (125) | 127 | 746 ms (749 ms) | **8.1 ms** (8.2 ms) | 9.04 s |
| Decode-heavy `128/1024`, C=8 | 16 | 0.34 | **348.2** (374) | 392 | 1.55 s (1.59 s) | 21.4 ms (21.7 ms) | 23.48 s |
| Decode-heavy `128/1024`, C=32 | 64 | 0.68 | **697.5** (822) | 785 | 2.68 s (4.00 s) | 43.2 ms (44.9 ms) | 46.86 s |
| Decode-heavy `128/1024`, C=64 | 128 | 0.82 | **843.1** (990) | 949 | 4.45 s (7.40 s) | 71.4 ms (74.6 ms) | 77.51 s |
| `4096/256`, C=64 (rerun of the run cut off at 14:42Z) | 256 | 0.77 | **196.6** (665) | 3,342 | 57.49 s (110.66 s) | 91.8 ms (276.5 ms) | 80.90 s |

- With short prompts, decode-only throughput scales to **843 output tok/s at C=64** (peak 990). The 1,024-token-prompt sweep plateaued at ~463 because prefill stalls the running decodes.
- Decode step time grows with batch: 8.1 ms at bs=1, ~21 ms at 8, ~43 ms at 32, ~71 ms at 64. More concurrent tokens activate more distinct experts out of the 896, so more expert weights are read each step.

- Single-stream decode runs at **~120 tok/s (8.1 ms/token)** on the full 2.8T model.
- With short (1,024-token) random prompts, throughput above C=1 is prefill-bound. The guide caps prefill at 4,096 tokens and 8 requests per batch, and each prefill batch stalls the running decodes, so TTFT and TPOT grow with concurrency. Long prompts prefill at ~4.4–4.6K tok/s in 4,096-token chunks (G4 increments; the G5B turn-0 prefills logged ~4.57K tok/s). The average rate drops to ~2.7K tok/s over a full 0→477K-token prompt.

---

## 6. Stage 1C — Expanded Single-Host Sweeps & `llm-d` Multi-Replica Routing (12-Layer, Dummy Weights)

**Configuration:** same 12-layer block as Stage 1B (`--load-format=dummy`, `TP=8 / EP=8` per `tpu7x-standard-4t`, `pallas_tpu_mla_sparse`, `fp8_e4m3` KV, page 128, HiCache `raiden` ratio 2), widened to `--context-length=131072`, `--max-running-requests=64`, `--tpu-compile-bs 1 2 4 8 16 32 64`, `--mem-fraction-static=0.75`, with a Tier-3 compile cache on GCS FUSE. Three replicas, one per Spot node, sit in the `llm-d` InferencePool behind the EPP (`approx-prefix-cache-producer` + `prefix-cache-affinity-filter` + `token-load-scorer`) and Envoy.

> [!IMPORTANT]
> Dummy weights and 12 of 93 layers: these numbers measure kernel, scheduler, cache and router behavior on TPU v7x, not output quality. The full model's per-token cost is much higher.

| Startup metric | Value |
| --- | --- |
| HBM per TensorCore after decode compile | ~`75.3 / 97.0 GB` |
| Cold start (no compile cache) | ~11 min to finish compiling |
| Restart with Tier-3 cache | ~8 min to `Ready` |
| Pod memory limit | `900Gi` required. At `800Gi` the HiCache host pool fails: `Requesting 44.69 GB but only have 41.53 GB free`. |
| Post-startup recompiles | None. `torch.compile` frozen with `fail_on_recompile`; a few XLA programs loaded lazily from the persistent cache at serving time. |

### 6.1 Direct to One Replica (`sglang.benchmark.serving`, uncached unless noted)

| Test | Prompts | Req/s | Output tok/s (peak) | Total tok/s | Mean TTFT (P99) | Mean TPOT (P99) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024/256`, C=1 | 8 | 0.77 | **196.5** (226) | 982.6 | 200 ms (800 ms) | 4.31 ms (4.45 ms) |
| `1024/256`, C=2 | 8 | 1.43 | **365.1** (444) | 1,825.5 | 300 ms (802 ms) | 4.28 ms (4.39 ms) |
| `1024/256`, C=4 | 16 | 2.86 | **731.9** (924) | 3,659.3 | 307 ms (807 ms) | 4.24 ms (4.32 ms) |
| `1024/256`, C=8 | 32 | 5.55 | **1,420.3** (1,636) | 7,101.7 | 250 ms (435 ms) | 4.65 ms (5.27 ms) |
| `1024/256`, C=16 | 64 | 8.40 | **2,150.3** (3,394) | 10,751.7 | 436 ms (1,280 ms) | 5.70 ms (8.46 ms) |
| `1024/256`, C=32 | 128 | 13.46 | **3,447.0** (6,412) | 17,234.8 | 650 ms (1,113 ms) | 6.71 ms (8.62 ms) |
| `1024/256`, C=64 | 256 | 16.13 | **4,130.4** (9,072) | 20,651.8 | 1,211 ms (2,188 ms) | 10.72 ms (15.07 ms) |
| `4096/256`, C=1 | 8 | 0.81 | **206.1** (226) | 3,504.1 | 149 ms (177 ms) | 4.26 ms (4.33 ms) |
| `4096/256`, C=4 | 16 | 2.51 | **642.4** (853) | 10,920.9 | 347 ms (548 ms) | 4.87 ms (5.70 ms) |
| `4096/256`, C=16 | 64 | 4.94 | **1,264.8** (3,360) | 21,502.5 | 1,152 ms (2,083 ms) | 8.15 ms (12.09 ms) |
| `4096/256`, C=64 | 256 | 6.27 | **1,606.2** (8,887) | 27,304.7 | 4,250 ms (8,242 ms) | 23.25 ms (39.32 ms) |
| Shared prefix `3072+1024/256`, C=16 | 48 | 8.09 | **2,069.9** (3,437) | 35,188.5 | 531 ms (982 ms) | 5.49 ms (7.39 ms) |
| Shared prefix `3072+1024/256`, C=32 | 96 | 12.33 | **3,155.3** (6,082) | 53,639.7 | 790 ms (1,470 ms) | 6.76 ms (9.45 ms) |
| Shared prefix `3072+1024/256`, C=64 | 192 | 14.99 | **3,837.6** (9,216) | **65,238.4** | 1,351 ms (2,485 ms) | 11.04 ms (15.43 ms) |

### 6.2 Long-Context Prefix Cache (single request, 16 output tokens)

Each longer prompt extends the previous one, so the "first request" prefills only the tokens beyond the previous, cached prompt (same method as G4 in Section 5.4).

| Prompt tokens | First request | New tokens prefilled (rate) | Repeat with cached prefix | Cached tokens | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3,895 | 803 ms | 3,895 | 610 ms | 3,840 | 1.3× |
| 15,607 | 490 ms | ~11.8K (~24K tok/s) | 198 ms | 15,488 | 2.5× |
| 34,057 | 702 ms | ~18.6K (~26K tok/s) | 199 ms | 34,048 | 3.5× |
| 71,107 | 1,356 ms | ~37K (~27K tok/s) | 193 ms | 71,040 | 7.0× |
| 130,007 | 4,262 ms | ~59K (~14K tok/s) | **246 ms** | 129,920 | **17.3×** |

### 6.3 Through the `llm-d` Router (3 Replicas)

| Test | Prompts | Req/s | Output tok/s (peak) | Total tok/s | Mean TTFT (P99) | Mean TPOT (P99) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1024/256`, C=16 | 32 | 6.28 | **1,606.5** (2,358) | 8,032.3 | 499 ms (1,441 ms) | 6.12 ms (9.20 ms) |
| `1024/256`, C=64 | 128 | 20.61 | **5,276.8** (9,308) | 26,384.1 | 812 ms (2,120 ms) | 8.03 ms (13.11 ms) |
| `1024/256`, C=128 | 256 | 28.39 | **7,266.9** (13,851) | 36,334.3 | 940 ms (1,840 ms) | 12.76 ms (17.33 ms) |
| `1024/256`, C=256 | 512 | 35.69 | **9,136.9** (13,713) | 45,684.6 | 2,480 ms (7,310 ms) | 14.41 ms (25.90 ms) |
| Multi-session shared prefix (16 groups × 8, `3072` system + `1024` question / `256`), C=32 | 128 | 11.38 | **2,914.0** (4,627) | 50,524.2 | **453 ms** (1,294 ms) | 8.15 ms (12.79 ms) |
| Same workload, C=64 | 128 | 19.52 | **4,996.1** (10,598) | **86,736.9** | 1,034 ms (2,494 ms) | 8.44 ms (14.56 ms) |
| Baseline: same workload direct to **one** replica, C=32 | 128 | 8.76 | 2,241.8 (5,472) | 38,951.5 | 1,219 ms (3,393 ms) | 9.28 ms (18.54 ms) |

**Takeaways**
- One replica scales from 197 output tok/s (C=1, 4.3 ms/token) to 4,130 tok/s at C=64 (10.7 ms/token). 4096-token prompts become prefill-bound at high concurrency (4.25 s mean TTFT at C=64).
- With its prefix cached (radix cache + HiCache), a 130K-token prompt returns in 246 ms. The first request at that length took 4.26 s, and it already had the previous ~71K-token prompt cached.
- Across 3 replicas, the router reached 9,137 output tok/s (45.7K total tok/s) at C=256, 2.2× the single-replica C=64 peak.
- On the multi-session shared-prefix workload at C=32, routing across 3 replicas cut mean TTFT from 1,219 ms to 453 ms versus one replica. That comparison mixes 3× hardware with routing; Section 6.4 isolates the routing policy.

### 6.4 A/B: `llm-d` Prefix-Aware Routing vs. Plain Kubernetes Load Balancing (Same 3 Replicas)

Same multi-session shared-prefix workload (16 groups × 8 prompts, `3072` system + `1024` question / `256` out) and the same seed per pair. Every replica's cache was flushed before each run. Baseline: a plain `ClusterIP` Service (kube-proxy random spread) over the same pods.

| Concurrency | Routing | Output tok/s | Total tok/s | Mean / Median / P99 TTFT | Mean TPOT | Mean E2E |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | kube-proxy random | 3,007.3 | 52,257 | 514 / 387 / 1,715 ms | 7.95 ms | 2,540 ms |
| 32 | **llm-d EPP** | **3,476.4 (+15.6%)** | **60,409** | **475 / 355 / 1,393 ms** | **6.69 ms** | **2,179 ms (−14%)** |
| 64 | kube-proxy random | 3,936.8 | 68,343 | 1,063 / 678 / 3,128 ms | 9.61 ms | 3,513 ms |
| 64 | **llm-d EPP** | **4,436.8 (+12.7%)** | **77,025** | **985 / 568 / 2,523 ms** | **9.04 ms** | **3,291 ms (−6%)** |

With the replica count held fixed, prefix-aware routing gave **+13–16% output throughput** and **~19% lower P99 TTFT** on this workload.

### 6.5 Official `llm-d` `inference-perf` Multi-Turn Benchmark (Adapted to the 12-Layer Replicas)

Same harness as the guide (`quay.io/inference-perf/inference-perf:v0.6.0`, `conversation_replay`, streaming completions, against the `kimi-k3-1-epp` router). Session sizes are scaled down to fit the replicas' 131,072-token context: system prompts 8K–90K tokens (lognormal, mean 30K), 5–30 turns, 1.5K input / 300 output tokens per turn, 1–20 s simulated tool-call pauses. Config: [benchmarks/inference-perf-12l-adapted.yaml](benchmarks/inference-perf-12l-adapted.yaml).

| Metric (3 replicas behind the `llm-d` router) | 8 conversations, 160 requests | 16 conversations, 320 requests (guide's concurrency) |
| --- | --- | --- |
| Success / failures | **160 / 0** in 189 s | **320 / 0** in 197 s |
| Prompt length per request | mean 42,001 · median 44,058 · P90 59,409 · max 68,190 | mean 38,755 · median 38,138 · P90 58,298 · max 79,420 |
| **Prefix-cache reuse across turns** | **91.1%** (6.12M of 6.72M prompt tokens) | **90.0%** (11.16M of 12.40M prompt tokens) |
| TTFT | mean **321 ms** · median 226 ms · P90 348 ms · P99 2.67 s | mean **383 ms** · median 244 ms · P90 462 ms · P99 4.06 s |
| TPOT | mean **5.33 ms** · median 4.47 ms · P90 5.21 ms | mean **5.41 ms** · median 4.67 ms · P90 6.07 ms |
| Request latency | mean 6.98 s · median 5.29 s · P90 12.09 s | mean 6.75 s · median 5.29 s · P90 12.56 s |
| Throughput | 35,494 input tok/s · 185 output tok/s · 0.85 req/s | **62,934 input tok/s** · 483 output tok/s · 1.62 req/s |

Doubling the conversations nearly doubled throughput (1.62 vs 0.85 req/s) while median TTFT moved only from 226 to 244 ms, despite ~40K-token prompts. The router kept each conversation on the replica that already held its prefix (~90% of prompt tokens served from cache).

---

## 7. Operational Findings (Spot TPU v7x on GKE)

- **Single-host vs. slice:** a `2x2x1` node needs one free tray; a `2x2x4` slice needs 4 ICI-connected trays free at once, and GCE rolls back the whole slice if any one fails. On contended Spot capacity, single-host nodes come back within minutes, while slices landed only a handful of times overnight and were reclaimed whole (all 4 VMs within ~1 s).
- **Scale-down locks don't stop Spot reclaim:** `scale-down-disabled`, autoscaling off, and fixed pool size keep GKE from removing nodes. Every node we lost was a GCE `compute.instances.preempted` event. A fixed-size MIG recreates preempted nodes automatically.
- **Multi-host TPU pools can't be resized by hand:** GKE `setSize` returns `Multi-host TPU pool manual resize is not supported`, and the gang-mode MIG only resizes to 0. MIG `resize-requests` work for queuing a 4-VM slice.
- **Persistent compile cache matters on Spot:** with the Tier-3 cache on GCS FUSE, a preempted 12-layer replica comes back in ~8 min instead of ~11+ min. For the full model (~38 min uncached startup vs. 34–62 min slice lifetimes), the cache decides whether a slice ever serves.
- **Share one compile-cache directory across ranks:** the cache key includes the node-normalized rank and `rank_to_device`, and node normalization assigns ranks per slice. Per-pod directories (keyed by LWS worker index) missed on every new slice. Object names are content keys, so one shared directory is safe.
- **`LeaderWorkerSet` rolling updates can stall on scarce capacity:** with `maxSurge=0, maxUnavailable=1`, the highest group index updates first. If it can't schedule, lower groups keep the old template revision even after they are recreated. Check `template-revision-hash` on the pods, and use `maxUnavailable` ≥ replicas when changing templates on Spot pools.
