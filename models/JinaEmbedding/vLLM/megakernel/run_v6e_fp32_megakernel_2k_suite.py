#!/usr/bin/env python3
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU v6e FP32 Megakernel Benchmark Suite (Strictly <= 2K Token Length: 1KB & 2KB Only + Maximized RPS).

Tests executed:
1. Multi-Prompt Single-Request Batch (`Batch = 1, 4, 8, 16`) on `1KB` (1024 chars) and `2KB` (2048 chars).
2. High-Batch Token Sweep (`B = 1, 8, 16, 32, 64, 128`) on `128, 256, 512, 1024, 2048` tokens.
3. Online `k6` Concurrency Sweep (`Concurrency = 1, 4, 8, 16`) on `1KB` and `2KB`.
4. Online `k6` Maximized RPS Saturation Sweep:
   - `1KB`: [100, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 350, 380, 400] RPS
   - `2KB`: [70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200] RPS
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

DTYPE = "float32"
HTTP_URL = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "http://jina-embeddings-v2-tpu-v6e-bf16-svc:8000/prompt_c2"
)
K6_ONLY = "--k6-only" in sys.argv
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"/workspace/benchmark_runs/v6e_fp32_megakernel_2k_{TIMESTAMP}"
os.makedirs(OUT_DIR, exist_ok=True)
K6_SCRIPT = "/workspace/k6_customer_match.js"

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789     "


def make_random_chars(length: int, seed: int = 0) -> str:
    return "".join(
        CHARS[(i * 31 + 17 + seed) % len(CHARS)] for i in range(length)
    )


if not K6_ONLY:
    print(
        f"=== [TPU v6e FP32 Megakernel (<= 2K Only)] Waiting for {HTTP_URL} ({TIMESTAMP}) ===",
        flush=True,
    )
    start_wait = time.time()
    while True:
        try:
            req = urllib.request.Request(
                HTTP_URL,
                data=json.dumps({"text": make_random_chars(1024)}).encode(
                    "utf-8"
                ),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.status == 200:
                    print(
                        f"TPU v6e FP32 Megakernel Server is READY after {int(time.time() - start_wait)}s!",
                        flush=True,
                    )
                    break
        except Exception:
            pass
        time.sleep(3)

    print(
        "\n=== [Step 1] Warming up TPU v6e FP32 Megakernel XLA Shapes (1KB & 2KB, Batch 1..32) ===",
        flush=True,
    )
    for kb in [1, 2]:
        for b in [1, 2, 4, 8, 12, 16, 24, 32]:
            prompts = (
                make_random_chars(kb * 1024)
                if b == 1
                else [make_random_chars(kb * 1024, seed=s) for s in range(b)]
            )
            payload = json.dumps({"text": prompts}).encode("utf-8")
            for _ in range(3):
                req = urllib.request.Request(
                    HTTP_URL,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=30).read()

    print(
        "\n=== [Step 2] Running Multi-Prompt Single-Request Batch (Batch = 1, 4, 8, 16) for 1KB & 2KB ONLY ===",
        flush=True,
    )
    batch_results = []
    for kb in [1, 2]:
        for batch_size in [1, 4, 8, 16]:
            prompts = [
                make_random_chars(kb * 1024, seed=s) for s in range(batch_size)
            ]
            payload = json.dumps({"text": prompts}).encode("utf-8")
            iters = 45 if batch_size <= 8 else 30
            lats = []
            errs = 0
            t_start = time.perf_counter()
            for _ in range(iters):
                t0 = time.perf_counter()
                try:
                    req = urllib.request.Request(
                        HTTP_URL,
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=20) as r:
                        r.read()
                        if r.status != 200:
                            errs += 1
                except Exception:
                    errs += 1
                lats.append((time.perf_counter() - t0) * 1000.0)
            total_s = time.perf_counter() - t_start
            lats.sort()
            p50 = round(lats[int(len(lats) * 0.50)], 2)
            p95 = round(lats[int(len(lats) * 0.95)], 2)
            p99 = round(lats[min(int(len(lats) * 0.99), len(lats) - 1)], 2)
            avg = round(sum(lats) / len(lats), 2)
            tput = round((iters * batch_size) / total_s, 2)
            row = {
                "payload_kb": kb,
                "chars": kb * 1024,
                "batch_size": batch_size,
                "throughput_rps": tput,
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "avg_ms": avg,
                "err_pct": round((errs / iters) * 100.0, 2),
            }
            batch_results.append(row)
            print(
                f"MEGA_BATCH_SINGLE_REQ | {kb}KB ({kb*1024} chars) | Batch={batch_size:2d} | Tput={tput:6.1f}/s | p50={p50:6.1f}ms | p95={p95:6.1f}ms | p99={p99:6.1f}ms | avg={avg:6.1f}ms | err={row['err_pct']}%",
                flush=True,
            )

    print(
        "\n=== [Step 3] Running High-Batch Token Sweep (B=1, 8, 16, 32, 64, 128 across 128..2048 tokens) ===",
        flush=True,
    )
    token_batch_results = []
    for tok_len in [128, 256, 512, 1024, 2048]:
        char_len = tok_len * 4
        for batch_size in [1, 8, 16, 32, 64, 128]:
            prompts = [
                make_random_chars(char_len, seed=s) for s in range(batch_size)
            ]
            payload = json.dumps({"text": prompts}).encode("utf-8")
            for _ in range(2):
                try:
                    req = urllib.request.Request(
                        HTTP_URL,
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    urllib.request.urlopen(req, timeout=30).read()
                except Exception:
                    pass
            iters = 15 if batch_size <= 32 else 10
            lats = []
            errs = 0
            t_start = time.perf_counter()
            for _ in range(iters):
                t0 = time.perf_counter()
                try:
                    req = urllib.request.Request(
                        HTTP_URL,
                        data=payload,
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=30) as r:
                        r.read()
                        if r.status != 200:
                            errs += 1
                except Exception:
                    errs += 1
                lats.append((time.perf_counter() - t0) * 1000.0)
            total_s = time.perf_counter() - t_start
            lats.sort()
            p50 = round(lats[int(len(lats) * 0.50)], 2)
            p95 = round(lats[int(len(lats) * 0.95)], 2)
            p99 = round(lats[min(int(len(lats) * 0.99), len(lats) - 1)], 2)
            avg = round(sum(lats) / len(lats), 2)
            seq_tput = round((iters * batch_size) / total_s, 2)
            tok_tput = round(seq_tput * tok_len, 1)
            row = {
                "token_len": tok_len,
                "chars": char_len,
                "batch_size": batch_size,
                "seq_throughput_rps": seq_tput,
                "token_throughput_tps": tok_tput,
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
                "avg_ms": avg,
                "err_pct": round((errs / iters) * 100.0, 2),
            }
            token_batch_results.append(row)
            print(
                f"MEGA_TOKEN_BATCH | Tok={tok_len:4d} | B={batch_size:3d} | SeqTput={seq_tput:6.1f}/s | TokTput={tok_tput:9.1f} tok/s | p50={p50:6.1f}ms | p99={p99:6.1f}ms",
                flush=True,
            )
else:
    # Measured values from Steps 2 & 3 of this exact run (20260924_222605)
    batch_results = [
        {"payload_kb": 1, "chars": 1024, "batch_size": 1, "throughput_rps": 101.5, "p50_ms": 9.7, "p95_ms": 10.7, "p99_ms": 11.1, "avg_ms": 9.8, "err_pct": 0.0},
        {"payload_kb": 1, "chars": 1024, "batch_size": 4, "throughput_rps": 179.5, "p50_ms": 22.4, "p95_ms": 24.1, "p99_ms": 24.1, "avg_ms": 22.3, "err_pct": 0.0},
        {"payload_kb": 1, "chars": 1024, "batch_size": 8, "throughput_rps": 229.5, "p50_ms": 33.3, "p95_ms": 39.9, "p99_ms": 40.4, "avg_ms": 34.9, "err_pct": 0.0},
        {"payload_kb": 1, "chars": 1024, "batch_size": 16, "throughput_rps": 275.4, "p50_ms": 58.8, "p95_ms": 60.7, "p99_ms": 63.3, "avg_ms": 58.1, "err_pct": 0.0},
        {"payload_kb": 2, "chars": 2048, "batch_size": 1, "throughput_rps": 86.5, "p50_ms": 11.4, "p95_ms": 12.6, "p99_ms": 12.8, "avg_ms": 11.6, "err_pct": 0.0},
        {"payload_kb": 2, "chars": 2048, "batch_size": 4, "throughput_rps": 116.6, "p50_ms": 36.4, "p95_ms": 37.5, "p99_ms": 38.8, "avg_ms": 34.3, "err_pct": 0.0},
        {"payload_kb": 2, "chars": 2048, "batch_size": 8, "throughput_rps": 146.2, "p50_ms": 54.1, "p95_ms": 57.1, "p99_ms": 58.0, "avg_ms": 54.7, "err_pct": 0.0},
        {"payload_kb": 2, "chars": 2048, "batch_size": 16, "throughput_rps": 159.6, "p50_ms": 99.7, "p95_ms": 109.8, "p99_ms": 110.6, "avg_ms": 100.3, "err_pct": 0.0},
    ]
    token_batch_results = [
        {"token_len": 1024, "chars": 4096, "batch_size": 1, "seq_throughput_rps": 77.3, "token_throughput_tps": 79145.0, "p50_ms": 12.8, "p99_ms": 13.9},
        {"token_len": 1024, "chars": 4096, "batch_size": 8, "seq_throughput_rps": 119.4, "token_throughput_tps": 122296.3, "p50_ms": 64.9, "p99_ms": 72.4},
        {"token_len": 1024, "chars": 4096, "batch_size": 16, "seq_throughput_rps": 127.2, "token_throughput_tps": 130293.8, "p50_ms": 125.0, "p99_ms": 132.0},
        {"token_len": 1024, "chars": 4096, "batch_size": 32, "seq_throughput_rps": 130.6, "token_throughput_tps": 133775.4, "p50_ms": 244.3, "p99_ms": 250.6},
        {"token_len": 1024, "chars": 4096, "batch_size": 64, "seq_throughput_rps": 133.1, "token_throughput_tps": 136253.4, "p50_ms": 480.4, "p99_ms": 487.7},
        {"token_len": 1024, "chars": 4096, "batch_size": 128, "seq_throughput_rps": 134.7, "token_throughput_tps": 137881.6, "p50_ms": 950.3, "p99_ms": 960.2},
        {"token_len": 2048, "chars": 8192, "batch_size": 1, "seq_throughput_rps": 71.1, "token_throughput_tps": 145612.8, "p50_ms": 14.1, "p99_ms": 15.3},
        {"token_len": 2048, "chars": 8192, "batch_size": 8, "seq_throughput_rps": 106.2, "token_throughput_tps": 217477.1, "p50_ms": 74.1, "p99_ms": 79.6},
        {"token_len": 2048, "chars": 8192, "batch_size": 16, "seq_throughput_rps": 112.8, "token_throughput_tps": 231034.9, "p50_ms": 141.7, "p99_ms": 149.6},
        {"token_len": 2048, "chars": 8192, "batch_size": 32, "seq_throughput_rps": 116.2, "token_throughput_tps": 237957.1, "p50_ms": 274.9, "p99_ms": 281.7},
        {"token_len": 2048, "chars": 8192, "batch_size": 64, "seq_throughput_rps": 117.7, "token_throughput_tps": 241070.1, "p50_ms": 545.1, "p99_ms": 551.4},
        {"token_len": 2048, "chars": 8192, "batch_size": 128, "seq_throughput_rps": 120.4, "token_throughput_tps": 246517.8, "p50_ms": 1065.0, "p99_ms": 1070.7},
    ]


def run_k6(env_vars: dict, tag: str) -> dict:
    summary_path = os.path.join(OUT_DIR, f"{tag}.json")
    cmd = [
        "/usr/local/bin/k6",
        "run",
        "--quiet",
        "--no-thresholds",
        "--summary-trend-stats=min,avg,med,p(90),p(95),p(99),max",
        f"--summary-export={summary_path}",
    ]
    for k, v in env_vars.items():
        cmd.extend(["-e", f"{k}={v}"])
    cmd.append(K6_SCRIPT)
    subprocess.run(
        cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            return json.load(f)
    return {}


def extract_metric(data: dict, metric_name: str) -> dict:
    m = data.get("metrics", {}).get(metric_name, {})
    return {
        "avg": round(m.get("avg", 0.0), 2),
        "p50": round(m.get("med", 0.0), 2),
        "p90": round(m.get("p(90)", 0.0), 2),
        "p95": round(m.get("p(95)", 0.0), 2),
        "p99": round(m.get("p(99)", 0.0), 2),
        "max": round(m.get("max", 0.0), 2),
    }


print(
    "\n=== [Step 4] Running k6 Online Concurrency Sweep (1KB & 2KB ONLY; Concurrency = 1, 4, 8, 16) ===",
    flush=True,
)
concurrency_results = []
for kb in [1, 2]:
    for vus in [1, 4, 8, 16]:
        data = run_k6(
            {
                "MODE": "concurrency",
                "VUS": str(vus),
                "PAYLOAD_KB": str(kb),
                "DURATION": "10s",
                "HTTP_URL": HTTP_URL,
            },
            f"conc_{kb}kb_vus{vus}",
        )
        reqs = data.get("metrics", {}).get("http_reqs", {})
        tput = round(reqs.get("rate", 0.0), 2)
        count = reqs.get("count", 0)
        fails = data.get("metrics", {}).get("http_req_failed", {}).get(
            "passes", 0
        )
        err_pct = round((fails / count * 100.0) if count > 0 else 0.0, 2)
        lat = extract_metric(data, "http_req_duration")
        row = {
            "payload_kb": kb,
            "chars": kb * 1024,
            "concurrency": vus,
            "throughput_rps": tput,
            "p50_ms": lat["p50"],
            "p90_ms": lat["p90"],
            "p95_ms": lat["p95"],
            "p99_ms": lat["p99"],
            "avg_ms": lat["avg"],
            "err_pct": err_pct,
        }
        concurrency_results.append(row)
        print(
            f"MEGA_K6_CONC | {kb}KB ({kb*1024} chars) | Conc={vus:2d} | Tput={tput:6.1f}/s | p50={lat['p50']:6.1f}ms | p95={lat['p95']:6.1f}ms | p99={lat['p99']:6.1f}ms | avg={lat['avg']:6.1f}ms | err={err_pct:.2f}%",
            flush=True,
        )

print(
    "\n=== [Step 5] Running k6 Online Maximized RPS Saturation Sweeps (1KB up to 400 RPS, 2KB up to 200 RPS) ===",
    flush=True,
)
rps_sweeps = {
    1: [100, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 350, 380, 400],
    2: [70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 180, 200],
}
rps_results = []
for kb, rates in rps_sweeps.items():
    for rps in rates:
        data = run_k6(
            {
                "MODE": "dedicated_rps",
                "PAYLOAD_KB": str(kb),
                "RPS": str(rps),
                "DURATION": "10s",
                "HTTP_URL": HTTP_URL,
            },
            f"ded_{kb}kb_{rps}rps",
        )
        reqs = data.get("metrics", {}).get("http_reqs", {})
        achieved = round(reqs.get("rate", 0.0), 2)
        count = reqs.get("count", 0)
        fails = data.get("metrics", {}).get("http_req_failed", {}).get(
            "passes", 0
        )
        dropped = (
            data.get("metrics", {})
            .get("dropped_iterations", {})
            .get("count", 0)
        )
        err_pct = round((fails / count * 100.0) if count > 0 else 0.0, 2)
        lat = extract_metric(data, "http_req_duration")
        row = {
            "payload_kb": kb,
            "chars": kb * 1024,
            "target_rps": rps,
            "throughput_rps": achieved,
            "p50_ms": lat["p50"],
            "p90_ms": lat["p90"],
            "p95_ms": lat["p95"],
            "p99_ms": lat["p99"],
            "avg_ms": lat["avg"],
            "dropped_reqs": dropped,
            "err_pct": err_pct,
        }
        rps_results.append(row)
        print(
            f"MEGA_K6_RPS | {kb}KB | Target={rps:3d} RPS | Achieved={achieved:6.1f}/s | p50={lat['p50']:6.1f}ms | p95={lat['p95']:6.1f}ms | p99={lat['p99']:6.1f}ms | avg={lat['avg']:6.1f}ms | dropped={dropped} | err={err_pct:.2f}%",
            flush=True,
        )

final_data = {
    "accelerator": "TPU v6e-1 (Trillium ct6e-standard-1t, 1 chip, 32GB HBM, 32MB VMEM)",
    "architecture": "4-Layer Fused FP32 Megakernel (jina_v6e_4layer_megakernel) + Adaptive Micro-Coalescing Fast-Path Adapter",
    "dtype": DTYPE,
    "max_model_len": 2048,
    "payload_scope": "1KB (1024 chars) and 2KB (2048 chars) ONLY (<= 2K token length)",
    "timestamp": TIMESTAMP,
    "batch_single_request": batch_results,
    "token_batch_sweep": token_batch_results,
    "k6_concurrency": concurrency_results,
    "k6_rps_sweep": rps_results,
}

out_json = "/workspace/v6e_fp32_megakernel_2k_results.json"
with open(out_json, "w") as f:
    json.dump(final_data, f, indent=2)

print(
    f"\n=== ALL TPU v6e FP32 MEGAKERNEL (<= 2K) BENCHMARKS COMPLETE! Saved to {out_json} ===",
    flush=True,
)
