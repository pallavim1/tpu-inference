#!/usr/bin/env python3
"""
Automated One-Click Saturation Benchmark Runner for jina-embeddings-v2 on Cloud TPU.

Direct match to BENCHMARKING_GUIDE.md (Section 7: Automated One-Click Saturation Benchmark Runner).
Validated for jina-v2-alibi-kernel (JAX ALiBi Flash-Attention Kernel + vLLM Pooling Runner).

Target SLA: P99 < 50.0 ms & Error Rate <= 1.0%.
Endpoint: http://jina-embedding-service:8000/prompt_c2 (or http://localhost:8000/prompt_c2)
Duration: 60 seconds per stage (STAGE_DURATION=60s).
Payloads: Palo Alto Networks (PANW) ATP specs + Full 8192 Context (1K, 2K, 5K, 7K, 8K bytes).
"""

import argparse
import concurrent.futures
import csv
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

# Optional psutil for system metrics if available
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Palo Alto Networks (PANW) ATP exact payload byte sizes + Full 8192 Context payload
DEFAULT_PAYLOADS = {
    "1K": 1024,
    "2K": 2048,
    "5K": 5120,
    "7K": 7168,
    "8K": 8192,  # Full 8192-token context length
}

# Stages matching BENCHMARKING_GUIDE.md baseline + full saturation sweeps
BASELINE_RPS = [1, 5, 7, 10, 20, 30, 40]
SATURATION_RPS = [50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]

DEFAULT_ENDPOINT = "http://jina-embedding-service:8000/prompt_c2"
DEFAULT_MODEL = "jinaai/jina-embeddings-v2-small-en"
DEFAULT_DURATION = 60.0  # Exactly 60s per stage per BENCHMARKING_GUIDE.md


def generate_text_payload(target_bytes: int) -> str:
    """Generates deterministic string matching target byte length exactly."""
    pattern = "Google Cloud TPU v6e vLLM JAX ALiBi Flash-Attention embedding kernel payload data token sequence. "
    pattern_bytes = pattern.encode("utf-8")
    repetitions = (target_bytes // len(pattern_bytes)) + 1
    full_bytes = (pattern * repetitions).encode("utf-8")[:target_bytes]
    return full_bytes.decode("utf-8", errors="ignore")


def build_request_body(endpoint: str, text: str, model_id: str) -> bytes:
    """Formats payload JSON bytes depending on endpoint path."""
    if "prompt_c2" in endpoint:
        payload = {"text": text, "model": model_id}
    else:
        # Standard OpenAI /v1/embeddings endpoint format
        payload = {"input": text, "model": model_id}
    return json.dumps(payload).encode("utf-8")


def send_single_request(url: str, body_bytes: bytes, timeout_sec: float) -> tuple:
    """Sends a single HTTP POST embedding request and measures latency in ms."""
    req = urllib.request.Request(
        url,
        data=body_bytes,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            _ = resp.read()
            t1 = time.perf_counter()
            if resp.status == 200:
                return (True, (t1 - t0) * 1000.0, None)
            else:
                return (False, (t1 - t0) * 1000.0, f"HTTP_{resp.status}")
    except urllib.error.HTTPError as e:
        t1 = time.perf_counter()
        return (False, (t1 - t0) * 1000.0, f"HTTP_{e.code}")
    except Exception as e:
        t1 = time.perf_counter()
        return (False, (t1 - t0) * 1000.0, str(e))


def calc_percentile(sorted_data: list, pct: float) -> float:
    """Calculates percentile on sorted data list."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


def run_single_stage(
    url: str,
    payload_label: str,
    target_bytes: int,
    target_rps: float,
    duration_sec: float,
    model_id: str,
    timeout_sec: float,
    max_workers: int
) -> dict:
    """Runs a single 60s RPS stage for a specific payload size using thread pool executor."""
    text_content = generate_text_payload(target_bytes)
    body_bytes = build_request_body(url, text_content, model_id)

    num_requests = int(target_rps * duration_sec)
    interval = 1.0 / target_rps

    cpu_before = psutil.cpu_percent(interval=None) if HAS_PSUTIL else 0.0
    mem_before = (psutil.virtual_memory().used / (1024**3)) if HAS_PSUTIL else 0.0

    latencies = []
    errors = []

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(num_requests):
            t_next = t_start + (i * interval)
            now = time.perf_counter()
            if t_next > now:
                time.sleep(t_next - now)

            fut = executor.submit(send_single_request, url, body_bytes, timeout_sec)
            futures.append(fut)

        for fut in concurrent.futures.as_completed(futures):
            success, lat_ms, err = fut.result()
            if success:
                latencies.append(lat_ms)
            else:
                errors.append(err)

    t_end = time.perf_counter()
    total_duration = t_end - t_start
    achieved_rps = len(latencies) / total_duration if total_duration > 0 else 0.0

    cpu_after = psutil.cpu_percent(interval=None) if HAS_PSUTIL else 0.0
    mem_after = (psutil.virtual_memory().used / (1024**3)) if HAS_PSUTIL else 0.0

    if not latencies:
        return {
            "payload": payload_label,
            "target_bytes": target_bytes,
            "target_rps": target_rps,
            "samples": 0,
            "achieved_rps": 0.0,
            "min_ms": None,
            "p50_ms": None,
            "avg_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "errors": len(errors),
            "error_rate_pct": 100.0,
            "sla_pass": False
        }

    sorted_lat = sorted(latencies)
    min_ms = sorted_lat[0]
    max_ms = sorted_lat[-1]
    avg_ms = sum(sorted_lat) / len(sorted_lat)
    p50 = calc_percentile(sorted_lat, 50.0)
    p90 = calc_percentile(sorted_lat, 90.0)
    p95 = calc_percentile(sorted_lat, 95.0)
    p99 = calc_percentile(sorted_lat, 99.0)

    total_attempts = len(latencies) + len(errors)
    error_rate = (len(errors) / total_attempts) * 100.0 if total_attempts > 0 else 0.0
    sla_pass = (p99 <= 50.0) and (error_rate <= 1.0)

    return {
        "payload": payload_label,
        "target_bytes": target_bytes,
        "target_rps": target_rps,
        "samples": len(latencies),
        "achieved_rps": round(achieved_rps, 2),
        "min_ms": round(min_ms, 2),
        "p50_ms": round(p50, 2),
        "avg_ms": round(avg_ms, 2),
        "p90_ms": round(p90, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "max_ms": round(max_ms, 2),
        "errors": len(errors),
        "error_rate_pct": round(error_rate, 2),
        "sla_pass": sla_pass,
        "cpu_usage_avg": round((cpu_before + cpu_after) / 2.0, 2),
        "mem_gb_avg": round((mem_before + mem_after) / 2.0, 2)
    }


def run_benchmark_suite(args):
    """Orchestrates multi-stage benchmark sweep across all payloads."""
    print("=" * 90)
    print("      JINA EMBEDDINGS V2 ON CLOUD TPU - AUTOMATED SATURATION BENCHMARK RUNNER")
    print("      (BENCHMARKING_GUIDE.md Section 7 Standard Configuration)")
    print(f" Target Endpoint: {args.url}")
    print(f" Model ID:        {args.model}")
    print(f" Stage Duration:  {args.duration} seconds per stage")
    print(f" SLA Target:      P99 < 50.0 ms & Error Rate <= 1.0%")
    print(f" Max Context Size: 8192 Bytes (8K Full Context)")
    print("=" * 90)

    rps_stages = BASELINE_RPS.copy()
    if args.phase in ["saturation", "all"]:
        for rps in SATURATION_RPS:
            if rps not in rps_stages:
                rps_stages.append(rps)

    all_stage_results = []
    saturation_boundaries = {}

    for payload_label, target_bytes in DEFAULT_PAYLOADS.items():
        print(f"\n>>> STARTING 60s SWEEP FOR PAYLOAD {payload_label} ({target_bytes} Bytes) <<<")
        print(f"{'Target RPS':<10} | {'Achieved RPS':<12} | {'p50 (ms)':<9} | {'p90 (ms)':<9} | {'p99 (ms)':<9} | {'Max (ms)':<9} | {'Errors':<6} | {'SLA Status'}")
        print("-" * 90)

        last_passed_rps = 0
        saturated = False

        for target_rps in rps_stages:
            if saturated and not args.force_full_sweep:
                print(f" Skipping target {target_rps} RPS (payload {payload_label} already saturated)")
                continue

            max_workers = min(int(target_rps * 4), 256)
            res = run_single_stage(
                url=args.url,
                payload_label=payload_label,
                target_bytes=target_bytes,
                target_rps=target_rps,
                duration_sec=args.duration,
                model_id=args.model,
                timeout_sec=args.timeout,
                max_workers=max_workers
            )

            all_stage_results.append(res)

            status_str = "PASS (P99 < 50ms)" if res["sla_pass"] else "SATURATED / FAIL"
            print(f"{res['target_rps']:<10} | {res['achieved_rps']:<12} | {str(res['p50_ms']):<9} | {str(res['p90_ms']):<9} | {str(res['p99_ms']):<9} | {str(res['max_ms']):<9} | {res['errors']:<6} | {status_str}")

            if res["sla_pass"]:
                last_passed_rps = target_rps
            else:
                if not saturated:
                    saturated = True
                    saturation_boundaries[payload_label] = {
                        "max_sla_pass_rps": last_passed_rps,
                        "saturation_boundary_rps": target_rps,
                        "p99_at_saturation": res["p99_ms"]
                    }

            time.sleep(args.cooldown)

        if payload_label not in saturation_boundaries:
            saturation_boundaries[payload_label] = {
                "max_sla_pass_rps": last_passed_rps,
                "saturation_boundary_rps": "Not Reached (Higher RPS needed)",
                "p99_at_saturation": None
            }

    # Print Summary Table
    print("\n" + "=" * 90)
    print("                        SATURATION & HEADROOM SUMMARY REFERENCE")
    print("=" * 90)
    print(f"{'Payload':<10} | {'Max SLA Pass RPS (P99 < 50ms)':<32} | {'Saturation Boundary RPS':<28}")
    print("-" * 90)
    for payload_label, boundary_info in saturation_boundaries.items():
        print(f"{payload_label:<10} | {boundary_info['max_sla_pass_rps']:<32} | {str(boundary_info['saturation_boundary_rps']):<28}")
    print("=" * 90)

    # Export Results
    os.makedirs(args.out_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(args.out_dir, f"jina_v2_alibi_saturation_{timestamp_str}.json")
    csv_path = os.path.join(args.out_dir, f"jina_v2_alibi_saturation_{timestamp_str}.csv")

    output_data = {
        "timestamp": timestamp_str,
        "endpoint": args.url,
        "model": args.model,
        "duration_per_stage_sec": args.duration,
        "saturation_boundaries": saturation_boundaries,
        "stages": all_stage_results
    }

    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    # Standard library CSV Export
    if all_stage_results:
        keys = list(all_stage_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_stage_results)

    print(f"\n[INFO] Benchmark results saved successfully:")
    print(f"  - JSON Report: {json_path}")
    print(f"  - CSV Summary: {csv_path}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Automated Saturation Benchmark Runner for jina-embeddings-v2 on Cloud TPU (Matching BENCHMARKING_GUIDE.md with 8K Context).")
    parser.add_argument("--url", type=str, default=DEFAULT_ENDPOINT, help="Target API endpoint URL.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model identifier.")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="Duration in seconds for each RPS stage (default: 60s per BENCHMARKING_GUIDE.md).")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Cooldown time in seconds between stages.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--phase", type=str, choices=["baseline", "saturation", "all"], default="all", help="Sweep phase to execute.")
    parser.add_argument("--force-full-sweep", action="store_true", help="Do not stop payload sweep early upon reaching saturation.")
    parser.add_argument("--out-dir", type=str, default="./benchmark_results", help="Directory to store output files.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark_suite(args)
