#!/usr/bin/env python3
import asyncio
import time
import json
import numpy as np
import aiohttp
import os
import psutil

SERVER_URL = os.environ.get("VLLM_SERVER_URL", "http://localhost:8000/v1/embeddings")
MODEL_ID = "jinaai/jina-embeddings-v2-small-en"

# Exact PANW payloads by byte size
PAYLOAD_SIZES = {
    "1K": 1024,
    "2K": 2048,
    "5K": 5120,
    "7K": 7168
}

TARGET_RPS_LIST = [1, 5, 7, 10, 20, 30, 40]
TEST_DURATION_SEC = 12

def generate_payload_text(target_bytes):
    base_unit = "Google Cloud TPU v6e benchmark payload data string token sequence. "
    repetitions = (target_bytes // len(base_unit)) + 1
    return (base_unit * repetitions)[:target_bytes]

async def send_single_request(session, payload, results, errors):
    t0 = time.perf_counter()
    try:
        async with session.post(SERVER_URL, json=payload, timeout=120) as resp:
            t1 = time.perf_counter()
            if resp.status == 200:
                await resp.read()
                results.append((t1 - t0) * 1000.0)
            else:
                errors.append(resp.status)
    except Exception as e:
        errors.append(str(e))

async def run_rps_test(payload_label, target_bytes, target_rps):
    text_content = generate_payload_text(target_bytes)
    payload = {
        "model": MODEL_ID,
        "input": text_content
    }
    
    results = []
    errors = []
    
    interval = 1.0 / target_rps
    num_requests = int(target_rps * TEST_DURATION_SEC)
    
    print(f"\n--- Testing Scenario {payload_label} ({target_bytes}B) @ Target {target_rps} RPS ({num_requests} reqs) ---", flush=True)
    
    cpu_before = psutil.cpu_percent(interval=None)
    mem_before = psutil.virtual_memory().used / (1024**3)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        t_start = time.perf_counter()
        
        for i in range(num_requests):
            t_next = t_start + (i * interval)
            now = time.perf_counter()
            if t_next > now:
                await asyncio.sleep(t_next - now)
            
            task = asyncio.create_task(send_single_request(session, payload, results, errors))
            tasks.append(task)
            
        await asyncio.gather(*tasks, return_exceptions=True)
        t_end = time.perf_counter()

    total_duration = t_end - t_start
    achieved_rps = len(results) / total_duration if total_duration > 0 else 0
    
    cpu_after = psutil.cpu_percent(interval=None)
    mem_after = psutil.virtual_memory().used / (1024**3)

    if not results:
        print(f"FAILED: 0 successful requests. Errors: {len(errors)}", flush=True)
        return None

    res_summary = {
        "scenario": f"prompt_c2_{target_bytes}b_rps{target_rps}",
        "payload": payload_label,
        "payload_bytes": target_bytes,
        "target_rps": target_rps,
        "samples": len(results),
        "achieved_rps": round(achieved_rps, 2),
        "min_ms": round(float(np.min(results)), 2),
        "p50_ms": round(float(np.percentile(results, 50)), 2),
        "avg_ms": round(float(np.mean(results)), 2),
        "p90_ms": round(float(np.percentile(results, 90)), 2),
        "p95_ms": round(float(np.percentile(results, 95)), 2),
        "p99_ms": round(float(np.percentile(results, 99)), 2),
        "max_ms": round(float(np.max(results)), 2),
        "errors": len(errors),
        "cpu_usage_avg": round((cpu_before + cpu_after)/2.0, 2),
        "mem_gb_avg": round((mem_before + mem_after)/2.0, 2)
    }
    
    print(f"Result: Achieved RPS={res_summary['achieved_rps']}, p50={res_summary['p50_ms']}ms, p90={res_summary['p90_ms']}ms, p99={res_summary['p99_ms']}ms, errors={res_summary['errors']}", flush=True)
    return res_summary

async def main():
    print("=== STARTING PANW-EQUIVALENT TPU v6e RPS BENCHMARK ===", flush=True)
    all_summaries = []
    
    for payload_label, target_bytes in PAYLOAD_SIZES.items():
        for rps in TARGET_RPS_LIST:
            res = await run_rps_test(payload_label, target_bytes, rps)
            if res:
                all_summaries.append(res)
            await asyncio.sleep(0.5)
            
    with open("tpu_panw_rps_results.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
        
    print("\n=== ALL RPS BENCHMARKS COMPLETED SUCCESSFULLY ===", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
