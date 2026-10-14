#!/usr/bin/env python3
"""End-to-End & Dedicated RPS Saturation Benchmark Suite for Jina Embeddings v2 on TPU v6e.

Validates the Opus JinaBert Megakernel v2 + Fused Device Pooling (commit d3439062):
  1. Single-HTTP-Request Multi-Prompt Batch (N = 1, 4, 8, 16) for 1KB (1,024 tok) & 2KB (2,048 tok)
  2. k6 Closed-Loop Concurrent Requests (VUS = 1, 4, 8, 16, 30s each) for 1KB & 2KB
  3. 2KB Dedicated RPS Saturation (60s per step, exact 2,048 tokens, p99 < 50 ms SLA)
  4. 1KB Dedicated RPS Saturation (60s per step, exact 1,024 tokens, p99 < 50 ms SLA)

Usage:
  # Run the full 4-part suite (auto-builds exact-token pools if missing):
  python3 run_rps_saturation_test.py --base-url http://127.0.0.1:8000 --metrics-url http://127.0.0.1:8001/metrics

  # Run ONLY the Dedicated RPS Saturation sweep (1K & 2K):
  python3 run_rps_saturation_test.py --saturation-only --base-url http://127.0.0.1:8000 --metrics-url http://127.0.0.1:8001/metrics

  # Run a quick saturation knee check around the d3439062 ceiling (480..550 RPS for 1K, 280..370 RPS for 2K):
  python3 run_rps_saturation_test.py --knee-only --base-url http://127.0.0.1:8000 --metrics-url http://127.0.0.1:8001/metrics
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

WORKDIR = os.path.dirname(os.path.abspath(__file__))
SLA_P99_MS = 50.0

COUNTERS = ("vllm:prompt_tokens_total", "vllm:request_success_total")
GAUGES = ("vllm:num_requests_running", "vllm:num_requests_waiting")


def log(msg):
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_pools(pool_dir, count=2000):
    p1 = os.path.join(pool_dir, "pool_1024tok.json")
    p2 = os.path.join(pool_dir, "pool_2048tok.json")
    if os.path.exists(p1) and os.path.exists(p2):
        return p1, p2
    log(f"Building exact 1,024-token and 2,048-token prompt pools in {pool_dir} (count={count})...")
    cmd = [sys.executable, os.path.join(WORKDIR, "build_token_pools.py"), "build", pool_dir, str(count)]
    subprocess.run(cmd, check=True)
    return p1, p2


def scrape(metrics_url):
    try:
        text = urllib.request.urlopen(metrics_url, timeout=10).read().decode()
    except Exception as e:
        log(f"[WARN] metrics scrape failed: {e}")
        return None
    out = {name: 0.0 for name in COUNTERS + GAUGES}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in out:
            out[name] += float(line.rsplit(" ", 1)[1])
    return out


def wait_drain(metrics_url, timeout_s=300):
    t0, zeros = time.time(), 0
    while time.time() - t0 < timeout_s:
        m = scrape(metrics_url)
        if m is None:
            time.sleep(2)
            return round(time.time() - t0, 1)
        busy = m["vllm:num_requests_running"] + m["vllm:num_requests_waiting"]
        zeros = zeros + 1 if busy == 0 else 0
        if zeros >= 3:
            return round(time.time() - t0, 1)
        time.sleep(0.5)
    log(f"[WARN] vLLM queue not drained after {timeout_s} s")
    return round(time.time() - t0, 1)


def percentile(sorted_vals, pct):
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def preflight(base_url, tier, n=3):
    with open(tier["pool"]) as f:
        pool = json.load(f)
    rows = []
    for text in pool[:n]:
        body = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            f"{base_url}/prompt_c2", data=body, headers={"Content-Type": "application/json"}
        )
        t0 = time.time()
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        ms = (time.time() - t0) * 1000
        rows.append(
            dict(
                chars=len(text),
                prompt_tokens=resp["usage"]["prompt_tokens"],
                dim=len(resp["data"][0]["embedding"]),
                latency_ms=round(ms, 1),
            )
        )
    ok = all(r["prompt_tokens"] == tier["tokens"] and r["dim"] == 512 for r in rows)
    return ok, rows, len(pool)


def save_state(out_dir, state):
    with open(os.path.join(out_dir, "full_eval_results.json"), "w") as f:
        json.dump(state, f, indent=2)


def run_single_http_batch_suite(base_url, metrics_url, out_dir, loaded_pools, state):
    log("=== [1/4] Single-HTTP-Request Multi-Prompt Batch (N = 1, 4, 8, 16) ===")
    for kb in [1, 2]:
        pool, expected_tok, label = loaded_pools[kb]
        for batch_size in [1, 4, 8, 16]:
            for w in range(8):
                prompts = [pool[(w * batch_size + i) % len(pool)] for i in range(batch_size)]
                body = json.dumps({"text": prompts}).encode("utf-8")
                req = urllib.request.Request(
                    f"{base_url}/prompt_c2", data=body, headers={"Content-Type": "application/json"}
                )
                urllib.request.urlopen(req, timeout=30).read()
            wait_drain(metrics_url)

            iters = 80 if batch_size <= 8 else 50
            lats = []
            errs = 0
            before = scrape(metrics_url)
            started = datetime.now(timezone.utc).isoformat(timespec="seconds")
            t_start = time.perf_counter()
            for it in range(iters):
                base_idx = (it + 10) * batch_size
                prompts = [pool[(base_idx + i) % len(pool)] for i in range(batch_size)]
                body = json.dumps({"text": prompts}).encode("utf-8")
                req = urllib.request.Request(
                    f"{base_url}/prompt_c2", data=body, headers={"Content-Type": "application/json"}
                )
                t0 = time.perf_counter()
                try:
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        resp.read()
                        if resp.status != 200:
                            errs += 1
                except Exception as e:
                    log(f"[ERR] batch={batch_size} kb={kb}: {e}")
                    errs += 1
                lats.append((time.perf_counter() - t0) * 1000.0)
            total_s = time.perf_counter() - t_start
            wait_drain(metrics_url)
            after = scrape(metrics_url)

            lats.sort()
            d_req = (after["vllm:request_success_total"] - before["vllm:request_success_total"]) if (before and after) else 0
            d_tok = (after["vllm:prompt_tokens_total"] - before["vllm:prompt_tokens_total"]) if (before and after) else 0
            row = {
                "mode": "single_http_batch",
                "started_utc": started,
                "payload_label": label,
                "payload_kb": kb,
                "tokens_per_prompt": expected_tok,
                "concurrency": batch_size,
                "iterations": iters,
                "total_prompts": iters * batch_size,
                "elapsed_s": round(total_s, 4),
                "throughput_rps": round((iters * batch_size) / total_s, 2),
                "min_ms": round(lats[0], 2),
                "p50_ms": round(percentile(lats, 50), 2),
                "avg_ms": round(sum(lats) / len(lats), 2),
                "p90_ms": round(percentile(lats, 90), 2),
                "p95_ms": round(percentile(lats, 95), 2),
                "p99_ms": round(percentile(lats, 99), 2),
                "max_ms": round(lats[-1], 2),
                "errors": errs,
                "server_requests": int(d_req),
                "server_tokens_per_prompt": round(d_tok / (iters * batch_size), 2) if d_tok else None,
            }
            state["single_http_batch"].append(row)
            save_state(out_dir, state)
            log(f"[SINGLE_HTTP_BATCH] {json.dumps(row)}")


def run_k6_concurrency_suite(base_url, metrics_url, out_dir, pools_map, state):
    log("=== [2/4] k6 Closed-Loop Concurrency (VUS = 1, 4, 8, 16, 30s each) ===")
    k6_js = os.path.join(out_dir, "k6_conc_exact_tokens.js")
    with open(k6_js, "w") as f:
        f.write("""import http from 'k6/http';
import { check } from 'k6';
import { scenario as k6scenario } from 'k6/execution';
import { Trend, Rate } from 'k6/metrics';
import { SharedArray } from 'k6/data';

const HTTP_URL = (__ENV.HTTP_URL || 'http://127.0.0.1:8000').replace(/\\/$/, '');
const VUS = parseInt(__ENV.VUS || '1', 10);
const DURATION = __ENV.DURATION || '30s';
const SCEN_NAME = __ENV.SCEN_NAME || 'prompt_c2_conc';
const TEXT_POOL = new SharedArray('text_pool', function () {
  return JSON.parse(open(__ENV.TEXT_POOL));
});

const promptLatency = new Trend('prompt_c2_latency_ms', true);
const errorRate = new Rate('error_rate');
const JSON_HEADERS = { headers: { 'Content-Type': 'application/json' } };

export const options = {
  scenarios: {
    [SCEN_NAME]: {
      executor: 'constant-vus',
      vus: VUS,
      duration: DURATION,
    },
  },
};

export default function () {
  const text = TEXT_POOL[k6scenario.iterationInTest % TEXT_POOL.length];
  const res = http.post(`${HTTP_URL}/prompt_c2`, JSON.stringify({ text }), JSON_HEADERS);
  const scen = k6scenario.name;
  promptLatency.add(res.timings.duration, { scenario: scen });
  const ok = check(res, { 'http 200': (r) => r.status === 200 });
  errorRate.add(!ok, { scenario: scen });
}
""")
    for kb in [1, 2]:
        pool_path, expected_tok, label = pools_map[kb]
        scen_name = f"prompt_c2_{kb*1024}b"
        for vus in [1, 4, 8, 16]:
            raw = os.path.join(out_dir, f"conc_{kb}k_vus{vus}.ndjson")
            summary = os.path.join(out_dir, f"conc_{kb}k_vus{vus}_summary")
            k6_log = os.path.join(out_dir, f"conc_{kb}k_vus{vus}_k6.log")
            wait_drain(metrics_url)
            before = scrape(metrics_url)
            started = datetime.now(timezone.utc).isoformat(timespec="seconds")
            cmd_k6 = (
                f"rm -f {raw} && k6 run --no-thresholds --out json={raw} "
                f"-e HTTP_URL={base_url} -e VUS={vus} -e DURATION=30s "
                f"-e SCEN_NAME={scen_name} -e TEXT_POOL={pool_path} "
                f"{k6_js} > {k6_log} 2>&1"
            )
            subprocess.run(cmd_k6, shell=True, check=False)
            wait_drain(metrics_url)
            after = scrape(metrics_url)
            cmd_ana = f"{sys.executable} {os.path.join(WORKDIR, 'analyze_k6_results.py')} {raw} --out {summary} > {summary}.log 2>&1"
            subprocess.run(cmd_ana, shell=True, check=False)
            with open(f"{summary}.json") as f:
                entry = json.load(f)[scen_name]
            met = entry["metrics"]["prompt_c2_latency_ms"]
            d_req = (after["vllm:request_success_total"] - before["vllm:request_success_total"]) if (before and after) else 0
            d_tok = (after["vllm:prompt_tokens_total"] - before["vllm:prompt_tokens_total"]) if (before and after) else 0
            row = {
                "mode": "k6_constant_vus",
                "started_utc": started,
                "payload_label": label,
                "payload_kb": kb,
                "tokens_per_prompt": expected_tok,
                "concurrency": vus,
                "duration": "30s",
                "count": met["count"],
                "throughput_rps": met["achieved_rps"],
                "min_ms": met["min"],
                "p50_ms": met["p50"],
                "avg_ms": met["avg"],
                "p90_ms": met["p90"],
                "p95_ms": met["p95"],
                "p99_ms": met["p99"],
                "max_ms": met["max"],
                "errors": entry.get("errors", 0),
                "server_requests": int(d_req),
                "server_tokens_per_request": round(d_tok / d_req, 2) if d_req else None,
            }
            state["k6_concurrency"].append(row)
            save_state(out_dir, state)
            log(f"[K6_CONCURRENCY] {json.dumps(row)}")
            if os.path.exists(raw):
                os.remove(raw)


def run_saturation_step(base_url, metrics_url, out_dir, tier, rps, duration, prefix, phase):
    raw = os.path.join(out_dir, f"{prefix}_{rps}rps.ndjson")
    summary = os.path.join(out_dir, f"{prefix}_{rps}rps_summary")
    k6_log = os.path.join(out_dir, f"{prefix}_{rps}rps_k6.log")
    wait_drain(metrics_url)
    before = scrape(metrics_url)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cmd_k6 = (
        f"cd {WORKDIR} && rm -f {raw} && "
        f"k6 run --no-thresholds --out json={raw} "
        f"-e SUITE=payload_size -e ENDPOINT=prompt_c2 -e HTTP_URL={base_url} "
        f"-e STAGE_DURATION={duration} -e PAYLOAD_RPS={rps} "
        f"-e PAYLOAD_VUS={tier['vus']} -e TEXT_POOL={tier['pool']} "
        f"{tier['script']} > {k6_log} 2>&1"
    )
    log(f"[K6 RUN] {phase} {tier['label']} {rps} RPS")
    k6_rc = subprocess.run(cmd_k6, shell=True, check=False).returncode
    drain_s = wait_drain(metrics_url)
    after = scrape(metrics_url)
    cmd_ana = f"{sys.executable} {os.path.join(WORKDIR, 'analyze_k6_results.py')} {raw} --out {summary} > {summary}.log 2>&1"
    subprocess.run(cmd_ana, shell=True, check=False)

    row = dict(
        tier=tier["label"],
        tokens=tier["tokens"],
        phase=phase,
        target_rps=rps,
        duration=duration,
        started_utc=started,
        k6_rc=k6_rc,
        drain_s=drain_s,
    )
    with open(f"{summary}.json") as f:
        entry = json.load(f)[tier["scenario"]]
    met = entry["metrics"]["prompt_c2_latency_ms"]
    row.update({k: met[k] for k in ("count", "achieved_rps", "min", "p50", "avg", "p90", "p95", "p99", "max")})
    row["errors"] = entry.get("errors", 0)
    row["status"] = "PASS" if met["p99"] < SLA_P99_MS else "SATURATED"
    if before and after:
        d_req = after["vllm:request_success_total"] - before["vllm:request_success_total"]
        d_tok = after["vllm:prompt_tokens_total"] - before["vllm:prompt_tokens_total"]
        row["server_requests"] = int(d_req)
        row["server_tokens_per_request"] = round(d_tok / d_req, 2) if d_req else None
    if os.path.exists(raw):
        os.remove(raw)
    log(f"[SATURATION_RESULT] {json.dumps(row)}")
    return row


def main():
    p = argparse.ArgumentParser(description="Run Jina Embeddings v2 TPU v6e Megakernel RPS Saturation Suite")
    p.add_argument("--base-url", default="http://127.0.0.1:8000", help="Adapter URL (default: http://127.0.0.1:8000)")
    p.add_argument("--metrics-url", default="http://127.0.0.1:8001/metrics", help="vLLM /metrics URL (default: http://127.0.0.1:8001/metrics)")
    p.add_argument("--pool-dir", default=os.path.join(WORKDIR, "pools"), help="Directory for exact-token JSON pools")
    p.add_argument("--out-dir", default=None, help="Output directory for JSON results and logs")
    p.add_argument("--duration", default="60s", help="Duration per saturation RPS step (default: 60s)")
    p.add_argument("--phase", default="all", choices=["all", "1k", "2k"], help="Which saturation payload tier(s) to run")
    p.add_argument("--saturation-only", action="store_true", help="Skip Single-HTTP Batch and k6 Concurrency; run only Dedicated Saturation")
    p.add_argument("--knee-only", action="store_true", help="Run only the high-RPS saturation knee steps (480..550 RPS for 1K, 280..370 RPS for 2K)")
    p.add_argument("--steps", default=None, help="Custom comma-separated RPS steps for a single --phase (e.g. --phase 1k --steps 500,510,520,530,540)")
    args = p.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or os.path.join(WORKDIR, f"results_{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(WORKDIR, "results"), exist_ok=True)

    p1024, p2048 = ensure_pools(args.pool_dir)
    pools_map = {
        1: (p1024, 1024, "1KB (1024 chars)"),
        2: (p2048, 2048, "2KB (2048 chars)"),
    }

    tiers = {
        "2k": dict(
            label="2K (2,048 tokens)",
            tokens=2048,
            vus=200,
            script=os.path.join(WORKDIR, "k6_2kb_saturation_test_2048tok.js"),
            pool=p2048,
            scenario="prompt_c2_2048b",
            steps=[280, 300, 320, 340, 350, 360, 370] if args.knee_only else [
                90, 95, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200,
                220, 240, 260, 280, 300, 310, 320, 330, 340, 350, 360, 370,
            ],
            max_consecutive_saturated=2,
            warmup_rps=45,
        ),
        "1k": dict(
            label="1K (1,024 tokens)",
            tokens=1024,
            vus=250,
            script=os.path.join(WORKDIR, "k6_1kb_saturation_test_1024tok.js"),
            pool=p1024,
            scenario="prompt_c2_1024b",
            steps=[480, 500, 510, 520, 530, 540, 550] if args.knee_only else [
                100, 120, 140, 160, 180, 190, 200, 220, 240, 260, 280, 300,
                320, 340, 360, 380, 400, 420, 440, 460, 480, 490, 500, 510,
                520, 530, 540, 550,
            ],
            max_consecutive_saturated=2,
            warmup_rps=50,
        ),
    }

    if args.steps:
        if args.phase == "all":
            raise SystemExit("--steps requires --phase 1k or --phase 2k")
        tiers[args.phase]["steps"] = [int(x) for x in args.steps.split(",")]

    active_tiers = ["2k", "1k"] if args.phase == "all" else [args.phase]

    meta = {
        "run_id": run_id,
        "base_url": args.base_url,
        "metrics_url": args.metrics_url,
        "sla_p99_ms": SLA_P99_MS,
        "started_utc": run_id,
        "tiers": {},
    }
    for key in active_tiers:
        t = tiers[key]
        ok, checks, pool_size = preflight(args.base_url, t)
        meta["tiers"][key] = dict(
            label=t["label"],
            tokens=t["tokens"],
            vus=t["vus"],
            steps=t["steps"],
            pool=t["pool"],
            pool_size=pool_size,
            pool_sha256=sha256(t["pool"]),
            script_sha256=sha256(t["script"]),
            preflight_ok=ok,
            preflight=checks,
        )
        log(f"[PREFLIGHT] {t['label']}: ok={ok} {checks}")
        if not ok:
            raise SystemExit(f"Preflight failed for {t['label']} (expected exact {t['tokens']} tokens and 512-dim embedding)")

    state = {
        "meta": meta,
        "single_http_batch": [],
        "k6_concurrency": [],
        "saturation_1k": [],
        "saturation_2k": [],
    }
    save_state(out_dir, state)

    if not args.saturation_only and not args.knee_only:
        loaded_pools = {}
        for kb, (path, tok, lbl) in pools_map.items():
            with open(path) as f:
                loaded_pools[kb] = (json.load(f), tok, lbl)
        run_single_http_batch_suite(args.base_url, args.metrics_url, out_dir, loaded_pools, state)
        run_k6_concurrency_suite(args.base_url, args.metrics_url, out_dir, pools_map, state)

    for key in active_tiers:
        t = tiers[key]
        log(f"=== Dedicated Saturation: {t['label']} ({args.duration} per step) ===")
        run_saturation_step(args.base_url, args.metrics_url, out_dir, t, t["warmup_rps"], "30s", f"warmup_{key}", "warmup")
        saturated_in_a_row = 0
        for rps in t["steps"]:
            row = run_saturation_step(args.base_url, args.metrics_url, out_dir, t, rps, args.duration, f"{key}_fp32", "sweep")
            state[f"saturation_{key}"].append(row)
            save_state(out_dir, state)
            saturated_in_a_row = saturated_in_a_row + 1 if row.get("status") != "PASS" else 0
            if saturated_in_a_row >= t["max_consecutive_saturated"]:
                break

    state["meta"]["finished_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_state(out_dir, state)
    log(f"ALL DONE. Results saved to {os.path.join(out_dir, 'full_eval_results.json')}")


if __name__ == "__main__":
    main()
