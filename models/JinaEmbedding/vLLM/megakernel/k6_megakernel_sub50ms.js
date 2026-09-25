// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Standalone k6 Benchmark Script for TPU v6e FP32 Megakernel (<50ms Latency SLA)
// Supports both:
//   1. Open-Loop Constant-Arrival-Rate Saturation Mode (MODE=rps, default)
//      Example (1KB @ 400 RPS):
//        k6 run --summary-trend-stats="avg,min,med,max,p(90),p(95),p(99)" \
//          -e TARGET_URL=http://10.241.2.7:8000/prompt_c2 \
//          -e PAYLOAD_KB=1 -e RATE=400 -e DURATION=12s \
//          k6_megakernel_sub50ms.js
//
//      Example (2KB @ 200 RPS):
//        k6 run --summary-trend-stats="avg,min,med,max,p(90),p(95),p(99)" \
//          -e TARGET_URL=http://10.241.2.7:8000/prompt_c2 \
//          -e PAYLOAD_KB=2 -e RATE=200 -e DURATION=12s \
//          k6_megakernel_sub50ms.js
//
//   2. Closed-Loop Fixed-Concurrency Mode (MODE=concurrency)
//      Example (1KB @ Concurrency=16):
//        k6 run --summary-trend-stats="avg,min,med,max,p(90),p(95),p(99)" \
//          -e TARGET_URL=http://10.241.2.7:8000/prompt_c2 \
//          -e MODE=concurrency -e PAYLOAD_KB=1 -e VUS=16 -e DURATION=15s \
//          k6_megakernel_sub50ms.js

import http from "k6/http";
import { check } from "k6";

const TARGET_URL = __ENV.TARGET_URL || "http://10.241.2.7:8000/prompt_c2";
const MODE = (__ENV.MODE || "rps").toLowerCase();
const PAYLOAD_KB = parseInt(__ENV.PAYLOAD_KB || "1", 10); // 1 (1KB = 1024 chars / ~256 tokens) or 2 (2KB = 2048 chars / ~512 tokens)
const RATE = parseInt(__ENV.RATE || (PAYLOAD_KB === 1 ? "400" : "200"), 10);
const VUS = parseInt(__ENV.VUS || "16", 10);
const DURATION = __ENV.DURATION || "12s";
const PRE_ALLOCATED_VUS = parseInt(__ENV.PRE_ALLOCATED_VUS || "64", 10);
const MAX_VUS = parseInt(__ENV.MAX_VUS || "128", 10);

// Exact word-tokenized payload matching run_v6e_fp32_megakernel_2k_suite.py
// 1KB -> 205 repetitions of "word " (~1,025 chars / 256 tokens)
// 2KB -> 410 repetitions of "word " (~2,050 chars / 512 tokens)
const WORDS_COUNT = PAYLOAD_KB * 205;
const payload = JSON.stringify({
  text: "word ".repeat(WORDS_COUNT),
});

const params = {
  headers: {
    "Content-Type": "application/json",
  },
  timeout: "10s",
};

export const options =
  MODE === "concurrency"
    ? {
        scenarios: {
          closed_loop_concurrency: {
            executor: "constant-vus",
            vus: VUS,
            duration: DURATION,
          },
        },
        thresholds: {
          http_req_failed: ["rate==0.00"],
          // Strict <50ms latency SLA threshold across median (p50) and tail (p99)
          http_req_duration: ["med<50", "p(95)<50", "p(99)<50"],
        },
      }
    : {
        scenarios: {
          constant_rate_sub50ms: {
            executor: "constant-arrival-rate",
            rate: RATE,
            timeUnit: "1s",
            duration: DURATION,
            preAllocatedVUs: PRE_ALLOCATED_VUS,
            maxVUs: MAX_VUS,
          },
        },
        thresholds: {
          http_req_failed: ["rate==0.00"],
          // Strict <50ms latency SLA threshold across median (p50) and tail (p99)
          http_req_duration: ["med<50", "p(95)<50", "p(99)<50"],
        },
      };

export default function () {
  const res = http.post(TARGET_URL, payload, params);
  check(res, {
    "status is 200": (r) => r.status === 200,
  });
}
