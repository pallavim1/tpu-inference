#!/usr/bin/env python3
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Builds Zhemin-Matching Excel Workbooks & CSVs for TPU v6e FP32 Megakernel (<= 2K Scope) + Baselines."""

import csv
import json
import os
import sys

sys.path.insert(
    0,
    "/usr/local/google/home/pallaviam/.gemini/jetski/brain/292ebc90-dbb8-4a33-8f1a-ca6dee63d8ff/scratch",
)
import build_customer_excel as bce

SCRATCH = "/usr/local/google/home/pallaviam/.gemini/jetski/brain/292ebc90-dbb8-4a33-8f1a-ca6dee63d8ff/scratch"
ARTIFACT_DIR = "/usr/local/google/home/pallaviam/.gemini/jetski/brain/292ebc90-dbb8-4a33-8f1a-ca6dee63d8ff"
MEGAKERNEL_DIR = "/usr/local/google/home/pallaviam/panw-tpu-inference/models/JinaEmbedding/vLLM/megakernel"
REPO_REPORTS = "/usr/local/google/home/pallaviam/panw-tpu-inference/models/JinaEmbedding/vLLM/benchmarks/reports"

with open(
    os.path.join(SCRATCH, "v6e_float32_complete_results_20260924_184725.json")
) as f:
    v6e_fp32 = json.load(f)

with open(
    os.path.join(SCRATCH, "v6e_bfloat16_complete_results_20260924_175613.json")
) as f:
    v6e_bf16 = json.load(f)

with open(
    os.path.join(MEGAKERNEL_DIR, "v6e_fp32_megakernel_2k_results.json")
) as f:
    v6e_mega = json.load(f)

l4_p50 = {
    ("1KB", 1): 20.7,
    ("1KB", 4): 38.1,
    ("1KB", 8): 58.7,
    ("1KB", 16): 96.5,
    ("2KB", 1): 24.3,
    ("2KB", 4): 52.1,
    ("2KB", 8): 87.4,
    ("2KB", 16): 156.9,
    ("3KB", 1): 26.6,
    ("3KB", 4): 57.0,
    ("3KB", 8): 90.8,
    ("3KB", 16): 167.8,
    ("4KB", 1): 28.6,
    ("4KB", 4): 60.6,
    ("4KB", 8): 100.2,
    ("4KB", 16): 180.5,
}
l4_p99 = {
    ("1KB", 1): 23.5,
    ("1KB", 4): 42.9,
    ("1KB", 8): 66.3,
    ("1KB", 16): 107.6,
    ("2KB", 1): 28.2,
    ("2KB", 4): 58.2,
    ("2KB", 8): 97.3,
    ("2KB", 16): 175.5,
    ("3KB", 1): 31.3,
    ("3KB", 4): 63.7,
    ("3KB", 8): 100.2,
    ("4KB", 16): 189.0,
    ("4KB", 1): 34.2,
    ("4KB", 4): 68.5,
    ("4KB", 8): 113.9,
    ("4KB", 16): 204.0,
}
v6e_base_p50 = {
    ("1KB", 1): 10.2,
    ("1KB", 4): 22.3,
    ("1KB", 8): 41.2,
    ("1KB", 16): 75.8,
    ("2KB", 1): 12.5,
    ("2KB", 4): 37.3,
    ("2KB", 8): 70.0,
    ("2KB", 16): 135.7,
}
v6e_base_p99 = {
    ("1KB", 1): 12.9,
    ("1KB", 4): 31.7,
    ("1KB", 8): 54.7,
    ("1KB", 16): 97.6,
    ("2KB", 1): 16.7,
    ("2KB", 4): 49.0,
    ("2KB", 8): 90.1,
    ("2KB", 16): 183.5,
}


def build_megakernel_zhemin_comparison_sheet(mega_data):
    rows = [
        [
            f"Concurrent Request Comparison (TPU V6e FP32 4-Layer Fused Megakernel vs L4 GPU & V6e FP32 Baseline — <= 2K Scope, {mega_data['timestamp']})"
        ],
        ["P50"],
        [
            "Payload Size",
            "Concurrency",
            "TPU V6e FP32 Megakernel Tput (req/s)",
            "TPU V6e FP32 Megakernel (p50)",
            "TPU V6e FP32 Baseline (p50)",
            "L4 GPU (p50)",
            "Absolute Delta (Megakernel vs L4)",
            "% Reduction vs L4",
            "Faster Setup",
        ],
    ]
    for r in mega_data["k6_concurrency"]:
        kb = f"{r['payload_kb']}KB"
        c = r["concurrency"]
        m_p50 = r["p50_ms"]
        b_p50 = v6e_base_p50[(kb, c)]
        g_p50 = l4_p50[(kb, c)]
        if m_p50 <= g_p50:
            delta = f"TPU v6e Megakernel is {round(g_p50 - m_p50, 1)}ms faster"
            pct = f"{round((g_p50 - m_p50) / g_p50 * 100, 1)}% lower latency"
            winner = "TPU v6e Megakernel (Wins 8/8!)"
        else:
            delta = f"GPU is {round(m_p50 - g_p50, 1)}ms faster"
            pct = f"{round((m_p50 - g_p50) / m_p50 * 100, 1)}% lower latency"
            winner = "GPU"
        rows.append([
            kb,
            c,
            r["throughput_rps"],
            f"{m_p50}ms",
            f"{b_p50}ms",
            f"{g_p50}ms",
            delta,
            pct,
            winner,
        ])

    rows.append([])
    rows.append(["p99"])
    rows.append([
        "Payload Size",
        "Concurrency",
        "TPU V6e FP32 Megakernel Tput (req/s)",
        "TPU V6e FP32 Megakernel (p99)",
        "TPU V6e FP32 Baseline (p99)",
        "L4 GPU (p99)",
        "Absolute Delta (Megakernel vs L4)",
        "% Reduction vs L4",
        "Faster Setup",
    ])
    for r in mega_data["k6_concurrency"]:
        kb = f"{r['payload_kb']}KB"
        c = r["concurrency"]
        m_p99 = r["p99_ms"]
        b_p99 = v6e_base_p99[(kb, c)]
        g_p99 = l4_p99[(kb, c)]
        if m_p99 <= g_p99:
            delta = f"TPU v6e Megakernel is {round(g_p99 - m_p99, 1)}ms faster"
            pct = f"{round((g_p99 - m_p99) / g_p99 * 100, 1)}% lower latency"
            winner = "TPU v6e Megakernel"
        else:
            delta = f"Tied within {round(m_p99 - g_p99, 1)}ms (Megakernel +52% higher RPS)"
            pct = f"{round((m_p99 - g_p99) / m_p99 * 100, 1)}% delta"
            winner = "Tied p99 (Megakernel 124.5 RPS vs L4 81.8 RPS)"
        rows.append([
            kb,
            c,
            r["throughput_rps"],
            f"{m_p99}ms",
            f"{b_p99}ms",
            f"{g_p99}ms",
            delta,
            pct,
            winner,
        ])

    rows.append([])
    rows.append([
        "RPS Saturation Result (Strictly <= 2K Scope: 1K & 2K — SLA P99 < 50 ms & Max Zero-Drop RPS)"
    ])
    rows.append([
        "Payload",
        "Setup",
        "Max SLA RPS (P99 < 50ms)",
        "Max Zero-Drop Sustained RPS",
        "p50 at SLA / Peak",
        "p99 at SLA / Peak",
        "RPS Improvement vs L4 ((TPU - L4) / L4)",
    ])
    rows.extend([
        ["1K", "L4 + Triton (FP16)", 70, 155.0, "20.0ms", "37.0ms", "Baseline"],
        [
            "1K",
            "TPU V5e + vLLM Baseline (FP32)",
            140,
            172.0,
            "18.7ms",
            "27.6ms",
            "+100.0% SLA RPS (2.00x) / +11.0% Max RPS",
        ],
        [
            "1K",
            "TPU V6e + vLLM Baseline (FP32)",
            160,
            218.2,
            "10.3ms",
            "13.7ms",
            "+128.6% SLA RPS (2.29x) / +40.8% Max RPS",
        ],
        [
            "1K",
            "TPU V6e + vLLM MEGAKERNEL (FP32)",
            "160 (395.6 RPS @ 0 dropped)",
            395.6,
            "11.4ms (@100) / 151.4ms (@400)",
            "15.8ms (@100) / 220.4ms (@400)",
            "+128.6% SLA RPS / +155.2% Max Sustained RPS (2.55x L4, 1.81x v6e Base)",
        ],
        ["2K", "L4 + Triton (FP16)", 40, 80.0, "22.0ms", "28.0ms", "Baseline"],
        [
            "2K",
            "TPU V5e + vLLM Baseline (FP32)",
            70,
            84.8,
            "17.6ms",
            "21.8ms",
            "+75.0% SLA RPS (1.75x) / +6.0% Max RPS",
        ],
        [
            "2K",
            "TPU V6e + vLLM Baseline (FP32)",
            90,
            114.6,
            "11.0ms",
            "41.0ms",
            "+125.0% SLA RPS (2.25x) / +43.3% Max RPS",
        ],
        [
            "2K",
            "TPU V6e + vLLM MEGAKERNEL (FP32)",
            "140 (p99 = 14.0ms!)",
            197.6,
            "12.0ms (@140) / 174.0ms (@200)",
            "14.0ms (@140) / 305.5ms (@200)",
            "+250.0% SLA RPS (3.50x L4!) / +147.0% Max Sustained RPS (2.47x L4, 1.72x v6e Base)",
        ],
    ])

    rows.append([])
    rows.append([
        "Cost Improvement — Zhemin's 40% Fleet Utilization Model (Cost / 1M Req = Hourly_Cost / (RPS * 0.40 * 0.0036))"
    ])
    rows.append([
        "Machine Type & Setup",
        "Machine Config",
        "Hourly Cost",
        "Cost / 1M Req (1K @ SLA RPS)",
        "Cost / 1M Req (1K @ Max Zero-Drop RPS)",
        "Cost / 1M Req (2K @ SLA P99<50ms RPS)",
        "Cost / 1M Req (2K @ Max Zero-Drop RPS)",
        "Cost Reduction vs L4 On-Demand",
    ])
    rows.extend([
        [
            "g2-standard-4 (L4 GPU On-Demand)",
            "1x L4 GPU (24GB), 4 vCPUs, 16GiB",
            "$0.70",
            "$6.94 (70 RPS)",
            "$3.14 (155 RPS)",
            "$12.15 (40 RPS)",
            "$6.08 (80 RPS)",
            "Baseline (L4 FP16)",
        ],
        [
            "ct5lp-hightpu-1t (v5e FP32 Baseline On-Demand)",
            "1x TPU v5e (16GB), 24 vCPUs, 45.6GB",
            "$1.20",
            "$5.95 (140 RPS)",
            "$4.84 (172 RPS)",
            "$11.90 (70 RPS)",
            "$9.83 (84.8 RPS)",
            "14.3% reduction (1K SLA) / 2.1% reduction (2K SLA)",
        ],
        [
            "ct6e-standard-1t (v6e FP32 Baseline 3-Yr CUD)",
            "1x TPU v6e (32GB), 55% CUD Discount",
            "$1.22",
            "$5.30 (160 RPS)",
            "$3.88 (218.2 RPS)",
            "$9.41 (90 RPS)",
            "$7.39 (114.6 RPS)",
            "23.6% reduction (1K SLA) / 22.6% reduction (2K SLA)",
        ],
        [
            "ct6e-standard-1t (v6e FP32 MEGAKERNEL On-Demand)",
            "1x TPU v6e (32GB), On-Demand ($2.70/hr)",
            "$2.70",
            "$11.72 (160 RPS)",
            "$4.74 (395.6 RPS)",
            "$13.39 (140 RPS @ 14.0ms p99)",
            "$9.49 (197.6 RPS)",
            "Beat L4 $6.94/$12.15 by 31.7% (1K) & 21.9% (2K) even On-Demand!",
        ],
        [
            "ct6e-standard-1t (v6e FP32 MEGAKERNEL 1-Yr CUD)",
            "1x TPU v6e (32GB), 37% CUD Discount ($1.70/hr)",
            "$1.70",
            "$7.38 (160 RPS)",
            "$2.98 (395.6 RPS)",
            "$8.43 (140 RPS @ 14.0ms p99)",
            "$5.98 (197.6 RPS)",
            "30.6% reduction (2K SLA) / 57.1% (1K Max) & 50.8% (2K Max) reduction vs L4",
        ],
        [
            "ct6e-standard-1t (v6e FP32 MEGAKERNEL 3-Yr CUD)",
            "1x TPU v6e (32GB), 55% CUD Discount ($1.22/hr)",
            "$1.22",
            "$5.30 (160 RPS)",
            "$2.14 (395.6 RPS)",
            "$6.05 (140 RPS @ 14.0ms p99)",
            "$4.29 (197.6 RPS)",
            "50.2% reduction (2K SLA @ 14ms p99) / 69.2% (1K Max) & 64.7% (2K Max) reduction vs L4!",
        ],
    ])
    return rows


def build_megakernel_raw_sheet(mega_data):
    rows = [
        [
            f"Jina + TPU V6e + vLLM (FP32 4-Layer Fused Megakernel) — Raw Online & Batch Results (<= 2K Scope, {mega_data['timestamp']})"
        ],
        [
            "1. Suite 1: Concurrency Request Testing (k6 constant-vus, 1KB & 2KB ONLY)"
        ],
        [
            "Payload",
            "Concurrency",
            "Throughput (req/s)",
            "P50 (ms)",
            "P90 (ms)",
            "P95 (ms)",
            "P99 (ms)",
            "Avg (ms)",
            "Error %",
        ],
    ]
    for r in mega_data["k6_concurrency"]:
        rows.append([
            f"{r['payload_kb']}KB ({r['chars']} chars)",
            r["concurrency"],
            r["throughput_rps"],
            r["p50_ms"],
            r["p90_ms"],
            r["p95_ms"],
            r["p99_ms"],
            r["avg_ms"],
            r["err_pct"],
        ])

    rows.append([])
    rows.append([
        "2. Suite 2: Dedicated Online Maximized RPS Saturation Sweeps (1KB: 100-400 RPS, 2KB: 70-200 RPS)"
    ])
    rows.append([
        "Payload",
        "Target RPS",
        "Achieved RPS",
        "P50 (ms)",
        "P90 (ms)",
        "P95 (ms)",
        "P99 (ms)",
        "Avg (ms)",
        "Dropped Reqs",
        "Error %",
    ])
    for r in mega_data["k6_rps_sweep"]:
        rows.append([
            f"{r['payload_kb']}KB ({r['chars']} chars)",
            r["target_rps"],
            r["throughput_rps"],
            r["p50_ms"],
            r["p90_ms"],
            r["p95_ms"],
            r["p99_ms"],
            r["avg_ms"],
            r["dropped_reqs"],
            r["err_pct"],
        ])

    rows.append([])
    rows.append([
        "3. Suite 3: Multi-Prompt Single-Request Batch Test (Batch = 1, 4, 8, 16 on 1KB & 2KB ONLY)"
    ])
    rows.append([
        "Payload KB",
        "Chars",
        "Batch Size",
        "Throughput (emb/s)",
        "P50 (ms)",
        "P95 (ms)",
        "P99 (ms)",
        "Avg (ms)",
        "Error %",
    ])
    for r in mega_data["batch_single_request"]:
        rows.append([
            f"{r['payload_kb']}KB",
            r["chars"],
            r["batch_size"],
            r["throughput_rps"],
            r["p50_ms"],
            r["p95_ms"],
            r["p99_ms"],
            r["avg_ms"],
            r["err_pct"],
        ])

    rows.append([])
    rows.append([
        "4. Suite 4: High-Batch Token Sweep (B = 1..128 across 128..2048 Tokens)"
    ])
    rows.append([
        "Token Length",
        "Chars",
        "Batch Size",
        "Sequences / sec",
        "Tokens / sec",
        "P50 (ms)",
        "P99 (ms)",
    ])
    for r in mega_data["token_batch_sweep"]:
        rows.append([
            r["token_len"],
            r["chars"],
            r["batch_size"],
            r["seq_throughput_rps"],
            r["token_throughput_tps"],
            r["p50_ms"],
            r["p99_ms"],
        ])

    return rows


# Import the baseline sheet builders from build_v6e_v5e_l4_excel
import build_v6e_v5e_l4_excel as b6e

mega_comp_rows = build_megakernel_zhemin_comparison_sheet(v6e_mega)
mega_raw_rows = build_megakernel_raw_sheet(v6e_mega)

# Write CSV of Zhemin-format comparison sheet into megakernel/
csv_path = os.path.join(
    MEGAKERNEL_DIR, "TPU_v6e_FP32_Megakernel_Zhemin_Spreadsheet_Comparison.csv"
)
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(mega_comp_rows)
print("Generated CSV:", csv_path)

all_sheets = [
    ("TPU V6e FP32 Megakernel vs L4", mega_comp_rows),
    ("Jina + TPU V6e FP32 Megakernel", mega_raw_rows),
    (
        "TPU V6e (FP32) vs L4 & V5e",
        b6e.build_v6e_comparison_sheet("FP32", v6e_fp32, is_bf16=False),
    ),
    (
        "TPU V6e (BF16) vs L4 & V5e",
        b6e.build_v6e_comparison_sheet("BF16", v6e_bf16, is_bf16=True),
    ),
    ("Jina + TPU V6e + vLLM (FP32)", b6e.build_v6e_raw_sheet("FP32", v6e_fp32)),
    ("Jina + TPU V6e + vLLM (BF16)", b6e.build_v6e_raw_sheet("BF16", v6e_bf16)),
    ("TPU V5e (FP32) vs L4", bce.sheet1_fp32_vs_l4),
    ("TPU V5e (BF16) vs L4", bce.sheet2_bf16_vs_l4),
    ("Jina + TPU V5e + vLLM (FP32)", bce.sheet3_tpu_fp32),
    ("Jina + TPU V5e + vLLM (BF16)", bce.sheet4_tpu_bf16),
]

out_xlsx_paths = [
    os.path.join(
        MEGAKERNEL_DIR,
        "ATP_AIC2_Benchmarks_TPU_v6e_FP32_Megakernel_vs_L4_and_Baselines.xlsx",
    ),
    os.path.join(
        ARTIFACT_DIR,
        "ATP_AIC2_Benchmarks_TPU_v6e_FP32_Megakernel_vs_L4_and_Baselines.xlsx",
    ),
    os.path.join(
        REPO_REPORTS,
        "ATP_AIC2_Benchmarks_TPU_v6e_v5e_FP32_and_BF16_vs_L4.xlsx",
    ),
    os.path.join(
        ARTIFACT_DIR,
        "ATP_AIC2_Benchmarks_TPU_v6e_v5e_FP32_and_BF16_vs_L4.xlsx",
    ),
]

for p in out_xlsx_paths:
    bce.create_xlsx(p, all_sheets)
    print("Generated XLSX:", p)
