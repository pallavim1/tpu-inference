/**
 * ray_serve_test Load Test for Ray Serve — HTTP and gRPC interfaces
 *
 * Endpoints:
 *   HTTP  POST /prompt_c2   (port 8000)  — prompt-c2, text 1 K–7 K bytes
 *   HTTP  POST /mcp_c2      (port 8000)  — mcp-c2,    text 100–400 bytes
 *   gRPC  atpdetection.Detection/Verdict   (port 9000)  — same text ranges
 *         Routed via gRPC metadata  application: prompt-c2 | mcp-c2
 *
 * ── Suites (SUITE env var) ─────────────────────────────────────────────────────
 *
 *   (default)      RPS sweep: 5 / 20 / 50 RPS, random payload in range, 60 s each
 *   payload_size   Fixed payload sizes, fixed RPS, 2 min each
 *                    prompt-c2: 1 K, 2 K, 5 K, 7 K
 *                    mcp-c2:    100, 200, 400 B
 *   rps_payload    Full matrix: every (RPS × payload size) combo, 2 min each
 *                    prompt-c2: 12 RPS levels × 4 sizes = 48 stages  (~1 h 46 min)
 *                    mcp-c2:    12 RPS levels × 3 sizes = 36 stages  (~1 h 19 min)
 *                    both:                              = 84 stages  (~3 h 5 min)
 *                    Use -e ENDPOINT=prompt_c2|mcp_c2 to run one endpoint only.
 *
 * ── Quick start ────────────────────────────────────────────────────────────────
 *
 *   # Default RPS sweep (HTTP)
 *   ray_serve_test run k6_ray_serve_test.js
 *9
 *   # Fixed payload-size suite (gRPC)
 *   ray_serve_test run -e SUITE=payload_size -e INTERFACE=grpc k6_ray_serve_test.js
 *
 *   # Full RPS × payload matrix, prompt-c2 only (HTTP, ~1 h 40 min)
 *   ray_serve_test run -e SUITE=rps_payload -e ENDPOINT=prompt_c2 k6_ray_serve_test.js
 *
 *   # Full RPS × payload matrix, both endpoints (gRPC, ~2 h 54 min)
 *   ray_serve_test run -e SUITE=rps_payload -e INTERFACE=grpc k6_ray_serve_test.js
 *
 *   # Ad-hoc: single endpoint, full control
 *   ray_serve_test run -e ENDPOINT=prompt_c2 -e RPS=30 -e VUS=60 -e DURATION=2m \
 *           -e MIN_SIZE=3072 -e MAX_SIZE=3072 k6_ray_serve_test.js
 *
 * ── Environment variables ──────────────────────────────────────────────────────
 *   INTERFACE    http | grpc                  (default: http)
 *   SUITE        payload_size | rps_payload   (default: RPS sweep)
 *   ENDPOINT     prompt_c2 | mcp_c2 | both   (default: both)
 *   HTTP_URL        Base HTTP URL                (default: http://localhost:8000)
 *   GRPC_HOST       gRPC host:port               (default: localhost:9000)
 *   GRPC_PROMPT_APP Ray Serve app name for prompt-c2 gRPC routing (default: prompt-c2)
 *   GRPC_MCP_APP    Ray Serve app name for mcp-c2 gRPC routing    (default: mcp-c2)
 *
 *   Ad-hoc only (activated when ENDPOINT + RPS are both set):
 *   RPS          Requests/s                   (default: 10)
 *   VUS          Pre-allocated VUs            (default: 20)
 *   DURATION     Stage duration               (default: 60s)
 *   MIN_SIZE     Min text bytes               (endpoint default)
 *   MAX_SIZE     Max text bytes               (endpoint default)
 *
 *   payload_size / rps_payload suites:
 *   PAYLOAD_RPS  RPS for payload_size suite   (default: 20)
 *   PAYLOAD_VUS  VUs for payload_size suite   (default: 40)
 */

import http from 'k6/http';
import grpc from 'k6/net/grpc';
import { check, sleep } from 'k6';
import { scenario as k6scenario } from 'k6/execution';
import { Trend, Rate, Counter } from 'k6/metrics';
import { textSummary } from 'https://jslib.k6.io/k6-summary/0.0.1/index.js';

// ── Config ────────────────────────────────────────────────────────────────────

const INTERFACE  = (__ENV.INTERFACE  || 'http').toLowerCase();   // 'http' | 'grpc'
const AD_HOC_EP  = __ENV.ENDPOINT || null;                        // null → built-in suite
const _HTTP_URL_RAW = __ENV.HTTP_URL || 'http://localhost:8000';
const HTTP_URL   = (_HTTP_URL_RAW.startsWith('http') ? _HTTP_URL_RAW : `http://${_HTTP_URL_RAW}`).replace(/\/$/, '');
const GRPC_HOST  = __ENV.GRPC_HOST  || 'ray-head-svc.atp-ray-serve.svc.cluster.local:9000';

const AD_HOC_RPS      = parseInt(__ENV.RPS      || '10');
const AD_HOC_VUS      = parseInt(__ENV.VUS      || '20');
const AD_HOC_DURATION = __ENV.DURATION          || '60s';

// Default payload size ranges per endpoint
const PROMPT_MIN = 1024;  // 1 K
const PROMPT_MAX = 7168;  // 7 K
const MCP_MIN    = 100;
const MCP_MAX    = 400;

// gRPC application names used for Ray Serve routing metadata.
// Ray Serve routes gRPC requests to the correct app via the "application" metadata key.
// Override with GRPC_PROMPT_APP / GRPC_MCP_APP if your deployment uses different names.
const GRPC_APP = {
  prompt_c2: __ENV.GRPC_PROMPT_APP || 'prompt-c2',
  mcp_c2:    __ENV.GRPC_MCP_APP    || 'mcp-c2',
};

// ── Pod metrics config ────────────────────────────────────────────────────────
// Set ENABLE_POD_METRICS=1 to collect CPU, memory, GPU utilization and GPU
// memory for all Ray worker nodes via the Ray dashboard API.
//
// All metrics are fetched from the Ray dashboard (port 8265) via a single call
// to /nodes?view=summary — no worker pod IPs, no K8s Metrics API RBAC, no SA
// token required.
//
// In-cluster usage:
//   ray_serve_test run -e ENABLE_POD_METRICS=1 \
//           -e SUITE=payload_size \
//           -e ENDPOINT=prompt_c2 \
//           -e RAY_DASHBOARD_URL=http://ray-head-svc.atp-ray-serve.svc.cluster.local:8265 \
//           k6_ray_serve_test.js
//
// Override defaults:
//   RAY_DASHBOARD_URL  Ray dashboard URL  (default: http://ray-head-svc.atp-ray-serve.svc.cluster.local:8265)
//   METRICS_INTERVAL   seconds between polls  (default: 15)
const ENABLE_POD_METRICS  = __ENV.ENABLE_POD_METRICS === '1';
const RAY_DASHBOARD_URL   = (__ENV.RAY_DASHBOARD_URL || 'http://ray-head-svc.atp-ray-serve.svc.cluster.local:8265').replace(/\/$/, '');
const METRICS_INTERVAL    = parseInt(__ENV.METRICS_INTERVAL || '15');  // seconds

// Time table populated by buildScenarios(); used in pollPodMetrics to tag
// pod metrics samples with the load scenario that was active at poll time.
// Format: [{name, startSec, endSec}], sorted by startSec.
let SCENARIO_TIME_TABLE = [];

// Wall-clock ms recorded on the first pollPodMetrics iteration.
// Used instead of k6instance.currentTestRunDuration which can be unreliable.
let _pollStartMs = 0;

// ── gRPC client (init context) ────────────────────────────────────────────────
// client.load() must run in the init context (module top-level).
// The first arg is an array of proto import paths relative to this script.

const grpcClient = new grpc.Client();
if (INTERFACE === 'grpc') {
  grpcClient.load(['proto'], 'detection.proto');
}

// Per-VU flag — each VU connects once and reuses the connection.
let grpcConnected = false;

// ── Custom metrics ─────────────────────────────────────────────────────────────

const promptLatency  = new Trend('prompt_c2_latency_ms', true);
const mcpLatency     = new Trend('mcp_c2_latency_ms',    true);
const errorRate      = new Rate('error_rate');
const promptCounter  = new Counter('prompt_c2_requests');
const mcpCounter     = new Counter('mcp_c2_requests');

// ── Pod resource metrics (optional — enabled by ENABLE_POD_METRICS=1) ─────────
const podCpuMillicores = new Trend('pod_cpu_millicores',  false);
const podMemoryMB      = new Trend('pod_memory_mb',       false);
const podGpuUtilPct    = new Trend('pod_gpu_util_pct',    false);
const podGpuMemUsedMB  = new Trend('pod_gpu_mem_used_mb', false);

// ── Helpers ────────────────────────────────────────────────────────────────────

const CHARSET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:\n';

function generateText(size) {
  let s = '';
  for (let i = 0; i < size; i++) {
    s += CHARSET[Math.floor(Math.random() * CHARSET.length)];
  }
  return s;
}

function randomText(minBytes, maxBytes) {
  const size = Math.floor(Math.random() * (maxBytes - minBytes + 1)) + minBytes;
  return generateText(size);
}



/** Parse a ray_serve_test duration string to seconds. Examples: '60s'→60, '2m'→120, '1h'→3600 */
function parseDurationToSec(d) {
  if (!d) return 0;
  const s = String(d);
  if (s.endsWith('h')) return parseInt(s) * 3600;
  if (s.endsWith('m')) return parseInt(s) * 60;
  return parseInt(s);  // bare number or 'Xs'
}

/** Return the load scenario name that is active at elapsedSec.
 *  Exact match first; falls back to the nearest scenario within METRICS_INTERVAL
 *  to absorb polling jitter and K8s API call latency drift.
 *  Returns null only if the poll is more than METRICS_INTERVAL away from every stage. */
function findActiveScenario(elapsedSec) {
  // Exact window match
  for (const entry of SCENARIO_TIME_TABLE) {
    if (elapsedSec >= entry.startSec && elapsedSec < entry.endSec) {
      return entry.name;
    }
  }
  // Nearest-scenario fallback: attribute the poll to the closest stage window
  // up to METRICS_INTERVAL seconds away (handles gap periods and timing drift).
  let best = null;
  let bestDist = METRICS_INTERVAL;
  for (const entry of SCENARIO_TIME_TABLE) {
    const dist = elapsedSec < entry.startSec ? entry.startSec - elapsedSec
               : elapsedSec >= entry.endSec  ? elapsedSec - entry.endSec
               : 0;
    if (dist < bestDist) { bestDist = dist; best = entry.name; }
  }
  return best;
}

// ── HTTP test functions ────────────────────────────────────────────────────────

const JSON_HEADERS = { headers: { 'Content-Type': 'application/json' } };

export function httpPromptC2() {
  const minSize = parseInt(__ENV.MIN_SIZE || PROMPT_MIN);
  const maxSize = parseInt(__ENV.MAX_SIZE || PROMPT_MAX);
  const res = http.post(
    `${HTTP_URL}/prompt_c2`,
    JSON.stringify({ text: randomText(minSize, maxSize) }),
    JSON_HEADERS,
  );
  const scen = k6scenario.name;
  promptLatency.add(res.timings.duration, { scenario: scen });
  promptCounter.add(1);
  const ok = check(res, { 'prompt_c2 http 200': (r) => r.status === 200 });
  errorRate.add(!ok, { scenario: scen });
}

export function httpMcpC2() {
  const minSize = parseInt(__ENV.MIN_SIZE || MCP_MIN);
  const maxSize = parseInt(__ENV.MAX_SIZE || MCP_MAX);
  const res = http.post(
    `${HTTP_URL}/mcp_c2`,
    JSON.stringify({ text: randomText(minSize, maxSize) }),
    JSON_HEADERS,
  );
  const scen = k6scenario.name;
  mcpLatency.add(res.timings.duration, { scenario: scen });
  mcpCounter.add(1);
  const ok = check(res, { 'mcp_c2 http 200': (r) => r.status === 200 });
  errorRate.add(!ok, { scenario: scen });
}

// ── gRPC test functions ────────────────────────────────────────────────────────

function ensureGrpcConnected() {
  if (!grpcConnected) {
    grpcClient.connect(GRPC_HOST, { plaintext: true });
    grpcConnected = true;
  }
}

function grpcVerdict(endpoint, minBytes, maxBytes) {
  ensureGrpcConnected();
  const text = randomText(minBytes, maxBytes);
  const start = Date.now();
  const res = grpcClient.invoke(
    'atpdetection.Detection/Verdict',
    { text },
    { metadata: { application: GRPC_APP[endpoint] } },
  );
  const latencyMs = Date.now() - start;
  const ok = check(res, {
    [`${endpoint} grpc OK`]: (r) => r && r.status === grpc.StatusOK,
  });
  errorRate.add(!ok, { scenario: k6scenario.name });
  return latencyMs;
}

export function grpcPromptC2() {
  const minSize = parseInt(__ENV.MIN_SIZE || PROMPT_MIN);
  const maxSize = parseInt(__ENV.MAX_SIZE || PROMPT_MAX);
  const scen = k6scenario.name;
  promptLatency.add(grpcVerdict('prompt_c2', minSize, maxSize), { scenario: scen });
  promptCounter.add(1);
}

export function grpcMcpC2() {
  const minSize = parseInt(__ENV.MIN_SIZE || MCP_MIN);
  const maxSize = parseInt(__ENV.MAX_SIZE || MCP_MAX);
  const scen = k6scenario.name;
  mcpLatency.add(grpcVerdict('mcp_c2', minSize, maxSize), { scenario: scen });
  mcpCounter.add(1);
}

// ── Pod metrics polling function ───────────────────────────────────────────────
//
// Runs as the sole VU in the pod_metrics scenario (added automatically when
// ENABLE_POD_METRICS=1). Each poll:
//   1. K8s Metrics API (direct, with SA bearer token) → CPU & memory per container
//   2. Ray Prometheus endpoint (port 8080) → GPU utilisation & memory per GPU
//
// HTTP calls to the metrics endpoints are tagged {type:"pod_metrics"} so they
// do not pollute the main latency metrics or thresholds.

export function pollPodMetrics() {
  if (_pollStartMs === 0) _pollStartMs = Date.now();
  const elapsedSec = (Date.now() - _pollStartMs) / 1000;

  const activeScenario = findActiveScenario(elapsedSec);
  console.log(`[pod_metrics] t=${elapsedSec.toFixed(1)}s → scenario=${activeScenario || 'NONE'}`);

  // Scenario tag — only set when a load stage is active, not between stages.
  const baseTag = activeScenario ? { scenario: activeScenario } : {};

  // ── Ray Dashboard API ─────────────────────────────────────────────────────────
  // /nodes?view=summary returns CPU, memory and GPU stats for ALL cluster nodes
  // via the head service — no worker pod IPs or K8s RBAC tokens required.
  //
  // Response fields used per node:
  //   cpu          — CPU usage % (e.g. 5.4 = 5.4%)
  //   cpus[0]      — total logical CPU count
  //   mem[0]       — memory used in bytes
  //   gpus[i].utilizationGpu  — GPU utilization %
  //   gpus[i].memoryUsed      — GPU memory used in MB
  //   gpus[i].index           — GPU index on the node
  const dashRes = http.get(`${RAY_DASHBOARD_URL}/nodes?view=summary`, {
    tags: { type: 'pod_metrics' },
  });

  if (dashRes.status !== 200) {
    console.warn(`[pod_metrics] Dashboard HTTP ${dashRes.status} — check RAY_DASHBOARD_URL (${RAY_DASHBOARD_URL})`);
    sleep(METRICS_INTERVAL);
    return;
  }

  try {
    const data  = JSON.parse(dashRes.body);
    const nodes = (data.data || {}).summary || [];

    for (const node of nodes) {
      const nodeTag = { node: node.ip, hostname: node.hostname, ...baseTag };

      // CPU: usage% × num_cpus × 10 → millicores
      if (typeof node.cpu === 'number' && Array.isArray(node.cpus) && node.cpus[0]) {
        podCpuMillicores.add(node.cpu * node.cpus[0] * 10, nodeTag);
      }

      // Memory: used bytes → MB
      // mem[0]=total, mem[1]=available, mem[2]=used%, mem[3]=used bytes
      if (Array.isArray(node.mem) && node.mem[3]) {
        podMemoryMB.add(node.mem[3] / (1024 * 1024), nodeTag);
      }

      // GPU: only present on worker nodes (head has gpus: [])
      for (const gpu of (node.gpus || [])) {
        const gpuTag = { node: node.ip, gpu: String(gpu.index), ...baseTag };
        podGpuUtilPct.add(gpu.utilizationGpu, gpuTag);   // already in %
        podGpuMemUsedMB.add(gpu.memoryUsed, gpuTag);     // already in MB
      }
    }
  } catch (e) {
    console.warn(`[pod_metrics] Dashboard parse error: ${e}`);
  }

  sleep(METRICS_INTERVAL);
}

// ── Ad-hoc default export ──────────────────────────────────────────────────────

export default function () {
  const ep = AD_HOC_EP || 'both';
  if (INTERFACE === 'grpc') {
    if (ep === 'mcp_c2')    { grpcMcpC2();    return; }
    if (ep === 'prompt_c2') { grpcPromptC2(); return; }
    grpcPromptC2(); grpcMcpC2();
  } else {
    if (ep === 'mcp_c2')    { httpMcpC2();    return; }
    if (ep === 'prompt_c2') { httpPromptC2(); return; }
    httpPromptC2(); httpMcpC2();
  }
}

// ── Scenario builder ──────────────────────────────────────────────────────────

function makeScenario(exec, rate, vus, startOffsetSec) {
  return {
    executor: 'constant-arrival-rate',
    exec,
    rate,
    timeUnit: '1s',
    duration: BUILTIN_STAGE_DURATION,
    preAllocatedVUs: vus,
    maxVUs: vus * 3,
    startTime: `${startOffsetSec}s`,
  };
}

// 2 stages (5 → 20 RPS) per endpoint. Stage duration configurable via STAGE_DURATION env var.
// ENDPOINT=prompt_c2 -e STAGE_DURATION=60s → ~2.5 min; default 15s → ~1 min.
const BUILTIN_STAGE_DURATION = __ENV.STAGE_DURATION || '15s';
const BUILTIN_STAGE_S        = parseDurationToSec(BUILTIN_STAGE_DURATION);
const OFFSET = BUILTIN_STAGE_S + 15; // stage duration + 15 s cooldown gap

function builtinScenarios() {
  // Choose exec names based on INTERFACE
  const promptExec = INTERFACE === 'grpc' ? 'grpcPromptC2' : 'httpPromptC2';
  const mcpExec    = INTERFACE === 'grpc' ? 'grpcMcpC2'    : 'httpMcpC2';

  const ep = AD_HOC_EP || 'both';
  const s  = {};

  if (ep === 'prompt_c2' || ep === 'both') {
    s.prompt_c2_5rps  = makeScenario(promptExec,  5,  10, 0 * OFFSET);
    s.prompt_c2_20rps = makeScenario(promptExec, 20,  40, 1 * OFFSET);
  }

  if (ep === 'mcp_c2' || ep === 'both') {
    const base = (ep === 'both') ? 2 : 0;  // offset past prompt stages if running both
    s.mcp_c2_5rps  = makeScenario(mcpExec,  5,  10, (base + 0) * OFFSET);
    s.mcp_c2_20rps = makeScenario(mcpExec, 20,  40, (base + 1) * OFFSET);
  }

  return s;
}

function adHocScenario() {
  return {
    adhoc: {
      executor: 'constant-arrival-rate',
      exec: 'default',
      rate: AD_HOC_RPS,
      timeUnit: '1s',
      duration: AD_HOC_DURATION,
      preAllocatedVUs: AD_HOC_VUS,
      maxVUs: AD_HOC_VUS * 3,
    },
  };
}

// Ad-hoc mode: ENDPOINT is explicitly set to prompt_c2 or mcp_c2 AND RPS/VUS/DURATION are provided.
const IS_AD_HOC = AD_HOC_EP && AD_HOC_EP !== 'both' && __ENV.RPS;

// Shared size / RPS constants (used by payload_size and rps_payload suites)
const RPS_LEVELS   = [1, 5, 7, 10, 20, 30, 40];
const PROMPT_SIZES = [1024, 2048, 5120, 7168];   // bytes
const MCP_SIZES    = [100, 200, 400];             // bytes

// ── Fixed-payload-size suite ──────────────────────────────────────────────────
//
// Sweeps through fixed payload sizes at a steady RPS (default 20).
// Activated by:  ray_serve_test run -e SUITE=payload_size  k6_ray_serve_test.js
//
// prompt-c2: 1 K, 2 K, 5 K, 7 K  — 2 min each → 8 min total
// mcp-c2:   100, 200, 400 bytes   — 2 min each → 6 min total (starts after prompt stages)
//
// Override throughput:  -e PAYLOAD_RPS=30 -e PAYLOAD_VUS=60
//
const IS_PAYLOAD_SUITE = __ENV.SUITE === 'payload_size';
const PAYLOAD_RPS      = parseInt(__ENV.PAYLOAD_RPS || '20');
const PAYLOAD_VUS      = parseInt(__ENV.PAYLOAD_VUS || '40');
const PAYLOAD_DURATION = __ENV.STAGE_DURATION || '30s';
const PAYLOAD_STAGE_S  = parseDurationToSec(PAYLOAD_DURATION) + 5;

function makeFixedPayloadScenario(exec, rps, vus, sizeBytes, startSec) {
  return {
    executor: 'constant-arrival-rate',
    exec,
    rate: rps,
    timeUnit: '1s',
    duration: PAYLOAD_DURATION,
    preAllocatedVUs: vus,
    maxVUs: vus * 3,
    startTime: `${startSec}s`,
    env: { MIN_SIZE: String(sizeBytes), MAX_SIZE: String(sizeBytes) },
  };
}

function payloadSizeScenarios() {
  const promptExec = INTERFACE === 'grpc' ? 'grpcPromptC2' : 'httpPromptC2';
  const s = {};
  s['prompt_c2_1024b'] = {
    executor: 'constant-arrival-rate',
    exec: promptExec,
    rate: PAYLOAD_RPS,
    timeUnit: '1s',
    duration: PAYLOAD_DURATION,
    preAllocatedVUs: Math.max(PAYLOAD_VUS, PAYLOAD_RPS * 2),
    maxVUs: Math.max(PAYLOAD_VUS * 4, PAYLOAD_RPS * 8),
    startTime: '0s',
    env: { MIN_SIZE: '1024', MAX_SIZE: '1024' },
  };
  return s;
}

// ── RPS × payload-size matrix suite ──────────────────────────────────────────
//
// Sweeps every combination of (RPS level × payload size) sequentially.
// 1 min per combination, 10 s gap between stages.
//
// prompt-c2:  9 RPS × 4 sizes = 36 stages  (~42 min)
// mcp-c2:     9 RPS × 3 sizes = 27 stages  (~32 min)
// both:                         63 stages  (~1 h 14 min)
//
// Order: for each payload size → sweep all RPS levels.
// This keeps one variable fixed per "block", making the result easy to graph.

const IS_RPS_PAYLOAD = __ENV.SUITE === 'rps_payload';
const MATRIX_DURATION = __ENV.STAGE_DURATION || '20s';
const MATRIX_STAGE_S = parseDurationToSec(MATRIX_DURATION) + 5;   // active + 5 s cooldown

// Optional payload-size filter for rps_payload: run only one size across all RPS levels.
// Example: -e SIZE_FILTER=5120  → runs prompt_c2_5120b_rps10 … rps100 only (~9 min)
const SIZE_FILTER = __ENV.SIZE_FILTER ? parseInt(__ENV.SIZE_FILTER) : null;

function rpsPayloadScenarios() {
  const promptExec = INTERFACE === 'grpc' ? 'grpcPromptC2' : 'httpPromptC2';
  const mcpExec    = INTERFACE === 'grpc' ? 'grpcMcpC2'    : 'httpMcpC2';
  const ep = AD_HOC_EP || 'both';
  const s  = {};
  let offset = 0;

  function addStages(exec, sizes) {
    const filtered = SIZE_FILTER ? sizes.filter(sz => sz === SIZE_FILTER) : sizes;
    for (const sz of filtered) {
      for (const rps of RPS_LEVELS) {
        const preVUs = Math.max(rps * 2, 20);
        const maxVUs = Math.max(rps * 8, 50);
        const epName = exec.includes('Prompt') || exec.includes('prompt') ? 'prompt_c2' : 'mcp_c2';
        s[`${epName}_${sz}b_rps${rps}`] = {
          executor: 'constant-arrival-rate',
          exec,
          rate: rps,
          timeUnit: '1s',
          duration: MATRIX_DURATION,
          preAllocatedVUs: preVUs,
          maxVUs: maxVUs,
          startTime: `${offset * MATRIX_STAGE_S}s`,
          env: { MIN_SIZE: String(sz), MAX_SIZE: String(sz) },
        };
        offset++;
      }
    }
  }

  if (ep === 'prompt_c2' || ep === 'both') addStages(promptExec, PROMPT_SIZES);
  if (ep === 'mcp_c2'    || ep === 'both') addStages(mcpExec,    MCP_SIZES);

  return s;
}

// ── Total duration calculator + scenario builder ──────────────────────────────

/** Estimate the total test wall-clock duration in seconds (used to size the
 *  pod_metrics constant-vus scenario so it runs for the full test). */
function computeTotalDurationSec() {
  if (IS_RPS_PAYLOAD) {
    const ep = AD_HOC_EP || 'both';
    const promptCount = SIZE_FILTER
      ? (PROMPT_SIZES.includes(SIZE_FILTER) ? 1 : 0)
      : PROMPT_SIZES.length;
    const mcpCount = SIZE_FILTER
      ? (MCP_SIZES.includes(SIZE_FILTER) ? 1 : 0)
      : MCP_SIZES.length;
    let stages = 0;
    if (ep === 'prompt_c2' || ep === 'both') stages += promptCount * RPS_LEVELS.length;
    if (ep === 'mcp_c2'    || ep === 'both') stages += mcpCount    * RPS_LEVELS.length;
    return stages * MATRIX_STAGE_S + 60;
  }
  if (IS_PAYLOAD_SUITE) {
    return (PROMPT_SIZES.length + MCP_SIZES.length) * PAYLOAD_STAGE_S + 60;
  }
  if (IS_AD_HOC) {
    const d = AD_HOC_DURATION.endsWith('m')
      ? parseInt(AD_HOC_DURATION) * 60
      : parseInt(AD_HOC_DURATION);
    return d + 30;
  }
  // Default RPS sweep: 2 stages per endpoint group, each OFFSET apart, last stage BUILTIN_STAGE_S long.
  // Total stages = numStageGroups × 2; last stage starts at (totalStages-1) × OFFSET.
  const ep = AD_HOC_EP || 'both';
  const numStageGroups = (ep === 'both') ? 2 : 1;
  const totalStages = numStageGroups * 2;
  return (totalStages - 1) * OFFSET + BUILTIN_STAGE_S + 15;  // last stage start + duration + buffer
}

/** Returns the final scenarios object, injecting a pod_metrics poller when
 *  ENABLE_POD_METRICS=1. Also populates SCENARIO_TIME_TABLE so pollPodMetrics
 *  can tag samples with whichever load scenario is active at poll time. */
function buildScenarios() {
  const base = IS_RPS_PAYLOAD  ? rpsPayloadScenarios()
             : IS_PAYLOAD_SUITE ? payloadSizeScenarios()
             : IS_AD_HOC        ? adHocScenario()
             :                    builtinScenarios();

  // Build lookup table: [{name, startSec, endSec}] sorted by start time.
  // Used by findActiveScenario() in pollPodMetrics to tag resource samples.
  SCENARIO_TIME_TABLE = Object.entries(base).map(([name, s]) => {
    const startSec = parseDurationToSec(s.startTime || '0s');
    return { name, startSec, endSec: startSec + parseDurationToSec(s.duration) };
  }).sort((a, b) => a.startSec - b.startSec);

  if (!ENABLE_POD_METRICS) return base;

  const totalSec = computeTotalDurationSec();
  base.pod_metrics = {
    executor: 'constant-vus',
    exec:     'pollPodMetrics',
    vus:      1,
    duration: `${totalSec}s`,
    startTime: '0s',
  };
  return base;
}

// ── handleSummary — stdout + local result files ────────────────────────────────
//
// Every run automatically writes two files into results/:
//   results/<suite>_<timestamp>.json   full ray_serve_test metric dump (all values)
//   results/<suite>_<timestamp>.txt    human-readable summary + throughput table
//
// To also capture per-request raw data, add:  --out json=results/raw_<timestamp>.json

export function handleSummary(data) {
  const promptCount = (data.metrics['prompt_c2_requests'] || { values: { count: 0 } }).values.count;
  const mcpCount    = (data.metrics['mcp_c2_requests']    || { values: { count: 0 } }).values.count;

  // Active seconds = sum of stage durations actually executed per endpoint.
  let promptActiveSec, mcpActiveSec;
  const MATRIX_ACTIVE_S = MATRIX_STAGE_S - 10;  // strip the 10 s cooldown gap
  if (IS_RPS_PAYLOAD) {
    promptActiveSec = PROMPT_SIZES.length * RPS_LEVELS.length * MATRIX_ACTIVE_S;
    mcpActiveSec    = MCP_SIZES.length    * RPS_LEVELS.length * MATRIX_ACTIVE_S;
  } else if (IS_PAYLOAD_SUITE) {
    promptActiveSec = PROMPT_SIZES.length * 120;
    mcpActiveSec    = MCP_SIZES.length    * 120;
  } else if (IS_AD_HOC) {
    promptActiveSec = parseDurationToSec(AD_HOC_DURATION);
    mcpActiveSec    = parseDurationToSec(AD_HOC_DURATION);
  } else {
    // Default builtin: 2 stages per endpoint
    promptActiveSec = 2 * BUILTIN_STAGE_S;
    mcpActiveSec    = 2 * BUILTIN_STAGE_S;
  }

  function rpsLine(label, count, activeSec) {
    if (count === 0) return '';
    const rps = (count / activeSec).toFixed(2);
    return `  ${label.padEnd(12)} ${String(count).padStart(7)} reqs in ${activeSec}s active  →  ${rps} RPS\n`;
  }

  const throughput =
    '\n── Throughput ──────────────────────────────────────────────────\n' +
    rpsLine('prompt_c2', promptCount, promptActiveSec) +
    rpsLine('mcp_c2',    mcpCount,    mcpActiveSec) +
    '────────────────────────────────────────────────────────────────\n';

  // ── Per-scenario breakdown table ─────────────────────────────────────────────
  const stageDurationSec = IS_RPS_PAYLOAD || IS_PAYLOAD_SUITE ? 120
                         : IS_AD_HOC ? parseDurationToSec(AD_HOC_DURATION)
                         : BUILTIN_STAGE_S;

  function buildScenarioTable() {
    const allKeys = Object.keys(data.metrics).filter(k => k.includes('latency'));
    console.log(`[debug] latency metric keys (${allKeys.length}): ${allKeys.slice(0, 10).join(' | ')}`);

    const rows = [];
    for (const [key, metric] of Object.entries(data.metrics)) {
      const match = key.match(/^(prompt_c2|mcp_c2)_latency_ms\{scenario:(.+)\}$/)
                 || key.match(/^(prompt_c2|mcp_c2)_latency_ms\{scenario="(.+)"\}$/);
      if (!match) continue;
      const scenName = match[2];
      const v = metric.values;
      const count = v['count'] || 0;

      // actual achieved RPS = count / stage duration
      const achievedRps = (count / stageDurationSec).toFixed(1);

      // error count from error_rate sub-metric
      const errKey = `error_rate{scenario:${scenName}}`;
      const errMetric = data.metrics[errKey];
      const errFails = errMetric ? (errMetric.values['fails'] || 0) : 0;

      // parse payload size from scenario name e.g. mcp_c2_100b_rps10 → 100
      const sizeMatch = scenName.match(/_(\d+)b_rps/);
      const payloadBytes = sizeMatch ? parseInt(sizeMatch[1]) : null;
      const payloadLabel = payloadBytes !== null
        ? (payloadBytes >= 1024 ? `${payloadBytes / 1024}K` : `${payloadBytes}B`)
        : '-';

      rows.push({
        scenario: scenName,
        payload: payloadLabel,
        p50:  Math.round(v['med']   || 0),
        p90:  Math.round(v['p(90)'] || 0),
        p95:  Math.round(v['p(95)'] || 0),
        p99:  Math.round(v['p(99)'] || 0),
        count,
        achievedRps,
        errors: errFails,
      });
    }
    if (rows.length === 0) return '';
    rows.sort((a, b) => a.scenario.localeCompare(b.scenario));
    const H = '─';
    let t = '\n── Per-scenario results ' + H.repeat(65) + '\n';
    t += 'Scenario                              payload    p50      p90      p95      p99    count     RPS  errors\n';
    t += H.repeat(103) + '\n';
    for (const r of rows) {
      t += `${r.scenario.padEnd(38)}${r.payload.padStart(7)}  ${String(r.p50).padStart(5)}  ${String(r.p90).padStart(7)}  ${String(r.p95).padStart(7)}  ${String(r.p99).padStart(7)}  ${String(r.count).padStart(7)}  ${String(r.achievedRps).padStart(7)}  ${String(r.errors).padStart(6)}\n`;
    }
    t += H.repeat(103) + '\n';
    return t;
  }

  const scenarioTable = buildScenarioTable();

  // ── Pod resource metrics table ────────────────────────────────────────────────
  function buildPodMetricsTable() {
    const cpuM  = data.metrics['pod_cpu_millicores'];
    const memM  = data.metrics['pod_memory_mb'];
    const gpuU  = data.metrics['pod_gpu_util_pct'];
    const gpuMM = data.metrics['pod_gpu_mem_used_mb'];
    if (!cpuM && !memM && !gpuU && !gpuMM) return '';

    const H = '─';
    let t = '\n── Pod Resource Metrics ' + H.repeat(55) + '\n';

    if (cpuM) {
      const v = cpuM.values;
      t += `  CPU (millicores)  avg=${Math.round(v.avg||0)}  p50=${Math.round(v.med||0)}  p90=${Math.round(v['p(90)']||0)}  max=${Math.round(v.max||0)}  samples=${v.count||0}\n`;
    }
    if (memM) {
      const v = memM.values;
      t += `  Memory (MB)       avg=${(v.avg||0).toFixed(0)}  p50=${(v.med||0).toFixed(0)}  p90=${(v['p(90)']||0).toFixed(0)}  max=${(v.max||0).toFixed(0)}  samples=${v.count||0}\n`;
    }
    if (gpuU) {
      const v = gpuU.values;
      t += `  GPU util (%)      avg=${(v.avg||0).toFixed(1)}  p50=${(v.med||0).toFixed(1)}  p90=${(v['p(90)']||0).toFixed(1)}  max=${(v.max||0).toFixed(1)}  samples=${v.count||0}\n`;
    }
    if (gpuMM) {
      const v = gpuMM.values;
      t += `  GPU mem (MB)      avg=${(v.avg||0).toFixed(0)}  p50=${(v.med||0).toFixed(0)}  p90=${(v['p(90)']||0).toFixed(0)}  max=${(v.max||0).toFixed(0)}  samples=${v.count||0}\n`;
    }
    t += H.repeat(78) + '\n';
    return t;
  }

  const podMetricsTable = ENABLE_POD_METRICS ? buildPodMetricsTable() : '';

  const plainSummary = textSummary(data, { indent: ' ', enableColors: false }) + throughput + scenarioTable + podMetricsTable;
  const colorSummary = textSummary(data, { indent: ' ', enableColors: true  }) + throughput + scenarioTable + podMetricsTable;

  // Build a timestamped filename suffix: YYYYMMDD_HHMMSS
  const now   = new Date();
  const ts    = now.toISOString().replace(/[-:T]/g, '').slice(0, 15);  // 20260427_143022
  const suite = IS_RPS_PAYLOAD  ? 'rps_payload'
              : IS_PAYLOAD_SUITE ? 'payload_size'
              : IS_AD_HOC        ? `adhoc_${AD_HOC_EP}_rps${AD_HOC_RPS}`
              :                    'default';
  const iface = INTERFACE;
  const ep    = (AD_HOC_EP && AD_HOC_EP !== 'both') ? AD_HOC_EP : 'both';
  const base  = `results/${suite}_${iface}_${ep}_${ts}`;

  return {
    stdout:            colorSummary,
    [`${base}txt`]:   plainSummary,
    [`${base}json`]:  JSON.stringify(data, null, 2),
  };
}

// ── Options ───────────────────────────────────────────────────────────────────

export const options = {
  scenarios: buildScenarios(),

  // Skip TLS verification globally so the K8s Metrics API (in-cluster HTTPS)
  // works without needing update-ca-certificates on the ray_serve_test pod.
  insecureSkipTLSVerify: true,

  thresholds: {
    'prompt_c2_latency_ms': ['p(95)<5000', 'p(99)<10000'],
    'mcp_c2_latency_ms':    ['p(95)<2000', 'p(99)<5000'],
    'error_rate':            ['rate<0.05'],
    'http_req_duration':     ['p(99)<15000'],
  },

  summaryTrendStats: ['min', 'med', 'avg', 'p(90)', 'p(95)', 'p(99)', 'max', 'count'],
};
