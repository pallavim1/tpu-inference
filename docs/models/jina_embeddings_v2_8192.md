# Serving and Benchmarking jina-embeddings-v2 (JinaBert) on TPU (8192 Context Length)

Step-by-step runbook for bringing up, serving, and benchmarking `jinaai/jina-embeddings-v2-small-en` — a JAX-native encoder-only embedding model with symmetric ALiBi Flash-Attention kernels — on a Cloud TPU VM (validated on TPU v5e & v6e, TP=1, full **8192-token context length**).

The model runs under vLLM's pooling runner with no KV cache. The symmetric ALiBi bias is computed dynamically inside the Flash-Attention kernel (`docs/developer_guides/encoder_alibi_kernel.md`), enabling full 8192-token sequence processing without extra HBM overhead.

---

## 1. Code Checkout & Repository Setup

```bash
git clone git@github.com:pallavim1/tpu-inference.git
cd tpu-inference
git checkout jina-v2-alibi-kernel
```

---

## 2. Python Environment Setup (uv, Python 3.12)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
uv venv ~/vllm-env --python 3.12 --seed    # --seed installs pip into the venv
source ~/vllm-env/bin/activate
```

---

## 3. Install vLLM TPU Stack & Overlay Repository Editable

```bash
python -m pip install vllm-tpu==0.26.0           # validated version
python -m pip install -e .                       # CRITICAL: editable install
python -m pip install onnxruntime transformers   # required for parity testing
```

> **IMPORTANT:** The editable install (`pip install -e .`) is **required** because `vllm serve` spawns API-server and engine subprocesses that import `tpu_inference` from site-packages. Without `-e .`, those processes will run a stale bundled copy.

Verify that `tpu_inference` points to your repository checkout:
```bash
python -c 'import tpu_inference, os; print(os.path.dirname(tpu_inference.__file__))'
```

---

## 4. Environment Patches (One-Shot, Idempotent)

```bash
python scripts/setup_jina_env.py
```

If `HF_TOKEN` is set to an unauthenticated or dummy secret (such as `"test"`), unset it to avoid HuggingFace Hub `401 Unauthorized` errors when downloading public model weights:
```bash
unset HF_TOKEN
```

---

## 5. Model & Kernel Validation Tests

Run the full validation suite to verify kernel correctness and numerical parity:

```bash
pytest tests/kernels/encoder_alibi_kernel_test.py -v  # ALiBi kernel vs dense reference (expect 5 passed)
pytest tests/models/jax/test_jina_bert.py -v -rs       # Parity vs official ONNX export (expect 3 passed)
pytest tests/e2e/test_jina_embeddings.py -v -rs        # Full vLLM engine integration path
```

---

## 6. Serve Model with `--max-model-len 8192`

Serve the model using `vllm serve` with sequence length and batch token limits expanded to **8192 tokens**:

```bash
vllm serve jinaai/jina-embeddings-v2-small-en \
  --runner pooling \
  --convert embed \
  --trust-remote-code \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --dtype float32 \
  --host 0.0.0.0 \
  --port 8000
```

### Key Parameter Rationale:
* `--max-model-len 8192`: Extends maximum model sequence length from the default 2048 to **8192 tokens**.
* `--max-num-batched-tokens 8192`: Must be $\ge$ `max_model_len`. Encoder models do not use KV caching and cannot chunk prefill; full sequence prompts must fit in a single token budget step.
* `--convert embed`: Required for vLLM to route `JinaBertForMaskedLM` model output to the `/v1/embeddings` and `/prompt_c2` endpoints.

---

## 7. Verifying Endpoints

### Short-Context Embedding Query
```bash
curl http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model": "jinaai/jina-embeddings-v2-small-en", "input": "Google Cloud TPU v5e jina embeddings test."}'
```
*Expect a 512-dimensional embedding vector.*

### Full 8192-Token Long-Context Smoke Test (~7,500 Tokens)
```bash
python3 - <<'PY'
import json, urllib.request

# Generate 7500+ token sequence
text = "Google Cloud TPU v5e JAX ALiBi Flash-Attention embedding kernel payload sequence. " * 750

req = urllib.request.Request(
    "http://localhost:8000/v1/embeddings",
    json.dumps({"model": "jinaai/jina-embeddings-v2-small-en", "input": text}).encode(),
    {"Content-Type": "application/json"}
)
r = json.load(urllib.request.urlopen(req))
print("Embedding Dimensions:", len(r["data"][0]["embedding"]), "| Token Usage:", r["usage"])
PY
```

---

## 8. Running Automated Saturation Benchmarks

To execute the automated k6 saturation benchmark suite matching Section 7 of `BENCHMARKING_GUIDE.md`:

```bash
python3 scripts/benchmarking/run_jina_v2_alibi_saturation_benchmark.py \
  --endpoint http://localhost:8000/prompt_c2 \
  --duration 60s \
  --max-model-len 8192
```

This runs load tests across 1K, 2K, 5K, 7K, and 8K token payloads at 60s stage durations and outputs latency percentiles ($P_{50}, P_{90}, P_{95}, P_{99}$) and throughput metrics.
