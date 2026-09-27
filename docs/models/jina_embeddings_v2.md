# Serving jina-embeddings-v2 (JinaBert) on TPU

Runbook for bringing up `jinaai/jina-embeddings-v2-small-en` — a JAX-native
encoder-only embedding model (symmetric ALiBi, GEGLU, post-LayerNorm) — on a
fresh TPU VM (validated on v6e, TP=1). The model runs under vLLM's pooling
runner with no KV cache; mean pooling executes in vLLM's CPU pooler.

## 1. Code

```bash
git clone https://github.com/shivajid/tpu-inference.git
cd tpu-inference
git checkout jina-v2-embeddings-clean
```

## 2. Python environment (uv, python 3.12)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
uv venv ~/vllm-env --python 3.12 --seed    # --seed installs pip into the venv
source ~/vllm-env/bin/activate
```

## 3. Install vLLM TPU stack, then overlay the repo editable

```bash
python -m pip install vllm-tpu==0.26.0           # version this runbook was validated against
python -m pip install -e .                       # CRITICAL — see below
python -m pip install onnxruntime transformers   # for the parity test
```

(Validated environment: `vllm-tpu 0.26.0`, `jax/jaxlib 0.11.0`,
`libtpu 0.0.44`, python 3.12.)

The editable install is **required**: `vllm serve` spawns API-server and
engine subprocesses that import `tpu_inference` from site-packages, not from
your checkout. Without `-e .` those processes silently run the stale bundled
copy. Verify (must print the repo path, not site-packages):

```bash
cd ~ && python -c 'import tpu_inference, os; print(os.path.dirname(tpu_inference.__file__))'
cd ~/tpu-inference
```

## 4. Environment patches (one shot, idempotent)

```bash
python scripts/setup_jina_env.py
export HF_TOKEN=<your-token>   # optional; avoids HF rate limits
```

The script applies three patches to the active environment (see its
docstring for details): `transformers.onnx` and `transformers.pytorch_utils`
stubs for APIs removed in transformers v5 but still imported (never executed
at inference) by the Jina remote-code config, and a `sitecustomize.py` hook
that registers the JinaBert architectures with vLLM's ModelRegistry in every
process — needed because the API-server process validates ModelConfig before
`vllm.general_plugins` load or `tpu_inference` is imported.

## 5. Validate

```bash
pytest tests/models/jax/test_jina_bert.py -v -rs   # parity vs official ONNX export; expect 3 passed
pytest tests/e2e/test_jina_embeddings.py -v -rs    # full engine path; expect 1 passed
pytest tests/models/jax/test_jina_bert_megakernel.py -v -rs -s  # vs XLA path; 67 tests
pytest tests/kernels/jina_bert_megakernel_test.py -v -rs  # vs dense reference; 75 tests
```

The parity test compares per-token hidden states and mean-pooled embeddings
against the model repo's official ONNX export (same weights, float32, no
remote code) and requires cosine similarity > 0.999. The megakernel tests
compare the encoder megakernel (both kernel versions, both matmul
precisions) against the per-layer XLA path and a dense reference (see the
kernel README). The host-side pooling paths and the switches below are
covered by `pytest tests/models/common/test_pooling.py tests/test_envs.py`
(no TPU needed).

## 6. Serve

```bash
vllm serve jinaai/jina-embeddings-v2-small-en --runner pooling --convert embed \
  --trust-remote-code --max-model-len 2048 --max-num-batched-tokens 2048 \
  --dtype float32
```

`--convert embed` is required: vLLM classifies the `JinaBertForMaskedLM`
architecture string as masked-LM and gates the embeddings API otherwise. No
pooler flag is needed — vLLM auto-detects mean pooling from the repo's
sentence-transformers configuration.

The encoder runs as one Pallas megakernel by default
(`tpu_inference/kernels/jina_bert_megakernel/README.md`). It accepts at most
2048 tokens per step, hence `--max-num-batched-tokens 2048`. Every speed-up
has a switch; unset means the default:

| Environment variable             | Default   | `0` / other value                                        |
| -------------------------------- | --------- | -------------------------------------------------------- |
| `USE_JINA_BERT_MEGAKERNEL`       | `1`       | `0`: per-layer XLA path                                  |
| `JINA_BERT_MEGAKERNEL_PRECISION` | `default` | `highest`: full-fp32 matmuls (default: bf16 MXU operands, fp32 accumulation, as the XLA path) |
| `JINA_BERT_MEGAKERNEL_VERSION`   | `v2`      | `v1`: the first megakernel                               |
| `TPU_POOLING_FAST_PATH`          | `1`       | `0`: vLLM's pooler under torchax's dispatch modes, and sampling metadata built every step (the previous behaviour) |
| `JINA_BERT_DEVICE_POOLING`       | `0`       | `1`: mean pooling on the TPU, see below                  |

`TPU_POOLING_FAST_PATH=1` runs the same vLLM pooler on the same host copy
of the hidden states, but as plain torch: the output is bitwise identical,
and in a CPU-only measurement of the pooler call (1x1024 to 1x2048 tokens)
it saved ~1.4 ms per step of torchax dispatch. `JINA_BERT_DEVICE_POOLING=1`
computes the per-request means on the TPU and copies only
`[num_requests, 512]` to the host (instead of `[num_tokens, 512]`, 4 MiB at
2048 tokens); vLLM's pooler still applies the embedding head
(normalization). It changes where the mean is computed, so it is off by
default; the means match the CPU pooler to ~1e-7.

Query:

```bash
curl localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model": "jinaai/jina-embeddings-v2-small-en", "input": "How is the weather today?"}'
```

Expect a 512-dimensional embedding.

## 7. Measure

**Model forward, no server.** With vLLM stopped (the script needs the
chip), from the repo root:

```bash
python scripts/benchmarking/jina/measure_jina_forward.py --output /tmp/after.json
```

It times the jitted forward (embeddings + encoder) for 1x1024, 2x1024
packed and 1x2048 tokens on the real checkpoint: megakernel (every kernel
version at this commit) and the per-layer XLA path, each at matmul precision
`default` and `highest`; and, for the default megakernel paths, forward +
host copy + vLLM's embed pooler for each pooling path (`old` =
`TPU_POOLING_FAST_PATH=0`, `fast`, `device`). 20 warm-up runs, then 200
timed runs to `block_until_ready`; the JSON has median, p90, mean and min
in ms. Useful flags: `--cases 1x2048`, `--precisions default`,
`--paths megakernel`, `--versions v2`, `--no-pooling`, `--runs 500`, and
`--trace-dir /tmp/jina-forward-trace` (a `jax.profiler` trace of 5 extra
runs of every measurement, annotated per measurement).

Before/after: at this commit the `v1` rows with pooling `old` are the
previous behaviour. To time an older commit itself, copy the script out of
the tree first (it runs against older code, timing what exists there):

```bash
cp scripts/benchmarking/jina/measure_jina_forward.py /tmp/
git checkout <old-commit> && python /tmp/measure_jina_forward.py --output /tmp/before.json
git checkout -   # back to this commit (the editable install follows the checkout)
```

**Device trace of the live server.** Start the server with vLLM's
profiler enabled (the TPU worker then records a `jax.profiler` trace
between the two calls below; the API server gets the endpoints):

```bash
vllm serve jinaai/jina-embeddings-v2-small-en --runner pooling --convert embed \
  --trust-remote-code --max-model-len 2048 --max-num-batched-tokens 2048 \
  --dtype float32 \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/tmp/jina-trace", "ignore_frontend": true}'
```

Start the load (e.g. the k6 step at the rate of interest), wait until it is
steady, then capture a few seconds:

```bash
curl -X POST localhost:8000/start_profile
sleep 3
curl -X POST localhost:8000/stop_profile
```

Open `/tmp/jina-trace` with xprof (`pip install xprof`, then
`xprof --logdir /tmp/jina-trace --port 8791`) or TensorBoard with
`tensorboard-plugin-profile`. The trace view shows, per engine step, the
host thread (`execute_model`, `_prepare_inputs`, the pooler) and the TPU
timeline (the `jina_bert_encoder_megakernel_v2` kernel, host transfers,
and with `JINA_BERT_DEVICE_POOLING=1` the pooling reduction).
`PYTHON_TRACER_LEVEL=0` in the server's environment turns off Python
function tracing (less overhead, fewer host details);
`PROFILE_SINGLE_DEVICE=1` limits device tracing to one chip. The profiler
adds host overhead, so read latencies from the unprofiled runs and use the
trace for the breakdown.

## Known limitations / follow-ups

- At most 2048 tokens per step (so `max_model_len` <= 2048): the megakernel
  keeps the whole step in VMEM and raises on longer steps, and the per-layer
  path's dense ALiBi bias tensor is O(heads × T²). The full 8192 context
  needs a different kernel design.
- float32 only (validated); bf16 pending validation against the fp32 baseline.
- TP=1 (model is ~33M params); slopes already shard with heads for TP > 1.
- The sitecustomize hook is a workaround; an upstream vLLM registration hook
  in the API-server process would remove it.
