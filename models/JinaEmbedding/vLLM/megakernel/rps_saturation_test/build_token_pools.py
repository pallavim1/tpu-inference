#!/usr/bin/env python3
"""Builds pools of random-character prompts with an exact token count.

The characters come from the same charset as the k6 saturation scripts
(k6_1kb_saturation_test.js / k6_2kb_saturation_test.js). Token counts use the
model's own fast tokenizer and include [CLS] and [SEP], so they equal the
prompt_tokens that vLLM reports for each request.

Usage:
  build_token_pools.py measure              # tokens in 1024/2048-byte k6 payloads
  build_token_pools.py build OUT_DIR COUNT  # COUNT prompts of 1024 and of 2048 tokens
"""
import json
import multiprocessing as mp
import os
import random
import statistics
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from transformers import AutoTokenizer  # noqa: E402

MODEL = "jinaai/jina-embeddings-v2-small-en"
CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?;:\n"
TARGETS = (1024, 2048)
_TOKENIZER = None


def tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        except Exception:
            _TOKENIZER = AutoTokenizer.from_pretrained(MODEL)
    return _TOKENIZER


def n_tokens(text):
    return len(tokenizer()(text, add_special_tokens=True)["input_ids"])


def rand_text(rng, n_chars):
    return "".join(rng.choice(CHARSET) for _ in range(n_chars))


def exact_prompt(args):
    """Returns a random-character prompt with exactly `target` tokens."""
    target, seed = args
    rng = random.Random(seed)
    n_chars = target * 2
    while True:
        text = rand_text(rng, n_chars)
        if n_tokens(text) < target:
            n_chars *= 2
            continue
        lo, hi = 1, len(text)
        while lo < hi:  # Shortest prefix with at least `target` tokens.
            mid = (lo + hi) // 2
            if n_tokens(text[:mid]) >= target:
                hi = mid
            else:
                lo = mid + 1
        for length in range(max(1, lo - 4), min(len(text), lo + 12) + 1):
            if n_tokens(text[:length]) == target:
                return text[:length]
        # The count skipped over `target` for this string; try a new one.


def measure():
    tok = tokenizer()
    print(f"tokenizer: {type(tok).__name__}, fast={tok.is_fast}, "
          f"model_max_length={tok.model_max_length}")
    rng = random.Random(0)
    for n_bytes in TARGETS:
        counts = [n_tokens(rand_text(rng, n_bytes)) for _ in range(300)]
        print(f"{n_bytes}-byte k6-style random text -> tokens incl. CLS/SEP: "
              f"min {min(counts)}, mean {statistics.mean(counts):.1f}, "
              f"max {max(counts)} (300 samples)")


def build(out_dir, count):
    os.makedirs(out_dir, exist_ok=True)
    with mp.Pool(min(24, os.cpu_count() or 8)) as pool:
        for target in TARGETS:
            jobs = [(target, target * 1_000_000 + i) for i in range(count)]
            prompts = pool.map(exact_prompt, jobs, chunksize=64)
            checked = pool.map(n_tokens, prompts, chunksize=256)
            assert all(c == target for c in checked), "token count mismatch"
            lens = [len(p) for p in prompts]
            path = os.path.join(out_dir, f"pool_{target}tok.json")
            with open(path, "w") as f:
                json.dump(prompts, f)
            print(f"{path}: {len(prompts)} prompts ({len(set(prompts))} unique), "
                  f"{target} tokens each (all verified); chars min {min(lens)}, "
                  f"mean {statistics.mean(lens):.0f}, max {max(lens)}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "measure":
        measure()
    elif len(sys.argv) == 4 and sys.argv[1] == "build":
        build(sys.argv[2], int(sys.argv[3]))
    else:
        sys.exit(__doc__)
