# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TPU v6e (Trillium) 4-Layer Fused FP32 Megakernel for JinaBert (`max_model_len = 2048` ONLY).

Strictly enforces `MAX_MODEL_LEN = 2048` (never compiling or padding above 2,048 tokens)
on the `jina-v2-embeddings-clean` branch.

Key Fusions Across All 4 JinaBert Encoder Layers:
1. Strict `MAX_MODEL_LEN = 2048` Enforcement (`q_len <= 2048`).
2. Single-Pass Pre-Layer-0 Padding, SegmentIds, and Symmetric ALiBi Bias (`ab`) Construction:
   - Eliminates 3x redundant `ab` (`[1, N, T, T]`) and `SegmentIds` computation across Layers 1-3.
   - Uses hardware DMA for `ab` inside VMEM-resident `_flash_attention_kernel_single_batch_single_step`
     instead of per-tile VPU `iota` synthesis.
3. Head-First QKV & Output Contractions (`einsum("td,dcnh->cnth")` and `einsum("nth,nhd->td")`)
   with `lax.Precision.HIGHEST` (IEEE FP32).
4. Fused Single-Reduction FP32 LayerNorm (`E[x]` and `E[x^2]` in one pass) and GeGLU MLP
   inside a single `jax.shard_map` + `lax.scan` 4-layer loop.
"""

import functools
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P

from tpu_inference.kernels.flash_attention.kernel import (
    BlockSizes,
    SegmentIds,
    _flash_attention,
)
from tpu_inference.utils import align_to

# Strictly cap JinaBert on jina-v2-embeddings-clean to 2048 tokens
MAX_MODEL_LEN = 2048

# TPU v6e (Trillium) provides 32 MB of VMEM per TensorCore
TPU_V6E_VMEM_LIMIT_BYTES = 32 * 1024 * 1024


def _fused_layer_norm_fp32(
    x: jax.Array,
    scale: jax.Array,
    bias: jax.Array,
    eps: float = 1e-12,
) -> jax.Array:
    """Single-pass FP32 LayerNorm using fused first/second moments."""
    x_f32 = x.astype(jnp.float32)
    mean = jnp.mean(x_f32, axis=-1, keepdims=True)
    mean_sq = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
    var = jnp.maximum(0.0, mean_sq - jnp.square(mean))
    rstd = lax.rsqrt(var + eps)
    normed = (x_f32 - mean) * rstd
    return (normed * scale.astype(jnp.float32) + bias.astype(jnp.float32)).astype(
        x.dtype
    )


def _build_segment_ids_once(
    seq_lens: jax.Array,
    q_len: int,
    padded_len: int,
) -> SegmentIds:
    """Constructs 2D `[1, padded_len]` SegmentIds ONCE for all 4 layers."""
    max_num_seqs = seq_lens.shape[0]
    zero_2_max_num_seqs = jnp.arange(0, max_num_seqs + 1, dtype=jnp.int32)
    sum_seq_len = jnp.sum(seq_lens)
    seq_lens_concat_zero = jnp.concatenate([
        seq_lens,
        jnp.asarray([q_len - sum_seq_len], dtype=jnp.int32),
    ])
    qkv_segment_ids = jnp.repeat(
        zero_2_max_num_seqs,
        seq_lens_concat_zero,
        total_repeat_length=q_len,
    )
    pad_size = padded_len - q_len
    padding_segment_id = max_num_seqs
    padded_seg = jnp.pad(
        qkv_segment_ids,
        (0, pad_size),
        constant_values=padding_segment_id,
    )
    padded_seg_2d = jnp.expand_dims(padded_seg, axis=0)
    return SegmentIds(q=padded_seg_2d, kv=padded_seg_2d)


def _build_alibi_bias_once(
    alibi_slopes: jax.Array,
    padded_len: int,
    sm_scale: float,
) -> jax.Array:
    """Constructs `[1, N_local, padded_len, padded_len]` symmetric ALiBi bias ONCE before Layer 0."""
    pos = jnp.arange(padded_len, dtype=jnp.int32)
    distance = jnp.abs(pos[:, None] - pos[None, :])
    alibi = -alibi_slopes.astype(jnp.float32)[:, None, None] * (
        distance.astype(jnp.float32)[None, :, :] / sm_scale
    )
    return jnp.expand_dims(alibi, axis=0)


def _select_v6e_block_sizes_2048(padded_len: int) -> BlockSizes:
    """Selects optimal TPU v6e VMEM block sizes for `padded_len <= 2048` (`MAX_MODEL_LEN = 2048`)."""
    if padded_len <= 128:
        block_q = 128
    elif padded_len <= 256:
        block_q = 256
    else:
        block_q = 512

    # For padded_len <= 2048, the entire KV sequence + ALiBi bias tile (~11 MB total)
    # fits inside TPU v6e's 32 MB VMEM in a single step!
    block_k = padded_len

    return BlockSizes(
        block_q=block_q,
        block_k_major=block_k,
        block_k=block_k,
        block_b=1,
    )


@functools.partial(
    jax.jit,
    static_argnames=("mesh", "sm_scale", "layer_norm_eps"),
)
def jina_v6e_4layer_megakernel(
    x: jax.Array,                  # [T, D] where T <= 2048
    seq_lens: jax.Array,           # [max_num_seqs]
    w_qkv: jax.Array,              # [4, D, 3, N, H]
    b_qkv: jax.Array,              # [4, 3, N, H]
    w_o: jax.Array,                # [4, N, H, D]
    b_o: jax.Array,                # [4, D]
    ln1_scale: jax.Array,          # [4, D]
    ln1_bias: jax.Array,           # [4, D]
    w_gated: jax.Array,            # [4, D, 2*I]
    w_down: jax.Array,             # [4, I, D]
    b_down: jax.Array,             # [4, D]
    ln2_scale: jax.Array,          # [4, D]
    ln2_bias: jax.Array,           # [4, D]
    alibi_slopes: jax.Array,       # [N]
    mesh: Mesh,
    sm_scale: float,
    layer_norm_eps: float = 1e-12,
) -> jax.Array:
    """Executes all 4 JinaBert layers inside a single `shard_map` + `scan` Megakernel (`T <= 2048`)."""
    q_len_total = x.shape[0]
    if q_len_total > MAX_MODEL_LEN:
        raise ValueError(
            f"jina_v6e_4layer_megakernel on jina-v2-embeddings-clean strictly supports "
            f"max_model_len <= {MAX_MODEL_LEN}, got {q_len_total}"
        )

    in_specs = (
        P(None, None),                     # x: [T, D]
        P(),                               # seq_lens: [max_num_seqs]
        P(None, None, None, "model", None),  # w_qkv: [4, D, 3, N, H]
        P(None, None, "model", None),      # b_qkv: [4, 3, N, H]
        P(None, "model", None, None),      # w_o: [4, N, H, D]
        P(None, None),                     # b_o: [4, D]
        P(None, None),                     # ln1_scale: [4, D]
        P(None, None),                     # ln1_bias: [4, D]
        P(None, None, "model"),            # w_gated: [4, D, 2*I]
        P(None, "model", None),            # w_down: [4, I, D]
        P(None, None),                     # b_down: [4, D]
        P(None, None),                     # ln2_scale: [4, D]
        P(None, None),                     # ln2_bias: [4, D]
        P("model"),                        # alibi_slopes: [N]
    )
    out_specs = P(None, None)

    def _megakernel_body(
        x_in,
        seq_lens_in,
        w_qkv_in,
        b_qkv_in,
        w_o_in,
        b_o_in,
        ln1_s_in,
        ln1_b_in,
        w_gated_in,
        w_down_in,
        b_down_in,
        ln2_s_in,
        ln2_b_in,
        alibi_in,
    ):
        q_len = x_in.shape[0]
        align_quantum = 512 if q_len > 256 else (256 if q_len > 128 else 128)
        padded_len = align_to(q_len, align_quantum)
        pad_tokens = padded_len - q_len

        # 1. Pad token activations ONCE before Layer 0
        if pad_tokens > 0:
            x_pad = jnp.pad(x_in, ((0, pad_tokens), (0, 0)), constant_values=0.0)
        else:
            x_pad = x_in

        # 2. Build SegmentIds and Symmetric ALiBi bias (`ab`) ONCE before Layer 0
        segment_ids = _build_segment_ids_once(seq_lens_in, q_len, padded_len)
        ab = _build_alibi_bias_once(alibi_in, padded_len, sm_scale)
        block_sizes = _select_v6e_block_sizes_2048(padded_len)

        layer_weights = (
            w_qkv_in,
            b_qkv_in,
            w_o_in,
            b_o_in,
            ln1_s_in,
            ln1_b_in,
            w_gated_in,
            w_down_in,
            b_down_in,
            ln2_s_in,
            ln2_b_in,
        )

        def _scan_layer(x_curr, weights_l):
            (
                wqkv_l,
                bqkv_l,
                wo_l,
                bo_l,
                ln1_s_l,
                ln1_b_l,
                wgated_l,
                wdown_l,
                bdown_l,
                ln2_s_l,
                ln2_b_l,
            ) = weights_l

            # Step A: Fused QKV Projection directly into Head-First [3, N_local, T_pad, H]
            qkv = (
                jnp.einsum(
                    "td,dcnh->cnth",
                    x_curr,
                    wqkv_l,
                    precision=lax.Precision.HIGHEST,
                )
                + bqkv_l[:, :, None, :]
            )
            q_bhtd = qkv[0:1]
            k_bhtd = qkv[1:2]
            v_bhtd = qkv[2:3]

            # Step B: Single-Step VMEM-Resident Pallas FlashAttention (32 MB VMEM, shared `ab`)
            attn_bhtd = _flash_attention(
                q_bhtd,
                k_bhtd,
                v_bhtd,
                ab,
                segment_ids,
                False,
                False,
                sm_scale,
                block_sizes,
                TPU_V6E_VMEM_LIMIT_BYTES,
                False,
            )

            # Step C: Direct Head-First Output Contraction + Residual + LayerNorm1
            attn_proj = (
                jnp.einsum(
                    "nth,nhd->td",
                    attn_bhtd[0],
                    wo_l,
                    precision=lax.Precision.HIGHEST,
                )
                + bo_l
            )
            x_mid = _fused_layer_norm_fp32(
                x_curr + attn_proj,
                ln1_s_l,
                ln1_b_l,
                eps=layer_norm_eps,
            )

            # Step D: Fused GeGLU MLP + Residual + LayerNorm2
            h_gated_full = jnp.dot(
                x_mid,
                wgated_l,
                precision=lax.Precision.HIGHEST,
            )
            half_dim = h_gated_full.shape[-1] // 2
            gated = h_gated_full[..., :half_dim]
            non_gated = h_gated_full[..., half_dim:]
            h_act = jax.nn.gelu(gated, approximate=False) * non_gated
            mlp_out = (
                jnp.dot(
                    h_act,
                    wdown_l,
                    precision=lax.Precision.HIGHEST,
                )
                + bdown_l
            )
            x_next = _fused_layer_norm_fp32(
                x_mid + mlp_out,
                ln2_s_l,
                ln2_b_l,
                eps=layer_norm_eps,
            )
            return x_next, None

        x_final_pad, _ = lax.scan(_scan_layer, x_pad, layer_weights)
        return x_final_pad[:q_len]

    return jax.shard_map(
        _megakernel_body,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )(
        x,
        seq_lens,
        w_qkv,
        b_qkv,
        w_o,
        b_o,
        ln1_scale,
        ln1_bias,
        w_gated,
        w_down,
        b_down,
        ln2_scale,
        ln2_bias,
        alibi_slopes,
    )
