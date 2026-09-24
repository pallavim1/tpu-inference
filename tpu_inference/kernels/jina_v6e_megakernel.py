# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU v6e (Trillium) 4-Layer Fused FP32 Megakernel for Jina-Embeddings-v2-Small-EN.

Architecture & Key Optimizations over Baseline `jina_bert.py`:
1. Cross-Layer Weight Stacking & QKV Fusion (`[4, ...]`):
   - Fuses `query`, `key`, `value` weights per layer into a single head-first tensor
     `w_qkv` of shape `[4, D, 3, N, H]` and `b_qkv` of shape `[4, 3, N, H]`.
   - Stacks `w_o` (`[4, N, H, D]`), `b_o` (`[4, D]`), `ln1_scale`/`ln1_bias` (`[4, D]`),
     `w_gated` (`[4, D, 2*I]`), `w_down` (`[4, I, D]`), `b_down` (`[4, D]`), and
     `ln2_scale`/`ln2_bias` (`[4, D]`).
2. Single `jax.shard_map` + `jax.lax.scan` Megakernel Boundary:
   - Replaces the 4 per-layer `jax.jit(jax.shard_map(...))` barriers with a SINGLE
     outer `jax.jit(jax.shard_map(...))` wrapping all 4 layers.
3. Single Precomputed `SegmentIds` & Token Padding:
   - Computes `align_to(T, 128)` padding and `SegmentIds` (`jnp.repeat`) ONCE before
     Layer 0 instead of repeating them 4 times across layers 0..3.
   - Projects `x_pad` (`[T_pad, D]`) directly into head-first `[3, N_local, T_pad, H]`
     layout via `einsum("td,dcnh->cnht")` and contracts attention output directly back
     to `[T_pad, D]` via `einsum("nth,nhd->td")`, eliminating all 16 `swapaxes` and
     `jnp.pad` HBM copy kernels across the 4 layers.
4. TPU v6e 32 MB VMEM Single-Step Pallas ALiBi FlashAttention:
   - Uses `vmem_limit_bytes = 32 * 1024 * 1024` (32 MB VMEM on Trillium v6e) and
     `block_q = min(512, T_pad), block_k_major = T_pad, block_k = T_pad` for `<= 2K`
     workloads so Pallas dispatches `_flash_attention_kernel_single_batch_single_step`
     completely inside VMEM without multi-pass online softmax rescaling.
"""

from __future__ import annotations

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.flash_attention.kernel import (
    BlockSizes,
    SegmentIds,
    _flash_attention,
)
from tpu_inference.utils import align_to

# TPU v6e (Trillium) provides 32 MB of VMEM per TensorCore.
TPU_V6E_VMEM_LIMIT_BYTES = 32 * 1024 * 1024


def _fused_layer_norm_fp32(
    x: jax.Array,
    scale: jax.Array,
    bias: jax.Array,
    eps: float = 1e-12,
) -> jax.Array:
    """In-register FP32 LayerNorm (mean/variance reduction + affine scale/shift)."""
    x_f32 = x.astype(jnp.float32)
    mean = jnp.mean(x_f32, axis=-1, keepdims=True)
    centered = x_f32 - mean
    var = jnp.mean(centered * centered, axis=-1, keepdims=True)
    inv_std = lax.rsqrt(var + eps)
    normed = centered * inv_std
    return (normed * scale.astype(jnp.float32) + bias.astype(jnp.float32)).astype(
        x.dtype
    )


def _build_segment_ids_once(
    seq_lens: jax.Array,
    q_len: int,
    padded_len: int,
) -> SegmentIds:
    """Builds SegmentIds ONCE for the entire 4-layer encoder megakernel."""
    max_num_seqs = seq_lens.shape[0]
    zero_2_max_num_seqs = jnp.arange(max_num_seqs + 1, dtype=jnp.int32)
    seq_lens_concat_zero = jnp.concatenate([
        seq_lens,
        jnp.array([0], dtype=seq_lens.dtype),
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


def _select_v6e_block_sizes(padded_len: int) -> BlockSizes:
    """Selects optimal TPU v6e VMEM block sizes for <= 2K token sequences.

    For padded_len <= 4096, the entire KV sequence fits comfortably inside
    TPU v6e's 32 MB VMEM, enabling `_flash_attention_kernel_single_batch_single_step`
    with large query tiles (up to 512 tokens per MXU program block).
    """
    if padded_len <= 128:
        block_q = 128
    elif padded_len <= 256:
        block_q = 256
    else:
        block_q = 512

    if padded_len <= 4096:
        block_k = padded_len
    else:
        block_k = 512

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
    x: jax.Array,                  # [T, D]
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
    """Executes all 4 JinaBert layers inside a single `shard_map` + `scan` Megakernel."""
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
        # Align to 512 when q_len > 256 so block_q=512 divides padded_len cleanly
        align_quantum = 512 if q_len > 256 else (256 if q_len > 128 else 128)
        padded_len = align_to(q_len, align_quantum)
        pad_tokens = padded_len - q_len

        # 1. Pad token activations ONCE before Layer 0
        if pad_tokens > 0:
            x_pad = jnp.pad(x_in, ((0, pad_tokens), (0, 0)), constant_values=0.0)
        else:
            x_pad = x_in

        # 2. Build SegmentIds ONCE before Layer 0 (eliminates 3x redundant jnp.repeat)
        segment_ids = _build_segment_ids_once(seq_lens_in, q_len, padded_len)
        block_sizes = _select_v6e_block_sizes(padded_len)

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
            # Zero swapaxes, zero intermediate padding!
            qkv = (
                jnp.einsum(
                    "td,dcnh->cnht",
                    x_curr,
                    wqkv_l,
                    precision=lax.Precision.HIGHEST,
                )
                + bqkv_l[:, :, None, :]
            )
            q_bhtd = qkv[0:1]  # [1, N_local, T_pad, H]
            k_bhtd = qkv[1:2]  # [1, N_local, T_pad, H]
            v_bhtd = qkv[2:3]  # [1, N_local, T_pad, H]

            # Step B: Single-Step VMEM-Resident Pallas ALiBi FlashAttention (32 MB VMEM)
            attn_bhtd = _flash_attention(
                q_bhtd,
                k_bhtd,
                v_bhtd,
                ab=None,
                segment_ids=segment_ids,
                alibi_slopes=alibi_in,
                save_residuals=False,
                causal=False,
                sm_scale=sm_scale,
                block_sizes=block_sizes,
                vmem_limit_bytes=TPU_V6E_VMEM_LIMIT_BYTES,
                debug=False,
            )

            # Step C: Direct Head-First Output Contraction + Residual + LayerNorm1
            # Contracts [N_local, T_pad, H] with [N_local, H, D] -> [T_pad, D]
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

        # Execute all 4 layers via lax.scan inside the single shard_map
        x_final_pad, _ = lax.scan(_scan_layer, x_pad, layer_weights)

        # 3. Unpad ONCE after Layer 3
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
