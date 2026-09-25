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
"""JAX-native JinaBert (jina-embeddings-v2) encoder-only embedding model
on `jina-v2-embeddings-clean` strictly supporting `max_model_len <= 2048`
with TPU v6e (Trillium) 4-Layer Fused FP32 Megakernel (`jina_v6e_4layer_megakernel`).
"""

import functools
import math
from typing import List, Optional, Tuple

from flax import nnx
from flax.typing import PRNGKey
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.kernels.jina_v6e_megakernel import (
    MAX_MODEL_LEN,
    jina_v6e_4layer_megakernel,
)
from tpu_inference.layers.common.attention_interface import (
    encoder_only_attention,
)
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax import JaxModule, JaxModuleList
from tpu_inference.layers.jax.layers import Embedder, JaxLayerNorm, JaxLinear
from tpu_inference.models.jax.jax_intermediate_tensor import (
    JaxIntermediateTensors,
)
from tpu_inference.models.jax.utils.weight_utils import (
    LoadableWithIterator,
    load_nnx_param_from_reshaped_torch,
)

init_fn = nnx.initializers.uniform()


def _get_alibi_slopes(num_heads: int) -> Tuple[float, ...]:
    """Compute the symmetric ALiBi slopes used by JinaBert."""

    def _powers_of_2(n: int) -> List[float]:
        start = 2.0 ** (-(2.0 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio**i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        slopes = _powers_of_2(num_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        slopes = (
            _powers_of_2(closest_power_of_2)
            + _get_alibi_slopes(2 * closest_power_of_2)[0::2][
                : num_heads - closest_power_of_2
            ]
        )
    return tuple(slopes)


def _set_weight_loader(
    param: nnx.Param,
    weight_sharding,
    mesh: Mesh,
    param_name: str,
    reshape_dims: Optional[Tuple[int, ...]] = None,
    permute_dims: Optional[Tuple[int, ...]] = None,
) -> None:
    """Attach an explicit HF->JAX weight loader to a param."""
    param.set_metadata(
        "weight_loader",
        functools.partial(
            load_nnx_param_from_reshaped_torch,
            permute_dims=permute_dims,
            reshape_dims=reshape_dims,
            weight_sharding=weight_sharding,
            mesh=mesh,
            param_name=param_name,
        ),
    )


class JinaBertEmbeddings(JaxModule):
    """Word + token_type embeddings + LayerNorm (no position embeddings; ALiBi handles position)."""

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs):
        self.word_embeddings = Embedder(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            dtype=dtype,
            vd_sharding=(("data", "model"), None),
            rngs=rng,
        )
        self.token_type_embeddings = nnx.Param(
            init_fn(
                rng.params(),
                (config.type_vocab_size, config.hidden_size),
                dtype,
            )
        )
        self.LayerNorm = JaxLayerNorm(
            num_features=config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            rngs=rng,
        )

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        words = self.word_embeddings.encode(input_ids)
        token_type_zero = self.token_type_embeddings.value[0]
        return self.LayerNorm(words + token_type_zero)


class JinaBertSelfAttention(JaxModule):
    """Multi-head bidirectional self-attention with symmetric ALiBi bias."""

    def __init__(
        self, config, dtype: jnp.dtype, rng: nnx.Rngs, mesh: Mesh, prefix: str
    ):
        hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        assert hidden_size % self.num_heads == 0
        self.head_dim_original = hidden_size // self.num_heads
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)
        self.mesh = mesh

        def _make_qkv_proj(name: str) -> JaxLinear:
            proj = JaxLinear(
                hidden_size,
                self.num_heads * self.head_dim,
                use_bias=True,
                dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn, (None, "model", None)),
                bias_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                prefix=f"{prefix}.{name}",
            )
            proj.weight = nnx.Param(
                init_fn(
                    rng.params(),
                    (hidden_size, self.num_heads, self.head_dim),
                    dtype,
                )
            )
            _set_weight_loader(
                proj.weight,
                (None, "model", None),
                mesh,
                f"{prefix}.{name}.weight",
                reshape_dims=(self.num_heads, self.head_dim, hidden_size),
                permute_dims=(2, 0, 1),
            )
            proj.bias = nnx.Param(
                init_fn(rng.params(), (self.num_heads, self.head_dim), dtype)
            )
            _set_weight_loader(
                proj.bias,
                ("model", None),
                mesh,
                f"{prefix}.{name}.bias",
                reshape_dims=(self.num_heads, self.head_dim),
            )
            return proj

        self.query = _make_qkv_proj("query")
        self.key = _make_qkv_proj("key")
        self.value = _make_qkv_proj("value")

        slopes = jnp.asarray(_get_alibi_slopes(self.num_heads), dtype=jnp.float32)
        self.alibi_slopes = jax.device_put(
            slopes,
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec("model")
            ),
        )

    def __call__(
        self, x: jax.Array, attention_metadata: AttentionMetadata
    ) -> jax.Array:
        q = jnp.einsum("td,dnh->tnh", x, self.query.weight.value) + self.query.bias.value
        k = jnp.einsum("td,dnh->tnh", x, self.key.weight.value) + self.key.bias.value
        v = jnp.einsum("td,dnh->tnh", x, self.value.weight.value) + self.value.bias.value

        sm_scale = self.head_dim_original**-0.5
        out_tnh = encoder_only_attention(
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            head_dim_original=self.head_dim_original,
            sm_scale=sm_scale,
            alibi_slopes=self.alibi_slopes,
        )
        t = out_tnh.shape[0]
        return jnp.reshape(out_tnh, (t, self.num_heads * self.head_dim))


class JinaBertSelfOutput(JaxModule):
    """Output projection + residual + post-LayerNorm (`attention.output.*`)."""

    def __init__(
        self,
        config,
        num_heads: int,
        head_dim: int,
        dtype: jnp.dtype,
        rng: nnx.Rngs,
        mesh: Mesh,
        prefix: str,
    ):
        hidden_size = config.hidden_size
        self.dense = JaxLinear(
            num_heads * head_dim,
            hidden_size,
            use_bias=True,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            bias_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rng,
            prefix=prefix + ".dense",
        )
        self.dense.weight = nnx.Param(
            init_fn(rng.params(), (num_heads, head_dim, hidden_size), dtype)
        )
        _set_weight_loader(
            self.dense.weight,
            ("model", None, None),
            mesh,
            prefix + ".dense.weight",
            reshape_dims=(hidden_size, num_heads, head_dim),
            permute_dims=(1, 2, 0),
        )
        self.LayerNorm = JaxLayerNorm(
            num_features=hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            rngs=rng,
        )

    def __call__(self, hidden_states: jax.Array, residual: jax.Array) -> jax.Array:
        out = (
            jnp.einsum("tnh,nhd->td", hidden_states, self.dense.weight.value)
            + self.dense.bias.value
        )
        return self.LayerNorm(out + residual)


class JinaBertAttention(JaxModule):

    def __init__(
        self, config, dtype: jnp.dtype, rng: nnx.Rngs, mesh: Mesh, prefix: str
    ):
        self_attention = JinaBertSelfAttention(
            config, dtype, rng, mesh, prefix=prefix + ".self"
        )
        setattr(self, "self", self_attention)
        self.output = JinaBertSelfOutput(
            config,
            self_attention.num_heads,
            self_attention.head_dim,
            dtype,
            rng,
            mesh,
            prefix=prefix + ".output",
        )

    def __call__(
        self, x: jax.Array, attention_metadata: AttentionMetadata
    ) -> jax.Array:
        attn_flat = getattr(self, "self")(x, attention_metadata)
        self_attn = getattr(self, "self")
        attn_tnh = jnp.reshape(
            attn_flat, (x.shape[0], self_attn.num_heads, self_attn.head_dim)
        )
        return self.output(attn_tnh, x)


class JinaBertGLUMLP(JaxModule):
    """GEGLU MLP with internal residual + post-LayerNorm."""

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, prefix: str):
        hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gated_layers = JaxLinear(
            hidden_size,
            2 * self.intermediate_size,
            use_bias=False,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, "model")),
            rngs=rng,
            prefix=prefix + ".gated_layers",
        )
        self.wo = JaxLinear(
            self.intermediate_size,
            hidden_size,
            use_bias=True,
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None)),
            bias_init=nnx.with_partitioning(init_fn, (None,)),
            rngs=rng,
            prefix=prefix + ".wo",
        )
        self.layernorm = JaxLayerNorm(
            num_features=hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            rngs=rng,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        h = self.gated_layers(x)
        gated = h[..., : self.intermediate_size]
        non_gated = h[..., self.intermediate_size :]
        h = jax.nn.gelu(gated, approximate=False) * non_gated
        h = self.wo(h)
        return self.layernorm(h + residual)


class JinaBertLayer(JaxModule):

    def __init__(
        self, config, dtype: jnp.dtype, rng: nnx.Rngs, mesh: Mesh, prefix: str
    ):
        self.attention = JinaBertAttention(
            config, dtype, rng, mesh, prefix=prefix + ".attention"
        )
        self.mlp = JinaBertGLUMLP(config, dtype, rng, prefix=prefix + ".mlp")

    def __call__(
        self, x: jax.Array, attention_metadata: AttentionMetadata
    ) -> jax.Array:
        x = self.attention(x, attention_metadata)
        return self.mlp(x)


class JinaBertEncoder(JaxModule):
    """4-Layer JinaBert Encoder powered by the TPU v6e Fused Megakernel (`MAX_MODEL_LEN = 2048`)."""

    def __init__(
        self, config, dtype: jnp.dtype, rng: nnx.Rngs, mesh: Mesh, prefix: str
    ):
        self.mesh = mesh
        self.layer_norm_eps = float(config.layer_norm_eps)
        self.layer = JaxModuleList([
            JinaBertLayer(
                config, dtype, rng, mesh, prefix=f"{prefix}.layer.{i}"
            )
            for i in range(config.num_hidden_layers)
        ])
        first_self = getattr(self.layer[0].attention, "self")
        self.sm_scale = float(first_self.head_dim_original**-0.5)
        self.alibi_slopes = first_self.alibi_slopes

    def __call__(
        self, x: jax.Array, attention_metadata: AttentionMetadata
    ) -> jax.Array:
        if x.shape[0] > MAX_MODEL_LEN:
            raise ValueError(
                f"JinaBertEncoder on jina-v2-embeddings-clean strictly supports "
                f"max_model_len <= {MAX_MODEL_LEN}, got {x.shape[0]}"
            )
        w_qkv = jnp.stack(
            [
                jnp.stack(
                    [
                        getattr(L.attention, "self").query.weight.value,
                        getattr(L.attention, "self").key.weight.value,
                        getattr(L.attention, "self").value.weight.value,
                    ],
                    axis=1,
                )
                for L in self.layer
            ],
            axis=0,
        )
        b_qkv = jnp.stack(
            [
                jnp.stack(
                    [
                        getattr(L.attention, "self").query.bias.value,
                        getattr(L.attention, "self").key.bias.value,
                        getattr(L.attention, "self").value.bias.value,
                    ],
                    axis=0,
                )
                for L in self.layer
            ],
            axis=0,
        )
        w_o = jnp.stack(
            [L.attention.output.dense.weight.value for L in self.layer], axis=0
        )
        b_o = jnp.stack(
            [L.attention.output.dense.bias.value for L in self.layer], axis=0
        )
        ln1_scale = jnp.stack(
            [L.attention.output.LayerNorm.weight.value for L in self.layer],
            axis=0,
        )
        ln1_bias = jnp.stack(
            [L.attention.output.LayerNorm.bias.value for L in self.layer],
            axis=0,
        )
        w_gated = jnp.stack(
            [L.mlp.gated_layers.weight.value for L in self.layer], axis=0
        )
        w_down = jnp.stack([L.mlp.wo.weight.value for L in self.layer], axis=0)
        b_down = jnp.stack([L.mlp.wo.bias.value for L in self.layer], axis=0)
        ln2_scale = jnp.stack(
            [L.mlp.layernorm.weight.value for L in self.layer], axis=0
        )
        ln2_bias = jnp.stack(
            [L.mlp.layernorm.bias.value for L in self.layer], axis=0
        )

        alibi_arr = jnp.asarray(self.alibi_slopes, dtype=jnp.float32)
        return jina_v6e_4layer_megakernel(
            x=x,
            seq_lens=attention_metadata.seq_lens,
            w_qkv=w_qkv,
            b_qkv=b_qkv,
            w_o=w_o,
            b_o=b_o,
            ln1_scale=ln1_scale,
            ln1_bias=ln1_bias,
            w_gated=w_gated,
            w_down=w_down,
            b_down=b_down,
            ln2_scale=ln2_scale,
            ln2_bias=ln2_bias,
            alibi_slopes=alibi_arr,
            mesh=self.mesh,
            sm_scale=self.sm_scale,
            layer_norm_eps=self.layer_norm_eps,
        )


class JinaBertModel(JaxModule):

    def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs, mesh: Mesh):
        config = vllm_config.model_config.hf_config
        # Strictly clamp max_position_embeddings to 2048 on jina-v2-embeddings-clean
        config.max_position_embeddings = min(
            getattr(config, "max_position_embeddings", MAX_MODEL_LEN),
            MAX_MODEL_LEN,
        )
        if vllm_config.model_config.max_model_len > MAX_MODEL_LEN:
            raise ValueError(
                f"JinaBertModel on jina-v2-embeddings-clean strictly supports "
                f"max_model_len <= {MAX_MODEL_LEN}, got {vllm_config.model_config.max_model_len}"
            )
        dtype = vllm_config.model_config.dtype
        self.embeddings = JinaBertEmbeddings(config, dtype, rng)
        self.encoder = JinaBertEncoder(
            config, dtype, rng, mesh, prefix="encoder"
        )

    def __call__(
        self, input_ids: jax.Array, attention_metadata: AttentionMetadata
    ) -> jax.Array:
        x = self.embeddings(input_ids)
        return self.encoder(x, attention_metadata)


class JinaBertForMaskedLM(JaxModule, LoadableWithIterator):
    """Embedding-only JinaBert ("JinaBertForMaskedLM") with TPU v6e 2048-Capped Megakernel."""

    is_pooling_model = True

    def __init__(
        self, vllm_config: VllmConfig, rng_key: jax.Array, mesh: Mesh
    ) -> None:
        self.vllm_config = vllm_config
        self.mesh = mesh
        rng = nnx.Rngs(rng_key)
        self.model = JinaBertModel(vllm_config, rng, mesh)

    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: AttentionMetadata,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: Optional[JaxIntermediateTensors] = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> Tuple[
        List[jax.Array], jax.Array, List[jax.Array], Optional[jax.Array]
    ]:
        assert inputs_embeds is None, (
            "JinaBert does not support external input embeddings"
        )
        hidden_states = self.model(input_ids, attention_metadata)
        return kv_caches, hidden_states, [], None

    def load_weights(self, weights):
        from tpu_inference.models.jax.utils.weight_utils import (
            JaxAutoWeightsLoader,
        )
        from tpu_inference.utils import to_torch_dtype

        torch_dtype = to_torch_dtype(self.vllm_config.model_config.dtype)

        def _filtered(weights_iter):
            for name, weight in weights_iter:
                if name.startswith("bert."):
                    name = name[len("bert.") :]
                if name.startswith("cls.") or name.startswith("pooler."):
                    continue
                yield name, weight.to(torch_dtype)

        loader = JaxAutoWeightsLoader(self, skip_prefixes=None)
        return loader.load_weights(_filtered(weights))
