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

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from vllm.config import VllmConfig

from tpu_inference import utils
from tpu_inference.kernels.jina_v6e_megakernel import (
    MAX_MODEL_LEN,
    jina_v6e_4layer_megakernel,
)
from tpu_inference.layers.common.attention_interface import \
    encoder_only_attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax import JaxModule, JaxModuleList
from tpu_inference.layers.jax.embed import JaxEmbed
from tpu_inference.layers.jax.linear import JaxEinsum, JaxLinear
from tpu_inference.layers.jax.norm import JaxLayerNorm
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.weight_utils import (
    LoadableWithIterator, load_nnx_param_from_reshaped_torch)

logger = init_logger(__name__)

init_fn = nnx.initializers.uniform()


def get_alibi_slopes(n_heads: int) -> List[float]:
    """Standard ALiBi head slopes (geometric sequence), incl. non-power-of-2."""

    def slopes_power_of_2(n):
        start = 2**(-(2**-(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        return slopes_power_of_2(n_heads)
    closest_power_of_2 = 2**math.floor(math.log2(n_heads))
    return (slopes_power_of_2(closest_power_of_2) +
            get_alibi_slopes(2 * closest_power_of_2)[0::2][:n_heads -
                                                           closest_power_of_2])


def _set_weight_loader(param: nnx.Param,
                       param_name: str,
                       reshape_dims: Optional[Tuple[int, ...]] = None,
                       permute_dims: Optional[Tuple[int, ...]] = None) -> None:
    """Attach an explicit HF->JAX weight loader to a param."""
    param.set_metadata(
        "weight_loader",
        functools.partial(load_nnx_param_from_reshaped_torch,
                          reshape_dims=reshape_dims,
                          permute_dims=permute_dims,
                          param_name=param_name))


class JinaBertEmbeddings(JaxModule):
    """word + token_type embeddings -> LayerNorm. No position embeddings."""

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs):
        self.word_embeddings = JaxEmbed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, ("model", None)),
            rngs=rng,
        )
        self.token_type_embeddings = JaxEmbed(
            num_embeddings=config.type_vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            embedding_init=nnx.with_partitioning(init_fn, (None, None)),
            rngs=rng,
        )
        self.LayerNorm = JaxLayerNorm(
            num_features=config.hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            rngs=rng,
        )
        _set_weight_loader(self.word_embeddings.weight,
                           "embeddings.word_embeddings.weight",
                           permute_dims=(0, 1))
        _set_weight_loader(self.token_type_embeddings.weight,
                           "embeddings.token_type_embeddings.weight",
                           permute_dims=(0, 1))

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        x = self.word_embeddings(input_ids)
        x = x + self.token_type_embeddings.weight.value[0]
        return self.LayerNorm(x)


class JinaBertSelfAttention(JaxModule):
    """QKV projections + encoder-only flash attention with ALiBi bias."""

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, mesh: Mesh,
                 prefix: str):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim_original = self.hidden_size // self.num_heads
        self.head_dim = utils.get_padded_head_dim(self.head_dim_original)
        sharding_size = mesh.shape["model"]
        self.num_heads = utils.get_padded_num_heads(self.num_heads,
                                                    sharding_size)
        self.mesh = mesh

        def qkv(name):
            proj = JaxEinsum(
                "TD,DNH->TNH",
                (self.hidden_size, self.num_heads, self.head_dim),
                bias_shape=(self.num_heads, self.head_dim),
                dtype=dtype,
                kernel_init=nnx.with_partitioning(init_fn,
                                                  (None, "model", None)),
                bias_init=nnx.with_partitioning(init_fn, ("model", None)),
                rngs=rng,
                prefix=f"{prefix}.{name}",
            )
            _set_weight_loader(proj.weight,
                               f"{prefix}.{name}.weight",
                               reshape_dims=(self.num_heads, self.head_dim,
                                             self.hidden_size),
                               permute_dims=(2, 0, 1))
            _set_weight_loader(proj.bias,
                               f"{prefix}.{name}.bias",
                               reshape_dims=(self.num_heads, self.head_dim),
                               permute_dims=(0, 1))
            return proj

        self.query = qkv("query")
        self.key = qkv("key")
        self.value = qkv("value")
        self.alibi_slopes = tuple(get_alibi_slopes(self.num_heads))

    def __call__(self, x: jax.Array,
                 attention_metadata: AttentionMetadata) -> jax.Array:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        return encoder_only_attention(
            q,
            k,
            v,
            attention_metadata,
            self.mesh,
            sm_scale=self.head_dim_original**-0.5,
            alibi_slopes=jnp.asarray(self.alibi_slopes, dtype=jnp.float32),
        )


class JinaBertSelfOutput(JaxModule):
    """attention output dense + residual + post-LayerNorm."""

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs,
                 num_heads: int, head_dim: int, prefix: str):
        hidden_size = config.hidden_size
        self.dense = JaxEinsum(
            "TNH,NHD->TD",
            (num_heads, head_dim, hidden_size),
            bias_shape=(hidden_size, ),
            dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, ("model", None, None)),
            bias_init=nnx.with_partitioning(init_fn, (None, )),
            rngs=rng,
            prefix=prefix + ".dense",
        )
        _set_weight_loader(self.dense.weight,
                           f"{prefix}.dense.weight",
                           reshape_dims=(hidden_size, num_heads, head_dim),
                           permute_dims=(1, 2, 0))
        self.LayerNorm = JaxLayerNorm(
            num_features=hidden_size,
            epsilon=config.layer_norm_eps,
            dtype=dtype,
            rngs=rng,
        )

    def __call__(self, x: jax.Array, residual: jax.Array) -> jax.Array:
        return self.LayerNorm(self.dense(x) + residual)


class JinaBertAttention(JaxModule):

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, mesh: Mesh,
                 prefix: str):
        self_attention = JinaBertSelfAttention(config,
                                               dtype,
                                               rng,
                                               mesh,
                                               prefix=prefix + ".self")
        setattr(self, "self", self_attention)
        self.output = JinaBertSelfOutput(
            config,
            dtype,
            rng,
            num_heads=self_attention.num_heads,
            head_dim=self_attention.head_dim,
            prefix=prefix + ".output",
        )

    def __call__(self, x: jax.Array,
                 attention_metadata: AttentionMetadata) -> jax.Array:
        attn = getattr(self, "self")(x, attention_metadata)
        return self.output(attn, x)


class JinaBertGLUMLP(JaxModule):
    """GEGLU MLP with internal residual + post-LayerNorm."""

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, prefix: str):
        hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        assert getattr(config, "feed_forward_type", "geglu") == "geglu", (
            "Only feed_forward_type='geglu' is supported")
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
            bias_init=nnx.with_partitioning(init_fn, (None, )),
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
        gated = h[..., :self.intermediate_size]
        non_gated = h[..., self.intermediate_size:]
        h = jax.nn.gelu(gated, approximate=False) * non_gated
        h = self.wo(h)
        return self.layernorm(h + residual)


class JinaBertLayer(JaxModule):

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, mesh: Mesh,
                 prefix: str):
        self.attention = JinaBertAttention(config,
                                           dtype,
                                           rng,
                                           mesh,
                                           prefix=prefix + ".attention")
        self.mlp = JinaBertGLUMLP(config, dtype, rng, prefix=prefix + ".mlp")

    def __call__(self, x: jax.Array,
                 attention_metadata: AttentionMetadata) -> jax.Array:
        x = self.attention(x, attention_metadata)
        return self.mlp(x)


class JinaBertEncoder(JaxModule):
    """4-Layer JinaBert Encoder powered by the TPU v6e Fused Megakernel (`MAX_MODEL_LEN = 2048`)."""

    def __init__(self, config, dtype: jnp.dtype, rng: nnx.Rngs, mesh: Mesh,
                 prefix: str):
        self.mesh = mesh
        self.layer_norm_eps = float(config.layer_norm_eps)
        self.layer = JaxModuleList([
            JinaBertLayer(config,
                          dtype,
                          rng,
                          mesh,
                          prefix=f"{prefix}.layer.{i}")
            for i in range(config.num_hidden_layers)
        ])
        first_self = getattr(self.layer[0].attention, "self")
        self.sm_scale = float(first_self.head_dim_original**-0.5)
        self.alibi_slopes = first_self.alibi_slopes

        # Precompute [1, num_heads, 2048, 2048] symmetric ALiBi bias ONCE at init time
        slopes_arr = jnp.asarray(self.alibi_slopes, dtype=jnp.float32)
        pos = jnp.arange(MAX_MODEL_LEN, dtype=jnp.int32)
        distance = jnp.abs(pos[:, None] - pos[None, :]).astype(jnp.float32)
        ab_2048 = jnp.expand_dims(
            -slopes_arr[:, None, None] * (distance[None, :, :] / self.sm_scale),
            axis=0,
        )
        ab_sharded = jax.device_put(
            ab_2048,
            NamedSharding(mesh, P(None, "model", None, None)),
        )
        self.ab_full = nnx.Variable(ab_sharded)

    def __call__(self, x: jax.Array,
                 attention_metadata: AttentionMetadata) -> jax.Array:
        if x.shape[0] > MAX_MODEL_LEN:
            raise ValueError(
                f"JinaBertEncoder on jina-v2-embeddings-clean strictly supports "
                f"max_model_len <= {MAX_MODEL_LEN}, got {x.shape[0]}"
            )
        w_qkv = jnp.stack([
            jnp.stack([
                getattr(L.attention, "self").query.weight.value,
                getattr(L.attention, "self").key.weight.value,
                getattr(L.attention, "self").value.weight.value,
            ], axis=1)
            for L in self.layer
        ], axis=0)
        b_qkv = jnp.stack([
            jnp.stack([
                getattr(L.attention, "self").query.bias.value,
                getattr(L.attention, "self").key.bias.value,
                getattr(L.attention, "self").value.bias.value,
            ], axis=0)
            for L in self.layer
        ], axis=0)
        w_o = jnp.stack(
            [L.attention.output.dense.weight.value for L in self.layer],
            axis=0)
        b_o = jnp.stack(
            [L.attention.output.dense.bias.value for L in self.layer],
            axis=0)
        ln1_scale = jnp.stack(
            [L.attention.output.LayerNorm.weight.value for L in self.layer],
            axis=0)
        ln1_bias = jnp.stack(
            [L.attention.output.LayerNorm.bias.value for L in self.layer],
            axis=0)
        w_gated = jnp.stack(
            [L.mlp.gated_layers.weight.value for L in self.layer], axis=0)
        w_down = jnp.stack(
            [L.mlp.wo.weight.value for L in self.layer], axis=0)
        b_down = jnp.stack(
            [L.mlp.wo.bias.value for L in self.layer], axis=0)
        ln2_scale = jnp.stack(
            [L.mlp.layernorm.weight.value for L in self.layer], axis=0)
        ln2_bias = jnp.stack(
            [L.mlp.layernorm.bias.value for L in self.layer], axis=0)

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
            ab_full=self.ab_full.value,
            mesh=self.mesh,
            sm_scale=self.sm_scale,
            layer_norm_eps=self.layer_norm_eps,
        )


class JinaBertModel(JaxModule):

    def __init__(self, vllm_config: VllmConfig, rng: nnx.Rngs, mesh: Mesh):
        config = vllm_config.model_config.hf_config
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
        self.encoder = JinaBertEncoder(config,
                                       dtype,
                                       rng,
                                       mesh,
                                       prefix="encoder")

    def __call__(self, input_ids: jax.Array,
                 attention_metadata: AttentionMetadata) -> jax.Array:
        x = self.embeddings(input_ids)
        return self.encoder(x, attention_metadata)


class JinaBertForMaskedLM(JaxModule, LoadableWithIterator):
    """Embedding-only JinaBert ("JinaBertForMaskedLM") with TPU v6e 2048-Capped Megakernel."""

    is_pooling_model = True

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
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
    ) -> Tuple[List[jax.Array], jax.Array, List[jax.Array],
               Optional[jax.Array]]:
        assert inputs_embeds is None, (
            "JinaBert does not support external input embeddings")
        hidden_states = self.model(input_ids, attention_metadata)
        return kv_caches, hidden_states, [], None

    def load_weights(self, weights):
        from tpu_inference.models.jax.utils.weight_utils import \
            JaxAutoWeightsLoader
        from tpu_inference.utils import to_torch_dtype

        torch_dtype = to_torch_dtype(self.vllm_config.model_config.dtype)

        def _filtered(weights_iter):
            for name, weight in weights_iter:
                if name.startswith("bert."):
                    name = name[len("bert."):]
                if name.startswith("cls.") or name.startswith("pooler."):
                    continue
                yield name, weight.to(torch_dtype)

        loader = JaxAutoWeightsLoader(self, skip_prefixes=None)
        return loader.load_weights(_filtered(weights))
