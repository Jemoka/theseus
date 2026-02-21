from typing import Optional, List, Type, Any, Tuple

import jax
import flax.linen as nn
import jax.numpy as jnp

from theseus.config import field
from theseus.model.module import Module
from theseus.model.layers import LayerNorm, NeoXMLP
from theseus.model.attention.grouped_self_attention import GroupedSelfAttention


class GPTNeoXDecoderBlock(Module):
    n_layers: int = field("architecture/n_layers", default=24)
    n_embd: int = field("architecture/n_embd", default=2048)
    n_head: int = field("architecture/n_head", default=32)
    n_kv_head: int = field("architecture/n_kv_head", default=-1)
    intermediate_size: int = field("architecture/intermediate_size", default=8192)
    dropout: float = field("architecture/dropout", default=0.0)
    attn_dropout: float = field("architecture/attn_dropout", default=0.0)
    rope_theta: float = field("architecture/rope_theta", default=10000.0)
    partial_rotary_factor: float = field(
        "architecture/partial_rotary_factor", default=1.0
    )
    layer_norm_eps: float = field("architecture/layer_norm_eps", default=1e-5)
    use_parallel_residual: bool = field(
        "architecture/use_parallel_residual", default=True
    )
    bias: bool = field("architecture/bias", default=True)
    attention_bias: bool = field("architecture/attention_bias", default=True)
    hidden_act: str = field("architecture/hidden_act", default="gelu_new")

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [LayerNorm, GroupedSelfAttention, NeoXMLP]

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return []

    def setup(self) -> None:
        self.ln_1 = LayerNorm(
            self.n_embd, self.bias, eps=self.layer_norm_eps, dtype=jnp.float32
        )
        self.attn = GroupedSelfAttention(
            n_embd=self.n_embd,
            n_layers=self.n_layers,
            n_head=self.n_head,
            n_kv_head=self.n_kv_head,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            rope_theta=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            use_sliding_window=False,
            attn_bias=self.attention_bias,
        )
        self.ln_2 = LayerNorm(
            self.n_embd, self.bias, eps=self.layer_norm_eps, dtype=jnp.float32
        )
        self.mlp = NeoXMLP(
            n_embd=self.n_embd,
            n_layers=self.n_layers,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            hidden_act=self.hidden_act,
            bias=self.bias,
        )
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        positions: Optional[jax.Array] = None,
    ) -> jax.Array:
        attn_out = self.attn(
            self.ln_1(x),
            padding_mask=padding_mask,
            deterministic=deterministic,
            sliding=False,
            positions=positions,
        )
        attn_out = self.dropout_layer(attn_out, deterministic=deterministic)

        if self.use_parallel_residual:
            mlp_out = self.mlp(self.ln_2(x), deterministic=deterministic)
            mlp_out = self.dropout_layer(mlp_out, deterministic=deterministic)
            x = x + attn_out + mlp_out
        else:
            x = x + attn_out
            mlp_out = self.mlp(self.ln_2(x), deterministic=deterministic)
            mlp_out = self.dropout_layer(mlp_out, deterministic=deterministic)
            x = x + mlp_out
        return x
