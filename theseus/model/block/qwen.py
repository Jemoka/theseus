from typing import Optional, List, Type, Any, Tuple

import jax

from theseus.config import field
from theseus.model.module import Module
from theseus.model.layers import RMSNorm, QwenMLP
from theseus.model.attention.grouped import GroupedSelfAttention


class QwenDecoderBlock(Module):
    n_layers: int = field("architecture/n_layers", default=32)
    n_embd: int = field("architecture/n_embd", default=4096)
    n_head: int = field("architecture/n_head", default=32)
    n_kv_head: int = field("architecture/n_kv_head", default=-1)
    intermediate_size: int = field("architecture/intermediate_size", default=22016)
    dropout: float = field("architecture/dropout", default=0.0)
    attn_dropout: float = field("architecture/attn_dropout", default=0.0)
    rope_theta: float = field("architecture/rope_theta", default=1e6)
    rms_norm_eps: float = field("architecture/rms_norm_eps", default=1e-6)
    use_sliding_window: bool = field("architecture/use_sliding_window", default=False)
    sliding_window: int = field("architecture/sliding_window", default=-1)
    bias: bool = field("architecture/bias", default=False)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [RMSNorm, GroupedSelfAttention, QwenMLP]

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return []

    def setup(self) -> None:
        self.rms_1 = RMSNorm(
            ndim=self.n_embd,
            eps=self.rms_norm_eps,
            param_dtype=self.param_dtype,
            activation_dtype=self.activation_dtype,
        )
        self.attn = GroupedSelfAttention(
            n_embd=self.n_embd,
            n_layers=self.n_layers,
            n_head=self.n_head,
            n_kv_head=self.n_kv_head,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            rope_theta=self.rope_theta,
            use_sliding_window=self.use_sliding_window,
            sliding_window=self.sliding_window,
            param_dtype=self.param_dtype,
            activation_dtype=self.activation_dtype,
        )
        self.rms_2 = RMSNorm(
            ndim=self.n_embd,
            eps=self.rms_norm_eps,
            param_dtype=self.param_dtype,
            activation_dtype=self.activation_dtype,
        )
        self.mlp = QwenMLP(
            n_embd=self.n_embd,
            n_layers=self.n_layers,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            bias=self.bias,
            param_dtype=self.param_dtype,
            activation_dtype=self.activation_dtype,
        )

    def __call__(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        sliding: bool = False,
        positions: Optional[jax.Array] = None,
    ) -> jax.Array:
        h = self.rms_1(x)
        h = self.attn(
            h,
            padding_mask=padding_mask,
            deterministic=deterministic,
            sliding=sliding,
            positions=positions,
        )
        x = x + h

        h = self.rms_2(x)
        h = self.mlp(h, deterministic=deterministic)
        x = x + h
        return x
