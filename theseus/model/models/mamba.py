"""Mamba-2 language model.

A pure selective state space model following the Mamba-2 architecture
(Dao & Gu, 2024).  No positional embeddings — position information
is implicit in the SSM recurrent state.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Any, List, Optional, Tuple, Type

from theseus.model.block.mamba import MambaBlock
from theseus.model.layers.rmsnorm import RMSNorm
from theseus.model.axes import Axes
from theseus.model.module import Module
from theseus.config import field, configure
from theseus.base.axis import Axis


class Mamba(Module):
    n_layers: int = field("architecture/n_layers", default=48)
    n_embd: int = field("architecture/n_embd", default=2048)
    block_size: int = field("architecture/block_size", default=2048)
    dropout: float = field("architecture/dropout", default=0.0)
    vocab_size: int = field("architecture/vocab_size", default=100288)

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return [
            (Axes.VOCAB.value, None),
            (Axes.N_EMBD.value, None),
            (Axes.N_EMBD_FF.value, Axis.SHARD),
            (Axes.N_SSM.value, Axis.SHARD),
        ]

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [MambaBlock, RMSNorm]

    def setup(self) -> None:
        assert self.vocab_size is not None

        self.wte: jax.Array = self.param(
            "wte",
            nn.with_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.VOCAB.value, Axes.N_EMBD.value),
            ),
            (self.vocab_size, self.n_embd),
            self._param_dtype,
        )  # type: ignore

        self.drop = nn.Dropout(rate=self.dropout)
        self.blocks = [configure(MambaBlock) for _ in range(self.n_layers)]
        self.ln_f = configure(RMSNorm)

    def embed(self, idx: jax.Array, deterministic: bool = False, **kwargs: Any) -> Any:
        """Token embeddings only — no positional encoding needed for SSMs."""
        x = jnp.take(self.wte, idx, axis=0).astype(self._activation_dtype)
        x = self.drop(x, deterministic=deterministic)
        return x

    def decode(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> Any:
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask, deterministic=deterministic)
        return x

    def unembed(self, x: jax.Array) -> Any:
        x = self.ln_f(x)
        logits = jnp.einsum("bth,vh->btv", x, self.wte.astype(self._activation_dtype))
        return logits

    def loss(self, logits: jax.Array, targets: jax.Array) -> jax.Array:
        logits_f32 = logits.astype(jnp.float32)
        logits_flat = logits_f32.reshape(-1, logits_f32.shape[-1])
        targets_flat = targets.reshape(-1)

        mask = targets_flat != -1
        targets_masked = jnp.where(mask, targets_flat, 0)

        loss = -jnp.sum(
            jax.nn.log_softmax(logits_flat, axis=-1)
            * jax.nn.one_hot(targets_masked, self.vocab_size)
            * mask[:, None]
        ) / mask.sum().clip(min=1)

        return loss

    def __call__(
        self,
        idx: jax.Array,
        targets: Optional[jax.Array] = None,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> Tuple[jax.Array, Optional[jax.Array]]:
        b, t = idx.shape
        assert t <= self.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        )

        x = self.embed(idx, deterministic, **kwargs)
        x = self.decode(
            x, padding_mask=padding_mask, deterministic=deterministic, **kwargs
        )
        logits = self.unembed(x)

        if targets is not None:
            loss = self.loss(logits, targets)
        else:
            loss = None

        return logits, loss
