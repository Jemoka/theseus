"""Hybrid Transformer + Mamba model.

Interleaves standard transformer (attention) blocks with Mamba-2 SSM
blocks, following the Jamba architecture pattern (Lieber et al., 2024).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Any, List, Optional, Tuple, Type

from theseus.model.block import Block
from theseus.model.block.mamba import MambaBlock
from theseus.model.layers import LayerNorm
from theseus.model.layers.rmsnorm import RMSNorm
from theseus.model.axes import Axes
from theseus.model.module import Module
from theseus.config import field, configure
from theseus.base.axis import Axis


class Hybrid(Module):
    """Hybrid Transformer + Mamba language model.

    ``mamba_layers`` controls which layers use Mamba blocks:
    - ``"even"``: even-indexed layers (0, 2, 4, ...) are Mamba
    - ``"odd"``: odd-indexed layers (1, 3, 5, ...) are Mamba
    - comma-separated indices: e.g. ``"0,2,4,6"`` for explicit control
    """

    n_layers: int = field("architecture/n_layers", default=32)
    n_embd: int = field("architecture/n_embd", default=2048)
    rope: bool = field("architecture/rope", default=True)
    block_size: int = field("architecture/block_size", default=2048)
    dropout: float = field("architecture/dropout", default=0.0)
    vocab_size: int = field("architecture/vocab_size", default=100288)
    mamba_layers: str = field("architecture/mamba_layers", default="even")

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return [
            (Axes.VOCAB.value, None),
            (Axes.BLOCK_SIZE.value, None),
            (Axes.N_EMBD.value, None),
            (Axes.N_EMBD_FF.value, Axis.SHARD),
            (Axes.N_EMBD_OUT.value, Axis.SHARD),
            (Axes.N_ATTN.value, Axis.SHARD),
            (Axes.N_SSM.value, Axis.SHARD),
        ]

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [Block, MambaBlock, LayerNorm, RMSNorm]

    def _parse_mamba_layers(self) -> set[int]:
        """Parse the mamba_layers config into a set of layer indices."""
        spec = self.mamba_layers.strip().lower()
        if spec == "even":
            return {i for i in range(self.n_layers) if i % 2 == 0}
        elif spec == "odd":
            return {i for i in range(self.n_layers) if i % 2 == 1}
        else:
            return {int(s.strip()) for s in spec.split(",") if s.strip()}

    def setup(self) -> None:
        assert self.vocab_size is not None

        mamba_indices = self._parse_mamba_layers()

        self.wte: jax.Array = self.param(
            "wte",
            nn.with_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.VOCAB.value, Axes.N_EMBD.value),
            ),
            (self.vocab_size, self.n_embd),
            self._param_dtype,
        )  # type: ignore

        # Positional embedding only used by transformer blocks (via RoPE in Block)
        if not self.rope:
            self.wpe: jax.Array = self.param(
                "wpe",
                nn.with_partitioning(
                    nn.initializers.normal(stddev=0.02),
                    (Axes.BLOCK_SIZE.value, Axes.N_EMBD.value),
                ),
                (self.block_size, self.n_embd),
                self._param_dtype,
            )  # type: ignore

        self.drop = nn.Dropout(rate=self.dropout)

        self.blocks = [
            configure(MambaBlock) if i in mamba_indices else configure(Block)
            for i in range(self.n_layers)
        ]

        self.ln_f = configure(RMSNorm)

    def embed(self, idx: jax.Array, deterministic: bool = False, **kwargs: Any) -> Any:
        _, t = idx.shape
        x = jnp.take(self.wte, idx, axis=0).astype(self._activation_dtype)

        if not self.rope:
            pos = jnp.arange(0, t)
            x = x + jnp.take(self.wpe, pos, axis=0).astype(self._activation_dtype)

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
