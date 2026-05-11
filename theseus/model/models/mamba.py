"""Mamba-2 language model.

A pure selective state space model following the Mamba-2 architecture
(Dao & Gu, 2024).  No positional embeddings — position information
is implicit in the SSM recurrent state.

Inherits GPT's ``loss`` and ``unembed``.  Overrides ``__call__`` to drop
the block-size assertion (SSMs have no fixed context-length limit), and
overrides ``setup``, ``embed``, and ``decode`` for SSM-specific structure.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Any, List, Optional, Tuple, Type

from theseus.model.block.mamba import MambaBlock
from theseus.model.layers.rmsnorm import RMSNorm
from theseus.model.axes import Axes
from theseus.model.models.base import GPT
from theseus.config import field, configure
from theseus.base.axis import Axis


class Mamba(GPT):
    """Mamba-2 language model — SSM-only, no attention.

    Overrides GPT's setup/embed/decode to use MambaBlock layers
    and skip positional embeddings.  ``loss`` and ``unembed`` are
    inherited unchanged.
    """

    # Override defaults for SSM-typical depth
    n_layers: int = field("architecture/n_layers", default=48)

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return [
            (Axes.VOCAB.value, None),
            (Axes.N_EMBD.value, None),
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

    def __call__(
        self,
        idx: jax.Array,
        targets: Optional[jax.Array] = None,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> Tuple[jax.Array, Optional[jax.Array]]:
        # SSMs have no fixed context-length limit — skip the block_size assertion.
        x = self.embed(idx, deterministic, **kwargs)
        x = self.decode(
            x, padding_mask=padding_mask, deterministic=deterministic, **kwargs
        )
        logits = self.unembed(x)
        loss = self.loss(logits, targets) if targets is not None else None
        return logits, loss
