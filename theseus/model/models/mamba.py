"""Mamba-2 language model.

A pure selective state space model following the Mamba-2 architecture
(Dao & Gu, 2024).  No positional embeddings — position information
is implicit in the SSM recurrent state.

Inherits GPT's ``loss``, ``unembed``, and ``__call__`` (the
embed → decode → unembed → loss pipeline is identical).  Only
``setup``, ``embed``, and ``decode`` differ.
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
    and skip positional embeddings.  ``loss``, ``unembed``, and
    ``__call__`` are inherited unchanged.
    """

    # Override defaults for SSM-typical depth
    n_layers: int = field("architecture/n_layers", default=48)

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
        # Use RMSNorm (not LayerNorm) — matches Mamba-2 convention
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
        # Override to use RMSNorm (self.ln_f is RMSNorm, not LayerNorm)
        x = self.ln_f(x)
        logits = jnp.einsum("bth,vh->btv", x, self.wte.astype(self._activation_dtype))
        return logits
