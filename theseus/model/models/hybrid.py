"""Hybrid Transformer + Mamba model.

Interleaves standard transformer (attention) blocks with Mamba-2 SSM
blocks, following the Jamba architecture pattern (Lieber et al., 2024).

Inherits GPT's ``loss``, ``unembed``, and ``__call__``.  Only
``setup``, ``components``, ``sharding``, and ``_parse_mamba_layers``
are new; ``embed`` and ``decode`` are minor overrides.
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
from theseus.model.models.base import GPT
from theseus.config import field, configure
from theseus.base.axis import Axis


class Hybrid(GPT):
    """Hybrid Transformer + Mamba language model.

    ``mamba_layers`` controls which layers use Mamba blocks:
    - ``"even"``: even-indexed layers (0, 2, 4, ...) are Mamba
    - ``"odd"``: odd-indexed layers (1, 3, 5, ...) are Mamba
    - comma-separated indices: e.g. ``"0,2,4,6"`` for explicit control
    """

    mamba_layers: str = field("architecture/mamba_layers", default="even")

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        # Union of transformer + SSM sharding axes
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

        # Use RMSNorm for final layer norm (consistent with Mamba layers)
        self.ln_f = configure(RMSNorm)

    def unembed(self, x: jax.Array) -> Any:
        # Override to use RMSNorm (self.ln_f is RMSNorm, not LayerNorm)
        x = self.ln_f(x)
        logits = jnp.einsum("bth,vh->btv", x, self.wte.astype(self._activation_dtype))
        return logits
