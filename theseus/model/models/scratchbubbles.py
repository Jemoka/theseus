"""
Scratchubbbles: thoughtbubbles except we can fork into any other token.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.model.axes import Axes
from theseus.model.layers import LayerNorm
from theseus.model.block.forking import ThoughtBlock
from theseus.model.block.scratching import ScratchingBlock
from theseus.model.models.thoughtbubbles import Thoughtbubbles
from theseus.config import configure

from typing import List, Any, Type


class Scratchbubbles(Thoughtbubbles):
    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [ThoughtBlock, ScratchingBlock, LayerNorm]

    def setup(self) -> None:
        assert self.vocab_size is not None
        assert self.block_size is not None

        # Token embedding table (no positional - using RoPE in attention)
        self.wte: jax.Array = self.param(
            "wte",
            nn.with_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.VOCAB.value, Axes.N_EMBD.value),
            ),
            (self.vocab_size, self.n_embd),
            jnp.float32,
        )  # type: ignore

        self.drop = nn.Dropout(rate=self.dropout)

        # Create blocks: ForkingBlock for layers in fork list, ThoughtBlock otherwise
        fork_set = set(self.fork)
        self.blocks = [
            configure(ScratchingBlock) if i in fork_set else configure(ThoughtBlock)
            for i in range(self.n_layers)
        ]

        self.ln_f = configure(LayerNorm)
