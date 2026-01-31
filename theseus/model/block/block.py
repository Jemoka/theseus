"""
Your most very basic indeed transformer Block
"""

import jax
import flax.linen as nn
from typing import Dict, Any

from theseus.model.attention import SelfAttention, RopeAttention
from theseus.model.layers.layer_norm import LayerNorm
from theseus.model.layers.mlp import MLP

from theseus.config import configure, field


class Block(nn.Module):
    rope: bool = field("architecture/rope", default=True)

    def setup(self) -> None:
        self.ln_1 = configure(LayerNorm)
        if self.rope:
            self.attn = configure(RopeAttention)
        else:
            self.attn = configure(SelfAttention)
        self.ln_2 = configure(LayerNorm)
        self.mlp = configure(MLP)

    def __call__(self, x: jax.Array, **kwargs: Dict[Any, Any]) -> jax.Array:
        x = x + self.attn(self.ln_1(x), **kwargs)
        x = x + self.mlp(self.ln_2(x))
        return x
