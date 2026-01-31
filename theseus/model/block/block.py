"""
Your most very basic indeed transformer Block
"""

import jax
from typing import Dict, Any, List, Type

from theseus.model.attention import SelfAttention, RopeAttention
from theseus.model.layers.layernorm import LayerNorm
from theseus.model.layers.mlp import MLP
from theseus.model.module import Module

from theseus.config import configure, field


class Block(Module):
    rope: bool = field("architecture/rope", default=True)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [LayerNorm, SelfAttention, RopeAttention, MLP]

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
