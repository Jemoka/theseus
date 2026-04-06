"""Mixture-of-experts transformer block."""

from typing import Any, List, Type

import jax

from theseus.config import configure, field
from theseus.model.attention import RopeAttention, SelfAttention
from theseus.model.block.block import Block
from theseus.model.layers.layernorm import LayerNorm
from theseus.model.moe import BiasBalancedMoE, MoE


class MoEBlock(Block):
    moe_implementation: str = field("architecture/moe/implementation", default="base")

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [LayerNorm, SelfAttention, RopeAttention, MoE, BiasBalancedMoE]

    def _moe_impl(self) -> Type[MoE]:
        implementations: dict[str, Type[MoE]] = {
            "base": MoE,
            "bias_balanced": BiasBalancedMoE,
            "deepseek": BiasBalancedMoE,
        }
        if self.moe_implementation not in implementations:
            raise ValueError(
                "Unknown architecture/moe/implementation "
                f"{self.moe_implementation!r}; expected one of "
                f"{sorted(implementations)}"
            )
        return implementations[self.moe_implementation]

    def setup(self) -> None:
        self.ln_1 = configure(LayerNorm)
        if self.rope:
            self.attn = configure(RopeAttention)
        else:
            self.attn = configure(SelfAttention)
        self.ln_2 = configure(LayerNorm)
        self.mlp = configure(self._moe_impl())

    def __call__(self, x: jax.Array, **kwargs: Any) -> jax.Array:
        deterministic = kwargs.get("deterministic", False)
        x = x + self.attn(self.ln_1(x), **kwargs)
        x = x + self.mlp(self.ln_2(x), deterministic=deterministic)
        return x
