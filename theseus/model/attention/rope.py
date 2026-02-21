"""
Attention module with Rotary Positional Encoding (RoPE).
"""

import jax
from typing import Any, Tuple

from theseus.model.layers.rope import RotaryPosEncoding
from theseus.model.attention.base import SelfAttention


class RopeAttention(SelfAttention):
    def setup(self) -> None:
        super().setup()
        # seq_dim=1 for (B, T, H, D) convention
        self.rope = RotaryPosEncoding(self.head_dim, seq_dim=1)

    def preprocess_qkv(
        self, q: jax.Array, k: jax.Array, v: jax.Array, **kwargs: Any
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        q, k = self.rope(q, k, t=kwargs.get("positions"))
        return q, k, v
