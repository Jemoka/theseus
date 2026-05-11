"""LaCT language model.

Test-Time Training Done Right (arXiv:2505.23884).  A GPT-style transformer
where every block is a ``LaCTBlock`` (SWA + TTT + FFN).  Inherits ``embed``,
``unembed``, ``loss``, and ``__call__`` from ``GPT`` unchanged; only ``setup``,
``components``, and ``sharding`` are new.

The fast-weight machinery lives inside ``LaCTBlock``; this class just stitches
the embedding / residual stream / unembedding around it.
"""

import jax
import flax.linen as nn

from typing import Any, List, Optional, Tuple, Type

from theseus.base.axis import Axis
from theseus.config import configure
from theseus.model.axes import Axes
from theseus.model.block.lact import LaCTBlock
from theseus.model.layers.layernorm import LayerNorm
from theseus.model.models.base import GPT


class LaCT(GPT):
    """LaCT language model — SWA + chunked-TTT layers throughout."""

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return [
            (Axes.VOCAB.value, None),
            (Axes.BLOCK_SIZE.value, None),
            (Axes.N_EMBD.value, None),
            (Axes.N_EMBD_FF.value, Axis.SHARD),
            (Axes.N_EMBD_OUT.value, Axis.SHARD),
            (Axes.N_ATTN.value, Axis.SHARD),
            (Axes.N_FW.value, Axis.SHARD),
        ]

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [LaCTBlock, LayerNorm]

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

        # RoPE is handled inside each block's SWA sublayer, so no wpe table here.
        self.drop = nn.Dropout(rate=self.dropout)
        self.blocks = [configure(LaCTBlock) for _ in range(self.n_layers)]
        self.ln_f = configure(LayerNorm)

    def embed(self, idx: jax.Array, deterministic: bool = False, **kwargs: Any) -> Any:
        """Token embeddings only — RoPE handles positions inside each block."""
        x = jax.numpy.take(self.wte, idx, axis=0).astype(self._activation_dtype)
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
            x = block(
                x, padding_mask=padding_mask, deterministic=deterministic, **kwargs
            )
        return x
