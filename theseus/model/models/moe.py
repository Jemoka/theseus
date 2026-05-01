from typing import Any, List, Type

import flax.linen as nn
import jax

from theseus.config import configure
from theseus.model.axes import Axes
from theseus.model.block import MoEBlock
from theseus.model.layers import LayerNorm
from theseus.model.models.base import GPT


class MoEGPT(GPT):
    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [MoEBlock, LayerNorm]

    def setup(self) -> None:
        assert self.vocab_size is not None
        assert self.block_size is not None

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
        self.blocks = [configure(MoEBlock) for _ in range(self.n_layers)]
        self.ln_f = configure(LayerNorm)
