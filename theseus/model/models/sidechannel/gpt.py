"""
SideChannelGPT: GPT with cross-attention side channels.

Selected layers (configured via cross_attn_layers) use SideChannelBlock
instead of standard Block. A lightweight SideChannelEncoder compresses raw
side-channel tokens into fixed-size latent vectors via learned-query
cross-attention + MLP.
"""

from typing import Optional, Tuple, List, Any, Type

import jax
import jax.numpy as jnp
import flax.linen as nn

from theseus.model.models.base import GPT
from theseus.model.block import Block
from theseus.model.block.sidechannel import SideChannelBlock
from theseus.model.attention.perceiver import SideChannelEncoder
from theseus.model.layers import LayerNorm
from theseus.model.axes import Axes

from theseus.config import field, configure


class SideChannelGPT(GPT):
    """GPT with cross-attention side channels.

    Side-channel tokens are processed by a lightweight encoder (learned queries
    cross-attend to embedded tokens + MLP), then injected via single-head
    gated cross-attention at selected layers.

    Tanh gates initialized to 0 => vanilla GPT behavior at init.
    """

    n_channels: int = field("architecture/sidechannel/n_channels", default=4)
    n_latents: int = field("architecture/sidechannel/n_latents", default=128)
    cross_attn_layers: List[int] = field(
        "architecture/sidechannel/cross_attn_layers",
        default_factory=lambda: [3, 7, 11, 15, 19, 23, 27, 31],
    )

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [Block, SideChannelBlock, SideChannelEncoder, LayerNorm]

    def setup(self) -> None:
        # NOTE: cannot call super().setup() because Flax forbids re-assigning
        # named submodules (blocks_0, etc.) once registered by the parent.
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
        self.encoder = configure(SideChannelEncoder)

        cross_set = set(self.cross_attn_layers)
        self.blocks = [
            configure(SideChannelBlock) if i in cross_set else configure(Block)
            for i in range(self.n_layers)
        ]
        self.ln_f = configure(LayerNorm)

    def encode_channels(
        self, sidechannel: jax.Array, deterministic: bool = False
    ) -> jax.Array:
        """Encode raw side-channel token IDs. Returns (B, N, K, C)."""
        B, N, L = sidechannel.shape
        flat_tokens = sidechannel.reshape(B * N, L)
        flat_embedded = jnp.take(self.wte, flat_tokens, axis=0).astype(
            self._activation_dtype
        )
        flat_compressed = self.encoder(flat_embedded, deterministic=deterministic)
        return flat_compressed.reshape(  # type: ignore[no-any-return]
            B, N, self.n_latents, self.n_embd
        )

    def decode(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Process through transformer blocks with cross-attention side channels."""
        channel_states = kwargs.get("channel_states")
        channel_mask = kwargs.get("channel_mask")

        for block in self.blocks:
            if isinstance(block, SideChannelBlock):
                x = block(
                    x,
                    channel_states=channel_states,
                    channel_mask=channel_mask,
                    padding_mask=padding_mask,
                    deterministic=deterministic,
                )
            else:
                x = block(x, padding_mask=padding_mask, deterministic=deterministic)
        return x

    def __call__(
        self,
        idx: jax.Array,
        targets: Optional[jax.Array] = None,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        sidechannel: Optional[jax.Array] = None,
        sidechannel_mask: Optional[jax.Array] = None,
        **kwargs: Any,
    ) -> Tuple[jax.Array, Optional[jax.Array]]:
        b, t = idx.shape
        assert t <= self.block_size

        x = self.embed(idx, deterministic, **kwargs)

        if sidechannel is None:
            sidechannel = jnp.zeros((b, self.n_channels, 1), dtype=idx.dtype)
        channel_states = self.encode_channels(sidechannel, deterministic)

        x = self.decode(
            x,
            padding_mask=padding_mask,
            deterministic=deterministic,
            channel_states=channel_states,
            channel_mask=sidechannel_mask,
        )

        logits = self.unembed(x)
        loss_val = self.loss(logits, targets) if targets is not None else None
        return logits, loss_val
