"""
Transformer blocks with gated cross-attention side channels.

SideChannelBlock: extends Block (GPT backbone) with cross-attention.
SideChannelQwenBlock: extends QwenDecoderBlock with cross-attention.
"""

from typing import Optional, List, Type, Any, Tuple, Dict

import jax

from theseus.config import configure, field
from theseus.model.module import Module
from theseus.model.attention import SelfAttention, RopeAttention
from theseus.model.attention.sidechannel import GroupedSidechannelCrossAttention
from theseus.model.attention.grouped import GroupedSelfAttention
from theseus.model.layers import LayerNorm, RMSNorm, QwenMLP
from theseus.model.layers.mlp import MLP


class SideChannelBlock(Module):
    """Block with self-attention + gated cross-attention + MLP.

    Extends the standard Block pattern for GPT backbone. When channel_states
    is None, the cross-attention is skipped entirely (standard block behavior).
    """

    rope: bool = field("architecture/rope", default=True)

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [
            LayerNorm,
            SelfAttention,
            RopeAttention,
            GroupedSidechannelCrossAttention,
            MLP,
        ]

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return []

    def setup(self) -> None:
        self.ln_1 = configure(LayerNorm)
        if self.rope:
            self.attn = configure(RopeAttention)
        else:
            self.attn = configure(SelfAttention)
        self.ln_cross = configure(LayerNorm)
        self.cross_attn = configure(GroupedSidechannelCrossAttention)
        self.ln_2 = configure(LayerNorm)
        self.mlp = configure(MLP)

    def __call__(
        self,
        x: jax.Array,
        channel_states: Optional[jax.Array] = None,
        channel_mask: Optional[jax.Array] = None,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Any,
    ) -> jax.Array:
        x = x + self.attn(
            self.ln_1(x), padding_mask=padding_mask, deterministic=deterministic, **kwargs
        )
        if channel_states is not None:
            x = x + self.cross_attn(
                self.ln_cross(x),
                channel_states,
                channel_mask,
                deterministic=deterministic,
            )
        x = x + self.mlp(self.ln_2(x), deterministic=deterministic)
        return x


class SideChannelQwenBlock(Module):
    """QwenDecoderBlock + gated cross-attention for Qwen backbone.

    When channel_states is None, behaves identically to QwenDecoderBlock.
    """

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [RMSNorm, GroupedSelfAttention, GroupedSidechannelCrossAttention, QwenMLP]

    @property
    def sharding(self) -> List[Tuple[str, Optional[Any]]]:
        return []

    def setup(self) -> None:
        self.rms_1 = configure(RMSNorm)
        self.attn = configure(GroupedSelfAttention)
        self.rms_cross = configure(RMSNorm)
        self.cross_attn = configure(GroupedSidechannelCrossAttention)
        self.rms_2 = configure(RMSNorm)
        self.mlp = configure(QwenMLP)

    def __call__(
        self,
        x: jax.Array,
        channel_states: Optional[jax.Array] = None,
        channel_mask: Optional[jax.Array] = None,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        sliding: bool = False,
        positions: Optional[jax.Array] = None,
    ) -> jax.Array:
        h = self.rms_1(x)
        h = self.attn(
            h,
            padding_mask=padding_mask,
            deterministic=deterministic,
            sliding=sliding,
            positions=positions,
        )
        x = x + h

        if channel_states is not None:
            h = self.rms_cross(x)
            h = self.cross_attn(
                h, channel_states, channel_mask, deterministic=deterministic
            )
            x = x + h

        h = self.rms_2(x)
        h = self.mlp(h, deterministic=deterministic)
        x = x + h
        return x
