"""
SideChannelQwen: Qwen with cross-attention side channels.

Extends Qwen: layers in cross_attn_layers use SideChannelQwenBlock.
A lightweight SideChannelEncoder compresses raw side-channel tokens.
Supports from_pretrained: loads base Qwen weights, initializes cross-attention
layers and encoder fresh (gates at 0 = no initial contribution).
"""

from typing import Optional, Tuple, List, Any, Type

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

from omegaconf import OmegaConf

from theseus.config import field, configure, patch
from theseus.model.models.contrib.qwen import Qwen, _from_hf_state_dict
from theseus.model.block.qwen import QwenDecoderBlock
from theseus.model.block.sidechannel import SideChannelQwenBlock
from theseus.model.attention.perceiver import SideChannelEncoder
from theseus.model.layers import RMSNorm
from theseus.model.axes import Axes


class SideChannelQwen(Qwen):
    """Qwen with cross-attention side channels.

    Selected layers (via cross_attn_layers) use SideChannelQwenBlock instead
    of QwenDecoderBlock. A lightweight encoder (learned-query cross-attn + MLP)
    compresses side-channel tokens.

    from_pretrained loads base Qwen weights; cross-attention layers and encoder
    are initialized fresh with tanh gates at 0 (preserving pretrained behavior
    exactly at init).
    """

    n_channels: int = field("architecture/sidechannel/n_channels", default=4)
    n_latents: int = field("architecture/sidechannel/n_latents", default=128)
    cross_attn_layers: List[int] = field(
        "architecture/sidechannel/cross_attn_layers",
        default_factory=lambda: [3, 7, 11, 15, 19, 23, 27, 31],
    )

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [QwenDecoderBlock, SideChannelQwenBlock, SideChannelEncoder, RMSNorm]

    def setup(self) -> None:
        assert self.vocab_size is not None
        assert self.block_size is not None

        self.wte: Any = self.param(
            "wte",
            nn.with_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.VOCAB.value, Axes.N_EMBD.value),
            ),
            (self.vocab_size, self.n_embd),
            self._param_dtype,
        )

        self.lm_head: Any = self.param(
            "lm_head",
            nn.with_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.VOCAB.value, Axes.N_EMBD.value),
            ),
            (self.vocab_size, self.n_embd),
            self._param_dtype,
        )

        self.drop = nn.Dropout(rate=self.dropout)

        # Side-channel encoder
        self.encoder = configure(SideChannelEncoder)

        # Layer types for sliding window
        self.layer_types = [
            "sliding"
            if (
                self.use_sliding_window
                and self.sliding_window > 0
                and i >= self.max_window_layers
            )
            else "full"
            for i in range(self.n_layers)
        ]

        # Create blocks: SideChannelQwenBlock for cross_attn_layers
        cross_set = set(self.cross_attn_layers)
        self.blocks = [
            configure(SideChannelQwenBlock)
            if i in cross_set
            else configure(QwenDecoderBlock)
            for i in range(self.n_layers)
        ]

        self.ln_f = configure(RMSNorm)

    def encode_channels(
        self, sidechannel: jax.Array, deterministic: bool = False
    ) -> jax.Array:
        """Encode raw side-channel token IDs through encoder.

        Args:
            sidechannel: (B, N, L) token IDs for N channels
            deterministic: whether to apply dropout

        Returns:
            (B, N, K, C) compressed channel states
        """
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
        channel_states: Optional[jax.Array] = None,
        channel_mask: Optional[jax.Array] = None,
        **kwargs: Any,
    ) -> jax.Array:
        """Process through blocks with optional cross-attention side channels."""
        b, t, _ = x.shape
        positions = jnp.arange(t)

        for i, block in enumerate(self.blocks):
            sliding = self.layer_types[i] == "sliding"
            mask = padding_mask

            if isinstance(block, SideChannelQwenBlock):
                x = block(
                    x,
                    channel_states=channel_states,
                    channel_mask=channel_mask,
                    padding_mask=mask,
                    deterministic=deterministic,
                    sliding=sliding,
                    positions=positions,
                )
            else:
                x = block(
                    x,
                    padding_mask=mask,
                    deterministic=deterministic,
                    sliding=sliding,
                    positions=positions,
                )
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
    ) -> Tuple[Any, Optional[Any]]:
        """Forward pass with optional side-channel inputs.

        Args:
            idx: (B, T) input token indices
            targets: (B, T) target token indices, -1 to ignore
            padding_mask: (B, T) bool, True=valid
            deterministic: if False, apply dropout
            sidechannel: (B, N, L) token IDs for N side channels
            sidechannel_mask: (B, T) int in [0..N-1]

        Returns:
            logits: (B, T, vocab_size)
            loss: cross-entropy loss if targets provided, else None
        """
        b, t = idx.shape
        assert t <= self.block_size, (
            f"Cannot forward sequence of length {t}, "
            f"block size is only {self.block_size}"
        )

        x = self.embed(idx, deterministic)

        # Encode side channels (always call encoder to ensure params exist)
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

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: str = "cpu",
        param_dtype: str = "float32",
        activation_dtype: str = "bfloat16",
    ) -> Any:
        """Load pretrained Qwen weights, adding fresh cross-attention + encoder params.

        The base Qwen weights are loaded into the self-attention, MLP, norms, and
        embeddings. Cross-attention layers (rms_cross, cross_attn) and the encoder
        are initialized fresh. Since tanh gates start at 0, the model produces
        identical outputs to vanilla Qwen at initialization.
        """
        import torch
        from transformers import Qwen2ForCausalLM

        torch_dtype = torch.float32
        hf_model = Qwen2ForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=None
        )
        hf_model.to(device)
        hf_model.eval()
        cfg = hf_model.config

        rope_theta = 10000.0
        if cfg.rope_parameters is not None and "rope_theta" in cfg.rope_parameters:
            rope_theta = cfg.rope_parameters["rope_theta"]

        with patch() as th_cfg:
            new_arch = OmegaConf.create(
                {
                    "n_layers": cfg.num_hidden_layers,
                    "n_embd": cfg.hidden_size,
                    "n_head": cfg.num_attention_heads,
                    "n_kv_head": cfg.num_key_value_heads,
                    "intermediate_size": cfg.intermediate_size,
                    "block_size": cfg.max_position_embeddings,
                    "vocab_size": cfg.vocab_size,
                    "dropout": 0.0,
                    "attn_dropout": float(cfg.attention_dropout),
                    "rope_theta": float(rope_theta),
                    "rms_norm_eps": float(cfg.rms_norm_eps),
                    "use_sliding_window": bool(cfg.use_sliding_window),
                    "sliding_window": int(cfg.sliding_window)
                    if cfg.sliding_window is not None
                    else -1,
                    "max_window_layers": int(cfg.max_window_layers),
                    "bias": False,
                    "attention_bias": True,
                    "partial_rotary_factor": 1.0,
                    "dtype": {"param": param_dtype, "activation": activation_dtype},
                }
            )
            if "architecture" in th_cfg:
                th_cfg.architecture = OmegaConf.merge(th_cfg.architecture, new_arch)
            else:
                th_cfg.architecture = new_arch

            model = configure(cls)
            dummy = jnp.zeros((1, 1), dtype=jnp.int32)
            abstract = jax.eval_shape(model.init, jax.random.PRNGKey(0), dummy)
            params = jax.tree_util.tree_map(
                lambda x: np.zeros(x.shape, x.dtype), abstract["params"]
            )

            # Load base Qwen weights into the shared params
            # _from_hf_state_dict fills: wte, lm_head, blocks_*/rms_1, blocks_*/rms_2,
            # blocks_*/attn/*, blocks_*/mlp/*, ln_f
            # It skips keys that don't exist in HF (rms_cross, cross_attn, encoder)
            params = _from_hf_state_dict(params, hf_model.state_dict(), model.n_layers)

            return model, params
