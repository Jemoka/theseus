import jax
import flax.linen as nn
import jax.numpy as jnp

from typing import Optional, Tuple, List, Any, Type, Dict

from theseus.model.block import Block
from theseus.model.layers import LayerNorm
from theseus.model.axes import Axes
from theseus.model.module import Module

from theseus.config import field, configure
from theseus.base.axis import Axis


class GPT(Module):
    rope: bool = field("architecture/rope")
    n_layers: int = field("architecture/n_layers")
    n_embd: int = field("architecture/n_embd")
    vocab_size: int = field("architecture/vocab_size")
    block_size: int = field("architecture/block_size")
    dropout: float = field("architecture/dropout")

    @property
    def sharding(self) -> List[Tuple[Axes, Optional[Axis]]]:
        return [
            (Axes.VOCAB, None),
            (Axes.BLOCK_SIZE, None),
            (Axes.N_EMBD, None),
            (Axes.N_EMBD_FF, Axis.SHARD),
            (Axes.N_EMBD_OUT, Axis.SHARD),
            (Axes.N_ATTN, Axis.SHARD),
        ]

    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [Block, LayerNorm]

    def setup(self) -> None:
        assert self.vocab_size is not None
        assert self.block_size is not None

        # Token embedding table
        self.wte: jax.Array = self.param(
            "wte",
            nn.with_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.VOCAB.value, Axes.N_EMBD.value),
            ),
            (self.vocab_size, self.n_embd),
            jnp.float32,
        )  # type: ignore

        # Positional embedding table (only when not using RoPE)
        if not self.rope:
            self.wpe: jax.Array = self.param(
                "wpe",
                nn.with_partitioning(
                    nn.initializers.normal(stddev=0.02),
                    (Axes.BLOCK_SIZE.value, Axes.N_EMBD.value),
                ),
                (self.block_size, self.n_embd),
                jnp.float32,
            )  # type: ignore

        self.drop = nn.Dropout(rate=self.dropout)
        self.blocks = [configure(Block) for _ in range(self.n_layers)]
        self.ln_f = configure(LayerNorm)

    def embed(
        self, idx: jax.Array, deterministic: bool = False, **kwargs: Dict[Any, Any]
    ) -> Any:
        """Compute token and positional embeddings given inputs."""

        _, t = idx.shape

        # Token embeddings
        x = jnp.take(self.wte, idx, axis=0).astype(jnp.bfloat16)

        # Positional embeddings (only when not using RoPE)
        if not self.rope:
            pos = jnp.arange(0, t)
            x = x + jnp.take(self.wpe, pos, axis=0).astype(jnp.bfloat16)

        x = self.drop(x, deterministic=deterministic)

        return x

    def decode(
        self,
        x: jax.Array,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Dict[Any, Any],
    ) -> Any:
        """Compute decoded residual channels given embeddings."""

        for block in self.blocks:
            x = block(x, padding_mask=padding_mask, deterministic=deterministic)
        return x

    def unembed(self, x: jax.Array) -> Any:
        """Compute output distribution."""

        x = self.ln_f(x)
        logits = jnp.einsum("bth,vh->btv", x, self.wte.astype(jnp.bfloat16))

        return logits

    def loss(self, logits: jax.Array, targets: jax.Array) -> jax.Array:
        """Compute cross-entropy loss given logits and targets."""

        logits_f32 = logits.astype(jnp.float32)
        logits_flat = logits_f32.reshape(-1, logits_f32.shape[-1])
        targets_flat = targets.reshape(-1)

        # Mask out ignore index (-1)
        mask = targets_flat != -1
        targets_masked = jnp.where(mask, targets_flat, 0)

        loss = -jnp.sum(
            jax.nn.log_softmax(logits_flat, axis=-1)
            * jax.nn.one_hot(targets_masked, self.vocab_size)
            * mask[:, None]
        ) / mask.sum().clip(min=1)

        return loss

    def __call__(
        self,
        idx: jax.Array,
        targets: Optional[jax.Array] = None,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
        **kwargs: Dict[Any, Any],
    ) -> Tuple[jax.Array, Optional[jax.Array]]:
        """
        Args:
            idx: Input token indices of shape (B, T).
            targets: Target token indices of shape (B, T). Use -1 to ignore positions.
            padding_mask: Boolean tensor of shape (B, T). True for valid tokens,
                False for padding tokens.
            deterministic: If False, applies dropout.

        Returns:
            logits: Output logits of shape (B, T, vocab_size).
            loss: Cross-entropy loss if targets provided, else None.
        """
        b, t = idx.shape
        assert t <= self.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        )

        # get embeddings
        x = self.embed(idx, deterministic, **kwargs)

        # Transformer blocks
        x = self.decode(
            x, padding_mask=padding_mask, deterministic=deterministic, **kwargs
        )

        # Final layer norm and logits
        logits = self.unembed(x)

        # Compute loss if targets provided
        if targets is not None:
            loss = self.loss(logits, targets)
        else:
            loss = None

        return logits, loss
