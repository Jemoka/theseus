import jax
import flax.linen as nn
import jax.numpy as jnp

from typing import Optional, Tuple

from theseus.model.block import Block
from theseus.model.layers import LayerNorm

from theseus.config import field, configure


class GPT(nn.Module):
    n_layers: int = field("architecture/n_layers")
    n_embd: int = field("architecture/n_embd")
    vocab_size: int = field("architecture/vocab_size")
    block_size: int = field("architecture/block_size")
    dropout: float = field("architecture/dropout")

    def setup(self) -> None:
        assert self.vocab_size is not None
        assert self.block_size is not None

        self.blocks = [configure(Block) for _ in range(self.n_layers)]
        self.ln_f = configure(LayerNorm)

    @nn.compact
    def __call__(
        self,
        idx: jax.Array,
        targets: Optional[jax.Array] = None,
        padding_mask: Optional[jax.Array] = None,
        deterministic: bool = False,
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

        # Shared token embedding table
        wte: jax.Array = self.param(
            "wte",
            nn.with_partitioning(
                nn.initializers.normal(stddev=0.02), ("vocab", "n_embd")
            ),
            (self.vocab_size, self.n_embd),
            jnp.float32,
        )  # type: ignore

        # Token embeddings
        x = jnp.take(wte, idx, axis=0).astype(jnp.bfloat16)

        if not deterministic:
            x = nn.Dropout(rate=self.dropout)(x, deterministic=False)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, padding_mask=padding_mask, deterministic=deterministic)

        # Final layer norm and logits
        x = self.ln_f(x)
        logits = jnp.einsum("bth,vh->btv", x, wte.astype(jnp.bfloat16))

        # Compute loss if targets provided
        if targets is not None:
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
        else:
            loss = None

        return logits, loss
