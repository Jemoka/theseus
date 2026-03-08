import math
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax

from theseus.config import configure
from theseus.model.axes import Axes
from theseus.model.block.forking import ForkingBlock
from theseus.model.attention.scratching import ScratchSparseCrossAttention


class ScratchingBlock(ForkingBlock):
    def setup(self) -> None:
        super().setup()
        self.ssca = configure(ScratchSparseCrossAttention)

    @nn.compact
    def fork(
        self,
        x: jax.Array,
        cumulative_scores: jax.Array,
        token_index: jax.Array,
        padding_mask: Optional[jax.Array],
        input_seq_len: Optional[int] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """
        Top-k forking: doubles tokens then selects top-k to keep.

        Args:
            x: Input tensor of shape (B, T, C).
            cumulative_scores: Log-probability scores of shape (B, T).
            token_index: Original token indices of shape (B, T).
            padding_mask: Boolean tensor of shape (B, T). True for valid.
            input_seq_len: Original input sequence length for ratio calc.

        Mocks:
            x = jnp.ones((4, config.architecture.block_size, config.architecture.n_embd))  # (B=4, T=block_size, C=n_embd)
            cumulative_scores = jnp.zeros((4, config.architecture.block_size))  # (B=4, T=block_size)
            token_index = jnp.tile(jnp.arange(config.architecture.block_size), (4, 1))  # (B=4, T=block_size)
            padding_mask = jnp.ones((4, config.architecture.block_size), dtype=bool)  # (B=4, T=block_size)

        Returns:
            Tuple of (forked_x, new_scores, new_token_indices).
        """

        batch_size = cumulative_scores.shape[0]

        # Compute current layer's forking scores
        curr_layer_forking_score = self.forking_score(self.forking_ln(x))

        # Update cumulative scores (in log space)
        forking_scores_cum = (
            self.clipped_logsigmoid(curr_layer_forking_score)
            + cumulative_scores[:, :, None]
        ).reshape(batch_size, -1)  # (batch_size, 2T)
        forking_scores_cum_for_topk = forking_scores_cum

        # Copy token index twice (for original and fork)
        forked_token_index = token_index.repeat(2, axis=-1)

        # Mark *leftmost* token of each original token with +inf (always keep)
        rolled = jnp.roll(forked_token_index, 1, axis=-1)
        is_leftmost = rolled != forked_token_index
        forking_scores_cum_for_topk = jnp.where(
            is_leftmost, float("+inf"), forking_scores_cum_for_topk
        )

        if padding_mask is not None:
            new_padding_mask = jnp.take_along_axis(
                padding_mask, forked_token_index, axis=-1
            )

            forking_scores_cum = jnp.where(
                new_padding_mask, forking_scores_cum, float("-inf")
            )
            forking_scores_cum_for_topk = jnp.where(
                new_padding_mask, forking_scores_cum_for_topk, float("-inf")
            )

            # Soft kill to maintain ratio with padding
            keep_p = self.max_block_size / self.block_size
            n_per_row = padding_mask.sum(axis=-1)
            k_per_row = jnp.floor(keep_p * n_per_row).astype(jnp.int32)

            score_order = jnp.argsort(forking_scores_cum_for_topk, axis=-1)[:, ::-1]
            rank_in_row = jnp.argsort(score_order, axis=-1)
            scores_to_kill = ~(rank_in_row < k_per_row[:, None])
            forking_scores_cum_for_topk = jnp.where(
                scores_to_kill, -jnp.inf, forking_scores_cum_for_topk
            )
            forking_scores_cum = jnp.where(
                scores_to_kill, float("-inf"), forking_scores_cum
            )

        # Perform top-k selection
        if input_seq_len is not None:
            k = int(math.ceil(input_seq_len * (self.max_block_size / self.block_size)))
        else:
            k = self.max_block_size

        k = min(k, forking_scores_cum_for_topk.shape[-1])
        _, top_k_indices = lax.top_k(forking_scores_cum_for_topk, k)
        top_k_indices = jnp.sort(top_k_indices, axis=-1)

        # Gather based on indices that survived
        orig_indices = top_k_indices // 2
        is_fork = (top_k_indices % 2) == 1  # we fork to the *right*

        # Gather x values
        batch_indices = jnp.arange(batch_size)[:, None]
        x_to_consider = x[batch_indices, orig_indices, :]

        # Add fork embedding
        fork_embedding = self.param(
            "fork_embedding",
            nn.with_partitioning(
                lambda rng, shape: jax.random.normal(rng, shape)
                * (1 / jnp.sqrt(self.n_embd)),
                (Axes.N_EMBD.value,),
            ),
            (self.n_embd,),
        )
        fork_embedding_bf16 = jnp.asarray(fork_embedding, dtype=jnp.bfloat16)

        x_to_consider = x_to_consider + (
            is_fork[:, :, None].astype(x.dtype) * fork_embedding_bf16
        )

        # "local forking attention"
        K = x  # full sequence of the original, pre-for kinputs
        V = (
            x + fork_embedding_bf16
        )  # full sequence of the original, pre-fork inputs, add fork embedding since this is what we'l be avging
        Q = (
            x_to_consider * is_fork[:, :, None]
        )  # the new forked tokens, in particular the non-zero entries, is what we used to gather

        # do the local mixing things to figure out the identity of the forked tokens
        forked_embds = self.ssca(
            Q,
            K,
            V,
            padding_mask=padding_mask,
            token_index=token_index,
            query_token_index=token_index.repeat(2, axis=1),
            cumulative_scores=cumulative_scores,
        )  # get rid of heads

        # and set forked embeddings
        x_to_consider = jnp.where(is_fork[:, :, None], forked_embds, x_to_consider)

        # Gather cumulative scores and token indices
        new_cumulative_scores = jnp.take_along_axis(
            forking_scores_cum, top_k_indices, axis=-1
        )
        new_token_indices = jnp.take_along_axis(token_index, orig_indices, axis=-1)
        self.sow("plots", "new_cumulative_scores", new_cumulative_scores)
        self.sow("plots", "embeddings", x_to_consider)

        return x_to_consider, new_cumulative_scores, new_token_indices
