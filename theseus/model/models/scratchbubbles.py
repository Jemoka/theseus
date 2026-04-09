"""
Scratchubbbles: thoughtbubbles except we can fork into any other token.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

import numpy as np

from theseus.model.axes import Axes
from theseus.model.layers import LayerNorm
from theseus.model.block.forking import ThoughtBlock
from theseus.model.block.scratching import ScratchingBlock
from theseus.model.models.thoughtbubbles import Thoughtbubbles
from theseus.config import configure

from typing import List, Any, Type, Dict


def pca(x: jax.Array, n_components: int) -> jax.Array:
    """x: (N, D) -> (N, n_components). Centered SVD."""
    x = x - x.mean(axis=0)
    U, S, Vt = jnp.linalg.svd(x, full_matrices=False)
    return U[:, :n_components] * S[:n_components]


def vectors_to_colors(embeddings: jax.Array) -> jax.Array:
    """(layers, seq_len, hidden) -> (layers, seq_len, 3) RGB."""
    flat = embeddings.reshape(-1, embeddings.shape[-1])
    coords = pca(flat, 3)
    for i in range(3):
        lo, hi = jnp.percentile(coords[:, i], jnp.array([2, 98]))
        coords = coords.at[:, i].set(
            jnp.clip((coords[:, i] - lo) / (hi - lo + 1e-8), 0, 1)
        )
    return coords.reshape(*embeddings.shape[:2], 3)


class Scratchbubbles(Thoughtbubbles):
    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [ThoughtBlock, ScratchingBlock, LayerNorm]

    @staticmethod
    def plot(intermediates: Any) -> Dict[str, Any]:
        """intermediates -> [figure]"""

        from matplotlib import pyplot as plt
        import seaborn as sns

        # TODO not exactly sure why each tuple has an extra
        # dim but it seems like we take the first one; perhaps
        # from initialization/multiple calls?

        ### plot embeddings ###
        embeddings = [intermediates["plots"]["embeddings"]] + [
            i["embeddings"]
            for i in intermediates["plots"].values()
            if isinstance(i, dict)
        ]
        max_seq_len = max([e[0].shape[1] for e in embeddings])
        padded_embeddings = []

        for i in embeddings:
            e = i[0]
            if e.shape[1] < max_seq_len:
                pad_width = ((0, 0), (0, max_seq_len - e.shape[1]), (0, 0))
                e = jnp.pad(e, pad_width)
            padded_embeddings.append(e)
        stacked_embeddings = jnp.array(padded_embeddings)
        embeddings_2d = stacked_embeddings[:, 0]  # only one sample batch axes

        # map to colors and plot: (layer x seq_len x 3)
        colors = vectors_to_colors(embeddings_2d.astype(jnp.float32))
        c = np.array(colors)  # [layer, seq_len, 3]

        fig, ax = plt.subplots(figsize=(12, 6))
        # imshow directly takes [H, W, 3] RGB in [0,1]
        ax.imshow(np.clip(c, 0, 1), aspect="auto", interpolation="nearest")

        ax.set_xlabel("seq position")
        ax.set_ylabel("layer")
        plt.tight_layout()

        plots = {"analysis/embeddings": fig}

        ### plot "extra" attention to see where tokens are attending ###
        weights = [
            i["ssca"]["scratching_attn_weights"][0][0][0]
            for i in intermediates["plots"].values()
            if isinstance(i, dict)
        ]
        max_queries = max([i.shape[0] for i in weights])
        max_keys = max([i.shape[-1] for i in weights])

        # pad weights to max shape so we can stack them
        padded_weights = []
        for w in weights:
            pad_q = max_queries - w.shape[0]
            pad_k = max_keys - w.shape[-1]
            if pad_q > 0 or pad_k > 0:
                w = jnp.pad(w, ((0, pad_q), (0, pad_k)), constant_values=-jnp.inf)
            padded_weights.append(w)

        # aaand softmax
        stacked_weights = jax.nn.softmax(jnp.stack(padded_weights), axis=-1)

        # plot
        for i, w in enumerate(stacked_weights):
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(w.astype(jnp.float32), ax=ax, cmap="viridis")

            ax.set_xlabel("token index (queries, forks)")
            ax.set_ylabel("token index (keys, seq)")
            ax.set_title(f"Attention Weights for Block {i}")
            plt.tight_layout()
            plots[f"analysis/attention_weights_block_{i}"] = fig

        ### make the rest of the plot ###
        plots.update(Thoughtbubbles.plot(intermediates))

        return plots

    def setup(self) -> None:
        assert self.vocab_size is not None
        assert self.block_size is not None

        # Token embedding table (no positional - using RoPE in attention)
        self.wte: jax.Array = self.param(
            "wte",
            nn.with_partitioning(
                nn.initializers.normal(stddev=0.02),
                (Axes.VOCAB.value, Axes.N_EMBD.value),
            ),
            (self.vocab_size, self.n_embd),
            jnp.float32,
        )  # type: ignore

        self.drop = nn.Dropout(rate=self.dropout)

        # Create blocks: ForkingBlock for layers in fork list, ThoughtBlock otherwise
        fork_set = set(self.fork)
        self.blocks = [
            configure(ScratchingBlock) if i in fork_set else configure(ThoughtBlock)
            for i in range(self.n_layers)
        ]

        self.ln_f = configure(LayerNorm)
