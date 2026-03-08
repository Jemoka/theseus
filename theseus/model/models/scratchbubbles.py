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


def vectors_to_colors(x: jax.Array) -> jax.Array:
    """Map vectors to RGB colors via PCA projection. (for plotting averages)

    Args:
        x: array of shape [..., N, H]
    Returns:
        array of shape [..., N, 3] with values in [0, 1]
    """
    H = x.shape[-1]

    # flatten to [M, H] for PCA
    flat = x.reshape(-1, H)
    mean = flat.mean(axis=0)
    centered = flat - mean

    # top 3 right singular vectors = top 3 PCA directions
    _, _, Vt = jnp.linalg.svd(centered, full_matrices=False)
    basis = Vt[:3]  # [3, H]

    # project all vectors (preserving batch dims)
    projected = (x - mean) @ basis.T  # [..., N, 3]

    # center at 0.5, scale so max absolute value hits 0 or 1
    scale = jnp.max(jnp.abs(projected)) + 1e-8
    colors = 0.5 + 0.5 * projected / scale

    return colors


class Scratchbubbles(Thoughtbubbles):
    @classmethod
    def components(cls) -> List[Type[Any]]:
        return [ThoughtBlock, ScratchingBlock, LayerNorm]

    @staticmethod
    def plot(intermediates: Any) -> Dict[str, Any]:
        """intermediates -> [figure]"""

        from matplotlib import pyplot as plt

        # TODO not exactly sure why each tuple has an extra
        # dim but it seems like we take the first one; perhaps
        # from initialization/multiple calls?

        # plot embeddings
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

        # make the rest of the plot
        plots = Thoughtbubbles.plot(intermediates)
        plots.update({"analysis/embeddings": fig})

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
