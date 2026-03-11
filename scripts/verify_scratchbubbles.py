#!/usr/bin/env python3
"""Verify Scratchbubbles model: init params + single forward pass."""

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from theseus.config import build, configuration, configure
from theseus.model.models.scratchbubbles import Scratchbubbles


def main() -> None:
    # Build config from all components the model needs
    all_types = Scratchbubbles.gather()
    config = build(*all_types)

    # Override to tiny sizes for a quick test
    OmegaConf.set_struct(config, False)
    config.architecture.n_layers = 4
    config.architecture.n_embd = 64
    config.architecture.n_head = 2
    config.architecture.block_size = 16
    config.architecture.max_block_size = 32
    config.architecture.vocab_size = 128
    config.architecture.dropout = 0.0
    config.architecture.fork = [1, 3]
    config.architecture.scratch_head_dim = 16
    OmegaConf.set_struct(config, True)

    with configuration(config):
        model = configure(Scratchbubbles)

        # Init params
        rng = jax.random.PRNGKey(42)
        dummy_input = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8]])  # (B=1, T=8)
        variables = model.init(rng, dummy_input, deterministic=True)

        print("Parameter shapes:")
        flat = jax.tree.leaves_with_path(variables["params"])
        for path, leaf in flat:
            key_str = "/".join(str(k) for k in path)
            print(f"  {key_str}: {leaf.shape}")

        # Forward pass
        logits, loss = model.apply(variables, dummy_input, deterministic=True)

        print(f"\nInput shape:  {dummy_input.shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Loss: {loss}")
        assert logits.shape == (1, 8, 128), f"Bad logits shape: {logits.shape}"

        # Forward pass with targets (check loss computation)
        targets = jnp.array([[2, 3, 4, 5, 6, 7, 8, 9]])
        logits2, loss2 = model.apply(
            variables, dummy_input, targets, deterministic=True
        )

        print("\nWith targets:")
        print(f"  Logits shape: {logits2.shape}")
        print(f"  Loss: {loss2}")
        assert loss2 is not None, "Loss should not be None when targets are provided"
        assert jnp.isfinite(loss2), f"Loss is not finite: {loss2}"

        # Verify residual averaging path is exercised:
        # With fork=[1,3], the sequence expands after forking layers.
        # The output shape (1, 8, 128) = (B, T_input, vocab_size) proves
        # residual_average collapsed the expanded sequence back to T_input.
        # If residual averaging were broken, we'd get (1, T_expanded, 128).
        # Also verify via mutable=["plots"] on apply:
        (logits_ra, _), plots = model.apply(
            variables,
            dummy_input,
            deterministic=True,
            mutable=["plots"],
        )
        assert logits_ra.shape == (1, 8, 128), (
            f"Residual averaging output shape wrong: {logits_ra.shape}"
        )
        print(f"\nResidual averaging verified: output shape {logits_ra.shape}")

        print("\nAll checks passed!")


if __name__ == "__main__":
    main()
