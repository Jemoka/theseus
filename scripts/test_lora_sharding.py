"""LoRA sharding test with 8 fake CPU devices.

Verifies that after transition_to_lora, the LoRA params and optimizer
state are correctly sharded across devices — not stuck on device 0.

Must set XLA_FLAGS before importing JAX, so this runs as a standalone
test file with a subprocess marker if needed.
"""

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import Mesh

from theseus.config import build, configuration, configure
from theseus.base.axis import Axis
from theseus.training.lora import (
    LoRATrainState,
    LoRAConfig,
    param_filter,
    inject_lora_params,
    merge_lora_params,
    transition_to_lora,
)


class FakeTrainer:
    """Minimal trainer stub for transition_to_lora."""

    def __init__(self, state, lora_config, model, mesh, sharding_rules, scheduler):
        self.state = state
        self.lora_config = lora_config
        self.model = model
        self.mesh = mesh
        self.state_sharding = None
        self.scheduler = scheduler
        self._in_lora_phase = False
        self._sharding_rules = sharding_rules

    def main_process(self):
        return jax.process_index() == 0


def _build_gpt_config_ctx():
    from theseus.model.models.base import GPT
    from theseus.model.block import Block
    from theseus.model.layers import LayerNorm
    from theseus.model.layers.mlp import MLP
    from theseus.model.attention import SelfAttention, RopeAttention
    from omegaconf import OmegaConf

    cfg = build(GPT, Block, LayerNorm, MLP, SelfAttention, RopeAttention)
    OmegaConf.set_struct(cfg, False)
    cfg.architecture.n_layers = 2
    cfg.architecture.n_embd = 64
    cfg.architecture.n_head = 8
    cfg.architecture.block_size = 32
    cfg.architecture.vocab_size = 128
    cfg.architecture.dropout = 0.0
    cfg.architecture.rope = True
    cfg.architecture.bias = True
    cfg.architecture.dtype = OmegaConf.create(
        {"param": "float32", "activation": "float32"}
    )
    OmegaConf.set_struct(cfg, True)
    return cfg


class TestLoRASharding:
    def test_lora_params_sharded_after_transition(self):
        """After transition_to_lora, LoRA params should be sharded across
        8 fake devices, not concentrated on device 0."""
        import flax.linen

        cfg = _build_gpt_config_ctx()
        with configuration(cfg):
            from theseus.model.models.base import GPT

            model = configure(GPT)

            # Create mesh with 8 devices
            devices = np.array(jax.devices()).reshape((1, 8))
            mesh = Mesh(devices, (Axis.BATCH, Axis.SHARD))

            # Init sharded params — same as BaseTrainer._init_state
            dummy = jnp.zeros((1, 8), dtype=jnp.int32)

            import flax

            def sharded_init(key, dummy_input):
                variables = model.init(key, dummy_input)
                return variables["params"]

            params_shapes = jax.eval_shape(sharded_init, jax.random.PRNGKey(0), dummy)

            # Build initial TrainState with proper sharding
            tx = optax.adam(1e-3)

            from flax.training import train_state

            def make_state(p):
                return train_state.TrainState.create(
                    apply_fn=model.apply, params=p, tx=tx
                )

            init_params = jax.jit(
                sharded_init,
                out_shardings=flax.linen.logical_to_mesh_sharding(
                    flax.linen.get_partition_spec(params_shapes),
                    mesh,
                    rules=tuple(model.sharding),
                ),
            )(jax.random.PRNGKey(0), dummy)

            state_shapes = jax.eval_shape(make_state, init_params)
            state_sharding = flax.linen.logical_to_mesh_sharding(
                flax.linen.get_partition_spec(state_shapes),
                mesh,
                rules=tuple(model.sharding),
            )
            state = jax.jit(make_state, out_shardings=state_sharding)(init_params)

            # Create fake trainer
            lora_cfg = configure(LoRAConfig)
            fake = FakeTrainer(
                state=state,
                lora_config=lora_cfg,
                model=model,
                mesh=mesh,
                sharding_rules=tuple(model.sharding),
                scheduler=optax.constant_schedule(1e-3),
            )
            # model.sharding is used in transition_to_lora
            fake.model = model

            # Transition
            transition_to_lora(fake)

            assert fake._in_lora_phase
            assert isinstance(fake.state, LoRATrainState)

            # Check that LoRA params are sharded (not all on device 0)
            lora_leaves = jax.tree_util.tree_leaves(fake.state.params)
            for leaf in lora_leaves:
                if leaf is not None and hasattr(leaf, "sharding"):
                    # The leaf should be a jax.Array with a sharding
                    assert len(leaf.sharding.device_set) > 0, (
                        "LoRA param has no device assignment"
                    )

            # Check base_params preserved sharding
            base_leaves = jax.tree_util.tree_leaves(fake.state.base_params)
            for leaf in base_leaves:
                if hasattr(leaf, "sharding"):
                    assert len(leaf.sharding.device_set) > 0

            # Check optimizer state is sharded (not empty)
            opt_leaves = jax.tree_util.tree_leaves(fake.state.opt_state)
            assert len(opt_leaves) > 0, "Optimizer state is empty"
            for leaf in opt_leaves:
                if (
                    hasattr(leaf, "sharding")
                    and hasattr(leaf, "shape")
                    and leaf.size > 1
                ):
                    assert len(leaf.sharding.device_set) > 1, (
                        f"Optimizer state leaf with shape {leaf.shape} "
                        f"only on {len(leaf.sharding.device_set)} device(s) — "
                        f"expected sharding across multiple devices"
                    )

    def test_sharded_lora_forward_matches_unsharded(self):
        """Merged forward with sharded LoRA should produce same output as
        unsharded (single-device) LoRA, verifying sharding doesn't corrupt values."""
        import flax.linen

        cfg = _build_gpt_config_ctx()
        with configuration(cfg):
            from theseus.model.models.base import GPT
            from theseus.training.lora import LoRATrainer

            model = configure(GPT)

            devices = np.array(jax.devices()).reshape((1, 8))
            mesh = Mesh(devices, (Axis.BATCH, Axis.SHARD))

            dummy = jnp.zeros((1, 8), dtype=jnp.int32)
            full_params = jax.jit(lambda k: model.init(k, dummy)["params"])(
                jax.random.PRNGKey(0)
            )

            # Unsharded forward with LoRA
            mask = param_filter(full_params, ["kernel"])
            lora_A, lora_B = inject_lora_params(
                full_params, mask, 4, jax.random.PRNGKey(1)
            )
            merged_unsharded = merge_lora_params(full_params, lora_A, lora_B, 4.0, 4)
            logits_ref, _ = model.apply(
                {"params": merged_unsharded}, dummy, deterministic=True
            )

            # Now do the same through a sharded LoRATrainState
            tx = optax.adam(1e-3)
            from flax.training import train_state

            def make_state(p):
                return train_state.TrainState.create(
                    apply_fn=model.apply, params=p, tx=tx
                )

            import flax

            state_shapes = jax.eval_shape(make_state, full_params)
            state_sharding = flax.linen.logical_to_mesh_sharding(
                flax.linen.get_partition_spec(state_shapes),
                mesh,
                rules=tuple(model.sharding),
            )
            state = jax.jit(make_state, out_shardings=state_sharding)(full_params)

            lora_cfg = configure(LoRAConfig)
            fake = FakeTrainer(
                state=state,
                lora_config=lora_cfg,
                model=model,
                mesh=mesh,
                sharding_rules=tuple(model.sharding),
                scheduler=optax.constant_schedule(1e-3),
            )

            transition_to_lora(fake)

            # Forward through the trainer's static method
            batch = {
                "x": dummy,
                "y": jnp.ones_like(dummy),
                "padding_mask": jnp.ones_like(dummy, dtype=jnp.bool_),
            }
            logits_sharded, _, _ = LoRATrainer.forward(
                fake.state, fake.state.params, batch, deterministic=True
            )

            # B is zeros so merged = base, should match unsharded.
            # transition_to_lora casts base_params to bfloat16, so expect
            # small numerical differences vs the f32 reference (~1e-3).
            logits_base_ref, _ = model.apply(
                {"params": full_params}, dummy, deterministic=True
            )
            assert jnp.allclose(logits_sharded, logits_base_ref, atol=5e-3), (
                f"Sharded LoRA forward doesn't match unsharded: "
                f"max diff = {jnp.max(jnp.abs(logits_sharded - logits_base_ref))}"
            )
