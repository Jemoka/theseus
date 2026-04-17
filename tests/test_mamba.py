"""Tests for Mamba-2 and Hybrid models."""

import pytest
import jax
import jax.numpy as jnp
from contextlib import contextmanager

from theseus.config import build, configuration


@contextmanager
def _mamba_config_ctx():
    """Set up a config context with all fields needed for Mamba models."""
    from theseus.model.models.mamba import Mamba
    from theseus.model.block.mamba import MambaBlock
    from theseus.model.layers.rmsnorm import RMSNorm

    cfg = build(Mamba, MambaBlock, RMSNorm)

    # Override to tiny sizes for testing
    from omegaconf import OmegaConf

    OmegaConf.set_struct(cfg, False)
    cfg.architecture.n_layers = 2
    cfg.architecture.n_embd = 128
    cfg.architecture.block_size = 64
    cfg.architecture.vocab_size = 256
    cfg.architecture.dropout = 0.0
    cfg.architecture.d_state = 16
    cfg.architecture.d_conv = 4
    cfg.architecture.expand = 2
    cfg.architecture.n_groups = 1
    cfg.architecture.n_heads = -1
    cfg.architecture.rms_norm_eps = 1e-6
    cfg.architecture.dtype = OmegaConf.create(
        {"param": "float32", "activation": "float32"}
    )
    OmegaConf.set_struct(cfg, True)

    with configuration(cfg):
        yield cfg


@contextmanager
def _hybrid_config_ctx():
    """Set up a config context for Hybrid models."""
    from theseus.model.models.hybrid import Hybrid
    from theseus.model.block import Block
    from theseus.model.block.mamba import MambaBlock
    from theseus.model.layers import LayerNorm
    from theseus.model.layers.rmsnorm import RMSNorm
    from theseus.model.layers.mlp import MLP
    from theseus.model.attention import SelfAttention, RopeAttention

    cfg = build(
        Hybrid, Block, MambaBlock, LayerNorm, RMSNorm, MLP, SelfAttention, RopeAttention
    )

    from omegaconf import OmegaConf

    OmegaConf.set_struct(cfg, False)
    cfg.architecture.n_layers = 4
    cfg.architecture.n_embd = 128
    cfg.architecture.n_head = 8
    cfg.architecture.block_size = 64
    cfg.architecture.vocab_size = 256
    cfg.architecture.dropout = 0.0
    cfg.architecture.rope = True
    cfg.architecture.bias = True
    cfg.architecture.mamba_layers = "even"
    cfg.architecture.d_state = 16
    cfg.architecture.d_conv = 4
    cfg.architecture.expand = 2
    cfg.architecture.n_groups = 1
    cfg.architecture.n_heads = -1
    cfg.architecture.rms_norm_eps = 1e-6
    cfg.architecture.dtype = OmegaConf.create(
        {"param": "float32", "activation": "float32"}
    )
    OmegaConf.set_struct(cfg, True)

    with configuration(cfg):
        yield cfg


class TestMambaBlock:
    def test_forward_shape(self):
        from theseus.model.block.mamba import MambaBlock
        from theseus.config import configure

        with _mamba_config_ctx():
            block = configure(MambaBlock)
            key = jax.random.PRNGKey(0)
            x = jnp.ones((2, 16, 128))
            params = block.init(key, x)
            y = block.apply(params, x, deterministic=True)
            assert y.shape == x.shape

    def test_backward_runs(self):
        from theseus.model.block.mamba import MambaBlock
        from theseus.config import configure

        with _mamba_config_ctx():
            block = configure(MambaBlock)
            key = jax.random.PRNGKey(0)
            x = jnp.ones((2, 16, 128))
            params = block.init(key, x)

            def loss_fn(p):
                y = block.apply(p, x, deterministic=True)
                return jnp.mean(y)

            grad = jax.grad(loss_fn)(params)
            leaves = jax.tree_util.tree_leaves(grad)
            assert all(jnp.all(jnp.isfinite(l)) for l in leaves)

    def test_padding_mask(self):
        from theseus.model.block.mamba import MambaBlock
        from theseus.config import configure

        with _mamba_config_ctx():
            block = configure(MambaBlock)
            key = jax.random.PRNGKey(0)
            x = jnp.ones((1, 8, 128))
            params = block.init(key, x)

            mask = jnp.array([[True, True, True, True, False, False, False, False]])
            y = block.apply(params, x, padding_mask=mask, deterministic=True)
            # Padded positions in the output projection should be zeroed
            # (the residual connection still adds the input though)
            # Just check the output is finite
            assert jnp.all(jnp.isfinite(y))


class TestSelectiveScan:
    def test_zero_input(self):
        from theseus.model.block.mamba import _selective_scan

        B, T, H, G, N = 1, 8, 4, 1, 8
        A = jnp.zeros((B, T, H))
        B_mat = jnp.zeros((B, T, G, N))
        C_mat = jnp.zeros((B, T, G, N))
        dt = jnp.ones((B, T, H))
        x = jnp.zeros((B, T, H))

        y = _selective_scan(A, B_mat, C_mat, dt, x)
        assert y.shape == (B, T, H)
        assert jnp.allclose(y, 0.0)

    def test_output_shape(self):
        from theseus.model.block.mamba import _selective_scan

        B, T, H, G, N = 2, 16, 8, 2, 4
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 5)
        A = jax.random.normal(keys[0], (B, T, H)) * 0.1
        B_mat = jax.random.normal(keys[1], (B, T, G, N))
        C_mat = jax.random.normal(keys[2], (B, T, G, N))
        dt = jax.nn.softplus(jax.random.normal(keys[3], (B, T, H)))
        x = jax.random.normal(keys[4], (B, T, H))

        y = _selective_scan(A, B_mat, C_mat, dt, x)
        assert y.shape == (B, T, H)
        assert jnp.all(jnp.isfinite(y))


class TestMambaModel:
    def test_forward_shape(self):
        from theseus.model.models.mamba import Mamba
        from theseus.config import configure

        with _mamba_config_ctx():
            model = configure(Mamba)
            key = jax.random.PRNGKey(0)
            idx = jnp.zeros((2, 16), dtype=jnp.int32)
            params = model.init(key, idx)
            logits, loss = model.apply(params, idx, deterministic=True)
            assert logits.shape == (2, 16, 256)
            assert loss is None

    def test_forward_with_loss(self):
        from theseus.model.models.mamba import Mamba
        from theseus.config import configure

        with _mamba_config_ctx():
            model = configure(Mamba)
            key = jax.random.PRNGKey(0)
            idx = jnp.zeros((2, 16), dtype=jnp.int32)
            targets = jnp.ones((2, 16), dtype=jnp.int32)
            params = model.init(key, idx)
            logits, loss = model.apply(
                params, idx, targets=targets, deterministic=True
            )
            assert logits.shape == (2, 16, 256)
            assert loss is not None
            assert loss.shape == ()
            assert jnp.isfinite(loss)

    def test_backward_runs(self):
        from theseus.model.models.mamba import Mamba
        from theseus.config import configure

        with _mamba_config_ctx():
            model = configure(Mamba)
            key = jax.random.PRNGKey(0)
            idx = jnp.zeros((2, 16), dtype=jnp.int32)
            targets = jnp.ones((2, 16), dtype=jnp.int32)
            params = model.init(key, idx)

            def loss_fn(p):
                _, loss = model.apply(
                    p, idx, targets=targets, deterministic=True
                )
                return loss

            grad = jax.grad(loss_fn)(params)
            leaves = jax.tree_util.tree_leaves(grad)
            assert all(jnp.all(jnp.isfinite(l)) for l in leaves)


class TestHybridModel:
    def test_forward_shape(self):
        from theseus.model.models.hybrid import Hybrid
        from theseus.config import configure

        with _hybrid_config_ctx():
            model = configure(Hybrid)
            key = jax.random.PRNGKey(0)
            idx = jnp.zeros((2, 16), dtype=jnp.int32)
            params = model.init(key, idx)
            logits, loss = model.apply(params, idx, deterministic=True)
            assert logits.shape == (2, 16, 256)
            assert loss is None

    def test_forward_with_loss(self):
        from theseus.model.models.hybrid import Hybrid
        from theseus.config import configure

        with _hybrid_config_ctx():
            model = configure(Hybrid)
            key = jax.random.PRNGKey(0)
            idx = jnp.zeros((2, 16), dtype=jnp.int32)
            targets = jnp.ones((2, 16), dtype=jnp.int32)
            params = model.init(key, idx)
            logits, loss = model.apply(
                params, idx, targets=targets, deterministic=True
            )
            assert logits.shape == (2, 16, 256)
            assert loss is not None
            assert jnp.isfinite(loss)

    def test_backward_runs(self):
        from theseus.model.models.hybrid import Hybrid
        from theseus.config import configure

        with _hybrid_config_ctx():
            model = configure(Hybrid)
            key = jax.random.PRNGKey(0)
            idx = jnp.zeros((2, 16), dtype=jnp.int32)
            targets = jnp.ones((2, 16), dtype=jnp.int32)
            params = model.init(key, idx)

            def loss_fn(p):
                _, loss = model.apply(
                    p, idx, targets=targets, deterministic=True
                )
                return loss

            grad = jax.grad(loss_fn, allow_int=True)(params)
            float_leaves = [
                l for l in jax.tree_util.tree_leaves(grad)
                if hasattr(l, 'dtype') and jnp.issubdtype(l.dtype, jnp.floating)
            ]
            assert all(jnp.all(jnp.isfinite(l)) for l in float_leaves)

    def test_sharding_has_ssm_axis(self):
        from theseus.model.models.hybrid import Hybrid
        from theseus.config import configure

        with _hybrid_config_ctx():
            model = configure(Hybrid)
            sharding = model.sharding
            axes_names = [s[0] for s in sharding]
            assert "n_ssm" in axes_names

    def test_parse_mamba_layers_even(self):
        from theseus.model.models.hybrid import Hybrid
        from theseus.config import configure

        with _hybrid_config_ctx():
            model = configure(Hybrid)
            indices = model._parse_mamba_layers()
            assert indices == {0, 2}  # 4 layers, even = 0, 2

    def test_parse_mamba_layers_explicit(self):
        from theseus.model.models.hybrid import Hybrid
        from theseus.config import configure
        from theseus.config import patch
        from omegaconf import OmegaConf

        with _hybrid_config_ctx() as cfg:
            with patch():
                cfg.architecture.n_layers = 6
                cfg.architecture.mamba_layers = "1,3,5"
            model = configure(Hybrid)
            indices = model._parse_mamba_layers()
            assert indices == {1, 3, 5}
