"""Tests for LoRA parameter utilities and model compatibility.

Inspired by scripts/llama_parity.py — tests LoRA injection, merging,
and forward pass for all model types.
"""

import pytest
import jax
import jax.numpy as jnp
from contextlib import contextmanager

from theseus.config import build, configuration, configure
from theseus.training.lora import (
    param_filter,
    inject_lora_params,
    merge_lora_params,
    count_lora_params,
)


# ---------------------------------------------------------------------------
# Config context helpers
# ---------------------------------------------------------------------------

@contextmanager
def _gpt_config_ctx():
    from theseus.model.models.base import GPT
    from theseus.model.block import Block
    from theseus.model.layers import LayerNorm
    from theseus.model.layers.mlp import MLP
    from theseus.model.attention import SelfAttention, RopeAttention

    cfg = build(GPT, Block, LayerNorm, MLP, SelfAttention, RopeAttention)
    from omegaconf import OmegaConf

    OmegaConf.set_struct(cfg, False)
    cfg.architecture.n_layers = 2
    cfg.architecture.n_embd = 64
    cfg.architecture.n_head = 4
    cfg.architecture.block_size = 32
    cfg.architecture.vocab_size = 128
    cfg.architecture.dropout = 0.0
    cfg.architecture.rope = True
    cfg.architecture.bias = True
    cfg.architecture.dtype = OmegaConf.create(
        {"param": "float32", "activation": "float32"}
    )
    OmegaConf.set_struct(cfg, True)
    with configuration(cfg):
        yield


@contextmanager
def _mamba_config_ctx():
    from theseus.model.models.mamba import Mamba
    from theseus.model.block.mamba import MambaBlock
    from theseus.model.layers.rmsnorm import RMSNorm

    cfg = build(Mamba, MambaBlock, RMSNorm)
    from omegaconf import OmegaConf

    OmegaConf.set_struct(cfg, False)
    cfg.architecture.n_layers = 2
    cfg.architecture.n_embd = 64
    cfg.architecture.block_size = 32
    cfg.architecture.vocab_size = 128
    cfg.architecture.dropout = 0.0
    cfg.architecture.d_state = 8
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
        yield


# ---------------------------------------------------------------------------
# Core LoRA utilities
# ---------------------------------------------------------------------------


class TestParamFilter:
    def test_filter_kernel(self):
        """Targeting 'kernel' should match Dense layer kernels."""
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            key = jax.random.PRNGKey(0)
            idx = jnp.zeros((1, 8), dtype=jnp.int32)
            params = model.init(key, idx)["params"]

            mask = param_filter(params, ["kernel"])
            mask_leaves = jax.tree_util.tree_leaves(mask)
            assert any(mask_leaves), "No kernels found in GPT params"
            assert not all(mask_leaves), "All params are kernels? Unexpected."

    def test_filter_empty(self):
        """Empty target list should match nothing."""
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            params = model.init(
                jax.random.PRNGKey(0), jnp.zeros((1, 8), dtype=jnp.int32)
            )["params"]

            mask = param_filter(params, [])
            assert not any(jax.tree_util.tree_leaves(mask))


class TestInjectLoRA:
    def test_shapes(self):
        """Injected A/B have correct shapes for targeted 2D params."""
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            params = model.init(
                jax.random.PRNGKey(0), jnp.zeros((1, 8), dtype=jnp.int32)
            )["params"]

            rank = 4
            mask = param_filter(params, ["kernel"])
            lora_A, lora_B = inject_lora_params(params, mask, rank, jax.random.PRNGKey(1))

            # Use tree_map to check shapes in aligned pairs
            def _check(p, m, a, b):
                if m and p.ndim == 2:
                    assert a is not None, "A should exist for targeted 2D param"
                    assert a.shape == (p.shape[0], rank)
                    assert b is not None, "B should exist for targeted 2D param"
                    assert b.shape == (rank, p.shape[1])
                return p  # return something to satisfy tree_map

            jax.tree_util.tree_map(
                _check, params, mask, lora_A, lora_B,
                is_leaf=lambda x: x is None,
            )

    def test_b_initialized_zero(self):
        """B matrices should be zeros so initial delta is zero."""
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            params = model.init(
                jax.random.PRNGKey(0), jnp.zeros((1, 8), dtype=jnp.int32)
            )["params"]

            mask = param_filter(params, ["kernel"])
            _, lora_B = inject_lora_params(params, mask, 4, jax.random.PRNGKey(1))

            for leaf in jax.tree_util.tree_leaves(lora_B):
                if leaf is not None and hasattr(leaf, "shape"):
                    assert jnp.allclose(leaf, 0.0)


class TestMergeLoRA:
    def test_zero_b_preserves_base(self):
        """With B=0, merged params should equal base."""
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            params = model.init(
                jax.random.PRNGKey(0), jnp.zeros((1, 8), dtype=jnp.int32)
            )["params"]

            mask = param_filter(params, ["kernel"])
            lora_A, lora_B = inject_lora_params(params, mask, 4, jax.random.PRNGKey(1))
            merged = merge_lora_params(params, lora_A, lora_B, alpha=4.0, rank=4)

            for base_leaf, merged_leaf in zip(
                jax.tree_util.tree_leaves(params),
                jax.tree_util.tree_leaves(merged),
            ):
                assert jnp.allclose(base_leaf, merged_leaf, atol=1e-6)

    def test_nonzero_delta(self):
        """With non-zero B, merged should differ from base."""
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            params = model.init(
                jax.random.PRNGKey(0), jnp.zeros((1, 8), dtype=jnp.int32)
            )["params"]

            mask = param_filter(params, ["kernel"])
            lora_A, lora_B = inject_lora_params(params, mask, 4, jax.random.PRNGKey(1))

            # Set B to non-zero
            lora_B_nz = jax.tree_util.tree_map(
                lambda b: jnp.ones_like(b) if b is not None else None,
                lora_B,
                is_leaf=lambda x: x is None,
            )
            merged = merge_lora_params(params, lora_A, lora_B_nz, alpha=4.0, rank=4)

            any_different = False
            for base_leaf, merged_leaf in zip(
                jax.tree_util.tree_leaves(params),
                jax.tree_util.tree_leaves(merged),
            ):
                if not jnp.allclose(base_leaf, merged_leaf, atol=1e-6):
                    any_different = True
                    break
            assert any_different


class TestCountLoRA:
    def test_count(self):
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            params = model.init(
                jax.random.PRNGKey(0), jnp.zeros((1, 8), dtype=jnp.int32)
            )["params"]

            rank = 4
            mask = param_filter(params, ["kernel"])
            lora_A, lora_B = inject_lora_params(params, mask, rank, jax.random.PRNGKey(1))

            count = count_lora_params(lora_A, lora_B)
            assert count > 0


# ---------------------------------------------------------------------------
# Model-specific forward pass tests
# ---------------------------------------------------------------------------


class TestLoRAForwardGPT:
    """Test LoRA merge + forward for GPT."""

    def test_forward_matches_base_with_zero_b(self):
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            key = jax.random.PRNGKey(0)
            idx = jnp.zeros((1, 8), dtype=jnp.int32)
            params = model.init(key, idx)["params"]

            mask = param_filter(params, ["kernel"])
            lora_A, lora_B = inject_lora_params(params, mask, 4, jax.random.PRNGKey(1))
            merged = merge_lora_params(params, lora_A, lora_B, alpha=4.0, rank=4)

            logits_base, _ = model.apply({"params": params}, idx, deterministic=True)
            logits_lora, _ = model.apply({"params": merged}, idx, deterministic=True)

            assert jnp.allclose(logits_base, logits_lora, atol=1e-5)

    def test_gradients_flow_through_lora(self):
        """Verify that gradients flow through LoRA B params."""
        with _gpt_config_ctx():
            from theseus.model.models.base import GPT

            model = configure(GPT)
            key = jax.random.PRNGKey(0)
            idx = jnp.zeros((1, 8), dtype=jnp.int32)
            targets = jnp.ones((1, 8), dtype=jnp.int32)
            params = model.init(key, idx)["params"]

            mask = param_filter(params, ["kernel"])
            lora_A, lora_B = inject_lora_params(params, mask, 4, jax.random.PRNGKey(1))

            # Set B to small non-zero
            lora_B_nz = jax.tree_util.tree_map(
                lambda b: jnp.ones_like(b) * 0.01 if b is not None else None,
                lora_B,
                is_leaf=lambda x: x is None,
            )

            def loss_fn(b):
                merged = merge_lora_params(params, lora_A, b, alpha=4.0, rank=4)
                _, loss = model.apply(
                    {"params": merged}, idx, targets=targets, deterministic=True
                )
                return loss

            grad = jax.grad(loss_fn, allow_int=True)(lora_B_nz)

            # Check non-None leaves have gradients
            has_grad = False
            for leaf in jax.tree_util.tree_leaves(grad):
                if leaf is not None and hasattr(leaf, "shape") and leaf.size > 0:
                    if jnp.any(leaf != 0):
                        has_grad = True
                        break
            assert has_grad


class TestLoRAForwardMamba:
    """Test LoRA merge + forward for Mamba."""

    def test_forward_matches_base_with_zero_b(self):
        with _mamba_config_ctx():
            from theseus.model.models.mamba import Mamba

            model = configure(Mamba)
            key = jax.random.PRNGKey(0)
            idx = jnp.zeros((1, 8), dtype=jnp.int32)
            params = model.init(key, idx)["params"]

            mask = param_filter(params, ["kernel"])
            lora_A, lora_B = inject_lora_params(params, mask, 4, jax.random.PRNGKey(1))
            merged = merge_lora_params(params, lora_A, lora_B, alpha=4.0, rank=4)

            logits_base, _ = model.apply({"params": params}, idx, deterministic=True)
            logits_lora, _ = model.apply({"params": merged}, idx, deterministic=True)

            assert jnp.allclose(logits_base, logits_lora, atol=1e-5)


class TestLoRAContribModels:
    """Test LoRA param filter + injection for contrib models.

    Contrib models use configure() internally, so they need config contexts.
    We build minimal configs for each model family.
    """

    def _test_model_lora(self, model, idx):
        """Generic LoRA test: filter, inject, merge, forward."""
        params = model.init(jax.random.PRNGKey(0), idx)["params"]

        mask = param_filter(params, ["kernel"])
        mask_leaves = jax.tree_util.tree_leaves(mask)
        assert any(mask_leaves), "No kernels found"

        rank = 4
        lora_A, lora_B = inject_lora_params(params, mask, rank, jax.random.PRNGKey(1))

        count = count_lora_params(lora_A, lora_B)
        assert count > 0

        # Merge with zero B should preserve output
        merged = merge_lora_params(params, lora_A, lora_B, alpha=4.0, rank=4)
        logits_base, _ = model.apply({"params": params}, idx, deterministic=True)
        logits_lora, _ = model.apply({"params": merged}, idx, deterministic=True)
        assert jnp.allclose(logits_base, logits_lora, atol=1e-4)

    def _contrib_config_ctx(self, model_cls, **overrides):
        """Build a config context for a contrib model with tiny dimensions."""
        from omegaconf import OmegaConf

        all_types = model_cls.gather()
        cfg = build(*all_types)
        OmegaConf.set_struct(cfg, False)

        # Defaults for all contrib models
        defaults = dict(
            n_layers=2, n_embd=64, n_head=4, n_kv_head=4,
            block_size=32, vocab_size=128, dropout=0.0, attn_dropout=0.0,
            intermediate_size=128, rope_theta=10000.0, rms_norm_eps=1e-6,
            layer_norm_eps=1e-5, bias=False, attention_bias=False,
            use_sliding_window=False, sliding_window=32, max_window_layers=2,
            use_parallel_residual=True, partial_rotary_factor=1.0,
        )
        defaults.update(overrides)
        for k, v in defaults.items():
            try:
                setattr(cfg.architecture, k, v)
            except Exception:
                pass  # field doesn't exist on this model

        cfg.architecture.dtype = OmegaConf.create(
            {"param": "float32", "activation": "float32"}
        )
        OmegaConf.set_struct(cfg, True)
        return configuration(cfg)

    def test_llama(self):
        from theseus.model.models.contrib.llama import Llama

        with self._contrib_config_ctx(Llama):
            model = configure(Llama)
            self._test_model_lora(model, jnp.zeros((1, 8), dtype=jnp.int32))

    def test_gpt_neox(self):
        from theseus.model.models.contrib.gpt_neox import GPTNeoX

        with self._contrib_config_ctx(GPTNeoX):
            model = configure(GPTNeoX)
            self._test_model_lora(model, jnp.zeros((1, 8), dtype=jnp.int32))
