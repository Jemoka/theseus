"""KV cache parity tests.

For each model, verifies that token-by-token autoregressive generation
with KV cache produces identical logits to a single full-sequence
forward pass.

Migrated from scripts/test_kv_cache.py.
"""

from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

from theseus.config import build, configuration


def _kv_cache_parity(
    model_cls: type,
    model_kwargs: dict[str, Any],
    seq_len: int = 8,
    vocab_size: int = 128,
    atol: float = 1e-4,
) -> None:
    """Assert that cached autoregressive logits match full-sequence logits."""
    model = model_cls(**model_kwargs)
    key = jax.random.PRNGKey(42)
    dummy = jnp.zeros((1, 1), dtype=jnp.int32)
    variables = model.init(key, dummy)
    params = variables["params"]

    tokens = jax.random.randint(jax.random.PRNGKey(0), (1, seq_len), 0, vocab_size)

    # Full forward pass
    full_logits, _ = model.apply({"params": params}, tokens, deterministic=True)
    full_logits = np.array(full_logits)

    # Cached: prefill half, then one-by-one
    prefill_len = seq_len // 2
    (prefill_logits, _), cache = model.apply(
        {"params": params}, tokens[:, :prefill_len],
        deterministic=True, mutable=["cache"],
    )
    cached_logits = [np.array(prefill_logits[0, i]) for i in range(prefill_len)]

    for i in range(prefill_len, seq_len):
        (step_logits, _), cache = model.apply(
            {"params": params, **cache}, tokens[:, i : i + 1],
            deterministic=True, mutable=["cache"],
        )
        cached_logits.append(np.array(step_logits[0, 0]))

    cached_arr = np.stack(cached_logits, axis=0)
    max_diff = float(np.max(np.abs(full_logits[0] - cached_arr)))
    assert max_diff < atol, f"max_diff={max_diff} exceeds atol={atol}"


def _build_config_ctx(model_cls, overrides: dict):
    """Build a config context from model's component tree with overrides."""
    cfg = build(*model_cls.gather())
    OmegaConf.set_struct(cfg, False)
    for k, v in overrides.items():
        OmegaConf.update(cfg, f"architecture.{k}", v)
    OmegaConf.set_struct(cfg, True)
    return configuration(cfg)


class TestKVCacheGPT:
    def test_gpt_rope(self):
        from theseus.model.models.base import GPT

        kwargs = dict(
            n_layers=2, n_embd=64, rope=True, block_size=32,
            dropout=0.0, vocab_size=128,
        )
        overrides = {**kwargs, "n_head": 4, "bias": True}
        with _build_config_ctx(GPT, overrides):
            _kv_cache_parity(GPT, kwargs, atol=2e-3)


class TestKVCacheLlama:
    def test_llama(self):
        from theseus.model.models.contrib.llama import Llama

        kwargs = dict(
            n_layers=2, n_embd=64, n_head=4, n_kv_head=2,
            intermediate_size=128, block_size=32, vocab_size=128,
            dropout=0.0, attn_dropout=0.0, rope_theta=10000.0,
            rms_norm_eps=1e-6, bias=False, attention_bias=False,
        )
        with _build_config_ctx(Llama, kwargs):
            _kv_cache_parity(Llama, kwargs)


class TestKVCacheQwen:
    def test_qwen(self):
        from theseus.model.models.contrib.qwen import Qwen

        kwargs = dict(
            n_layers=2, n_embd=64, n_head=4, n_kv_head=2,
            intermediate_size=128, block_size=32, vocab_size=128,
            dropout=0.0, attn_dropout=0.0, rope_theta=1e6,
            rms_norm_eps=1e-6, use_sliding_window=False,
            sliding_window=-1, max_window_layers=28, bias=False,
        )
        with _build_config_ctx(Qwen, kwargs):
            _kv_cache_parity(Qwen, kwargs)


class TestKVCacheGPTNeoX:
    def test_gpt_neox(self):
        from theseus.model.models.contrib.gpt_neox import GPTNeoX

        kwargs = dict(
            n_layers=2, n_embd=64, n_head=4, n_kv_head=4,
            intermediate_size=128, block_size=32, vocab_size=128,
            dropout=0.0, attn_dropout=0.0, rope_theta=10000.0,
            partial_rotary_factor=0.25, layer_norm_eps=1e-5,
            bias=True, attention_bias=True, use_parallel_residual=True,
        )
        with _build_config_ctx(GPTNeoX, kwargs):
            _kv_cache_parity(GPTNeoX, kwargs)
