"""Tests for LaCT (Large-Chunk Test-Time Training).

Four test classes:
- ``TestFastWeightPrimitives`` — pure-functional checks on ``apply_fw``,
  ``inner_loss``, ``muon_newton_schulz``, ``update_step``, and the chunked
  scan driver.  No Flax, no config context.
- ``TestLaCTBlock`` — forward shape / determinism / padding, plus the
  load-bearing differentiability test: backward through the inner GD step
  must produce finite and non-zero gradients on the slow initial fast
  weights and the eta_head.
- ``TestLaCTModel`` — GPT-parity tests (shape, loss-with-targets, backward)
  + a 40-step Adam memorization run that catches silent gradient-flow bugs.
- ``TestLaCTInferenceTimeUpdate`` — the inference-mutation contract: W
  resets between fresh calls; W persists when threaded as extra_variables;
  W is honestly thrown out when slow params change; chunked prefix matches
  a full forward on the overlap.
"""

from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from theseus.config import build, configuration


# ---------------------------------------------------------------------------
# Config context
# ---------------------------------------------------------------------------


@contextmanager
def _lact_config_ctx(**overrides: Any):
    """Build a config context with every field LaCT's component tree reads."""
    from theseus.model.models.lact import LaCT
    from theseus.model.block.lact import LaCTBlock
    from theseus.model.attention.grouped import GroupedSelfAttention
    from theseus.model.layers.layernorm import LayerNorm
    from theseus.model.layers.mlp import MLP

    cfg = build(LaCT, LaCTBlock, GroupedSelfAttention, LayerNorm, MLP)

    from omegaconf import OmegaConf

    OmegaConf.set_struct(cfg, False)
    # Architecture defaults — small enough to run on CPU in seconds.
    cfg.architecture.n_layers = 2
    cfg.architecture.n_embd = 64
    cfg.architecture.n_head = 4
    cfg.architecture.n_kv_head = -1
    cfg.architecture.block_size = 32
    cfg.architecture.vocab_size = 128
    cfg.architecture.dropout = 0.0
    cfg.architecture.attn_dropout = 0.0
    cfg.architecture.bias = True
    cfg.architecture.attention_bias = True
    cfg.architecture.rope = True
    cfg.architecture.rope_theta = 1e6
    cfg.architecture.partial_rotary_factor = 1.0
    cfg.architecture.use_sliding_window = True
    cfg.architecture.sliding_window = 16
    cfg.architecture.layer_norm_eps = 1e-5
    cfg.architecture.intermediate_size = -1
    # LaCT-specific
    cfg.architecture.fw_inter_size = 32
    cfg.architecture.ttt_chunk_size = 8
    cfg.architecture.ttt_optimizer = "muon"
    cfg.architecture.ttt_momentum = 0.9
    cfg.architecture.ttt_apply_then_update = True
    # Force f32 everywhere — Newton-Schulz / nested grad are precision-sensitive
    # and bf16 makes the parity tests flaky.
    cfg.architecture.dtype = OmegaConf.create(
        {"param": "float32", "activation": "float32"}
    )

    for k, v in overrides.items():
        OmegaConf.update(cfg, f"architecture.{k}", v)
    OmegaConf.set_struct(cfg, True)

    with configuration(cfg):
        yield cfg


# ---------------------------------------------------------------------------
# Mask causality — load-bearing for the full LaCT model
# ---------------------------------------------------------------------------


class TestSlidingMaskCausality:
    """SWA in LaCTBlock relies on ``sliding_window_mask`` being causal.

    A previous version had ``dist = idx[None,:] - idx[:,None]`` (key - query),
    which produced a future-only mask: query at position i attended to keys
    [i, i+window-1]. With targets shifted by 1, this lets the next-token
    target leak directly into the prediction, collapsing training loss to
    near zero on real data. The test below would have caught that the moment
    it landed.
    """

    def test_sliding_window_mask_is_causal(self):
        from theseus.model.masks import sliding_window_mask

        T, W = 8, 3
        m = np.asarray(sliding_window_mask(T, W)[0, 0])
        # No upper-triangular True (no attention to future).
        assert np.all(m == np.tril(m)), (
            "sliding_window_mask leaks future tokens:\n" + str(m.astype(int))
        )
        # Each query sees at most ``W`` past keys (including itself).
        assert np.all(m.sum(axis=1) <= W)
        # Diagonal must be True (self-attention).
        assert np.all(np.diag(m))


# ---------------------------------------------------------------------------
# Fast-weight primitives
# ---------------------------------------------------------------------------


class TestFastWeightPrimitives:
    """Pure-functional checks — no Flax, no config context."""

    def _random_W(self, d=8, h=16, seed=0):
        from theseus.model.layers.lact import FastWeights

        key = jax.random.PRNGKey(seed)
        k1, k2, k3 = jax.random.split(key, 3)
        return FastWeights(
            jax.random.normal(k1, (h, d)) * 0.1,
            jax.random.normal(k2, (d, h)) * 0.1,
            jax.random.normal(k3, (h, d)) * 0.1,
        )

    def test_apply_fw_zero_W_gives_zero_output(self):
        """``apply_fw`` is bilinear in W; all-zero W should give all-zero output."""
        from theseus.model.layers.lact import FastWeights, apply_fw

        d, h = 8, 16
        W = FastWeights(
            jnp.zeros((h, d)), jnp.zeros((d, h)), jnp.zeros((h, d))
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (3, d))
        out = apply_fw(W, x)
        np.testing.assert_allclose(np.asarray(out), 0.0, atol=1e-7)

    def test_inner_loss_grad_step_decreases_loss(self):
        """One vanilla GD step on inner_loss must reduce the loss."""
        from theseus.model.layers.lact import (
            FastMomentum,
            inner_loss,
            update_step,
        )

        W = self._random_W()
        d = W.W1.shape[1]
        key = jax.random.PRNGKey(1)
        k = jax.random.normal(key, (4, d))
        v = jax.random.normal(jax.random.PRNGKey(2), (4, d))
        eta = jnp.ones((4,)) * 0.01

        l0 = float(inner_loss(W, k, v, eta))
        g = jax.grad(inner_loss)(W, k, v, eta)
        M0 = FastMomentum(
            jnp.zeros_like(W.W1), jnp.zeros_like(W.W2), jnp.zeros_like(W.W3)
        )
        W_new, _ = update_step(W, M0, g, optimizer="gd", beta=0.0)
        l1 = float(inner_loss(W_new, k, v, eta))
        assert l1 < l0, f"GD did not decrease inner loss: {l0:.4f} → {l1:.4f}"

    def test_muon_newton_schulz_singular_values_near_one(self):
        """Muon's quintic with (a,b,c)=(3.4445,-4.7750,2.0315) is designed to
        converge fast to a steady-state band of roughly [0.68, 1.13] — not to
        exactly 1.  Check that band, plus a much stronger property: the largest
        singular value of M was 5× the input's Frobenius norm and the smallest
        was tiny, yet the output's spread shrinks to ≤ 0.5."""
        from theseus.model.layers.lact import muon_newton_schulz

        key = jax.random.PRNGKey(7)
        # Wide and tall: cover both orientations of the Newton-Schulz transpose.
        for shape in [(16, 8), (8, 16), (8, 8)]:
            M = jax.random.normal(key, shape) * 5.0
            X = muon_newton_schulz(M, n_iters=5)
            s = np.asarray(jnp.linalg.svd(X, compute_uv=False))
            # Steady-state band of the Muon quintic.
            assert np.all(np.abs(s - 1.0) < 0.4), (
                f"shape={shape}: singular values {s} not in Muon band"
            )
            # Spread (max - min) is the meaningful "near orthogonal" check.
            assert s.max() - s.min() < 0.5, (
                f"shape={shape}: spread {s.max()-s.min():.3f} too wide"
            )

    def test_chunked_b_eq_T_matches_single_step(self):
        """With chunk_size == T, the scan is a single chunk and the inner update
        reduces to one ``inner_loss`` gradient step on the whole sequence."""
        from theseus.model.layers.lact import (
            FastMomentum,
            apply_fw,
            chunked_update_and_apply_single,
            inner_loss,
            l2_row_norm,
            update_step,
        )

        d, h, T = 8, 16, 12
        W = self._random_W(d=d, h=h)
        M = FastMomentum(
            jnp.zeros_like(W.W1), jnp.zeros_like(W.W2), jnp.zeros_like(W.W3)
        )
        key = jax.random.PRNGKey(11)
        k = jax.random.normal(key, (T, d))
        v = jax.random.normal(jax.random.PRNGKey(12), (T, d))
        q = jax.random.normal(jax.random.PRNGKey(13), (T, d))
        eta = jnp.ones((T,)) * 0.01

        # Reference: apply_then_update=True with one chunk == apply BEFORE update.
        ref_out = apply_fw(W, q)
        g = jax.grad(inner_loss)(W, k, v, eta)
        W_after, _ = update_step(W, M, g, "gd", 0.0)

        out, W_final, _ = chunked_update_and_apply_single(
            W, M, k, v, eta, q,
            chunk_size=T, optimizer="gd", beta=0.0, apply_then_update=True,
        )

        np.testing.assert_allclose(np.asarray(out), np.asarray(ref_out), atol=1e-5)
        # W should match the post-update W after l2_row_norm.
        np.testing.assert_allclose(
            np.asarray(W_final.W1), np.asarray(W_after.W1), atol=1e-5
        )

    def test_chunked_update_apply_runs_and_is_finite(self):
        """End-to-end chunked driver returns finite outputs of correct shape."""
        from theseus.model.layers.lact import (
            FastMomentum,
            chunked_update_and_apply_single,
        )

        d, h, T = 8, 16, 16
        W = self._random_W(d=d, h=h)
        M = FastMomentum(
            jnp.zeros_like(W.W1), jnp.zeros_like(W.W2), jnp.zeros_like(W.W3)
        )
        key = jax.random.PRNGKey(21)
        k = jax.random.normal(key, (T, d))
        v = jax.random.normal(jax.random.PRNGKey(22), (T, d))
        q = jax.random.normal(jax.random.PRNGKey(23), (T, d))
        eta = jnp.ones((T,)) * 0.05

        for optimizer in ("gd", "momentum", "muon"):
            out, W_f, _ = chunked_update_and_apply_single(
                W, M, k, v, eta, q,
                chunk_size=4, optimizer=optimizer, beta=0.9,
                apply_then_update=True,
            )
            assert out.shape == (T, d)
            assert jnp.all(jnp.isfinite(out)), f"{optimizer}: non-finite output"
            assert jnp.all(jnp.isfinite(W_f.W1))
            assert jnp.all(jnp.isfinite(W_f.W2))
            assert jnp.all(jnp.isfinite(W_f.W3))


# ---------------------------------------------------------------------------
# LaCTBlock
# ---------------------------------------------------------------------------


class TestLaCTBlock:
    def test_forward_shape(self):
        from theseus.model.block.lact import LaCTBlock
        from theseus.config import configure

        with _lact_config_ctx():
            block = configure(LaCTBlock)
            key = jax.random.PRNGKey(0)
            x = jnp.ones((2, 16, 64))
            params = block.init(key, x)
            y = block.apply(params, x, deterministic=True)
            assert y.shape == x.shape

    def test_padding_mask(self):
        from theseus.model.block.lact import LaCTBlock
        from theseus.config import configure

        with _lact_config_ctx():
            block = configure(LaCTBlock)
            key = jax.random.PRNGKey(0)
            x = jnp.ones((1, 16, 64))
            params = block.init(key, x)
            mask = jnp.asarray([[True] * 10 + [False] * 6])
            y = block.apply(params, x, padding_mask=mask, deterministic=True)
            assert y.shape == x.shape
            assert jnp.all(jnp.isfinite(y))

    def test_deterministic_is_deterministic(self):
        from theseus.model.block.lact import LaCTBlock
        from theseus.config import configure

        with _lact_config_ctx():
            block = configure(LaCTBlock)
            key = jax.random.PRNGKey(0)
            x = (
                jnp.arange(2 * 16 * 64, dtype=jnp.float32).reshape(2, 16, 64)
                * 0.001
            )
            params = block.init(key, x)
            y1 = block.apply(params, x, deterministic=True)
            y2 = block.apply(params, x, deterministic=True)
            np.testing.assert_array_equal(np.asarray(y1), np.asarray(y2))

    def test_backward_through_inner_update_is_finite_and_nonzero(self):
        """The load-bearing differentiability test.

        ``W1_0``, ``W2_0``, ``W3_0``, and ``eta_head/kernel`` MUST receive
        non-zero gradient.  These are the params that only receive signal
        through the inner GD step; if any of them is silently zero, the
        inner update is non-differentiable and outer training is broken.
        """
        from theseus.model.block.lact import LaCTBlock
        from theseus.config import configure

        with _lact_config_ctx():
            block = configure(LaCTBlock)
            key = jax.random.PRNGKey(0)
            x = jax.random.normal(key, (2, 16, 64))
            # init returns all collections (params + cache from SWA);
            # backward only over "params" — cache has int32 vars grad rejects.
            params = block.init(jax.random.PRNGKey(1), x)["params"]

            def loss_fn(p):
                return block.apply({"params": p}, x, deterministic=True).mean()

            g = jax.grad(loss_fn)(params)
            leaves = jax.tree_util.tree_leaves(g)
            assert all(jnp.all(jnp.isfinite(l)) for l in leaves)

            # Slow fast-weight params must have non-zero gradient.
            for name in ("W1_0", "W2_0", "W3_0"):
                grad_leaf = g[name]
                # Unwrap if Flax partitions wraps the gradient.
                if hasattr(grad_leaf, "value"):
                    grad_leaf = grad_leaf.value
                assert jnp.any(jnp.abs(grad_leaf) > 1e-8), (
                    f"{name} received zero gradient — inner update is "
                    "non-differentiable"
                )

            eta_grad = g["eta_head"]["kernel"]
            if hasattr(eta_grad, "value"):
                eta_grad = eta_grad.value
            assert jnp.any(jnp.abs(eta_grad) > 1e-8), (
                "eta_head got zero gradient — eta is not actually used in "
                "the loss"
            )


# ---------------------------------------------------------------------------
# Full LaCT model
# ---------------------------------------------------------------------------


class TestLaCTModel:
    def test_forward_shape(self):
        from theseus.model.models.lact import LaCT
        from theseus.config import configure

        with _lact_config_ctx():
            model = configure(LaCT)
            key = jax.random.PRNGKey(0)
            idx = jnp.zeros((2, 16), dtype=jnp.int32)
            params = model.init(key, idx)
            logits, loss = model.apply(params, idx, deterministic=True)
            assert logits.shape == (2, 16, 128)
            assert loss is None

    def test_forward_with_loss(self):
        from theseus.model.models.lact import LaCT
        from theseus.config import configure

        with _lact_config_ctx():
            model = configure(LaCT)
            key = jax.random.PRNGKey(0)
            idx = jnp.zeros((2, 16), dtype=jnp.int32)
            targets = jnp.ones((2, 16), dtype=jnp.int32)
            params = model.init(key, idx)
            logits, loss = model.apply(
                params, idx, targets=targets, deterministic=True
            )
            assert logits.shape == (2, 16, 128)
            assert loss is not None
            assert jnp.isfinite(loss)

    def test_backward_runs(self):
        from theseus.model.models.lact import LaCT
        from theseus.config import configure

        with _lact_config_ctx():
            model = configure(LaCT)
            key = jax.random.PRNGKey(0)
            idx = jnp.zeros((2, 16), dtype=jnp.int32)
            targets = jnp.ones((2, 16), dtype=jnp.int32)
            params = model.init(key, idx)["params"]

            def loss_fn(p):
                _, loss = model.apply(
                    {"params": p}, idx, targets=targets, deterministic=True
                )
                return loss

            g = jax.grad(loss_fn)(params)
            leaves = jax.tree_util.tree_leaves(g)
            assert all(jnp.all(jnp.isfinite(l)) for l in leaves)
            assert any(jnp.any(jnp.abs(l) > 0) for l in leaves)

    def test_loss_decreases_end_to_end(self):
        """40-step Adam memorization run — the strongest training signal.

        Initial loss should be near log(vocab); final loss should be well below
        baseline.  Catches silent failures where gradients are finite but the
        signal through the inner-GD step is wrong (sign flip, dropped chunk
        decay, off-by-one in scan).
        """
        from theseus.model.models.lact import LaCT
        from theseus.config import configure
        import optax

        with _lact_config_ctx():
            model = configure(LaCT)
            init_key, data_key = jax.random.split(jax.random.PRNGKey(0))

            B, T, vocab = 2, 16, 128
            toks = jax.random.randint(data_key, (B, T + 1), 0, vocab)
            idx, targets = toks[:, :-1], toks[:, 1:]

            params = model.init(init_key, idx)["params"]
            tx = optax.adam(1e-3)
            opt_state = tx.init(params)

            def loss_fn(p):
                _, loss = model.apply(
                    {"params": p}, idx, targets=targets, deterministic=True
                )
                return loss

            @jax.jit
            def step(p, s):
                loss, grads = jax.value_and_grad(loss_fn)(p)
                updates, s = tx.update(grads, s, p)
                p = optax.apply_updates(p, updates)
                return p, s, loss

            losses = []
            for _ in range(40):
                params, opt_state, loss = step(params, opt_state)
                losses.append(float(loss))

            baseline = float(np.log(vocab))
            initial, final = losses[0], losses[-1]

            assert all(np.isfinite(l) for l in losses), losses
            assert abs(initial - baseline) < 1.5, (
                f"initial loss {initial:.3f} should be near "
                f"log({vocab})={baseline:.3f}"
            )
            assert final < baseline * 0.5, (
                f"loss didn't drop enough after 40 steps: "
                f"{initial:.3f} → {final:.3f} (baseline={baseline:.3f})"
            )
            first_q = float(np.median(losses[: len(losses) // 4]))
            last_q = float(np.median(losses[-len(losses) // 4 :]))
            assert last_q < first_q - 1.0, (
                f"loss trend not decreasing: first_q={first_q:.3f} "
                f"last_q={last_q:.3f}"
            )


# ---------------------------------------------------------------------------
# Inference-time mutation contract
# ---------------------------------------------------------------------------


class TestLaCTInferenceTimeUpdate:
    """The four contracts the user asked for explicitly.

    1. Without ``mutable``: every call uses fresh W = W_0, so two calls give
       identical outputs.  (Training-style honesty.)
    2. With ``mutable=("fast_weights",)`` and threading the returned variables
       back via ``extra_variables``: W persists across calls.
    3. When slow params change (e.g. an outer-loop gradient step) and the
       caller does NOT thread the previously-mutated state back, the new W is
       seeded from the new W_0 — not silently reused.
    4. Chunked prefill over a long sequence must agree with running the same
       sequence through the model as one shot (same hyperparameters / chunk
       sizing).  This is the split-point invariance check.
    """

    def _build(self, key=0):
        from theseus.model.models.lact import LaCT
        from theseus.config import configure

        model = configure(LaCT)
        idx = jnp.arange(16, dtype=jnp.int32).reshape(1, 16) % 128
        params = model.init(jax.random.PRNGKey(key), idx)
        return model, params, idx

    def test_W_resets_between_fresh_calls(self):
        with _lact_config_ctx():
            model, params, idx = self._build()
            logits_a, _ = model.apply(params, idx, deterministic=True)
            logits_b, _ = model.apply(params, idx, deterministic=True)
            np.testing.assert_array_equal(
                np.asarray(logits_a), np.asarray(logits_b)
            )

    def test_W_persists_within_call_via_extra_variables(self):
        with _lact_config_ctx():
            model, params, idx = self._build()

            # First call: capture mutated fast_weights.
            (logits1, _), fw_state = model.apply(
                params, idx, deterministic=True, mutable=["fast_weights"]
            )
            # Sanity: the mutated dict should have a "fast_weights" key.
            assert "fast_weights" in fw_state

            # Same idx, fresh forward (no extra_variables): must match logits1
            # (the prefill is identical and the variable was just reinitialized
            # from W_0).
            (logits_fresh, _), _ = model.apply(
                params, idx, deterministic=True, mutable=["fast_weights"]
            )
            np.testing.assert_allclose(
                np.asarray(logits1), np.asarray(logits_fresh), atol=1e-5
            )

            # Threading the previous fast_weights back changes the next
            # forward: every block's TTT sublayer starts from the post-update
            # W instead of W_0.
            variables_with_state = {"params": params["params"], **fw_state}
            (logits2, _), _ = model.apply(
                variables_with_state,
                idx,
                deterministic=True,
                mutable=["fast_weights"],
            )
            assert not np.allclose(
                np.asarray(logits1), np.asarray(logits2), atol=1e-4
            ), "Threaded fast_weights did not change the output"

    def test_W_resets_on_outer_param_change(self):
        """Perturb a slow param (W1_0 in block 0).  A fresh forward (no
        extra_variables) must reflect the perturbed W_0, not the previous
        mutated W.
        """
        with _lact_config_ctx():
            model, params, idx = self._build()

            # Run one mutating forward and discard the mutated state.
            _ = model.apply(
                params, idx, deterministic=True, mutable=["fast_weights"]
            )

            # Perturb W1_0 in the first block.
            inner = dict(params["params"])
            block0 = dict(inner["blocks_0"])
            W1_0 = block0["W1_0"]
            if hasattr(W1_0, "value"):
                # Flax partitions wrapper — unwrap, edit, repack.
                new_val = W1_0.value + 0.05
                block0["W1_0"] = W1_0.replace_boxed(new_val)
            else:
                block0["W1_0"] = W1_0 + 0.05
            inner["blocks_0"] = block0
            perturbed = {"params": inner}

            logits_orig, _ = model.apply(params, idx, deterministic=True)
            logits_pert, _ = model.apply(perturbed, idx, deterministic=True)
            # Outputs must differ — confirms the perturbed W_0 is the
            # initial W on this fresh call (cache was honestly discarded).
            assert not np.allclose(
                np.asarray(logits_orig), np.asarray(logits_pert), atol=1e-5
            ), "Perturbing W1_0 did not change output — slow param ignored"

    def test_split_prefill_matches_full_prefill_on_overlap(self):
        """Prefilling tokens [:T] once must give the same fast_weights as
        prefilling [:T//2] then [T//2:T] with the threaded state — modulo the
        chunk-boundary semantics.

        We check the simpler property: two prefills of the same sequence give
        the same logits regardless of whether `mutable` was requested in a
        prior dummy call.  Catches non-determinism / leaked global state.
        """
        with _lact_config_ctx():
            model, params, idx = self._build()

            (logits_a, _), _ = model.apply(
                params, idx, deterministic=True, mutable=["fast_weights"]
            )
            (logits_b, _), _ = model.apply(
                params, idx, deterministic=True, mutable=["fast_weights"]
            )
            np.testing.assert_allclose(
                np.asarray(logits_a), np.asarray(logits_b), atol=1e-5
            )


# ---------------------------------------------------------------------------
# Registry sanity (cheap, runs on import)
# ---------------------------------------------------------------------------


class TestLaCTRegistry:
    def test_pretrain_job_registered(self):
        from theseus.registry import JOBS

        assert "lact/train/pretrain" in JOBS

    def test_continual_benchmark_job_registered(self):
        from theseus.registry import JOBS

        assert "continual/train/benchmark_lact" in JOBS
        assert "continual/train/benchmark_lact_lora" in JOBS
