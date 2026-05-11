"""Tests for Mamba-2 and Hybrid models.

The selective-scan tests are pinned against a sequential reference
implementation (``_naive_scan``) so any algorithmic drift in the chunked
SSD path will fail loudly.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from contextlib import contextmanager

from theseus.config import build, configuration


# ---------------------------------------------------------------------------
# Reference selective scan
# ---------------------------------------------------------------------------


def _naive_scan(
    A: jax.Array,        # (B, T, H)
    B_mat: jax.Array,    # (B, T, G, N)
    C_mat: jax.Array,    # (B, T, G, N)
    dt: jax.Array,       # (B, T, H)
    x: jax.Array,        # (B, T, H) or (B, T, H, P)
) -> jax.Array:
    """Sequential Mamba-2 selective scan — slow but obviously correct.

    Implements
        state[t] = exp(A[t] * dt[t]) * state[t-1] + B[t] * dt[t] * x[t]
        y[t]     = C[t] · state[t]
    timestep-by-timestep with a Python for-loop.  Used as ground truth for
    the chunked SSD implementation in ``_ssd_scan``.
    """
    squeeze = x.ndim == 3
    if squeeze:
        x = x[..., None]

    batch, T, H, P = x.shape
    G = B_mat.shape[2]
    N = B_mat.shape[3]
    hpg = H // G

    # Broadcast B/C to per-head via group sharing (no allocation: broadcast
    # only). Reshape back to (B, T, H, N).
    B_exp = jnp.broadcast_to(
        B_mat[:, :, :, None, :], (batch, T, G, hpg, N)
    ).reshape(batch, T, H, N)
    C_exp = jnp.broadcast_to(
        C_mat[:, :, :, None, :], (batch, T, G, hpg, N)
    ).reshape(batch, T, H, N)

    state = jnp.zeros((batch, H, N, P), dtype=x.dtype)
    ys = []
    for t in range(T):
        a_t = jnp.exp(A[:, t] * dt[:, t])  # (B, H)
        state = state * a_t[:, :, None, None]
        delta = (
            B_exp[:, t, :, :, None]
            * dt[:, t, :, None, None]
            * x[:, t, :, None, :]
        )
        state = state + delta
        y_t = (C_exp[:, t, :, :, None] * state).sum(axis=2)  # (B, H, P)
        ys.append(y_t)

    y = jnp.stack(ys, axis=1)  # (B, T, H, P)
    return y[..., 0] if squeeze else y


def _random_inputs(B, T, H, G, N, *, P=None, seed=0):
    """Sample a plausible parameter regime: A < 0, dt > 0."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 5)
    # A in [-2, 0] — typical post-init range for -exp(A_log)
    A = -jnp.abs(jax.random.normal(keys[0], (B, T, H))) * 0.5
    B_mat = jax.random.normal(keys[1], (B, T, G, N)) * 0.3
    C_mat = jax.random.normal(keys[2], (B, T, G, N)) * 0.3
    dt = jax.nn.softplus(jax.random.normal(keys[3], (B, T, H)) - 1.0)  # ~0.3
    if P is None:
        x = jax.random.normal(keys[4], (B, T, H))
    else:
        x = jax.random.normal(keys[4], (B, T, H, P))
    return A, B_mat, C_mat, dt, x


# ---------------------------------------------------------------------------
# Config contexts
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Selective scan: correctness + invariants
# ---------------------------------------------------------------------------


class TestSelectiveScan:
    """Pin the chunked SSD scan against a sequential reference."""

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
        A, B_mat, C_mat, dt, x = _random_inputs(B, T, H, G, N)
        y = _selective_scan(A, B_mat, C_mat, dt, x)
        assert y.shape == (B, T, H)
        assert jnp.all(jnp.isfinite(y))

    @pytest.mark.parametrize(
        "B,T,H,G,N,P",
        [
            (1, 8, 4, 1, 8, 1),     # tiny, P=1
            (2, 16, 8, 2, 4, 2),    # grouped, P=2
            (1, 32, 4, 4, 4, 4),    # G == H (no group sharing), P=head_dim
            (2, 64, 6, 3, 8, 3),    # multi-chunk-aligned
        ],
    )
    def test_ssd_matches_naive(self, B, T, H, G, N, P):
        """SSD chunked scan reproduces the sequential recurrence."""
        from theseus.model.block.mamba import _ssd_scan

        A, B_mat, C_mat, dt, x = _random_inputs(B, T, H, G, N, P=P)
        y_ssd = _ssd_scan(A, B_mat, C_mat, dt, x)
        y_naive = _naive_scan(A, B_mat, C_mat, dt, x)
        np.testing.assert_allclose(y_ssd, y_naive, rtol=2e-4, atol=2e-4)

    @pytest.mark.parametrize("chunk_size", [2, 4, 8, 16, 32])
    def test_ssd_chunk_size_invariant(self, chunk_size):
        """The chunk_size partitioning is purely a perf knob, not semantic."""
        from theseus.model.block.mamba import _ssd_scan

        B, T, H, G, N, P = 1, 32, 4, 1, 8, 3
        A, B_mat, C_mat, dt, x = _random_inputs(B, T, H, G, N, P=P, seed=1)

        ref = _naive_scan(A, B_mat, C_mat, dt, x)
        out = _ssd_scan(A, B_mat, C_mat, dt, x, chunk_size=chunk_size)
        np.testing.assert_allclose(out, ref, rtol=2e-4, atol=2e-4)

    def test_ssd_seq_len_not_multiple_of_chunk(self):
        """Padding is silent: T=11, chunk_size=4 still matches the naive scan."""
        from theseus.model.block.mamba import _ssd_scan

        B, T, H, G, N, P = 1, 11, 2, 1, 4, 2
        A, B_mat, C_mat, dt, x = _random_inputs(B, T, H, G, N, P=P, seed=2)

        out = _ssd_scan(A, B_mat, C_mat, dt, x, chunk_size=4)
        ref = _naive_scan(A, B_mat, C_mat, dt, x)
        assert out.shape == (B, T, H, P)
        np.testing.assert_allclose(out, ref, rtol=2e-4, atol=2e-4)

    def test_ssd_seq_len_smaller_than_chunk(self):
        """T < chunk_size: single chunk after padding."""
        from theseus.model.block.mamba import _ssd_scan

        B, T, H, G, N, P = 1, 3, 2, 1, 4, 2
        A, B_mat, C_mat, dt, x = _random_inputs(B, T, H, G, N, P=P, seed=3)

        out = _ssd_scan(A, B_mat, C_mat, dt, x, chunk_size=8)
        ref = _naive_scan(A, B_mat, C_mat, dt, x)
        np.testing.assert_allclose(out, ref, rtol=2e-4, atol=2e-4)

    def test_ssd_no_decay_is_running_sum(self):
        """A = 0 ⇒ no decay; y[t] = C[t]·sum_{j≤t} B[j]·dt[j]·x[j]."""
        from theseus.model.block.mamba import _ssd_scan

        B, T, H, G, N, P = 1, 12, 2, 1, 4, 2
        key = jax.random.PRNGKey(7)
        keys = jax.random.split(key, 4)
        A = jnp.zeros((B, T, H))
        B_mat = jax.random.normal(keys[0], (B, T, G, N)) * 0.3
        C_mat = jax.random.normal(keys[1], (B, T, G, N)) * 0.3
        dt = jnp.ones((B, T, H)) * 0.1
        x = jax.random.normal(keys[2], (B, T, H, P))

        out = _ssd_scan(A, B_mat, C_mat, dt, x, chunk_size=4)
        ref = _naive_scan(A, B_mat, C_mat, dt, x)
        np.testing.assert_allclose(out, ref, rtol=2e-4, atol=2e-4)

    def test_ssd_zero_dt_zero_output(self):
        """dt = 0 zeros both the decay exponent and the input contribution;
        the state never accumulates, so y is identically 0."""
        from theseus.model.block.mamba import _ssd_scan

        B, T, H, G, N, P = 2, 16, 4, 1, 4, 2
        key = jax.random.PRNGKey(11)
        keys = jax.random.split(key, 4)
        A = -jnp.abs(jax.random.normal(keys[0], (B, T, H)))
        B_mat = jax.random.normal(keys[1], (B, T, G, N))
        C_mat = jax.random.normal(keys[2], (B, T, G, N))
        dt = jnp.zeros((B, T, H))
        x = jax.random.normal(keys[3], (B, T, H, P))

        out = _ssd_scan(A, B_mat, C_mat, dt, x, chunk_size=4)
        np.testing.assert_allclose(out, jnp.zeros_like(out), atol=1e-6)

    def test_ssd_gradients_match_naive(self):
        """Backprop through SSD must agree with backprop through the naive
        recurrence on every input (A, B, C, dt, x)."""
        from theseus.model.block.mamba import _ssd_scan

        B, T, H, G, N, P = 1, 12, 4, 2, 4, 2
        A, B_mat, C_mat, dt, x = _random_inputs(B, T, H, G, N, P=P, seed=42)

        def loss_ssd(args):
            return _ssd_scan(*args, chunk_size=4).sum()

        def loss_naive(args):
            return _naive_scan(*args).sum()

        args = (A, B_mat, C_mat, dt, x)
        g_ssd = jax.grad(loss_ssd)(args)
        g_naive = jax.grad(loss_naive)(args)

        for s, n, name in zip(g_ssd, g_naive, ["A", "B", "C", "dt", "x"]):
            np.testing.assert_allclose(
                s, n, rtol=2e-3, atol=2e-3,
                err_msg=f"gradient mismatch in {name}",
            )

    def test_ssd_state_carries_across_chunks(self):
        """A single impulse at t=0 should still reach y[t] for t in any
        chunk, with the expected geometric decay.  This catches state-prop
        bugs (off-by-one in state_at_start, dropped chunk_decay, etc.)."""
        from theseus.model.block.mamba import _ssd_scan

        B, T, H, G, N, P = 1, 16, 1, 1, 1, 1
        A = jnp.full((B, T, H), -0.1)        # constant log decay
        dt = jnp.ones((B, T, H))
        B_mat = jnp.zeros((B, T, G, N)).at[:, 0].set(1.0)  # impulse only at t=0
        C_mat = jnp.ones((B, T, G, N))
        x = jnp.zeros((B, T, H, P)).at[:, 0].set(1.0)

        out = _ssd_scan(A, B_mat, C_mat, dt, x, chunk_size=4)[..., 0]  # (B, T, H)
        # y[t] should equal exp(-0.1)^t  (decay applied once per step after t=0)
        expected = jnp.exp(-0.1 * jnp.arange(T)).astype(out.dtype)
        np.testing.assert_allclose(out[0, :, 0], expected, rtol=2e-4, atol=2e-4)


# ---------------------------------------------------------------------------
# MambaBlock
# ---------------------------------------------------------------------------


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
            # All gradients should match the corresponding param shapes.
            param_leaves = jax.tree_util.tree_leaves(params)
            for g_leaf, p_leaf in zip(leaves, param_leaves):
                assert g_leaf.shape == p_leaf.shape

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
            assert jnp.all(jnp.isfinite(y))
            # The mask is applied multiplicatively to the SSM output before the
            # residual add, so masked-out positions still see the input residual.
            # We can at least check the shape and finiteness.
            assert y.shape == x.shape

    def test_deterministic_is_deterministic(self):
        """Two passes with deterministic=True should be bit-identical."""
        from theseus.model.block.mamba import MambaBlock
        from theseus.config import configure

        with _mamba_config_ctx():
            block = configure(MambaBlock)
            key = jax.random.PRNGKey(0)
            x = jnp.arange(2 * 16 * 128, dtype=jnp.float32).reshape(2, 16, 128) * 0.001
            params = block.init(key, x)

            y1 = block.apply(params, x, deterministic=True)
            y2 = block.apply(params, x, deterministic=True)
            np.testing.assert_array_equal(np.asarray(y1), np.asarray(y2))

    def test_block_varying_seq_lens(self):
        """Block should accept any T (chunk-size padding is internal)."""
        from theseus.model.block.mamba import MambaBlock
        from theseus.config import configure

        with _mamba_config_ctx():
            block = configure(MambaBlock)
            key = jax.random.PRNGKey(0)
            # init at T=16, run at T=7, 16, 33
            x_init = jnp.ones((1, 16, 128))
            params = block.init(key, x_init)

            for T in (7, 16, 33):
                x = jnp.ones((1, T, 128)) * 0.1
                y = block.apply(params, x, deterministic=True)
                assert y.shape == (1, T, 128)
                assert jnp.all(jnp.isfinite(y))


# ---------------------------------------------------------------------------
# Full Mamba model
# ---------------------------------------------------------------------------


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
            # Catch the silent-zero-gradient case: at least *some* gradient must
            # be non-zero, otherwise the loss isn't actually connected to params.
            assert any(jnp.any(jnp.abs(l) > 0) for l in leaves)

    def test_loss_decreases_end_to_end(self):
        """A short Adam memorization run on a fixed micro-batch.

        Initial loss should be near the uniform-prior baseline of
        ``log(vocab_size)``; after a few dozen steps the model should
        memorize the batch and the loss should fall well below that
        baseline.  Catches the silent class of failures where the model
        compiles and gradients stay finite but the backward signal is
        wrong (sign flip in the scan, dropped chunk decay, off-by-one in
        state propagation) — none of which fail the existing finite/shape
        assertions but all of which prevent learning.
        """
        from theseus.model.models.mamba import Mamba
        from theseus.config import configure
        import optax

        with _mamba_config_ctx():
            model = configure(Mamba)
            init_key, data_key = jax.random.split(jax.random.PRNGKey(0))

            B, T, vocab = 2, 16, 256
            # Random next-token prediction targets on a fixed batch — the task
            # is pure memorization, well within a 2-layer Mamba's capacity.
            toks = jax.random.randint(data_key, (B, T + 1), 0, vocab)
            idx, targets = toks[:, :-1], toks[:, 1:]

            params = model.init(init_key, idx)
            tx = optax.adam(1e-3)
            opt_state = tx.init(params)

            def loss_fn(p):
                _, loss = model.apply(p, idx, targets=targets, deterministic=True)
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

            baseline = float(np.log(vocab))  # ~5.545: uniform prior
            initial, final = losses[0], losses[-1]

            # 1. Loss is finite throughout — no NaN explosions from the scan.
            assert all(np.isfinite(l) for l in losses), losses

            # 2. Initial loss is near the uniform-prior baseline (catches
            #    pathological init or a model that hard-codes some token).
            assert abs(initial - baseline) < 1.5, (
                f"initial loss {initial:.3f} should be near "
                f"log({vocab})={baseline:.3f}"
            )

            # 3. Substantial drop: memorization should clear half the baseline.
            assert final < baseline * 0.5, (
                f"loss didn't drop enough after 40 steps: "
                f"{initial:.3f} → {final:.3f} (baseline={baseline:.3f})"
            )

            # 4. Monotone-ish trend: last-quarter median sits at least 1 nat
            #    below first-quarter median.  Tolerant of stepwise noise but
            #    fails if the run is flat or oscillating.
            first_q = float(np.median(losses[: len(losses) // 4]))
            last_q = float(np.median(losses[-len(losses) // 4 :]))
            assert last_q < first_q - 1.0, (
                f"loss trend not decreasing enough: "
                f"first_quarter_median={first_q:.3f} "
                f"last_quarter_median={last_q:.3f}"
            )


# ---------------------------------------------------------------------------
# Hybrid (transformer + Mamba) model
# ---------------------------------------------------------------------------


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
