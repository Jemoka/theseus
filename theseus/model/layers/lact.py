"""Pure-functional primitives for LaCT fast weights.

LaCT (arXiv:2505.23884) maintains a small SwiGLU MLP ``f_W(x) = W2 (silu(W1 x) * W3 x)``
whose weights ``W = (W1, W2, W3)`` are gradient-stepped once per chunk of ``b`` tokens
on the inner loss ``L(W) = - sum_i eta_i * <f_W(k_i), v_i>``.  These functions are
deliberately Flax-free: they operate on plain JAX arrays / NamedTuples so the
training path can call ``jax.grad(inner_loss)`` directly and the inference path can
plug into a Flax ``self.variable("fast_weights", ...)`` cell.

Per-sequence W is the unit of computation — every batch element runs its own
chunked scan. We expose single-sequence functions and a ``jax.vmap``-batched
wrapper to keep both shapes legible.
"""

from typing import NamedTuple, Tuple, cast

import jax
import jax.numpy as jnp


class FastWeights(NamedTuple):
    """Per-sequence fast-weight matrices for a single LaCT block.

    Shapes (no batch axis, single sequence):
        W1: (h, d) — input-side gate projection
        W2: (d, h) — output projection
        W3: (h, d) — input-side up projection
    where ``d`` is the model hidden dim and ``h`` is the fast-weight intermediate dim.
    Batched usage adds a leading ``(B, ...)`` axis via ``jax.vmap``.
    """

    W1: jax.Array
    W2: jax.Array
    W3: jax.Array


class FastMomentum(NamedTuple):
    """Optimizer momentum buffers, shape-matched to ``FastWeights``."""

    M1: jax.Array
    M2: jax.Array
    M3: jax.Array


def fw_zeros_like(W: FastWeights) -> FastMomentum:
    """Zero momentum buffers with the same shape/dtype as ``W``."""
    return FastMomentum(
        jnp.zeros_like(W.W1), jnp.zeros_like(W.W2), jnp.zeros_like(W.W3)
    )


def apply_fw(W: FastWeights, x: jax.Array) -> jax.Array:
    """SwiGLU forward through fast weights.

    Args:
        W: FastWeights for a single sequence (W1/W3: (h, d), W2: (d, h)).
        x: (..., d) input.

    Returns:
        (..., d) output ``W2 @ (silu(W1 @ x) * (W3 @ x))``.
    """
    h1 = jax.nn.silu(jnp.einsum("hd,...d->...h", W.W1, x))
    h3 = jnp.einsum("hd,...d->...h", W.W3, x)
    return jnp.einsum("dh,...h->...d", W.W2, h1 * h3)


def inner_loss(W: FastWeights, k: jax.Array, v: jax.Array, eta: jax.Array) -> jax.Array:
    """Negative-dot-product inner loss for one chunk of one sequence.

    Args:
        W: FastWeights for this sequence.
        k, v: (b, d) keys / values for ``b`` tokens.
        eta: (b,) per-token inner learning rates (zero on padded positions).

    Returns:
        Scalar ``- sum_i eta_i * <f_W(k_i), v_i>``.  No lower bound by construction;
        the L2 row-norm in ``update_step`` is what keeps the iterate stable across
        chunks.
    """
    fk = apply_fw(W, k)  # (b, d)
    per_token = jnp.einsum("bd,bd->b", fk, v)
    return -jnp.sum(eta * per_token)


def l2_row_norm(W: FastWeights, eps: float = 1e-6) -> FastWeights:
    """Renormalize each row to unit L2 along the input axis.

    Acts as weight decay and bounds the iterate after every chunk update.  "Row"
    here means the input-axis vector: for W1 (h, d) we normalize axis=-1 (so each
    of the h hidden neurons keeps a unit-norm input-side weight); same convention
    for W2 (d, h) and W3 (h, d).
    """

    def _norm(M: jax.Array) -> jax.Array:
        n = jnp.linalg.norm(M, axis=-1, keepdims=True)
        return M / jnp.maximum(n, eps)

    return FastWeights(_norm(W.W1), _norm(W.W2), _norm(W.W3))


def muon_newton_schulz(M: jax.Array, n_iters: int = 5, eps: float = 1e-7) -> jax.Array:
    """5-iteration Newton-Schulz polynomial that maps M ≈ U S V^T → U V^T.

    Implements the quintic from the Muon paper (arXiv:2502.16982): per iteration
    ``X ← a·X + b·(XX^T)X + c·(XX^T)^2 X`` with ``(a, b, c) = (3.4445, -4.7750,
    2.0315)`` and the input first rescaled to Frobenius norm 1.  Cheap because it
    only needs matmuls in the smaller of the two matrix dims (we transpose tall
    matrices so ``XX^T`` is the smaller side).

    Args:
        M: 2D matrix.
        n_iters: number of Newton-Schulz iterations.

    Returns:
        Matrix of the same shape, with singular values all ≈ 1.
    """
    a, b_, c = 3.4445, -4.7750, 2.0315
    X = M / (jnp.linalg.norm(M) + eps)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T

    def body(carry: jax.Array, _: jax.Array) -> Tuple[jax.Array, None]:
        A = carry @ carry.T
        return a * carry + b_ * (A @ carry) + c * (A @ A @ carry), None

    X, _ = jax.lax.scan(body, X, jnp.arange(n_iters))
    if transposed:
        X = X.T
    return cast(jax.Array, X)


def _orthogonalize_fw(g: FastWeights) -> FastWeights:
    """Apply Muon Newton-Schulz to each matrix in a fast-weight gradient."""
    return FastWeights(
        muon_newton_schulz(g.W1),
        muon_newton_schulz(g.W2),
        muon_newton_schulz(g.W3),
    )


def update_step(
    W: FastWeights,
    M: FastMomentum,
    g: FastWeights,
    optimizer: str,
    beta: float,
) -> Tuple[FastWeights, FastMomentum]:
    """One inner optimizer step for a single sequence.

    Args:
        W: current fast weights.
        M: momentum (zeros when ``optimizer == "gd"``).
        g: gradient ``∂inner_loss/∂W`` for this chunk.
        optimizer: one of ``"gd"``, ``"momentum"``, ``"muon"``.
        beta: momentum coefficient (ignored for ``"gd"``).

    Returns:
        ``(W_new, M_new)``.  ``W`` is renormalized after the step via
        ``l2_row_norm`` regardless of optimizer.
    """
    if optimizer == "gd":
        delta = g
        new_M = M
    elif optimizer == "momentum":
        new_M = FastMomentum(
            beta * M.M1 + g.W1,
            beta * M.M2 + g.W2,
            beta * M.M3 + g.W3,
        )
        delta = FastWeights(new_M.M1, new_M.M2, new_M.M3)
    elif optimizer == "muon":
        new_M = FastMomentum(
            beta * M.M1 + g.W1,
            beta * M.M2 + g.W2,
            beta * M.M3 + g.W3,
        )
        delta = _orthogonalize_fw(FastWeights(new_M.M1, new_M.M2, new_M.M3))
    else:
        raise ValueError(f"unknown ttt optimizer: {optimizer!r}")

    W_new = FastWeights(
        W.W1 - delta.W1,
        W.W2 - delta.W2,
        W.W3 - delta.W3,
    )
    return l2_row_norm(W_new), new_M


def _pad_to_multiple(arr: jax.Array, b: int, axis: int) -> Tuple[jax.Array, int]:
    """Right-pad ``arr`` along ``axis`` so its length becomes a multiple of ``b``."""
    T = arr.shape[axis]
    pad = (-T) % b
    if pad == 0:
        return arr, 0
    pad_widths = [(0, 0)] * arr.ndim
    pad_widths[axis] = (0, pad)
    return jnp.pad(arr, pad_widths), pad


def chunked_update_and_apply_single(
    W0: FastWeights,
    M0: FastMomentum,
    K: jax.Array,
    V: jax.Array,
    eta: jax.Array,
    Q: jax.Array,
    chunk_size: int,
    optimizer: str,
    beta: float,
    apply_then_update: bool,
) -> Tuple[jax.Array, FastWeights, FastMomentum]:
    """Run the chunked TTT inner loop for one sequence.

    Args:
        W0, M0: initial fast weights and momentum (typically the slow params and zeros).
        K, V, Q: (T, d) tensors.  Q is the query, K/V the key/value the chunk's
            inner loss is computed against.
        eta: (T,) per-token learning rates.  Pre-zero padded positions before calling.
        chunk_size: tokens per inner-loop chunk (``b`` in the paper).
        optimizer: ``"gd"`` | ``"momentum"`` | ``"muon"``.
        beta: momentum coefficient.
        apply_then_update: when True, the chunk's queries see the pre-update W
            (shifted block-causal order, LM default); when False, queries see the
            post-update W (full block-causal order).

    Returns:
        ``(out, W_final, M_final)``: out is (T, d) — the TTT-layer output before any
        residual / projection.  ``W_final``/``M_final`` are the state after the last
        chunk's update, used by the inference path to persist across forward calls.
    """
    T = K.shape[0]
    K_p, pad = _pad_to_multiple(K, chunk_size, axis=0)
    V_p, _ = _pad_to_multiple(V, chunk_size, axis=0)
    Q_p, _ = _pad_to_multiple(Q, chunk_size, axis=0)
    eta_p, _ = _pad_to_multiple(eta, chunk_size, axis=0)
    T_pad = T + pad
    n_chunks = T_pad // chunk_size

    d = K.shape[-1]
    K_c = K_p.reshape(n_chunks, chunk_size, d)
    V_c = V_p.reshape(n_chunks, chunk_size, d)
    Q_c = Q_p.reshape(n_chunks, chunk_size, d)
    eta_c = eta_p.reshape(n_chunks, chunk_size)

    def step(
        carry: Tuple[FastWeights, FastMomentum], c: jax.Array
    ) -> Tuple[Tuple[FastWeights, FastMomentum], jax.Array]:
        W, M = carry
        K_i, V_i, Q_i, eta_i = K_c[c], V_c[c], Q_c[c], eta_c[c]
        if apply_then_update:
            out_i = apply_fw(W, Q_i)
            g = jax.grad(inner_loss)(W, K_i, V_i, eta_i)
            W, M = update_step(W, M, g, optimizer, beta)
        else:
            g = jax.grad(inner_loss)(W, K_i, V_i, eta_i)
            W, M = update_step(W, M, g, optimizer, beta)
            out_i = apply_fw(W, Q_i)
        return (W, M), out_i

    (W_final, M_final), outs = jax.lax.scan(step, (W0, M0), jnp.arange(n_chunks))
    out_flat = outs.reshape(T_pad, d)
    return out_flat[:T], W_final, M_final


def chunked_update_and_apply(
    W0: FastWeights,
    M0: FastMomentum,
    K: jax.Array,
    V: jax.Array,
    eta: jax.Array,
    Q: jax.Array,
    chunk_size: int,
    optimizer: str,
    beta: float,
    apply_then_update: bool,
) -> Tuple[jax.Array, FastWeights, FastMomentum]:
    """Batched wrapper around ``chunked_update_and_apply_single``.

    Args:
        W0, M0: batched FastWeights / FastMomentum with leading batch axis.
        K, V, Q: (B, T, d).
        eta: (B, T).
        Other args: see ``chunked_update_and_apply_single``.

    Returns:
        ``(out (B, T, d), W_final (B, ...), M_final (B, ...))``.
    """
    return jax.vmap(
        chunked_update_and_apply_single,
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None, None),
        out_axes=(0, 0, 0),
    )(W0, M0, K, V, eta, Q, chunk_size, optimizer, beta, apply_then_update)


def batch_broadcast(W: FastWeights, batch_size: int) -> FastWeights:
    """Broadcast a single-sequence FastWeights to ``(B, ...)``."""
    return FastWeights(
        jnp.broadcast_to(W.W1, (batch_size,) + W.W1.shape),
        jnp.broadcast_to(W.W2, (batch_size,) + W.W2.shape),
        jnp.broadcast_to(W.W3, (batch_size,) + W.W3.shape),
    )


def batch_zeros_momentum(W_batched: FastWeights) -> FastMomentum:
    """Zero momentum matching a batched FastWeights pytree."""
    return FastMomentum(
        jnp.zeros_like(W_batched.W1),
        jnp.zeros_like(W_batched.W2),
        jnp.zeros_like(W_batched.W3),
    )
