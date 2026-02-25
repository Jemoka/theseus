from dataclasses import dataclass
from theseus.config import field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from optax._src import base


# Polar Express coefficients for orthogonalization.
# From https://arxiv.org/pdf/2505.16932 (computed for num_iters=5, safety_factor=2e-2, cushion=2)
_POLAR_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


@dataclass
class MuonConfig:
    # Muon hyperparameters
    momentum: float = field("optimization/muon_momentum", default=0.95)
    ns_steps: int = field("optimization/muon_ns_steps", default=5)
    muon_beta2: float = field("optimization/muon_beta2", default=0.95)
    weight_decay: float = field("optimization/muon_weight_decay", default=0.0)

    # LR multipliers relative to the base schedule LR (matrix params define the "1x" baseline)
    # Example raw values: matrix=0.02, embedding=0.3, unembedding=0.004, scalar=0.5
    # → multipliers: matrix=1.0, embedding=15.0, unembedding=0.2, scalar=25.0
    matrix_lr_multiplier: float = field("optimization/matrix_lr_multiplier", default=1.0)
    embedding_lr_multiplier: float = field("optimization/embedding_lr_multiplier", default=15.0)
    unembedding_lr_multiplier: float = field("optimization/unembedding_lr_multiplier", default=0.2)
    scalar_lr_multiplier: float = field("optimization/scalar_lr_multiplier", default=25.0)

    # AdamW hyperparameters for embedding / unembedding / scalar params
    adam_beta1: float = field("optimization/adam_beta1", default=0.8)
    adam_beta2: float = field("optimization/adam_beta2", default=0.95)
    adam_eps: float = field("optimization/adam_eps", default=1e-10)


class _MuonState(NamedTuple):
    momentum: Any        # pytree matching params: first-moment buffers
    second_moment: Any   # pytree matching params: factored second-moment buffers
    count: Any           # scalar step counter


def _polar_express(X: jax.Array, num_steps: int) -> jax.Array:
    """Polar Express orthogonalization for a 2D matrix (JAX port of the PyTorch kernel).

    Iteratively applies quintic Newton-Schulz-style iterations to compute the
    polar factor U of X ≈ U S V^T.  The coefficients maximise the slope at zero
    so the update is near-orthogonal after `num_steps` passes.
    """
    X = X / (jnp.linalg.norm(X) * 1.02 + 1e-6)
    if X.shape[0] >= X.shape[1]:  # tall matrix
        for a, b, c in _POLAR_COEFFS[:num_steps]:
            A = X.T @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:  # wide matrix
        for a, b, c in _POLAR_COEFFS[:num_steps]:
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X


def scale_by_muon(
    momentum: float = 0.95,
    ns_steps: int = 5,
    beta2: float = 0.95,
) -> base.GradientTransformation:
    """Core Muon gradient transformation.

    Applies three sequential operations to each update:
      1. Nesterov momentum (EMA of gradients with look-ahead blend)
      2. Polar Express orthogonalization for 2-D+ parameters
      3. NorMuon per-row/col adaptive variance reduction

    Parameters with fewer than 2 dimensions receive only the momentum step.

    Args:
        momentum: EMA coefficient for the first-moment buffer.
        ns_steps:  Number of Polar Express iterations (1–5; 5 recommended).
        beta2:     EMA coefficient for the factored second-moment buffer.

    Returns:
        An :class:`optax.GradientTransformation`.
    """

    def init_fn(params):
        mu = jax.tree.map(jnp.zeros_like, params)

        def _make_nu(p):
            if p.ndim >= 2:
                # Factored second moment: reduce along the smaller dimension
                if p.shape[-2] >= p.shape[-1]:
                    return jnp.zeros((*p.shape[:-1], 1), dtype=jnp.float32)
                else:
                    return jnp.zeros((*p.shape[:-2], 1, p.shape[-1]), dtype=jnp.float32)
            return jnp.zeros(p.shape, dtype=jnp.float32)

        nu = jax.tree.map(_make_nu, params)
        return _MuonState(momentum=mu, second_moment=nu, count=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None):
        mu, nu, count = state

        # --- 1. Nesterov momentum (mirrors PyTorch's lerp_ pair) ---
        new_mu = jax.tree.map(
            lambda m, g: momentum * m + (1 - momentum) * g,
            mu, updates,
        )
        g = jax.tree.map(
            lambda new_m, orig_g: (1 - momentum) * orig_g + momentum * new_m,
            new_mu, updates,
        )

        # --- 2. Polar Express orthogonalization for matrix params ---
        def _orth(grad):
            if grad.ndim < 2:
                return grad
            orig_shape = grad.shape
            g2d = grad.reshape(grad.shape[0], -1).astype(jnp.float32)
            g_orth = _polar_express(g2d, ns_steps)
            # Rectangular correction: scale by sqrt(max(1, M/N))
            rect_scale = jnp.sqrt(jnp.maximum(1.0, g2d.shape[0] / g2d.shape[1]))
            return (g_orth * rect_scale).reshape(orig_shape).astype(grad.dtype)

        g = jax.tree.map(_orth, g)

        # --- 3. NorMuon variance reduction ---
        def _update_nu(grad, v):
            if grad.ndim < 2:
                return v
            g_f = grad.astype(jnp.float32)
            if grad.shape[-2] >= grad.shape[-1]:
                v_mean = jnp.mean(g_f ** 2, axis=-1, keepdims=True)
            else:
                v_mean = jnp.mean(g_f ** 2, axis=-2, keepdims=True)
            return (1 - beta2) * v_mean + beta2 * v

        new_nu = jax.tree.map(_update_nu, g, nu)

        def _apply_normuon(grad, new_v):
            if grad.ndim < 2:
                return grad
            g_f = grad.astype(jnp.float32)
            if grad.shape[-2] >= grad.shape[-1]:
                v_mean = jnp.mean(g_f ** 2, axis=-1, keepdims=True)
                red_dim = grad.shape[-1]
            else:
                v_mean = jnp.mean(g_f ** 2, axis=-2, keepdims=True)
                red_dim = grad.shape[-2]

            v_norm = jnp.sqrt(jnp.sum(v_mean * red_dim))
            step_size = jnp.maximum(new_v, 1e-10) ** -0.5
            scaled_sq = (v_mean * red_dim) * step_size ** 2
            v_norm_new = jnp.sqrt(jnp.sum(scaled_sq))
            final_scale = step_size * (v_norm / jnp.maximum(v_norm_new, 1e-10))
            return (grad * final_scale).astype(grad.dtype)

        g = jax.tree.map(_apply_normuon, g, new_nu)

        return g, _MuonState(momentum=new_mu, second_moment=new_nu, count=count + 1)

    return base.GradientTransformation(init_fn, update_fn)


def _cautious_weight_decay(weight_decay: float) -> base.GradientTransformation:
    """Apply weight decay only where the update and parameter agree in sign.

    This is the "cautious" variant from the reference implementation:
      update += weight_decay * param * (sign(update) == sign(param))
    """

    def init_fn(params):
        return ()

    def update_fn(updates, state, params=None):
        if params is None or weight_decay == 0.0:
            return updates, state
        def _apply(u, p):
            mask = (u * p >= 0).astype(u.dtype)
            return u + weight_decay * p * mask
        return jax.tree.map(_apply, updates, params), state

    return base.GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Default parameter labeller
# ---------------------------------------------------------------------------

_EMBED_KWS = {"embed", "wte", "wpe", "embedding", "token_embed"}
_UNEMBED_KWS = {"lm_head", "unembed", "unembedding"}


def _label_params(params: Any) -> Any:
    """Classify each parameter leaf for :func:`optax.multi_transform`.

    Labels:
      ``'matrix'``      – 2-D+ params that are not embeddings/output (→ Muon)
      ``'embedding'``   – 2-D params whose path contains an embedding keyword
      ``'unembedding'`` – 2-D params whose path contains an unembedding keyword
      ``'scalar'``      – 0-D or 1-D params (biases, norms, scalars)

    The path matching is intentionally conservative and keyword-based; pass a
    custom ``param_labels`` callable to :func:`muon` for non-standard layouts.
    """

    def _classify(path, leaf):
        path_str = ".".join(
            k.key if hasattr(k, "key") else str(k) for k in path
        ).lower()
        if leaf.ndim <= 1:
            return "scalar"
        if any(kw in path_str for kw in _UNEMBED_KWS):
            return "unembedding"
        if any(kw in path_str for kw in _EMBED_KWS):
            return "embedding"
        return "matrix"

    return jax.tree_util.tree_map_with_path(_classify, params)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def muon(
    lr: optax._src.base.Schedule | float,
    cfg: MuonConfig,
    param_labels=None,
) -> optax.GradientTransformation:
    """Muon + AdamW mixed optimizer, mirroring the :func:`adamw` API.

    Applies the Muon update (momentum → Polar-Express orthogonalization →
    NorMuon variance reduction) to matrix-shaped parameters, and standard
    AdamW to embedding, unembedding, and scalar parameters.  Each group is
    scaled by its own LR multiplier relative to the base schedule.

    Args:
        lr:           Base LR schedule (``step → float``) or constant float.
                      Matrix params receive ``lr * cfg.matrix_lr_multiplier``.
        cfg:          :class:`MuonConfig` with all hyperparameters.
        param_labels: Optional callable ``params → labels_pytree``.  If
                      ``None``, :func:`_label_params` is used (path-keyword
                      heuristic).  Pass a custom callable for architectures
                      with non-standard naming.

    Returns:
        An :class:`optax.GradientTransformation`.
    """

    def _scaled(multiplier: float) -> optax._src.base.Schedule | float:
        if callable(lr):
            return lambda step: lr(step) * multiplier
        return lr * multiplier

    matrix_tx = optax.chain(
        scale_by_muon(cfg.momentum, cfg.ns_steps, cfg.muon_beta2),
        _cautious_weight_decay(cfg.weight_decay),
        optax.scale_by_learning_rate(_scaled(cfg.matrix_lr_multiplier)),
    )

    def _adamw_tx(multiplier: float) -> optax.GradientTransformation:
        return optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(b1=cfg.adam_beta1, b2=cfg.adam_beta2, eps=cfg.adam_eps),
            optax.scale_by_learning_rate(_scaled(multiplier)),
        )

    return optax.multi_transform(
        transforms={
            "matrix": matrix_tx,
            "embedding": _adamw_tx(cfg.embedding_lr_multiplier),
            "unembedding": _adamw_tx(cfg.unembedding_lr_multiplier),
            "scalar": _adamw_tx(cfg.scalar_lr_multiplier),
        },
        param_labels=param_labels if param_labels is not None else _label_params,
    )
