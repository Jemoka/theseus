from dataclasses import dataclass

import numpy as np

from theseus.config import field


@dataclass
class MokConfig:
    weighting: list[float] = field(
        "optimization/mok/weights", default_factory=lambda: [0.5, 0.5]
    )
    eps_min: float = field("optimization/mok/eps_min", default=1e-6)
    eps_max: float = field("optimization/mok/eps_max", default=0.5)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))  # type: ignore[no-any-return]


def mok_reward(
    scores: np.ndarray,
    config: MokConfig,
    progress: float = 1.0,
) -> np.ndarray:
    r"""MoK multi-objective scalarization. ``(N, k) -> (N,)``.

    Given per-rollout per-channel raw scores ``scores[n, i]``:

      1. Squash each channel to ``[0, 1]`` via sigmoid.
      2. Weight by ``config.weighting`` (renormalized to sum to 1) and append a
         residual channel so each row defines a distribution over ``k+1``
         categories::

            r̂_w = [w_1·r_1, ..., w_k·r_k, 1 - Σ_i w_i·r_i]

      3. Build the target distribution ``ŵ = [w_1·(1-ε), ..., w_k·(1-ε), ε]``.
      4. Return the per-rollout reward ``-D_KL(r̂_w || ŵ)``. Higher is better.

    ``progress ∈ [0, 1]`` linearly anneals ``ε`` from ``eps_max`` (early) to
    ``eps_min`` (late). Defaults to ``1.0`` so callers without a training-
    progress signal (e.g. eval pipelines) get ``ε = eps_min``.
    """
    if scores.ndim != 2:
        raise ValueError(f"mok_reward expects (N, k); got shape {scores.shape}.")
    _, k = scores.shape
    if len(config.weighting) != k:
        raise ValueError(
            f"MokConfig.weighting has {len(config.weighting)} entries but "
            f"scores has {k} channels."
        )

    s = _sigmoid(scores.astype(np.float32))
    weights = np.asarray(config.weighting, dtype=np.float32)
    weights = weights / weights.sum()

    eps = float(config.eps_max - (config.eps_max - config.eps_min) * progress)

    r_w = s * weights[None, :]  # (N, k)
    residual = 1.0 - r_w.sum(axis=-1, keepdims=True)  # (N, 1)
    r_w_hat = np.concatenate([r_w, residual], axis=-1)  # (N, k+1)
    w_hat = np.concatenate([weights * (1.0 - eps), np.array([eps], dtype=np.float32)])

    kl = np.sum(
        r_w_hat * (np.log(r_w_hat + 1e-10) - np.log(w_hat[None, :] + 1e-10)),
        axis=-1,
    )
    return -kl  # type: ignore[no-any-return]
